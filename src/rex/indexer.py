"""AST-based indexer for Python packages.

Walks .venv, parses all .py files, extracts symbols without importing.
"""

from __future__ import annotations

import ast
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True, slots=True)
class Symbol:
    """A documented Python symbol."""

    name: str  # e.g., "from_pretrained"
    qualified_name: str  # e.g., "transformers.Sam3Processor.from_pretrained"
    symbol_type: str  # "class", "function", "method", "module"
    signature: str | None  # e.g., "(cls, pretrained_model_name, **kwargs)"
    docstring: str | None
    file_path: str
    line_no: int
    # For navigation: what this symbol references
    bases: tuple[str, ...]  # For classes: base class names
    return_annotation: str | None  # For functions/methods


class ASTExtractor(ast.NodeVisitor):
    """Extract symbols from a Python AST."""

    def __init__(self, file_path: str, module_name: str):
        self.file_path = file_path
        self.module_name = module_name
        self.symbols: list[Symbol] = []
        self._class_stack: list[str] = []

    def _get_qualified_name(self, name: str) -> str:
        if self._class_stack:
            return f"{self.module_name}.{'.'.join(self._class_stack)}.{name}"
        return f"{self.module_name}.{name}"

    def _get_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Extract function signature as string."""
        parts = []

        # Handle positional-only args (before /)
        posonlyargs = node.args.posonlyargs
        args = node.args.args
        defaults = node.args.defaults

        # Calculate default offset
        num_defaults = len(defaults)
        num_args = len(args)
        default_offset = num_args - num_defaults

        all_positional = posonlyargs + args
        num_posonly = len(posonlyargs)

        for i, arg in enumerate(all_positional):
            param = arg.arg
            if arg.annotation:
                param += f": {ast.unparse(arg.annotation)}"

            # Check if this arg has a default
            # Defaults apply to the rightmost arguments
            arg_index_in_args = i - num_posonly if i >= num_posonly else -1
            if arg_index_in_args >= 0 and arg_index_in_args >= default_offset:
                default_idx = arg_index_in_args - default_offset
                param += f" = {ast.unparse(defaults[default_idx])}"

            parts.append(param)

            # Add / after positional-only args
            if i == num_posonly - 1 and num_posonly > 0:
                parts.append("/")

        # *args
        if node.args.vararg:
            vararg = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                vararg += f": {ast.unparse(node.args.vararg.annotation)}"
            parts.append(vararg)
        elif node.args.kwonlyargs:
            parts.append("*")

        # Keyword-only args
        kw_defaults = node.args.kw_defaults
        for i, arg in enumerate(node.args.kwonlyargs):
            param = arg.arg
            if arg.annotation:
                param += f": {ast.unparse(arg.annotation)}"
            if kw_defaults[i] is not None:
                param += f" = {ast.unparse(kw_defaults[i])}"
            parts.append(param)

        # **kwargs
        if node.args.kwarg:
            kwarg = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                kwarg += f": {ast.unparse(node.args.kwarg.annotation)}"
            parts.append(kwarg)

        sig = f"({', '.join(parts)})"

        # Return annotation
        if node.returns:
            sig += f" -> {ast.unparse(node.returns)}"

        return sig

    def _get_docstring(self, node: ast.AST) -> str | None:
        """Extract docstring from a node."""
        if not (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module))
            and node.body
        ):
            return None
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                return first.value.value
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.name.startswith("_") and not node.name.startswith("__"):
            return  # skip private classes (but keep dunders)
        bases = tuple(ast.unparse(b) for b in node.bases)
        self.symbols.append(
            Symbol(
                name=node.name,
                qualified_name=self._get_qualified_name(node.name),
                symbol_type="class",
                signature=f"({', '.join(bases)})" if bases else None,
                docstring=self._get_docstring(node),
                file_path=self.file_path,
                line_no=node.lineno,
                bases=bases,
                return_annotation=None,
            )
        )
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node, is_async=True)

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool = False
    ) -> None:
        if node.name.startswith("_") and not node.name.startswith("__"):
            return  # skip private functions (but keep dunders)
        symbol_type = "method" if self._class_stack else "function"
        if is_async:
            symbol_type = f"async_{symbol_type}"

        return_annotation = ast.unparse(node.returns) if node.returns else None

        self.symbols.append(
            Symbol(
                name=node.name,
                qualified_name=self._get_qualified_name(node.name),
                symbol_type=symbol_type,
                signature=self._get_signature(node),
                docstring=self._get_docstring(node),
                file_path=self.file_path,
                line_no=node.lineno,
                bases=(),
                return_annotation=return_annotation,
            )
        )



def parse_file(file_path: Path, module_name: str) -> list[Symbol]:
    """Parse a Python file and extract all symbols."""
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, ValueError):
        return []

    # Add module-level docstring as a symbol
    extractor = ASTExtractor(str(file_path), module_name)

    module_doc = extractor._get_docstring(tree)
    if module_doc:
        extractor.symbols.append(
            Symbol(
                name=module_name.split(".")[-1],
                qualified_name=module_name,
                symbol_type="module",
                signature=None,
                docstring=module_doc,
                file_path=str(file_path),
                line_no=1,
                bases=(),
                return_annotation=None,
            )
        )

    try:
        extractor.visit(tree)
    except RecursionError:
        # Some files have deeply nested AST (e.g., generated math expressions)
        pass
    return extractor.symbols


def find_venv(start_dir: Path | None = None) -> Path | None:
    """Find .venv directory starting from given dir or cwd."""
    if start_dir is None:
        start_dir = Path.cwd()

    current = start_dir.resolve()
    while current != current.parent:
        venv = current / ".venv"
        if venv.is_dir():
            return venv
        current = current.parent
    # Fallback: check home directory
    home_venv = Path.home() / ".venv"
    if home_venv.is_dir():
        return home_venv
    return None


def find_site_packages(venv: Path) -> Path | None:
    """Find site-packages directory in a venv."""
    lib = venv / "lib"
    if not lib.exists():
        # Windows layout
        site_packages = venv / "Lib" / "site-packages"
        if site_packages.exists():
            return site_packages
        return None

    # Unix layout: lib/pythonX.Y/site-packages
    for python_dir in lib.iterdir():
        if python_dir.name.startswith("python"):
            site_packages = python_dir / "site-packages"
            if site_packages.exists():
                return site_packages
    return None


_SKIP_SUFFIXES = {".dist-info", ".egg-info", ".so", ".pyd", ".pth"}


def _has_python_content(directory: Path) -> bool:
    """Check if directory has Python content (namespace package heuristic).

    Checks one level deep for .py/.pyi files or subdirs with __init__.py.
    """
    for child in directory.iterdir():
        if child.is_file() and child.suffix in (".py", ".pyi"):
            return True
        if child.is_dir() and (child / "__init__.py").exists():
            return True
    return False


def _is_package(entry: Path) -> tuple[str, Path] | None:
    """Check if a directory entry is a Python package or module.

    Returns (name, path) or None.
    Recognizes both regular packages (__init__.py) and
    namespace packages (PEP 420 — no __init__.py).
    """
    if entry.name.startswith(("_", ".")):
        return None
    if entry.suffix in _SKIP_SUFFIXES:
        return None
    if entry.is_dir():
        if (entry / "__init__.py").exists():
            return entry.name, entry
        if _has_python_content(entry):
            return entry.name, entry
    if entry.suffix == ".py":
        return entry.stem, entry
    return None


def _iter_pth_packages(
    site_packages: Path, pth_file: Path, seen: set[str],
) -> Iterator[tuple[str, Path]]:
    """Yield packages from directories listed in a .pth file."""
    try:
        text = pth_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "import ")):
            continue
        pth_dir = (site_packages / line).resolve()
        if not pth_dir.is_dir():
            continue
        for sub in pth_dir.iterdir():
            pkg = _is_package(sub)
            if pkg and pkg[0] not in seen:
                seen.add(pkg[0])
                yield pkg


def iter_packages(site_packages: Path) -> Iterator[tuple[str, Path]]:
    """Iterate over packages in site-packages.

    Yields (package_name, package_path) tuples.
    Also scans .pth files for packages with non-standard layouts
    (e.g. rerun_sdk.pth -> rerun_sdk/rerun/).
    """
    seen: set[str] = set()

    for entry in site_packages.iterdir():
        if entry.suffix == ".pth":
            yield from _iter_pth_packages(site_packages, entry, seen)
            continue
        pkg = _is_package(entry)
        if pkg and pkg[0] not in seen:
            seen.add(pkg[0])
            yield pkg


def index_package(package_name: str, package_path: Path) -> Iterator[Symbol]:
    """Index a single package, yielding symbols."""
    if package_path.is_file():
        # Single .py file module
        yield from parse_file(package_path, package_name)
        return

    # Directory package — include _-prefixed files (public symbols
    # are often defined there and re-exported via __init__.py)
    for file_path in iter_py_files(package_path):
        rel_path = file_path.relative_to(package_path.parent)
        parts = list(rel_path.parts)

        if parts[-1] in ("__init__.py", "__init__.pyi"):
            parts = parts[:-1]
        else:
            parts[-1] = Path(parts[-1]).stem  # Remove .py/.pyi

        module_name = ".".join(parts)
        yield from parse_file(file_path, module_name)


def iter_py_files(directory: Path) -> Iterator[Path]:
    """Yield .py/.pyi file paths under directory, skipping hidden/cache dirs.

    When both foo.py and foo.pyi exist, only foo.py is yielded.
    Stubs are used as fallback for compiled extensions (e.g. .so/.pyd).
    """
    for root, dirs, files in os.walk(directory):
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and d != "__pycache__"
        ]
        py_files = {f for f in files if f.endswith(".py")}
        for f in py_files:
            yield Path(root) / f
        # .pyi stubs only when no .py counterpart exists
        for f in files:
            if f.endswith(".pyi") and f[:-1] not in py_files:
                yield Path(root) / f


def index_directory(directory: Path) -> Iterator[Symbol]:
    """Index all .py files in a directory tree, yielding symbols.

    Skips __pycache__, hidden dirs (.*), and .venv dirs.
    """
    directory = directory.resolve()

    for file_path in iter_py_files(directory):
        rel_path = file_path.relative_to(directory)
        parts = list(rel_path.parts)

        if parts[-1] in ("__init__.py", "__init__.pyi"):
            parts = parts[:-1]
            if not parts:
                continue
        else:
            parts[-1] = Path(parts[-1]).stem  # Remove .py/.pyi

        module_name = ".".join(parts)
        yield from parse_file(file_path, module_name)


def index_site_packages(site_packages: Path, progress_callback=None) -> Iterator[Symbol]:
    """Index all packages in a site-packages directory."""
    packages = list(iter_packages(site_packages))
    for i, (pkg_name, pkg_path) in enumerate(packages):
        if progress_callback:
            progress_callback(pkg_name, i + 1, len(packages))
        yield from index_package(pkg_name, pkg_path)


def index_venv(venv: Path, progress_callback=None) -> Iterator[Symbol]:
    """Index all packages in a venv."""
    site_packages = find_site_packages(venv)
    if site_packages is None:
        return
    yield from index_site_packages(site_packages, progress_callback)
