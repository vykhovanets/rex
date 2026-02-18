"""AST-based indexer for Python packages.

Walks .venv, parses all .py files, extracts symbols without importing.
"""

from __future__ import annotations

import ast
import logging
import os
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReExport:
    """A re-exported name found in an __init__.py."""

    local_name: str  # name as exposed (alias or original)
    source_module: str  # absolute dotted source module
    source_name: str  # name in source (before 'as')
    is_module: bool  # True when importing a subpackage/module


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



def _resolve_relative_import(
    level: int, module: str | None, current_package: str,
) -> str:
    """Convert a relative import to an absolute module path.

    ``from ._impl.input import slider`` inside ``marimo._plugins.ui``
    with *level=1* and *module='_impl.input'* yields
    ``marimo._plugins.ui._impl.input``.
    """
    parts = current_package.split(".")
    if level > len(parts):
        raise ValueError(
            f"Attempted relative import beyond top-level package: "
            f"level={level}, package={current_package!r}"
        )
    base = ".".join(parts[: len(parts) - level + 1])
    if module:
        return f"{base}.{module}"
    return base


def extract_reexports(init_path: Path, module_name: str) -> list[ReExport]:
    """Parse an ``__init__.py`` and return structured re-export list.

    Walks top-level statements plus the bodies of ``try/except`` and ``if``
    blocks (but NOT function/class bodies) looking for ``import`` and
    ``from … import`` statements.
    """
    try:
        source = init_path.read_text(encoding="utf-8", errors="replace")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source, filename=str(init_path))
    except (SyntaxError, ValueError):
        return []

    results: list[ReExport] = []

    def _walk_stmts(stmts: list[ast.stmt]) -> None:
        for node in stmts:
            if isinstance(node, ast.ImportFrom):
                if node.names and len(node.names) == 1 and node.names[0].name == "*":
                    log.debug("Skipping star import in %s", module_name)
                    continue
                if node.level and node.level > 0:
                    # relative import
                    try:
                        source_mod = _resolve_relative_import(
                            node.level, node.module, module_name,
                        )
                    except ValueError:
                        continue
                else:
                    source_mod = node.module or ""
                for alias in node.names:
                    results.append(ReExport(
                        local_name=alias.asname or alias.name,
                        source_module=source_mod,
                        source_name=alias.name,
                        is_module=False,
                    ))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    results.append(ReExport(
                        local_name=alias.asname or alias.name,
                        source_module=alias.name,
                        source_name=alias.name,
                        is_module=True,
                    ))
            # Recurse into compound statements (try/except, if/else)
            elif isinstance(node, (ast.Try, )):
                _walk_stmts(node.body)
                for handler in node.handlers:
                    _walk_stmts(handler.body)
                _walk_stmts(node.orelse)
                _walk_stmts(node.finalbody)
            elif isinstance(node, ast.If):
                _walk_stmts(node.body)
                _walk_stmts(node.orelse)
            # Skip FunctionDef, AsyncFunctionDef, ClassDef bodies

    _walk_stmts(tree.body)
    return results


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


def _find_init_files(package_path: Path, package_name: str) -> dict[str, Path]:
    """Recursively collect ``__init__.py`` files under *package_path*.

    Walks into ``_``-prefixed dirs (private sub-packages often hold the
    actual definitions). Returns ``{dotted_module_name: init_path}``.
    """
    result: dict[str, Path] = {}
    init = package_path / "__init__.py"
    if init.is_file():
        result[package_name] = init

    for root, dirs, files in os.walk(package_path):
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and d != "__pycache__"
        ]
        for d in list(dirs):
            sub_init = Path(root) / d / "__init__.py"
            if sub_init.is_file():
                rel = Path(root) / d
                rel_parts = rel.relative_to(package_path.parent).parts
                mod_name = ".".join(rel_parts)
                result[mod_name] = sub_init
    return result


def _module_to_path(module_name: str, package_path: Path) -> Path | None:
    """Convert a dotted module name to a ``.py`` or ``.pyi`` file path.

    Used for resolving source modules that are plain files (not packages).
    """
    parts = module_name.split(".")
    base = package_path.parent  # site-packages
    rel = Path(*parts)
    for suffix in (".py", ".pyi"):
        candidate = base / rel.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def _parse_module_cached(
    module_name: str,
    init_files: dict[str, Path],
    package_path: Path,
    cache: dict[str, list[Symbol]],
) -> list[Symbol]:
    """Parse a module (via ``__init__.py`` or ``.py``), with caching."""
    if module_name in cache:
        return cache[module_name]

    path: Path | None = init_files.get(module_name)
    if path is None:
        path = _module_to_path(module_name, package_path)
    if path is None:
        cache[module_name] = []
        return []

    symbols = parse_file(path, module_name)
    cache[module_name] = symbols
    return symbols


def _resolve_reexports(
    package_name: str, package_path: Path,
) -> Iterator[Symbol]:
    """Follow ``__init__.py`` re-exports, yielding symbols with public names.

    BFS from the top-level ``__init__.py``.  For each init file, calls
    ``extract_reexports()`` and either queues sub-packages (module re-exports)
    or resolves individual symbols (symbol re-exports).
    """
    init_files = _find_init_files(package_path, package_name)
    if package_name not in init_files:
        return  # no top-level __init__.py

    parse_cache: dict[str, list[Symbol]] = {}
    seen_qnames: set[str] = set()
    visited_mods: set[str] = set()  # source modules already processed

    # queue entries: (source_module_name, public_prefix)
    queue: deque[tuple[str, str]] = deque()
    queue.append((package_name, package_name))

    while queue:
        src_mod, pub_prefix = queue.popleft()
        if src_mod in visited_mods:
            continue
        visited_mods.add(src_mod)

        init_path = init_files.get(src_mod)
        if init_path is None:
            continue

        reexports = extract_reexports(init_path, src_mod)

        for rex in reexports:
            if rex.is_module:
                # ``import subpkg`` or ``import subpkg as alias``
                # Queue sub-package with its public prefix
                sub_pub = f"{pub_prefix}.{rex.local_name}"
                queue.append((rex.source_module, sub_pub))
                continue

            # Check if the source_name is itself a sub-package with __init__
            sub_mod = f"{rex.source_module}.{rex.source_name}"
            if sub_mod in init_files:
                sub_pub = f"{pub_prefix}.{rex.local_name}"
                queue.append((sub_mod, sub_pub))
                continue

            # Symbol re-export — parse the source module and find the symbol
            symbols = _parse_module_cached(
                rex.source_module, init_files, package_path, parse_cache,
            )
            for sym in symbols:
                if sym.name != rex.source_name:
                    continue

                # Build public qualified name
                if sym.symbol_type in ("class", "module"):
                    pub_qname = f"{pub_prefix}.{rex.local_name}"
                elif sym.symbol_type in ("method", "async_method"):
                    # Method of a re-exported class — skip standalone,
                    # these are yielded when the parent class is processed
                    continue
                else:
                    pub_qname = f"{pub_prefix}.{rex.local_name}"

                if pub_qname in seen_qnames:
                    continue
                seen_qnames.add(pub_qname)

                yield Symbol(
                    name=rex.local_name,
                    qualified_name=pub_qname,
                    symbol_type=sym.symbol_type,
                    signature=sym.signature,
                    docstring=sym.docstring,
                    file_path=sym.file_path,
                    line_no=sym.line_no,
                    bases=sym.bases,
                    return_annotation=sym.return_annotation,
                )

                # If it's a class, also yield its methods with remapped names
                if sym.symbol_type == "class":
                    class_prefix_old = f"{sym.qualified_name}."
                    for child in symbols:
                        if not child.qualified_name.startswith(class_prefix_old):
                            continue
                        # Only direct children (one level deep)
                        rest = child.qualified_name[len(class_prefix_old):]
                        if "." in rest:
                            continue
                        child_pub = f"{pub_qname}.{child.name}"
                        if child_pub in seen_qnames:
                            continue
                        seen_qnames.add(child_pub)
                        yield Symbol(
                            name=child.name,
                            qualified_name=child_pub,
                            symbol_type=child.symbol_type,
                            signature=child.signature,
                            docstring=child.docstring,
                            file_path=child.file_path,
                            line_no=child.line_no,
                            bases=child.bases,
                            return_annotation=child.return_annotation,
                        )
                break  # found the symbol, stop iterating


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
    """Index a single package, yielding symbols.

    **Phase 1**: Walk public directories (``iter_py_files`` skips ``_*`` dirs)
    and yield symbols with filesystem-based qualified names.

    **Phase 2**: Follow ``__init__.py`` re-exports (``_resolve_reexports``)
    to yield symbols from private modules with their *public* qualified names.

    A ``seen_qnames`` set prevents duplicates across both phases.
    """
    if package_path.is_file():
        # Single .py file module — no re-exports to resolve
        yield from parse_file(package_path, package_name)
        return

    seen_qnames: set[str] = set()

    # Phase 1: walk public dirs (iter_py_files now skips _* dirs)
    for file_path in iter_py_files(package_path):
        rel_path = file_path.relative_to(package_path.parent)
        parts = list(rel_path.parts)

        if parts[-1] in ("__init__.py", "__init__.pyi"):
            parts = parts[:-1]
        else:
            parts[-1] = Path(parts[-1]).stem  # Remove .py/.pyi

        module_name = ".".join(parts)
        for sym in parse_file(file_path, module_name):
            seen_qnames.add(sym.qualified_name)
            yield sym

    # Phase 2: resolve re-exports from __init__.py files
    for sym in _resolve_reexports(package_name, package_path):
        if sym.qualified_name not in seen_qnames:
            seen_qnames.add(sym.qualified_name)
            yield sym


def iter_py_files(directory: Path) -> Iterator[Path]:
    """Yield .py/.pyi file paths under directory, skipping private/hidden dirs.

    Skips ``_``-prefixed, ``.``-prefixed, and ``__pycache__`` directories.
    When both foo.py and foo.pyi exist, only foo.py is yielded.
    Stubs are used as fallback for compiled extensions (e.g. .so/.pyd).
    """
    for root, dirs, files in os.walk(directory):
        dirs[:] = [
            d for d in dirs
            if not d.startswith((".", "_"))
            and d != "__pycache__"
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
