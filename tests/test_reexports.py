"""Tests for re-export indexing: following __init__.py re-exports to build public API names."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from rex.indexer import (
    ReExport,
    Symbol,
    _find_init_files,
    _resolve_reexports,
    _resolve_relative_import,
    extract_reexports,
    index_directory,
    index_package,
    iter_py_files,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_py(path: Path, source: str) -> Path:
    """Write dedented Python source to a file, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source))
    return path


# ===========================================================================
# Unit tests: _resolve_relative_import
# ===========================================================================


class TestResolveRelativeImport:
    """Test relative-to-absolute module path conversion."""

    def test_level_1_with_module(self):
        # from ._impl.input import slider  (inside marimo._plugins.ui)
        result = _resolve_relative_import(1, "_impl.input", "marimo._plugins.ui")
        assert result == "marimo._plugins.ui._impl.input"

    def test_level_1_without_module(self):
        # from . import something  (inside marimo._plugins.ui)
        result = _resolve_relative_import(1, None, "marimo._plugins.ui")
        assert result == "marimo._plugins.ui"

    def test_level_2_with_module(self):
        # from .._other import foo  (inside marimo._plugins.ui)
        result = _resolve_relative_import(2, "_other", "marimo._plugins.ui")
        assert result == "marimo._plugins._other"

    def test_level_3(self):
        # from ... import bar  (inside marimo._plugins.ui)
        result = _resolve_relative_import(3, None, "marimo._plugins.ui")
        assert result == "marimo"

    def test_level_beyond_top_raises(self):
        with pytest.raises(ValueError, match="beyond top-level"):
            _resolve_relative_import(4, None, "a.b.c")

    def test_level_1_simple_package(self):
        # from .sub import X  (inside pkg)
        result = _resolve_relative_import(1, "sub", "pkg")
        assert result == "pkg.sub"


# ===========================================================================
# Unit tests: extract_reexports
# ===========================================================================


class TestExtractReexports:
    """Test AST-based extraction of re-exports from __init__.py files."""

    def test_from_import(self, tmp_path):
        init = _write_py(
            tmp_path / "__init__.py",
            """\
            from .models import User
            from .views import render
            """,
        )
        results = extract_reexports(init, "mypkg")
        assert len(results) == 2
        assert results[0] == ReExport("User", "mypkg.models", "User", False)
        assert results[1] == ReExport("render", "mypkg.views", "render", False)

    def test_aliased_import(self, tmp_path):
        init = _write_py(
            tmp_path / "__init__.py",
            """\
            from ._internal import Engine as PublicEngine
            """,
        )
        results = extract_reexports(init, "mypkg")
        assert results[0].local_name == "PublicEngine"
        assert results[0].source_name == "Engine"

    def test_absolute_import(self, tmp_path):
        init = _write_py(
            tmp_path / "__init__.py",
            """\
            from other_pkg.sub import Tool
            """,
        )
        results = extract_reexports(init, "mypkg")
        assert results[0].source_module == "other_pkg.sub"
        assert results[0].source_name == "Tool"

    def test_import_as_module(self, tmp_path):
        init = _write_py(
            tmp_path / "__init__.py",
            """\
            import numpy as np
            """,
        )
        results = extract_reexports(init, "mypkg")
        assert results[0] == ReExport("np", "numpy", "numpy", True)

    def test_try_except(self, tmp_path):
        init = _write_py(
            tmp_path / "__init__.py",
            """\
            try:
                from ._accel import fast_func
            except ImportError:
                from ._fallback import fast_func
            """,
        )
        results = extract_reexports(init, "mypkg")
        # Both branches captured
        assert len(results) == 2
        names = [r.local_name for r in results]
        assert names == ["fast_func", "fast_func"]

    def test_if_block(self, tmp_path):
        init = _write_py(
            tmp_path / "__init__.py",
            """\
            import sys
            if sys.version_info >= (3, 11):
                from ._new import feature
            else:
                from ._compat import feature
            """,
        )
        results = extract_reexports(init, "mypkg")
        feature_results = [r for r in results if r.local_name == "feature"]
        assert len(feature_results) == 2

    def test_empty_init(self, tmp_path):
        init = _write_py(tmp_path / "__init__.py", "")
        results = extract_reexports(init, "mypkg")
        assert results == []

    def test_star_import_skipped(self, tmp_path):
        init = _write_py(
            tmp_path / "__init__.py",
            """\
            from .models import *
            from .views import render
            """,
        )
        results = extract_reexports(init, "mypkg")
        # Star import skipped, only explicit import kept
        assert len(results) == 1
        assert results[0].local_name == "render"

    def test_relative_level_2(self, tmp_path):
        init = _write_py(
            tmp_path / "__init__.py",
            """\
            from .._parent import Base
            """,
        )
        results = extract_reexports(init, "top.sub.pkg")
        assert results[0].source_module == "top.sub._parent"

    def test_function_body_ignored(self, tmp_path):
        init = _write_py(
            tmp_path / "__init__.py",
            """\
            from .models import User

            def _lazy_load():
                from .heavy import BigThing
                return BigThing
            """,
        )
        results = extract_reexports(init, "mypkg")
        # Only the top-level import, not the one inside the function
        assert len(results) == 1
        assert results[0].local_name == "User"


# ===========================================================================
# Unit tests: _find_init_files
# ===========================================================================


class TestFindInitFiles:
    """Test recursive __init__.py collection."""

    def test_finds_nested_inits(self, tmp_path):
        pkg = tmp_path / "mypkg"
        _write_py(pkg / "__init__.py", "")
        _write_py(pkg / "sub" / "__init__.py", "")
        _write_py(pkg / "sub" / "deep" / "__init__.py", "")
        _write_py(pkg / "_private" / "__init__.py", "")

        result = _find_init_files(pkg, "mypkg")
        assert "mypkg" in result
        assert "mypkg.sub" in result
        assert "mypkg.sub.deep" in result
        assert "mypkg._private" in result  # walks into _private

    def test_skips_hidden_dirs(self, tmp_path):
        pkg = tmp_path / "mypkg"
        _write_py(pkg / "__init__.py", "")
        _write_py(pkg / ".hidden" / "__init__.py", "")

        result = _find_init_files(pkg, "mypkg")
        assert "mypkg" in result
        assert "mypkg..hidden" not in result
        assert len(result) == 1


# ===========================================================================
# Integration tests: _resolve_reexports
# ===========================================================================


class TestResolveReexports:
    """Integration tests for the full re-export resolution pipeline."""

    def test_symbol_from_private_module_gets_public_name(self, tmp_path):
        """Core case: symbol in _impl gets public qualified_name."""
        pkg = tmp_path / "mypkg"
        _write_py(
            pkg / "__init__.py",
            """\
            from ._impl import Widget
            """,
        )
        _write_py(
            pkg / "_impl.py",
            """\
            class Widget:
                \"\"\"A widget.\"\"\"

                def render(self) -> str:
                    return "<widget>"
            """,
        )

        symbols = list(_resolve_reexports("mypkg", pkg))
        qnames = {s.qualified_name for s in symbols}
        assert "mypkg.Widget" in qnames
        assert "mypkg.Widget.render" in qnames

    def test_nested_chain(self, tmp_path):
        """pkg → pkg._internal.sub → actual definition."""
        pkg = tmp_path / "mypkg"
        _write_py(
            pkg / "__init__.py",
            """\
            from ._internal import sub
            """,
        )
        _write_py(
            pkg / "_internal" / "__init__.py",
            """\
            from . import sub
            """,
        )
        _write_py(
            pkg / "_internal" / "sub" / "__init__.py",
            """\
            from ._core import Engine
            """,
        )
        _write_py(
            pkg / "_internal" / "sub" / "_core.py",
            """\
            class Engine:
                \"\"\"The engine.\"\"\"

                def start(self) -> None:
                    pass
            """,
        )

        symbols = list(_resolve_reexports("mypkg", pkg))
        qnames = {s.qualified_name for s in symbols}
        assert "mypkg.sub.Engine" in qnames
        assert "mypkg.sub.Engine.start" in qnames

    def test_class_methods_remapped(self, tmp_path):
        """Class methods get the public prefix alongside the class."""
        pkg = tmp_path / "mypkg"
        _write_py(
            pkg / "__init__.py",
            """\
            from ._models import User
            """,
        )
        _write_py(
            pkg / "_models.py",
            """\
            class User:
                def __init__(self, name: str) -> None:
                    self.name = name

                def save(self) -> bool:
                    return True
            """,
        )

        symbols = list(_resolve_reexports("mypkg", pkg))
        qnames = {s.qualified_name for s in symbols}
        assert "mypkg.User" in qnames
        assert "mypkg.User.__init__" in qnames
        assert "mypkg.User.save" in qnames

    def test_aliased_module_reexport(self, tmp_path):
        """import _ai as ai → queue with public prefix 'pkg.ai'."""
        pkg = tmp_path / "mypkg"
        _write_py(
            pkg / "__init__.py",
            """\
            from . import _ai as ai
            """,
        )
        _write_py(
            pkg / "_ai" / "__init__.py",
            """\
            from ._core import Model
            """,
        )
        _write_py(
            pkg / "_ai" / "_core.py",
            """\
            class Model:
                def predict(self) -> str:
                    return "yes"
            """,
        )

        symbols = list(_resolve_reexports("mypkg", pkg))
        qnames = {s.qualified_name for s in symbols}
        assert "mypkg.ai.Model" in qnames
        assert "mypkg.ai.Model.predict" in qnames

    def test_circular_import_doesnt_loop(self, tmp_path):
        """Circular re-exports don't cause infinite loop."""
        pkg = tmp_path / "mypkg"
        _write_py(
            pkg / "__init__.py",
            """\
            from . import sub
            """,
        )
        _write_py(
            pkg / "sub" / "__init__.py",
            """\
            from .. import sub  # circular back to parent
            """,
        )

        # Should complete without hanging
        symbols = list(_resolve_reexports("mypkg", pkg))
        # No crash is the main assertion
        assert isinstance(symbols, list)

    def test_public_symbols_not_duplicated(self, tmp_path):
        """Same symbol re-exported from multiple paths → single entry."""
        pkg = tmp_path / "mypkg"
        _write_py(
            pkg / "__init__.py",
            """\
            from ._core import Tool
            from ._compat import Tool
            """,
        )
        _write_py(
            pkg / "_core.py",
            """\
            class Tool:
                pass
            """,
        )
        _write_py(
            pkg / "_compat.py",
            """\
            class Tool:
                pass
            """,
        )

        symbols = list(_resolve_reexports("mypkg", pkg))
        tool_syms = [s for s in symbols if s.name == "Tool"]
        assert len(tool_syms) == 1

    def test_function_reexport(self, tmp_path):
        """Functions (not just classes) are re-exported correctly."""
        pkg = tmp_path / "mypkg"
        _write_py(
            pkg / "__init__.py",
            """\
            from ._utils import helper
            """,
        )
        _write_py(
            pkg / "_utils.py",
            """\
            def helper(x: int) -> int:
                \"\"\"Help.\"\"\"
                return x + 1
            """,
        )

        symbols = list(_resolve_reexports("mypkg", pkg))
        helpers = [s for s in symbols if s.name == "helper"]
        assert len(helpers) == 1
        assert helpers[0].qualified_name == "mypkg.helper"
        assert helpers[0].symbol_type == "function"
        assert "x: int" in helpers[0].signature


# ===========================================================================
# Integration tests: index_package with re-exports
# ===========================================================================


class TestIndexPackageReexports:
    """Test the full two-phase index_package pipeline."""

    def test_private_dirs_skipped_in_walk(self, tmp_path):
        """Phase 1 walk skips _private dirs."""
        pkg = tmp_path / "mypkg"
        _write_py(pkg / "__init__.py", "")
        _write_py(pkg / "public.py", "class Public: pass\n")
        _write_py(pkg / "_private" / "__init__.py", "")
        _write_py(pkg / "_private" / "hidden.py", "class Hidden: pass\n")

        symbols = list(index_package("mypkg", pkg))
        qnames = {s.qualified_name for s in symbols}

        assert "mypkg.public.Public" in qnames
        # _private.hidden.Hidden should NOT appear with its filesystem name
        assert "mypkg._private.hidden.Hidden" not in qnames

    def test_reexported_symbols_have_public_names(self, tmp_path):
        """Phase 2 adds re-exported symbols with public qualified_names."""
        pkg = tmp_path / "mypkg"
        _write_py(
            pkg / "__init__.py",
            """\
            from ._internal import Gadget
            """,
        )
        _write_py(pkg / "_internal.py", "class Gadget:\n    pass\n")

        symbols = list(index_package("mypkg", pkg))
        qnames = {s.qualified_name for s in symbols}

        assert "mypkg.Gadget" in qnames

    def test_file_path_and_line_no_point_to_definition(self, tmp_path):
        """Re-exported symbol's file_path/line_no point to actual source."""
        pkg = tmp_path / "mypkg"
        _write_py(
            pkg / "__init__.py",
            """\
            from ._core import MyClass
            """,
        )
        _write_py(
            pkg / "_core.py",
            """\
            # line 1: comment
            # line 2: comment
            class MyClass:
                pass
            """,
        )

        symbols = list(index_package("mypkg", pkg))
        my_class = next(s for s in symbols if s.qualified_name == "mypkg.MyClass")
        assert my_class.file_path.endswith("_core.py")
        assert my_class.line_no == 3

    def test_index_directory_skips_private(self, tmp_path):
        """index_directory also uses iter_py_files which skips _* dirs."""
        _write_py(tmp_path / "app.py", "class App: pass\n")
        _write_py(tmp_path / "_internal" / "secret.py", "class Secret: pass\n")

        symbols = list(index_directory(tmp_path))
        names = {s.name for s in symbols}
        assert "App" in names
        assert "Secret" not in names

    def test_iter_py_files_skips_underscore_dirs(self, tmp_path):
        """iter_py_files excludes _prefixed directories."""
        pkg = tmp_path / "mypkg"
        _write_py(pkg / "public.py", "x = 1\n")
        _write_py(pkg / "_private" / "hidden.py", "y = 2\n")
        _write_py(pkg / "__pycache__" / "cached.py", "z = 3\n")

        files = list(iter_py_files(pkg))
        file_names = {f.name for f in files}
        assert "public.py" in file_names
        assert "hidden.py" not in file_names
        assert "cached.py" not in file_names

    def test_single_file_module_no_reexports(self, tmp_path):
        """Single .py file module returns symbols without re-export phase."""
        mod = _write_py(
            tmp_path / "single.py",
            """\
            def solo() -> None:
                pass
            """,
        )

        symbols = list(index_package("single", mod))
        assert any(s.name == "solo" for s in symbols)
