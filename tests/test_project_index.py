"""Tests for project source code indexing (index_directory + build_index project_dirs)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from rex.indexer import Symbol, find_venv, parse_file
from rex.storage import build_index, search

PROJECT_DIR = Path("/Users/avykhova/Code/karb/rex")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_py(path: Path, source: str) -> Path:
    """Write dedented Python source to a file, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source))
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_project(tmp_path):
    """Create a small project tree under tmp_path for indexing tests."""
    # top-level module
    _write_py(
        tmp_path / "app.py",
        '''\
        """App module."""

        def run_app(config: dict) -> None:
            """Start the application."""
            pass
        ''',
    )

    # package with __init__ and submodule
    _write_py(
        tmp_path / "mypackage" / "__init__.py",
        '''\
        """My package."""
        ''',
    )
    _write_py(
        tmp_path / "mypackage" / "models.py",
        '''\
        """Models module."""

        class UserModel:
            """A user model."""

            def save(self) -> bool:
                """Persist the user."""
                return True

        class AdminModel(UserModel):
            """An admin model."""
            pass
        ''',
    )

    # nested subpackage
    _write_py(
        tmp_path / "mypackage" / "utils" / "__init__.py",
        "",
    )
    _write_py(
        tmp_path / "mypackage" / "utils" / "helpers.py",
        '''\
        def helper_func(x: int, y: int = 0) -> int:
            """Add two numbers."""
            return x + y
        ''',
    )

    # dirs that should be skipped
    _write_py(
        tmp_path / "__pycache__" / "cached.py",
        "class Cached: pass\n",
    )
    _write_py(
        tmp_path / ".hidden" / "secret.py",
        "class Secret: pass\n",
    )
    _write_py(
        tmp_path / ".venv" / "lib" / "pkg.py",
        "class VenvPkg: pass\n",
    )

    return tmp_path


@pytest.fixture(scope="session")
def venv():
    """Locate the project .venv."""
    v = find_venv(PROJECT_DIR)
    assert v is not None, ".venv not found for project"
    return v


@pytest.fixture(scope="session")
def ensure_index(venv):
    """Ensure the venv index is built before tests run."""
    build_index(venv)
    return venv


# ---------------------------------------------------------------------------
# A) index_directory basics
# ---------------------------------------------------------------------------


class TestIndexDirectoryBasics:
    """Test that index_directory walks a directory and yields symbols."""


    def test_extracts_classes_and_functions(self, sample_project):
        from rex.indexer import index_directory

        symbols = list(index_directory(sample_project))
        names = [s.name for s in symbols]

        assert "run_app" in names
        assert "UserModel" in names
        assert "AdminModel" in names
        assert "save" in names
        assert "helper_func" in names


    def test_handles_nested_subdirectories(self, sample_project):
        from rex.indexer import index_directory

        symbols = list(index_directory(sample_project))
        qualified_names = [s.qualified_name for s in symbols]

        # helpers.py is inside mypackage/utils/
        helper_matches = [qn for qn in qualified_names if "helper_func" in qn]
        assert len(helper_matches) > 0


    def test_skips_pycache_hidden_and_venv_dirs(self, sample_project):
        from rex.indexer import index_directory

        symbols = list(index_directory(sample_project))
        names = [s.name for s in symbols]

        assert "Cached" not in names, "__pycache__ should be skipped"
        assert "Secret" not in names, "hidden dirs should be skipped"
        assert "VenvPkg" not in names, ".venv dirs should be skipped"


    def test_empty_directory_returns_empty(self, tmp_path):
        from rex.indexer import index_directory

        symbols = list(index_directory(tmp_path))
        assert symbols == []


    def test_symbol_types_are_correct(self, sample_project):
        from rex.indexer import index_directory

        symbols = list(index_directory(sample_project))
        by_name = {s.name: s for s in symbols}

        assert by_name["UserModel"].symbol_type == "class"
        assert by_name["run_app"].symbol_type == "function"
        assert by_name["save"].symbol_type == "method"


    def test_signatures_are_extracted(self, sample_project):
        from rex.indexer import index_directory

        symbols = list(index_directory(sample_project))
        by_name = {s.name: s for s in symbols}

        assert "config: dict" in by_name["run_app"].signature
        assert "-> None" in by_name["run_app"].signature
        assert "y: int = 0" in by_name["helper_func"].signature


# ---------------------------------------------------------------------------
# B) build_index with project_dirs
# ---------------------------------------------------------------------------


class TestBuildIndexWithProjectDirs:
    """Test that build_index accepts project_dirs and indexes them."""


    def test_project_symbols_appear_in_search(self, sample_project, venv):
        build_index(venv, force=True, project_dirs=[sample_project])

        results = search("UserModel", venv=venv)
        names = [s.name for s in results]
        assert "UserModel" in names


    def test_project_and_venv_symbols_coexist(self, sample_project, venv):
        build_index(venv, force=True, project_dirs=[sample_project])

        # Project symbol
        proj_results = search("UserModel", venv=venv)
        assert len(proj_results) > 0

        # Venv symbol (typer is installed in the venv)
        venv_results = search("Typer", venv=venv)
        assert len(venv_results) > 0


# ---------------------------------------------------------------------------
# C) Search finds project symbols end-to-end
# ---------------------------------------------------------------------------


class TestSearchFindsProjectSymbols:
    """End-to-end: index project dirs, then search for unique symbols."""


    def test_search_finds_unique_class(self, tmp_path, venv):
        _write_py(
            tmp_path / "unique_mod.py",
            '''\
            class UniqueTestClassXyz123:
                """A very unique class for testing."""

                def unique_method_abc789(self) -> str:
                    return "hello"
            ''',
        )

        build_index(venv, force=True, project_dirs=[tmp_path])

        results = search("UniqueTestClassXyz123", venv=venv)
        assert len(results) > 0
        assert any(s.name == "UniqueTestClassXyz123" for s in results)


    def test_search_finds_unique_function(self, tmp_path, venv):
        _write_py(
            tmp_path / "funcs.py",
            '''\
            def standalone_unique_func_qrs456(x: int) -> int:
                """A unique standalone function."""
                return x * 2
            ''',
        )

        build_index(venv, force=True, project_dirs=[tmp_path])

        results = search("standalone_unique_func_qrs456", venv=venv)
        assert len(results) > 0
        assert any(s.name == "standalone_unique_func_qrs456" for s in results)


    def test_search_finds_unique_method(self, tmp_path, venv):
        _write_py(
            tmp_path / "with_method.py",
            '''\
            class SomeHolder:
                def unique_method_holder_xyz(self) -> None:
                    pass
            ''',
        )

        build_index(venv, force=True, project_dirs=[tmp_path])

        results = search("unique_method_holder_xyz", venv=venv)
        assert len(results) > 0
        assert any(s.name == "unique_method_holder_xyz" for s in results)
