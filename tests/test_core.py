"""Tests for rex indexer and storage modules."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from rex.indexer import Symbol, find_site_packages, find_venv, parse_file
from rex.storage import build_index, get_members, get_stats, get_symbol, search

PROJECT_DIR = Path("/Users/avykhova/Code/karb/rex")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def venv():
    """Locate the project .venv."""
    v = find_venv(PROJECT_DIR)
    assert v is not None, ".venv not found for project"
    return v


@pytest.fixture(scope="session")
def ensure_index(venv):
    """Ensure the index is built before storage tests run."""
    build_index(venv)
    return venv


# ---------------------------------------------------------------------------
# A) Indexer tests
# ---------------------------------------------------------------------------


class TestFindVenv:
    def test_find_venv_from_project_dir(self):
        result = find_venv(PROJECT_DIR)
        assert result is not None
        assert result.name == ".venv"
        assert result.is_dir()

    def test_find_venv_from_subdirectory(self):
        result = find_venv(PROJECT_DIR / "src")
        assert result is not None
        assert result.name == ".venv"

    def test_find_venv_returns_none_from_tmp(self):
        result = find_venv(Path("/tmp"))
        assert result is None


class TestFindSitePackages:
    def test_find_site_packages(self, venv):
        sp = find_site_packages(venv)
        assert sp is not None
        assert sp.name == "site-packages"
        assert sp.is_dir()


class TestParseFile:
    def test_parse_extracts_class_and_methods(self, tmp_path):
        source = textwrap.dedent('''\
            """Module doc."""

            class Greeter:
                """A greeter class."""

                def greet(self, name: str) -> str:
                    """Say hello."""
                    return f"Hello, {name}"

            def standalone(x, y=10):
                """A standalone function."""
                return x + y
        ''')
        py_file = tmp_path / "example.py"
        py_file.write_text(source)

        symbols = parse_file(py_file, "example")

        names = [s.name for s in symbols]
        assert "example" in names  # module-level docstring symbol
        assert "Greeter" in names
        assert "greet" in names
        assert "standalone" in names

        # Check types
        by_name = {s.name: s for s in symbols}
        assert by_name["example"].symbol_type == "module"
        assert by_name["Greeter"].symbol_type == "class"
        assert by_name["greet"].symbol_type == "method"
        assert by_name["standalone"].symbol_type == "function"

        # Check signatures
        assert "name: str" in by_name["greet"].signature
        assert "-> str" in by_name["greet"].signature
        assert "y = 10" in by_name["standalone"].signature

        # Check qualified names
        assert by_name["greet"].qualified_name == "example.Greeter.greet"
        assert by_name["standalone"].qualified_name == "example.standalone"

    def test_parse_empty_file(self, tmp_path):
        py_file = tmp_path / "empty.py"
        py_file.write_text("")
        symbols = parse_file(py_file, "empty")
        assert symbols == []

    def test_parse_syntax_error(self, tmp_path):
        py_file = tmp_path / "bad.py"
        py_file.write_text("def broken(:\n")
        symbols = parse_file(py_file, "bad")
        assert symbols == []


# ---------------------------------------------------------------------------
# B) Storage tests (against real index)
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_typer(self, ensure_index):
        results = search("Typer", venv=ensure_index)
        assert len(results) > 0
        names = [s.name for s in results]
        assert any("Typer" in n for n in names)

    def test_search_basemodel(self, ensure_index):
        results = search("BaseModel", venv=ensure_index)
        assert len(results) > 0

    def test_search_empty_returns_empty(self, ensure_index):
        results = search("", venv=ensure_index)
        assert results == []

    def test_search_special_char_parenthesis(self, ensure_index):
        results = search("(", venv=ensure_index)
        assert isinstance(results, list)

    def test_search_special_char_plus(self, ensure_index):
        results = search("+", venv=ensure_index)
        assert isinstance(results, list)

    def test_search_special_char_at(self, ensure_index):
        results = search("@", venv=ensure_index)
        assert isinstance(results, list)

    def test_search_special_keyword_NOT(self, ensure_index):
        results = search("NOT", venv=ensure_index)
        assert isinstance(results, list)

    def test_search_special_char_quote(self, ensure_index):
        results = search('"', venv=ensure_index)
        assert isinstance(results, list)

    def test_search_nonexistent(self, ensure_index):
        results = search("nonexistent_xyz_abc", venv=ensure_index)
        assert results == []

    def test_search_type_filter(self, ensure_index):
        results = search("BaseModel", venv=ensure_index, symbol_type="class")
        assert len(results) > 0
        assert all(s.symbol_type == "class" for s in results)


class TestGetSymbol:
    def test_get_symbol_existing(self, ensure_index):
        hits = search("Typer", venv=ensure_index, limit=1)
        assert len(hits) > 0
        qn = hits[0].qualified_name

        sym = get_symbol(qn, venv=ensure_index)
        assert sym is not None
        assert isinstance(sym, Symbol)
        assert sym.qualified_name == qn

    def test_get_symbol_nonexistent(self, ensure_index):
        result = get_symbol("nonexistent.does.not.exist", venv=ensure_index)
        assert result is None


class TestGetMembers:
    def test_get_members_known_class(self, ensure_index):
        hits = search("BaseModel", venv=ensure_index, symbol_type="class")
        assert len(hits) > 0
        qn = hits[0].qualified_name

        members = get_members(qn, venv=ensure_index)
        assert isinstance(members, list)
        assert len(members) > 0
        for m in members:
            assert m.qualified_name.startswith(qn + ".")


class TestGetStats:
    def test_get_stats_keys(self, ensure_index):
        stats = get_stats(venv=ensure_index)
        assert isinstance(stats, dict)
        assert "total_symbols" in stats
        assert "by_type" in stats
        assert "packages" in stats
        assert "venv" in stats
        assert "db_path" in stats
        assert stats["total_symbols"] > 0


# ---------------------------------------------------------------------------
# C) Bug regression tests
# ---------------------------------------------------------------------------


class TestBugRegressions:
    def test_empty_query_returns_empty(self, ensure_index):
        results = search("", venv=ensure_index)
        assert results == []

    def test_whitespace_query_returns_empty(self, ensure_index):
        results = search("   ", venv=ensure_index)
        assert results == []

    @pytest.mark.parametrize("char", ["(", "+", "@", "NOT", '"'])
    def test_special_chars_dont_crash(self, ensure_index, char):
        # Should not raise any exception
        results = search(char, venv=ensure_index)
        assert isinstance(results, list)

    def test_dotted_query_fallback(self, ensure_index):
        """Dotted query that won't match a class should fall back gracefully."""
        results = search("SomeNonexistentClass.method", venv=ensure_index)
        assert isinstance(results, list)
