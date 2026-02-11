"""Tests for rex indexer and storage modules."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from rex.indexer import Symbol, find_site_packages, find_venv, parse_file
from rex.storage import (
    SearchResult,
    build_index,
    clean_index,
    get_db_path,
    get_members,
    get_stats,
    get_symbol,
    search,
)

PROJECT_DIR = Path("/Users/avykhova/Code/karb/rex")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def db_path_fn(tmp_path_factory):
    """DI function returning a temp DB path for test isolation."""
    db = tmp_path_factory.mktemp("rex") / "test.db"
    return lambda: db


@pytest.fixture(scope="session")
def indexed_db(db_path_fn):
    """Build index into temp DB, return the db_path_fn."""
    venv = find_venv(PROJECT_DIR)
    assert venv is not None, ".venv not found for project"
    build_index(venv, db_path_fn=db_path_fn)
    return db_path_fn


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

    def test_find_venv_returns_none_from_tmp(self, tmp_path, monkeypatch):
        # Mock home to prevent ~/.venv fallback
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "nohome"))
        result = find_venv(Path("/tmp"))
        assert result is None


class TestFindVenvHomeFallback:
    def test_home_venv_fallback(self, tmp_path, monkeypatch):
        """When no .venv in parents, falls back to ~/.venv."""
        home_venv = tmp_path / "fakehome" / ".venv"
        home_venv.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "fakehome"))
        result = find_venv(Path("/tmp"))
        assert result == home_venv

    def test_no_fallback_when_no_home_venv(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "fakehome"))
        result = find_venv(Path("/tmp"))
        assert result is None


class TestFindSitePackages:
    def test_find_site_packages(self):
        venv = find_venv(PROJECT_DIR)
        assert venv is not None
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
    def test_search_typer(self, indexed_db):
        result = search("Typer", db_path_fn=indexed_db)
        assert len(result.symbols) > 0
        names = [s.name for s in result.symbols]
        assert any("Typer" in n for n in names)

    def test_search_basemodel(self, indexed_db):
        result = search("BaseModel", db_path_fn=indexed_db)
        assert len(result.symbols) > 0

    def test_search_empty_returns_empty(self, indexed_db):
        result = search("", db_path_fn=indexed_db)
        assert result.symbols == []

    def test_search_returns_search_result(self, indexed_db):
        result = search("Typer", db_path_fn=indexed_db)
        assert isinstance(result, SearchResult)

    def test_search_special_char_parenthesis(self, indexed_db):
        result = search("(", db_path_fn=indexed_db)
        assert isinstance(result, SearchResult)

    def test_search_special_char_plus(self, indexed_db):
        result = search("+", db_path_fn=indexed_db)
        assert isinstance(result, SearchResult)

    def test_search_special_char_at(self, indexed_db):
        result = search("@", db_path_fn=indexed_db)
        assert isinstance(result, SearchResult)

    def test_search_special_keyword_NOT(self, indexed_db):
        result = search("NOT", db_path_fn=indexed_db)
        assert isinstance(result, SearchResult)

    def test_search_special_char_quote(self, indexed_db):
        result = search('"', db_path_fn=indexed_db)
        assert isinstance(result, SearchResult)

    def test_search_nonexistent(self, indexed_db):
        result = search("nonexistent_xyz_abc", db_path_fn=indexed_db)
        assert result.symbols == []

    def test_search_type_filter(self, indexed_db):
        result = search("BaseModel", db_path_fn=indexed_db, symbol_type="class")
        assert len(result.symbols) > 0
        assert all(s.symbol_type == "class" for s in result.symbols)


class TestGetSymbol:
    def test_get_symbol_existing(self, indexed_db):
        hits = search("Typer", db_path_fn=indexed_db, limit=1).symbols
        assert len(hits) > 0
        qn = hits[0].qualified_name

        sym = get_symbol(qn, db_path_fn=indexed_db)
        assert sym is not None
        assert isinstance(sym, Symbol)
        assert sym.qualified_name == qn

    def test_get_symbol_short_name(self, indexed_db):
        sym = get_symbol("BaseModel", db_path_fn=indexed_db)
        assert sym is not None
        assert sym.name == "BaseModel"
        assert sym.symbol_type == "class"

    def test_get_symbol_nonexistent(self, indexed_db):
        result = get_symbol("nonexistent.does.not.exist", db_path_fn=indexed_db)
        assert result is None


class TestGetMembers:
    def test_get_members_known_class(self, indexed_db):
        hits = search("BaseModel", db_path_fn=indexed_db, symbol_type="class").symbols
        assert len(hits) > 0
        qn = hits[0].qualified_name

        members = get_members(qn, db_path_fn=indexed_db)
        assert isinstance(members, list)
        assert len(members) > 0
        for m in members:
            assert m.qualified_name.startswith(qn + ".")

    def test_get_members_short_name(self, indexed_db):
        members = get_members("BaseModel", db_path_fn=indexed_db)
        assert len(members) > 0
        assert all(m.qualified_name.startswith("pydantic.") for m in members)

    def test_get_members_dunders_sorted_last(self, indexed_db):
        members = get_members("BaseModel", db_path_fn=indexed_db)
        regular = [m for m in members if not m.name.startswith("__")]
        dunders = [m for m in members if m.name.startswith("__")]
        if regular and dunders:
            # All regular members should appear before all dunders
            last_regular = members.index(regular[-1])
            first_dunder = members.index(dunders[0])
            assert last_regular < first_dunder


class TestGetStats:
    def test_get_stats_keys(self, indexed_db):
        stats = get_stats(db_path_fn=indexed_db)
        assert isinstance(stats, dict)
        assert "total_symbols" in stats
        assert "by_type" in stats
        assert "packages" in stats
        assert "db_path" in stats
        assert stats["total_symbols"] > 0


class TestGetDbPathGlobal:
    def test_returns_global_path(self):
        db_path = get_db_path()
        assert str(db_path).endswith("rex/rex.db")
        assert ".local/state/rex" in str(db_path)


class TestIncrementalIndex:
    def test_incremental_skips_unchanged(self, db_path_fn):
        """Index once, index again -- second time returns -1 (up to date)."""
        venv = find_venv(PROJECT_DIR)
        assert venv is not None

        # Use a fresh DB for this test
        count1 = build_index(venv, db_path_fn=db_path_fn, force=True)
        assert count1 > 0

        count2 = build_index(venv, db_path_fn=db_path_fn)
        assert count2 == -1  # nothing changed


class TestCleanIndex:
    def test_clean_removes_dead_packages(self, tmp_path):
        """Index a temp package, delete it, clean should remove it."""
        db = tmp_path / "clean_test.db"
        db_fn = lambda: db

        # Create a fake site-packages with one package
        fake_venv = tmp_path / "fakevenv"
        sp = fake_venv / "lib" / "python3.14" / "site-packages"
        pkg_dir = sp / "fakepkg"
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "__init__.py").write_text('"""Fake package."""\n')

        build_index(fake_venv, db_path_fn=db_fn, force=True)

        # Verify package was indexed
        result = search("fakepkg", db_path_fn=db_fn)
        assert len(result.symbols) > 0

        # Delete the package directory
        import shutil

        shutil.rmtree(str(pkg_dir))

        # Clean should find it
        removed = clean_index(db_path_fn=db_fn)
        assert "fakepkg" in removed

        # Verify fakepkg is gone from search
        result = search("fakepkg", db_path_fn=db_fn)
        assert not any(s.qualified_name.startswith("fakepkg.") for s in result.symbols)

    def test_clean_keeps_live_packages(self, tmp_path):
        """Clean should not remove packages that still exist."""
        db = tmp_path / "clean_live.db"
        db_fn = lambda: db

        fake_venv = tmp_path / "fakevenv"
        sp = fake_venv / "lib" / "python3.14" / "site-packages"
        pkg_dir = sp / "livepkg"
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "__init__.py").write_text('"""Live package."""\n')

        build_index(fake_venv, db_path_fn=db_fn, force=True)

        removed = clean_index(db_path_fn=db_fn)
        assert removed == []


class TestPackagesTable:
    def test_packages_populated_after_index(self, tmp_path):
        """After indexing, packages table should have entries."""
        import sqlite3

        db = tmp_path / "pkg_table.db"
        db_fn = lambda: db

        fake_venv = tmp_path / "fakevenv"
        sp = fake_venv / "lib" / "python3.14" / "site-packages"
        pkg_dir = sp / "testpkg"
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "__init__.py").write_text('"""Test."""\ndef hello(): pass\n')

        build_index(fake_venv, db_path_fn=db_fn, force=True)

        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM packages").fetchall()
        conn.close()

        assert len(rows) > 0
        names = [r["name"] for r in rows]
        assert "testpkg" in names


# ---------------------------------------------------------------------------
# C) Bug regression tests
# ---------------------------------------------------------------------------


class TestBugRegressions:
    def test_empty_query_returns_empty(self, indexed_db):
        result = search("", db_path_fn=indexed_db)
        assert result.symbols == []

    def test_whitespace_query_returns_empty(self, indexed_db):
        result = search("   ", db_path_fn=indexed_db)
        assert result.symbols == []

    @pytest.mark.parametrize("char", ["(", "+", "@", "NOT", '"'])
    def test_special_chars_dont_crash(self, indexed_db, char):
        # Should not raise any exception
        result = search(char, db_path_fn=indexed_db)
        assert isinstance(result, SearchResult)

    def test_dotted_query_fallback(self, indexed_db):
        """Dotted query that won't match a class should fall back gracefully."""
        result = search("SomeNonexistentClass.method", db_path_fn=indexed_db)
        assert isinstance(result, SearchResult)
