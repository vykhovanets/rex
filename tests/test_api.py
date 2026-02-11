"""Tests for rex.api -- the shared formatting/lookup layer.

These define the expected interface that both CLI and MCP should use,
eliminating duplicated formatting logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rex.indexer import Symbol
from rex.api import format_symbol_detail, format_symbol_line, search_suggestion, show_symbol
from rex.storage import SearchResult, search


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
    from rex.indexer import find_venv
    from rex.storage import build_index

    venv = find_venv(Path("/Users/avykhova/Code/karb/rex"))
    assert venv is not None, ".venv not found"
    build_index(venv, db_path_fn=db_path_fn)
    return db_path_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_symbol(
    *,
    name: str = "greet",
    qualified_name: str = "mypackage.utils.Greeter.greet",
    symbol_type: str = "method",
    signature: str | None = "(self, name: str) -> str",
    docstring: str | None = "Say hello to someone.",
    file_path: str = "/path/to/mypackage/utils.py",
    line_no: int = 42,
    bases: tuple[str, ...] = (),
    return_annotation: str | None = "str",
) -> Symbol:
    return Symbol(
        name=name,
        qualified_name=qualified_name,
        symbol_type=symbol_type,
        signature=signature,
        docstring=docstring,
        file_path=file_path,
        line_no=line_no,
        bases=bases,
        return_annotation=return_annotation,
    )


# ---------------------------------------------------------------------------
# A) format_symbol_line
# ---------------------------------------------------------------------------


class TestFormatSymbolLine:
    """format_symbol_line(sym, sig_max=50) -> str

    Produces a single-line summary suitable for search result listings.
    Must contain at minimum: symbol type, name, and truncated signature.
    """

    def test_returns_string(self):
        sym = _make_symbol()
        result = format_symbol_line(sym)
        assert isinstance(result, str)

    def test_contains_name(self):
        sym = _make_symbol(name="my_func")
        result = format_symbol_line(sym)
        assert "my_func" in result

    def test_contains_symbol_type(self):
        sym = _make_symbol(symbol_type="function")
        result = format_symbol_line(sym)
        assert "function" in result

    def test_contains_signature(self):
        sym = _make_symbol(signature="(x, y)")
        result = format_symbol_line(sym)
        assert "(x, y)" in result

    def test_no_signature_does_not_crash(self):
        sym = _make_symbol(signature=None)
        result = format_symbol_line(sym)
        assert isinstance(result, str)
        assert sym.name in result

    def test_long_signature_truncated_at_default(self):
        long_sig = "(" + ", ".join(f"arg{i}: int" for i in range(20)) + ")"
        assert len(long_sig) > 50
        sym = _make_symbol(signature=long_sig)
        result = format_symbol_line(sym)
        # The signature portion in the result should be <= 50 chars
        # and end with "..." if truncated
        assert "..." in result

    def test_long_signature_truncated_custom_max(self):
        sig = "(a, b, c, d, e, f, g)"  # 21 chars
        sym = _make_symbol(signature=sig)
        result = format_symbol_line(sym, sig_max=10)
        assert "..." in result

    def test_short_signature_not_truncated(self):
        sym = _make_symbol(signature="(x)")
        result = format_symbol_line(sym, sig_max=50)
        assert "..." not in result
        assert "(x)" in result

    def test_is_single_line(self):
        sym = _make_symbol()
        result = format_symbol_line(sym)
        assert "\n" not in result


# ---------------------------------------------------------------------------
# B) format_symbol_detail
# ---------------------------------------------------------------------------


class TestFormatSymbolDetail:
    """format_symbol_detail(sym) -> str

    Produces multi-line plain text detail for a single symbol.
    Used by both CLI `show` command and MCP `rex_show` tool.
    """

    def test_returns_string(self):
        sym = _make_symbol()
        result = format_symbol_detail(sym)
        assert isinstance(result, str)

    def test_contains_qualified_name(self):
        sym = _make_symbol(qualified_name="pkg.mod.Cls.method")
        result = format_symbol_detail(sym)
        assert "pkg.mod.Cls.method" in result

    def test_contains_symbol_type(self):
        sym = _make_symbol(symbol_type="class")
        result = format_symbol_detail(sym)
        assert "class" in result

    def test_contains_signature_when_present(self):
        sym = _make_symbol(signature="(self, x: int)")
        result = format_symbol_detail(sym)
        assert "(self, x: int)" in result

    def test_no_signature_does_not_crash(self):
        sym = _make_symbol(signature=None)
        result = format_symbol_detail(sym)
        assert isinstance(result, str)

    def test_contains_docstring_when_present(self):
        sym = _make_symbol(docstring="This function does something useful.")
        result = format_symbol_detail(sym)
        assert "This function does something useful." in result

    def test_no_docstring_does_not_crash(self):
        sym = _make_symbol(docstring=None)
        result = format_symbol_detail(sym)
        assert isinstance(result, str)

    def test_long_docstring_truncated(self):
        long_doc = "A" * 2000
        sym = _make_symbol(docstring=long_doc)
        result = format_symbol_detail(sym)
        # Should truncate around 1500 chars (matching current behavior)
        assert len(result) < len(long_doc) + 200
        assert "..." in result

    def test_contains_file_location(self):
        sym = _make_symbol(file_path="/some/path.py", line_no=99)
        result = format_symbol_detail(sym)
        assert "/some/path.py" in result
        assert "99" in result

    def test_is_multiline(self):
        sym = _make_symbol()
        result = format_symbol_detail(sym)
        assert "\n" in result


# ---------------------------------------------------------------------------
# C) show_symbol
# ---------------------------------------------------------------------------


class TestShowSymbol:
    """show_symbol(name, db_path_fn) -> Symbol | list[str]

    Exact lookup by qualified name. If not found, falls back to fuzzy
    search and returns a list of suggestion strings ("did you mean").
    """

    def test_exact_match_returns_symbol(self, indexed_db):
        # Find a real qualified name via storage layer
        from rex.storage import search as raw_search

        hits = raw_search("Symbol", db_path_fn=indexed_db, limit=1).symbols
        if not hits:
            pytest.skip("No symbols indexed -- cannot test exact match")
        qn = hits[0].qualified_name

        result = show_symbol(qn, db_path_fn=indexed_db)
        assert isinstance(result, Symbol)
        assert result.qualified_name == qn

    def test_nonexistent_returns_suggestions(self, indexed_db):
        # Not an exact qualified name, but a valid prefix that FTS can match
        result = show_symbol("BaseMod", db_path_fn=indexed_db)
        assert isinstance(result, list)
        assert len(result) > 0
        for s in result:
            assert isinstance(s, str)

    def test_completely_unknown_returns_empty_list(self, indexed_db):
        result = show_symbol("zzz_nonexistent_xyz.does.not.exist", db_path_fn=indexed_db)
        assert isinstance(result, list)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# D) search_suggestion
# ---------------------------------------------------------------------------


class TestSearchSuggestion:
    """search_suggestion(query, result) -> str | None

    Context-aware hint based on result quality and environment.
    Returns None for exact FTS5 matches, hints for everything else.
    """

    def test_exact_match_no_suggestion(self, indexed_db):
        result = search("BaseModel", db_path_fn=indexed_db)
        assert search_suggestion("BaseModel", result) is None

    def test_fuzzy_only_in_project_shows_approximate(self, indexed_db):
        result = search("BaseModl", db_path_fn=indexed_db)
        hint = search_suggestion("BaseModl", result)
        assert hint is not None
        assert "approximate" in hint.lower()
        assert "uv add" in hint
        assert "Not in a Python project" not in hint

    def test_fuzzy_only_outside_project(self, indexed_db, monkeypatch):
        monkeypatch.setattr("rex.api.find_venv", lambda: None)
        result = search("BaseModl", db_path_fn=indexed_db)
        hint = search_suggestion("BaseModl", result)
        assert hint is not None
        assert "approximate" in hint.lower()
        assert "Not in a Python project" in hint

    def test_empty_outside_project(self, monkeypatch):
        monkeypatch.setattr("rex.api.find_venv", lambda: None)
        result = SearchResult()
        hint = search_suggestion("torch", result)
        assert "Not in a Python project" in hint

    def test_empty_in_project_suggests_uv_add(self, indexed_db):
        result = search("nonexistentpkg", db_path_fn=indexed_db)
        hint = search_suggestion("nonexistentpkg", result)
        assert hint is not None
        assert "uv add" in hint
        # Should not guess the package name from the query
        assert "nonexistentpkg" not in hint
