"""Tests for rex.api — the shared formatting/lookup layer.

These define the expected interface that both CLI and MCP should use,
eliminating duplicated formatting logic.

Since rex.api doesn't exist yet, all tests are marked xfail.
"""

from __future__ import annotations

import pytest

from rex.indexer import Symbol

# We expect rex.api to provide these three functions.
# Since the module doesn't exist yet, gate the import behind xfail.
try:
    from rex.api import format_symbol_detail, format_symbol_line, show_symbol

    _API_AVAILABLE = True
except ImportError:
    _API_AVAILABLE = False
    format_symbol_line = None  # type: ignore[assignment]
    format_symbol_detail = None  # type: ignore[assignment]
    show_symbol = None  # type: ignore[assignment]

pytestmark = pytest.mark.xfail(
    not _API_AVAILABLE,
    reason="rex.api not implemented yet",
    raises=(TypeError, AttributeError, NameError),
)


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
    """show_symbol(name, venv) -> Symbol | list[str]

    Exact lookup by qualified name. If not found, falls back to fuzzy
    search and returns a list of suggestion strings ("did you mean").
    """

    def test_exact_match_returns_symbol(self, ensure_index):
        # Find a real qualified name via storage layer
        from rex.storage import search as raw_search

        hits = raw_search("Symbol", venv=ensure_index, limit=1)
        if not hits:
            pytest.skip("No symbols indexed — cannot test exact match")
        qn = hits[0].qualified_name

        result = show_symbol(qn, venv=ensure_index)
        assert isinstance(result, Symbol)
        assert result.qualified_name == qn

    def test_nonexistent_returns_suggestions(self, ensure_index):
        # Not an exact qualified name, but a valid prefix that FTS can match
        result = show_symbol("BaseMod", venv=ensure_index)
        assert isinstance(result, list)
        assert len(result) > 0
        for s in result:
            assert isinstance(s, str)

    def test_completely_unknown_returns_empty_list(self, ensure_index):
        result = show_symbol("zzz_nonexistent_xyz.does.not.exist", venv=ensure_index)
        assert isinstance(result, list)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Fixtures (reuse from test_core)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def venv():
    from rex.indexer import find_venv

    v = find_venv(Path("/Users/avykhova/Code/karb/rex"))
    assert v is not None, ".venv not found"
    return v


@pytest.fixture(scope="session")
def ensure_index(venv):
    from rex.storage import build_index

    build_index(venv)
    return venv


from pathlib import Path
