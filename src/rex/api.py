"""Shared API layer — used by both CLI and MCP server."""

from __future__ import annotations

from pathlib import Path

from .indexer import Symbol
from .storage import get_symbol, search


def format_symbol_line(sym: Symbol, sig_max: int = 50) -> str:
    """One-line summary: type, name, truncated signature."""
    sig = sym.signature or ""
    if len(sig) > sig_max:
        sig = sig[: sig_max - 3] + "..."
    return f"{sym.symbol_type:8} {sym.name}{sig}"


def format_symbol_detail(sym: Symbol) -> str:
    """Multi-line plain text detail view."""
    lines = [f"{sym.symbol_type}: {sym.qualified_name}"]
    if sym.signature:
        lines.append(f"signature: {sym.signature}")
    if sym.docstring:
        doc = sym.docstring if len(sym.docstring) <= 1500 else sym.docstring[:1500] + "..."
        lines.append(f"docstring: {doc}")
    lines.append(f"location: {sym.file_path}:{sym.line_no}")
    return "\n".join(lines)


def show_symbol(name: str, venv: Path | None = None) -> Symbol | list[str]:
    """Look up a symbol by exact qualified name.

    Returns the Symbol if found, or a list of suggestion qualified names.
    Empty list means nothing found at all.
    """
    sym = get_symbol(name, venv=venv)
    if sym is not None:
        return sym

    results = search(name, venv=venv, limit=5)
    return [r.qualified_name for r in results]
