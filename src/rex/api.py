"""Shared API layer — used by both CLI and MCP server."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .indexer import Symbol, find_venv
from .storage import SearchResult, get_db_path, get_members, get_symbol, search


def format_symbol_line(sym: Symbol, sig_max: int = 50) -> str:
    """One-line summary: type, name, truncated signature."""
    sig = sym.signature or ""
    if len(sig) > sig_max:
        sig = sig[: sig_max - 3] + "..."
    return f"{sym.symbol_type:8} {sym.name}{sig}"


def format_symbol_detail(
    sym: Symbol,
    db_path_fn: Callable[[], Path] = get_db_path,
) -> str:
    """Multi-line plain text detail view.

    For classes/modules, appends a brief member listing.
    """
    lines = [f"{sym.symbol_type}: {sym.qualified_name}"]
    if sym.signature:
        lines.append(f"signature: {sym.signature}")
    if sym.docstring:
        doc = sym.docstring if len(sym.docstring) <= 1500 else sym.docstring[:1500] + "..."
        lines.append(f"docstring: {doc}")
    lines.append(f"location: {sym.file_path}:{sym.line_no}")

    if sym.symbol_type in ("class", "module"):
        members = get_members(sym.qualified_name, db_path_fn=db_path_fn)
        if members:
            lines.append("members:")
            for m in members:
                sig = m.signature or ""
                if len(sig) > 50:
                    sig = sig[:47] + "..."
                lines.append(f"  {m.symbol_type:8} {m.name}{sig}")

    return "\n".join(lines)


def search_suggestion(
    query: str,
    result: SearchResult,
    project_path_fn: Callable[[], Path] = Path.cwd,
) -> str | None:
    """Context-aware hint for search results. None if no hint needed."""
    if result.symbols and not result.fuzzy_only:
        return None  # exact match

    in_project = find_venv(start_dir=project_path_fn()) is not None

    if result.fuzzy_only:
        if in_project:
            return (
                "Showing approximate matches (no exact results). "
                "Package may not be installed — try: uv add <package>"
            )
        return (
            "Showing approximate matches (no exact results). "
            "Not in a Python project — no sources to reindex."
        )

    # No results at all
    if not in_project:
        return "Not in a Python project (no .venv found)."

    return "No results. Package may not be installed — try: uv add <package>"


def show_symbol(
    name: str,
    db_path_fn: Callable[[], Path] = get_db_path,
    project_path_fn: Callable[[], Path] = Path.cwd,
) -> Symbol | list[str]:
    """Look up a symbol by exact qualified name.

    Returns the Symbol if found, or a list of suggestion qualified names.
    Empty list means nothing found at all.
    """
    sym = get_symbol(name, db_path_fn=db_path_fn, project_path_fn=project_path_fn)
    if sym is not None:
        return sym

    result = search(name, db_path_fn=db_path_fn, project_path_fn=project_path_fn, limit=5)
    return [r.qualified_name for r in result.symbols]
