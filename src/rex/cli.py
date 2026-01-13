"""Fast Python documentation browser CLI."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from .indexer import find_venv
from .storage import build_index, get_members, get_stats, get_symbol, search
from .tui import run_tui

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


# =============================================================================
# Display Helpers
# =============================================================================


def format_signature(symbol) -> Text:
    """Format symbol signature with syntax highlighting."""
    if symbol.symbol_type == "class":
        prefix = "class "
        sig = symbol.signature or ""
    elif symbol.symbol_type.endswith("method"):
        prefix = "def " if "async" not in symbol.symbol_type else "async def "
        sig = symbol.signature or "()"
    elif symbol.symbol_type.endswith("function"):
        prefix = "def " if "async" not in symbol.symbol_type else "async def "
        sig = symbol.signature or "()"
    else:
        prefix = ""
        sig = ""

    code = f"{prefix}{symbol.name}{sig}"
    return Syntax(code, "python", theme="monokai", word_wrap=True)


def format_symbol(symbol, show_path: bool = True) -> Panel:
    """Format a symbol for display."""
    content = Text()

    # Qualified name
    content.append(symbol.qualified_name, style="bold cyan")
    content.append("\n\n")

    # Signature
    if symbol.signature or symbol.symbol_type in ("class", "method", "function"):
        sig_text = format_signature(symbol)
        content.append_text(Text.from_markup("[dim]Signature:[/dim]\n"))
        console.print(sig_text)

    # Docstring
    if symbol.docstring:
        content.append("\n")
        # Truncate long docstrings
        doc = symbol.docstring
        if len(doc) > 2000:
            doc = doc[:2000] + "\n... (truncated)"
        content.append(doc, style="white")

    # File location
    if show_path:
        content.append("\n\n")
        content.append(f"{symbol.file_path}:{symbol.line_no}", style="dim")

    title = f"[{symbol.symbol_type}]"
    return Panel(content, title=title, title_align="left", border_style="blue")


# =============================================================================
# Commands
# =============================================================================


@app.command()
def index(
    force: bool = typer.Option(False, "-f", "--force", help="Force reindex"),
) -> None:
    """Build or rebuild the symbol index for .venv."""
    venv = find_venv()
    if venv is None:
        typer.echo("Error: No .venv found in current directory or parents", err=True)
        raise typer.Exit(1)

    typer.echo(f"Indexing: {venv}")

    def progress(pkg: str, current: int, total: int) -> None:
        typer.echo(f"  [{current}/{total}] {pkg}", nl=False)
        typer.echo("\r", nl=False)

    count = build_index(venv, force=force, progress_callback=progress)

    if count == -1:
        typer.echo("Index is up to date (use -f to force rebuild)")
    else:
        typer.echo(f"  ✓ Indexed {count:,} symbols")


@app.command()
def find(
    query: str = typer.Argument(..., help="Symbol to search for"),
    limit: int = typer.Option(20, "-n", "--limit", help="Max results"),
    type_filter: str = typer.Option(None, "-t", "--type", help="Filter by type"),
) -> None:
    """Search for symbols matching query."""
    results = search(query, limit=limit, symbol_type=type_filter)

    if not results:
        typer.echo(f"No symbols found matching: {query}")
        raise typer.Exit(1)

    for sym in results:
        sig = sym.signature or ""
        if len(sig) > 50:
            sig = sig[:47] + "..."
        typer.echo(f"{sym.symbol_type:8} {sym.qualified_name}{sig}")


@app.command()
def show(
    name: str = typer.Argument(..., help="Qualified symbol name"),
) -> None:
    """Show detailed info for a symbol."""
    symbol = get_symbol(name)

    if symbol is None:
        # Try fuzzy search
        results = search(name, limit=5)
        if results:
            typer.echo(f"Symbol not found: {name}")
            typer.echo("Did you mean:")
            for r in results:
                typer.echo(f"  {r.qualified_name}")
        else:
            typer.echo(f"Symbol not found: {name}", err=True)
        raise typer.Exit(1)

    console.print(format_symbol(symbol))


@app.command()
def members(
    name: str = typer.Argument(..., help="Class or module name"),
) -> None:
    """List members of a class or module."""
    results = get_members(name)

    if not results:
        typer.echo(f"No members found for: {name}")
        raise typer.Exit(1)

    for sym in results:
        sig = sym.signature or ""
        if len(sig) > 40:
            sig = sig[:37] + "..."
        typer.echo(f"{sym.symbol_type:8} {sym.name}{sig}")


@app.command()
def stats() -> None:
    """Show index statistics."""
    info = get_stats()

    if "error" in info:
        typer.echo(f"Error: {info['error']}", err=True)
        if "venv" in info:
            typer.echo(f"  venv: {info['venv']}")
            typer.echo("  Run 'pydocs index' to build the index")
        raise typer.Exit(1)

    typer.echo(f"venv: {info['venv']}")
    typer.echo(f"Symbols: {info['total_symbols']:,}")
    typer.echo(f"Packages: {info['packages']}")
    typer.echo("By type:")
    for stype, count in sorted(info["by_type"].items()):
        typer.echo(f"  {stype}: {count:,}")


@app.command()
def edit(
    name: str = typer.Argument(..., help="Symbol name to open"),
) -> None:
    """Open symbol in $EDITOR at the correct line."""
    import os
    import subprocess

    symbol = get_symbol(name)
    if symbol is None:
        results = search(name, limit=1)
        if results:
            symbol = results[0]
        else:
            typer.echo(f"Symbol not found: {name}", err=True)
            raise typer.Exit(1)

    editor = os.environ.get("EDITOR", "vim")
    file_path = symbol.file_path
    line_no = symbol.line_no

    # Most editors support +line syntax
    if editor in ("vim", "nvim", "vi"):
        cmd = [editor, f"+{line_no}", file_path]
    elif editor in ("code", "code-insiders"):
        cmd = [editor, "--goto", f"{file_path}:{line_no}"]
    elif editor == "subl":
        cmd = [editor, f"{file_path}:{line_no}"]
    else:
        cmd = [editor, file_path]

    subprocess.run(cmd)


@app.command()
def browse(
    query: str = typer.Argument("", help="Initial search query"),
) -> None:
    """Open interactive documentation browser (TUI)."""
    run_tui(query)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
