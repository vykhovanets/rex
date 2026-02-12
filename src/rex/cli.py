"""Rex CLI — thin wrapper over core API."""

from __future__ import annotations

from pathlib import Path

import click
import typer

from .api import format_symbol_detail, format_symbol_line, search_suggestion, show_symbol
from .storage import build_index, clean_index, get_members, get_stats, search


class _DefaultFind(typer.core.TyperGroup):
    """Treat unrecognized commands as `find` queries."""

    def resolve_command(self, ctx: click.Context, args: list[str]):  # type: ignore[override]
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            return super().resolve_command(ctx, ["find"] + args)


app = typer.Typer(
    cls=_DefaultFind,
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

# ANSI styling
DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
RESET = "\033[0m"


@app.command()
def index(
    force: bool = typer.Option(False, "-f", "--force", help="Force reindex"),
    project: list[Path] = typer.Option([], "-p", "--project", help="Project dirs to index alongside .venv"),
) -> None:
    """Build or rebuild the symbol index."""
    def progress(pkg: str, current: int, total: int) -> None:
        typer.echo(f"  [{current}/{total}] {pkg}", nl=False)
        typer.echo("\r", nl=False)

    project_dirs = [p.resolve() for p in project] if project else None
    try:
        count = build_index(force=force, progress_callback=progress, project_dirs=project_dirs)
    except RuntimeError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    if count == -1:
        typer.echo("Index is up to date (use -f to force rebuild)")
    else:
        typer.echo(f"  Indexed {count:,} symbols")


@app.command()
def find(
    query: list[str] = typer.Argument(..., help="Symbol to search for"),
    limit: int = typer.Option(20, "-n", "--limit", help="Max results"),
    type_filter: str = typer.Option(None, "-t", "--type", help="Filter by type"),
) -> None:
    """Search for symbols matching query."""
    query_str = " ".join(query)
    try:
        result = search(query_str, limit=limit, symbol_type=type_filter)
    except RuntimeError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    hint = search_suggestion(query_str, result)
    if hint:
        typer.echo(f"{DIM}{hint}{RESET}", err=True)

    if not result.symbols:
        raise typer.Exit(1)

    # Single exact match among fuzzy noise → auto-show detail
    hit = result.unique_exact
    if hit:
        typer.echo(format_symbol_detail(hit))
        return

    for sym in result.symbols:
        sig = sym.signature or ""
        if len(sig) > 50:
            sig = sig[:47] + "..."
        typer.echo(f"{DIM}{sym.symbol_type:8}{RESET} {BOLD}{sym.name}{RESET}{CYAN}{sig}{RESET}")
        typer.echo(f"{sym.file_path}:{sym.line_no}")


@app.command()
def show(
    name: str = typer.Argument(..., help="Qualified symbol name"),
) -> None:
    """Show detailed info for a symbol."""
    result = show_symbol(name)

    if isinstance(result, list):
        if result:
            typer.echo(f"Symbol not found: {name}")
            typer.echo("Did you mean:")
            for qn in result:
                typer.echo(f"  {qn}")
        else:
            typer.echo(f"Symbol not found: {name}", err=True)
        raise typer.Exit(1)

    typer.echo(format_symbol_detail(result))


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
        typer.echo(f"  {format_symbol_line(sym)}")


@app.command()
def stats() -> None:
    """Show index statistics."""
    info = get_stats()

    if "error" in info:
        typer.echo(f"Error: {info['error']}", err=True)
        raise typer.Exit(1)

    typer.echo(f"DB: {info['db_path']}")
    typer.echo(f"Symbols: {info['total_symbols']:,}")
    typer.echo(f"Packages: {info['packages']}")
    typer.echo("By type:")
    for stype, count in sorted(info["by_type"].items()):
        typer.echo(f"  {stype}: {count:,}")


@app.command()
def clean() -> None:
    """Remove packages whose source paths no longer exist."""
    removed = clean_index()
    if removed:
        for name in removed:
            typer.echo(f"  Removed: {name}")
        typer.echo(f"Cleaned {len(removed)} stale packages")
    else:
        typer.echo("No stale packages found")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
