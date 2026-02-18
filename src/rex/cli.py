"""Rex CLI — thin wrapper over core API."""

from __future__ import annotations

import json
import re
import tomllib
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
ITALIC = "\033[3m"
BLUE_ITALIC = "\033[3;34m"
RESET = "\033[0m"

# Matches `: type_annotation` and `= default_value` inside param lists
_ANNOTATION = re.compile(r"(:\s*)([^,)=]+)")
_DEFAULT = re.compile(r"(=\s*)([^,)]+)")


def _style_sig(sig: str) -> str:
    """Colorize a signature: parens cyan, types dim, -> cyan, ... cyan."""
    if not sig:
        return sig

    # Pop trailing truncation marker
    truncated = sig.endswith("...")
    if truncated:
        sig = sig[:-3]

    parts = sig.split(" -> ", 1)
    params = parts[0]

    # Dim type annotations and default values
    params = _ANNOTATION.sub(
        lambda m: DIM + m.group(1) + m.group(2) + RESET, params,
    )
    params = _DEFAULT.sub(
        lambda m: DIM + m.group(1) + m.group(2) + RESET, params,
    )

    # Cyan braces
    params = params.replace("(", f"{CYAN}({RESET}")
    params = params.replace(")", f"{CYAN}){RESET}")

    result = params
    if len(parts) > 1:
        result += f" {CYAN}->{RESET} {DIM}{parts[1]}{RESET}"

    if truncated:
        result += f"{CYAN}...{RESET}"

    return result


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
        typer.echo(f"{BLUE_ITALIC}{hint}{RESET}", err=True)

    if not result.symbols:
        raise typer.Exit(1)

    # Single exact match among fuzzy noise → auto-show detail
    hit = result.unique_exact
    if hit:
        typer.echo(format_symbol_detail(hit))
        return

    for i, sym in enumerate(result.symbols, 1):
        sig = sym.signature or ""
        if len(sig) > 50:
            sig = sig[:47] + "..."
        styled = _style_sig(sig)
        typer.echo(f"{ITALIC}{i:2}.{RESET} {DIM}{sym.symbol_type:8}{RESET} {BOLD}{sym.name}{RESET}{styled}")
        typer.echo(f"    {DIM}{sym.file_path}:{sym.line_no}{RESET}")


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


_REX_MCP_KEY = "rex"
_REX_MCP_CONFIG = {
    "command": "uv",
    "args": ["run", "rex-mcp", "serve", "${PWD}"],
    "autoApprove": ["rex_find", "rex_show", "rex_members"],
}
_REX_CODEX_CONFIG = {
    "command": "uv",
    "args": ["run", "rex-mcp", "serve", "${PWD}"],
    "enabled_tools": ["rex_find", "rex_show", "rex_members"],
}


def _toml_value(v: object) -> str:
    """Format a Python value as a TOML literal."""
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(i) for i in v) + "]"
    msg = f"unsupported TOML type: {type(v)}"
    raise TypeError(msg)


def _dump_toml(data: dict, _prefix: str = "") -> str:
    """Minimal TOML serializer for simple nested configs."""
    lines: list[str] = []
    nested: list[tuple[str, dict]] = []

    for key, value in data.items():
        if isinstance(value, dict):
            nested.append((f"{_prefix}{key}", value))
        else:
            lines.append(f"{key} = {_toml_value(value)}")

    for path, section in nested:
        has_subtables = any(isinstance(v, dict) for v in section.values())
        if has_subtables:
            sub = _dump_toml(section, f"{path}.")
            if lines:
                lines.append("")
            lines.append(sub.rstrip())
        else:
            if lines:
                lines.append("")
            lines.append(f"[{path}]")
            for k, v in section.items():
                lines.append(f"{k} = {_toml_value(v)}")

    return "\n".join(lines) + "\n"


def _init_claude_mcp(path: Path) -> None:
    """Register Rex in .mcp.json (Claude Code)."""
    mcp_path = path.resolve() / ".mcp.json"

    if mcp_path.exists():
        data = json.loads(mcp_path.read_text())
        servers = data.get("mcpServers", {})

        if _REX_MCP_KEY in servers:
            if servers[_REX_MCP_KEY] == _REX_MCP_CONFIG:
                typer.echo("Rex already configured in .mcp.json")
                return
            servers[_REX_MCP_KEY] = _REX_MCP_CONFIG
            typer.echo("Updated Rex config in .mcp.json")
        else:
            servers[_REX_MCP_KEY] = _REX_MCP_CONFIG
            typer.echo("Added Rex to existing .mcp.json")

        data["mcpServers"] = servers
    else:
        data = {"mcpServers": {_REX_MCP_KEY: _REX_MCP_CONFIG}}
        typer.echo("Created .mcp.json with Rex MCP")

    mcp_path.write_text(json.dumps(data, indent=2) + "\n")


def _init_codex_config(path: Path) -> None:
    """Register Rex in .codex/config.toml (Codex CLI)."""
    codex_dir = path.resolve() / ".codex"
    config_path = codex_dir / "config.toml"

    if config_path.exists():
        data = tomllib.loads(config_path.read_text())
        servers = data.get("mcp_servers", {})

        if _REX_MCP_KEY in servers:
            if servers[_REX_MCP_KEY] == _REX_CODEX_CONFIG:
                typer.echo("Rex already configured in .codex/config.toml")
                return
            servers[_REX_MCP_KEY] = _REX_CODEX_CONFIG
            typer.echo("Updated Rex config in .codex/config.toml")
        else:
            servers[_REX_MCP_KEY] = _REX_CODEX_CONFIG
            typer.echo("Added Rex to existing .codex/config.toml")

        data["mcp_servers"] = servers
    else:
        codex_dir.mkdir(parents=True, exist_ok=True)
        data = {"mcp_servers": {_REX_MCP_KEY: _REX_CODEX_CONFIG}}
        typer.echo("Created .codex/config.toml with Rex MCP")

    config_path.write_text(_dump_toml(data))


@app.command("init-mcp")
def init_mcp(
    path: Path = typer.Option(".", "-p", "--path", help="Project root"),
) -> None:
    """Register Rex MCP server for Claude Code and Codex CLI."""
    _init_claude_mcp(path)
    _init_codex_config(path)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
