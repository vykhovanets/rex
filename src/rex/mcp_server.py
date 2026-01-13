"""MCP server exposing rex functionality to Claude Code.

Usage:
    uv run rex-mcp install   # Register with Claude Code
    uv run rex-mcp uninstall # Remove registration
    uv run rex-mcp serve     # Run server (called by Claude Code)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .storage import build_index, get_members, get_stats, get_symbol, search

# =============================================================================
# MCP Server
# =============================================================================

server = Server("rex")


def _short_name(qualified_name: str) -> str:
    """Extract short name: 'pkg.mod.Class.method' -> 'Class.method'."""
    parts = qualified_name.split(".")
    if len(parts) <= 2:
        return qualified_name
    # Keep last 2 parts (Class.method or module.function)
    return ".".join(parts[-2:])


def _first_line(docstring: str | None) -> str:
    """Extract first meaningful line of docstring."""
    if not docstring:
        return ""
    line = docstring.strip().split("\n")[0].strip()
    if len(line) > 80:
        line = line[:77] + "..."
    return line


def _format_symbol(sym) -> str:
    """Format symbol for LLM consumption - concise but complete."""
    lines = [f"{sym.symbol_type}: {sym.qualified_name}"]
    if sym.signature:
        lines.append(f"signature: {sym.signature}")
    if sym.docstring:
        # Truncate very long docstrings for context efficiency
        doc = sym.docstring if len(sym.docstring) <= 1500 else sym.docstring[:1500] + "..."
        lines.append(f"docstring: {doc}")
    lines.append(f"location: {sym.file_path}:{sym.line_no}")
    return "\n".join(lines)


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Expose rex tools to Claude Code."""
    return [
        Tool(
            name="rex_search",
            description="Search Python symbols in .venv packages. Returns matching classes, functions, methods with signatures.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'from_pretrained', 'Sam3Processor')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 15)",
                        "default": 15,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="rex_show",
            description="Get full documentation for a symbol by qualified name. Returns signature, docstring, file location.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Qualified name (e.g., 'transformers.AutoModel.from_pretrained')",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="rex_members",
            description="List members of a class or module. Use after rex_show to explore structure.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Qualified name of class/module",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="rex_index",
            description="Build symbol index for .venv packages. Run if search returns empty or after installing new packages.",
            inputSchema={
                "type": "object",
                "properties": {
                    "force": {
                        "type": "boolean",
                        "description": "Force rebuild even if up-to-date",
                        "default": False,
                    },
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name == "rex_search":
        query = arguments["query"]
        limit = arguments.get("limit", 15)
        results = search(query, limit=limit)

        if not results:
            stats = get_stats()
            if "error" in stats:
                return [TextContent(type="text", text="No index found. Run rex_index first.")]
            return [TextContent(type="text", text=f"No symbols found for: {query}")]

        output = [f"Found {len(results)} results for '{query}':\n"]
        for i, sym in enumerate(results, 1):
            short = _short_name(sym.qualified_name)
            summary = _first_line(sym.docstring)
            sig = sym.signature or ""
            if len(sig) > 50:
                sig = sig[:47] + "..."
            # Format: numbered, short name + sig, then summary on next line
            output.append(f"{i}. {short}{sig}  [{sym.symbol_type}]")
            if summary:
                output.append(f"   {summary}")
            output.append(f"   → {sym.qualified_name}")

        return [TextContent(type="text", text="\n".join(output))]

    elif name == "rex_show":
        qname = arguments["name"]
        symbol = get_symbol(qname)

        if symbol is None:
            # Try fuzzy search
            results = search(qname, limit=3)
            if results:
                suggestions = "\n".join(f"  {r.qualified_name}" for r in results)
                return [TextContent(type="text", text=f"Symbol not found: {qname}\nDid you mean:\n{suggestions}")]
            return [TextContent(type="text", text=f"Symbol not found: {qname}")]

        return [TextContent(type="text", text=_format_symbol(symbol))]

    elif name == "rex_members":
        qname = arguments["name"]
        results = get_members(qname)

        if not results:
            return [TextContent(type="text", text=f"No members found for: {qname}")]

        output = [f"Members of {qname}:\n"]
        for sym in results:
            sig = sym.signature or ""
            if len(sig) > 50:
                sig = sig[:47] + "..."
            output.append(f"  {sym.symbol_type:8} {sym.name}{sig}")

        return [TextContent(type="text", text="\n".join(output))]

    elif name == "rex_index":
        force = arguments.get("force", False)
        try:
            count = build_index(force=force)
            if count == -1:
                return [TextContent(type="text", text="Index is up-to-date.")]
            return [TextContent(type="text", text=f"Indexed {count:,} symbols.")]
        except RuntimeError as e:
            return [TextContent(type="text", text=f"Error: {e}")]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def run_server():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


# =============================================================================
# CLI for install/uninstall
# =============================================================================


def get_settings_path() -> Path:
    """Get project-local Claude settings path."""
    return Path.cwd() / ".claude" / "settings.json"


def load_settings() -> dict:
    """Load existing settings or return empty dict."""
    path = get_settings_path()
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_settings(settings: dict) -> None:
    """Save settings to file."""
    path = get_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n")


def install() -> None:
    """Register rex MCP server with Claude Code."""
    settings = load_settings()

    if "mcpServers" not in settings:
        settings["mcpServers"] = {}

    settings["mcpServers"]["rex"] = {
        "command": "uv",
        "args": ["run", "rex-mcp", "serve"],
    }

    save_settings(settings)
    print(f"✓ Registered rex MCP server in {get_settings_path()}")
    print("  Restart Claude Code to activate.")


def uninstall() -> None:
    """Remove rex MCP server registration."""
    settings = load_settings()

    if "mcpServers" in settings and "rex" in settings["mcpServers"]:
        del settings["mcpServers"]["rex"]
        if not settings["mcpServers"]:
            del settings["mcpServers"]
        save_settings(settings)
        print("✓ Removed rex MCP server registration.")
    else:
        print("rex MCP server not registered.")


def main() -> None:
    """CLI entry point."""
    import asyncio

    if len(sys.argv) < 2:
        print("Usage: rex-mcp <command>")
        print("Commands:")
        print("  install   - Register with Claude Code")
        print("  uninstall - Remove registration")
        print("  serve     - Run MCP server")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "install":
        install()
    elif cmd == "uninstall":
        uninstall()
    elif cmd == "serve":
        asyncio.run(run_server())
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
