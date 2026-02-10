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

from .api import format_symbol_detail, format_symbol_line, show_symbol
from .storage import build_index, clean_index, get_members, get_stats, search

# =============================================================================
# MCP Server
# =============================================================================

server = Server("rex")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Expose rex tools to Claude Code."""
    return [
        Tool(
            name="rex_find",
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
                    "symbol_type": {
                        "type": "string",
                        "description": "Filter by symbol type (class, function, method, module)",
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
            description="Build symbol index for .venv packages and optionally project source code. Run if search returns empty or after installing new packages. Pass project_dirs to also index project source files for better search coverage.",
            inputSchema={
                "type": "object",
                "properties": {
                    "force": {
                        "type": "boolean",
                        "description": "Force rebuild even if up-to-date",
                        "default": False,
                    },
                    "project_dirs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Directories of project source code to index alongside .venv (e.g., ['./src', '.'])",
                    },
                },
            },
        ),
        Tool(
            name="rex_stats",
            description="Show index statistics: total symbols, packages, types.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="rex_clean",
            description="Remove stale packages whose source files no longer exist on disk.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name == "rex_find":
        query = arguments["query"]
        limit = arguments.get("limit", 15)
        symbol_type = arguments.get("symbol_type")
        results = search(query, limit=limit, symbol_type=symbol_type)

        if not results:
            stats = get_stats()
            if "error" in stats:
                return [TextContent(type="text", text="No index found. Run rex_index first.")]
            return [TextContent(type="text", text=f"No symbols found for: {query}")]

        output = [f"Found {len(results)} results for '{query}':\n"]
        for sym in results:
            output.append(format_symbol_line(sym))
            output.append(f"   \u2192 {sym.qualified_name}")

        return [TextContent(type="text", text="\n".join(output))]

    elif name == "rex_show":
        qname = arguments["name"]
        result = show_symbol(qname)

        if isinstance(result, list):
            if result:
                suggestions = "\n".join(f"  {qn}" for qn in result)
                return [TextContent(type="text", text=f"Symbol not found: {qname}\nDid you mean:\n{suggestions}")]
            return [TextContent(type="text", text=f"Symbol not found: {qname}")]

        return [TextContent(type="text", text=format_symbol_detail(result))]

    elif name == "rex_members":
        qname = arguments["name"]
        results = get_members(qname)

        if not results:
            return [TextContent(type="text", text=f"No members found for: {qname}")]

        output = [f"Members of {qname}:\n"]
        for sym in results:
            output.append(f"  {format_symbol_line(sym)}")

        return [TextContent(type="text", text="\n".join(output))]

    elif name == "rex_index":
        force = arguments.get("force", False)
        raw_dirs = arguments.get("project_dirs")
        project_dirs = [Path(d).resolve() for d in raw_dirs] if raw_dirs else None
        try:
            count = build_index(force=force, project_dirs=project_dirs)
            if count == -1:
                return [TextContent(type="text", text="Index is up-to-date.")]
            msg = f"Indexed {count:,} symbols."
            if project_dirs:
                msg += f" (including {len(project_dirs)} project dir(s))"
            return [TextContent(type="text", text=msg)]
        except RuntimeError as e:
            return [TextContent(type="text", text=f"Error: {e}")]

    elif name == "rex_stats":
        info = get_stats()
        if "error" in info:
            return [TextContent(type="text", text=info["error"])]
        lines = [
            f"DB: {info['db_path']}",
            f"Symbols: {info['total_symbols']:,}",
            f"Packages: {info['packages']}",
            "By type:",
        ]
        for stype, count in sorted(info["by_type"].items()):
            lines.append(f"  {stype}: {count:,}")
        return [TextContent(type="text", text="\n".join(lines))]

    elif name == "rex_clean":
        removed = clean_index()
        if removed:
            lines = [f"Removed {len(removed)} stale packages:"]
            for pkg_name in removed:
                lines.append(f"  {pkg_name}")
            return [TextContent(type="text", text="\n".join(lines))]
        return [TextContent(type="text", text="No stale packages found.")]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def run_server():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


# =============================================================================
# CLI for install/uninstall
# =============================================================================


def get_mcp_json_path() -> Path:
    """Get project's .mcp.json path."""
    return Path.cwd() / ".mcp.json"


def load_mcp_json() -> dict:
    """Load existing .mcp.json or return empty dict."""
    path = get_mcp_json_path()
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_mcp_json(config: dict) -> None:
    """Save .mcp.json file."""
    path = get_mcp_json_path()
    path.write_text(json.dumps(config, indent=2) + "\n")


def install() -> None:
    """Register rex MCP server in project's .mcp.json."""
    config = load_mcp_json()

    if "mcpServers" not in config:
        config["mcpServers"] = {}

    config["mcpServers"]["rex"] = {
        "command": "rex-mcp",
        "args": ["serve"],
    }

    save_mcp_json(config)
    print(f"\u2713 Registered rex MCP server in {get_mcp_json_path()}")
    print("  Restart Claude Code to activate.")
    print("  Note: Project MCP servers require approval on first use.")


def uninstall() -> None:
    """Remove rex MCP server registration."""
    config = load_mcp_json()

    if "mcpServers" in config and "rex" in config["mcpServers"]:
        del config["mcpServers"]["rex"]
        if not config["mcpServers"]:
            del config["mcpServers"]
        save_mcp_json(config)
        print("\u2713 Removed rex MCP server registration.")
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
