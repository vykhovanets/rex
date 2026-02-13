"""MCP server exposing rex functionality to Claude Code.

Usage:
    uv tool install rex-index
    claude mcp add rex -s user -- rex-mcp serve
"""

from __future__ import annotations

import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .api import format_symbol_detail, format_symbol_line, search_suggestion, show_symbol
from .storage import get_members, search

# =============================================================================
# MCP Server
# =============================================================================

server = Server(
    "rex",
    instructions=(
        "ALWAYS use Rex before writing code that calls library APIs. "
        "The correct signature is already on the system in .venv — "
        "don't guess from training data, don't waste tokens on web search. "
        "Rex returns exact signatures from the actually-installed source: "
        "no hallucinated parameters, no version mismatch, correct on the first try. "
        "~200 tokens, sub-second. "
        "rex_find to search → rex_show for full detail → rex_members to list class/module members."
    ),
)


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
                        "description": "Filter by symbol type (class, function, method, module, async_function, async_method)",
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
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name == "rex_find":
        query = arguments["query"]
        limit = arguments.get("limit", 15)
        symbol_type = arguments.get("symbol_type")
        result = search(query, limit=limit, symbol_type=symbol_type)

        if not result.symbols:
            hint = search_suggestion(query, result)
            msg = hint or f"No symbols found for: {query}"
            return [TextContent(type="text", text=msg)]

        # Single exact match among fuzzy noise → auto-show detail
        hit = result.unique_exact
        if hit:
            return [TextContent(type="text", text=format_symbol_detail(hit))]

        output = []
        hint = search_suggestion(query, result)
        if hint:
            output.append(f"Note: {hint}\n")
        output.append(f"Found {len(result.symbols)} results for '{query}':\n")
        for sym in result.symbols:
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

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def run_server():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    """CLI entry point — only 'serve' is needed."""
    import asyncio

    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        asyncio.run(run_server())
    else:
        print("Usage: rex-mcp serve")
        print()
        print("Register with Claude Code:")
        print("  claude mcp add rex -s user -- rex-mcp serve")
        sys.exit(1)


if __name__ == "__main__":
    main()
