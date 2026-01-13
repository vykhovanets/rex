# rex 🇺🇦

> reference explorer

Sub-second Python API browser. Faster than asking an LLM.  
Analyses `.venv` packages, that already in the project.  
*Bonus*: MCP for [Claude Code](https://github.com/anthropics/claude-code).  

## Install

```bash
uv add git+https://github.com/vykhovanets/rex.git
uv run rex index        # Build symbol index (~30s once)
uv run rex-mcp install  # Register with Claude Code
```

## CLI

```bash
rex find <query>    # Search symbols
rex show <name>     # Full documentation
rex members <name>  # List class/module members
rex edit <name>     # Open in $EDITOR at line
rex browse          # Interactive TUI browser
rex stats           # Index statistics
```

## MCP Tools

Claude Code can use rex too - same speed, no round-trips to the web.

| Tool | Description |
|------|-------------|
| `rex_search` | Find symbols by name |
| `rex_show` | Get full docs for a symbol |
| `rex_members` | List class/module members |
| `rex_index` | Rebuild index |

## License

MIT
