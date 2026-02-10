# rex 🇺🇦

> reference explorer

Sub-second Python symbol search for `.venv` packages
and project source code.
*Bonus*: MCP server for
[Claude Code](https://github.com/anthropics/claude-code).

## Install

Global (available everywhere):

```bash
uv tool install git+https://github.com/vykhovanets/rex.git
rex index                       # Index .venv packages (~30s once)
rex index -p ./src              # Also index project source
```

As project dependency:

```bash
uv add git+https://github.com/vykhovanets/rex.git
uv run rex index
```

MCP server for Claude Code:

```bash
claude mcp add rex -- \
  uvx --from git+https://github.com/vykhovanets/rex.git rex-mcp serve
```

## How it works

Rex stores a single global index at
`~/.local/state/rex/rex.db`. Indexing is **incremental** —
only packages whose files changed since the last run are
re-indexed. First run takes ~30s; subsequent runs are
instant.

## CLI

```bash
rex find <query>       # Search symbols (unquoted OK)
rex find -t class Base # Filter by type
rex show <name>        # Full docs by qualified name
rex members <name>     # List class/module members
rex stats              # Index statistics
rex index              # Build/update index
rex index -f           # Force full rebuild
rex index -p ./src     # Also index project source
rex clean              # Remove stale packages
```

## MCP Tools

Claude Code can use rex too — same speed, no
round-trips to the web.

| Tool          | Description                       |
|---------------|-----------------------------------|
| `rex_find`    | Search symbols (+ type filter)    |
| `rex_show`    | Full docs for a symbol            |
| `rex_members` | List class/module members         |
| `rex_index`   | Build/update index                |
| `rex_stats`   | Index statistics                  |
| `rex_clean`   | Remove stale packages             |

![example](./example.jpg)

## License

MIT
