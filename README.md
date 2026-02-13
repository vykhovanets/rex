# rex 🇺🇦

> reference explorer

Sub-second Python symbol search for `.venv` packages
and project source code.
*Bonus*: MCP server for
[Claude Code](https://github.com/anthropics/claude-code).

## Install

### Claude Code plugin

```bash
claude plugin marketplace add vykhovanets/rex
claude plugin install rex@rex
```

MCP tools + SKILL.md with usage guidance.

### CLI + MCP

```bash
uv tool install rex-index
claude mcp add rex -s user -- rex-mcp serve
```

Rex available in every project, no per-project setup.

### Project dependency

```bash
uv add rex-index
uv run rex init-mcp
```

Adds Rex to `.mcp.json` with pre-approved tools.
Commit both files — your team gets Rex on `uv sync`.

## How it works

Rex stores a single global index at
`~/.local/state/rex/rex.db`. Indexing is **incremental** —
only packages whose files changed since the last run are
re-indexed. First run takes ~30s; subsequent runs are
instant. Project source code is indexed automatically when
a `pyproject.toml` or `setup.py` is detected.

Search has three phases:
1. **FTS5** — exact and prefix matches via SQLite full-text
2. **Fuzzy** — typo-tolerant fallback (rapidfuzz)
3. **Auto-reindex** — if nothing found and index is stale

When results are fuzzy-only or empty, rex shows a
contextual hint (e.g. "approximate matches" or
"try: uv add <package>").

## CLI

```bash
rex find <query>       # Search symbols (unquoted OK)
rex find -t class Base # Filter by type
rex show <name>        # Full docs by qualified name
rex members <name>     # List class/module members
rex stats              # Index statistics
rex index              # Build/update index
rex index -f           # Force full rebuild
rex index -p ./lib     # Also index extra directories
rex clean              # Remove stale packages
rex init-mcp           # Register MCP in .mcp.json
```

## MCP Tools

Claude Code can use rex too — same speed, no
round-trips to the web.

| Tool          | Description                   |
|---------------|-------------------------------|
| `rex_find`    | Search symbols (+ type filter)|
| `rex_show`    | Full docs for a symbol        |
| `rex_members` | List class/module members     |

## License

Apache-2.0
