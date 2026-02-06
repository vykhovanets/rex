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

## CLI

```bash
rex find <query>       # Search symbols (unquoted multi-word OK)
rex show <name>        # Full documentation by qualified name
rex members <name>     # List class/module members
rex stats              # Index statistics
rex index              # Rebuild index
rex index -p .         # Rebuild including project source
```

## MCP Tools

Claude Code can use rex too — same speed, no
round-trips to the web.

| Tool | Description |
|--------------|--------------------------------------|
| `rex_search` | Find symbols by name |
| `rex_show` | Get full docs for a symbol |
| `rex_members`| List class/module members |
| `rex_index` | Rebuild index (supports project_dirs)|

![example](./example.jpg)

## License

MIT
