# Rex

Rex provides sub-second Python symbol search for .venv packages.

## Tools

- `rex_find` — search symbols by keyword. Supports `limit` and `symbol_type` (class/function/method/module) filters.
- `rex_show` — get full docs for a symbol by qualified name.
- `rex_members` — list class/module members.

## When to use

Use Rex instead of web search or .venv grep when you need:
- Function signatures and parameter types
- Class members and inheritance
- Docstrings from installed packages
- Quick API exploration before writing code

## Workflow

1. `rex_find` to search → 2. `rex_show` for details → 3. `rex_members` to explore

Rex auto-indexes on first use. No manual setup needed.
