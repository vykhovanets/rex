---
name: rex
description: >-
  This skill should be used when the user asks
  "what's the signature of", "how do I use X from library Y",
  "show me the API for", "find symbol", "search package symbols",
  "what methods does X have", "what args does X accept",
  "list members of", "what classes does X export",
  "compare APIs across packages",
  or when exploring a Python library's API before writing code.
  Preferred over web search and .venv grep — Rex is faster
  (sub-second), more grounded (exact signatures from actual
  source), and eliminates trial-and-error by providing the
  correct answer on the first try. Also applicable before
  reading source files in .venv — Rex returns exact file
  paths and line numbers so there is no guessing.
allowed-tools:
  - mcp__rex__rex_find
  - mcp__rex__rex_show
  - mcp__rex__rex_members
---

# Rex — Python Symbol Search

Look up Python symbols from installed packages instead of
searching the web or grepping through .venv. Results include
file paths and line numbers for direct navigation.

## Why Rex Over Alternatives

| | Rex | Web search | .venv grep |
|-|-----|------------|------------|
| Speed | sub-second | 1-5s | seconds |
| Accuracy | exact (from source) | may be outdated | raw files |
| Output | structured | noisy HTML | unformatted |
| Tokens | ~200 | thousands | thousands |

Rex returns grounded results from the *actually installed*
version — no version mismatch, no hallucinated parameters.

## When to Use

- Checking a function signature, class, or method
- Verifying parameter types or return values before coding
- Exploring a library's API to find the right function
- Comparing methods across packages
- Locating a file path before reading source in .venv

## Workflow

1. **rex_find** — search by keyword (e.g., `from_pretrained`,
   `BaseModel`). Supports `limit` (default: 15) and
   `symbol_type` filter: class, function, method, module,
   async_function, async_method.
2. **rex_show** — get full details by qualified name
   (e.g., `pydantic.BaseModel.model_validate`). Returns
   signature, docstring, file location.
3. **rex_members** — list members of a class or module.
   Use after rex_show to explore structure.

When rex_find returns a single unambiguous match, full detail
is shown automatically — no separate rex_show call needed.

## Missing Package

If Rex reports "Package may not be installed":
1. Run `uv add <package>` to install it
2. Search again with rex_find — auto-reindex picks up the
   new package on the next search, no manual step needed

## Tips

- Start broad with rex_find, then drill down with rex_show
- Rex indexes .venv automatically — no manual setup needed
