"""Bump version across all Rex files.

Usage: uv run src/bump.py 0.2.0
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TARGETS = [
    ("src/rex/__init__.py", re.compile(r'(__version__\s*=\s*")([^"]+)(")')),
    ("pyproject.toml", re.compile(r'(^version\s*=\s*")([^"]+)(")', re.MULTILINE)),
    (".claude-plugin/plugin.json", re.compile(r'("version":\s*")([^"]+)(")')),
    (".claude-plugin/marketplace.json", re.compile(r'("version":\s*")([^"]+)(")')),
]


def bump(new_version: str) -> None:
    for relpath, pattern in TARGETS:
        path = ROOT / relpath
        text = path.read_text()
        match = pattern.search(text)
        if not match:
            print(f"  SKIP {relpath} (pattern not found)")
            continue
        old = match.group(2)
        if old == new_version:
            print(f"  OK   {relpath} (already {new_version})")
            continue
        text = pattern.sub(rf"\g<1>{new_version}\3", text, count=1)
        path.write_text(text)
        print(f"  {old} → {new_version}  {relpath}")


def _parse_version(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def _current_version() -> str:
    path = ROOT / "src/rex/__init__.py"
    match = re.search(r'__version__\s*=\s*"([^"]+)"', path.read_text())
    return match.group(1) if match else "0.0.0"


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: uv run src/bump.py <version>")
        sys.exit(1)

    version = sys.argv[1].lstrip("v")
    if not re.match(r"^\d+\.\d+\.\d+$", version):
        print(f"Invalid version: {version}")
        sys.exit(1)

    current = _current_version()
    if _parse_version(version) <= _parse_version(current):
        print(f"Error: {version} is not higher than current {current}")
        sys.exit(1)

    print(f"Bumping {current} → {version}:")
    bump(version)


if __name__ == "__main__":
    main()
