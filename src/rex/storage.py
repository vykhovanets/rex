"""SQLite storage with FTS5 for fast symbol search."""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

from .indexer import (
    Symbol,
    find_site_packages,
    find_venv,
    index_directory,
    index_package,
    index_site_packages,
    iter_packages,
)


def get_db_path() -> Path:
    """Get the global database path."""
    state_dir = Path.home() / ".local" / "state" / "rex"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "rex.db"


@contextmanager
def get_connection(db_path: Path):
    """Get a database connection with proper settings."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for better concurrency
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
    finally:
        conn.close()


def create_schema(conn: sqlite3.Connection) -> None:
    """Create the database schema."""
    conn.executescript(
        """
        -- Main symbols table
        CREATE TABLE IF NOT EXISTS symbols (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            qualified_name TEXT NOT NULL UNIQUE,
            symbol_type TEXT NOT NULL,
            signature TEXT,
            docstring TEXT,
            file_path TEXT NOT NULL,
            line_no INTEGER NOT NULL,
            bases TEXT,  -- JSON array
            return_annotation TEXT
        );

        -- FTS5 index for fast search
        CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
            name,
            qualified_name,
            docstring,
            content='symbols',
            content_rowid='id',
            tokenize='porter unicode61'
        );

        -- Triggers to keep FTS in sync
        CREATE TRIGGER IF NOT EXISTS symbols_ai AFTER INSERT ON symbols BEGIN
            INSERT INTO symbols_fts(rowid, name, qualified_name, docstring)
            VALUES (new.id, new.name, new.qualified_name, new.docstring);
        END;

        CREATE TRIGGER IF NOT EXISTS symbols_ad AFTER DELETE ON symbols BEGIN
            INSERT INTO symbols_fts(symbols_fts, rowid, name, qualified_name, docstring)
            VALUES ('delete', old.id, old.name, old.qualified_name, old.docstring);
        END;

        -- Packages metadata table
        CREATE TABLE IF NOT EXISTS packages (
            name TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            mtime REAL NOT NULL,
            symbol_count INTEGER NOT NULL DEFAULT 0
        );

        -- Index for common queries
        CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
        CREATE INDEX IF NOT EXISTS idx_symbols_type ON symbols(symbol_type);
        CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_path);
    """
    )
    conn.commit()


def insert_symbols(conn: sqlite3.Connection, symbols: Iterator[Symbol]) -> int:
    """Insert symbols into the database. Returns count."""
    count = 0
    batch = []
    batch_size = 1000

    for symbol in symbols:
        batch.append(
            (
                symbol.name,
                symbol.qualified_name,
                symbol.symbol_type,
                symbol.signature,
                symbol.docstring,
                symbol.file_path,
                symbol.line_no,
                ",".join(symbol.bases) if symbol.bases else None,
                symbol.return_annotation,
            )
        )
        count += 1

        if len(batch) >= batch_size:
            conn.executemany(
                """
                INSERT OR REPLACE INTO symbols
                (name, qualified_name, symbol_type, signature, docstring,
                 file_path, line_no, bases, return_annotation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                batch,
            )
            batch = []

    if batch:
        conn.executemany(
            """
            INSERT OR REPLACE INTO symbols
            (name, qualified_name, symbol_type, signature, docstring,
             file_path, line_no, bases, return_annotation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            batch,
        )

    conn.commit()
    return count


def build_index(
    venv: Path | None = None,
    force: bool = False,
    progress_callback=None,
    project_dirs: list[Path] | None = None,
    db_path_fn: Callable[[], Path] = get_db_path,
) -> int:
    """Build or rebuild the index incrementally."""
    if venv is None:
        venv = find_venv()
    if venv is None:
        raise RuntimeError("No .venv found")

    db_path = db_path_fn()
    site_packages = find_site_packages(venv)
    if site_packages is None:
        raise RuntimeError(f"No site-packages found in {venv}")

    with get_connection(db_path) as conn:
        create_schema(conn)

        packages = list(iter_packages(site_packages))
        total_new = 0
        updated_packages = 0

        for i, (pkg_name, pkg_path) in enumerate(packages):
            if progress_callback:
                progress_callback(pkg_name, i + 1, len(packages))

            current_mtime = pkg_path.stat().st_mtime

            if not force:
                row = conn.execute(
                    "SELECT mtime FROM packages WHERE name = ?",
                    (pkg_name,),
                ).fetchone()
                if row and row["mtime"] == current_mtime:
                    continue  # up to date

            # Delete old symbols for this package
            conn.execute(
                "DELETE FROM symbols WHERE qualified_name LIKE ? OR qualified_name = ?",
                (f"{pkg_name}.%", pkg_name),
            )

            # Index and insert new symbols
            symbols = list(index_package(pkg_name, pkg_path))
            count = insert_symbols(conn, iter(symbols))

            # Upsert packages metadata
            conn.execute(
                "INSERT OR REPLACE INTO packages (name, source_path, mtime, symbol_count) VALUES (?, ?, ?, ?)",
                (pkg_name, str(pkg_path), current_mtime, count),
            )
            conn.commit()
            total_new += count
            updated_packages += 1

        # Handle project dirs
        if project_dirs:
            for proj_dir in project_dirs:
                proj_dir = proj_dir.resolve()
                source_key = f"project:{proj_dir}"
                current_mtime = proj_dir.stat().st_mtime

                if not force:
                    row = conn.execute(
                        "SELECT mtime FROM packages WHERE source_path = ?",
                        (source_key,),
                    ).fetchone()
                    if row and row["mtime"] == current_mtime:
                        continue

                # Delete old project symbols
                old_pkgs = conn.execute(
                    "SELECT name FROM packages WHERE source_path = ?",
                    (source_key,),
                ).fetchall()
                for old_pkg in old_pkgs:
                    conn.execute(
                        "DELETE FROM symbols WHERE qualified_name LIKE ? OR qualified_name = ?",
                        (f"{old_pkg['name']}.%", old_pkg['name']),
                    )
                conn.execute(
                    "DELETE FROM packages WHERE source_path = ?", (source_key,)
                )

                symbols = list(index_directory(proj_dir))
                count = insert_symbols(conn, iter(symbols))

                conn.execute(
                    "INSERT OR REPLACE INTO packages (name, source_path, mtime, symbol_count) VALUES (?, ?, ?, ?)",
                    (f"project:{proj_dir.name}", source_key, current_mtime, count),
                )
                conn.commit()
                total_new += count
                updated_packages += 1

    if updated_packages == 0 and not force:
        return -1
    return total_new


def clean_index(db_path_fn: Callable[[], Path] = get_db_path) -> list[str]:
    """Remove packages whose source paths no longer exist."""
    db_path = db_path_fn()
    if not db_path.exists():
        return []

    removed = []
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT name, source_path FROM packages").fetchall()
        for row in rows:
            # For project dirs, source_path starts with "project:"
            if row["source_path"].startswith("project:"):
                check_path = Path(row["source_path"][len("project:"):])
            else:
                check_path = Path(row["source_path"])

            if not check_path.exists():
                pkg_name = row["name"]
                conn.execute(
                    "DELETE FROM symbols WHERE qualified_name LIKE ? OR qualified_name = ?",
                    (f"{pkg_name}.%", pkg_name),
                )
                conn.execute("DELETE FROM packages WHERE name = ?", (pkg_name,))
                removed.append(pkg_name)

        if removed:
            conn.commit()

    return removed


def ensure_db(db_path_fn: Callable[[], Path] = get_db_path) -> Path:
    """Return DB path, building index if DB doesn't exist yet."""
    db_path = db_path_fn()
    if not db_path.exists():
        build_index(db_path_fn=db_path_fn)
    return db_path


def row_to_symbol(row: sqlite3.Row) -> Symbol:
    """Convert a database row to a Symbol."""
    bases = tuple(row["bases"].split(",")) if row["bases"] else ()
    return Symbol(
        name=row["name"],
        qualified_name=row["qualified_name"],
        symbol_type=row["symbol_type"],
        signature=row["signature"],
        docstring=row["docstring"],
        file_path=row["file_path"],
        line_no=row["line_no"],
        bases=bases,
        return_annotation=row["return_annotation"],
    )


# Characters that are FTS5 operators or cause syntax errors
_FTS5_SPECIAL = re.compile(r'[+\-@^"(){}:!|&~<>]')


def _sanitize_fts_query(query: str) -> str:
    """Build a safe FTS5 query from user input.

    Strips FTS5 operators, splits underscores, applies prefix matching.
    Returns empty string if no usable tokens remain.
    """
    terms = query.split()
    fts_parts = []
    for term in terms:
        # Remove FTS5 special characters
        clean = _FTS5_SPECIAL.sub(" ", term)
        subtokens = clean.replace("_", " ").split()
        for subtoken in subtokens:
            subtoken = subtoken.strip()
            if subtoken:
                escaped = subtoken.replace('"', '""')
                fts_parts.append(f"{escaped}*")
    return " ".join(fts_parts)


def search(
    query: str,
    limit: int = 50,
    symbol_type: str | None = None,
    db_path_fn: Callable[[], Path] = get_db_path,
) -> list[Symbol]:
    """Search for symbols matching query."""
    if not query or not query.strip():
        return []

    db_path = ensure_db(db_path_fn)

    # Handle "Class.method" style queries — try inheritance search first,
    # but fall through to normal FTS if it returns nothing
    if "." in query:
        results = _search_with_inheritance(query, db_path, limit)
        if results:
            return results

    with get_connection(db_path) as conn:
        fts_query = _sanitize_fts_query(query)

        if fts_query:
            sql = """
                SELECT s.*, bm25(symbols_fts) as rank
                FROM symbols s
                JOIN symbols_fts ON s.id = symbols_fts.rowid
                WHERE symbols_fts MATCH ?
            """
            params: list = [fts_query]

            if symbol_type:
                sql += " AND s.symbol_type = ?"
                params.append(symbol_type)

            sql += " ORDER BY rank LIMIT ?"
            params.append(limit)

            try:
                rows = conn.execute(sql, params).fetchall()
                return [row_to_symbol(row) for row in rows]
            except sqlite3.OperationalError:
                pass  # Fall through to LIKE

        # LIKE fallback — also search docstrings for consistency with FTS
        terms = query.split()
        if not terms:
            return []

        sql = "SELECT * FROM symbols WHERE 1=1"
        params = []
        for term in terms:
            sql += " AND (name LIKE ? OR qualified_name LIKE ? OR docstring LIKE ?)"
            params.extend([f"%{term}%", f"%{term}%", f"%{term}%"])
        if symbol_type:
            sql += " AND symbol_type = ?"
            params.append(symbol_type)
        sql += " ORDER BY name LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()

        return [row_to_symbol(row) for row in rows]


def _search_with_inheritance(query: str, db_path: Path, limit: int) -> list[Symbol]:
    """Search with inheritance awareness for 'Class.method' patterns."""
    parts = query.rsplit(".", 1)
    if len(parts) != 2:
        return []

    class_part, method_part = parts
    class_name = class_part.split(".")[-1]  # Get just the class name

    results = []

    with get_connection(db_path) as conn:
        # Find the class
        class_row = conn.execute(
            "SELECT * FROM symbols WHERE name = ? AND symbol_type = 'class' LIMIT 1",
            (class_name,),
        ).fetchone()

        if not class_row:
            return []

        class_sym = row_to_symbol(class_row)
        bases_to_check = list(class_sym.bases)
        checked_bases = set()

        # First, look for direct method on this class
        direct = conn.execute(
            """SELECT * FROM symbols
               WHERE qualified_name LIKE ? AND name LIKE ?
               ORDER BY qualified_name""",
            (f"%{class_name}.{method_part}%", f"%{method_part}%"),
        ).fetchall()
        results.extend(row_to_symbol(r) for r in direct)

        # Then search in base classes for the method
        while bases_to_check and len(results) < limit:
            base = bases_to_check.pop(0)
            if base in checked_bases:
                continue
            checked_bases.add(base)

            # Extract just the class name from base (could be "module.ClassName")
            base_name = base.split(".")[-1]

            # Find method in base class
            base_methods = conn.execute(
                """SELECT * FROM symbols
                   WHERE qualified_name LIKE ? AND name LIKE ?
                   LIMIT ?""",
                (f"%.{base_name}.{method_part}%", f"%{method_part}%", limit - len(results)),
            ).fetchall()
            results.extend(row_to_symbol(r) for r in base_methods)

            # Get base class to check its bases too
            base_class = conn.execute(
                "SELECT * FROM symbols WHERE name = ? AND symbol_type = 'class' LIMIT 1",
                (base_name,),
            ).fetchone()
            if base_class:
                base_sym = row_to_symbol(base_class)
                bases_to_check.extend(base_sym.bases)

    return results[:limit]


def get_symbol(
    qualified_name: str,
    db_path_fn: Callable[[], Path] = get_db_path,
) -> Symbol | None:
    """Get a symbol by its qualified name."""
    db_path = ensure_db(db_path_fn)

    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM symbols WHERE qualified_name = ?", (qualified_name,)
        ).fetchone()
        if row:
            return row_to_symbol(row)
    return None


def get_members(
    qualified_name: str,
    db_path_fn: Callable[[], Path] = get_db_path,
) -> list[Symbol]:
    """Get members of a class or module."""
    db_path = ensure_db(db_path_fn)

    with get_connection(db_path) as conn:
        # Find symbols that start with this qualified name
        prefix = qualified_name + "."
        rows = conn.execute(
            """
            SELECT * FROM symbols
            WHERE qualified_name LIKE ?
            AND qualified_name NOT LIKE ?
            ORDER BY name
        """,
            (prefix + "%", prefix + "%.%"),
        ).fetchall()
        return [row_to_symbol(row) for row in rows]


def get_stats(db_path_fn: Callable[[], Path] = get_db_path) -> dict:
    """Get index statistics."""
    db_path = db_path_fn()
    if not db_path.exists():
        return {"error": "No index found. Run 'rex index' first."}

    with get_connection(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        by_type = dict(
            conn.execute(
                "SELECT symbol_type, COUNT(*) FROM symbols GROUP BY symbol_type"
            ).fetchall()
        )
        packages = conn.execute("SELECT COUNT(*) FROM packages").fetchone()[0]

    return {
        "db_path": str(db_path),
        "total_symbols": total,
        "by_type": by_type,
        "packages": packages,
    }
