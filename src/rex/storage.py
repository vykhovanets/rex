"""SQLite storage with FTS5 for fast symbol search."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .indexer import Symbol, find_site_packages, find_venv, index_venv


def get_db_path(venv: Path) -> Path:
    """Get the database path for a venv."""
    return venv / ".pydocs.db"


def needs_reindex(venv: Path) -> bool:
    """Check if the index needs to be rebuilt."""
    db_path = get_db_path(venv)
    if not db_path.exists():
        return True

    site_packages = find_site_packages(venv)
    if site_packages is None:
        return True

    # Check if any package was modified after the index
    db_mtime = db_path.stat().st_mtime

    for entry in site_packages.iterdir():
        if entry.stat().st_mtime > db_mtime:
            return True

    return False


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


def build_index(venv: Path | None = None, force: bool = False, progress_callback=None) -> int:
    """Build or rebuild the index for a venv."""
    if venv is None:
        venv = find_venv()
    if venv is None:
        raise RuntimeError("No .venv found")

    if not force and not needs_reindex(venv):
        return -1  # Already up to date

    db_path = get_db_path(venv)

    # Remove old database
    if db_path.exists():
        db_path.unlink()

    with get_connection(db_path) as conn:
        create_schema(conn)
        symbols = index_venv(venv, progress_callback)
        count = insert_symbols(conn, symbols)

    return count


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


def search(
    query: str, venv: Path | None = None, limit: int = 50, symbol_type: str | None = None
) -> list[Symbol]:
    """Search for symbols matching query."""
    if venv is None:
        venv = find_venv()
    if venv is None:
        return []

    db_path = get_db_path(venv)
    if not db_path.exists():
        return []

    # Handle "Class.method" style queries - search for inherited methods too
    if "." in query:
        results = _search_with_inheritance(query, venv, limit)
        if results:
            return results

    with get_connection(db_path) as conn:
        # Use FTS5 for search
        # Split query into terms and create proper FTS query
        terms = query.split()

        # Build FTS query - avoid phrase matching (quotes) since unicode61
        # tokenizer splits on underscores, making phrase matches fail for
        # identifiers like "post_process_instance_segmentation"
        fts_parts = []
        for term in terms:
            # Split underscored terms into individual tokens for better matching
            subtokens = term.replace("_", " ").split()
            for subtoken in subtokens:
                # Use prefix matching without quotes for flexibility
                escaped = subtoken.replace('"', '""')
                fts_parts.append(f"{escaped}*")
        fts_query = " ".join(fts_parts)

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
        except sqlite3.OperationalError:
            # FTS query failed, fall back to LIKE
            # For multiple terms, require all to match
            sql = "SELECT * FROM symbols WHERE 1=1"
            params = []
            for term in terms:
                sql += " AND (name LIKE ? OR qualified_name LIKE ?)"
                params.extend([f"%{term}%", f"%{term}%"])
            if symbol_type:
                sql += " AND symbol_type = ?"
                params.append(symbol_type)
            sql += " LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()

        return [row_to_symbol(row) for row in rows]


def _search_with_inheritance(query: str, venv: Path, limit: int) -> list[Symbol]:
    """Search with inheritance awareness for 'Class.method' patterns."""
    parts = query.rsplit(".", 1)
    if len(parts) != 2:
        return []

    class_part, method_part = parts
    class_name = class_part.split(".")[-1]  # Get just the class name

    db_path = get_db_path(venv)
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


def get_symbol(qualified_name: str, venv: Path | None = None) -> Symbol | None:
    """Get a symbol by its qualified name."""
    if venv is None:
        venv = find_venv()
    if venv is None:
        return None

    db_path = get_db_path(venv)
    if not db_path.exists():
        return None

    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM symbols WHERE qualified_name = ?", (qualified_name,)
        ).fetchone()
        if row:
            return row_to_symbol(row)
    return None


def get_members(qualified_name: str, venv: Path | None = None) -> list[Symbol]:
    """Get members of a class or module."""
    if venv is None:
        venv = find_venv()
    if venv is None:
        return []

    db_path = get_db_path(venv)
    if not db_path.exists():
        return []

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


def get_stats(venv: Path | None = None) -> dict:
    """Get index statistics."""
    if venv is None:
        venv = find_venv()
    if venv is None:
        return {"error": "No venv found"}

    db_path = get_db_path(venv)
    if not db_path.exists():
        return {"error": "No index found", "venv": str(venv)}

    with get_connection(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        by_type = dict(
            conn.execute(
                "SELECT symbol_type, COUNT(*) FROM symbols GROUP BY symbol_type"
            ).fetchall()
        )
        packages = conn.execute(
            """
            SELECT COUNT(DISTINCT substr(qualified_name, 1, instr(qualified_name, '.') - 1))
            FROM symbols
        """
        ).fetchone()[0]

    return {
        "venv": str(venv),
        "db_path": str(db_path),
        "total_symbols": total,
        "by_type": by_type,
        "packages": packages,
    }
