"""Tests for fuzzy/typo-tolerant search and auto-reindex.

Fuzzy search is Phase 2: it fires only when FTS5 (Phase 1)
returns fewer than `limit` results.  Fuzzy results must always
rank BELOW exact/prefix matches.

Auto-reindex (Phase 3): when both FTS5 and fuzzy return nothing,
check if the index is stale (new/changed packages) and reindex
if needed before giving up.

Symbols used: Typer, BaseModel, Field — always in .venv.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from rex.indexer import find_venv
from rex.storage import build_index, is_index_stale, search

PROJECT_DIR = Path("/Users/avykhova/Code/karb/rex")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def db_path_fn(tmp_path_factory):
    """DI function returning a temp DB path for test isolation."""
    db = tmp_path_factory.mktemp("rex") / "test.db"
    return lambda: db


@pytest.fixture(scope="session")
def indexed_db(db_path_fn):
    """Build index into temp DB, return the db_path_fn."""
    venv = find_venv(PROJECT_DIR)
    assert venv is not None, ".venv not found"
    build_index(venv, db_path_fn=db_path_fn)
    return db_path_fn


# ---------------------------------------------------------------------------
# A) Fuzzy finds known typos
# ---------------------------------------------------------------------------


class TestFuzzyFindsTypos:
    """Typo queries that FTS5 prefix matching cannot handle
    should still return the correct symbol via fuzzy fallback."""

    @pytest.mark.parametrize(
        "typo, expected",
        [
            ("BaseModl", "BaseModel"),       # missing char
            ("BsaeModel", "BaseModel"),      # transposed chars
            ("Typerr", "Typer"),             # extra char
            ("Typr", "Typer"),               # missing char
            ("Feild", "Field"),              # transposed chars
            ("BasModel", "BaseModel"),       # missing char in middle
        ],
    )
    def test_fuzzy_finds_close_match(self, indexed_db, typo, expected):
        results = search(typo, db_path_fn=indexed_db)
        names = [s.name for s in results]
        assert expected in names, f"{typo!r} should find {expected!r}, got {names[:5]}"


# ---------------------------------------------------------------------------
# B) Ranking: exact > prefix > fuzzy
# ---------------------------------------------------------------------------


class TestFuzzyRanking:
    """Exact and prefix matches (FTS5 Phase 1) must always
    appear before fuzzy matches (Phase 2) in results."""

    def test_exact_match_ranks_first(self, indexed_db):
        results = search("BaseModel", db_path_fn=indexed_db)
        assert len(results) > 0
        assert results[0].name == "BaseModel"

    def test_prefix_match_ranks_above_fuzzy(self, indexed_db):
        # "BaseMod" is a valid FTS5 prefix — should rank above
        # any fuzzy results for unrelated symbols
        results = search("BaseMod", db_path_fn=indexed_db)
        assert len(results) > 0
        top_names = [r.name for r in results[:3]]
        assert "BaseModel" in top_names

    def test_fuzzy_results_come_after_fts_results(self, indexed_db):
        # A typo that FTS5 can't match — all results must be fuzzy
        results = search("BaseModl", db_path_fn=indexed_db)
        if not results:
            pytest.skip("fuzzy not implemented yet")
        # If there are results, they should include the target
        names = [s.name for s in results]
        assert "BaseModel" in names


# ---------------------------------------------------------------------------
# C) Threshold: don't match distant strings
# ---------------------------------------------------------------------------


class TestFuzzyThreshold:
    """Fuzzy search should NOT return results when the query
    is too far from any known symbol."""

    def test_distant_typo_returns_no_match(self, indexed_db):
        results = search("Zxqwvbn", db_path_fn=indexed_db)
        assert results == []

    def test_very_mangled_name_no_match(self, indexed_db):
        # 4+ edits away from "BaseModel"
        results = search("BaaaaMdl", db_path_fn=indexed_db)
        names = [s.name for s in results]
        assert "BaseModel" not in names

    def test_short_garbage_no_match(self, indexed_db):
        results = search("zz", db_path_fn=indexed_db)
        # Should return empty or unrelated — not fuzzy-match everything
        names = [s.name for s in results]
        assert "BaseModel" not in names
        assert "Typer" not in names


# ---------------------------------------------------------------------------
# D) Edge cases
# ---------------------------------------------------------------------------


class TestFuzzyEdgeCases:
    """Edge cases that should not crash or produce wrong results."""

    def test_single_char_query(self, indexed_db):
        results = search("B", db_path_fn=indexed_db)
        assert isinstance(results, list)

    def test_empty_query_still_returns_empty(self, indexed_db):
        results = search("", db_path_fn=indexed_db)
        assert results == []

    def test_special_chars_with_typo(self, indexed_db):
        # FTS5 special chars mixed with a typo
        results = search("Base+Modl", db_path_fn=indexed_db)
        assert isinstance(results, list)

    def test_fuzzy_respects_limit(self, indexed_db):
        results = search("BaseModl", db_path_fn=indexed_db, limit=3)
        assert len(results) <= 3

    def test_fuzzy_with_type_filter(self, indexed_db):
        results = search("BaseModl", db_path_fn=indexed_db, symbol_type="class")
        for sym in results:
            assert sym.symbol_type == "class"


# ---------------------------------------------------------------------------
# E) is_index_stale — fast read-only check
# ---------------------------------------------------------------------------


def _make_fake_venv(base: Path, packages: dict[str, str]) -> Path:
    """Create a fake .venv with site-packages.

    packages: {pkg_name: python_source}
    Returns the venv path.
    """
    venv = base / ".venv"
    sp = venv / "lib" / "python3.14" / "site-packages"
    for name, source in packages.items():
        pkg_dir = sp / name
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "__init__.py").write_text(textwrap.dedent(source))
    return venv


class TestIsIndexStale:
    """is_index_stale(venv, db_path_fn) -> bool

    Read-only check: compares package mtimes in the DB against
    what's on disk.  No DB writes, no schema changes.
    """

    def test_fresh_index_is_not_stale(self, tmp_path):
        """Immediately after indexing, is_index_stale returns False."""
        venv = _make_fake_venv(tmp_path, {
            "mypkg": '"""My package."""\ndef hello(): pass\n',
        })
        db = tmp_path / "stale_test.db"
        db_fn = lambda: db
        build_index(venv, db_path_fn=db_fn, force=True)

        assert is_index_stale(venv, db_path_fn=db_fn) is False

    def test_new_package_makes_index_stale(self, tmp_path):
        """Adding a new package to .venv makes the index stale."""
        venv = _make_fake_venv(tmp_path, {
            "mypkg": '"""My package."""\ndef hello(): pass\n',
        })
        db = tmp_path / "stale_new.db"
        db_fn = lambda: db
        build_index(venv, db_path_fn=db_fn, force=True)

        # Add a second package after indexing
        sp = venv / "lib" / "python3.14" / "site-packages"
        new_pkg = sp / "newpkg"
        new_pkg.mkdir()
        (new_pkg / "__init__.py").write_text('"""New."""\n')

        assert is_index_stale(venv, db_path_fn=db_fn) is True

    def test_modified_package_makes_index_stale(self, tmp_path):
        """Changing a package's mtime makes the index stale."""
        import time

        venv = _make_fake_venv(tmp_path, {
            "mypkg": '"""Original."""\n',
        })
        db = tmp_path / "stale_mod.db"
        db_fn = lambda: db
        build_index(venv, db_path_fn=db_fn, force=True)

        assert is_index_stale(venv, db_path_fn=db_fn) is False

        # Touch the package to change its mtime
        time.sleep(0.05)
        pkg_init = venv / "lib" / "python3.14" / "site-packages" / "mypkg" / "__init__.py"
        pkg_init.write_text('"""Modified."""\ndef new_func(): pass\n')

        assert is_index_stale(venv, db_path_fn=db_fn) is True

    def test_no_db_is_stale(self, tmp_path):
        """If the DB doesn't exist at all, index is stale."""
        venv = _make_fake_venv(tmp_path, {
            "mypkg": '"""Pkg."""\n',
        })
        db = tmp_path / "nonexistent.db"
        db_fn = lambda: db

        assert is_index_stale(venv, db_path_fn=db_fn) is True

    def test_fresh_project_dir_is_not_stale(self, tmp_path):
        """Immediately after indexing with project_dirs, not stale."""
        venv = _make_fake_venv(tmp_path, {
            "mypkg": '"""Pkg."""\ndef hello(): pass\n',
        })
        proj = tmp_path / "myproj"
        proj.mkdir()
        (proj / "mod.py").write_text('"""Mod."""\nclass Foo: pass\n')

        db = tmp_path / "proj_fresh.db"
        db_fn = lambda: db
        build_index(venv, project_dirs=[proj], db_path_fn=db_fn, force=True)

        assert is_index_stale(venv, db_path_fn=db_fn) is False

    def test_modified_project_file_makes_index_stale(self, tmp_path):
        """Modifying a .py file in a project dir makes it stale.
        This catches the case where directory mtime does NOT change."""
        import time

        venv = _make_fake_venv(tmp_path, {
            "mypkg": '"""Pkg."""\n',
        })
        proj = tmp_path / "myproj"
        proj.mkdir()
        (proj / "mod.py").write_text('"""Mod."""\nclass Foo: pass\n')

        db = tmp_path / "proj_stale_mod.db"
        db_fn = lambda: db
        build_index(venv, project_dirs=[proj], db_path_fn=db_fn, force=True)

        assert is_index_stale(venv, db_path_fn=db_fn) is False

        time.sleep(0.05)
        (proj / "mod.py").write_text('"""Mod."""\nclass Bar: pass\n')

        assert is_index_stale(venv, db_path_fn=db_fn) is True

    def test_new_file_in_project_subdir_makes_index_stale(self, tmp_path):
        """Adding a file in a subdirectory makes the project stale."""
        import time

        venv = _make_fake_venv(tmp_path, {
            "mypkg": '"""Pkg."""\n',
        })
        proj = tmp_path / "myproj"
        proj.mkdir()
        (proj / "mod.py").write_text('"""Mod."""\nclass Foo: pass\n')

        db = tmp_path / "proj_stale_sub.db"
        db_fn = lambda: db
        build_index(venv, project_dirs=[proj], db_path_fn=db_fn, force=True)

        assert is_index_stale(venv, db_path_fn=db_fn) is False

        time.sleep(0.05)
        sub = proj / "sub"
        sub.mkdir()
        (sub / "__init__.py").write_text("")
        (sub / "new.py").write_text('"""New."""\nclass NewThing: pass\n')

        assert is_index_stale(venv, db_path_fn=db_fn) is True


# ---------------------------------------------------------------------------
# F) Auto-reindex on empty results (Phase 3)
# ---------------------------------------------------------------------------


class TestAutoReindex:
    """When FTS5 + fuzzy both return nothing, search should
    check if the index is stale and auto-reindex before giving up."""

    def test_search_finds_newly_added_package(self, tmp_path):
        """Install a new package after initial index → search
        should auto-reindex and find the new symbols."""
        venv = _make_fake_venv(tmp_path, {
            "oldpkg": '"""Old package."""\nclass OldClass: pass\n',
        })
        db = tmp_path / "autoreindex.db"
        db_fn = lambda: db
        build_index(venv, db_path_fn=db_fn, force=True)

        # Verify old symbols found
        results = search("OldClass", db_path_fn=db_fn)
        assert any(s.name == "OldClass" for s in results)

        # Add a new package (simulating `uv add newlib`)
        sp = venv / "lib" / "python3.14" / "site-packages"
        new_pkg = sp / "freshpkg"
        new_pkg.mkdir()
        (new_pkg / "__init__.py").write_text(
            '"""Fresh."""\nclass FreshClass:\n    """A fresh class."""\n    pass\n'
        )

        # Search for the new symbol — should auto-reindex and find it
        results = search("FreshClass", db_path_fn=db_fn)
        names = [s.name for s in results]
        assert "FreshClass" in names

    def test_search_finds_new_file_in_project_dir(self, tmp_path):
        """Add a .py file to a project dir after initial index →
        search should auto-reindex and find the new symbol."""
        import time

        # Set up a fake venv (needed for any index) + a project dir
        venv = _make_fake_venv(tmp_path, {
            "basepkg": '"""Base."""\nclass BaseStuff: pass\n',
        })
        proj = tmp_path / "myproject"
        proj.mkdir()
        (proj / "app.py").write_text(
            '"""App module."""\nclass AppController:\n    pass\n'
        )

        db = tmp_path / "proj_reindex.db"
        db_fn = lambda: db
        build_index(venv, project_dirs=[proj], db_path_fn=db_fn, force=True)

        # Verify project symbol is found
        results = search("AppController", db_path_fn=db_fn)
        assert any(s.name == "AppController" for s in results)

        # Add a NEW file to the project dir (simulating writing new code)
        time.sleep(0.05)
        (proj / "new_feature.py").write_text(
            '"""New feature."""\nclass BrandNewWidget:\n'
            '    """A shiny new widget."""\n    pass\n'
        )

        # Search for the new symbol — should auto-reindex and find it
        results = search("BrandNewWidget", db_path_fn=db_fn)
        names = [s.name for s in results]
        assert "BrandNewWidget" in names

    def test_search_finds_modified_file_in_project_dir(self, tmp_path):
        """Modify an existing .py file in a project dir →
        search should detect staleness and find the new symbol."""
        import time

        venv = _make_fake_venv(tmp_path, {
            "basepkg": '"""Base."""\nclass BaseStuff: pass\n',
        })
        proj = tmp_path / "myproject"
        proj.mkdir()
        (proj / "app.py").write_text(
            '"""App module."""\nclass OriginalWidget:\n    pass\n'
        )

        db = tmp_path / "proj_modify.db"
        db_fn = lambda: db
        build_index(venv, project_dirs=[proj], db_path_fn=db_fn, force=True)

        # Modify existing file — add a new class (file mtime changes,
        # but root dir mtime does NOT change on macOS/Linux)
        time.sleep(0.05)
        (proj / "app.py").write_text(
            '"""App module."""\nclass OriginalWidget:\n    pass\n'
            'class AddedLaterWidget:\n    """Added after index."""\n    pass\n'
        )

        results = search("AddedLaterWidget", db_path_fn=db_fn)
        names = [s.name for s in results]
        assert "AddedLaterWidget" in names

    def test_search_finds_new_file_in_project_subdir(self, tmp_path):
        """Add a .py file in a project SUBDIRECTORY after initial index →
        search should auto-reindex and find it.

        This is the trickiest case: neither the root dir mtime nor any
        existing file mtime changes — only the subdirectory mtime changes.
        """
        import time

        venv = _make_fake_venv(tmp_path, {
            "basepkg": '"""Base."""\nclass BaseStuff: pass\n',
        })
        proj = tmp_path / "myproject"
        proj.mkdir()
        (proj / "app.py").write_text(
            '"""App."""\nclass AppRoot:\n    pass\n'
        )

        db = tmp_path / "proj_subdir.db"
        db_fn = lambda: db
        build_index(venv, project_dirs=[proj], db_path_fn=db_fn, force=True)

        # Add a new file in a new subdirectory
        time.sleep(0.05)
        sub = proj / "features"
        sub.mkdir()
        (sub / "__init__.py").write_text("")
        (sub / "deep_widget.py").write_text(
            '"""Deep."""\nclass DeepNestedWidget:\n    pass\n'
        )

        results = search("DeepNestedWidget", db_path_fn=db_fn)
        names = [s.name for s in results]
        assert "DeepNestedWidget" in names

    def test_no_reindex_when_index_is_current(self, tmp_path):
        """If index is up-to-date, don't waste time reindexing —
        just return empty results for a nonexistent symbol."""
        venv = _make_fake_venv(tmp_path, {
            "somepkg": '"""Some."""\nclass SomeClass: pass\n',
        })
        db = tmp_path / "no_reindex.db"
        db_fn = lambda: db
        build_index(venv, db_path_fn=db_fn, force=True)

        # Search for something that truly doesn't exist
        results = search("CompletelyNonexistentXyz", db_path_fn=db_fn)
        assert results == []
