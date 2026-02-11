"""Rex benchmark — rerun-sdk symbol search quality and timing.

Tests that rex can find, show, and list members for symbols in the
rerun-sdk package. Covers: archetypes, server API, logging functions,
styling, custom components, and Protocol classes.

Usage:
    uv run python bench/bench_rerun.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rex.storage import SearchResult, get_symbol, search, get_members


@dataclass
class Query:
    op: str  # "find", "show", "members"
    query: str
    expect: list[str]  # substrings to match in results
    description: str = ""


@dataclass
class QueryResult:
    op: str
    query: str
    expect: list[str]
    found: list[str]
    hits: list[str]
    misses: list[str]
    elapsed_ms: float
    description: str = ""

    @property
    def score(self) -> float:
        return len(self.hits) / len(self.expect) if self.expect else 1.0


@dataclass
class SuiteResult:
    name: str
    queries: list[QueryResult] = field(default_factory=list)

    @property
    def score(self) -> float:
        return (
            sum(q.score for q in self.queries) / len(self.queries)
            if self.queries
            else 0.0
        )

    @property
    def total_ms(self) -> float:
        return sum(q.elapsed_ms for q in self.queries)


# ---------------------------------------------------------------------------
# Query suites — grouped by API surface, not by domain task
# ---------------------------------------------------------------------------

ARCHETYPE_LOOKUP = [
    Query("find", "Scalars", ["Scalars"],
          "core scalar archetype"),
    Query("find", "TextLog", ["TextLog"],
          "text log archetype"),
    Query("find", "SeriesLines", ["SeriesLines"],
          "line series styling"),
    Query("find", "TextDocument", ["TextDocument"],
          "markdown text document"),
    Query("find", "BarChart", ["BarChart"],
          "bar chart archetype"),
    Query("find", "GeoPoints", ["GeoPoints"],
          "geo points archetype"),
    Query("find", "Points2D", ["Points2D"],
          "2D points archetype"),
    Query("find", "DynamicArchetype", ["DynamicArchetype"],
          "custom archetype"),
    Query("find", "AnyValues", ["AnyValues"],
          "ad-hoc custom components"),
]

NAME_RESOLUTION = [
    Query("show", "Scalars", ["Scalar"],
          "bare class name"),
    Query("show", "rerun.Scalars", ["Scalar"],
          "package.Class shorthand"),
    Query("show", "rerun.archetypes.scalars.Scalars", ["Scalar"],
          "full qualified name"),
    Query("show", "SeriesLines", ["color", "width"],
          "archetype with styling fields"),
    Query("show", "TextDocument", ["text", "media_type"],
          "archetype field inspection"),
    Query("show", "BarChart", ["values"],
          "archetype field inspection"),
    Query("show", "DynamicArchetype", ["archetype", "component"],
          "custom archetype details"),
    Query("show", "AnyValues", ["AnyValues"],
          "ad-hoc components details"),
]

SERVER_AND_LOGGING = [
    Query("find", "send_columns", ["send_columns"],
          "batch logging function"),
    Query("find", "rerun server", ["rerun.server"],
          "server API for reading rrd files"),
    Query("find", "TimeColumn", ["TimeColumn"],
          "time column for batch logging"),
    Query("find", "set_time", ["set_time"],
          "set timeline for point logging"),
    Query("find", "Recording", ["Recording"],
          "recording from compiled bindings (.pyi stub)"),
    Query("show", "Recording", ["schema", "recording"],
          "recording class from .pyi stub"),
]

MEMBERS_INSPECTION = [
    Query("members", "TimeColumn", ["timeline_name", "as_arrow_array"],
          "Protocol class members"),
    Query("members", "RecordingStream", ["log", "send_columns"],
          "recording stream key methods"),
    Query("members", "Recording", ["schema", "recording_id"],
          "compiled extension members from .pyi"),
]

SUITES = {
    "Archetype lookup": ARCHETYPE_LOOKUP,
    "Name resolution": NAME_RESOLUTION,
    "Server & logging": SERVER_AND_LOGGING,
    "Members inspection": MEMBERS_INSPECTION,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_query(q: Query) -> QueryResult:
    t0 = time.perf_counter()
    found_names: list[str] = []

    if q.op == "find":
        result: SearchResult = search(q.query, limit=15)
        found_names = [s.qualified_name for s in result.symbols]
    elif q.op == "show":
        sym = get_symbol(q.query)
        if sym is not None:
            text = f"{sym.qualified_name} {sym.signature or ''} {sym.docstring or ''}"
            found_names = [text]
    elif q.op == "members":
        found_names = [m.name for m in get_members(q.query)]

    elapsed = (time.perf_counter() - t0) * 1000

    hits, misses = [], []
    for exp in q.expect:
        if any(exp.lower() in f.lower() for f in found_names):
            hits.append(exp)
        else:
            misses.append(exp)

    return QueryResult(
        op=q.op, query=q.query, expect=q.expect,
        found=found_names[:5], hits=hits, misses=misses,
        elapsed_ms=elapsed, description=q.description,
    )


def run_benchmark() -> dict[str, SuiteResult]:
    results = {}
    for name, queries in SUITES.items():
        sr = SuiteResult(name=name)
        for q in queries:
            sr.queries.append(run_query(q))
        results[name] = sr
    return results


def print_report(results: dict[str, SuiteResult]):
    total_queries = 0
    total_hits = 0
    total_expect = 0
    total_ms = 0.0

    print("=" * 72)
    print("REX BENCHMARK — rerun-sdk")
    print("=" * 72)

    for name, sr in results.items():
        print(f"\n{'─' * 72}")
        print(f"{name}  |  score: {sr.score:.0%}  |"
              f"  {sr.total_ms:.0f}ms")
        print(f"{'─' * 72}")

        for qr in sr.queries:
            icon = "✓" if qr.score == 1.0 else (
                "◐" if qr.score > 0 else "✗")
            print(f"  {icon} {qr.op:7} {qr.query:<30} "
                  f"{qr.score:.0%}  {qr.elapsed_ms:6.1f}ms"
                  f"  {qr.description}")
            if qr.misses:
                print(f"    MISS: {qr.misses}")
            if qr.found and qr.score < 1.0:
                print(f"    GOT:  {[f[:60] for f in qr.found[:3]]}")

        total_queries += len(sr.queries)
        total_hits += sum(len(qr.hits) for qr in sr.queries)
        total_expect += sum(len(qr.expect) for qr in sr.queries)
        total_ms += sr.total_ms

    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Suites:       {len(results)}")
    print(f"  Queries:      {total_queries}")
    print(f"  Expectations: {total_expect}")
    if total_expect:
        print(f"  Hits:         {total_hits} / {total_expect}"
              f"  ({total_hits / total_expect:.0%})")
    print(f"  Total time:   {total_ms:.0f}ms")
    if total_queries:
        print(f"  Avg per query:{total_ms / total_queries:.1f}ms")

    print(f"\n  {'Suite':<30} {'Score':>6} {'Time':>8}")
    print(f"  {'─' * 30} {'─' * 6} {'─' * 8}")
    for name, sr in results.items():
        print(f"  {name:<30} {sr.score:>5.0%}"
              f" {sr.total_ms:>7.0f}ms")


if __name__ == "__main__":
    results = run_benchmark()
    print_report(results)
