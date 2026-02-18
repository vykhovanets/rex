"""Rex benchmark — marimo symbol search quality and timing.

Tests that rex can find, show, and list members for symbols in the
marimo package. Covers: UI input elements, data/chart widgets,
stateless display components, core API, and name resolution.

Usage:
    uv run python bench/bench_marimo.py
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
# Query suites — grouped by API surface
# ---------------------------------------------------------------------------

UI_INPUT_ELEMENTS = [
    Query("find", "slider", ["slider"],
          "range slider widget"),
    Query("find", "button", ["button"],
          "click button"),
    Query("find", "checkbox", ["checkbox"],
          "boolean checkbox"),
    Query("find", "dropdown", ["dropdown"],
          "selection dropdown"),
    Query("find", "radio", ["radio"],
          "radio button group"),
    Query("find", "multiselect", ["multiselect"],
          "multi-selection widget"),
    Query("find", "marimo text", ["_impl.input.text"],
          "text input field (qualified to marimo)"),
    Query("find", "marimo number", ["_impl.input.number"],
          "number input field (qualified to marimo)"),
    Query("find", "code_editor", ["code_editor"],
          "code editor widget"),
    Query("find", "file_browser", ["file_browser"],
          "file browser widget"),
    Query("find", "form", ["form"],
          "form wrapper"),
    Query("find", "marimo switch", ["_impl.switch.switch"],
          "toggle switch widget (qualified to marimo)"),
]

UI_DATA_ELEMENTS = [
    Query("find", "marimo dataframe", ["dataframe.dataframe"],
          "dataframe viewer/editor"),
    Query("find", "marimo table", ["_impl.table"],
          "interactive table"),
    Query("find", "plotly", ["marimo"],
          "plotly chart widget"),
    Query("find", "altair_chart", ["altair_chart"],
          "altair chart widget"),
    Query("find", "marimo chat", ["chat.chat"],
          "chat interface widget"),
    Query("find", "microphone", ["microphone"],
          "audio capture widget"),
]

STATELESS_DISPLAY = [
    Query("find", "accordion", ["marimo"],
          "collapsible accordion"),
    Query("find", "callout", ["marimo"],
          "styled callout box"),
    Query("find", "mermaid", ["marimo"],
          "mermaid diagram renderer"),
    Query("find", "carousel", ["marimo"],
          "image/content carousel"),
    Query("find", "tabs", ["marimo"],
          "tabbed container"),
    Query("find", "sidebar", ["marimo"],
          "sidebar layout element"),
    Query("find", "nav_menu", ["marimo"],
          "navigation menu"),
    Query("find", "marimo video", ["marimo"],
          "video display"),
]

CORE_API = [
    Query("find", "MarimoIslandGenerator", ["MarimoIslandGenerator"],
          "island mode generator"),
    Query("find", "create_asgi_app", ["create_asgi_app"],
          "ASGI app factory"),
    Query("find", "UIElement", ["UIElement"],
          "base UI element class"),
    Query("find", "marimo Cell", ["marimo"],
          "notebook cell class"),
    Query("find", "marimo App", ["marimo"],
          "application class"),
]

NAME_RESOLUTION = [
    Query("show", "slider", ["slider"],
          "bare class name"),
    Query("show", "file_browser", ["file_browser"],
          "bare compound name"),
    Query("show", "marimo._plugins.ui._impl.input.slider",
          ["slider", "start", "stop"],
          "full qualified name"),
    Query("show", "UIElement", ["UIElement"],
          "base class from internal module"),
    Query("show", "MarimoIslandGenerator", ["MarimoIslandGenerator"],
          "top-level re-exported class"),
    Query("show", "create_asgi_app", ["create_asgi_app"],
          "top-level re-exported function"),
]

MEMBERS_INSPECTION = [
    Query("members", "slider", ["__init__"],
          "slider class members"),
    Query("members", "UIElement", ["__init__"],
          "base UI element members"),
    Query("members",
          "marimo._plugins.ui._impl.file_browser.file_browser",
          ["__init__"],
          "file_browser class members"),
]

SUITES = {
    "UI input elements": UI_INPUT_ELEMENTS,
    "UI data elements": UI_DATA_ELEMENTS,
    "Stateless display": STATELESS_DISPLAY,
    "Core API": CORE_API,
    "Name resolution": NAME_RESOLUTION,
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
            text = (f"{sym.qualified_name} "
                    f"{sym.signature or ''} "
                    f"{sym.docstring or ''}")
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
    print("REX BENCHMARK — marimo")
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
