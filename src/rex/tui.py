"""TUI navigator with IDE-style back/forward navigation."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Footer, Header, Input, ListItem, ListView, Static

from .indexer import Symbol
from .storage import get_members, get_symbol, search


# =============================================================================
# Source Code Reading
# =============================================================================


def read_source_context(file_path: str, line_no: int, context: int = 40) -> str:
    """Read source code around a line number."""
    try:
        path = Path(file_path)
        if not path.exists():
            return ""

        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(0, line_no - 1)
        end = min(len(lines), line_no + context)

        # Find the end of the function/class definition by tracking indentation
        indent = None
        for i in range(start, min(len(lines), start + context)):
            line = lines[i]
            if indent is None and line.strip():
                indent = len(line) - len(line.lstrip())
            elif indent is not None and line.strip():
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= indent and i > start + 2:
                    if not line.strip().startswith((")", "]", "}", "@", "#")):
                        end = i
                        break

        return "\n".join(lines[start:end])
    except Exception:
        return ""


def extract_type_names(text: str) -> list[str]:
    """Extract potential type names from a signature or annotation."""
    # Match capitalized words that look like class names
    pattern = r'\b([A-Z][a-zA-Z0-9_]*(?:\[[^\]]+\])?)\b'
    matches = re.findall(pattern, text)
    # Filter out common non-type words
    skip = {"None", "True", "False", "Optional", "Union", "List", "Dict", "Tuple", "Set", "Any", "Callable"}
    return [m.split("[")[0] for m in matches if m.split("[")[0] not in skip]


def escape_markup(text: str) -> str:
    """Escape Rich markup characters in text."""
    return text.replace("[", "\\[").replace("]", "\\]")


# =============================================================================
# Widgets
# =============================================================================


class TypeLink(Static):
    """A clickable type name that navigates to that type."""

    class Clicked(Message):
        def __init__(self, type_name: str) -> None:
            self.type_name = type_name
            super().__init__()

    def __init__(self, type_name: str, display: str | None = None) -> None:
        super().__init__(display or type_name, classes="type-link")
        self.type_name = type_name

    def on_click(self) -> None:
        self.post_message(self.Clicked(self.type_name))


class FileLink(Static):
    """A clickable file path that opens in editor."""

    class Clicked(Message):
        def __init__(self, file_path: str, line_no: int) -> None:
            self.file_path = file_path
            self.line_no = line_no
            super().__init__()

    def __init__(self, file_path: str, line_no: int) -> None:
        display = f"{file_path}:{line_no}"
        super().__init__(f"[link]{display}[/link]", classes="file-link")
        self.file_path = file_path
        self.line_no = line_no

    def on_click(self) -> None:
        self.post_message(self.Clicked(self.file_path, self.line_no))


class SymbolDetail(VerticalScroll):
    """Scrollable display for symbol details with clickable elements."""

    def __init__(self) -> None:
        super().__init__(id="detail-panel")
        self.current_symbol: Symbol | None = None

    def compose(self) -> ComposeResult:
        yield Static("", id="detail-header")
        yield Static("", id="detail-signature")
        yield Static("", id="detail-meta")
        yield Static("", id="detail-doc")
        yield Static("", id="detail-source-header")
        yield Static("", id="detail-file-link")
        yield Static("", id="detail-source")

    def update_symbol(self, symbol: Symbol | None) -> None:
        self.current_symbol = symbol

        header = self.query_one("#detail-header", Static)
        signature = self.query_one("#detail-signature", Static)
        meta = self.query_one("#detail-meta", Static)
        doc = self.query_one("#detail-doc", Static)
        source_header = self.query_one("#detail-source-header", Static)
        file_link = self.query_one("#detail-file-link", Static)
        source = self.query_one("#detail-source", Static)

        if symbol is None:
            for w in [header, signature, meta, doc, source_header, file_link, source]:
                w.update("")
            return

        # Header - escape brackets in qualified_name to prevent markup errors
        safe_qname = escape_markup(symbol.qualified_name)
        header.update(f"[bold cyan]{safe_qname}[/]\n[dim]{symbol.symbol_type}[/]")

        # Signature with clickable types
        if symbol.signature:
            if symbol.symbol_type == "class":
                sig = f"class {symbol.name}{symbol.signature}"
            elif "async" in symbol.symbol_type:
                sig = f"async def {symbol.name}{symbol.signature}"
            elif symbol.symbol_type in ("method", "function"):
                sig = f"def {symbol.name}{symbol.signature}"
            else:
                sig = f"{symbol.name}{symbol.signature}"

            # Highlight type names
            sig_display = self._highlight_types(sig)
            signature.update(f"\n[bold yellow]━━━ Signature ━━━[/]\n{sig_display}")
        else:
            signature.update("")

        # Meta info (bases, return type)
        meta_lines = []
        if symbol.bases:
            bases_display = ", ".join(f"[magenta]{escape_markup(b)}[/]" for b in symbol.bases)
            meta_lines.append(f"[bold]Inherits:[/] {bases_display}")
        if symbol.return_annotation:
            ret_display = self._highlight_types(symbol.return_annotation)
            meta_lines.append(f"[bold]Returns:[/] {ret_display}")
        meta.update("\n".join(meta_lines) + ("\n" if meta_lines else ""))

        # Docstring - escape markup to prevent crashes from brackets in docs
        if symbol.docstring:
            docstring = symbol.docstring.strip()
            if len(docstring) > 5000:
                docstring = docstring[:5000] + "\n... (truncated)"
            safe_docstring = escape_markup(docstring)
            doc.update(f"\n[bold yellow]━━━ Documentation ━━━[/]\n{safe_docstring}")
        else:
            doc.update("")

        # Source code
        source_code = read_source_context(symbol.file_path, symbol.line_no)
        if source_code:
            source_header.update(f"\n[bold yellow]━━━ Source Code ━━━[/]")
            # Display file path without Rich link markup to avoid parsing issues
            file_link.update(
                f"[dim]{symbol.file_path}:{symbol.line_no}[/]  "
                f"[bold cyan]\\[press 'e' to edit][/]"
            )
            # Escape markup in source
            source_escaped = escape_markup(source_code)
            source.update(f"\n[white]{source_escaped}[/]")
        else:
            source_header.update("")
            file_link.update("")
            source.update("")

        self.scroll_home(animate=False)

    def _highlight_types(self, text: str) -> str:
        """Highlight type names in text - escape brackets first."""
        # Escape all brackets to prevent Rich markup conflicts
        escaped = text.replace("[", "\\[").replace("]", "\\]")
        return f"[green]{escaped}[/]"


class SymbolListItem(ListItem):
    """A list item representing a symbol."""

    def __init__(self, symbol: Symbol) -> None:
        super().__init__()
        self.symbol = symbol

    def compose(self) -> ComposeResult:
        sig = self.symbol.signature or ""
        if len(sig) > 45:
            sig = sig[:42] + "..."

        text = Text()
        text.append(f"{self.symbol.symbol_type:8}", style="dim")
        text.append(" ")
        text.append(self.symbol.name, style="bold")
        text.append(sig, style="cyan")

        yield Static(text)


# =============================================================================
# Main Application
# =============================================================================


class PyDocsApp(App):
    """Interactive Python documentation browser."""

    CSS = """
    #search-container {
        dock: top;
        height: 3;
        padding: 0 1;
    }

    #search-input {
        width: 100%;
    }

    #main-container {
        height: 1fr;
    }

    #results-list {
        width: 40%;
        border: solid green;
        min-width: 30;
    }

    #detail-panel {
        width: 60%;
        border: solid blue;
        padding: 1;
    }

    #status-bar {
        dock: bottom;
        height: 1;
        background: $surface;
        padding: 0 1;
    }

    .type-link {
        color: $primary;
        text-style: underline;
    }

    .type-link:hover {
        background: $primary-darken-2;
    }

    .file-link {
        color: $text-muted;
    }

    .file-link:hover {
        color: $primary;
        text-style: underline;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+o", "go_back", "Back", show=True),
        Binding("ctrl+i", "go_forward", "Forward", show=True),
        Binding("enter", "dive_in", "Dive In", show=True),
        Binding("e", "edit", "Edit", show=True),
        Binding("m", "members", "Members", show=True),
        Binding("g", "goto_type", "Goto Type", show=True),
        Binding("/", "focus_search", "Search", show=True),
        Binding("escape", "focus_list", "List", show=False),
        Binding("tab", "toggle_focus", "Toggle Focus", show=False),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("q", "quit", "Quit", show=True),
    ]

    def __init__(self, initial_query: str = "") -> None:
        super().__init__()
        self.initial_query = initial_query
        self.nav_stack: list[str] = []
        self.nav_index: int = -1
        self.current_results: list[Symbol] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="search-container"):
            yield Input(placeholder="Search symbols... (e.g., Sam3Processor.from_pretrained)", id="search-input")
        with Horizontal(id="main-container"):
            yield ListView(id="results-list")
            yield SymbolDetail()
        yield Static("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        if self.initial_query:
            search_input = self.query_one("#search-input", Input)
            search_input.value = self.initial_query
            self.do_search(self.initial_query)
        self.query_one("#results-list", ListView).focus()

    def do_search(self, query: str) -> None:
        """Perform search and update results."""
        if not query.strip():
            self.current_results = []
            self.update_results([])
            return

        results = search(query, limit=100)
        self.current_results = results
        self.update_results(results)

    def update_results(self, results: list[Symbol]) -> None:
        """Update the results list."""
        results_list = self.query_one("#results-list", ListView)
        results_list.clear()

        for symbol in results:
            results_list.append(SymbolListItem(symbol))

        self.update_status(f"{len(results)} results")

    def update_status(self, msg: str) -> None:
        nav_info = f"[{self.nav_index + 1}/{len(self.nav_stack)}]" if self.nav_stack else ""
        self.query_one("#status-bar", Static).update(f"{nav_info} {msg}")

    def navigate_to(self, qualified_name: str) -> None:
        """Navigate to a symbol, updating the stack."""
        if self.nav_index < len(self.nav_stack) - 1:
            self.nav_stack = self.nav_stack[: self.nav_index + 1]

        self.nav_stack.append(qualified_name)
        self.nav_index = len(self.nav_stack) - 1

        symbol = get_symbol(qualified_name)
        if symbol:
            detail = self.query_one(SymbolDetail)
            detail.update_symbol(symbol)
            self.update_status(f"Viewing: {symbol.name}")

    def get_highlighted_symbol(self) -> Symbol | None:
        """Get the currently highlighted symbol."""
        results_list = self.query_one("#results-list", ListView)
        if results_list.highlighted_child and isinstance(
            results_list.highlighted_child, SymbolListItem
        ):
            return results_list.highlighted_child.symbol
        return None

    def open_in_editor(self, file_path: str, line_no: int) -> None:
        """Open a file in the user's editor."""
        editor = os.environ.get("EDITOR", "vim")

        with self.suspend():
            if editor in ("vim", "nvim", "vi"):
                subprocess.run([editor, f"+{line_no}", file_path])
            elif editor in ("code", "code-insiders"):
                subprocess.run([editor, "--goto", f"{file_path}:{line_no}"])
            elif editor == "subl":
                subprocess.run([editor, f"{file_path}:{line_no}"])
            else:
                subprocess.run([editor, file_path])

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    @on(Input.Changed, "#search-input")
    def on_search_change(self, event: Input.Changed) -> None:
        self.do_search(event.value)

    @on(Input.Submitted, "#search-input")
    def on_search_submitted(self, event: Input.Submitted) -> None:
        self.query_one("#results-list", ListView).focus()

    @on(ListView.Selected, "#results-list")
    def on_result_selected(self, event: ListView.Selected) -> None:
        if isinstance(event.item, SymbolListItem):
            self.navigate_to(event.item.symbol.qualified_name)

    @on(ListView.Highlighted, "#results-list")
    def on_result_highlighted(self, event: ListView.Highlighted) -> None:
        if isinstance(event.item, SymbolListItem):
            detail = self.query_one(SymbolDetail)
            detail.update_symbol(event.item.symbol)

    @on(TypeLink.Clicked)
    def on_type_clicked(self, event: TypeLink.Clicked) -> None:
        """Navigate to a clicked type."""
        self.query_one("#search-input", Input).value = event.type_name
        self.do_search(event.type_name)
        self.query_one("#results-list", ListView).focus()

    @on(FileLink.Clicked)
    def on_file_clicked(self, event: FileLink.Clicked) -> None:
        """Open clicked file in editor."""
        self.open_in_editor(event.file_path, event.line_no)

    # -------------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------------

    def action_go_back(self) -> None:
        """Go back in navigation history."""
        if self.nav_index > 0:
            self.nav_index -= 1
            qname = self.nav_stack[self.nav_index]
            symbol = get_symbol(qname)
            if symbol:
                detail = self.query_one(SymbolDetail)
                detail.update_symbol(symbol)
                self.update_status(f"Back to: {symbol.name}")

    def action_go_forward(self) -> None:
        """Go forward in navigation history."""
        if self.nav_index < len(self.nav_stack) - 1:
            self.nav_index += 1
            qname = self.nav_stack[self.nav_index]
            symbol = get_symbol(qname)
            if symbol:
                detail = self.query_one(SymbolDetail)
                detail.update_symbol(symbol)
                self.update_status(f"Forward to: {symbol.name}")

    def action_dive_in(self) -> None:
        """Dive into the selected symbol (show its members)."""
        symbol = self.get_highlighted_symbol()
        if symbol:
            self.navigate_to(symbol.qualified_name)
            members = get_members(symbol.qualified_name)
            if members:
                self.current_results = members
                self.update_results(members)
                search_input = self.query_one("#search-input", Input)
                search_input.value = f"{symbol.qualified_name}."

    def action_members(self) -> None:
        """Show members of the currently viewed symbol."""
        if self.nav_index >= 0:
            qname = self.nav_stack[self.nav_index]
            members = get_members(qname)
            if members:
                self.current_results = members
                self.update_results(members)

    def action_goto_type(self) -> None:
        """Go to a type mentioned in the current symbol's signature."""
        detail = self.query_one(SymbolDetail)
        if not detail.current_symbol:
            return

        # Extract types from signature and return annotation
        types_to_check = []
        if detail.current_symbol.signature:
            types_to_check.extend(extract_type_names(detail.current_symbol.signature))
        if detail.current_symbol.return_annotation:
            types_to_check.extend(extract_type_names(detail.current_symbol.return_annotation))
        if detail.current_symbol.bases:
            types_to_check.extend(detail.current_symbol.bases)

        if types_to_check:
            # Search for the first type
            first_type = types_to_check[0]
            self.query_one("#search-input", Input).value = first_type
            self.do_search(first_type)
            self.update_status(f"Searching for type: {first_type}")

    def action_edit(self) -> None:
        """Open the current symbol in $EDITOR."""
        symbol = self.get_highlighted_symbol()
        if not symbol:
            detail = self.query_one(SymbolDetail)
            symbol = detail.current_symbol

        if symbol:
            self.open_in_editor(symbol.file_path, symbol.line_no)

    def action_focus_search(self) -> None:
        """Focus the search input."""
        self.query_one("#search-input", Input).focus()

    def action_focus_list(self) -> None:
        """Focus the results list."""
        self.query_one("#results-list", ListView).focus()

    def action_toggle_focus(self) -> None:
        """Toggle focus between list and detail panel."""
        detail = self.query_one(SymbolDetail)
        results = self.query_one("#results-list", ListView)
        if detail.has_focus:
            results.focus()
        else:
            detail.focus()

    def action_cursor_down(self) -> None:
        """Move cursor down in list (vim j)."""
        results_list = self.query_one("#results-list", ListView)
        results_list.action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move cursor up in list (vim k)."""
        results_list = self.query_one("#results-list", ListView)
        results_list.action_cursor_up()


def run_tui(query: str = "") -> None:
    """Run the TUI application."""
    app = PyDocsApp(initial_query=query)
    app.run()
