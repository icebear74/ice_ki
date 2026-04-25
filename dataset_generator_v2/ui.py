#!/usr/bin/env python3
"""
Rich + questionary UI helpers for ice_ki Video Manager.

Single import point for all TUI elements:  console, ask_*, print_*, make_table,
Choice, Separator.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich import box
import questionary
from questionary import Choice, Separator, Style as _QStyle
from typing import Any, Callable, List, Optional

__all__ = [
    "console",
    "Choice", "Separator",
    "ask_text", "ask_int", "ask_confirm", "ask_select", "ask_checkbox",
    "print_success", "print_error", "print_warn", "print_info",
    "print_banner", "print_rule", "make_table",
]

console = Console()

# ── questionary style — ice-blue / slate theme ────────────────────────────────
_QS = _QStyle([
    ("qmark",       "fg:#5bc0de bold"),   # leading question mark
    ("question",    "bold"),
    ("answer",      "fg:#5bc0de bold"),   # confirmed answer
    ("pointer",     "fg:#5bc0de bold"),   # ❯ cursor
    ("highlighted", "fg:#ffffff bold"),   # currently-highlighted option
    ("selected",    "fg:#5bc0de"),        # ticked checkbox item
    ("separator",   "fg:#555555"),
    ("instruction", "fg:#888888"),
    ("text",        ""),
    ("disabled",    "fg:#666666 italic"),
])

_NAV  = "(↑↓ navigate  Enter select  Ctrl+C cancel)"
_TICK = "(↑↓ navigate  Space toggle  Enter confirm  Ctrl+C cancel)"


# ── Input helpers ──────────────────────────────────────────────────────────────

def ask_text(
    prompt: str,
    default: str = "",
    validate: Optional[Callable] = None,
) -> Optional[str]:
    """Styled text input.  Returns None if cancelled (Ctrl+C)."""
    kwargs: dict = {"default": default, "style": _QS}
    if validate:
        kwargs["validate"] = validate
    return questionary.text(prompt, **kwargs).ask()


def ask_int(
    prompt: str,
    default: Optional[int] = None,
    min_val: int = 0,
) -> Optional[int]:
    """Integer input with inline validation.  Returns None if cancelled."""
    def _v(val: str) -> Any:
        try:
            n = int(val)
        except ValueError:
            return "Please enter a valid integer"
        if n < min_val:
            return f"Value must be ≥ {min_val}"
        return True

    result = questionary.text(
        prompt,
        default="" if default is None else str(default),
        validate=_v,
        style=_QS,
    ).ask()
    return None if result is None else int(result)


def ask_confirm(prompt: str, default: bool = True) -> Optional[bool]:
    """Yes/no prompt.  Returns None if cancelled."""
    return questionary.confirm(prompt, default=default, style=_QS).ask()


def ask_select(prompt: str, choices: List, instruction: str = "") -> Optional[Any]:
    """Arrow-key single-select.  Returns the chosen value or None if cancelled."""
    return questionary.select(
        prompt,
        choices=choices,
        instruction=instruction or _NAV,
        style=_QS,
    ).ask()


def ask_checkbox(
    prompt: str,
    choices: List,
    instruction: str = "",
) -> Optional[List]:
    """Arrow-key multi-select checkboxes.  Returns list of values or None."""
    return questionary.checkbox(
        prompt,
        choices=choices,
        instruction=instruction or _TICK,
        style=_QS,
    ).ask()


# ── Display helpers ────────────────────────────────────────────────────────────

def print_success(msg: str) -> None:
    console.print(f"  [bold green]✓[/]  {msg}")


def print_error(msg: str) -> None:
    console.print(f"  [bold red]✗[/]  {msg}")


def print_warn(msg: str) -> None:
    console.print(f"  [yellow]⚠[/]  {msg}")


def print_info(msg: str) -> None:
    console.print(f"  [cyan]ℹ[/]  {msg}")


def print_banner(
    videos: int = 0,
    categories: int = 0,
    unsaved_cfg: bool = False,
    unsaved_tpl: bool = False,
) -> None:
    """Print the app header panel with live stats."""
    flags: List[str] = []
    if unsaved_cfg:
        flags.append("[yellow bold]config unsaved ⚡[/]")
    if unsaved_tpl:
        flags.append("[yellow bold]templates unsaved ⚡[/]")
    flag_str = "  " + "  ".join(flags) if flags else ""
    subtitle = (
        f"[dim]videos: [bold]{videos}[/]   categories: [bold]{categories}[/][/]{flag_str}"
    )
    console.print(
        Panel(
            f"[bold cyan]❄  ice_ki Video Manager[/]  [dim]v2 · dataset generator[/]\n{subtitle}",
            box=box.ROUNDED,
            border_style="cyan",
            padding=(0, 2),
        )
    )


def print_rule(title: str = "") -> None:
    console.print(Rule(title, style="dim cyan"))


def make_table(*headers: str, box_style=box.SIMPLE_HEAVY) -> Table:
    """Return a pre-styled Rich Table ready for rows."""
    t = Table(
        box=box_style,
        border_style="dim cyan",
        header_style="bold cyan",
        show_edge=True,
        padding=(0, 1),
    )
    for h in headers:
        t.add_column(h)
    return t
