#!/usr/bin/env python3
"""
Curses TUI toolkit for ice_ki Video Manager.

Provides bordered menus, input dialogs, confirm dialogs and scrollable
detail views — all as floating popup windows on a styled desktop background.
Works perfectly over SSH.
"""

import curses
from typing import Any, List, Optional, Tuple

# ── Colour pair indices ────────────────────────────────────────────────────────
_P_NORMAL  = 1   # default text (white on dark)
_P_TITLE   = 2   # box / dialog title (bright yellow)
_P_SELECT  = 3   # highlighted menu row (black on cyan)
_P_BORDER  = 4   # box borders (bright cyan)
_P_STATUS  = 5   # status / footer bar (black on cyan)
_P_ERROR   = 6   # error text (bright red)
_P_SUCCESS = 7   # success text (bright green)
_P_INPUT   = 8   # active input field (black on white)
_P_DIM     = 9   # secondary / hint text
_P_CHECK   = 10  # checked checkbox item (bright green)
_P_HEADER  = 11  # app header bar (black on cyan)


def setup(stdscr) -> None:
    """Initialise colours, cursor and keypad for the full-screen TUI."""
    curses.curs_set(0)
    stdscr.keypad(True)

    if not curses.has_colors():
        return

    curses.start_color()
    try:
        curses.use_default_colors()
        bg = -1
    except Exception:
        bg = curses.COLOR_BLACK

    curses.init_pair(_P_NORMAL,  curses.COLOR_WHITE,   bg)
    curses.init_pair(_P_TITLE,   curses.COLOR_YELLOW,  bg)
    curses.init_pair(_P_SELECT,  curses.COLOR_BLACK,   curses.COLOR_CYAN)
    curses.init_pair(_P_BORDER,  curses.COLOR_CYAN,    bg)
    curses.init_pair(_P_STATUS,  curses.COLOR_BLACK,   curses.COLOR_CYAN)
    curses.init_pair(_P_ERROR,   curses.COLOR_RED,     bg)
    curses.init_pair(_P_SUCCESS, curses.COLOR_GREEN,   bg)
    curses.init_pair(_P_INPUT,   curses.COLOR_BLACK,   curses.COLOR_WHITE)
    curses.init_pair(_P_DIM,     curses.COLOR_WHITE,   bg)
    curses.init_pair(_P_CHECK,   curses.COLOR_GREEN,   bg)
    curses.init_pair(_P_HEADER,  curses.COLOR_BLACK,   curses.COLOR_CYAN)


# ── Low-level drawing helpers ──────────────────────────────────────────────────

def _a(pair: int, bold: bool = False, dim: bool = False) -> int:
    attr = curses.color_pair(pair)
    if bold:
        attr |= curses.A_BOLD
    if dim:
        attr |= curses.A_DIM
    return attr


def _safe(win, y: int, x: int, text: str, attr: int = 0) -> None:
    """addstr that never raises on out-of-bounds writes."""
    try:
        h, w = win.getmaxyx()
        if y < 0 or y >= h or x < 0 or x >= w:
            return
        avail = w - x
        if avail <= 0:
            return
        win.addstr(y, x, text[:avail], attr)
    except curses.error:
        pass


def _fill(win, attr: int) -> None:
    h, w = win.getmaxyx()
    for y in range(h):
        try:
            win.addstr(y, 0, " " * w, attr)
        except curses.error:
            pass


def _border(win, title: str = "", fg: int = _P_BORDER) -> None:
    """Draw a single-line box border with an optional centred title."""
    h, w = win.getmaxyx()
    ba = _a(fg, bold=True)
    try:
        win.attron(ba)
        win.border(
            curses.ACS_VLINE, curses.ACS_VLINE,
            curses.ACS_HLINE, curses.ACS_HLINE,
            curses.ACS_ULCORNER, curses.ACS_URCORNER,
            curses.ACS_LLCORNER, curses.ACS_LRCORNER,
        )
        win.attroff(ba)
    except curses.error:
        pass
    if title:
        label = f"  {title}  "
        tx = max(1, (w - len(label)) // 2)
        _safe(win, 0, tx, label, _a(_P_TITLE, bold=True))


def _popup(stdscr, height: int, width: int):
    """Create a centred floating window."""
    sh, sw = stdscr.getmaxyx()
    height = min(height, sh)
    width  = min(width,  sw)
    y = max(0, (sh - height) // 2)
    x = max(0, (sw - width)  // 2)
    win = curses.newwin(height, width, y, x)
    win.keypad(True)
    _fill(win, _a(_P_NORMAL))
    return win


# ── Background desktop ─────────────────────────────────────────────────────────

def draw_background(
    stdscr,
    app_name:  str = "❄  ice_ki Video Manager  v2",
    stats:     str = "",
    status:    str = "",
    is_error:  bool = False,
) -> None:
    """
    Redraw the full desktop:  header bar at top, status bar at bottom,
    empty work area in between.
    """
    stdscr.erase()
    sh, sw = stdscr.getmaxyx()

    # header
    hdr = f"  {app_name}  "
    if stats:
        pad = sw - len(hdr) - len(stats) - 2
        if pad > 0:
            hdr = hdr + " " * pad + stats + "  "
    _safe(stdscr, 0, 0, hdr.ljust(sw), _a(_P_HEADER, bold=True))

    # status bar
    if status:
        status_attr = _a(_P_ERROR, bold=True) if is_error else _a(_P_STATUS)
        _safe(stdscr, sh - 1, 0, ("  " + status).ljust(sw), status_attr)
    else:
        _safe(stdscr, sh - 1, 0, "  ↑↓/k/j=navigate   Enter=select   Esc/q=cancel".ljust(sw), _a(_P_STATUS))

    stdscr.refresh()


# ── menu_box ───────────────────────────────────────────────────────────────────

def menu_box(
    stdscr,
    title: str,
    items: List[Tuple[str, Any]],
    selected: int = 0,
) -> Optional[Any]:
    """
    Show a scrollable bordered menu.

    items  : List of (label, value).  A separator is ("───...", None).
    Returns: value of selected item, or None if cancelled (Esc / q).
    """
    if not items:
        return None

    labels     = [lbl for lbl, _ in items]
    sep_set    = {i for i, (lbl, _) in enumerate(items) if lbl.startswith("─") or lbl == ""}
    real_items = [i for i in range(len(items)) if i not in sep_set]
    if not real_items:
        return None

    sh, sw  = stdscr.getmaxyx()
    max_lbl = max((len(lbl) for lbl in labels if not lbl.startswith("─")), default=20)
    width   = min(max(max_lbl + 6, len(title) + 8, 42), sw - 4)
    inner_w = width - 2

    visible = min(len(items), sh - 8)
    height  = visible + 4   # top-border + items + hint-line + bottom-border

    win    = _popup(stdscr, height, width)
    cur    = max(0, min(selected, len(items) - 1))
    # skip to first real item
    while cur in sep_set and cur < len(items) - 1:
        cur += 1
    offset = 0

    hint = " ↑↓ navigate   Enter select   Esc cancel "

    while True:
        _fill(win, _a(_P_NORMAL))
        _border(win, title)

        # item rows
        for row_i in range(visible):
            idx = row_i + offset
            if idx >= len(items):
                break
            lbl = labels[idx]
            y   = row_i + 1

            if idx in sep_set:
                sep_line = ("─" * (inner_w - 2))
                _safe(win, y, 1, " " + sep_line + " ", _a(_P_BORDER))
                continue

            if idx == cur:
                _safe(win, y, 1, (" " + lbl + " ").ljust(inner_w), _a(_P_SELECT, bold=True))
            else:
                _safe(win, y, 2, lbl[:inner_w - 2], _a(_P_NORMAL))

        # hint at bottom
        _safe(win, height - 2, max(1, (width - len(hint)) // 2), hint, _a(_P_DIM, dim=True))

        # scroll indicator
        if len(real_items) > visible:
            frac = real_items.index(cur) / max(len(real_items) - 1, 1)
            bar_y = 1 + int(frac * (visible - 1))
            _safe(win, bar_y, width - 1, "█", _a(_P_BORDER))

        win.refresh()
        key = win.getch()

        def _prev():
            nonlocal cur
            c = cur
            while c > 0:
                c -= 1
                if c not in sep_set:
                    cur = c
                    return

        def _next():
            nonlocal cur
            c = cur
            while c < len(items) - 1:
                c += 1
                if c not in sep_set:
                    cur = c
                    return

        if key in (curses.KEY_UP, ord('k')):
            _prev()
        elif key in (curses.KEY_DOWN, ord('j')):
            _next()
        elif key == curses.KEY_PPAGE:
            for _ in range(visible):
                _prev()
        elif key == curses.KEY_NPAGE:
            for _ in range(visible):
                _next()
        elif key == curses.KEY_HOME:
            cur = real_items[0]
        elif key == curses.KEY_END:
            cur = real_items[-1]
        elif key in (curses.KEY_ENTER, ord('\n'), ord('\r')):
            if cur not in sep_set:
                val = items[cur][1]
                del win
                return val
        elif key in (27, ord('q'), curses.KEY_F10):
            del win
            return None

        # scroll offset
        if cur < offset:
            offset = cur
        elif cur >= offset + visible:
            offset = cur - visible + 1


# ── confirm_box ────────────────────────────────────────────────────────────────

def confirm_box(
    stdscr,
    message: str,
    title: str = "Confirm",
    default: bool = True,
) -> bool:
    """
    Bordered Yes / No dialog.  Returns True for Yes, False otherwise.
    """
    lines  = message.split("\n")
    sh, sw = stdscr.getmaxyx()
    width  = min(max(max(len(l) for l in lines) + 6, len(title) + 8, 44), sw - 4)
    height = len(lines) + 6   # border(2) + gap(1) + lines + gap(1) + buttons(1) + gap(1)

    win = _popup(stdscr, height, width)
    sel = 0 if default else 1   # 0=Yes 1=No

    yes_lbl = "  Yes  "
    no_lbl  = "   No  "
    btn_gap = 2
    btns_w  = len(yes_lbl) + len(no_lbl) + btn_gap
    yes_x   = (width - btns_w) // 2
    no_x    = yes_x + len(yes_lbl) + btn_gap

    while True:
        _fill(win, _a(_P_NORMAL))
        _border(win, title)

        for i, line in enumerate(lines):
            _safe(win, i + 2, 2, line[:width - 4], _a(_P_NORMAL))

        ya = _a(_P_SELECT, bold=True) if sel == 0 else _a(_P_NORMAL)
        na = _a(_P_SELECT, bold=True) if sel == 1 else _a(_P_NORMAL)
        _safe(win, height - 2, yes_x, yes_lbl, ya)
        _safe(win, height - 2, no_x,  no_lbl,  na)

        win.refresh()
        key = win.getch()

        if key in (curses.KEY_LEFT, curses.KEY_RIGHT, ord('\t')):
            sel = 1 - sel
        elif key in (curses.KEY_ENTER, ord('\n'), ord('\r')):
            del win
            return sel == 0
        elif key in (ord('y'), ord('Y')):
            del win
            return True
        elif key in (ord('n'), ord('N'), 27):
            del win
            return False


# ── message_box ───────────────────────────────────────────────────────────────

def message_box(
    stdscr,
    lines: List[str],
    title: str = "",
    ok_text: str = " OK / any key ",
) -> None:
    """
    Scrollable bordered info / read-only panel.
    Press any navigation key to scroll, Enter/Space/Esc to close.
    """
    if not lines:
        lines = ["(empty)"]

    sh, sw = stdscr.getmaxyx()
    max_line = max((len(l) for l in lines), default=0)
    width    = min(max(max_line + 4, len(title) + 8, 46), sw - 4)
    height   = min(len(lines) + 5, sh - 4)
    inner_h  = height - 4   # top-border + title-gap + inner + btn + bottom-border

    win    = _popup(stdscr, height, width)
    offset = 0

    while True:
        _fill(win, _a(_P_NORMAL))
        _border(win, title)

        for i in range(inner_h):
            idx = i + offset
            if idx >= len(lines):
                break
            line = lines[idx]
            # simple colour markup: lines starting with special prefixes
            if line.startswith("✓") or line.startswith("  ✓"):
                attr = _a(_P_SUCCESS, bold=True)
            elif line.startswith("✗") or line.startswith("  ✗") or line.startswith("ERROR"):
                attr = _a(_P_ERROR, bold=True)
            elif line.startswith("──") or line.startswith("═"):
                attr = _a(_P_BORDER, bold=True)
            elif line.startswith("▸") or line.startswith("  ▸") or line.startswith("["):
                attr = _a(_P_TITLE, bold=True)
            else:
                attr = _a(_P_NORMAL)
            _safe(win, i + 2, 2, line[:width - 4], attr)

        # scroll counter
        total = len(lines)
        end   = min(offset + inner_h, total)
        if total > inner_h:
            scrollinfo = f" {offset+1}–{end}/{total} ↑↓ scroll "
        else:
            scrollinfo = ""
        _safe(win, height - 2, 2, scrollinfo + ok_text, _a(_P_DIM, dim=True))

        win.refresh()
        key = win.getch()

        if key in (curses.KEY_UP, ord('k')):
            offset = max(0, offset - 1)
        elif key in (curses.KEY_DOWN, ord('j')):
            offset = min(max(0, total - inner_h), offset + 1)
        elif key == curses.KEY_PPAGE:
            offset = max(0, offset - inner_h)
        elif key == curses.KEY_NPAGE:
            offset = min(max(0, total - inner_h), offset + inner_h)
        elif key == curses.KEY_HOME:
            offset = 0
        elif key == curses.KEY_END:
            offset = max(0, total - inner_h)
        else:
            break   # any other key closes

    del win


# ── input_box ─────────────────────────────────────────────────────────────────

def input_box(
    stdscr,
    prompt: str,
    default: str = "",
    title: str = "Input",
    max_len: int = 200,
) -> Optional[str]:
    """
    Single-line text input dialog.
    Returns the entered string, or None if cancelled (Esc).
    """
    sh, sw = stdscr.getmaxyx()
    width  = min(max(len(prompt) + 8, len(title) + 8, 50), sw - 4)
    height = 7

    win = _popup(stdscr, height, width)
    _fill(win, _a(_P_NORMAL))
    _border(win, title)
    _safe(win, 2, 2, prompt[:width - 4], _a(_P_NORMAL, bold=True))
    _safe(win, 5, 2, " Esc=cancel   Enter=confirm ", _a(_P_DIM, dim=True))

    inner_w = width - 4
    buf     = list(default)
    pos     = len(buf)

    try:
        curses.curs_set(1)
    except Exception:
        pass

    while True:
        # draw input field
        ds      = max(0, pos - inner_w + 1)
        display = "".join(buf)[ds:ds + inner_w]
        _safe(win, 3, 2, display.ljust(inner_w), _a(_P_INPUT))
        cx = 2 + min(pos - ds, inner_w - 1)
        try:
            win.move(3, cx)
        except curses.error:
            pass
        win.refresh()

        key = win.getch()

        if key in (curses.KEY_ENTER, ord('\n'), ord('\r')):
            break
        elif key == 27:
            try:
                curses.curs_set(0)
            except Exception:
                pass
            del win
            return None
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            if pos > 0:
                buf.pop(pos - 1)
                pos -= 1
        elif key == curses.KEY_DC:
            if pos < len(buf):
                buf.pop(pos)
        elif key == curses.KEY_LEFT:
            pos = max(0, pos - 1)
        elif key == curses.KEY_RIGHT:
            pos = min(len(buf), pos + 1)
        elif key == curses.KEY_HOME:
            pos = 0
        elif key == curses.KEY_END:
            pos = len(buf)
        elif 32 <= key <= 126 and len(buf) < max_len:
            buf.insert(pos, chr(key))
            pos += 1

    try:
        curses.curs_set(0)
    except Exception:
        pass
    del win
    return "".join(buf)


# ── int_box ───────────────────────────────────────────────────────────────────

def int_box(
    stdscr,
    prompt: str,
    default: int = 0,
    min_val: int = 0,
    title: str = "Input",
) -> Optional[int]:
    """Integer input dialog with inline validation."""
    while True:
        raw = input_box(stdscr, prompt, str(default), title)
        if raw is None:
            return None
        raw = raw.strip()
        if not raw:
            return default
        try:
            val = int(raw)
        except ValueError:
            message_box(stdscr, ["Please enter a valid whole number."], "Error")
            continue
        if val < min_val:
            message_box(stdscr, [f"Value must be ≥ {min_val}."], "Error")
            continue
        return val


# ── checkbox_box ──────────────────────────────────────────────────────────────

def checkbox_box(
    stdscr,
    title: str,
    item_labels: List[str],
    preselected: Optional[List[int]] = None,
) -> Optional[List[int]]:
    """
    Multi-select checkbox list.
    Returns sorted list of selected indices, or None if cancelled.
    """
    if not item_labels:
        return []

    selected = set(preselected or [])
    sh, sw   = stdscr.getmaxyx()
    max_lbl  = max(len(s) for s in item_labels)
    width    = min(max(max_lbl + 8, len(title) + 8, 46), sw - 4)
    inner_w  = width - 2
    visible  = min(len(item_labels), sh - 8)
    height   = visible + 4

    win    = _popup(stdscr, height, width)
    cur    = 0
    offset = 0

    hint = " Space=toggle  a=all  n=none  Enter=confirm  Esc=cancel "

    while True:
        _fill(win, _a(_P_NORMAL))
        _border(win, title)

        for i in range(visible):
            idx = i + offset
            if idx >= len(item_labels):
                break
            lbl    = item_labels[idx]
            marker = "[✓]" if idx in selected else "[ ]"
            line   = f" {marker} {lbl}"
            y      = i + 1
            if idx == cur:
                attr = _a(_P_SELECT, bold=True)
            elif idx in selected:
                attr = _a(_P_CHECK, bold=True)
            else:
                attr = _a(_P_NORMAL)
            _safe(win, y, 1, line[:inner_w - 1].ljust(inner_w - 1), attr)

        # footer with count + hint
        count_str = f" {len(selected)}/{len(item_labels)} "
        _safe(win, height - 2, 1, count_str, _a(_P_TITLE, bold=True))
        _safe(win, height - 2, 1 + len(count_str), hint[:inner_w - len(count_str) - 1], _a(_P_DIM, dim=True))

        if cur < offset:
            offset = cur
        elif cur >= offset + visible:
            offset = cur - visible + 1

        win.refresh()
        key = win.getch()

        if key in (curses.KEY_UP, ord('k')):
            cur = max(0, cur - 1)
        elif key in (curses.KEY_DOWN, ord('j')):
            cur = min(len(item_labels) - 1, cur + 1)
        elif key == curses.KEY_PPAGE:
            cur = max(0, cur - visible)
        elif key == curses.KEY_NPAGE:
            cur = min(len(item_labels) - 1, cur + visible)
        elif key == ord(' '):
            if cur in selected:
                selected.discard(cur)
            else:
                selected.add(cur)
        elif key == ord('a'):
            selected = set(range(len(item_labels)))
        elif key == ord('n'):
            selected.clear()
        elif key in (curses.KEY_ENTER, ord('\n'), ord('\r')):
            del win
            return sorted(selected)
        elif key in (27, ord('q')):
            del win
            return None

    return None


# ── text_table ────────────────────────────────────────────────────────────────

def text_table(
    headers: List[str],
    rows:    List[List[str]],
    max_col: int = 35,
) -> List[str]:
    """
    Format a 2-D table as a list of text lines (for use with message_box).
    Columns are automatically sized.
    """
    if not rows and not headers:
        return ["(empty)"]

    all_rows = [headers] + rows
    n_cols   = max(len(r) for r in all_rows)
    widths   = [0] * n_cols
    for row in all_rows:
        for i, cell in enumerate(row):
            widths[i] = min(max(widths[i], len(str(cell))), max_col)

    sep  = "─┼─".join("─" * w for w in widths)
    sep  = "──" + sep + "──"

    def fmt_row(row):
        cells = [str(row[i]).ljust(widths[i])[:widths[i]] if i < len(row) else " " * widths[i]
                 for i in range(n_cols)]
        return "  " + "  │  ".join(cells) + "  "

    lines = []
    lines.append(sep)
    lines.append(fmt_row(headers))
    lines.append(sep)
    for row in rows:
        lines.append(fmt_row(row))
    lines.append(sep)
    return lines
