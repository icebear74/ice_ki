"""
Terminal UI Utilities

Helper functions for terminal-based UI display.
Ported from original train.py to maintain feature parity.
"""

import re
import sys

# ANSI Color Codes
C_GREEN   = "\033[92m"
C_GRAY    = "\033[90m"
C_RESET   = "\033[0m"
C_BOLD    = "\033[1m"
C_CYAN    = "\033[96m"
C_RED     = "\033[91m"
C_YELLOW  = "\033[93m"
C_MAGENTA = "\033[95m"
C_BORDER  = "\033[36m"   # Regular cyan — visible on both dark and light terminals

# ANSI Control Codes
ANSI_HOME       = "\033[H"
ANSI_CLEAR      = "\033[2J"
ANSI_HIDE_CURSOR = "\033[?25l"
ANSI_SHOW_CURSOR = "\033[?25h"

# ANSI Escape Sequence Pattern (for stripping colours from text)
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

# Display Mode Names (kept for reference)
DISPLAY_MODE_NAMES = [
    "2-Column Detailed (Backward | Forward)",
    "Grouped by Trunk → Sorted by Activity",
    "Flat List → Sorted by Position",
    "Flat List → Sorted by Activity"
]


# ── Utility ────────────────────────────────────────────────────────────────────

def get_visible_len(text):
    """Return the visible length of *text* (ANSI escape codes excluded)."""
    return len(ANSI_ESCAPE.sub('', text))


def _truncate(content, max_cw):
    """Truncate *content* to *max_cw* visible characters, appending '…' if cut."""
    if get_visible_len(content) <= max_cw:
        return content
    truncated   = ""
    visible     = 0
    in_ansi     = False
    ansi_buf    = ""
    for ch in content:
        if ch == '\033':
            in_ansi  = True
            ansi_buf = ch
        elif in_ansi:
            ansi_buf += ch
            if ch in 'mHJKSTfABCDsu':
                truncated += ansi_buf
                in_ansi  = False
                ansi_buf = ""
        else:
            if visible < max_cw - 1:
                truncated += ch
                visible   += 1
            else:
                truncated += "…"
                break
    return truncated + C_RESET


# ── String-returning border helpers (no print side-effects) ───────────────────

def _line(content, ui_w):
    """Return one bordered line string (no trailing newline)."""
    max_cw   = ui_w - 4
    content  = _truncate(content, max_cw)
    vis      = get_visible_len(content)
    padding  = max(0, max_cw - vis)
    return f" {C_BORDER}║{C_RESET} {content}{' ' * padding} {C_BORDER}║{C_RESET}"


def _two_cols(left, right, ui_w):
    """Return a two-column bordered line string (no trailing newline)."""
    col_w   = (ui_w - 7) // 2
    lv      = get_visible_len(left)
    rv      = get_visible_len(right) if right else 0
    lpad    = max(0, col_w - lv)
    rpad    = max(0, col_w - rv)
    left_s  = f"{left}{' ' * lpad}"
    right_s = f"{right}{' ' * rpad}" if right else (' ' * col_w)
    return f" {C_BORDER}║{C_RESET} {left_s} {C_BORDER}│{C_RESET} {right_s} {C_BORDER}║{C_RESET}"


def _sep(ui_w, style='single'):
    """Return a horizontal separator string (no trailing newline)."""
    if style == 'double':
        return f" {C_BORDER}╠{'═' * (ui_w - 2)}╣{C_RESET}"
    elif style == 'thin':
        return f" {C_BORDER}╟{'·' * (ui_w - 2)}╢{C_RESET}"
    else:
        return f" {C_BORDER}╟{'─' * (ui_w - 2)}╢{C_RESET}"


def _hdr(ui_w):
    """Return the top border string (no trailing newline)."""
    return f" {C_BORDER}╔{'═' * (ui_w - 2)}╗{C_RESET}"


def _ftr(ui_w):
    """Return the bottom border string (no trailing newline)."""
    return f" {C_BORDER}╚{'═' * (ui_w - 2)}╝{C_RESET}"


# ── Print wrappers (kept for backward compatibility) ──────────────────────────

def print_line(content, ui_w):
    sys.stdout.write(_line(content, ui_w) + "\n")


def print_two_columns(left_content, right_content, ui_w):
    sys.stdout.write(_two_cols(left_content, right_content, ui_w) + "\n")


def print_separator(ui_w, style='single'):
    sys.stdout.write(_sep(ui_w, style) + "\n")


def print_header(ui_w):
    sys.stdout.write(_hdr(ui_w) + "\n")


def print_footer(ui_w):
    sys.stdout.write(_ftr(ui_w) + "\n")


# ── Progress bars ─────────────────────────────────────────────────────────────

def make_bar(percent, width):
    """Green ASCII progress bar (0-100 %)."""
    width  = max(5, width)
    filled = max(0, min(width, int((percent / 100.0) * width)))
    return f"{C_GREEN}{'█' * filled}{C_GRAY}{'░' * (width - filled)}{C_RESET}"


def make_bar_fusion(percent, width):
    """Cyan ASCII progress bar for fusion layers."""
    width  = max(5, width)
    filled = max(0, min(width, int((percent / 100.0) * width)))
    return f"{C_CYAN}{'█' * filled}{C_GRAY}{'░' * (width - filled)}{C_RESET}"


def make_bar_final_fusion(percent, width):
    """Yellow ASCII progress bar for the final fusion layer."""
    width  = max(5, width)
    filled = max(0, min(width, int((percent / 100.0) * width)))
    return f"{C_YELLOW}{'█' * filled}{C_GRAY}{'░' * (width - filled)}{C_RESET}"


def make_adamw_magic_eye(momentum, width=20):
    """
    AdamW 'Magic Eye' — tube radio style momentum indicator.

    - High SNR (>0.55): gradients consistent → push right  [  |====>  ]
    - Low SNR  (<0.45): gradients noisy     → brake left   [  <====|  ]
    - Medium           : balanced           [   <=|=>   ]
    """
    width      = max(10, width)
    normalized = min(1.0, max(0.0, momentum))
    center     = width // 2
    needle_pos = int(normalized * (width - 1))

    if normalized > 0.55:
        bar = ['·'] * width
        for i in range(center, min(needle_pos, width)):
            bar[i] = '='
        bar[center] = '|'
        if needle_pos < width:
            bar[needle_pos] = '>'
        return f"[{C_GREEN}{''.join(bar)}{C_RESET}]"

    elif normalized < 0.45:
        bar = ['·'] * width
        for i in range(max(0, needle_pos + 1), center + 1):
            bar[i] = '='
        bar[center] = '|'
        if needle_pos >= 0:
            bar[needle_pos] = '<'
        return f"[{C_YELLOW}{''.join(bar)}{C_RESET}]"

    else:
        bar = ['·'] * width
        bar[center - 1] = '='
        bar[center]     = '|'
        bar[center + 1] = '='
        if needle_pos < center:
            bar[needle_pos] = '<'
        elif needle_pos > center:
            bar[needle_pos] = '>'
        return f"[{C_CYAN}{''.join(bar)}{C_RESET}]"


def make_peak_activity_bar(peak_value, width=60):
    """
    Gradient bar visualising peak layer activity on a 0.0–2.0+ scale.

    Zones: green (0–0.5) → cyan (0.5–1.0) → yellow (1.0–1.5) → red (1.5–2.0+).
    Includes a ▼ position indicator and a scale line with aligned labels.
    """
    width    = max(20, width)
    position = min(peak_value / 2.0, 1.0)
    bar_pos  = min(int(position * (width - 1)), width - 1)

    if peak_value < 0.5:
        color = C_GREEN;              label = "Normal"
    elif peak_value < 1.0:
        color = C_CYAN;               label = "Moderate"
    elif peak_value < 1.5:
        color = C_YELLOW;             label = "High"
    elif peak_value < 2.0:
        color = "\033[38;5;208m";     label = "Very High"
    else:
        color = C_RED;                label = "EXTREME"

    # Gradient bar
    bar = ""
    for i in range(width):
        frac = i / width
        if frac < 0.25:
            bar += C_GREEN  + "█" + C_RESET
        elif frac < 0.50:
            bar += C_CYAN   + "█" + C_RESET
        elif frac < 0.75:
            bar += C_YELLOW + "█" + C_RESET
        else:
            bar += C_RED    + "█" + C_RESET

    # Position indicator
    indicator_line = " " * bar_pos + f"{color}▼{C_RESET}"

    # Scale labels placed at their exact proportional positions
    scale = [' '] * (width + 6)

    def _put(lbl, frac):
        pos = int(frac * (width - 1))
        for j, ch in enumerate(lbl):
            idx = pos + j
            if 0 <= idx < len(scale):
                scale[idx] = ch

    _put("0.0", 0.0)
    _put("0.5", 0.25)
    _put("1.0", 0.50)
    _put("1.5", 0.75)
    # "2.0+" sits at the right edge — shift left by label length so it doesn't overflow
    _put("2.0+", max(0.0, 1.0 - 4.0 / width))

    scale_line = ''.join(scale).rstrip()
    return f"{bar}\n{indicator_line}\n{scale_line}  {color}{label}{C_RESET}"


def make_size_bar(trained, target, width):
    """Progress bar for size-distribution tracking."""
    width = max(5, width)
    pct   = min(100.0, (trained / target) * 100.0) if target > 0 else 0.0
    filled = max(0, min(width, int((pct / 100.0) * width)))
    color  = C_GREEN if pct >= 90.0 else C_CYAN if pct >= 50.0 else C_YELLOW
    return f"{color}{'█' * filled}{C_GRAY}{'░' * (width - filled)}{C_RESET}"


# ── Misc helpers ──────────────────────────────────────────────────────────────

def format_time(seconds):
    """Format *seconds* into a human-readable string like '2d 5h 30m'."""
    if seconds < 0:
        return "N/A"
    days    = int(seconds // 86400)
    hours   = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{days}d {hours}h {minutes}m" if days > 0 else f"{hours}h {minutes}m"


def clear_screen():
    print(ANSI_CLEAR, end='')


def move_cursor_home():
    print(ANSI_HOME, end='')


def hide_cursor():
    print(ANSI_HIDE_CURSOR, end='')


def show_cursor():
    print(ANSI_SHOW_CURSOR, end='')
