"""
Terminal UI Utilities for Dataset Generator

ANSI color codes and helper functions for terminal display.
Similar to vsr_plusplus_NEU/utils/ui_terminal.py
"""

import atexit
import re
import sys

# ANSI Color Codes
C_GREEN = "\033[92m"
C_GRAY = "\033[90m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_CYAN = "\033[96m"
C_RED = "\033[91m"
C_YELLOW = "\033[93m"
C_MAGENTA = "\033[95m"
C_BLUE = "\033[94m"
C_WHITE = "\033[97m"

# ANSI Control Codes
ANSI_HOME = "\033[H"
ANSI_CLEAR = "\033[2J"
ANSI_HIDE_CURSOR = "\033[?25l"
ANSI_SHOW_CURSOR = "\033[?25h"

# ANSI Escape Sequence Pattern (for stripping colors from text)
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

# Track whether we have hidden the cursor so we register atexit only once
# and so show_cursor() is always safe to call even if called multiple times.
_cursor_hidden = False


def hide_cursor():
    """Hide terminal cursor and register an atexit handler to restore it."""
    global _cursor_hidden
    sys.stdout.write(ANSI_HIDE_CURSOR)
    sys.stdout.flush()
    if not _cursor_hidden:
        _cursor_hidden = True
        atexit.register(show_cursor)  # guaranteed restore on any normal exit


def show_cursor():
    """Show terminal cursor."""
    global _cursor_hidden
    sys.stdout.write(ANSI_SHOW_CURSOR)
    sys.stdout.flush()
    _cursor_hidden = False


def clear_screen():
    """Clear screen and move cursor to home"""
    sys.stdout.write(ANSI_CLEAR + ANSI_HOME)
    sys.stdout.flush()


def get_visible_len(text):
    """
    Get visible length of text (excluding ANSI escape codes)
    
    Args:
        text: String potentially containing ANSI codes
    
    Returns:
        int: Visible length of text
    """
    return len(ANSI_ESCAPE.sub('', text))


def make_bar(percent, width, color=C_GREEN):
    """
    Create an ASCII progress bar
    
    Args:
        percent: Percentage (0-100)
        width: Width of the bar in characters
        color: ANSI color code for filled portion
    
    Returns:
        str: Formatted progress bar with ANSI colors
    """
    width = max(5, width)
    filled = max(0, min(width, int((percent / 100.0) * width)))
    return f"{color}{'█' * filled}{C_GRAY}{'░' * (width - filled)}{C_RESET}"


def format_time(seconds):
    """
    Format seconds into readable time string
    
    Args:
        seconds: Time in seconds
    
    Returns:
        str: Formatted time (e.g., "2h 34m", "45m 12s", "23s")
    """
    if seconds < 0 or seconds > 1e9:
        return "N/A"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def format_number(num):
    """
    Format number with thousands separators
    
    Args:
        num: Number to format
    
    Returns:
        str: Formatted number (e.g., "1,234,567")
    """
    return f"{int(num):,}"


def print_header(title, width=100):
    """
    Print a centered header with borders
    
    Args:
        title: Header title
        width: Total width of header
    """
    title_len = get_visible_len(title)
    padding = (width - title_len - 2) // 2
    line = "═" * width
    
    print(f"{C_CYAN}{line}{C_RESET}")
    print(f"{C_CYAN}║{C_RESET}{' ' * padding}{C_BOLD}{title}{C_RESET}{' ' * (width - padding - title_len - 2)}{C_CYAN}║{C_RESET}")
    print(f"{C_CYAN}{line}{C_RESET}")


def print_section_header(title):
    """
    Print a section header
    
    Args:
        title: Section title
    """
    print(f"\n{C_BOLD}{C_YELLOW}▸ {title}{C_RESET}")
    print(f"{C_GRAY}{'─' * 80}{C_RESET}")
