"""
Terminal UI Utilities

Helper functions for terminal-based UI display.
Ported from original train.py to maintain feature parity.
"""

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

# ANSI Control Codes
ANSI_HOME = "\033[H"
ANSI_CLEAR = "\033[2J"
ANSI_HIDE_CURSOR = "\033[?25l"
ANSI_SHOW_CURSOR = "\033[?25h"

# ANSI Escape Sequence Pattern (for stripping colors from text)
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

# Display Mode Names
DISPLAY_MODE_NAMES = [
    "2-Column Detailed (Backward | Forward)",
    "Grouped by Trunk → Sorted by Activity",
    "Flat List → Sorted by Position",
    "Flat List → Sorted by Activity"
]


def get_visible_len(text):
    """
    Get visible length of text (excluding ANSI escape codes)
    
    Args:
        text: String potentially containing ANSI codes
    
    Returns:
        int: Visible length of text
    """
    return len(ANSI_ESCAPE.sub('', text))


def make_bar(percent, width):
    """
    Create an ASCII progress bar
    
    Args:
        percent: Percentage (0-100)
        width: Width of the bar in characters
    
    Returns:
        str: Formatted progress bar with ANSI colors (green)
    """
    width = max(5, width)
    filled = max(0, min(width, int((percent / 100.0) * width)))
    return f"{C_GREEN}{'█' * filled}{C_GRAY}{'░' * (width - filled)}{C_RESET}"


def make_bar_fusion(percent, width):
    """
    Create an ASCII progress bar for fusion layers (cyan color)
    
    Args:
        percent: Percentage (0-100)
        width: Width of the bar in characters
    
    Returns:
        str: Formatted progress bar with ANSI colors (cyan for fusion layers)
    """
    width = max(5, width)
    filled = max(0, min(width, int((percent / 100.0) * width)))
    return f"{C_CYAN}{'█' * filled}{C_GRAY}{'░' * (width - filled)}{C_RESET}"


def make_bar_final_fusion(percent, width):
    """
    Create an ASCII progress bar for final fusion layer (yellow color)
    
    Args:
        percent: Percentage (0-100)
        width: Width of the bar in characters
    
    Returns:
        str: Formatted progress bar with ANSI colors (yellow for final fusion)
    """
    width = max(5, width)
    filled = max(0, min(width, int((percent / 100.0) * width)))
    return f"{C_YELLOW}{'█' * filled}{C_GRAY}{'░' * (width - filled)}{C_RESET}"


def make_adamw_magic_eye(momentum, width=20):
    """
    Create AdamW "Magic Eye" visualization (tube radio style)
    
    Shows momentum as a push/brake indicator:
    - High momentum (>0.5): Push right [       |====>  ]
    - Low momentum (<0.5): Brake left [  <====|       ]
    - Medium momentum: Balanced [   <=|=>   ]
    
    Args:
        momentum: Momentum value (typically 0-2, normalized to 0-1 range)
        width: Width of the bar in characters
    
    Returns:
        str: Formatted magic eye bar with ANSI colors
    """
    width = max(10, width)
    
    # Normalize momentum to 0-1 range (assume typical range is 0-2)
    normalized = min(1.0, max(0.0, momentum / 2.0))
    
    # Calculate center position
    center = width // 2
    
    # Calculate needle position based on momentum
    # normalized=0.5 -> center, normalized=1.0 -> right, normalized=0.0 -> left
    needle_pos = int(normalized * (width - 1))
    
    # Build the visualization
    if normalized > 0.55:  # Push right
        # More momentum = push to the right
        fill_start = center
        fill_end = needle_pos
        bar = ['·'] * width
        for i in range(fill_start, min(fill_end, width)):
            bar[i] = '='
        bar[center] = '|'
        if needle_pos < width:
            bar[needle_pos] = '>'
        result = ''.join(bar)
        return f"[{C_GREEN}{result}{C_RESET}]"
    
    elif normalized < 0.45:  # Brake left
        # Less momentum = brake to the left
        fill_start = needle_pos
        fill_end = center
        bar = ['·'] * width
        for i in range(max(0, fill_start + 1), fill_end + 1):
            bar[i] = '='
        bar[center] = '|'
        if needle_pos >= 0:
            bar[needle_pos] = '<'
        result = ''.join(bar)
        return f"[{C_YELLOW}{result}{C_RESET}]"
    
    else:  # Balanced
        bar = ['·'] * width
        bar[center - 1] = '='
        bar[center] = '|'
        bar[center + 1] = '='
        if needle_pos < center:
            bar[needle_pos] = '<'
        elif needle_pos > center:
            bar[needle_pos] = '>'
        result = ''.join(bar)
        return f"[{C_CYAN}{result}{C_RESET}]"


def format_time(seconds):
    """
    Format seconds into human-readable time string
    
    Args:
        seconds: Time in seconds
    
    Returns:
        str: Formatted time (e.g., "2d 5h 30m" or "3h 45m")
    """
    if seconds < 0:
        return "N/A"
    
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    else:
        return f"{hours}h {minutes}m"


def clear_screen():
    """Clear the terminal screen"""
    print(ANSI_CLEAR, end='')


def move_cursor_home():
    """Move cursor to home position (top-left)"""
    print(ANSI_HOME, end='')


def hide_cursor():
    """Hide terminal cursor"""
    print(ANSI_HIDE_CURSOR, end='')


def show_cursor():
    """Show terminal cursor"""
    print(ANSI_SHOW_CURSOR, end='')


def print_line(content, ui_width):
    """
    Print a single line within UI borders
    
    Args:
        content: Content to print
        ui_width: Total UI width
    """
    visible_len = get_visible_len(content)
    max_content_width = ui_width - 4  # Account for borders " ║ " and " ║"
    
    # Truncate content if it exceeds maximum width
    if visible_len > max_content_width:
        # Find where to cut (accounting for ANSI codes)
        truncated = ""
        current_visible = 0
        in_ansi = False
        ansi_buffer = ""
        
        for char in content:
            if char == '\033':
                in_ansi = True
                ansi_buffer = char
            elif in_ansi:
                ansi_buffer += char
                if char in 'mHJKSTfABCDsu':  # ANSI sequence terminators
                    truncated += ansi_buffer
                    in_ansi = False
                    ansi_buffer = ""
            else:
                if current_visible < max_content_width - 3:  # Leave space for "..."
                    truncated += char
                    current_visible += 1
                elif current_visible == max_content_width - 3:
                    truncated += "..."
                    current_visible += 3
                    break
        
        content = truncated + C_RESET
        visible_len = max_content_width
    
    # Pad to exact width
    padding = max(0, max_content_width - visible_len)
    sys.stdout.write(f" ║ {content}{' ' * padding} ║\n")


def print_two_columns(left_content, right_content, ui_width):
    """
    Print two columns within UI borders
    
    Args:
        left_content: Left column content
        right_content: Right column content (can be None)
        ui_width: Total UI width
    """
    col_width = (ui_width - 7) // 2  # For two-column layout
    
    # Get visible lengths (accounting for ANSI codes)
    left_vis = get_visible_len(left_content)
    right_vis = get_visible_len(right_content) if right_content else 0
    
    # Pad to exact column width
    left_pad = max(0, col_width - left_vis)
    right_pad = max(0, col_width - right_vis)
    
    left_str = f"{left_content}{' ' * left_pad}"
    right_str = f"{right_content}{' ' * right_pad}" if right_content else (' ' * col_width)
    
    sys.stdout.write(f" ║ {left_str} │ {right_str} ║\n")


def print_separator(ui_width, style='single'):
    """
    Print a separator line
    
    Args:
        ui_width: Total UI width
        style: 'single', 'double', or 'thin'
    """
    if style == 'double':
        sys.stdout.write(f" {C_GRAY}╠{'═'*(ui_width-2)}╣{C_RESET}\n")
    elif style == 'thin':
        sys.stdout.write(f" {C_GRAY}╟{'·'*(ui_width-2)}╢{C_RESET}\n")
    else:  # single
        sys.stdout.write(f" {C_GRAY}╟{'─'*(ui_width-2)}╢{C_RESET}\n")


def print_header(ui_width):
    """Print UI header"""
    sys.stdout.write(f" {C_GRAY}╔{'═'*(ui_width-2)}╗{C_RESET}\n")


def print_footer(ui_width):
    """Print UI footer"""
    sys.stdout.write(f" {C_GRAY}╚{'═'*(ui_width-2)}╝{C_RESET}\n")
