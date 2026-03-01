"""
Dataset Generator Terminal Display

Main UI drawing function for dataset generation progress.
Similar to vsr_plusplus_NEU/utils/ui_display.py but for dataset generation.
"""

import sys
import shutil
from .terminal_ui import *

# Color cycle for dynamically-assigned categories
_CAT_COLORS = [C_RED, C_CYAN, C_MAGENTA, C_YELLOW, C_GREEN, C_WHITE]

# Well-known category → fixed color so existing configs keep their colour
_KNOWN_COLORS = {
    'master':    C_RED,
    'space':     C_CYAN,
    'toon':      C_MAGENTA,
    'universal': C_YELLOW,
}


def _category_color(cat_key, index):
    """Return a terminal color for *cat_key*, falling back to a cycle."""
    return _KNOWN_COLORS.get(cat_key, _CAT_COLORS[index % len(_CAT_COLORS)])


def _category_display_name(cat_key):
    """Human-readable name for a category key."""
    return cat_key.capitalize()


def _size_label(key):
    """Display label for a format-size key (e.g. '720_169' → '169', '540' → '540')."""
    if '_' in key:
        return key.split('_', 1)[1]
    return key


def draw_dataset_ui(state):
    """
    Draw the complete dataset generation UI
    
    Args:
        state: Dictionary containing current state:
            - current_video_name: str
            - current_video_index: int
            - total_videos: int
            - current_video_progress: dict (per category: created, target, percent)
            - overall_progress: dict (per category: created, target, percent)
            - patch_distribution: dict (category -> size -> count/target)
            - eta: dict (category -> time string)
            - scenes_processed: int
            - patches_created_total: int
            - avg_time_per_scene: float
            - categories: list of category keys (dynamic, from config)
            - format_sizes: list of size keys (dynamic, from config)
    """
    # Get terminal size
    term_width, term_height = shutil.get_terminal_size((100, 50))
    
    # Clear screen and hide cursor
    clear_screen()
    
    # Header
    print_header("🎬 DATASET GENERATOR - ECHTZEIT FORTSCHRITT", term_width)
    
    # Current Video Section
    _draw_current_video_section(state, term_width)
    
    # Overall Progress Section
    _draw_overall_progress_section(state, term_width)
    
    # Patch Distribution Table
    _draw_patch_distribution_table(state, term_width)
    
    # Statistics and ETA
    _draw_statistics_and_eta(state, term_width)
    
    # Flush output
    sys.stdout.flush()


def _draw_current_video_section(state, width):
    """Draw current video progress section"""
    print_section_header("AKTUELLER FILM")
    
    # Video info
    video_name = state.get('current_video_name', 'Warte auf Start...')
    current_idx = state.get('current_video_index', 0)
    total_videos = state.get('total_videos', 0)
    
    print(f"  Film {C_BOLD}{current_idx}{C_RESET} / {total_videos}")
    print(f"  {C_CYAN}{video_name}{C_RESET}")
    print()
    
    # Progress bars – only for categories present in the config
    current_progress = state.get('current_video_progress', {})
    categories = state.get('categories') or list(current_progress.keys())
    
    bar_width = min(50, width - 40)
    
    for idx, cat_key in enumerate(categories):
        color = _category_color(cat_key, idx)
        cat_name = _category_display_name(cat_key)
        cat_data = current_progress.get(cat_key, {'created': 0, 'target': 0, 'percent': 0.0})
        created = cat_data.get('created', 0)
        target = cat_data.get('target', 0)
        percent = cat_data.get('percent', 0.0)
        
        label = f"  {cat_name:10s}"
        numbers = f"{created:5d} / {target:5d} ({percent:5.1f}%)"
        bar = make_bar(percent, bar_width, color)
        
        print(f"{label}  {numbers}  {bar}")


def _draw_overall_progress_section(state, width):
    """Draw overall progress section"""
    print_section_header("GESAMTFORTSCHRITT ÜBER ALLE FILME")
    
    overall_progress = state.get('overall_progress', {})
    categories = state.get('categories') or list(overall_progress.keys())
    
    bar_width = min(50, width - 45)
    
    for idx, cat_key in enumerate(categories):
        color = _category_color(cat_key, idx)
        cat_name = _category_display_name(cat_key)
        cat_data = overall_progress.get(cat_key, {'created': 0, 'target': 0, 'percent': 0.0})
        created = cat_data.get('created', 0)
        target = cat_data.get('target', 0)
        percent = cat_data.get('percent', 0.0)
        
        label = f"  {cat_name:10s}"
        numbers = f"{format_number(created):>8s} / {format_number(target):>8s} ({percent:5.1f}%)"
        bar = make_bar(percent, bar_width, color)
        
        print(f"{label}  {numbers}  {bar}")


def _draw_patch_distribution_table(state, width):
    """Draw patch distribution table"""
    print_section_header("PATCH-VERTEILUNG NACH KATEGORIE UND GRÖẞE")
    
    patch_dist = state.get('patch_distribution', {})
    categories = state.get('categories') or list(patch_dist.keys())

    # Determine size columns dynamically from config, falling back to what's in the data
    format_sizes = state.get('format_sizes') or []
    if not format_sizes:
        # Collect all size keys present across all categories
        seen = []
        for cat_data in patch_dist.values():
            for k in cat_data:
                if k not in seen:
                    seen.append(k)
        format_sizes = seen or ['540', '169', '720']

    # Map raw size keys to display labels (e.g. '720_169' → '169')
    size_labels = [_size_label(k) for k in format_sizes]

    # Table header
    col_w = 15
    header = f"  {'Kategorie':<12s}"
    for lbl in size_labels:
        header += f"  {lbl:>{col_w}s}"
    header += f"  {'Gesamt':>10s}"
    print(header)
    print(f"{C_GRAY}  {'─' * (len(header) - 10)}{C_RESET}")
    
    for idx, cat_key in enumerate(categories):
        color = _category_color(cat_key, idx)
        cat_name = _category_display_name(cat_key)
        cat_data = patch_dist.get(cat_key, {})
        
        total_count = 0
        cells = []
        for size_key, lbl in zip(format_sizes, size_labels):
            # patch_distribution is keyed by raw format name (e.g. '720_169').
            # Fall back to the short label ('169') for backward compatibility with
            # any state dicts produced before this fix.
            entry = cat_data.get(size_key) or cat_data.get(lbl) or {'count': 0, 'target': 0}
            count = entry.get('count', 0)
            target = entry.get('target', 0)
            total_count += count
            cells.append(f"{count:5d}/{target:5d}")
        
        row = f"  {color}{cat_name:<12s}{C_RESET}"
        for cell in cells:
            row += f"  {cell:>{col_w}s}"
        row += f"  {total_count:>10d}"
        print(row)


def _draw_statistics_and_eta(state, width):
    """Draw statistics and ETA section"""
    print_section_header("STATISTIKEN & GESCHÄTZTE RESTZEIT")

    frames_read   = state.get('frames_read_total', 0)       # raw frames decoded by FFmpeg
    frames_total  = state.get('frames_processed_total', 0)  # center-frame assignments evaluated
    patches_total = state.get('patches_created_total', 0)
    skipped       = max(0, frames_total - patches_total)
    avg_time      = state.get('avg_time_per_scene', 0.0)

    # Line 1: raw frames decoded  |  assignments evaluated  |  patches saved
    line1 = (
        f"  Gelesen: {C_GREEN}{format_number(frames_read):>8s}{C_RESET}  |  "
        f"Auswertungen: {C_GREEN}{format_number(frames_total):>8s}{C_RESET}  |  "
        f"Patches: {C_GREEN}{format_number(patches_total):>8s}{C_RESET}  |  "
        f"Übersprungen: {C_GREEN}{format_number(skipped):>6s}{C_RESET}"
    )
    if avg_time > 0:
        line1 += f"  |  Ø Zeit/Szene: {C_GREEN}{avg_time:>5.1f}s{C_RESET}"
    print(line1)

    print()
    
    # ETA – only configured categories + total
    eta_data = state.get('eta', {})
    categories = state.get('categories') or [k for k in eta_data if k != 'total']
    
    print("  ETA pro Kategorie:")
    for idx, cat_key in enumerate(categories):
        color = _category_color(cat_key, idx)
        cat_name = _category_display_name(cat_key)
        eta_value = eta_data.get(cat_key, None)
        if eta_value is None:
            eta_str = 'N/A'
        elif isinstance(eta_value, (int, float)):
            eta_str = format_time(eta_value)
        else:
            eta_str = str(eta_value)
        print(f"  {color}{cat_name:10s}: {eta_str:>15s}{C_RESET}")
    
    # Total ETA
    total_eta = eta_data.get('total', None)
    if total_eta is None:
        total_str = 'N/A'
    elif isinstance(total_eta, (int, float)):
        total_str = format_time(total_eta)
    else:
        total_str = str(total_eta)
    print(f"  {C_WHITE}{C_BOLD}{'GESAMT':10s}: {total_str:>15s}{C_RESET}")
    
    print()
