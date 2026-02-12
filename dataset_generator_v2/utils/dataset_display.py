"""
Dataset Generator Terminal Display

Main UI drawing function for dataset generation progress.
Similar to vsr_plusplus_NEU/utils/ui_display.py but for dataset generation.
"""

import sys
import shutil
from .terminal_ui import *


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
    
    # Progress bars for each category
    current_progress = state.get('current_video_progress', {})
    
    categories = [
        ('master', 'Master', C_RED),
        ('space', 'Space', C_CYAN),
        ('toon', 'Toon', C_MAGENTA),
        ('universal', 'Universal', C_YELLOW)
    ]
    
    bar_width = min(50, width - 40)
    
    for cat_key, cat_name, color in categories:
        cat_data = current_progress.get(cat_key, {'created': 0, 'target': 0, 'percent': 0.0})
        created = cat_data.get('created', 0)
        target = cat_data.get('target', 0)
        percent = cat_data.get('percent', 0.0)
        
        # Category label and numbers
        label = f"  {cat_name:10s}"
        numbers = f"{created:5d} / {target:5d} ({percent:5.1f}%)"
        
        # Progress bar
        bar = make_bar(percent, bar_width, color)
        
        print(f"{label}  {numbers}  {bar}")


def _draw_overall_progress_section(state, width):
    """Draw overall progress section"""
    print_section_header("GESAMTFORTSCHRITT ÜBER ALLE FILME")
    
    overall_progress = state.get('overall_progress', {})
    
    categories = [
        ('master', 'Master', C_RED),
        ('space', 'Space', C_CYAN),
        ('toon', 'Toon', C_MAGENTA),
        ('universal', 'Universal', C_YELLOW)
    ]
    
    bar_width = min(50, width - 45)
    
    for cat_key, cat_name, color in categories:
        cat_data = overall_progress.get(cat_key, {'created': 0, 'target': 150000, 'percent': 0.0})
        created = cat_data.get('created', 0)
        target = cat_data.get('target', 150000)
        percent = cat_data.get('percent', 0.0)
        
        # Category label and numbers
        label = f"  {cat_name:10s}"
        numbers = f"{format_number(created):>8s} / {format_number(target):>8s} ({percent:5.1f}%)"
        
        # Progress bar
        bar = make_bar(percent, bar_width, color)
        
        print(f"{label}  {numbers}  {bar}")


def _draw_patch_distribution_table(state, width):
    """Draw patch distribution table"""
    print_section_header("PATCH-VERTEILUNG NACH KATEGORIE UND GRÖẞE")
    
    patch_dist = state.get('patch_distribution', {})
    
    # Table header - use actual format sizes from config
    header = f"  {'Kategorie':<12s}  {'540':>15s}  {'720':>15s}  {'720_169':>15s}  {'Gesamt':>10s}"
    print(header)
    print(f"{C_GRAY}  {'─' * (len(header) - 10)}{C_RESET}")
    
    categories = [
        ('master', 'Master', C_RED),
        ('space', 'Space', C_CYAN),
        ('toon', 'Toon', C_MAGENTA),
        ('universal', 'Universal', C_YELLOW)
    ]
    
    for cat_key, cat_name, color in categories:
        cat_data = patch_dist.get(cat_key, {})
        
        # Get counts for each size - using actual format sizes
        size_540 = cat_data.get('540', {'count': 0, 'target': 0})
        size_720 = cat_data.get('720', {'count': 0, 'target': 0})
        size_720_169 = cat_data.get('720_169', {'count': 0, 'target': 0})
        
        count_540 = size_540.get('count', 0)
        target_540 = size_540.get('target', 0)
        count_720 = size_720.get('count', 0)
        target_720 = size_720.get('target', 0)
        count_720_169 = size_720_169.get('count', 0)
        target_720_169 = size_720_169.get('target', 0)
        
        total_count = count_540 + count_720 + count_720_169
        
        # Format row
        row_540 = f"{count_540:5d}/{target_540:5d}"
        row_720 = f"{count_720:5d}/{target_720:5d}"
        row_720_169 = f"{count_720_169:5d}/{target_720_169:5d}"
        
        print(f"  {color}{cat_name:<12s}{C_RESET}  {row_540:>15s}  {row_720:>15s}  {row_720_169:>15s}  {total_count:>10d}")


def _draw_statistics_and_eta(state, width):
    """Draw statistics and ETA section"""
    print_section_header("STATISTIKEN & GESCHÄTZTE RESTZEIT")
    
    # Statistics
    scenes_processed = state.get('scenes_processed', 0)
    patches_total = state.get('patches_created_total', 0)
    avg_time = state.get('avg_time_per_scene', 0.0)
    
    stats_line = f"  Szenen: {C_GREEN}{scenes_processed:>6d}{C_RESET}  |  "
    stats_line += f"Patches: {C_GREEN}{format_number(patches_total):>8s}{C_RESET}  |  "
    stats_line += f"Ø Zeit/Szene: {C_GREEN}{avg_time:>5.1f}s{C_RESET}"
    print(stats_line)
    
    print()
    
    # ETA for each category
    eta_data = state.get('eta', {})
    
    categories = [
        ('master', 'Master', C_RED),
        ('space', 'Space', C_CYAN),
        ('toon', 'Toon', C_MAGENTA),
        ('universal', 'Universal', C_YELLOW),
        ('total', 'GESAMT', C_WHITE)
    ]
    
    print("  ETA pro Kategorie:")
    for cat_key, cat_name, color in categories:
        eta_value = eta_data.get(cat_key, None)
        # Convert float seconds to readable time string
        if eta_value is None:
            eta_str = 'N/A'
        elif isinstance(eta_value, (int, float)):
            eta_str = format_time(eta_value)
        else:
            eta_str = str(eta_value)
        
        if cat_key == 'total':
            print(f"  {color}{C_BOLD}{cat_name:10s}: {eta_str:>15s}{C_RESET}")
        else:
            print(f"  {color}{cat_name:10s}: {eta_str:>15s}{C_RESET}")
    
    print()
