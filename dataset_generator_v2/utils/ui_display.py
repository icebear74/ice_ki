"""
Dataset Generator Display Module

Professional GUI display using box-drawing characters.
Based on vsr_plusplus/utils/ui_display.py design.
"""

import time
from datetime import timedelta
from .ui_terminal import *


def draw_dataset_generator_ui(generator):
    """
    Draw the complete dataset generator UI with box drawing
    
    Args:
        generator: DatasetGeneratorV2 instance with all state
    """
    # Clear screen and move cursor home
    clear_and_home()
    
    # Get terminal width
    ui_w = get_terminal_width()
    ui_w = min(ui_w, 120)  # Cap at 120 for readability
    
    # Calculate statistics
    elapsed = time.time() - generator.start_time
    elapsed_str = format_time(elapsed)
    
    current_idx = generator.tracker.status['progress']['current_video_index']
    total_videos = generator.tracker.status['progress']['total_videos']
    completed_videos = generator.tracker.status['progress']['completed_videos']
    
    # ETA calculation
    if completed_videos > 0:
        avg_time_per_video = elapsed / completed_videos
        remaining_videos = total_videos - completed_videos
        eta_seconds = avg_time_per_video * remaining_videos
        eta_str = format_time(eta_seconds)
    else:
        eta_str = "Calculating..."
    
    # Extraction speed
    if elapsed > 0:
        extractions_per_sec = generator.extractions_count / elapsed
        speed_str = f"{extractions_per_sec:.1f} /sec"
    else:
        speed_str = "..."
    
    # === HEADER ===
    title = f"{C_BOLD}{C_CYAN}DATASET GENERATOR v2.0 - LIVE{C_RESET}"
    print_header(ui_w, title)
    
    # === CURRENT VIDEO ===
    current_movie = generator.current_video_name if generator.current_video_name else "Initializing..."
    print_line(f"{C_YELLOW}🎬 CURRENT: {current_movie[:60]}{C_RESET}", ui_w)
    
    checkpoint = generator.tracker.get_video_checkpoint(current_idx)
    if checkpoint and checkpoint.get('status') == 'in_progress':
        done = checkpoint.get('extractions_done', 0)
        target = checkpoint.get('extractions_target', 1)
        progress_pct = (done / target * 100) if target > 0 else 0
        
        # Progress bar for current video
        bar_width = ui_w - 40  # Leave space for text
        bar = make_bar(progress_pct, bar_width)
        print_line(f"Progress: {bar} {progress_pct:.1f}% ({done:,}/{target:,})", ui_w)
    
    print_separator(ui_w, 'double')
    
    # === OVERALL PROGRESS ===
    print_line(f"{C_BOLD}📊 OVERALL PROGRESS{C_RESET}", ui_w)
    print_separator(ui_w, 'thin')
    
    # Videos progress
    completion_pct = (completed_videos/total_videos*100) if total_videos > 0 else 0
    bar_width = ui_w - 45
    bar = make_bar(completion_pct, bar_width)
    print_line(f"Videos: {bar} {completion_pct:.1f}%", ui_w)
    
    # Two column stats
    print_two_columns(
        f"Completed: {C_GREEN}{completed_videos}{C_RESET}/{total_videos}",
        f"Remaining: {C_CYAN}{total_videos - completed_videos}{C_RESET}",
        ui_w
    )
    print_two_columns(
        f"Elapsed: {C_CYAN}{elapsed_str}{C_RESET}",
        f"ETA: {C_YELLOW}{eta_str}{C_RESET}",
        ui_w
    )
    
    # Extraction stats
    success_rate = (generator.success_count/generator.extractions_count*100) if generator.extractions_count > 0 else 0
    print_separator(ui_w, 'thin')
    print_two_columns(
        f"Extractions: {C_CYAN}{generator.extractions_count:,}{C_RESET}",
        f"Speed: {C_GREEN}{speed_str}{C_RESET}",
        ui_w
    )
    print_two_columns(
        f"Successful: {C_GREEN}{generator.success_count:,}{C_RESET}",
        f"Success Rate: {C_GREEN}{success_rate:.1f}%{C_RESET}",
        ui_w
    )
    
    print_separator(ui_w, 'double')
    
    # === CATEGORY PROGRESS ===
    print_line(f"{C_BOLD}📦 CATEGORY PROGRESS{C_RESET}", ui_w)
    print_separator(ui_w, 'thin')
    
    # Calculate bar width for categories
    cat_bar_width = max(20, ui_w - 65)
    
    for cat_name in sorted(generator.config.get('category_targets', {}).keys()):
        stats = generator.tracker.status['category_stats'].get(cat_name, {})
        images = stats.get('images_created', 0)
        target = stats.get('target', 1)
        progress = (images / target * 100) if target > 0 else 0
        
        # Calculate ETA for this category
        if images > 0 and elapsed > 0:
            rate = images / elapsed
            remaining = target - images
            eta_secs = remaining / rate if rate > 0 else 0
            eta_str_cat = format_time(eta_secs) if eta_secs > 0 and eta_secs < 1e6 else "Complete"
        else:
            eta_str_cat = "..."
        
        # Color bar based on progress
        if progress >= 90:
            bar = make_bar(progress, cat_bar_width)
        elif progress >= 50:
            bar = make_bar_cyan(progress, cat_bar_width)
        else:
            bar = make_bar_yellow(progress, cat_bar_width)
        
        # Category line
        cat_display = f"{cat_name.upper():<12}"
        print_line(f"{cat_display} {bar} {progress:5.1f}%", ui_w)
        
        # Stats line
        print_two_columns(
            f"  Images: {C_CYAN}{images:,}{C_RESET} / {target:,}",
            f"  ETA: {C_YELLOW}{eta_str_cat}{C_RESET}",
            ui_w
        )
        
        if cat_name != sorted(generator.config.get('category_targets', {}).keys())[-1]:
            print_separator(ui_w, 'thin')
    
    print_separator(ui_w, 'double')
    
    # === DISK USAGE ===
    print_line(f"{C_BOLD}💾 DISK USAGE{C_RESET}", ui_w)
    print_separator(ui_w, 'thin')
    
    total_disk = sum(s.get('disk_usage_gb', 0) for s in generator.tracker.status['category_stats'].values())
    categories = sorted(generator.config.get('category_targets', {}).keys())
    
    # Show categories in two columns if more than 4
    if len(categories) <= 4:
        for cat_name in categories:
            usage = generator.tracker.status['category_stats'].get(cat_name, {}).get('disk_usage_gb', 0)
            print_two_columns(
                f"{cat_name.upper()}: {C_CYAN}{usage:.2f} GB{C_RESET}",
                "",
                ui_w
            )
    else:
        # Two column layout for many categories
        for i in range(0, len(categories), 2):
            cat1 = categories[i]
            usage1 = generator.tracker.status['category_stats'].get(cat1, {}).get('disk_usage_gb', 0)
            left = f"{cat1.upper()}: {C_CYAN}{usage1:.2f} GB{C_RESET}"
            
            if i + 1 < len(categories):
                cat2 = categories[i+1]
                usage2 = generator.tracker.status['category_stats'].get(cat2, {}).get('disk_usage_gb', 0)
                right = f"{cat2.upper()}: {C_CYAN}{usage2:.2f} GB{C_RESET}"
            else:
                right = ""
            
            print_two_columns(left, right, ui_w)
    
    print_separator(ui_w, 'thin')
    print_line(f"Total: {C_BOLD}{C_GREEN}{total_disk:.2f} GB{C_RESET}", ui_w)
    
    print_separator(ui_w, 'double')
    
    # === CONTROLS ===
    status_icon = f"{C_GREEN}●{C_RESET}" if not generator.paused else f"{C_YELLOW}●{C_RESET}"
    status_text = f"{C_GREEN}RUNNING{C_RESET}" if not generator.paused else f"{C_YELLOW}PAUSED{C_RESET}"
    print_line(f"{C_BOLD}⚙️  LIVE CONTROLS{C_RESET} {status_icon} {status_text}", ui_w)
    print_separator(ui_w, 'thin')
    
    print_two_columns(
        f"Workers: {C_BOLD}{C_CYAN}{generator.workers}{C_RESET} cores",
        f"[SPACE] Pause/Resume",
        ui_w
    )
    print_two_columns(
        "[+/-] Adjust workers",
        "[q] Quit | [Ctrl+C] Save & Exit",
        ui_w
    )
    
    # === FOOTER ===
    print_footer(ui_w)
    
    # Flush output
    sys.stdout.flush()
