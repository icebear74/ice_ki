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
            - active_streams: list of per-worker dicts (stream_id, video_name,
              gpu_index, gpu_name, state, frames_processed, patches_created,
              write_queue_depth) — populated during parallel extraction
            - n_active_streams: int — quick count of running streams
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

    # Active Streams (multi-GPU) — only when parallel workers are active
    _draw_active_streams_section(state, term_width)

    # Timing Breakdown (only when data is available)
    _draw_timing_breakdown(state, term_width)
    
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
        # "created" and "target" are GT-Bilder (= Szenen); each GT-Bild has one
        # stacked LR counterpart containing n_frames frames.
        numbers = f"{created:5d} GT / {target:5d} GT ({percent:5.1f}%)"
        bar = make_bar(percent, bar_width, color)
        
        print(f"{label}  {numbers}  {bar}")


def _draw_overall_progress_section(state, width):
    """Draw overall progress section"""
    print_section_header("GESAMTFORTSCHRITT ÜBER ALLE FILME")
    
    overall_progress = state.get('overall_progress', {})
    categories = state.get('categories') or list(overall_progress.keys())
    
    bar_width = min(50, width - 50)  # 5 chars wider margin to fit the added ' GT' suffixes
    
    for idx, cat_key in enumerate(categories):
        color = _category_color(cat_key, idx)
        cat_name = _category_display_name(cat_key)
        cat_data = overall_progress.get(cat_key, {'created': 0, 'target': 0, 'percent': 0.0})
        created = cat_data.get('created', 0)
        target = cat_data.get('target', 0)
        percent = cat_data.get('percent', 0.0)
        
        label = f"  {cat_name:10s}"
        # target = category_targets[category] = the user-configured GT-Bilder goal
        numbers = f"{format_number(created):>8s} GT / {format_number(target):>8s} GT ({percent:5.1f}%)"
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

    # Use real gt_size labels (e.g. "1152×648") when available; fall back to
    # the legacy _size_label truncation only when the mapping is absent.
    format_labels_map = state.get('format_labels', {})
    size_labels = [
        format_labels_map.get(k) or _size_label(k)
        for k in format_sizes
    ]

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
            # patch_distribution is keyed by raw format name (e.g. '1152_169').
            # Fall back to the short label for backward compatibility.
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

    # Parallel-worker status line (only shown while workers are running)
    parallel_status = state.get('parallel_status', '')
    if parallel_status:
        print(f"  {C_YELLOW}{parallel_status}{C_RESET}")
        print()

    frames_read   = state.get('frames_read_total', 0)       # frames actually piped from FFmpeg to Python
    frames_total  = state.get('frames_processed_total', 0)  # center-frame assignments evaluated
    gt_total      = state.get('patches_created_total', 0)   # GT-Bilder saved (= Szenen, cross-category sum)
    skipped       = max(0, frames_total - gt_total)
    avg_time      = state.get('avg_time_per_scene', 0.0)
    live_fps      = state.get('live_fps', 0.0)
    live_sps      = state.get('live_sps', 0.0)

    # Piped      = frames actually piped from FFmpeg stdout to Python (selected frames only)
    # Szenen      = center-frame assignments evaluated (= unique scene positions across categories)
    # GT-Bilder   = GT files saved; each GT-Bild corresponds to exactly 1 scene and 1 stacked LR
    #               file containing n_frames LR frames (e.g. 7 × 30 000 = 210 000 LR frames)
    # Übersprungen = assignments that produced no GT file (black frame or crop failure)
    line1 = (
        f"  Piped: {C_GREEN}{format_number(frames_read):>8s}{C_RESET}  |  "
        f"Szenen: {C_GREEN}{format_number(frames_total):>8s}{C_RESET}  |  "
        f"GT-Bilder: {C_GREEN}{format_number(gt_total):>8s}{C_RESET}  |  "
        f"Übersprungen: {C_GREEN}{format_number(skipped):>6s}{C_RESET}"
    )
    if avg_time > 0:
        line1 += f"  |  Ø Zeit/Szene: {C_GREEN}{avg_time:>5.1f}s{C_RESET}"
    print(line1)

    # FPS / SPS throughput line (only shown once the pipeline has started)
    if live_fps > 0 or live_sps > 0:
        fps_str = f"{live_fps:>7.1f}" if live_fps > 0 else "    N/A"
        sps_str = f"{live_sps:>7.2f}" if live_sps > 0 else "    N/A"
        print(
            f"  FPS (piped): {C_CYAN}{fps_str}{C_RESET}  |  "
            f"SPS (scenes/s): {C_CYAN}{sps_str}{C_RESET}"
        )

    # Decode backend (CPU / GPU N + pipeline tier) — always shown
    backend = state.get("decode_backend", "")
    if backend and backend != "–":
        print(f"  Decode-Backend : {C_CYAN}{backend}{C_RESET}")

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


def _draw_active_streams_section(state, width):
    """Draw the per-stream / per-GPU status table for parallel extraction."""
    streams = state.get('active_streams', [])
    if not streams:
        return  # nothing to show when no parallel workers are active

    _state_color = {
        "queued":  C_WHITE,
        "running": C_GREEN,
        "done":    C_CYAN,
        "error":   C_RED,
    }

    print_section_header("AKTIVE STREAMS (Multi-GPU)")
    header = (
        f"  {'St':>2}  {'GPU':>3}  {'Name':<35}  {'State':>7}"
        f"  {'Frames':>8}  {'Patches':>7}  {'Queue':>5}"
    )
    print(f"{C_GRAY}{header}{C_RESET}")
    print(f"  {'─' * (width - 4)}")

    for s in streams:
        sid     = s.get('stream_id', 0)
        gidx    = s.get('gpu_index', -1)
        vname   = s.get('video_name', '?')[:35]
        st      = s.get('state', 'queued')
        frames  = s.get('frames_processed', 0)
        patches = s.get('patches_created', 0)
        qdepth  = s.get('write_queue_depth', 0)
        scol    = _state_color.get(st, C_WHITE)
        # Queue warning colour: yellow when > 128, red when > 200 (maxsize=256)
        qcol    = C_RED if qdepth > 200 else (C_YELLOW if qdepth > 128 else C_GREEN)
        print(
            f"  {C_BOLD}{sid:>2}{C_RESET}  "
            f"{C_MAGENTA}{gidx:>3}{C_RESET}  "
            f"{C_CYAN}{vname:<35}{C_RESET}  "
            f"{scol}{st:>7}{C_RESET}  "
            f"{C_GREEN}{frames:>8,d}{C_RESET}  "
            f"{C_YELLOW}{patches:>7,d}{C_RESET}  "
            f"{qcol}{qdepth:>5d}{C_RESET}"
        )

    n_running = state.get('n_active_streams', 0)
    total     = len(streams)
    print(
        f"\n  {C_BOLD}{n_running}/{total}{C_RESET} Streams aktiv  "
        f"{C_GRAY}(round-robin GPU-Zuweisung){C_RESET}"
    )
    print()


def _draw_timing_breakdown(state, width):
    """Draw per-phase timing breakdown to help identify the performance bottleneck."""
    t = state.get('timing_phases', {})
    nc  = t.get('n_centers', 0)
    nf  = t.get('n_frames_buf', 0)
    np_ = t.get('n_patches', 0)
    if nc == 0:
        return  # No timing data yet — skip section

    print_section_header("TIMING BREAKDOWN (Bottleneck-Analyse)")

    ms_buf   = t.get('t_buf_s',   0.0) / max(nf, 1) * 1000
    ms_black = t.get('t_black_s', 0.0) / nc * 1000
    ms_patch = t.get('t_patch_s', 0.0) / nc * 1000
    ms_write = t.get('t_write_s', 0.0) / max(np_, 1) * 1000
    ms_total = ms_buf + ms_black + ms_patch + ms_write
    qsize    = t.get('q_size_last', 0)

    def _bar(ms, total_ms, w=20):
        frac = (ms / total_ms) if total_ms > 0 else 0.0
        filled = int(frac * w)
        return '█' * filled + '░' * (w - filled)

    def _phase_line(label, ms, unit, bar):
        return (
            f"  {C_CYAN}{label:<18s}{C_RESET}"
            f"  {C_GREEN}{ms:>8.1f}ms{C_RESET}/{unit}  "
            f"{C_YELLOW}{bar}{C_RESET}"
        )

    print(_phase_line("Buffer/Frame",   ms_buf,   "frame", _bar(ms_buf,   ms_total)))
    print(_phase_line("Schwarz-Check",  ms_black, "ctr",   _bar(ms_black, ms_total)))
    print(_phase_line("Patch-Erzeugung",ms_patch, "ctr",   _bar(ms_patch, ms_total)))
    print(_phase_line("Write-Queue",    ms_write, "patch", _bar(ms_write, ms_total)))

    # Warn if the write queue is backing up (> 80% full, maxsize=256)
    q_warn = f"  {C_RED}⚠ Queue fast voll!{C_RESET}" if qsize > 200 else ""
    print(
        f"  {'Write-Queue Tiefe':<18s}  {C_GREEN}{qsize:>3d}{C_RESET}/256{q_warn}"
    )

    # Theoretical max SPS from Python-side work alone (excludes FFmpeg decode time)
    if ms_total > 0:
        max_sps = 1000.0 / ms_total
        print(
            f"  {'Python-seitig max':<18s}  {C_CYAN}{max_sps:>6.2f}{C_RESET} SPS "
            f"(exkl. FFmpeg-Decode)"
        )
        # If Python-side max >> actual SPS, FFmpeg decode is the bottleneck.
        live_sps = state.get('live_sps', 0.0)
        if live_sps > 0 and max_sps > live_sps * 2:
            print(
                f"  {C_YELLOW}→ Bottleneck: FFmpeg-Decode "
                f"(Python {max_sps:.1f} SPS >> Actual {live_sps:.2f} SPS){C_RESET}"
            )
        elif live_sps > 0 and max_sps <= live_sps * 1.3:
            print(
                f"  {C_RED}→ Bottleneck: Python-Verarbeitung "
                f"(Patch-Erzeugung oder Write-Queue){C_RESET}"
            )
    print()
