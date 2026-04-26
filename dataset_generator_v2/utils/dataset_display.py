"""
Dataset Generator Terminal Display  –  Midnight Commander-style boxed TUI.

Panels (top → bottom):
  ┌─ title bar ──────────────────────────────────────────────────────────────┐
  ├─ GPU 0 ──────────────────┬─ GPU 1 ─────────────────────────────────────┤
  │ per-stream status (left)  │ per-stream status (right)                   │
  ├─ WRITE QUEUE ────────────────────────────────────────────────────────────┤
  ├─ PRODUCTION PROGRESS ────────────────────────────────────────────────────┤
  │ per-category bar + degradation breakdown                                 │
  ├─ PERFORMANCE & ETA ──────────────────────────────────────────────────────┤
  └──────────────────────────────────────────────────────────────────────────┘

The layout adapts to terminal width.  Below 80 columns GPU panels stack
vertically instead of side-by-side.
"""

import sys
import shutil

from .terminal_ui import (
    ANSI_ESCAPE, C_RESET, C_BOLD, C_CYAN, C_GRAY, C_GREEN, C_RED,
    C_YELLOW, C_MAGENTA, C_BLUE, C_WHITE,
    ANSI_HOME, ANSI_CLEAR, ANSI_HIDE_CURSOR,
    get_visible_len, make_bar, format_time, format_number,
)

# ── Colour helpers ────────────────────────────────────────────────────────────

_CAT_COLORS = [C_RED, C_CYAN, C_MAGENTA, C_YELLOW, C_GREEN, C_WHITE]
_KNOWN_COLORS = {
    'master':    C_RED,
    'space':     C_CYAN,
    'toon':      C_MAGENTA,
    'universal': C_YELLOW,
}
_DEG_COLORS = [C_YELLOW, C_GREEN, C_CYAN, C_MAGENTA, C_WHITE, C_RED]


def _category_color(cat_key, index=0):
    return _KNOWN_COLORS.get(cat_key, _CAT_COLORS[index % len(_CAT_COLORS)])


def _category_display_name(cat_key):
    return cat_key.capitalize()


def _size_label(key):
    """Compact display label for a template name (e.g. '1152_169' → '1152×…')."""
    return key.replace('_', '×') if '_' in key else key


# ── Box-drawing primitives ────────────────────────────────────────────────────

_BC = C_CYAN        # box colour
_BG = C_GRAY        # box dimmed colour


def _box_top(width, title="", color=_BC):
    """Top border: ╔══ title ═══╗"""
    if title:
        t = f" {title} "
        tl = get_visible_len(t)
        avail = width - 2
        left = max(0, (avail - tl) // 2)
        right = max(0, avail - tl - left)
        return f"{color}╔{'═' * left}{C_BOLD}{t}{C_RESET}{color}{'═' * right}╗{C_RESET}"
    return f"{color}╔{'═' * (width - 2)}╗{C_RESET}"


def _box_mid(width, color=_BC):
    """Middle divider: ╠══════╣"""
    return f"{color}╠{'═' * (width - 2)}╣{C_RESET}"


def _box_bot(width, color=_BC):
    """Bottom border: ╚══════╝"""
    return f"{color}╚{'═' * (width - 2)}╝{C_RESET}"


def _box_row(content, width, color=_BC, pad=1):
    """Content row with left/right borders: ║ content   ║"""
    inner = width - 2 - 2 * pad
    vl = get_visible_len(content)
    space = max(0, inner - vl)
    p = ' ' * pad
    return f"{color}║{C_RESET}{p}{content}{' ' * space}{p}{color}║{C_RESET}"


def _box_empty(width, color=_BC):
    """Empty content row."""
    return _box_row("", width, color)


def _trunc(s, maxlen):
    """Truncate string to maxlen visible chars, appending '…' if needed."""
    if get_visible_len(s) <= maxlen:
        return s
    # Strip ANSI before truncating to avoid cutting inside escape codes
    plain = ANSI_ESCAPE.sub('', s)
    if len(plain) <= maxlen:
        return s
    return plain[:max(0, maxlen - 1)] + '…'


def _bar_row(label, pct, done, target, bar_width, color, label_width=12):
    """
    One progress-bar row suitable for inside a box.

    label      category / template name (left-aligned, padded to label_width)
    pct        0–100
    done       count already done (int)
    target     total target (int)
    bar_width  width of the █░ bar
    color      ANSI colour for the bar fill
    """
    lbl = f"{color}{label:<{label_width}}{C_RESET}"
    bar = make_bar(pct, bar_width, color)
    cnt = f"{C_BOLD}{format_number(done):>8}{C_RESET}{C_GRAY} /{format_number(target):>8}  ({pct:5.1f}%){C_RESET}"
    return f"{lbl} {bar} {cnt}"


# ── GPU stream panel (one column) ─────────────────────────────────────────────

def _gpu_panel_lines(stream, width):
    """
    Render the content lines for one GPU/stream panel.
    Returns a list of raw strings (no borders), each will be wrapped with _box_row.
    """
    gpu_idx   = stream.get("gpu_index", -1)
    gpu_name  = stream.get("gpu_name", "GPU")
    state     = stream.get("state", "idle")
    video     = stream.get("video_name", "—")
    fps       = stream.get("live_fps", 0.0)
    patches   = stream.get("patches_created", 0)
    wq        = stream.get("write_queue_depth", 0)
    pipeline  = stream.get("pipeline", "libplacebo")
    n_done    = stream.get("n_videos_done", 0)
    inner_w   = width - 4   # 2 borders + 2 padding

    # State colour
    if state == "running":
        sc = C_GREEN
        sl = "▶ running"
    elif state == "error":
        sc = C_RED
        sl = "✖ error"
    else:
        sc = C_GRAY
        sl = "· idle"

    # GPU header
    gpu_label = f"GPU {gpu_idx}" if gpu_idx >= 0 else "CPU"
    header = f"{C_BOLD}{C_CYAN}{gpu_label}{C_RESET}  {C_GRAY}{gpu_name}{C_RESET}"
    state_str = f"{sc}{sl}{C_RESET}"

    # Film name (truncated)
    film_line = f"{C_GRAY}Film :{C_RESET} {C_WHITE}{_trunc(video, inner_w - 8)}{C_RESET}"

    # Speed / pipeline
    fps_str = f"{fps:6.1f} fps" if state == "running" else "     — fps"
    speed_line = (
        f"{C_GRAY}Speed:{C_RESET} {C_GREEN if fps > 0 else C_GRAY}{fps_str}{C_RESET}  "
        f"{C_GRAY}{pipeline}{C_RESET}"
    )

    # Patches / write queue
    patch_line = (
        f"{C_GRAY}Patch:{C_RESET} {C_BOLD}{format_number(patches):>7}{C_RESET} this film  "
        f"{C_GRAY}WQ:{C_RESET} {wq}"
    )

    # Videos done counter
    done_line = f"{C_GRAY}Done :{C_RESET} {n_done} film(s) this session"

    return [
        f"{header}  {state_str}",
        film_line,
        speed_line,
        patch_line,
        done_line,
    ]


def _render_gpu_panels(streams, term_width):
    """
    Render all GPU/stream panels.  Two panels side-by-side when width ≥ 90,
    stacked vertically otherwise.
    """
    lines_out = []
    if not streams:
        # No active streams yet — show placeholder
        pw = min(term_width, 80)
        lines_out.append(_box_top(pw, " STREAMS – awaiting start "))
        lines_out.append(_box_row(f"{C_GRAY}No streams active yet.{C_RESET}", pw))
        lines_out.append(_box_bot(pw))
        return lines_out

    side_by_side = term_width >= 90 and len(streams) >= 2

    if side_by_side:
        # Two panels next to each other
        pw = (term_width - 3) // 2  # width per panel (3 = gap)
        # Group streams in pairs
        for pair_start in range(0, len(streams), 2):
            left  = streams[pair_start]
            right = streams[pair_start + 1] if pair_start + 1 < len(streams) else None

            left_title  = f"  GPU {left['gpu_index']} · {left['gpu_name'][:20]}  "
            right_title = (
                f"  GPU {right['gpu_index']} · {right['gpu_name'][:20]}  "
                if right else ""
            )

            lines_out.append(
                _box_top(pw, left_title)
                + "  "
                + (_box_top(pw, right_title) if right else "")
            )

            left_rows  = _gpu_panel_lines(left, pw)
            right_rows = _gpu_panel_lines(right, pw) if right else []
            max_rows = max(len(left_rows), len(right_rows))

            for r in range(max_rows):
                lr = left_rows[r]  if r < len(left_rows)  else ""
                rr = right_rows[r] if r < len(right_rows) else ""
                lines_out.append(
                    _box_row(lr, pw) + "  " + (_box_row(rr, pw) if right else "")
                )

            lines_out.append(
                _box_bot(pw) + "  " + (_box_bot(pw) if right else "")
            )
    else:
        # Stacked / single
        pw = min(term_width, 80)
        for stream in streams:
            gpu_idx = stream.get("gpu_index", -1)
            gpu_title = f"  GPU {gpu_idx} · {stream.get('gpu_name', '')[:28]}  "
            lines_out.append(_box_top(pw, gpu_title))
            for row in _gpu_panel_lines(stream, pw):
                lines_out.append(_box_row(row, pw))
            lines_out.append(_box_bot(pw))

    return lines_out


# ── Write-queue panel ─────────────────────────────────────────────────────────

def _render_write_queue_panel(state, term_width):
    """One-line write-queue status panel."""
    pw = min(term_width, 100)
    streams = state.get("active_streams", [])
    wq_total = sum(s.get("write_queue_depth", 0) for s in streams)
    fmt = state.get("output_format", "BMP")
    n_active = state.get("n_active_streams", 0)
    output_fmt_color = C_GREEN if fmt == "BMP" else C_YELLOW

    parts = [
        f"{C_GRAY}Writers:{C_RESET} {C_BOLD}{n_active}{C_RESET} active",
        f"{C_GRAY}Queue depth:{C_RESET} {C_BOLD}{wq_total:>4}{C_RESET}",
        f"{C_GRAY}Format:{C_RESET} {output_fmt_color}{C_BOLD}{fmt}{C_RESET}",
        f"{C_GRAY}Backpressure:{C_RESET} "
        + (f"{C_RED}HIGH{C_RESET}" if wq_total > 200 else
           f"{C_YELLOW}mod{C_RESET}" if wq_total > 80 else
           f"{C_GREEN}ok{C_RESET}"),
    ]
    content = "  ·  ".join(parts)

    lines = []
    lines.append(_box_top(pw, " WRITE QUEUE "))
    lines.append(_box_row(content, pw))
    lines.append(_box_bot(pw))
    return lines


# ── Production progress panel ─────────────────────────────────────────────────

def _aggregate_degrade_counts(state):
    """
    Aggregate per-stream degrade_counts into a global dict.

    Returns: {category: {template_name: count}}
    """
    agg = {}
    for s in state.get("active_streams", []):
        for cat, tdict in s.get("degrade_counts", {}).items():
            agg.setdefault(cat, {})
            for tname, cnt in tdict.items():
                agg[cat][tname] = agg[cat].get(tname, 0) + cnt
    return agg


def _render_production_panel(state, term_width):
    pw = min(term_width, 100)
    lines = []
    lines.append(_box_top(pw, " PRODUCTION PROGRESS "))

    cats       = state.get("categories", [])
    ovr        = state.get("overall_progress", {})
    cat_tgts   = {}  # category → target (built from overall_progress or patch_distribution)
    for cat, info in ovr.items():
        if isinstance(info, dict):
            cat_tgts[cat] = info.get("target", 0)

    # Also pull degradation breakdown (from live streams)
    degrade_global = _aggregate_degrade_counts(state)

    if not cats:
        lines.append(_box_row(f"{C_GRAY}No categories configured.{C_RESET}", pw))
        lines.append(_box_bot(pw))
        return lines

    # Bar width: inner – label – count area
    inner_w  = pw - 4
    label_w  = max(8, min(14, len(max(cats, key=len))))
    count_w  = 28   # " 123,456 / 200,000  (80.0%)"
    bar_w    = max(8, inner_w - label_w - count_w - 2)

    for i, cat in enumerate(cats):
        color = _category_color(cat, i)
        info  = ovr.get(cat, {})
        done  = int(info.get("created", 0)) if isinstance(info, dict) else 0
        tgt   = int(info.get("target",  0)) if isinstance(info, dict) else 0
        pct   = min(100.0, 100.0 * done / tgt) if tgt > 0 else 0.0

        row = _bar_row(cat, pct, done, tgt, bar_w, color, label_w)
        lines.append(_box_row(row, pw))

        # Degradation breakdown for this category (if available)
        dc = degrade_global.get(cat, {})
        if dc:
            cat_total_patches = sum(dc.values())
            sorted_deg = sorted(dc.items(), key=lambda kv: -kv[1])
            parts = []
            for di, (dname, dcnt) in enumerate(sorted_deg):
                dc_color = _DEG_COLORS[di % len(_DEG_COLORS)]
                dpct = 100.0 * dcnt / cat_total_patches if cat_total_patches else 0
                parts.append(
                    f"{dc_color}{dname}{C_RESET}{C_GRAY}:{dpct:.0f}%{C_RESET}"
                )
            deg_row = f"  {C_GRAY}╰ degrade: {C_RESET}" + f"{C_GRAY} · {C_RESET}".join(parts)
            lines.append(_box_row(deg_row, pw))

    lines.append(_box_bot(pw))
    return lines


# ── ETA / performance panel ───────────────────────────────────────────────────

def _render_eta_panel(state, term_width):
    pw = min(term_width, 100)
    lines = []
    lines.append(_box_top(pw, " PERFORMANCE & ETA "))

    # Global throughput from active streams
    streams   = state.get("active_streams", [])
    total_fps = sum(s.get("live_fps", 0.0) for s in streams)
    n_streams = max(1, state.get("n_active_streams", 0))
    videos_idx = state.get("current_video_index", 0)
    total_vids = state.get("total_videos", 0)
    overall    = state.get("overall_progress", {})
    output_fmt = state.get("output_format", "BMP")
    fmt_color  = C_GREEN if output_fmt == "BMP" else C_YELLOW

    # Per-stream FPS summary
    fps_parts = [
        f"{C_GRAY}Videos:{C_RESET} {videos_idx}/{total_vids}",
        f"{C_GRAY}Streams:{C_RESET} {C_BOLD}{n_streams}{C_RESET} active",
        f"{C_GRAY}FPS:{C_RESET} {C_BOLD}{total_fps:6.1f}{C_RESET}",
        f"{C_GRAY}Output:{C_RESET} {fmt_color}{C_BOLD}{output_fmt}{C_RESET}",
    ]
    lines.append(_box_row("  ".join(fps_parts), pw))

    # ETA per category
    eta_dict   = state.get("eta", {})
    cats       = state.get("categories", [])
    eta_parts  = []
    for i, cat in enumerate(cats):
        color  = _category_color(cat, i)
        info   = overall.get(cat, {})
        done   = int(info.get("created", 0)) if isinstance(info, dict) else 0
        tgt    = int(info.get("target",  0)) if isinstance(info, dict) else 0
        eta_v  = eta_dict.get(cat, "—")
        if isinstance(eta_v, (int, float)) and eta_v > 0:
            eta_s = format_time(eta_v)
        else:
            eta_s = str(eta_v) if eta_v != 0 else "—"
        eta_parts.append(
            f"{color}{_category_display_name(cat)}{C_RESET}{C_GRAY}:{C_RESET}{eta_s}"
        )
    if eta_parts:
        lines.append(_box_row(
            f"{C_GRAY}ETA:  {C_RESET}" + f"  {C_GRAY}│{C_RESET}  ".join(eta_parts), pw
        ))

    lines.append(_box_bot(pw))
    return lines


# ── Title bar ─────────────────────────────────────────────────────────────────

def _render_title(term_width):
    pw = min(term_width, 100)
    title = (
        f"{C_BOLD}🎬  DATASET GENERATOR V2{C_RESET}  "
        f"{C_GRAY}·{C_RESET}  Multi-Stream  "
        f"{C_GRAY}·{C_RESET}  {C_GREEN}libplacebo{C_RESET}  "
        f"{C_GRAY}·{C_RESET}  {C_YELLOW}BMP{C_RESET}/{C_CYAN}PNG{C_RESET}"
    )
    t_inner = f" {title} "
    tl = get_visible_len(t_inner)
    fill = max(0, pw - 2)
    left = max(0, (fill - tl) // 2)
    right = max(0, fill - tl - left)
    top  = f"{C_CYAN}╔{'═' * fill}╗{C_RESET}"
    mid  = f"{C_CYAN}║{C_RESET}{'═' * left}{t_inner}{'═' * right}{C_CYAN}║{C_RESET}"
    bot  = f"{C_CYAN}╚{'═' * fill}╝{C_RESET}"
    return [top, mid, bot]


# ── Main entry point ──────────────────────────────────────────────────────────

def draw_dataset_ui(state):
    """
    Draw the complete Midnight Commander-style dataset-generation dashboard.

    The layout (top → bottom):
      1. Title bar (generator name + mode indicators)
      2. GPU panels (side-by-side when ≥2 streams and term_width ≥ 90)
      3. Write-queue status
      4. Production progress (per-category bars + degradation breakdown)
      5. Performance & ETA

    All sections are boxed with Unicode box-drawing characters.

    Args:
        state: The ``ui_state`` dict from DatasetGeneratorV2UHD.  Expected keys:
            active_streams, n_active_streams, n_gpus_available,
            overall_progress, categories, eta, current_video_index,
            total_videos, output_format, patches_created_total.
    """
    term_width, _term_height = shutil.get_terminal_size((100, 50))

    out = []

    # 1. Title
    out.extend(_render_title(term_width))

    # 2. GPU stream panels
    streams = state.get("active_streams", [])
    out.extend(_render_gpu_panels(streams, term_width))

    # 3. Write queue
    out.extend(_render_write_queue_panel(state, term_width))

    # 4. Production progress
    out.extend(_render_production_panel(state, term_width))

    # 5. ETA
    out.extend(_render_eta_panel(state, term_width))

    sys.stdout.write(ANSI_CLEAR + ANSI_HOME)
    sys.stdout.write("\n".join(out))
    sys.stdout.write("\n")
    sys.stdout.flush()
