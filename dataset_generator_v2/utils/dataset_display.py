"""
Dataset Generator Terminal Display  –  Midnight Commander-style boxed TUI.

Panels (top → bottom):
  ┌─ title bar ──────────────────────────────────────────────────────────────┐
  ├─ GPU 0 ──────────────────┬─ GPU 1 ─────────────────────────────────────┤
  │ per-stream status (left)  │ per-stream status (right)                   │
  ├─ WRITE QUEUE ────────────────────────────────────────────────────────────┤
  ├─ PLAN SUMMARY ───────────────────────────────────────────────────────────┤
  │ total / done / running / pending / failed plan items                     │
  ├─ PRODUCTION PROGRESS ────────────────────────────────────────────────────┤
  │ per-category bar + per-format-template bars + degradation breakdown      │
  ├─ PERFORMANCE & ETA ──────────────────────────────────────────────────────┤
  └──────────────────────────────────────────────────────────────────────────┘

The layout adapts to terminal width.  Below 80 columns GPU panels stack
vertically instead of side-by-side.

Plan-driven fields read from ``ui_state``:
  plan_summary  – global stats from GenerationPlan.get_global_stats()
  current_plan_items  – per-stream {plan_item_id, queue_position, …}
  patch_distribution  – {cat: {fmt: {count, target, deg_planned, deg_completed}}}
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
_FMT_COLORS = [C_CYAN, C_GREEN, C_YELLOW, C_MAGENTA, C_WHITE, C_BLUE]


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

    # Plan metadata for this stream
    plan_id   = stream.get("plan_item_id", "")
    queue_pos = stream.get("queue_position", 0)
    planned_total = stream.get("planned_total", 0)

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

    # Plan item ID + queue position
    plan_line = (
        f"{C_GRAY}Plan :{C_RESET} "
        f"{C_CYAN}{plan_id}{C_RESET}"
        f"{C_GRAY}  pos:{C_RESET}{queue_pos}"
        f"  {C_GRAY}planned:{C_RESET}{format_number(planned_total)}"
        f"  {C_GRAY}done:{C_RESET}{format_number(patches)}"
    ) if plan_id else ""

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

    rows = [
        f"{header}  {state_str}",
        film_line,
    ]
    if plan_line:
        rows.append(plan_line)
    rows += [speed_line, patch_line, done_line]
    return rows


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


# ── Plan summary panel ────────────────────────────────────────────────────────

def _render_plan_summary_panel(state, term_width):
    """
    Plan summary block: total plan items vs done / running / pending / failed.

    Uses ``state["plan_summary"]`` (= GenerationPlan.get_global_stats()).
    """
    pw = min(term_width, 100)
    lines = []
    lines.append(_box_top(pw, " PLAN SUMMARY "))

    ps = state.get("plan_summary", {})
    if not ps:
        lines.append(_box_row(
            f"{C_GRAY}Plan not yet created (Phase 3 pending…){C_RESET}", pw
        ))
        lines.append(_box_bot(pw))
        return lines

    n_total   = ps.get("n_items_total",   0)
    n_done    = ps.get("n_items_done",    0)
    n_running = ps.get("n_items_running", 0)
    n_pending = ps.get("n_items_pending", 0)
    n_failed  = ps.get("n_items_failed",  0)

    planned_total   = ps.get("planned_total",   0)
    completed_total = ps.get("completed_total", 0)
    pct_overall = (completed_total / planned_total * 100) if planned_total > 0 else 0.0

    # ── Item counts row ──────────────────────────────────────────────────
    item_row = (
        f"{C_GRAY}Items:{C_RESET}  "
        f"{C_BOLD}{n_total}{C_RESET} total  "
        f"{C_GREEN}✔ {n_done} done{C_RESET}  "
        f"{C_CYAN}▶ {n_running} running{C_RESET}  "
        f"{C_GRAY}· {n_pending} pending{C_RESET}  "
        + (f"{C_RED}✖ {n_failed} failed{C_RESET}" if n_failed else "")
    )
    lines.append(_box_row(item_row, pw))

    # ── Global planned vs completed ──────────────────────────────────────
    inner_w = pw - 4
    label_w = 10
    count_w = 28
    bar_w   = max(8, inner_w - label_w - count_w - 2)

    global_row = _bar_row(
        "TOTAL", pct_overall, completed_total, planned_total,
        bar_w, C_GREEN, label_w
    )
    lines.append(_box_row(global_row, pw))

    # ── Per-category planned vs completed ────────────────────────────────
    planned_per_cat   = ps.get("planned_per_category",   {})
    completed_per_cat = ps.get("completed_per_category", {})
    for i, cat in enumerate(sorted(planned_per_cat)):
        pl  = planned_per_cat.get(cat, 0)
        co  = completed_per_cat.get(cat, 0)
        pct = (co / pl * 100) if pl > 0 else 0.0
        color = _category_color(cat, i)
        cat_row = _bar_row(cat, pct, co, pl, bar_w, color, label_w)
        lines.append(_box_row(cat_row, pw))

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
    """
    Production progress panel.

    Shows per-category overall progress bars, then for each category a
    per-format-template breakdown with planned vs completed, and inside each
    format a degradation-template breakdown.

    Layout per category:
        ████████░░  CATEGORY   12,345 / 80,000  (15.4%)
          ╰ fmt uhd_169  ████░░  5,000 / 40,000  (12.5%)
                   degrade: web_medium:60%  mpeg2:40%
          ╰ fmt hd_169   ████░░  7,345 / 40,000  (18.4%)
                   degrade: …
    """
    pw = min(term_width, 100)
    lines = []
    lines.append(_box_top(pw, " PRODUCTION PROGRESS "))

    cats       = state.get("categories", [])
    ovr        = state.get("overall_progress", {})
    patch_dist = state.get("patch_distribution", {})
    ps         = state.get("plan_summary", {})

    # Plan-level planned/completed per degradation template (global, all categories).
    plan_deg_planned   = ps.get("planned_per_degradation_template", {})
    plan_deg_completed = ps.get("completed_per_degradation_template", {})

    if not cats:
        lines.append(_box_row(f"{C_GRAY}No categories configured.{C_RESET}", pw))
        lines.append(_box_bot(pw))
        return lines

    inner_w  = pw - 4
    label_w  = max(8, min(14, len(max(cats, key=len))))
    count_w  = 28
    bar_w    = max(8, inner_w - label_w - count_w - 2)

    for i, cat in enumerate(cats):
        color = _category_color(cat, i)
        info  = ovr.get(cat, {})
        done  = int(info.get("created", 0)) if isinstance(info, dict) else 0
        tgt   = int(info.get("target",  0)) if isinstance(info, dict) else 0
        pct   = min(100.0, 100.0 * done / tgt) if tgt > 0 else 0.0

        # Category-level bar
        row = _bar_row(cat, pct, done, tgt, bar_w, color, label_w)
        lines.append(_box_row(row, pw))

        # ── Per-format-template sub-rows ──────────────────────────────────
        cat_fmts = patch_dist.get(cat, {})
        fmt_items = sorted(cat_fmts.items(), key=lambda kv: -kv[1].get("target", 0))
        for fi, (fmt_name, fmt_data) in enumerate(fmt_items):
            fmt_target    = fmt_data.get("target",  0)
            fmt_completed = fmt_data.get("count",   0)
            fmt_pct = (fmt_completed / fmt_target * 100) if fmt_target > 0 else 0.0

            # Indent + compact label
            fmt_color = _FMT_COLORS[fi % len(_FMT_COLORS)]
            fmt_lbl   = _trunc(fmt_name, 14)

            # Narrower bar for the format sub-row (indent of 4 chars)
            fmt_bar_w = max(4, bar_w - 4)
            fmt_bar   = make_bar(fmt_pct, fmt_bar_w, fmt_color)
            fmt_cnt   = (
                f"{C_BOLD}{format_number(fmt_completed):>7}{C_RESET}"
                f"{C_GRAY}/{format_number(fmt_target):>7}  ({fmt_pct:5.1f}%){C_RESET}"
            )
            fmt_row = (
                f"  {C_GRAY}╰{C_RESET} "
                f"{fmt_color}{fmt_lbl:<14}{C_RESET} "
                f"{fmt_bar} {fmt_cnt}"
            )
            lines.append(_box_row(fmt_row, pw))

            # ── Degradation breakdown for this format ─────────────────────
            # Show planned counts from the plan and completed from live data.
            deg_planned_for_fmt: dict = fmt_data.get("deg_planned", {})
            deg_completed_all:   dict = fmt_data.get("deg_completed", {})  # global

            if deg_planned_for_fmt:
                deg_total_planned = sum(deg_planned_for_fmt.values())
                sorted_deg = sorted(
                    deg_planned_for_fmt.items(), key=lambda kv: -kv[1]
                )
                parts = []
                for di, (dname, dplanned) in enumerate(sorted_deg):
                    dc_color  = _DEG_COLORS[di % len(_DEG_COLORS)]
                    dpct_plan = 100.0 * dplanned / deg_total_planned if deg_total_planned else 0
                    dcompleted = deg_completed_all.get(dname, 0)
                    parts.append(
                        f"{dc_color}{_trunc(dname, 16)}{C_RESET}"
                        f"{C_GRAY}:{format_number(dcompleted)}"
                        f"/{format_number(dplanned)} ({dpct_plan:.0f}%){C_RESET}"
                    )
                deg_row = f"    {C_GRAY}╰ degrade: {C_RESET}" + f"{C_GRAY} · {C_RESET}".join(parts)
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

    # ETA per category (plan-driven remaining / rate)
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

    # Global ETA from plan
    eta_total = eta_dict.get("total", 0)
    if isinstance(eta_total, (int, float)) and eta_total > 0:
        ps = state.get("plan_summary", {})
        planned = ps.get("planned_total", 0)
        completed = ps.get("completed_total", 0)
        remaining = max(0, planned - completed)
        lines.append(_box_row(
            f"{C_GRAY}Global ETA (plan):{C_RESET}  "
            f"{C_BOLD}{format_time(eta_total)}{C_RESET}  "
            f"{C_GRAY}remaining:{C_RESET} {format_number(remaining)} patches",
            pw,
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
      4. Plan summary  (total/done/running/pending/failed items + global bar)
      5. Production progress  (per-category + per-format + per-degradation)
      6. Performance & ETA  (plan-driven remaining work)

    All sections are boxed with Unicode box-drawing characters.

    Args:
        state: The ``ui_state`` dict from DatasetGeneratorV2UHD.  Expected keys:
            active_streams, n_active_streams, n_gpus_available,
            overall_progress, categories, eta, current_video_index,
            total_videos, output_format, patches_created_total,
            plan_summary, current_plan_items, patch_distribution.
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

    # 4. Plan summary
    out.extend(_render_plan_summary_panel(state, term_width))

    # 5. Production progress (per-category + per-format + per-degradation)
    out.extend(_render_production_panel(state, term_width))

    # 6. ETA
    out.extend(_render_eta_panel(state, term_width))

    sys.stdout.write(ANSI_CLEAR + ANSI_HOME)
    sys.stdout.write("\n".join(out))
    sys.stdout.write("\n")
    sys.stdout.flush()

