"""
Dataset Generator Terminal Display  –  Midnight Commander-style boxed TUI.

Layout (top → bottom)
─────────────────────
  Title bar
  GPU stream panels  (side-by-side when both fit in terminal, else stacked)
  Write-queue status (one line)
  Plan summary       (item counts + global bar + per-category bars)
  Production progress (per-category → per-format → per-degradation, vertical hierarchy)
  Performance & ETA

Width-safety rules
──────────────────
* Every panel is capped at ``term_width``.
* ``_box_row`` always truncates content to ``inner_w`` so no line can ever
  escape the box border.
* Side-by-side GPU panels are only used when each panel has at least 38 usable
  columns AND the combined pair fits inside ``term_width``.
* All content strings are explicitly truncated with ``_trunc`` before being
  passed to ``_box_row``.

Plan-driven fields (from ``ui_state``)
───────────────────────────────────────
  plan_summary       – GenerationPlan.get_global_stats()
  current_plan_items – per-stream {plan_item_id, queue_position, …}
  patch_distribution – {cat: {fmt: {count, target, deg_planned, deg_completed}}}
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

# Minimum usable inner width for a panel column in side-by-side mode.
_MIN_SIDE_INNER = 34
# Minimum terminal width required for side-by-side GPU panels (2 panels × min
# inner + 2×4 border/padding + 2 gap).
_MIN_SIDE_BY_SIDE = 2 * (_MIN_SIDE_INNER + 4) + 2


def _category_color(cat_key, index=0):
    return _KNOWN_COLORS.get(cat_key, _CAT_COLORS[index % len(_CAT_COLORS)])


def _category_display_name(cat_key):
    return cat_key.capitalize()


# ── Box-drawing primitives ────────────────────────────────────────────────────

_BC = C_CYAN   # box border colour


def _box_top(width, title="", color=_BC):
    """Top border ╔══ title ═══╗ (width = total visual chars including borders)."""
    inner = width - 2
    if title:
        t_plain = " " + title.strip() + " "
        tl = get_visible_len(t_plain)
        tl = min(tl, inner)          # never wider than the inner area
        left  = max(0, (inner - tl) // 2)
        right = max(0, inner - tl - left)
        return (
            f"{color}╔{'═' * left}"
            f"{C_BOLD}{t_plain[:tl]}{C_RESET}"
            f"{color}{'═' * right}╗{C_RESET}"
        )
    return f"{color}╔{'═' * inner}╗{C_RESET}"


def _box_bot(width, color=_BC):
    """Bottom border ╚══════╝"""
    return f"{color}╚{'═' * (width - 2)}╝{C_RESET}"


def _box_row(content, width, color=_BC, pad=1):
    """
    Content row ║ content … ║.

    The content is **always truncated** to fit exactly inside the box so that
    no rendered line can exceed *width* visible characters.
    """
    inner = width - 2 - 2 * pad
    inner = max(0, inner)
    vl = get_visible_len(content)
    if vl > inner:
        # Strip ANSI, truncate plain text, add ellipsis
        plain = ANSI_ESCAPE.sub('', content)
        content = plain[:max(0, inner - 1)] + '…'
        vl = get_visible_len(content)
    space = max(0, inner - vl)
    p = ' ' * pad
    return f"{color}║{C_RESET}{p}{content}{' ' * space}{p}{color}║{C_RESET}"


def _trunc(s, maxlen):
    """Truncate *s* to *maxlen* visible chars, appending '…' if needed."""
    maxlen = max(1, maxlen)
    if get_visible_len(s) <= maxlen:
        return s
    plain = ANSI_ESCAPE.sub('', s)
    return plain[:max(0, maxlen - 1)] + '…'


def _abbrev_num(n: int) -> str:
    """Abbreviated number for compact displays: 1_234_567 → '1.2M'."""
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _bar_row(label, pct, done, target, inner_w, color, label_w=10):
    """
    Build a progress-bar row that always fits within *inner_w* visible chars.

    Algorithm
    ---------
    1. Try the full count field  "  1,234,567 /  9,999,999  ( 55.5%)"
    2. If the bar would be too narrow (< 4 chars), fall back to
       abbreviated counts "1.2M/10M (55%)"
    3. As a last resort strip the bar entirely and show label + pct only.
    4. Hard-clip the final string to inner_w so the result *always* fits.
    """
    inner_w = max(8, inner_w)

    def _try(done_s, target_s, pct_fmt):
        cnt = f"  {done_s}/{target_s} ({pct_fmt})"
        cnt_w = len(cnt)
        bw = inner_w - label_w - 1 - cnt_w   # 1 for the space before bar
        return cnt, cnt_w, bw

    # Full format
    done_f   = format_number(done)
    target_f = format_number(target)
    pct_f    = f"{pct:5.1f}%"
    cnt, cnt_w, bar_w = _try(done_f, target_f, pct_f)

    if bar_w < 4:
        # Abbreviated format
        done_a, target_a, pct_a = _abbrev_num(done), _abbrev_num(target), f"{pct:.0f}%"
        cnt, cnt_w, bar_w = _try(done_a, target_a, pct_a)

    lbl = _trunc(label, label_w)
    lbl_padded = f"{color}{lbl:<{label_w}}{C_RESET}"
    cnt_colored = f"{C_GRAY}{cnt}{C_RESET}"

    if bar_w >= 4:
        row = f"{lbl_padded} {make_bar(pct, bar_w, color)}{cnt_colored}"
    else:
        # No room for bar – just label + pct
        pct_short = f" ({pct:.0f}%)"
        row = f"{lbl_padded}{C_GRAY}{pct_short}{C_RESET}"

    # Hard-clip (safety net)
    vl = get_visible_len(row)
    if vl > inner_w:
        plain = ANSI_ESCAPE.sub('', row)
        row = plain[:max(1, inner_w - 1)] + '…'

    return row



# ── GPU stream panel ──────────────────────────────────────────────────────────

def _gpu_panel_lines(stream, inner_w):
    """
    Build the content lines for one GPU/stream panel.

    Every line is truncated to *inner_w* before being returned so that callers
    can pass these strings directly to ``_box_row`` without any further width
    checks.

    Args:
        stream:  stream state dict from ``ui_state["active_streams"]``.
        inner_w: available visible characters inside the box (width - 4).
    """
    gpu_idx  = stream.get("gpu_index", -1)
    gpu_name = stream.get("gpu_name", "GPU")
    state    = stream.get("state", "idle")
    video    = stream.get("video_name", "—")
    fps      = stream.get("live_fps", 0.0)
    patches  = stream.get("patches_created", 0)
    wq       = stream.get("write_queue_depth", 0)
    pipeline = stream.get("pipeline", "libplacebo")
    n_done   = stream.get("n_videos_done", 0)

    plan_id       = stream.get("plan_item_id", "")
    queue_pos     = stream.get("queue_position", 0)
    planned_total = stream.get("planned_total", 0)

    if state == "running":
        sc, sl = C_GREEN, "▶ running"
    elif state == "error":
        sc, sl = C_RED, "✖ error"
    else:
        sc, sl = C_GRAY, "· idle"

    gpu_label = f"GPU {gpu_idx}" if gpu_idx >= 0 else "CPU"

    def _t(s):
        return _trunc(s, inner_w)

    header = _t(
        f"{C_BOLD}{C_CYAN}{gpu_label}{C_RESET}  "
        f"{C_GRAY}{gpu_name}{C_RESET}  {sc}{sl}{C_RESET}"
    )
    film_line = _t(
        f"{C_GRAY}Film :{C_RESET} {C_WHITE}{_trunc(video, inner_w - 8)}{C_RESET}"
    )

    rows = [header, film_line]

    # Plan item line (only when there is active plan data)
    if plan_id:
        # Show ID in short form: first 8 chars
        short_id = plan_id[:8]
        plan_line = _t(
            f"{C_GRAY}Plan :{C_RESET} "
            f"{C_CYAN}#{short_id}{C_RESET}"
            f"{C_GRAY} pos:{C_RESET}{queue_pos}"
            f"  {C_GRAY}plan:{C_RESET}{format_number(planned_total)}"
            f"  {C_GRAY}done:{C_RESET}{format_number(patches)}"
        )
        rows.append(plan_line)

    fps_str = f"{fps:6.1f} fps" if state == "running" else "     — fps"
    rows.append(_t(
        f"{C_GRAY}Speed:{C_RESET} {C_GREEN if fps > 0 else C_GRAY}{fps_str}{C_RESET}"
        f"  {C_GRAY}{pipeline}{C_RESET}"
    ))
    rows.append(_t(
        f"{C_GRAY}Patch:{C_RESET} {C_BOLD}{format_number(patches):>7}{C_RESET} this film"
        f"  {C_GRAY}WQ:{C_RESET}{wq}"
    ))
    rows.append(_t(
        f"{C_GRAY}Done :{C_RESET} {n_done} film(s) this session"
    ))
    return rows


def _render_gpu_panels(streams, term_width):
    """
    Render GPU/stream panels.

    Layout decision:
    - Two panels side-by-side when ``term_width >= _MIN_SIDE_BY_SIDE`` **and**
      there are at least two streams.
    - Otherwise panels are stacked vertically.

    This is determined once per call so there is never a layout that exceeds
    the terminal width.
    """
    lines_out = []

    if not streams:
        pw = min(term_width, 80)
        lines_out.append(_box_top(pw, "STREAMS – awaiting start"))
        lines_out.append(_box_row(f"{C_GRAY}No streams active yet.{C_RESET}", pw))
        lines_out.append(_box_bot(pw))
        return lines_out

    use_side_by_side = (term_width >= _MIN_SIDE_BY_SIDE) and (len(streams) >= 2)

    if use_side_by_side:
        # Each panel gets exactly half the terminal minus 1 for the gap character.
        # (gap = 1 space between two panels)
        pw = (term_width - 1) // 2
        for pair_start in range(0, len(streams), 2):
            left  = streams[pair_start]
            right = streams[pair_start + 1] if pair_start + 1 < len(streams) else None

            left_title  = f"GPU {left.get('gpu_index', '?')} · {_trunc(left.get('gpu_name', ''), 20)}"
            right_title = (
                f"GPU {right.get('gpu_index', '?')} · {_trunc(right.get('gpu_name', ''), 20)}"
                if right else ""
            )

            lines_out.append(
                _box_top(pw, left_title) + " "
                + (_box_top(pw, right_title) if right else "")
            )

            inner_w = pw - 4  # 2 borders + 2×1 padding
            left_rows  = _gpu_panel_lines(left, inner_w)
            right_rows = _gpu_panel_lines(right, inner_w) if right else []
            n_rows = max(len(left_rows), len(right_rows))

            for r in range(n_rows):
                lr = left_rows[r]  if r < len(left_rows)  else ""
                rr = right_rows[r] if r < len(right_rows) else ""
                lines_out.append(
                    _box_row(lr, pw) + " " + (_box_row(rr, pw) if right else "")
                )

            lines_out.append(
                _box_bot(pw) + " " + (_box_bot(pw) if right else "")
            )
    else:
        # Stacked – single column, full width (capped at 100 for readability)
        pw = min(term_width, 100)
        for stream in streams:
            idx   = stream.get("gpu_index", -1)
            title = f"GPU {idx} · {_trunc(stream.get('gpu_name', ''), 30)}"
            lines_out.append(_box_top(pw, title))
            inner_w = pw - 4
            for row in _gpu_panel_lines(stream, inner_w):
                lines_out.append(_box_row(row, pw))
            lines_out.append(_box_bot(pw))

    return lines_out


# ── Write-queue panel ─────────────────────────────────────────────────────────

def _render_write_queue_panel(state, term_width):
    pw = min(term_width, 100)
    streams  = state.get("active_streams", [])
    wq_total = sum(s.get("write_queue_depth", 0) for s in streams)
    fmt      = state.get("output_format", "BMP")
    n_active = state.get("n_active_streams", 0)
    fc = C_GREEN if fmt == "BMP" else C_YELLOW

    bp_str = (
        f"{C_RED}HIGH{C_RESET}"   if wq_total > 200 else
        f"{C_YELLOW}mod{C_RESET}" if wq_total >  80 else
        f"{C_GREEN}ok{C_RESET}"
    )
    content = (
        f"{C_GRAY}Writers:{C_RESET} {C_BOLD}{n_active}{C_RESET} active"
        f"   {C_GRAY}Queue:{C_RESET} {C_BOLD}{wq_total}{C_RESET}"
        f"   {C_GRAY}Format:{C_RESET} {fc}{C_BOLD}{fmt}{C_RESET}"
        f"   {C_GRAY}Backpressure:{C_RESET} {bp_str}"
    )
    return [
        _box_top(pw, "WRITE QUEUE"),
        _box_row(content, pw),
        _box_bot(pw),
    ]


# ── Plan summary panel ────────────────────────────────────────────────────────

def _render_plan_summary_panel(state, term_width):
    """
    Plan summary: item counts + global progress bar + per-category bars.

    All rows are computed to fit within *term_width*.
    """
    pw = min(term_width, 100)
    inner_w = pw - 4
    lines = [_box_top(pw, "PLAN SUMMARY")]

    ps = state.get("plan_summary", {})
    if not ps:
        lines.append(_box_row(f"{C_GRAY}Plan not yet created (Phase 3 pending…){C_RESET}", pw))
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

    # ── Item count row ────────────────────────────────────────────────────
    item_row = (
        f"{C_GRAY}Items:{C_RESET} {C_BOLD}{n_total}{C_RESET}"
        f"  {C_GREEN}✔{n_done}{C_RESET}"
        f"  {C_CYAN}▶{n_running}{C_RESET}"
        f"  {C_GRAY}·{n_pending}{C_RESET}"
        + (f"  {C_RED}✖{n_failed}{C_RESET}" if n_failed else "")
    )
    lines.append(_box_row(item_row, pw))

    # ── Global bar ────────────────────────────────────────────────────────
    label_w = 10
    lines.append(_box_row(
        _bar_row("TOTAL", pct_overall, completed_total, planned_total, inner_w, C_GREEN, label_w),
        pw,
    ))

    # ── Per-category bars ─────────────────────────────────────────────────
    for i, cat in enumerate(sorted(ps.get("planned_per_category", {}))):
        pl = ps["planned_per_category"].get(cat, 0)
        co = ps.get("completed_per_category", {}).get(cat, 0)
        pct = (co / pl * 100) if pl > 0 else 0.0
        lines.append(_box_row(
            _bar_row(cat, pct, co, pl, inner_w, _category_color(cat, i), label_w),
            pw,
        ))

    lines.append(_box_bot(pw))
    return lines


# ── Production progress panel ─────────────────────────────────────────────────

def _render_production_panel(state, term_width):
    """
    Production progress panel with a readable three-level hierarchy:

        category                    ████████░  12,345 / 80,000  (15.4%)
          ╰ format_template_A       ████░░     5,000 / 40,000  (12.5%)
              ╰ degrade_template_X  3,000 / 25,000  (40%)
              ╰ degrade_template_Y  2,000 / 15,000  (60%)
          ╰ format_template_B       ████░░     7,345 / 40,000  (18.4%)
              ╰ degrade_template_X  …

    Every line is constrained to ``term_width`` through ``_box_row``.
    Degradation entries are shown one per line (vertical) instead of one
    long horizontal string to avoid overflow.
    """
    pw = min(term_width, 100)
    inner_w = pw - 4
    lines = [_box_top(pw, "PRODUCTION PROGRESS")]

    cats       = state.get("categories", [])
    ovr        = state.get("overall_progress", {})
    patch_dist = state.get("patch_distribution", {})

    if not cats:
        lines.append(_box_row(f"{C_GRAY}No categories configured.{C_RESET}", pw))
        lines.append(_box_bot(pw))
        return lines

    cat_label_w = max(6, min(12, max(len(c) for c in cats)))

    for i, cat in enumerate(cats):
        color = _category_color(cat, i)
        info  = ovr.get(cat, {})
        done  = int(info.get("created", 0)) if isinstance(info, dict) else 0
        tgt   = int(info.get("target",  0)) if isinstance(info, dict) else 0
        pct   = min(100.0, 100.0 * done / tgt) if tgt > 0 else 0.0

        # Category-level bar (full inner_w)
        lines.append(_box_row(
            _bar_row(cat, pct, done, tgt, inner_w, color, cat_label_w),
            pw,
        ))

        # ── Format-template sub-rows (indented 2 chars) ───────────────────
        cat_fmts  = patch_dist.get(cat, {})
        fmt_items = sorted(cat_fmts.items(), key=lambda kv: -kv[1].get("target", 0))

        for fi, (fmt_name, fmt_data) in enumerate(fmt_items):
            fmt_target    = fmt_data.get("target",  0)
            fmt_completed = fmt_data.get("count",   0)
            fmt_pct = (fmt_completed / fmt_target * 100) if fmt_target > 0 else 0.0
            fc = _FMT_COLORS[fi % len(_FMT_COLORS)]

            # Indent 2: "  ╰ "  = 4 visible chars
            fmt_indent   = 4
            fmt_label_w  = max(6, min(16, inner_w // 4))
            fmt_inner_w  = inner_w - fmt_indent

            fmt_bar_row = (
                f"  {C_GRAY}╰{C_RESET} "
                + _bar_row(
                    _trunc(fmt_name, fmt_label_w),
                    fmt_pct, fmt_completed, fmt_target,
                    fmt_inner_w, fc, fmt_label_w,
                )
            )
            lines.append(_box_row(fmt_bar_row, pw))

            # ── Degradation sub-rows (indented 6 chars, one per template) ─
            deg_planned_for_fmt: dict = fmt_data.get("deg_planned", {})
            deg_completed_all:   dict = fmt_data.get("deg_completed", {})

            if deg_planned_for_fmt:
                deg_total_planned = sum(deg_planned_for_fmt.values())
                sorted_deg = sorted(deg_planned_for_fmt.items(), key=lambda kv: -kv[1])

                for di, (dname, dplanned) in enumerate(sorted_deg):
                    dc_color  = _DEG_COLORS[di % len(_DEG_COLORS)]
                    dpct_plan = (100.0 * dplanned / deg_total_planned
                                 if deg_total_planned else 0.0)
                    dcompleted = deg_completed_all.get(dname, 0)
                    # Short display: "      ╰ dname_trunc  done/planned (pct%)"
                    d_lbl  = _trunc(dname, 18)
                    d_line = (
                        f"      {C_GRAY}╰{C_RESET} "
                        f"{dc_color}{d_lbl}{C_RESET}"
                        f"  {C_GRAY}{format_number(dcompleted)}"
                        f"/{format_number(dplanned)}"
                        f" ({dpct_plan:.0f}%){C_RESET}"
                    )
                    lines.append(_box_row(d_line, pw))

    lines.append(_box_bot(pw))
    return lines


# ── ETA / performance panel ───────────────────────────────────────────────────

def _render_eta_panel(state, term_width):
    pw = min(term_width, 100)
    lines = [_box_top(pw, "PERFORMANCE & ETA")]

    streams    = state.get("active_streams", [])
    total_fps  = sum(s.get("live_fps", 0.0) for s in streams)
    n_active   = max(1, state.get("n_active_streams", 0))
    vid_idx    = state.get("current_video_index", 0)
    total_vids = state.get("total_videos", 0)
    output_fmt = state.get("output_format", "BMP")
    fc         = C_GREEN if output_fmt == "BMP" else C_YELLOW

    lines.append(_box_row(
        f"{C_GRAY}Videos:{C_RESET} {vid_idx}/{total_vids}"
        f"   {C_GRAY}Streams:{C_RESET} {C_BOLD}{n_active}{C_RESET}"
        f"   {C_GRAY}FPS:{C_RESET} {C_BOLD}{total_fps:.1f}{C_RESET}"
        f"   {C_GRAY}Output:{C_RESET} {fc}{C_BOLD}{output_fmt}{C_RESET}",
        pw,
    ))

    # ETA per category
    eta_dict = state.get("eta", {})
    cats     = state.get("categories", [])
    eta_parts = []
    for i, cat in enumerate(cats):
        color = _category_color(cat, i)
        eta_v = eta_dict.get(cat, "—")
        eta_s = format_time(eta_v) if isinstance(eta_v, (int, float)) and eta_v > 0 else "—"
        eta_parts.append(f"{color}{_category_display_name(cat)}{C_RESET}{C_GRAY}:{C_RESET}{eta_s}")

    if eta_parts:
        lines.append(_box_row(
            f"{C_GRAY}ETA:  {C_RESET}" + "  │  ".join(eta_parts),
            pw,
        ))

    # Global plan-based ETA
    eta_total = eta_dict.get("total", 0)
    ps = state.get("plan_summary", {})
    if isinstance(eta_total, (int, float)) and eta_total > 0 and ps:
        planned   = ps.get("planned_total", 0)
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
        f"{C_BOLD}DATASET GENERATOR V2{C_RESET}"
        f"  {C_GRAY}·{C_RESET}  Multi-Stream"
        f"  {C_GRAY}·{C_RESET}  {C_GREEN}libplacebo{C_RESET}"
        f"  {C_GRAY}·{C_RESET}  {C_YELLOW}BMP{C_RESET}/{C_CYAN}PNG{C_RESET}"
    )
    # Centre inside box
    inner     = pw - 2
    title_vis = get_visible_len(title)
    left_pad  = max(0, (inner - title_vis) // 2)
    right_pad = max(0, inner - title_vis - left_pad)
    return [
        f"{C_CYAN}╔{'═' * inner}╗{C_RESET}",
        f"{C_CYAN}║{C_RESET}{' ' * left_pad}{title}{' ' * right_pad}{C_CYAN}║{C_RESET}",
        f"{C_CYAN}╚{'═' * inner}╝{C_RESET}",
    ]


# ── Main entry point ──────────────────────────────────────────────────────────

def draw_dataset_ui(state):
    """
    Draw the complete dataset-generation dashboard.

    Called from ``_update_terminal_ui()`` (heartbeat thread) and from
    ``run()`` once right before execution starts.  **Must not be called**
    during Phases 1–3 (scan / distribution / planning) because the plan does
    not yet exist and the GUI cannot show accurate planned-vs-completed data.

    All panels obey the current terminal width: no line ever escapes its box.

    Args:
        state: ``ui_state`` dict from DatasetGeneratorV2UHD.
    """
    term_width, _ = shutil.get_terminal_size((100, 50))
    # Never render wider than the actual terminal
    term_width = max(40, term_width)

    out = []
    out.extend(_render_title(term_width))
    out.extend(_render_gpu_panels(state.get("active_streams", []), term_width))
    out.extend(_render_write_queue_panel(state, term_width))
    out.extend(_render_plan_summary_panel(state, term_width))
    out.extend(_render_production_panel(state, term_width))
    out.extend(_render_eta_panel(state, term_width))

    sys.stdout.write(ANSI_CLEAR + ANSI_HOME)
    sys.stdout.write("\n".join(out))
    sys.stdout.write("\n")
    sys.stdout.flush()
