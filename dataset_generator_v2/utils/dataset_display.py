"""
Dataset Generator Terminal Display  –  Midnight Commander-style boxed TUI.

Layout (top → bottom)
─────────────────────
  Title bar
  GPU stream panels        (side-by-side when both fit in terminal, else stacked)
  Current film panel       (plan item detail for each running stream)
  Pipeline  │  Plan Summary     (side-by-side when terminal is wide enough)
  Production Progress  │  Performance & ETA   (side-by-side when wide enough)

Width-safety rules
──────────────────
* Every panel is capped at ``term_width``.
* ``_box_row`` always truncates content to ``inner_w`` – no line can ever
  escape the box border.
* Side-by-side GPU panels are only used when each panel has at least
  _MIN_SIDE_INNER usable columns AND the combined pair fits inside term_width.
* All content strings are explicitly truncated with ``_trunc`` before being
  passed to ``_box_row``.

Flicker-free rendering
──────────────────────
* draw_dataset_ui() uses cursor home (ANSI_HOME) instead of full clear.
* Each line is padded to term_width visible chars so it fully overwrites the
  previous content at that position.
* After all rendered lines, \033[J erases any stale lines from a previous
  taller render without causing a visible full-screen flash.
* On terminal resize (SIGWINCH or detected size change) a full ANSI clear is
  issued before the next redraw to eliminate leftover artefacts.

Resize handling
───────────────
* register_resize_handler() installs a SIGWINCH handler (Unix only) that sets
  a flag causing the next draw_dataset_ui() call to do a full clear.  Call it
  from the main thread alongside other signal.signal() registrations.
* draw_dataset_ui() also compares terminal dimensions between renders; any
  change triggers the same full-clear path.

Plan-driven fields (from ``ui_state``)
───────────────────────────────────────
  plan_summary        – GenerationPlan.get_global_stats()
  current_plan_items  – per-stream {plan_item_id, queue_position, …}
  patch_distribution  – {cat: {fmt: {count, target, deg_planned, deg_completed}}}
  live_sps            – global patches/second
  wq_capacity         – write-queue max capacity (int)
  total_streams       – total configured stream count
"""

import signal as _signal_mod
import sys
import shutil

from .terminal_ui import (
    ANSI_ESCAPE, C_RESET, C_BOLD, C_CYAN, C_GRAY, C_SILVER, C_GREEN, C_RED,
    C_YELLOW, C_MAGENTA, C_BLUE, C_WHITE,
    ANSI_HOME, ANSI_CLEAR,
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

# Minimum usable inner width for one column in side-by-side mode.
_MIN_SIDE_INNER = 34
# Minimum terminal width for side-by-side GPU panels and two-column layouts.
_MIN_SIDE_BY_SIDE = 2 * (_MIN_SIDE_INNER + 4) + 2

# Track how many lines the previous render used so we can erase stale lines.
_prev_line_count: int = 0

# Resize-tracking state
_prev_term_size: tuple = (0, 0)
_needs_clear: bool = False
_sigwinch_registered: bool = False


def _sigwinch_handler(signum, frame):
    """SIGWINCH handler – mark that a full screen clear is needed."""
    global _needs_clear
    _needs_clear = True


def register_resize_handler():
    """
    Install a SIGWINCH handler (Unix only) so terminal resizes trigger a full
    clear before the next redraw.  Must be called from the **main thread**
    alongside other ``signal.signal()`` registrations.
    """
    global _sigwinch_registered
    if not _sigwinch_registered:
        if hasattr(_signal_mod, 'SIGWINCH'):
            _signal_mod.signal(_signal_mod.SIGWINCH, _sigwinch_handler)
        _sigwinch_registered = True


def _category_color(cat_key, index=0):
    return _KNOWN_COLORS.get(cat_key, _CAT_COLORS[index % len(_CAT_COLORS)])


def _category_display_name(cat_key):
    return cat_key.capitalize()


# ── Box-drawing primitives ────────────────────────────────────────────────────

_BC = C_CYAN   # box border colour


def _box_top(width, title="", color=_BC):
    """Top border ╔══ title ═══╗"""
    inner = width - 2
    if title:
        t_plain = " " + title.strip() + " "
        tl = min(get_visible_len(t_plain), inner)
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


def _box_div(width, color=_BC):
    """Inner divider ╠══════╣"""
    return f"{color}╠{'═' * (width - 2)}╣{C_RESET}"


def _box_row(content, width, color=_BC, pad=1):
    """
    Content row ║ content … ║.
    Always truncates to inner width so no line can overflow the box.
    """
    inner = max(0, width - 2 - 2 * pad)
    vl = get_visible_len(content)
    if vl > inner:
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
    """Compact number: 1_234_567 → '1.2M'."""
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _bar_row(label, pct, done, target, inner_w, color, label_w=10):
    """
    Progress-bar row that always fits within *inner_w* visible chars.

    Falls back from full count format → abbreviated → label+pct only.
    Hard-clips the final string as a safety net.
    """
    inner_w = max(8, inner_w)

    def _try(done_s, target_s, pct_fmt):
        cnt = f"  {done_s}/{target_s} ({pct_fmt})"
        bw = inner_w - label_w - 1 - len(cnt)
        return cnt, bw

    done_f, target_f, pct_f = format_number(done), format_number(target), f"{pct:5.1f}%"
    cnt, bar_w = _try(done_f, target_f, pct_f)

    if bar_w < 4:
        done_a, target_a = _abbrev_num(done), _abbrev_num(target)
        cnt, bar_w = _try(done_a, target_a, f"{pct:.0f}%")

    lbl = _trunc(label, label_w)
    lbl_p = f"{color}{lbl:<{label_w}}{C_RESET}"
    cnt_c = f"{C_SILVER}{cnt}{C_RESET}"

    if bar_w >= 4:
        row = f"{lbl_p} {make_bar(pct, bar_w, color)}{cnt_c}"
    else:
        row = f"{lbl_p}{C_SILVER} ({pct:.0f}%){C_RESET}"

    vl = get_visible_len(row)
    if vl > inner_w:
        plain = ANSI_ESCAPE.sub('', row)
        row = plain[:max(1, inner_w - 1)] + '…'
    return row


# ── Two-column layout helper ──────────────────────────────────────────────────

def _render_panels_side_by_side(left_lines, right_lines, pw):
    """
    Interleave two lists of pre-rendered panel lines into a side-by-side view.

    Both panels should have been rendered with width *pw*.  Missing rows on the
    shorter side are replaced by blank lines of length *pw*.
    """
    height = max(len(left_lines), len(right_lines))
    empty = ' ' * pw
    result = []
    for i in range(height):
        l = left_lines[i] if i < len(left_lines) else empty
        r = right_lines[i] if i < len(right_lines) else empty
        result.append(l + ' ' + r)
    return result


# ── GPU stream panel ──────────────────────────────────────────────────────────

def _gpu_panel_lines(stream, inner_w):
    """
    Content lines for one GPU/stream panel – every line truncated to inner_w.
    """
    gpu_idx         = stream.get("gpu_index", -1)
    gpu_name        = stream.get("gpu_name", "GPU")
    state           = stream.get("state", "idle")
    video           = stream.get("video_name", "—")
    fps             = stream.get("live_fps", 0.0)
    sps             = stream.get("live_sps", 0.0)
    patches         = stream.get("patches_created", 0)
    wq              = stream.get("write_queue_depth", 0)
    pq              = stream.get("proc_queue_size", 0)
    n_w_active      = stream.get("n_workers_active", 0)
    n_w_total       = stream.get("n_workers_total", 0)
    pipeline        = stream.get("pipeline", "libplacebo")
    is_sw_vk        = stream.get("is_software_vulkan", False)
    n_done          = stream.get("n_videos_done", 0)
    plan_id         = stream.get("plan_item_id", "")
    queue_pos       = stream.get("queue_position", 0)
    planned_total   = stream.get("planned_total", 0)

    if state == "running":
        sc, sl = C_GREEN, "▶ running"
    elif state == "error":
        sc, sl = C_RED, "✖ error"
    else:
        sc, sl = C_SILVER, "· idle"

    # GPU label: show Vulkan index, or "auto" when FFmpeg picks any device.
    if gpu_idx >= 0:
        gpu_label = f"GPU {gpu_idx}"
    else:
        gpu_label = "GPU auto"

    # Software-Vulkan warning badge.
    _sw_badge = f"  {C_YELLOW}⚠SW{C_RESET}" if is_sw_vk else ""

    def _t(s):
        return _trunc(s, inner_w)

    rows = [
        _t(f"{C_BOLD}{C_CYAN}{gpu_label}{C_RESET}  {C_SILVER}{gpu_name}{C_RESET}"
           f"{_sw_badge}  {sc}{sl}{C_RESET}"),
        _t(f"{C_SILVER}Film :{C_RESET} {C_WHITE}{_trunc(video, inner_w - 8)}{C_RESET}"),
    ]

    if plan_id:
        short_id = plan_id[:8]
        remaining = max(0, planned_total - patches)
        rows.append(_t(
            f"{C_SILVER}Plan :{C_RESET} {C_CYAN}#{short_id}{C_RESET}"
            f"{C_SILVER} pos:{C_RESET}{queue_pos}"
            f"  {C_SILVER}plan:{C_RESET}{_abbrev_num(planned_total)}"
            f"  {C_SILVER}done:{C_RESET}{_abbrev_num(patches)}"
            f"  {C_SILVER}rem:{C_RESET}{_abbrev_num(remaining)}"
        ))

    fps_str = f"{fps:6.1f}" if state == "running" else "     —"
    sps_str = f"{sps:6.1f}" if sps > 0 else "     —"
    rows.append(_t(
        f"{C_SILVER}Input:{C_RESET} {C_GREEN if fps > 0 else C_SILVER}{fps_str}{C_RESET} fps"
        f"  {C_SILVER}SPS:{C_RESET} {C_GREEN if sps > 0 else C_SILVER}{sps_str}{C_RESET}"
        f"  {C_SILVER}{pipeline}{C_RESET}"
    ))

    # Processing queue + worker utilisation row.
    if n_w_total > 0:
        _wu_pct = int(n_w_active / n_w_total * 100)
        _wu_col = C_RED if _wu_pct >= 90 else (C_GREEN if _wu_pct >= 20 else C_YELLOW)
        rows.append(_t(
            f"{C_SILVER}ProcQ:{C_RESET} {C_CYAN}{pq:>3}{C_RESET}/32"
            f"  {C_SILVER}Workers:{C_RESET} {_wu_col}{n_w_active}/{n_w_total}{C_RESET}"
            f"  {C_SILVER}WQ:{C_RESET}{wq}"
        ))
    else:
        rows.append(_t(
            f"{C_SILVER}Patch:{C_RESET} {C_BOLD}{_abbrev_num(patches):>7}{C_RESET} this film"
            f"  {C_SILVER}ProcQ:{C_RESET}{pq}"
            f"  {C_SILVER}WQ:{C_RESET}{wq}"
        ))

    rows.append(_t(f"{C_SILVER}Done :{C_RESET} {n_done} film(s) this session"))
    return rows


def _render_gpu_panels(streams, term_width):
    lines_out = []

    if not streams:
        pw = min(term_width, 80)
        lines_out += [
            _box_top(pw, "STREAMS – awaiting start"),
            _box_row(f"{C_SILVER}No streams active yet.{C_RESET}", pw),
            _box_bot(pw),
        ]
        return lines_out

    use_side_by_side = (term_width >= _MIN_SIDE_BY_SIDE) and (len(streams) >= 2)

    def _panel_title(s):
        idx = s.get("gpu_index", -1)
        label = f"GPU {idx}" if idx >= 0 else "GPU auto"
        sw_tag = " ⚠SW" if s.get("is_software_vulkan") else ""
        return f"{label}{sw_tag} · {_trunc(s.get('gpu_name',''), 20)}"

    if use_side_by_side:
        pw = (term_width - 1) // 2
        for pair_start in range(0, len(streams), 2):
            left  = streams[pair_start]
            right = streams[pair_start + 1] if pair_start + 1 < len(streams) else None

            lt = _panel_title(left)
            rt = _panel_title(right) if right else ""
            lines_out.append(_box_top(pw, lt) + " " + (_box_top(pw, rt) if right else ""))

            inner_w = pw - 4
            lr = _gpu_panel_lines(left, inner_w)
            rr = _gpu_panel_lines(right, inner_w) if right else []
            for r in range(max(len(lr), len(rr))):
                l_row = lr[r] if r < len(lr) else ""
                r_row = rr[r] if r < len(rr) else ""
                lines_out.append(
                    _box_row(l_row, pw) + " " + (_box_row(r_row, pw) if right else "")
                )
            lines_out.append(_box_bot(pw) + " " + (_box_bot(pw) if right else ""))
    else:
        pw = min(term_width, 100)
        for stream in streams:
            lines_out.append(_box_top(pw, _panel_title(stream)))
            for row in _gpu_panel_lines(stream, pw - 4):
                lines_out.append(_box_row(row, pw))
            lines_out.append(_box_bot(pw))

    return lines_out


# ── Current-film detail panel ─────────────────────────────────────────────────

def _render_current_film_panel(state, term_width):
    """
    Dedicated sub-panel for the currently running film(s).

    Shows per-stream: plan item ID, overall progress bar, and a breakdown
    by category / format / degradation.  Basic stats (pos, planned, done,
    remaining) are already visible in the GPU panel above so they are not
    repeated here.
    """
    pw = min(term_width, 100)
    inner_w = pw - 4

    streams = [s for s in state.get("active_streams", []) if s.get("state") == "running"]
    if not streams:
        return []

    lines = [_box_top(pw, "CURRENT FILM STATUS")]

    for si, ss in enumerate(streams):
        if si > 0:
            lines.append(_box_div(pw))

        plan_id   = ss.get("plan_item_id", "")
        video     = _trunc(ss.get("video_name", "—"), inner_w - 10)
        planned   = ss.get("planned_total", 0)
        done      = ss.get("patches_created", 0)
        pct       = min(100.0, 100.0 * done / planned) if planned > 0 else 0.0

        lines.append(_box_row(
            f"{C_BOLD}{C_WHITE}{video}{C_RESET}"
            + (f"  {C_SILVER}#{plan_id[:12]}{C_RESET}" if plan_id else ""),
            pw,
        ))
        # Overall bar for this film
        lines.append(_box_row(
            _bar_row("  Film", pct, done, planned, inner_w, C_GREEN, 7),
            pw,
        ))

        # Per-category breakdown (in-flight from live stream state)
        per_cat = ss.get("patches_per_category", {})
        planned_per_cat = ss.get("planned_per_category", {})
        if per_cat or planned_per_cat:
            all_cats = sorted(set(list(per_cat.keys()) + list(planned_per_cat.keys())))
            for ci, cat in enumerate(all_cats):
                cat_done    = per_cat.get(cat, 0)
                cat_planned = planned_per_cat.get(cat, planned)  # fallback to film total
                cat_pct     = min(100.0, 100.0 * cat_done / cat_planned) if cat_planned > 0 else 0.0
                col = _category_color(cat, ci)
                lines.append(_box_row(
                    _bar_row(f"    {cat}", cat_pct, cat_done, cat_planned,
                             inner_w, col, max(8, min(14, len(cat) + 4))),
                    pw,
                ))

        # Per-format planned (static from plan; live counts only available after film)
        planned_per_fmt = ss.get("planned_per_format", {})
        if planned_per_fmt and inner_w > 40:
            for fi, (fmt_name, fmt_planned) in enumerate(
                sorted(planned_per_fmt.items(), key=lambda kv: -kv[1])[:4]
            ):
                fc = _FMT_COLORS[fi % len(_FMT_COLORS)]
                fmt_lbl = _trunc(f"      ╰ {fmt_name}", 22)
                lines.append(_box_row(
                    f"{C_SILVER}{fmt_lbl}{C_RESET}"
                    f"  {C_SILVER}plan:{C_RESET} {fc}{_abbrev_num(fmt_planned)}{C_RESET}",
                    pw,
                ))

    lines.append(_box_bot(pw))
    return lines


# ── Write-queue / writer diagnostics panel ────────────────────────────────────

def _render_write_queue_panel(state, term_width):
    """
    Pipeline diagnostics panel: processing queue, workers, write queue.

    Shows the full pipeline state in one place so bottlenecks are visible:
      FFmpeg → [proc_queue] → processing workers → [write_queue] → disk
    """
    pw = min(term_width, 100)
    inner_w = pw - 4
    streams   = state.get("active_streams", [])
    wq_total  = sum(s.get("write_queue_depth", 0) for s in streams)
    wq_cap    = max(1, state.get("wq_capacity", 500))
    wq_pct    = min(100.0, 100.0 * wq_total / wq_cap)
    fmt       = state.get("output_format", "BMP")
    n_active  = state.get("n_active_streams", 0)
    n_total   = max(n_active, state.get("total_streams", len(streams)))
    n_idle    = max(0, n_total - n_active)
    live_sps  = state.get("live_sps", 0.0)
    fc = C_GREEN if fmt == "BMP" else C_YELLOW

    # Aggregated processing-queue and worker stats across all active streams.
    pq_total       = sum(s.get("proc_queue_size", 0) for s in streams)
    nwa_total      = sum(s.get("n_workers_active", 0) for s in streams)
    nwt_total      = sum(s.get("n_workers_total",  0) for s in streams)

    # Backpressure colour and label for write queue
    if wq_pct >= 80:
        bp_color, bp_label = C_RED,    "HIGH"
    elif wq_pct >= 40:
        bp_color, bp_label = C_YELLOW, "moderate"
    else:
        bp_color, bp_label = C_GREEN,  "ok"

    lines = [_box_top(pw, "PIPELINE: FFmpeg → ProcQ → Workers → WriteQ → Disk")]

    # Row 1: processing queue
    pq_pct = min(100.0, 100.0 * pq_total / 32) if 32 > 0 else 0.0
    pq_color = C_RED if pq_pct >= 90 else (C_YELLOW if pq_pct >= 50 else C_GREEN)
    pq_bar_w = max(8, inner_w - 30)
    lines.append(_box_row(
        f"{C_SILVER}ProcQ:{C_RESET} {pq_total:>3}/32"
        f"  {make_bar(pq_pct, pq_bar_w, pq_color)}"
        f"  {pq_color}{pq_pct:4.0f}%{C_RESET}",
        pw,
    ))

    # Row 2: processing workers
    if nwt_total > 0:
        wu_pct = int(nwa_total / nwt_total * 100)
        wu_col = C_RED if wu_pct >= 90 else (C_GREEN if wu_pct >= 20 else C_YELLOW)
        wu_bar_w = max(8, inner_w - 35)
        lines.append(_box_row(
            f"{C_SILVER}Workers:{C_RESET} {wu_col}{nwa_total}/{nwt_total}{C_RESET}"
            f"  {make_bar(wu_pct, wu_bar_w, wu_col)}"
            f"  {wu_col}{wu_pct:3d}%{C_RESET}",
            pw,
        ))
    else:
        lines.append(_box_row(
            f"{C_SILVER}Workers:{C_RESET} {C_SILVER}not yet started{C_RESET}",
            pw,
        ))

    # Row 3: write queue
    bar_label = f"WriteQ {wq_total}/{wq_cap}"
    queue_bar_w = max(8, inner_w - len(bar_label) - 14)
    lines.append(_box_row(
        f"{C_SILVER}{bar_label}{C_RESET}"
        f"  {make_bar(wq_pct, queue_bar_w, bp_color)}"
        f"  {bp_color}{C_BOLD}{wq_pct:4.0f}%{C_RESET}"
        f"  {C_SILVER}pressure:{C_RESET} {bp_color}{bp_label}{C_RESET}",
        pw,
    ))

    # Row 4: stream activity + SPS
    sps_str = f"{live_sps:6.1f}" if live_sps > 0 else "     —"
    lines.append(_box_row(
        f"{C_SILVER}Streams:{C_RESET}"
        f"  {C_GREEN}▶ {n_active} active{C_RESET}"
        f"  {C_SILVER}· {n_idle} idle{C_RESET}"
        f"  {C_SILVER}Format:{C_RESET} {fc}{C_BOLD}{fmt}{C_RESET}"
        f"  {C_SILVER}SPS:{C_RESET} {C_BOLD}{sps_str}{C_RESET}",
        pw,
    ))

    lines.append(_box_bot(pw))
    return lines


# ── Plan summary panel ────────────────────────────────────────────────────────

def _render_plan_summary_panel(state, term_width):
    """
    Plan summary with live in-flight patches shown separately.

    In-flight patches are patches being created by currently running streams
    that have not yet been persisted to the plan (plan.completed_total only
    updates after a film finishes).  Showing them here makes progress feel live.
    """
    pw = min(term_width, 100)
    inner_w = pw - 4
    lines = [_box_top(pw, "PLAN SUMMARY")]

    ps = state.get("plan_summary", {})
    if not ps:
        lines.append(_box_row(f"{C_SILVER}Plan not yet created (Phase 3 pending…){C_RESET}", pw))
        lines.append(_box_bot(pw))
        return lines

    n_total   = ps.get("n_items_total",   0)
    n_done    = ps.get("n_items_done",    0)
    n_running = ps.get("n_items_running", 0)
    n_pending = ps.get("n_items_pending", 0)
    n_failed  = ps.get("n_items_failed",  0)

    planned_total   = ps.get("planned_total",   0)
    completed_total = ps.get("completed_total", 0)

    # In-flight: patches being made right now (not yet in plan)
    in_flight = sum(
        s.get("patches_created", 0)
        for s in state.get("active_streams", [])
        if s.get("state") == "running"
    )
    effective_done = completed_total + in_flight
    pct_overall = (effective_done / planned_total * 100) if planned_total > 0 else 0.0

    # Item counts
    lines.append(_box_row(
        f"{C_SILVER}Items:{C_RESET} {C_BOLD}{n_total}{C_RESET}"
        f"  {C_GREEN}✔ done:{n_done}{C_RESET}"
        f"  {C_CYAN}▶ running:{n_running}{C_RESET}"
        f"  {C_SILVER}· pending:{n_pending}{C_RESET}"
        + (f"  {C_RED}✖ failed:{n_failed}{C_RESET}" if n_failed else ""),
        pw,
    ))

    # Patches: completed (plan) + in-flight + remaining
    remaining = max(0, planned_total - effective_done)
    lines.append(_box_row(
        f"{C_SILVER}Patches:{C_RESET}"
        f"  {C_GREEN}done:{_abbrev_num(completed_total)}{C_RESET}"
        + (f"  {C_CYAN}+live:{_abbrev_num(in_flight)}{C_RESET}" if in_flight > 0 else "")
        + f"  {C_YELLOW}rem:{_abbrev_num(remaining)}{C_RESET}"
        + f"  {C_SILVER}total:{_abbrev_num(planned_total)}{C_RESET}",
        pw,
    ))

    # Global bar (effective done = plan_done + in-flight)
    label_w = 10
    lines.append(_box_row(
        _bar_row("TOTAL", pct_overall, effective_done, planned_total, inner_w, C_GREEN, label_w),
        pw,
    ))

    # Per-category bars
    planned_per_cat   = ps.get("planned_per_category", {})
    completed_per_cat = ps.get("completed_per_category", {})
    for i, cat in enumerate(sorted(planned_per_cat)):
        pl = planned_per_cat.get(cat, 0)
        co = completed_per_cat.get(cat, 0)
        # Add in-flight for this category
        in_flight_cat = sum(
            s.get("patches_per_category", {}).get(cat, 0)
            for s in state.get("active_streams", [])
            if s.get("state") == "running"
        )
        effective_co = co + in_flight_cat
        pct = (effective_co / pl * 100) if pl > 0 else 0.0
        row = _bar_row(cat, pct, effective_co, pl, inner_w, _category_color(cat, i), label_w)
        if in_flight_cat > 0:
            row = _trunc(row, inner_w - 9) + f" {C_CYAN}+{_abbrev_num(in_flight_cat)}{C_RESET}"
        lines.append(_box_row(row, pw))

    lines.append(_box_bot(pw))
    return lines


# ── Production progress panel ─────────────────────────────────────────────────

def _render_production_panel(state, term_width):
    """
    Three-level hierarchy: category → format → degradation template.

    Shows effective progress (plan_done + in-flight from active streams).
    Every line is constrained to term_width via _box_row.
    """
    pw = min(term_width, 100)
    inner_w = pw - 4
    lines = [_box_top(pw, "PRODUCTION PROGRESS")]

    cats       = state.get("categories", [])
    ovr        = state.get("overall_progress", {})
    patch_dist = state.get("patch_distribution", {})

    if not cats:
        lines.append(_box_row(f"{C_SILVER}No categories configured.{C_RESET}", pw))
        lines.append(_box_bot(pw))
        return lines

    cat_label_w = max(6, min(12, max(len(c) for c in cats)))

    for i, cat in enumerate(cats):
        color = _category_color(cat, i)
        info  = ovr.get(cat, {})
        done  = int(info.get("created", 0)) if isinstance(info, dict) else 0
        tgt   = int(info.get("target",  0)) if isinstance(info, dict) else 0

        # Add in-flight patches for this category
        in_flight_cat = sum(
            s.get("patches_per_category", {}).get(cat, 0)
            for s in state.get("active_streams", [])
            if s.get("state") == "running"
        )
        effective_done = done + in_flight_cat
        pct = min(100.0, 100.0 * effective_done / tgt) if tgt > 0 else 0.0

        lines.append(_box_row(
            _bar_row(cat, pct, effective_done, tgt, inner_w, color, cat_label_w),
            pw,
        ))

        # ── Format-template sub-rows (indented 4 chars) ───────────────────
        cat_fmts  = patch_dist.get(cat, {})
        fmt_items = sorted(cat_fmts.items(), key=lambda kv: -kv[1].get("target", 0))

        for fi, (fmt_name, fmt_data) in enumerate(fmt_items):
            fmt_target    = fmt_data.get("target",  0)
            fmt_completed = fmt_data.get("count",   0)
            fmt_pct = (fmt_completed / fmt_target * 100) if fmt_target > 0 else 0.0
            fc = _FMT_COLORS[fi % len(_FMT_COLORS)]

            fmt_label_w = max(6, min(16, inner_w // 4))
            fmt_inner_w = inner_w - 4  # 4 for "  ╰ "

            fmt_bar_row = (
                f"  {C_SILVER}╰{C_RESET} "
                + _bar_row(
                    _trunc(fmt_name, fmt_label_w),
                    fmt_pct, fmt_completed, fmt_target,
                    fmt_inner_w, fc, fmt_label_w,
                )
            )
            lines.append(_box_row(fmt_bar_row, pw))

            # ── Degradation sub-rows (one per template) ───────────────────
            deg_planned: dict  = fmt_data.get("deg_planned", {})
            deg_completed: dict = fmt_data.get("deg_completed", {})

            if deg_planned:
                deg_total = sum(deg_planned.values())
                for di, (dname, dplanned) in enumerate(
                    sorted(deg_planned.items(), key=lambda kv: -kv[1])
                ):
                    dc_color = _DEG_COLORS[di % len(_DEG_COLORS)]
                    dpct_plan = (100.0 * dplanned / deg_total) if deg_total else 0.0
                    dcompleted = deg_completed.get(dname, 0)
                    d_lbl = _trunc(dname, 18)
                    lines.append(_box_row(
                        f"      {C_SILVER}╰{C_RESET} "
                        f"{dc_color}{d_lbl}{C_RESET}"
                        f"  {C_SILVER}{_abbrev_num(dcompleted)}"
                        f"/{_abbrev_num(dplanned)}"
                        f" ({dpct_plan:.0f}%){C_RESET}",
                        pw,
                    ))

    lines.append(_box_bot(pw))
    return lines


# ── ETA / performance panel ───────────────────────────────────────────────────

def _render_eta_panel(state, term_width):
    pw = min(term_width, 100)
    lines = [_box_top(pw, "PERFORMANCE & ETA")]

    streams    = state.get("active_streams", [])
    total_fps  = sum(s.get("live_fps", 0.0) for s in streams)
    live_sps   = state.get("live_sps", 0.0)
    n_active   = max(1, state.get("n_active_streams", 0))
    vid_idx    = state.get("current_video_index", 0)
    total_vids = state.get("total_videos", 0)
    output_fmt = state.get("output_format", "BMP")
    fc         = C_GREEN if output_fmt == "BMP" else C_YELLOW

    fps_str = f"{total_fps:.1f}"
    sps_str = f"{live_sps:.1f}" if live_sps > 0 else "—"

    lines.append(_box_row(
        f"{C_SILVER}Videos:{C_RESET} {vid_idx}/{total_vids}"
        f"   {C_SILVER}Streams:{C_RESET} {C_BOLD}{n_active}{C_RESET} active"
        f"   {C_SILVER}FPS:{C_RESET} {C_BOLD}{fps_str}{C_RESET}"
        f"   {C_GREEN}SPS:{C_RESET} {C_BOLD}{sps_str}{C_RESET}"
        f"   {C_SILVER}Output:{C_RESET} {fc}{C_BOLD}{output_fmt}{C_RESET}",
        pw,
    ))

    # ETA per category
    eta_dict  = state.get("eta", {})
    cats      = state.get("categories", [])
    eta_parts = []
    for i, cat in enumerate(cats):
        color = _category_color(cat, i)
        eta_v = eta_dict.get(cat, "—")
        eta_s = format_time(eta_v) if isinstance(eta_v, (int, float)) and eta_v > 0 else "—"
        eta_parts.append(
            f"{color}{_category_display_name(cat)}{C_RESET}{C_SILVER}:{C_RESET}{eta_s}"
        )

    if eta_parts:
        lines.append(_box_row(
            f"{C_SILVER}ETA:  {C_RESET}" + "  │  ".join(eta_parts),
            pw,
        ))

    # Global plan-based ETA
    eta_total = eta_dict.get("total", 0)
    ps = state.get("plan_summary", {})
    if isinstance(eta_total, (int, float)) and eta_total > 0 and ps:
        planned   = ps.get("planned_total", 0)
        completed = ps.get("completed_total", 0)
        in_flight = sum(
            s.get("patches_created", 0)
            for s in state.get("active_streams", [])
            if s.get("state") == "running"
        )
        remaining = max(0, planned - completed - in_flight)
        lines.append(_box_row(
            f"{C_SILVER}Global ETA:{C_RESET}  "
            f"{C_BOLD}{format_time(eta_total)}{C_RESET}"
            f"  {C_SILVER}remaining:{C_RESET} {C_BOLD}{_abbrev_num(remaining)}{C_RESET} patches",
            pw,
        ))

    lines.append(_box_bot(pw))
    return lines


# ── Title bar ─────────────────────────────────────────────────────────────────

def _render_title(term_width):
    pw = min(term_width, 100)
    title = (
        f"{C_BOLD}DATASET GENERATOR V2{C_RESET}"
        f"  {C_SILVER}·{C_RESET}  Multi-Stream"
        f"  {C_SILVER}·{C_RESET}  {C_GREEN}libplacebo{C_RESET}"
        f"  {C_SILVER}·{C_RESET}  {C_YELLOW}BMP{C_RESET}/{C_CYAN}PNG{C_RESET}"
    )
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
    Draw (or redraw in-place) the complete dataset-generation dashboard.

    Uses cursor-homing (ANSI_HOME) instead of a full-screen clear to avoid
    visible flicker.  Each rendered line is right-padded with spaces to
    overwrite the previous content at that position.  After the last rendered
    line, \033[J erases any stale rows from a taller previous render.

    On terminal resize (detected by comparing terminal dimensions between
    renders, or by the SIGWINCH handler set via register_resize_handler()) a
    full ANSI_CLEAR is issued first so that leftover artefacts disappear.

    Layout:
      Title bar
      GPU stream panels  (side-by-side when wide enough)
      Current film status
      Pipeline  |  Plan Summary   (side-by-side when wide enough)
      Production Progress  |  Performance & ETA  (side-by-side when wide enough)

    Must only be called AFTER Phase 3 (plan creation), because before that
    the plan does not exist and the panels cannot show accurate data.

    Args:
        state: ``ui_state`` dict from DatasetGeneratorV2UHD.
    """
    global _prev_line_count, _needs_clear, _prev_term_size

    term_width, term_height = shutil.get_terminal_size((100, 50))
    term_width = max(40, term_width)
    current_size = (term_width, term_height)

    # Full clear if the terminal was resized or SIGWINCH fired.
    if _needs_clear or (_prev_term_size != (0, 0) and _prev_term_size != current_size):
        sys.stdout.write(ANSI_CLEAR + ANSI_HOME)
        sys.stdout.flush()
        _prev_line_count = 0
        _needs_clear = False
    _prev_term_size = current_size

    # Decide whether to use two-column layout for the lower panels.
    use_two_col = term_width >= _MIN_SIDE_BY_SIDE
    pw_half = (term_width - 1) // 2

    out = []
    out.extend(_render_title(term_width))
    out.extend(_render_gpu_panels(state.get("active_streams", []), term_width))
    out.extend(_render_current_film_panel(state, term_width))

    # Pipeline + Plan Summary (side-by-side when wide enough)
    if use_two_col:
        pipeline_lines = _render_write_queue_panel(state, pw_half)
        plan_lines     = _render_plan_summary_panel(state, pw_half)
        out.extend(_render_panels_side_by_side(pipeline_lines, plan_lines, pw_half))
    else:
        out.extend(_render_write_queue_panel(state, term_width))
        out.extend(_render_plan_summary_panel(state, term_width))

    # Production Progress + Performance & ETA (side-by-side when wide enough)
    if use_two_col:
        prod_lines = _render_production_panel(state, pw_half)
        eta_lines  = _render_eta_panel(state, pw_half)
        out.extend(_render_panels_side_by_side(prod_lines, eta_lines, pw_half))
    else:
        out.extend(_render_production_panel(state, term_width))
        out.extend(_render_eta_panel(state, term_width))

    # ── Flicker-free write ────────────────────────────────────────────────
    # Move cursor to top-left (no clear).  Each line is padded to term_width
    # so it fully overwrites the previous content at that position.
    buf = [ANSI_HOME]
    for line in out:
        vis_len = get_visible_len(line)
        pad = max(0, term_width - vis_len)
        buf.append(line + ' ' * pad)

    # Erase stale lines from a taller previous render.
    current_lines = len(out)
    for _ in range(max(0, _prev_line_count - current_lines)):
        buf.append(' ' * term_width)
    _prev_line_count = current_lines

    sys.stdout.write('\n'.join(buf))
    sys.stdout.write('\n')
    sys.stdout.flush()
