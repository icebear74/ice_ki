"""
UI Display Module — Priority-tile dashboard for VSR++ training.

Tiles are built as lists of strings and laid out by priority so the most
important information is always visible regardless of terminal height.

Priority ladder (lower = always shown first):
  1  header   — title + training score          (always)
  2  progress — step / epoch / ETA bars         (always)
  3  losses   — l1 / ms / grad / LR / speed     (always)
  4  adaptive — mode / plateau / AdamW magic eye
  5  quality  — KI vs LR quality metrics
  6  activity — averaged layer activity
 99  footer   — status bar + keyboard hints     (always, printed last)
"""

import sys
import shutil
import numpy as np

from .ui_terminal import (
    C_GREEN, C_GRAY, C_RESET, C_BOLD, C_CYAN, C_RED, C_YELLOW,
    C_MAGENTA, C_BORDER,
    _line, _two_cols, _sep, _hdr, _ftr,
    make_bar, make_bar_fusion, make_bar_final_fusion,
    make_adamw_magic_eye, make_peak_activity_bar, make_size_bar,
    get_visible_len, format_time,
    clear_screen, move_cursor_home, hide_cursor, show_cursor,
    # kept for external callers
    print_line, print_two_columns, print_separator, print_header, print_footer,
)

# Convergence thresholds
CONVERGENCE_SLOPE_THRESHOLD = -0.00005
DIVERGENCE_SLOPE_THRESHOLD  =  0.00005

# Global state
activity_history = {i + 1: [] for i in range(64)}
loss_history     = []
TREND_WINDOW     = 50
last_term_size   = (0, 0)
last_display_mode = -1


# ── Layer helpers ─────────────────────────────────────────────────────────────

def is_fusion_layer(name):
    return "Fuse" in name or "Fusion" in name

def is_final_fusion(name):
    return "Final Fusion" in name

def get_bar_for_layer(name, percent, width):
    if is_final_fusion(name):
        return make_bar_final_fusion(percent, width)
    elif is_fusion_layer(name):
        return make_bar_fusion(percent, width)
    return make_bar(percent, width)


# ── Trend / convergence ───────────────────────────────────────────────────────

def calculate_trends(activities):
    trends = []
    for layer_id, current_val in enumerate(activities, 1):
        if layer_id not in activity_history:
            activity_history[layer_id] = []
        activity_history[layer_id].append(current_val)
        if len(activity_history[layer_id]) > TREND_WINDOW:
            activity_history[layer_id].pop(0)
        if len(activity_history[layer_id]) >= 20:
            recent = np.mean(activity_history[layer_id][-10:])
            old    = np.mean(activity_history[layer_id][-20:-10])
            trend  = ((recent - old) / (old + 1e-8)) * 100
        else:
            trend = 0.0
        trends.append(trend)
    return trends


def calculate_convergence_status(loss_hist):
    global loss_history
    loss_history = loss_hist
    if len(loss_hist) < 100:
        return f"{C_YELLOW}Warming up…{C_RESET}"
    recent = loss_hist[-100:]
    x      = np.arange(len(recent))
    slope  = np.polyfit(x, recent, 1)[0]
    if slope < CONVERGENCE_SLOPE_THRESHOLD:
        return f"{C_GREEN}Converging ✓{C_RESET}"
    elif abs(slope) < DIVERGENCE_SLOPE_THRESHOLD:
        return f"{C_CYAN}Plateauing ⚠{C_RESET}"
    else:
        return f"{C_RED}Diverging ✗{C_RESET}"


# ── Activity data extraction ──────────────────────────────────────────────────

def get_activity_data(model):
    """Extract per-layer activity tuples (name, percent, trend, raw_value)."""
    m = model.module if hasattr(model, 'module') else model
    if not hasattr(m, 'get_layer_activity'):
        return [(f"Layer {i+1}", 0, 0, 0.0) for i in range(32)]

    activity_dict = m.get_layer_activity()
    if not activity_dict:
        return [(f"Layer {i+1}", 0, 0, 0.0) for i in range(32)]

    backward          = activity_dict.get('backward_trunk', [])
    backward_fuse     = activity_dict.get('backward_fuse', 0.0)
    backward_align    = activity_dict.get('backward_align_flow', None)
    forward           = activity_dict.get('forward_trunk', [])
    forward_fuse      = activity_dict.get('forward_fuse', 0.0)
    forward_align     = activity_dict.get('forward_align_flow', None)
    fusion            = activity_dict.get('fusion', 0.0)

    activities_with_names = []

    def _add_fusion(name, activity):
        if isinstance(activity, list):
            if len(activity) == 3:
                activities_with_names.append((f"{name} 3x3",  float(activity[0]) if activity[0] is not None else 0.0))
                activities_with_names.append((f"{name} 1x1",  float(activity[1]) if activity[1] is not None else 0.0))
                activities_with_names.append((f"{name} Gate", float(activity[2]) if activity[2] is not None else 0.0))
            elif len(activity) == 2:
                activities_with_names.append((f"{name} 3x3", float(activity[0]) if activity[0] is not None else 0.0))
                activities_with_names.append((f"{name} 1x1", float(activity[1]) if activity[1] is not None else 0.0))
            elif len(activity) > 0:
                avg = sum(float(a) if a is not None else 0.0 for a in activity) / len(activity)
                activities_with_names.append((name, avg))
            else:
                activities_with_names.append((name, 0.0))
        elif activity is not None:
            activities_with_names.append((name, float(activity)))
        else:
            activities_with_names.append((name, 0.0))

    for i, act in enumerate(backward):
        activities_with_names.append((f"Backward {i+1}", float(act) if act is not None else 0.0))
    _add_fusion("Backward Fuse", backward_fuse)
    if backward_align is not None:
        activities_with_names.append(("Backward Align", float(backward_align)))

    for i, act in enumerate(forward):
        activities_with_names.append((f"Forward {i+1}", float(act) if act is not None else 0.0))
    _add_fusion("Forward Fuse", forward_fuse)
    if forward_align is not None:
        activities_with_names.append(("Forward Align", float(forward_align)))

    _add_fusion("Final Fusion", fusion)

    # Sanitise NaN / inf
    activities_with_names = [
        (n, v if np.isfinite(v) else 0.0)
        for n, v in activities_with_names
    ]

    raw_vals = [v for _, v in activities_with_names]
    if not raw_vals:
        return [(f"Layer {i+1}", 0, 0, 0.0) for i in range(32)]

    trends  = calculate_trends(raw_vals)
    max_val = max(raw_vals) if max(raw_vals) > 1e-12 else 1e-12

    return [
        (name, int((v / max_val) * 100), trends[i], v)
        for i, (name, v) in enumerate(activities_with_names)
    ]


# ── Tile builders ─────────────────────────────────────────────────────────────

def _tile_header(ui_w, paused, score_pct, score_color, score_icon, score_label, components_str):
    """Header tile: app title + training score.  Always shown."""
    T = [_hdr(ui_w)]
    title = f"{C_BOLD}{C_CYAN}◈  VSR++ TRAINING{C_RESET}"
    if paused:
        title += f"  {C_YELLOW}[ PAUSED ]{C_RESET}"
    T.append(_line(title, ui_w))
    T.append(_sep(ui_w, 'double'))
    T.append(_line(
        f"{C_BOLD}⭐ SCORE{C_RESET}  {score_icon} "
        f"{score_color}{C_BOLD}{score_pct:.0f}%  {score_label}{C_RESET}"
        f"  {C_BORDER}│{C_RESET}  {components_str}",
        ui_w,
    ))
    T.append(_sep(ui_w, 'double'))
    return T                      # 5 lines


def _tile_progress(ui_w, step, max_steps, total_prog, epoch,
                   current_epoch_step, steps_per_epoch, epoch_prog,
                   total_eta, epoch_eta, bar_w):
    """Progress tile: step/epoch progress bars + ETA."""
    T = []
    T.append(_line(f"{C_BOLD}📊 PROGRESS{C_RESET}", ui_w))
    T.append(_sep(ui_w))
    T.append(_two_cols(
        f"Step   {C_CYAN}{C_BOLD}{step:,}{C_RESET} / {max_steps:,}",
        f"Epoch  {C_CYAN}{C_BOLD}{epoch}{C_RESET}  ({current_epoch_step:,} / {steps_per_epoch:,})",
        ui_w,
    ))
    T.append(_line(f"Total  {make_bar(total_prog, bar_w)} {total_prog:>5.1f}%  ETA {total_eta}", ui_w))
    T.append(_line(f"Epoch  {make_bar(epoch_prog, bar_w)} {epoch_prog:>5.1f}%  ETA {epoch_eta}", ui_w))
    T.append(_sep(ui_w, 'double'))
    return T                      # 6 lines


def _tile_losses(ui_w, l1, ms, grad, perc, total,
                 w_l1, w_ms, w_grad, w_perc,
                 lr, lr_phase_str, it_time, convergence):
    """Loss & metrics tile."""
    T = []
    T.append(_line(f"{C_BOLD}📉 LOSSES & METRICS{C_RESET}", ui_w))
    T.append(_sep(ui_w))
    T.append(_two_cols(
        f"L1   {C_CYAN}{l1:.5f}{C_RESET}  w:{C_GREEN}{w_l1:.2f}{C_RESET}",
        f"MS   {C_CYAN}{ms:.5f}{C_RESET}  w:{C_GREEN}{w_ms:.2f}{C_RESET}",
        ui_w,
    ))
    T.append(_two_cols(
        f"Grad {C_CYAN}{grad:.5f}{C_RESET}  w:{C_GREEN}{w_grad:.2f}{C_RESET}",
        f"Perc {C_CYAN}{perc:.5f}{C_RESET}  w:{C_GREEN}{w_perc:.2f}{C_RESET}",
        ui_w,
    ))
    T.append(_line(f"Total  {C_BOLD}{C_CYAN}{total:.5f}{C_RESET}", ui_w))
    T.append(_sep(ui_w, 'thin'))
    ips = 1.0 / it_time if it_time > 0 else 0.0
    T.append(_two_cols(
        f"LR  {C_GREEN}{lr:.2e}{C_RESET}  {lr_phase_str}",
        f"Speed  {C_CYAN}{ips:.2f} it/s{C_RESET}  ({it_time:.2f}s/it)",
        ui_w,
    ))
    T.append(_line(f"Convergence  {convergence}", ui_w))
    T.append(_sep(ui_w, 'double'))
    return T                      # 9 lines


def _tile_adaptive(ui_w, adaptive_status, adam_momentum):
    """Adaptive system tile."""
    T = []
    T.append(_line(f"{C_BOLD}🔧 ADAPTIVE SYSTEM{C_RESET}", ui_w))
    T.append(_sep(ui_w))

    mode            = adaptive_status.get('mode', 'Stable')
    is_cooldown     = adaptive_status.get('is_cooldown', False)
    cooldown_rem    = adaptive_status.get('cooldown_remaining', 0)
    plateau         = adaptive_status.get('plateau_counter', 0)
    patience        = adaptive_status.get('plateau_patience', 100)
    aggressive      = adaptive_status.get('aggressive_mode', False)
    lr_boost_ready  = adaptive_status.get('lr_boost_available', False)
    grad_clip       = adaptive_status.get('grad_clip', 1.0)

    mode_color  = C_RED if mode == 'Aggressive' else C_GREEN
    mode_icon   = '🔴' if mode == 'Aggressive' else '🟢'

    cool_str = (f"{C_YELLOW}ACTIVE{C_RESET} ({cooldown_rem} steps)"
                if is_cooldown else f"{C_GREEN}Idle{C_RESET}")

    if plateau > patience * 1.5:
        plat_str = f"{C_RED}{plateau}  ⚠{C_RESET}"
    elif plateau > patience * 0.75:
        plat_str = f"{C_YELLOW}{plateau}{C_RESET}"
    else:
        plat_str = f"{C_GREEN}{plateau}{C_RESET}"

    boost_str = f"{C_GREEN}Ready ⚡{C_RESET}" if lr_boost_ready else f"{C_YELLOW}Cooldown{C_RESET}"

    T.append(_two_cols(
        f"Mode   {mode_icon} {mode_color}{mode}{C_RESET}",
        f"Cooldown  {cool_str}",
        ui_w,
    ))
    T.append(_two_cols(
        f"Plateau   {plat_str} steps",
        f"LR Boost  {boost_str}",
        ui_w,
    ))
    T.append(_two_cols(
        f"Grad Clip  {C_CYAN}{grad_clip:.3f}{C_RESET}",
        f"Aggressive  {C_RED if aggressive else C_GRAY}{'ON' if aggressive else 'off'}{C_RESET}",
        ui_w,
    ))
    T.append(_sep(ui_w, 'thin'))
    eye = make_adamw_magic_eye(adam_momentum, width=25)
    T.append(_line(f"AdamW SNR  {eye}  {C_CYAN}{adam_momentum:.3f}{C_RESET}", ui_w))
    T.append(_sep(ui_w, 'double'))
    return T                      # 8 lines


def _tile_quality(ui_w, ki_quality, lr_quality, improvement, ki_to_gt, lr_to_gt):
    """Quality metrics tile."""
    T = []
    T.append(_line(f"{C_BOLD}🎯 QUALITY{C_RESET}", ui_w))
    T.append(_sep(ui_w))
    imp_sign  = "+" if improvement >= 0 else ""
    imp_color = C_GREEN if improvement >= 0 else C_RED
    T.append(_two_cols(
        f"LR  {C_YELLOW}{lr_quality:>5.1f}%{C_RESET}",
        f"KI  {C_GREEN}{ki_quality:>5.1f}%{C_RESET}",
        ui_w,
    ))
    T.append(_line(
        f"Improvement  {C_BOLD}{imp_color}{imp_sign}{improvement:.1f}%{C_RESET}",
        ui_w,
    ))
    if ki_to_gt is not None and lr_to_gt is not None:
        ki_s = "+" if ki_to_gt >= 0 else ""
        lr_s = "+" if lr_to_gt >= 0 else ""
        T.append(_two_cols(
            f"KI→GT  {C_CYAN}{ki_s}{ki_to_gt:.1f}%{C_RESET}",
            f"LR→GT  {C_CYAN}{lr_s}{lr_to_gt:.1f}%{C_RESET}",
            ui_w,
        ))
    T.append(_sep(ui_w, 'double'))
    return T                      # 5-6 lines


def _tile_activity(ui_w, activities, bar_w):
    """
    Layer activity tile — shows group averages, not individual layers.

    Groups: Backward trunk / Forward trunk / Fusion & Align
    Also shows the single peak layer.
    """
    T = []

    # Separate layers into groups
    bwd_vals  = [raw for n, _, _, raw in activities
                 if "Backward" in n and "Fuse" not in n and "Align" not in n]
    fwd_vals  = [raw for n, _, _, raw in activities
                 if "Forward"  in n and "Fuse" not in n and "Align" not in n]
    fus_vals  = [raw for n, _, _, raw in activities
                 if "Fuse" in n or "Fusion" in n or "Align" in n]

    all_vals = [raw for _, _, _, raw in activities]
    max_val  = max(all_vals) if all_vals and max(all_vals) > 1e-12 else 1e-12

    peak_tuple = max(activities, key=lambda x: x[3]) if activities else ("?", 0, 0, 0.0)
    peak_name, peak_raw = peak_tuple[0], peak_tuple[3]

    if peak_raw > 2.0:
        pk_col = C_RED;    pk_lbl = "EXTREME 🔥🔥"
    elif peak_raw > 1.5:
        pk_col = C_YELLOW; pk_lbl = "very high 🔥"
    elif peak_raw > 1.0:
        pk_col = C_CYAN;   pk_lbl = "high ⚡"
    elif peak_raw > 0.5:
        pk_col = C_GREEN;  pk_lbl = "moderate"
    else:
        pk_col = C_GREEN;  pk_lbl = "normal ✓"

    T.append(_two_cols(
        f"{C_BOLD}⚡ LAYER ACTIVITY{C_RESET}",
        f"Peak  {C_BOLD}{pk_col}{peak_name}{C_RESET}  "
        f"{pk_col}{peak_raw:.3f}  {pk_lbl}{C_RESET}",
        ui_w,
    ))
    T.append(_sep(ui_w, 'thin'))

    def _avg_row(vals, label, use_fusion_bar=False):
        if not vals:
            return _line(f"  {label:<10}  {C_GRAY}no data{C_RESET}", ui_w)
        avg  = sum(vals) / len(vals)
        pct  = int((avg / max_val) * 100)
        bar  = make_bar_fusion(pct, bar_w) if use_fusion_bar else make_bar(pct, bar_w)
        n    = len(vals)
        return _line(f"  {label:<10}  {bar}  {pct:>3}%  ({n} layers)", ui_w)

    T.append(_avg_row(bwd_vals, "Backward"))
    T.append(_avg_row(fwd_vals, "Forward"))
    if fus_vals:
        T.append(_avg_row(fus_vals, "Fusion", use_fusion_bar=True))

    T.append(_sep(ui_w, 'double'))
    return T                      # 6-7 lines


def _tile_footer(ui_w, step, config, grad_clip):
    """Footer tile: status bar + keyboard hints.  Always shown last."""
    T = []

    val_every  = config.get('VAL_STEP_EVERY',  config.get('val_step_every',  500))
    save_every = config.get('SAVE_STEP_EVERY', config.get('save_step_every', 10000))
    batch      = config.get('BATCH_SIZE', 4)
    accum      = config.get('ACCUMULATION_STEPS', 1)

    nv = (val_every  - (step % val_every))  if val_every  > 0 else 0
    ns = (save_every - (step % save_every)) if save_every > 0 else 0

    T.append(_line(
        f"VAL IN {C_CYAN}{nv:<5}{C_RESET}  "
        f"SAVE IN {C_CYAN}{ns:<6}{C_RESET}  "
        f"BATCH {C_CYAN}{batch}×{accum}={batch*accum}{C_RESET}  "
        f"CLIP {C_CYAN}{grad_clip:.3f}{C_RESET}",
        ui_w,
    ))
    T.append(_ftr(ui_w))
    T.append(_line(
        f"{C_CYAN}P{C_RESET} Pause  "
        f"{C_CYAN}V{C_RESET} Validate  "
        f"{C_CYAN}C{C_RESET} Checkpoint  "
        f"{C_CYAN}Q{C_RESET} Quit  "
        f"{C_CYAN}ESC{C_RESET} Emergency Stop",
        ui_w,
    ))
    return T                      # 3 lines


# ── Main draw entry point ─────────────────────────────────────────────────────

def draw_ui(step, epoch, losses, it_time, activities, config, num_images,
            steps_per_epoch, current_epoch_step, adaptive_status=None,
            paused=False, quality_metrics=None, lr_info=None,
            total_eta="Calculating...", epoch_eta="Calculating...",
            adam_momentum=0.0, val_iter_per_sec=0.0):
    """
    Draw the complete training dashboard.

    Tiles are stacked in priority order.  If the terminal is too short to fit
    all tiles, lower-priority tiles (layer activity, quality, adaptive) are
    silently omitted until everything fits.
    """
    global last_term_size, last_display_mode

    term_size    = shutil.get_terminal_size()
    display_mode = config.get("DISPLAY_MODE", 0)

    if term_size != last_term_size or display_mode != last_display_mode:
        clear_screen()
        last_term_size    = term_size
        last_display_mode = display_mode

    move_cursor_home()
    hide_cursor()

    rows = term_size.lines
    ui_w = max(80, term_size.columns - 4)
    bar_w = min(50, max(15, ui_w - 58))

    # ── Extract values ────────────────────────────────────────────────────────
    l1   = losses.get('l1',         0.0)
    ms   = losses.get('ms',         0.0)
    grad = losses.get('grad',       0.0)
    perc = losses.get('perceptual', 0.0)
    tot  = losses.get('total',      0.0)

    if lr_info:
        current_lr = lr_info.get('lr',    0.0)
        lr_phase   = lr_info.get('phase', 'unknown')
    else:
        current_lr = 0.0
        lr_phase   = 'unknown'

    lr_phase_str = {
        'warmup':          f'{C_YELLOW}WARMUP{C_RESET}',
        'cosine':          f'{C_GREEN}COSINE{C_RESET}',
        'plateau_reduced': f'{C_RED}PLATEAU{C_RESET}',
    }.get(lr_phase, lr_phase)

    if quality_metrics:
        lr_quality  = quality_metrics.get('lr_quality',  0.0)
        ki_quality  = quality_metrics.get('ki_quality',  0.0)
        improvement = quality_metrics.get('improvement', 0.0)
        ki_to_gt    = quality_metrics.get('ki_to_gt',    None)
        lr_to_gt    = quality_metrics.get('lr_to_gt',    None)
    else:
        lr_quality = ki_quality = improvement = 0.0
        ki_to_gt   = lr_to_gt  = None

    if adaptive_status:
        w_l1   = adaptive_status.get('l1_weight',          0.7)
        w_ms   = adaptive_status.get('ms_weight',          0.2)
        w_grad = adaptive_status.get('grad_weight',        0.1)
        w_perc = adaptive_status.get('perceptual_weight',  0.0)
        grad_clip = adaptive_status.get('grad_clip',       1.0)
    else:
        w_l1      = 0.7
        w_ms      = 0.2
        w_grad    = 0.1
        w_perc    = config.get('PERCEPTUAL_WEIGHT', 0.0)
        grad_clip = config.get('GRAD_CLIP',         1.0)

    max_steps   = config.get("MAX_STEPS", 100000)
    total_prog  = (step / max_steps)         * 100 if max_steps        > 0 else 0.0
    epoch_prog  = (current_epoch_step / steps_per_epoch) * 100 if steps_per_epoch > 0 else 0.0

    # ── Training score ────────────────────────────────────────────────────────
    global loss_history

    score_total = 0.0
    score_max   = 0.0
    score_parts = []

    if len(loss_history) >= 100:
        recent = loss_history[-100:]
        slope  = np.polyfit(np.arange(len(recent)), recent, 1)[0]
        if slope < CONVERGENCE_SLOPE_THRESHOLD:
            score_total += 30.0; score_parts.append(f"Trend:{C_GREEN}↓Conv{C_RESET}")
        elif abs(slope) < DIVERGENCE_SLOPE_THRESHOLD:
            score_total += 20.0; score_parts.append(f"Trend:{C_CYAN}Plateau{C_RESET}")
        else:
            score_total +=  5.0; score_parts.append(f"Trend:{C_RED}↑Div{C_RESET}")
    else:
        score_total += 15.0;     score_parts.append(f"Trend:{C_YELLOW}WarmUp{C_RESET}")
    score_max += 30.0

    if quality_metrics and ki_quality > 0:
        score_total += (ki_quality / 100.0) * 40.0
        score_max   += 40.0
        qc = C_GREEN if ki_quality >= 70 else C_YELLOW if ki_quality >= 50 else C_RED
        score_parts.append(f"Quality:{qc}{ki_quality:.0f}%{C_RESET}")

    if adaptive_status:
        plateau  = adaptive_status.get('plateau_counter', 0)
        patience = adaptive_status.get('plateau_patience', 100)
        if plateau < patience * 0.75:
            score_total += 30.0; score_parts.append(f"Stab:{C_GREEN}Stable{C_RESET}")
        elif plateau < patience * 1.5:
            score_total += 20.0; score_parts.append(f"Stab:{C_YELLOW}Moderate{C_RESET}")
        else:
            score_total += 10.0; score_parts.append(f"Stab:{C_RED}Unstable{C_RESET}")
        score_max += 30.0

    score_pct = (score_total / score_max * 100.0) if score_max > 0 else 0.0

    if score_max == 0:
        sc_col = C_GRAY;   sc_icon = "⚪"; sc_lbl = "NO DATA"
    elif score_pct >= 80:
        sc_col = C_GREEN;  sc_icon = "🟢"; sc_lbl = "EXCELLENT"
    elif score_pct >= 60:
        sc_col = C_CYAN;   sc_icon = "🔵"; sc_lbl = "GOOD"
    elif score_pct >= 40:
        sc_col = C_YELLOW; sc_icon = "🟡"; sc_lbl = "MODERATE"
    else:
        sc_col = C_RED;    sc_icon = "🔴"; sc_lbl = "NEEDS ATTENTION"

    convergence   = calculate_convergence_status(loss_history)
    components_str = "  ".join(score_parts)

    # ── Build all tiles ───────────────────────────────────────────────────────
    #  (priority, name, lines[])
    tiles = []

    tiles.append((1, 'header', _tile_header(
        ui_w, paused, score_pct, sc_col, sc_icon, sc_lbl, components_str,
    )))

    tiles.append((2, 'progress', _tile_progress(
        ui_w, step, max_steps, total_prog, epoch,
        current_epoch_step, steps_per_epoch, epoch_prog,
        total_eta, epoch_eta, bar_w,
    )))

    tiles.append((3, 'losses', _tile_losses(
        ui_w, l1, ms, grad, perc, tot,
        w_l1, w_ms, w_grad, w_perc,
        current_lr, lr_phase_str, it_time, convergence,
    )))

    if adaptive_status:
        tiles.append((4, 'adaptive', _tile_adaptive(ui_w, adaptive_status, adam_momentum)))

    if quality_metrics and ki_quality > 0:
        tiles.append((5, 'quality', _tile_quality(
            ui_w, ki_quality, lr_quality, improvement, ki_to_gt, lr_to_gt,
        )))

    if activities:
        tiles.append((6, 'activity', _tile_activity(ui_w, activities, bar_w)))

    tiles.append((99, 'footer', _tile_footer(ui_w, step, config, grad_clip)))

    # ── Priority budget allocation ────────────────────────────────────────────
    # Core tiles (header, progress, losses, footer) are always rendered even if
    # they slightly overflow — they carry critical information.
    # Optional tiles (adaptive, quality, activity) are added in strict priority
    # order; we stop at the first one that no longer fits so we never show a
    # less-important tile without a more-important one above it.
    budget = rows - 2   # reserve 2 lines so the shell prompt is never covered

    CORE_PRIORITIES = {1, 2, 3, 99}
    always_tiles   = [(p, n, t) for p, n, t in tiles if p in CORE_PRIORITIES]
    optional_tiles = sorted(
        [(p, n, t) for p, n, t in tiles if p not in CORE_PRIORITIES],
        key=lambda x: x[0],
    )

    for _, _, t in always_tiles:
        budget -= len(t)

    selected = []
    for p, n, t in optional_tiles:
        if budget >= len(t):
            budget -= len(t)
            selected.append((p, n, t))
        else:
            break   # strict: stop at first tile that no longer fits

    all_tiles = sorted(always_tiles + selected, key=lambda x: x[0])

    # ── Render ────────────────────────────────────────────────────────────────
    footer_lines = next(t for p, n, t in all_tiles if n == 'footer')
    body_tiles   = [(p, n, t) for p, n, t in all_tiles if n != 'footer']

    output = []
    for _, _, t in body_tiles:
        output.extend(t)
    output.extend(footer_lines)

    # Fill any remaining rows with blank cleared lines so no old content
    # from previous renders or external print() calls bleeds through.
    used_rows  = len(output)
    blank_line = ' ' * (ui_w + 4)  # +4 for the box border characters
    for _ in range(max(0, rows - used_rows - 1)):
        output.append(blank_line)

    sys.stdout.write('\n'.join(output) + '\n')
    sys.stdout.flush()


# ── Size distribution helpers (used by external callers) ─────────────────────

def print_size_distribution_panel(size_stats, ui_width=120):
    """Print size-distribution tracking panel (direct-print, not tiled)."""
    if not size_stats or 'size_stats' not in size_stats:
        return

    stats          = size_stats['size_stats']
    category_order = sorted(stats.keys())

    print_separator(ui_width, style='double')
    print_line(f"{C_BOLD}Size Distribution Progress{C_RESET}", ui_width)
    print_separator(ui_width, style='thin')

    for category in category_order:
        cat_stats = stats[category]
        trained   = cat_stats['images_trained']
        target    = cat_stats['target_images']
        pct       = cat_stats['percentage_complete']
        bar       = make_size_bar(trained, target, 30)
        line      = f"{category:>15}: {bar} {trained:>8,} / {target:>8,}  ({pct:>6.2f}%)"
        print_line(line, ui_width)

    print_separator(ui_width, style='thin')
    total = size_stats.get('total_images_trained', 0)
    print_line(f"Total Images Trained: {C_BOLD}{total:,}{C_RESET}", ui_width)


def format_size_stats_compact(size_stats):
    """Return size stats as a compact status string."""
    if not size_stats or 'size_stats' not in size_stats:
        return "No size stats"

    stats  = size_stats['size_stats']
    parts  = []
    for cat in sorted(stats.keys()):
        s      = stats[cat]
        target = s['target_images']
        trained = s['images_trained']
        pct    = (trained / target * 100.0) if target > 0 else 0.0
        parts.append(f"{cat.split('_')[0]}: {pct:.0f}%")
    return " | ".join(parts)
