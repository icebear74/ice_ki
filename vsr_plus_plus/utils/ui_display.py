"""
UI Display Module

Complete GUI display functions for training visualization.
Ported from original train.py to maintain full feature parity.
"""

import sys
import shutil
import numpy as np
from .ui_terminal import *


# Global state for UI
activity_history = {i+1: [] for i in range(64)}  # Support up to 64 layers
loss_history = []
TREND_WINDOW = 50
last_term_size = (0, 0)
last_display_mode = -1


def is_fusion_layer(layer_name):
    """Check if a layer is a fusion layer based on its name"""
    return "Fuse" in layer_name or "Fusion" in layer_name


def is_final_fusion(layer_name):
    """Check if a layer is the final fusion layer"""
    return "Final Fusion" in layer_name


def get_bar_for_layer(layer_name, percent, width):
    """Get appropriate bar style based on layer type"""
    if is_final_fusion(layer_name):
        return make_bar_final_fusion(percent, width)
    elif is_fusion_layer(layer_name):
        return make_bar_fusion(percent, width)
    else:
        return make_bar(percent, width)


def calculate_trends(activities):
    """
    Calculate activity trends for each layer
    
    Args:
        activities: List of current activity values
    
    Returns:
        list: Trend percentages for each layer
    """
    trends = []
    for layer_id, current_val in enumerate(activities, 1):
        if layer_id not in activity_history:
            activity_history[layer_id] = []
        
        activity_history[layer_id].append(current_val)
        if len(activity_history[layer_id]) > TREND_WINDOW:
            activity_history[layer_id].pop(0)
        
        if len(activity_history[layer_id]) >= 20:
            recent = np.mean(activity_history[layer_id][-10:])
            old = np.mean(activity_history[layer_id][-20:-10])
            trend = ((recent - old) / (old + 1e-8)) * 100
        else:
            trend = 0.0
        trends.append(trend)
    return trends


def calculate_convergence_status(loss_hist):
    """
    Calculate convergence status from loss history
    
    Args:
        loss_hist: List of recent loss values
    
    Returns:
        str: Status string with color codes
    """
    global loss_history
    loss_history = loss_hist  # Update global
    
    if len(loss_hist) < 100:
        return "Warming up..."
    
    recent = loss_hist[-100:]
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent, 1)[0]
    
    if slope < -0.00005:
        return f"{C_GREEN}Converging ✓{C_RESET}"
    elif abs(slope) < 0.00005:
        return f"{C_CYAN}Plateauing ⚠{C_RESET}"
    else:
        return f"{C_RED}Diverging ✗{C_RESET}"


def get_activity_data(model):
    """
    Extract layer activity data from model
    
    Args:
        model: VSR model instance
    
    Returns:
        list: List of tuples (layer_name, activity_percent, trend, raw_value)
    """
    # Unwrap DataParallel if needed
    m = model.module if hasattr(model, 'module') else model
    
    if not hasattr(m, 'get_layer_activity'):
        # Model doesn't have activity tracking
        return [(f"Layer {i+1}", 0, 0, 0.0) for i in range(32)]
    
    activity_dict = m.get_layer_activity()
    
    if not activity_dict:
        return [(f"Layer {i+1}", 0, 0, 0.0) for i in range(32)]
    
    # Combine backward and forward trunk activities + fusion layers into a single list
    backward = activity_dict.get('backward_trunk', [])
    backward_fuse = activity_dict.get('backward_fuse', 0.0)
    forward = activity_dict.get('forward_trunk', [])
    forward_fuse = activity_dict.get('forward_fuse', 0.0)
    fusion = activity_dict.get('fusion', 0.0)
    
    # Build combined list with names
    activities_with_names = []
    
    # Backward trunk blocks
    for i, act in enumerate(backward):
        activities_with_names.append((f"Backward {i+1}", float(act) if act is not None else 0.0))
    
    # Backward fuse layer
    activities_with_names.append(("Backward Fuse", float(backward_fuse) if backward_fuse is not None else 0.0))
    
    # Forward trunk blocks
    for i, act in enumerate(forward):
        activities_with_names.append((f"Forward {i+1}", float(act) if act is not None else 0.0))
    
    # Forward fuse layer
    activities_with_names.append(("Forward Fuse", float(forward_fuse) if forward_fuse is not None else 0.0))
    
    # Final fusion layer
    activities_with_names.append(("Final Fusion", float(fusion) if fusion is not None else 0.0))
    
    # Extract just the values for trend calculation
    activities_raw = [val for name, val in activities_with_names]
    
    if not activities_raw:
        return [(f"Layer {i+1}", 0, 0, 0.0) for i in range(32)]
    
    trends = calculate_trends(activities_raw)
    max_val = max(activities_raw) if max(activities_raw) > 1e-12 else 1e-12
    
    # Build final list with layer names
    activities = [(name, int((v / max_val) * 100), trends[i], v) 
                  for i, (name, v) in enumerate(activities_with_names)]
    
    return activities


def draw_ui(step, epoch, losses, it_time, activities, config, num_images, 
            steps_per_epoch, current_epoch_step, adaptive_status=None, 
            paused=False, quality_metrics=None, lr_info=None, 
            total_eta="Calculating...", epoch_eta="Calculating..."):
    """
    Draw the complete training UI
    
    Args:
        step: Global training step
        epoch: Current epoch
        losses: Dict with loss components ('l1', 'ms', 'grad', 'total')
        it_time: Time per iteration (seconds)
        activities: List of (layer_id, activity%, trend, raw_value) tuples
        config: Training configuration dict
        num_images: Total number of training images
        steps_per_epoch: Number of steps per epoch
        current_epoch_step: Current step within epoch
        adaptive_status: Adaptive system status dict (optional)
        paused: Whether training is paused
        quality_metrics: Dict with quality info (optional)
        lr_info: Dict with LR info ('lr', 'phase') (optional)
    """
    global last_term_size, last_display_mode
    
    term_size = shutil.get_terminal_size()
    display_mode = config.get("DISPLAY_MODE", 0)
    
    # Clear screen if terminal size changed OR display mode changed
    if term_size != last_term_size or display_mode != last_display_mode:
        clear_screen()
        last_term_size = term_size
        last_display_mode = display_mode
    
    # Move cursor to home and hide cursor
    move_cursor_home()
    hide_cursor()
    
    # Calculate UI width
    ui_w = max(90, term_size.columns - 4)
    col_width = (ui_w - 7) // 2
    # Standardize all progress bars to 30 characters
    BAR_LENGTH = 30
    bar_width_single = BAR_LENGTH
    bar_width_double = BAR_LENGTH
    
    # Calculate progress (total_eta and epoch_eta are passed as parameters from trainer)
    max_steps = config.get("MAX_STEPS", 100000)
    total_prog = (step / max_steps) * 100 if max_steps > 0 else 0
    epoch_prog = (current_epoch_step / steps_per_epoch) * 100 if steps_per_epoch > 0 else 0
    
    # Extract values
    l1_loss = losses.get('l1', 0.0)
    ms_loss = losses.get('ms', 0.0)
    grad_loss = losses.get('grad', 0.0)
    total_loss = losses.get('total', 0.0)
    
    # LR info
    if lr_info:
        current_lr = lr_info.get('lr', 0.0)
        lr_phase = lr_info.get('phase', 'unknown')
    else:
        current_lr = 0.0
        lr_phase = 'unknown'
    
    # Quality metrics
    if quality_metrics:
        lr_quality = quality_metrics.get('lr_quality', 0.0)
        ki_quality = quality_metrics.get('ki_quality', 0.0)
        improvement = quality_metrics.get('improvement', 0.0)
        ki_to_gt = quality_metrics.get('ki_to_gt', None)
        lr_to_gt = quality_metrics.get('lr_to_gt', None)
    else:
        lr_quality = ki_quality = improvement = 0.0
        ki_to_gt = lr_to_gt = None
    
    # Adaptive status
    if adaptive_status:
        l1_weight = adaptive_status.get('l1_weight', 0.7)
        ms_weight = adaptive_status.get('ms_weight', 0.2)
        grad_weight = adaptive_status.get('grad_weight', 0.1)
        grad_clip = adaptive_status.get('grad_clip', 1.0)
        aggressive = adaptive_status.get('aggressive_mode', False)
    else:
        l1_weight = 0.7
        ms_weight = 0.2
        grad_weight = 0.1
        grad_clip = config.get('GRAD_CLIP', 1.0)
        aggressive = False
    
    # === HEADER ===
    print_header(ui_w)
    
    title = f"{C_BOLD}VSR++ TRAINING{C_RESET}"
    if paused:
        title += f" {C_YELLOW}[PAUSED]{C_RESET}"
    print_line(f"{title:^{ui_w-4}}", ui_w)
    
    print_separator(ui_w, 'double')
    
    # === PROGRESS ===
    print_line(f"{C_BOLD}📊 PROGRESS{C_RESET}", ui_w)
    print_separator(ui_w, 'single')
    
    # Calculate progress percentage
    progress_pct = (step / max_steps) * 100 if max_steps > 0 else 0
    
    print_line(f"Step: {C_BOLD}{step:,}{C_RESET} / {max_steps:,} ({progress_pct:.1f}%) | ETA: {total_eta}", ui_w)
    print_line(f"Epoch: {C_BOLD}{epoch}{C_RESET} (Step {current_epoch_step}/{steps_per_epoch})", ui_w)
    
    print_line(f"Total:  {make_bar(total_prog, bar_width_single)} {total_prog:>5.1f}% ETA: {total_eta}", ui_w)
    print_line(f"Epoch:  {make_bar(epoch_prog, bar_width_single)} {epoch_prog:>5.1f}% ETA: {epoch_eta}", ui_w)
    
    print_separator(ui_w, 'double')
    
    # === LOSS & METRICS ===
    print_line(f"{C_BOLD}📉 LOSS & METRICS{C_RESET}", ui_w)
    print_separator(ui_w, 'single')
    
    print_two_columns(
        f"L1:   {C_CYAN}{l1_loss:.6f}{C_RESET} (w:{l1_weight:.2f})",
        f"MS:   {C_CYAN}{ms_loss:.6f}{C_RESET} (w:{ms_weight:.2f})",
        ui_w
    )
    print_two_columns(
        f"Grad: {C_CYAN}{grad_loss:.6f}{C_RESET} (w:{grad_weight:.2f})",
        f"Total: {C_BOLD}{C_CYAN}{total_loss:.6f}{C_RESET}",
        ui_w
    )
    
    print_separator(ui_w, 'single')
    
    # Learning Rate
    lr_phase_str = {
        'warmup': f'{C_YELLOW}WARMUP{C_RESET}',
        'cosine': f'{C_GREEN}COSINE{C_RESET}',
        'plateau_reduced': f'{C_RED}PLATEAU{C_RESET}'
    }.get(lr_phase, lr_phase)
    
    print_two_columns(
        f"LR: {C_GREEN}{current_lr:.6f}{C_RESET} ({lr_phase_str})",
        f"Speed: {C_CYAN}{it_time:.2f}s/it{C_RESET}",
        ui_w
    )
    
    # Convergence status
    convergence = calculate_convergence_status(loss_history)
    print_line(f"Convergence: {convergence}", ui_w)
    
    print_separator(ui_w, 'double')
    
    # === QUALITY (if available) ===
    if quality_metrics and ki_quality > 0:
        print_line(f"{C_BOLD}🎯 QUALITY{C_RESET}", ui_w)
        print_separator(ui_w, 'single')
        
        print_two_columns(
            f"LR:  {C_YELLOW}{lr_quality:>5.1f}%{C_RESET}",
            f"KI:  {C_GREEN}{ki_quality:>5.1f}%{C_RESET}",
            ui_w
        )
        
        # Improvement (Sum of per-image KI-LR differences)
        imp_sign = "+" if improvement >= 0 else ""
        imp_color = C_GREEN if improvement >= 0 else C_RED
        print_line(f"Improvement (Sum): {C_BOLD}{imp_color}{imp_sign}{improvement:.1f}%{C_RESET}", ui_w)
        
        # Display GT differences if available
        if ki_to_gt is not None and lr_to_gt is not None:
            ki_gt_sign = "+" if ki_to_gt >= 0 else ""
            lr_gt_sign = "+" if lr_to_gt >= 0 else ""
            print_two_columns(
                f"KI to GT: {C_CYAN}{ki_gt_sign}{ki_to_gt:.1f}%{C_RESET}",
                f"LR to GT: {C_CYAN}{lr_gt_sign}{lr_to_gt:.1f}%{C_RESET}",
                ui_w
            )
        
        print_separator(ui_w, 'double')
    
    # === ADAPTIVE SYSTEM ===
    if adaptive_status:
        print_line(f"{C_BOLD}🔧 ADAPTIVE SYSTEM{C_RESET}", ui_w)
        print_separator(ui_w, 'single')
        
        print_two_columns(
            f"Grad Clip: {C_CYAN}{grad_clip:.3f}{C_RESET}",
            f"Aggressive: {C_RED if aggressive else C_GRAY}{'YES' if aggressive else 'NO'}{C_RESET}",
            ui_w
        )
        
        print_separator(ui_w, 'double')
    
    # === LAYER ACTIVITY ===
    available_lines = term_size.lines - 30  # Lines available for layer display
    
    # Calculate layer counts for display
    n_blocks = config.get('N_BLOCKS', 32)
    total_layers = len(activities) if activities else 0
    fusion_layers = total_layers - n_blocks if total_layers > n_blocks else 0
    
    print_line(f"{C_BOLD}⚡ LAYER ACTIVITY{C_RESET} - Mode: {DISPLAY_MODE_NAMES[display_mode]}", ui_w)
    print_line(f"ResidualBlocks: {C_CYAN}{n_blocks}{C_RESET} | Total Layers: {C_CYAN}{total_layers}{C_RESET} (incl. {fusion_layers} fusion)", ui_w)
    print_separator(ui_w, 'single')
    
    # Display based on mode
    _draw_activity_display(activities, display_mode, available_lines, ui_w, 
                          bar_width_single, bar_width_double)
    
    # === FOOTER ===
    print_separator(ui_w, 'double')
    
    nv = config.get('VAL_STEP_EVERY', 500) - (step % config.get('VAL_STEP_EVERY', 500))
    ns = config.get('SAVE_STEP_EVERY', 10000) - (step % config.get('SAVE_STEP_EVERY', 10000))
    batch = config.get('BATCH_SIZE', 4)
    accum = config.get('ACCUMULATION_STEPS', 1)
    
    print_line(f"VAL IN: {nv:<5} │ SAVE IN: {ns:<5} │ BATCH: {batch}x{accum}={batch*accum} │ GRAD CLIP: {grad_clip:.3f}", ui_w)
    
    print_footer(ui_w)
    
    # Control hints
    sys.stdout.write(f"{' ' * ((ui_w - 55) // 2)}{C_BOLD}( ENTER: Config | S: Next View | P: Pause | V: Val ){C_RESET}\n")
    sys.stdout.flush()


def _draw_activity_display(activities, display_mode, available_lines, ui_w, 
                           bar_width_single, bar_width_double):
    """
    Draw layer activity based on selected display mode
    
    Args:
        activities: List of (layer_id, activity%, trend, raw_value) tuples
        display_mode: Display mode (0-3)
        available_lines: Available terminal lines for display
        ui_w: UI width
        bar_width_single: Width for single-column bars
        bar_width_double: Width for double-column bars
    """
    if not activities:
        print_line("No activity data available", ui_w)
        return
    
    num_activities = len(activities)
    
    if display_mode == 0:
        # MODE 0: Grouped by Trunk → Sorted by Position
        # Assume: First half = backward, second half = forward
        half = num_activities // 2
        backward = activities[:half]
        forward = activities[half:]
        
        backward_overall = int(np.mean([act for _, act, _, _ in backward])) if backward else 0
        forward_overall = int(np.mean([act for _, act, _, _ in forward])) if forward else 0
        
        print_line(f"{C_BOLD}🔥 BACKWARD TRUNK{C_RESET} - Overall: {make_bar(backward_overall, BAR_LENGTH)} {backward_overall}%", ui_w)
        print_separator(ui_w, 'single')
        
        if available_lines >= num_activities:
            for name, act, trend, raw in backward:
                # Fixed width for name to align bars
                print_line(f"{name:<15}: {get_bar_for_layer(name, act, bar_width_single)} {act:>3}%", ui_w)
        else:
            for row in range(0, len(backward), 2):
                left = backward[row]
                right = backward[row+1] if row+1 < len(backward) else None
                left_str = f"{left[0]:<13}:{get_bar_for_layer(left[0], left[1], bar_width_double)}{left[1]:3}%"
                right_str = f"{right[0]:<13}:{get_bar_for_layer(right[0], right[1], bar_width_double)}{right[1]:3}%" if right else ""
                print_two_columns(left_str, right_str, ui_w)
        
        print_separator(ui_w, 'double')
        print_line(f"{C_BOLD}⚡ FORWARD TRUNK{C_RESET} - Overall: {make_bar(forward_overall, BAR_LENGTH)} {forward_overall}%", ui_w)
        print_separator(ui_w, 'single')
        
        if available_lines >= num_activities:
            for name, act, trend, raw in forward:
                # Fixed width for name to align bars
                print_line(f"{name:<15}: {get_bar_for_layer(name, act, bar_width_single)} {act:>3}%", ui_w)
        else:
            for row in range(0, len(forward), 2):
                left = forward[row]
                right = forward[row+1] if row+1 < len(forward) else None
                left_str = f"{left[0]:<13}:{get_bar_for_layer(left[0], left[1], bar_width_double)}{left[1]:3}%"
                right_str = f"{right[0]:<13}:{get_bar_for_layer(right[0], right[1], bar_width_double)}{right[1]:3}%" if right else ""
                print_two_columns(left_str, right_str, ui_w)
    
    elif display_mode == 1:
        # MODE 1: Grouped by Trunk → Sorted by Activity
        half = num_activities // 2
        backward = sorted(activities[:half], key=lambda x: x[1], reverse=True)
        forward = sorted(activities[half:], key=lambda x: x[1], reverse=True)
        
        backward_overall = int(np.mean([act for _, act, _, _ in backward])) if backward else 0
        forward_overall = int(np.mean([act for _, act, _, _ in forward])) if forward else 0
        
        print_line(f"{C_BOLD}🔥 BACKWARD TRUNK (sorted){C_RESET} - Overall: {make_bar(backward_overall, BAR_LENGTH)} {backward_overall}%", ui_w)
        print_separator(ui_w, 'single')
        
        if available_lines >= num_activities:
            for name, act, trend, raw in backward:
                print_line(f"{name:<15}: {get_bar_for_layer(name, act, bar_width_single)} {act:>3}%", ui_w)
        else:
            for row in range(0, len(backward), 2):
                left = backward[row]
                right = backward[row+1] if row+1 < len(backward) else None
                left_str = f"{left[0]:<13}:{get_bar_for_layer(left[0], left[1], bar_width_double)}{left[1]:3}%"
                right_str = f"{right[0]:<13}:{get_bar_for_layer(right[0], right[1], bar_width_double)}{right[1]:3}%" if right else ""
                print_two_columns(left_str, right_str, ui_w)
        
        print_separator(ui_w, 'double')
        print_line(f"{C_BOLD}⚡ FORWARD TRUNK (sorted){C_RESET} - Overall: {make_bar(forward_overall, BAR_LENGTH)} {forward_overall}%", ui_w)
        print_separator(ui_w, 'single')
        
        if available_lines >= num_activities:
            for name, act, trend, raw in forward:
                print_line(f"{name:<15}: {get_bar_for_layer(name, act, bar_width_single)} {act:>3}%", ui_w)
        else:
            for row in range(0, len(forward), 2):
                left = forward[row]
                right = forward[row+1] if row+1 < len(forward) else None
                left_str = f"{left[0]:<13}:{get_bar_for_layer(left[0], left[1], bar_width_double)}{left[1]:3}%"
                right_str = f"{right[0]:<13}:{get_bar_for_layer(right[0], right[1], bar_width_double)}{right[1]:3}%" if right else ""
                print_two_columns(left_str, right_str, ui_w)
    
    elif display_mode == 2:
        # MODE 2: Flat List → Sorted by Position
        if available_lines >= num_activities:
            for name, act, trend, raw in activities:
                print_line(f"{name:<15}: {get_bar_for_layer(name, act, bar_width_single)} {act:>3}%", ui_w)
        else:
            for row in range(0, num_activities, 2):
                left = activities[row]
                right = activities[row+1] if row+1 < num_activities else None
                left_str = f"{left[0]:<13}:{get_bar_for_layer(left[0], left[1], bar_width_double)}{left[1]:3}%"
                right_str = f"{right[0]:<13}:{get_bar_for_layer(right[0], right[1], bar_width_double)}{right[1]:3}%" if right else ""
                print_two_columns(left_str, right_str, ui_w)
    
    else:  # display_mode == 3
        # MODE 3: Flat List → Sorted by Activity
        sorted_acts = sorted(activities, key=lambda x: x[1], reverse=True)
        
        if available_lines >= num_activities:
            for name, act, trend, raw in sorted_acts:
                print_line(f"{name:<15}: {get_bar_for_layer(name, act, bar_width_single)} {act:>3}%", ui_w)
        else:
            for row in range(0, num_activities, 2):
                left = sorted_acts[row]
                right = sorted_acts[row+1] if row+1 < num_activities else None
                left_str = f"{left[0]:<13}:{get_bar_for_layer(left[0], left[1], bar_width_double)}{left[1]:3}%"
                right_str = f"{right[0]:<13}:{get_bar_for_layer(right[0], right[1], bar_width_double)}{right[1]:3}%" if right else ""
                print_two_columns(left_str, right_str, ui_w)
