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
            total_eta="Calculating...", epoch_eta="Calculating...", adam_momentum=0.0):
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
        adam_momentum: AdamW momentum value (optional)
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
    
    # Use dynamic widths for progress bars based on terminal size
    # Allocate space for labels and percentage, use remaining for bar
    LABEL_AND_PERCENTAGE_SPACE = 50  # Space for labels, numbers, borders
    DOUBLE_COLUMN_OVERHEAD = 60  # Additional space for 2-column layouts
    
    total_available = ui_w - 4  # Account for borders
    bar_width_single = min(50, max(20, total_available - LABEL_AND_PERCENTAGE_SPACE))
    bar_width_double = min(30, max(15, (total_available - DOUBLE_COLUMN_OVERHEAD) // 2))
    
    # Calculate progress percentages (ETAs already calculated and passed from trainer)
    max_steps = config.get("MAX_STEPS", 100000)
    total_prog = (step / max_steps) * 100 if max_steps > 0 else 0
    epoch_prog = (current_epoch_step / steps_per_epoch) * 100 if steps_per_epoch > 0 else 0
    
    # Extract values
    l1_loss = losses.get('l1', 0.0)
    ms_loss = losses.get('ms', 0.0)
    grad_loss = losses.get('grad', 0.0)
    perceptual_loss = losses.get('perceptual', 0.0)
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
        perceptual_weight = adaptive_status.get('perceptual_weight', 0.0)
        grad_clip = adaptive_status.get('grad_clip', 1.0)
        aggressive = adaptive_status.get('aggressive_mode', False)
        # NEW: Get cooldown status
        is_cooldown = adaptive_status.get('is_cooldown', False)
        cooldown_remaining = adaptive_status.get('cooldown_remaining', 0)
        adaptive_mode = adaptive_status.get('mode', 'Stable')
    else:
        l1_weight = 0.7
        ms_weight = 0.2
        grad_weight = 0.1
        perceptual_weight = config.get('PERCEPTUAL_WEIGHT', 0.0)
        grad_clip = config.get('GRAD_CLIP', 1.0)
        aggressive = False
        is_cooldown = False
        cooldown_remaining = 0
        adaptive_mode = 'Stable'
    
    # === HEADER ===
    print_header(ui_w)
    
    title = f"{C_BOLD}VSR++ TRAINING{C_RESET}"
    if paused:
        title += f" {C_YELLOW}[PAUSED]{C_RESET}"
    print_line(f"{title:^{ui_w-4}}", ui_w)
    
    print_separator(ui_w, 'double')
    
    # === TRAINING SCORE (Prominent indicator of training quality/performance) ===
    # Calculate training score based on multiple factors
    score_components = []
    score_total = 0.0
    score_max = 0.0
    
    # 1. Loss trend (up to 30 points) - based on convergence
    if len(loss_history) >= 100:
        recent = loss_history[-100:]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        
        if slope < -0.00005:  # Converging
            loss_trend_score = 30.0
            loss_trend_status = f"{C_GREEN}Converging{C_RESET}"
        elif abs(slope) < 0.00005:  # Plateauing
            loss_trend_score = 20.0
            loss_trend_status = f"{C_CYAN}Plateau{C_RESET}"
        else:  # Diverging
            loss_trend_score = 5.0
            loss_trend_status = f"{C_RED}Diverging{C_RESET}"
    else:
        loss_trend_score = 15.0
        loss_trend_status = f"{C_YELLOW}Warming Up{C_RESET}"
    score_total += loss_trend_score
    score_max += 30.0
    score_components.append(f"Trend:{loss_trend_status}")
    
    # 2. Quality metrics (up to 40 points) - if available
    if quality_metrics and ki_quality > 0:
        # KI quality (0-100%) -> 0-40 points
        quality_score = (ki_quality / 100.0) * 40.0
        score_total += quality_score
        score_max += 40.0
        
        quality_color = C_GREEN if ki_quality >= 70 else C_YELLOW if ki_quality >= 50 else C_RED
        score_components.append(f"Quality:{quality_color}{ki_quality:.0f}%{C_RESET}")
    
    # 3. Learning stability (up to 30 points) - based on plateau counter and adaptive mode
    if adaptive_status:
        plateau = adaptive_status.get('plateau_counter', 0)
        adaptive_mode_val = adaptive_status.get('mode', 'Stable')
        
        if plateau < 150:
            stability_score = 30.0
            stability_status = f"{C_GREEN}Stable{C_RESET}"
        elif plateau < 300:
            stability_score = 20.0
            stability_status = f"{C_YELLOW}Moderate{C_RESET}"
        else:
            stability_score = 10.0
            stability_status = f"{C_RED}Unstable{C_RESET}"
        
        score_total += stability_score
        score_max += 30.0
        score_components.append(f"Stability:{stability_status}")
    
    # Calculate overall score percentage
    if score_max > 0:
        training_score_pct = (score_total / score_max) * 100.0
    else:
        training_score_pct = 50.0  # Default if no components available
    
    # Determine color and icon based on score
    if training_score_pct >= 80:
        score_color = C_GREEN
        score_icon = "🟢"
        score_label = "EXCELLENT"
    elif training_score_pct >= 60:
        score_color = C_CYAN
        score_icon = "🔵"
        score_label = "GOOD"
    elif training_score_pct >= 40:
        score_color = C_YELLOW
        score_icon = "🟡"
        score_label = "MODERATE"
    else:
        score_color = C_RED
        score_icon = "🔴"
        score_label = "NEEDS ATTENTION"
    
    # Display prominent training score
    print_line(f"{C_BOLD}⭐ TRAINING SCORE: {score_icon} {score_color}{training_score_pct:.1f}%{C_RESET} {C_BOLD}({score_label}){C_RESET}", ui_w)
    
    # Show component breakdown
    components_str = " | ".join(score_components)
    print_line(f"   {components_str}", ui_w)
    
    print_separator(ui_w, 'double')
    
    # === PROGRESS ===
    print_line(f"{C_BOLD}📊 PROGRESS{C_RESET}", ui_w)
    print_separator(ui_w, 'single')
    
    print_line(f"Step: {C_BOLD}{step:,}{C_RESET} / {max_steps:,} ({total_prog:.1f}%) | ETA: {total_eta}", ui_w)
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
        f"Perc: {C_CYAN}{perceptual_loss:.6f}{C_RESET} (w:{perceptual_weight:.2f})",
        ui_w
    )
    print_line(f"Total: {C_BOLD}{C_CYAN}{total_loss:.6f}{C_RESET}", ui_w)
    
    # Perceptual Loss Status - show details if enabled
    if perceptual_weight > 0:
        perc_status = f"{C_GREEN}ACTIVE{C_RESET}" if perceptual_loss > 0 else f"{C_YELLOW}ENABLED (0){C_RESET}"
        print_separator(ui_w, 'thin')
        print_two_columns(
            f"Perceptual: {perc_status} (w:{perceptual_weight:.2f})",
            f"Type: {C_MAGENTA}VGG16{C_RESET} (ImageNet)",
            ui_w
        )
        # Show perceptual loss is active (VGG weights are frozen, not learning)
        if perceptual_loss > 0.001:
            learning_indicator = f"{C_GREEN}●{C_RESET} VGG Features Active"
        else:
            learning_indicator = f"{C_YELLOW}○{C_RESET} Initializing"
        print_line(f"  {learning_indicator} | Pretrained frozen weights", ui_w)
    else:
        print_separator(ui_w, 'thin')
        print_line(f"Perceptual: {C_GRAY}DISABLED{C_RESET} (w:0.00)", ui_w)
    
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
        
        # Mode
        mode_color = C_RED if adaptive_mode == 'Aggressive' else C_GREEN
        mode_icon = '🔴' if adaptive_mode == 'Aggressive' else '🟢'
        mode_display = f"{mode_icon} {mode_color}{adaptive_mode}{C_RESET}"
        print_line(f"Mode: {mode_display}", ui_w)
        
        # Cooldown status
        if is_cooldown:
            cooldown_display = f"⏸️  {C_YELLOW}ACTIVE{C_RESET} ({cooldown_remaining} steps remaining)"
        else:
            cooldown_display = f"✅ {C_GREEN}Inactive{C_RESET}"
        print_line(f"Cooldown: {cooldown_display}", ui_w)
        
        # Plateau counter
        plateau = adaptive_status.get('plateau_counter', 0)
        if plateau > 300:
            plateau_color = C_RED
            plateau_icon = '🚨'
            plateau_text = f"{plateau} steps (WARNING)"
        elif plateau > 150:
            plateau_color = C_YELLOW
            plateau_icon = '🟡'
            plateau_text = f"{plateau} steps"
        else:
            plateau_color = C_GREEN
            plateau_icon = '🟢'
            plateau_text = f"{plateau} steps"
        print_line(f"Plateau: {plateau_icon} {plateau_color}{plateau_text}{C_RESET}", ui_w)
        
        # LR Boost status
        lr_boost_available = adaptive_status.get('lr_boost_available', False)
        if lr_boost_available:
            boost_display = f"⚡ {C_GREEN}Ready{C_RESET}"
        else:
            boost_display = f"⏳ {C_YELLOW}Cooldown{C_RESET}"
        print_line(f"LR Boost: {boost_display}", ui_w)
        
        print_separator(ui_w, 'thin')
        
        # Weight bars with current values
        print_two_columns(
            f"Grad Clip: {C_CYAN}{grad_clip:.3f}{C_RESET}",
            f"Aggressive: {C_RED if aggressive else C_GRAY}{'YES' if aggressive else 'NO'}{C_RESET}",
            ui_w
        )
        
        # AdamW Magic Eye
        print_separator(ui_w, 'thin')
        from .ui_terminal import make_adamw_magic_eye
        magic_eye = make_adamw_magic_eye(adam_momentum, width=25)
        print_line(f"AdamW Momentum: {magic_eye} {C_CYAN}{adam_momentum:.4f}{C_RESET}", ui_w)
        
        print_separator(ui_w, 'double')
    
    # === RUNTIME CONFIGURATION (New config parameters) ===
    # Display key runtime config parameters accessible via ENTER menu
    print_line(f"{C_BOLD}⚙️ RUNTIME CONFIG{C_RESET} (Press ENTER to edit)", ui_w)
    print_separator(ui_w, 'single')
    
    # Display key parameters
    plateau_threshold = config.get('plateau_safety_threshold', 500)
    plateau_patience = config.get('plateau_patience', 200)
    cooldown_dur = config.get('cooldown_duration', 50)
    
    print_two_columns(
        f"Plateau Threshold: {C_CYAN}{plateau_threshold}{C_RESET} steps",
        f"Plateau Patience: {C_CYAN}{plateau_patience}{C_RESET} steps",
        ui_w
    )
    print_two_columns(
        f"Cooldown Duration: {C_CYAN}{cooldown_dur}{C_RESET} steps",
        f"LR Range: {C_GREEN}{config.get('min_lr', 1e-6):.2e}{C_RESET} - {C_GREEN}{config.get('max_lr', 1e-4):.2e}{C_RESET}",
        ui_w
    )
    
    # TensorBoard settings
    tb_interval = config.get('log_tboard_every', 50)
    val_interval = config.get('val_step_every', 500)
    save_interval = config.get('save_step_every', 10000)
    
    print_separator(ui_w, 'thin')
    print_two_columns(
        f"TensorBoard Log: Every {C_CYAN}{tb_interval}{C_RESET} steps",
        f"Validation: Every {C_CYAN}{val_interval}{C_RESET} steps",
        ui_w
    )
    print_line(f"Checkpoint Save: Every {C_CYAN}{save_interval}{C_RESET} steps", ui_w)
    
    print_separator(ui_w, 'double')
    
    # === PEAK LAYER ACTIVITY (Enhanced visibility) ===
    if activities and len(activities) > 0:
        # Find peak activity
        # activities is a list of tuples: (name, percent, trend, raw_value)
        if isinstance(activities, dict):
            peak_value = max(activities.values())
            peak_layer_name = ""
            for layer_name, value in activities.items():
                if value == peak_value:
                    peak_layer_name = layer_name
                    break
        else:
            # For list of tuples, find the one with max raw_value (index 3)
            peak_tuple = max(activities, key=lambda x: x[3] if isinstance(x, tuple) and len(x) > 3 else 0)
            if isinstance(peak_tuple, tuple) and len(peak_tuple) > 3:
                peak_layer_name = peak_tuple[0]
                peak_value = peak_tuple[3]  # raw_value is at index 3
            else:
                peak_layer_name = "Unknown"
                peak_value = 0.0
        
        # Determine color and icon based on peak value
        if peak_value > 2.0:
            peak_color = C_RED
            peak_icon = "🔥🔥🔥"
            peak_status = "EXTREME"
        elif peak_value > 1.5:
            peak_color = C_YELLOW
            peak_icon = "🔥🔥"
            peak_status = "VERY HIGH"
        elif peak_value > 1.0:
            peak_color = C_CYAN
            peak_icon = "🔥"
            peak_status = "HIGH"
        elif peak_value > 0.5:
            peak_color = C_GREEN
            peak_icon = "⚡"
            peak_status = "MODERATE"
        else:
            peak_color = C_GREEN
            peak_icon = "✓"
            peak_status = "NORMAL"
        
        # Create peak activity bar (0.0 - 2.0 scale)
        from .ui_terminal import make_peak_activity_bar
        peak_bar = make_peak_activity_bar(peak_value, width=ui_w - 40)
        
        # Enhanced header with icon and color
        print_line(f"{C_BOLD}{peak_icon} PEAK LAYER ACTIVITY - {peak_color}{peak_status}{C_RESET}", ui_w)
        print_separator(ui_w, 'thin')
        
        # Show peak layer with enhanced formatting
        print_line(f"Layer: {C_BOLD}{peak_color}{peak_layer_name}{C_RESET} | Value: {C_BOLD}{peak_color}{peak_value:.3f}{C_RESET}", ui_w)
        print_separator(ui_w, 'thin')
        
        # Visual bar display
        print_line(peak_bar, ui_w)
        print_separator(ui_w, 'thin')
        
        # Warning if extreme with prominent display
        if peak_value > 2.0:
            print_line(f"{C_BOLD}{C_RED}⚠️  🔴 EXTREME ACTIVITY! Check training stability! 🔴 ⚠️{C_RESET}", ui_w)
        elif peak_value > 1.5:
            print_line(f"{C_BOLD}{C_YELLOW}⚠️  Unusually high activity - Monitor closely{C_RESET}", ui_w)
        elif peak_value > 1.0:
            print_line(f"{C_CYAN}ℹ️  High activity detected - This is normal during training{C_RESET}", ui_w)
        else:
            print_line(f"{C_GREEN}✓ Activity levels within normal range{C_RESET}", ui_w)
        
        # Stream overview removed as per user request
        
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
    
    # Control hints - Enhanced with all action buttons
    print_line(f"{C_BOLD}⌨️  KEYBOARD SHORTCUTS{C_RESET}", ui_w)
    print_separator(ui_w, 'thin')
    print_two_columns(
        f"{C_CYAN}P{C_RESET} Pause/Resume  │  {C_CYAN}V{C_RESET} Validation",
        f"{C_CYAN}S{C_RESET} Change View  │  {C_CYAN}ENTER{C_RESET} Config",
        ui_w
    )
    print_two_columns(
        f"{C_CYAN}C{C_RESET} Save Checkpoint  │  {C_CYAN}Q{C_RESET} Quit",
        f"{C_CYAN}ESC{C_RESET} Emergency Stop",
        ui_w
    )
    sys.stdout.write("\n")
    sys.stdout.flush()


def _draw_activity_display(activities, display_mode, available_lines, ui_w, 
                           bar_width_single, bar_width_double):
    """
    Draw layer activity based on selected display mode
    
    MODE 0: 2-Column Detailed Layout (NEW!)
      - Left column: BACKWARD TRUNK
      - Right column: FORWARD TRUNK
      - Bottom center: FINAL FUSION
    
    MODE 1: Grouped by Trunk → Sorted by Activity
    MODE 2: Flat List → Sorted by Position  
    MODE 3: Flat List → Sorted by Activity
    
    Args:
        activities: List of (layer_name, activity%, trend, raw_value) tuples
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
        # MODE 0: NEW 2-Column Detailed Layout
        # Separate layers by type
        backward_layers = []
        forward_layers = []
        fusion_layers = []
        
        for name, act, trend, raw in activities:
            if "Backward" in name and "Fuse" not in name:
                backward_layers.append((name, act, trend, raw))
            elif "Forward" in name and "Fuse" not in name:
                forward_layers.append((name, act, trend, raw))
            else:
                fusion_layers.append((name, act, trend, raw))
        
        # Calculate overall activity
        backward_overall = int(np.mean([act for _, act, _, _ in backward_layers])) if backward_layers else 0
        forward_overall = int(np.mean([act for _, act, _, _ in forward_layers])) if forward_layers else 0
        
        # Print headers side-by-side
        print_two_columns(
            f"{C_BOLD}🔥 BACKWARD TRUNK{C_RESET} ({backward_overall}%)",
            f"{C_BOLD}⚡ FORWARD TRUNK{C_RESET} ({forward_overall}%)",
            ui_w
        )
        print_separator(ui_w, 'thin')
        
        # Print layers side-by-side
        max_rows = max(len(backward_layers), len(forward_layers))
        for i in range(max_rows):
            left_str = ""
            right_str = ""
            
            if i < len(backward_layers):
                name, act, trend, raw = backward_layers[i]
                # Extract number from "Backward N" format (e.g., "Backward 1" -> "B1")
                if name.startswith("Backward ") and name.split()[-1].isdigit():
                    short_name = f"B{name.split()[-1]}"
                else:
                    short_name = name[:4]  # Fallback: first 4 chars
                left_str = f"{short_name:>4}: {get_bar_for_layer(name, act, bar_width_double)} {act:>3}%"
            
            if i < len(forward_layers):
                name, act, trend, raw = forward_layers[i]
                # Extract number from "Forward N" format (e.g., "Forward 1" -> "F1")
                if name.startswith("Forward ") and name.split()[-1].isdigit():
                    short_name = f"F{name.split()[-1]}"
                else:
                    short_name = name[:4]  # Fallback: first 4 chars
                right_str = f"{short_name:>4}: {get_bar_for_layer(name, act, bar_width_double)} {act:>3}%"
            
            print_two_columns(left_str, right_str, ui_w)
        
        # Print fusion layers centered at bottom
        if fusion_layers:
            print_separator(ui_w, 'thin')
            for name, act, trend, raw in fusion_layers:
                if "Final" in name:
                    # Final fusion gets special treatment - centered
                    print_line(f"{C_BOLD}{name:^20}{C_RESET}: {get_bar_for_layer(name, act, bar_width_single)} {act:>3}%", ui_w)
                else:
                    # Other fusion layers (Backward Fuse, Forward Fuse)
                    print_line(f"{name:<20}: {get_bar_for_layer(name, act, bar_width_single)} {act:>3}%", ui_w)
    
    elif display_mode == 1:
        # MODE 1: Grouped by Trunk → Sorted by Activity
        # Separate layers by type
        backward_layers = []
        forward_layers = []
        fusion_layers = []
        
        for name, act, trend, raw in activities:
            if "Backward" in name and "Fuse" not in name:
                backward_layers.append((name, act, trend, raw))
            elif "Forward" in name and "Fuse" not in name:
                forward_layers.append((name, act, trend, raw))
            else:
                fusion_layers.append((name, act, trend, raw))
        
        # Sort by activity
        backward_layers = sorted(backward_layers, key=lambda x: x[1], reverse=True)
        forward_layers = sorted(forward_layers, key=lambda x: x[1], reverse=True)
        
        backward_overall = int(np.mean([act for _, act, _, _ in backward_layers])) if backward_layers else 0
        forward_overall = int(np.mean([act for _, act, _, _ in forward_layers])) if forward_layers else 0
        
        print_line(f"{C_BOLD}🔥 BACKWARD TRUNK (sorted){C_RESET} - Overall: {make_bar(backward_overall, bar_width_single)} {backward_overall}%", ui_w)
        print_separator(ui_w, 'single')
        
        if available_lines >= num_activities:
            for name, act, trend, raw in backward_layers:
                print_line(f"{name:<20}: {get_bar_for_layer(name, act, bar_width_single)} {act:>3}%", ui_w)
        else:
            for row in range(0, len(backward_layers), 2):
                left = backward_layers[row]
                right = backward_layers[row+1] if row+1 < len(backward_layers) else None
                left_str = f"{left[0]:<18}:{get_bar_for_layer(left[0], left[1], bar_width_double)}{left[1]:3}%"
                right_str = f"{right[0]:<18}:{get_bar_for_layer(right[0], right[1], bar_width_double)}{right[1]:3}%" if right else ""
                print_two_columns(left_str, right_str, ui_w)
        
        print_separator(ui_w, 'double')
        print_line(f"{C_BOLD}⚡ FORWARD TRUNK (sorted){C_RESET} - Overall: {make_bar(forward_overall, bar_width_single)} {forward_overall}%", ui_w)
        print_separator(ui_w, 'single')
        
        if available_lines >= num_activities:
            for name, act, trend, raw in forward_layers:
                print_line(f"{name:<20}: {get_bar_for_layer(name, act, bar_width_single)} {act:>3}%", ui_w)
        else:
            for row in range(0, len(forward_layers), 2):
                left = forward_layers[row]
                right = forward_layers[row+1] if row+1 < len(forward_layers) else None
                left_str = f"{left[0]:<18}:{get_bar_for_layer(left[0], left[1], bar_width_double)}{left[1]:3}%"
                right_str = f"{right[0]:<18}:{get_bar_for_layer(right[0], right[1], bar_width_double)}{right[1]:3}%" if right else ""
                print_two_columns(left_str, right_str, ui_w)
        
        # Print fusion layers at end
        if fusion_layers:
            print_separator(ui_w, 'double')
            for name, act, trend, raw in fusion_layers:
                print_line(f"{name:<20}: {get_bar_for_layer(name, act, bar_width_single)} {act:>3}%", ui_w)
    
    elif display_mode == 2:
        # MODE 2: Flat List → Sorted by Position
        if available_lines >= num_activities:
            for name, act, trend, raw in activities:
                print_line(f"{name:<20}: {get_bar_for_layer(name, act, bar_width_single)} {act:>3}%", ui_w)
        else:
            for row in range(0, num_activities, 2):
                left = activities[row]
                right = activities[row+1] if row+1 < num_activities else None
                left_str = f"{left[0]:<18}:{get_bar_for_layer(left[0], left[1], bar_width_double)}{left[1]:3}%"
                right_str = f"{right[0]:<18}:{get_bar_for_layer(right[0], right[1], bar_width_double)}{right[1]:3}%" if right else ""
                print_two_columns(left_str, right_str, ui_w)
    
    else:  # display_mode == 3
        # MODE 3: Flat List → Sorted by Activity
        sorted_acts = sorted(activities, key=lambda x: x[1], reverse=True)
        
        if available_lines >= num_activities:
            for name, act, trend, raw in sorted_acts:
                print_line(f"{name:<20}: {get_bar_for_layer(name, act, bar_width_single)} {act:>3}%", ui_w)
        else:
            for row in range(0, num_activities, 2):
                left = sorted_acts[row]
                right = sorted_acts[row+1] if row+1 < num_activities else None
                left_str = f"{left[0]:<18}:{get_bar_for_layer(left[0], left[1], bar_width_double)}{left[1]:3}%"
                right_str = f"{right[0]:<18}:{get_bar_for_layer(right[0], right[1], bar_width_double)}{right[1]:3}%" if right else ""
                print_two_columns(left_str, right_str, ui_w)
