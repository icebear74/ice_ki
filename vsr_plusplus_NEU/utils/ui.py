"""
UI Utilities - Box drawing and display functions
"""

# ANSI Color Codes
C_GREEN = "\033[92m"
C_CYAN = "\033[96m"
C_RED = "\033[91m"
C_YELLOW = "\033[93m"
C_BOLD = "\033[1m"
C_RESET = "\033[0m"
ANSI_CLEAR = "\033[2J"
ANSI_HOME = "\033[H"


def draw_box(title, lines, width=75):
    """
    Draw a box with title and content lines
    
    Args:
        title: Box title
        lines: List of content lines
        width: Box width
    """
    print("╔" + "═" * width + "╗")
    
    # Title
    padding = (width - len(title)) // 2
    print("║" + " " * padding + title + " " * (width - padding - len(title)) + "║")
    
    print("╠" + "═" * width + "╣")
    
    # Content lines
    for line in lines:
        # Remove ANSI codes for length calculation
        import re
        clean_line = re.sub(r'\033\[[0-9;]*m', '', line)
        padding = width - len(clean_line)
        print("║ " + line + " " * padding + "║")
    
    print("╚" + "═" * width + "╝")


def draw_auto_tune_results(config):
    """
    Display auto-tune results in formatted box
    
    Args:
        config: Auto-tuned configuration
    """
    print("\n")
    
    lines = [
        f"{C_GREEN}✅ OPTIMAL CONFIGURATION FOUND{C_RESET}",
        "",
        f"  Features: {config['n_feats']} | Batch: {config['batch_size']} | Blocks: {config['n_blocks']}",
        f"  Accumulation Steps: {config['accumulation_steps']}",
        "",
        f"  ⏱️  Speed: {config['measured_speed']:.2f}s/iter",
        f"  💾 VRAM: {config['measured_vram']:.2f}GB",
        f"  🔢 Params: {config['total_params']/1e6:.2f}M",
        "",
        f"{C_YELLOW}📸 Press ENTER to continue training...{C_RESET}"
    ]
    
    draw_box("🔧 AUTO-TUNING COMPLETE", lines, width=71)


def print_checkpoint_info(checkpoints):
    """
    Display checkpoint table
    
    Args:
        checkpoints: List of checkpoint info dicts
    """
    if not checkpoints:
        print(f"\n{C_YELLOW}No checkpoints found{C_RESET}\n")
        return
    
    print(f"\n{C_CYAN}{'='*80}{C_RESET}")
    print(f"{C_BOLD}Available Checkpoints:{C_RESET}")
    print(f"{C_CYAN}{'='*80}{C_RESET}")
    
    print(f"{'Step':<12} {'Type':<15} {'Quality':<12} {'Loss':<12} {'Size':<10}")
    print(f"{'-'*80}")
    
    for ckpt in checkpoints:
        step = f"{ckpt['step']:,}" if ckpt['step'] > 0 else "Emergency"
        ckpt_type = ckpt.get('type', 'regular')
        quality = f"{ckpt.get('quality', 0)*100:.1f}%" if 'quality' in ckpt else "N/A"
        loss = f"{ckpt.get('loss', 0):.4f}" if 'loss' in ckpt else "N/A"
        size = f"{ckpt.get('size_mb', 0):.1f}MB" if 'size_mb' in ckpt else "N/A"
        
        print(f"{step:<12} {ckpt_type:<15} {quality:<12} {loss:<12} {size:<10}")
    
    print(f"{C_CYAN}{'='*80}{C_RESET}\n")


def draw_ui(step, epoch, losses, lr, speed, vram, activities, config, metrics, adaptive_info):
    """
    Draw main training UI with box drawing characters
    
    Args:
        step: Current global step
        epoch: Current epoch
        losses: Dict with loss components
        lr: Current learning rate
        speed: Training speed (s/iter)
        vram: VRAM usage (GB)
        activities: Dict with trunk activities
        config: Model configuration
        metrics: Validation metrics
        adaptive_info: Adaptive system info
    """
    print(f"{ANSI_CLEAR}{ANSI_HOME}")
    
    # Header
    print(f"{C_CYAN}{'='*80}{C_RESET}")
    print(f"{C_BOLD}VSR++ Training - Step {step:,} | Epoch {epoch}{C_RESET}")
    print(f"{C_CYAN}{'='*80}{C_RESET}\n")
    
    # Loss Information
    print(f"{C_BOLD}Loss Components:{C_RESET}")
    print(f"  Total: {C_GREEN}{losses.get('total', 0):.6f}{C_RESET} | "
          f"L1: {losses.get('l1', 0):.6f} | "
          f"MS: {losses.get('ms', 0):.6f} | "
          f"Grad: {losses.get('grad', 0):.6f}")
    
    # Training Info
    print(f"\n{C_BOLD}Training Info:{C_RESET}")
    print(f"  LR: {C_CYAN}{lr:.2e}{C_RESET} | "
          f"Speed: {speed:.2f}s/iter | "
          f"VRAM: {vram:.2f}GB")
    
    # Model Config
    print(f"\n{C_BOLD}Model Config:{C_RESET}")
    print(f"  Features: {config.get('n_feats', 128)} | "
          f"Blocks: {config.get('n_blocks', 32)} | "
          f"Batch: {config.get('batch_size', 4)}")
    
    # Quality Metrics
    if metrics:
        print(f"\n{C_BOLD}Quality Metrics:{C_RESET}")
        print(f"  LR Quality: {C_YELLOW}{metrics.get('lr_quality', 0)*100:.1f}%{C_RESET} | "
              f"KI Quality: {C_GREEN}{metrics.get('ki_quality', 0)*100:.1f}%{C_RESET} | "
              f"Improvement: {C_CYAN}{metrics.get('improvement', 0)*100:.1f}%{C_RESET}")
    
    # Adaptive System
    if adaptive_info:
        print(f"\n{C_BOLD}Adaptive System:{C_RESET}")
        l1_w, ms_w, grad_w = adaptive_info.get('loss_weights', (0.6, 0.2, 0.2))
        print(f"  Loss Weights: L1={l1_w:.2f} | MS={ms_w:.2f} | Grad={grad_w:.2f}")
        print(f"  Grad Clip: {adaptive_info.get('grad_clip', 1.5):.3f}")
        
        if adaptive_info.get('aggressive_mode'):
            print(f"  {C_RED}⚡ AGGRESSIVE MODE ACTIVE{C_RESET}")
    
    # Activity
    if activities:
        back_act = activities.get('backward_trunk', [])
        forw_act = activities.get('forward_trunk', [])
        if back_act and forw_act:
            avg_back = sum(back_act) / len(back_act) if back_act else 0
            avg_forw = sum(forw_act) / len(forw_act) if forw_act else 0
            print(f"\n{C_BOLD}Block Activity:{C_RESET}")
            print(f"  Backward Trunk: {avg_back:.4f} | Forward Trunk: {avg_forw:.4f}")
    
    print(f"\n{C_CYAN}{'='*80}{C_RESET}")
