import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import cv2, os, time, shutil, sys, glob, json, re, random
import numpy as np
from datetime import datetime
from model_vsrppp_v2 import VSRTriplePlus_3x 
import select
import termios
import tty
import subprocess
import socket
from adaptive_system import FullAdaptiveSystem

# ============================================================
# --- KONFIGURATION & PFADE ---
# ============================================================
DATA_ROOT      = "/mnt/data/training/Universal/Mastermodell/Learn"
CONFIG_FILE    = os.path.join(DATA_ROOT, "train_config.json")
DATASET_ROOT   = "/mnt/data/training/Dataset/Universal/Mastermodell"

N_BLOCKS       = 32
BATCH_SIZE     = 6
NUM_WORKERS    = 4
PATCH_GT, PATCH_LR = 540, 180

defaults = {
    "LR_EXPONENT": -5,
    "WEIGHT_DECAY": 0.001,
    "MAX_STEPS": 100000,
    "VAL_STEP_EVERY": 500,
    "SAVE_STEP_EVERY": 5000,
    "LOG_TBOARD_EVERY": 100,
    "LOG_LAYERS_EVERY": 100,
    "HIST_STEP_EVERY": 500,
    "ACCUMULATION_STEPS": 1,
    "DISPLAY_MODE": 0,
    "GRAD_CLIP": 1.5
}

activity_history = {i+1: [] for i in range(N_BLOCKS)}
loss_history = []
TREND_WINDOW = 50
WINDOW_SIZE = 50
ema_grads = {}
alpha = 0.2
last_term_size = (0, 0)
last_quality_metrics = {
    'lr_quality': 0.0,
    'ki_quality': 0.0,
    'improvement': 0.0
}

C_GREEN, C_GRAY, C_RESET, C_BOLD, C_CYAN, C_RED, C_YELLOW = "\033[92m", "\033[90m", "\033[0m", "\033[1m", "\033[96m", "\033[91m", "\033[93m"
ANSI_HOME, ANSI_CLEAR = "\033[H", "\033[2J"
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

DISPLAY_MODE_NAMES = [
    "Grouped by Trunk → Sorted by Position",
    "Grouped by Trunk → Sorted by Activity",
    "Flat List → Sorted by Position",
    "Flat List → Sorted by Activity"
]

# --- QUALITY METRICS (IN %) ---
def calculate_psnr(img1, img2):
    """Calculate PSNR between two images (tensor 0-1 range)"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 50.0  # Perfect = 50 dB
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images (simplified, returns 0-1)"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = F.avg_pool2d(img1, 11, stride=1, padding=5)
    mu2 = F.avg_pool2d(img2, 11, stride=1, padding=5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 ** 2, 11, stride=1, padding=5) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, 11, stride=1, padding=5) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, 11, stride=1, padding=5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

def quality_to_percent(psnr, ssim):
    """
    Convert PSNR + SSIM to a single Quality Score (0-100%)
    
    PSNR scale:
      20 dB = 0%  (sehr schlecht)
      30 dB = 33% (mittelmäßig)
      40 dB = 66% (gut)
      50 dB = 100% (perfekt)
    
    SSIM: bereits 0-1 (direkt zu %)
    
    Final: 50% SSIM + 50% PSNR
    """
    # PSNR normalisieren (20-50 dB → 0-100%)
    psnr_percent = min(100.0, max(0.0, (psnr - 20.0) * 3.33))
    
    # SSIM zu %
    ssim_percent = ssim * 100.0
    
    # Kombiniert (50/50 Gewichtung)
    quality = (psnr_percent * 0.5) + (ssim_percent * 0.5)
    
    return quality

# --- HELPER FUNKTIONEN ---
def cleanup_logs(log_base, min_events=15):
    if not os.path.exists(log_base): return
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    subdirs = [d for d in os.listdir(log_base) if os.path.isdir(os.path.join(log_base, d)) and d != "active_run"]
    if not subdirs: return

    print(f"\n{C_CYAN}🔍 Hybrid-Check von {len(subdirs)} Log-Ordnern...{C_RESET}")
    deleted = 0
    for i, subdir in enumerate(subdirs):
        prog = (i + 1) / len(subdirs) * 100
        sys.stdout.write(f"\r {C_GRAY}[{C_GREEN}{'█' * int(prog//5)}{C_GRAY}{'░' * (20 - int(prog//5))}{C_GRAY}] {prog:>5.1f}%{C_RESET}")
        sys.stdout.flush()
        path = os.path.join(log_base, subdir)
        try:
            size_mb = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))) / (1024*1024)
            if size_mb < 0.5: shutil.rmtree(path); deleted += 1; continue
            if size_mb > 20.0: continue
            ea = EventAccumulator(path, size_guidance={'scalars': 20})
            ea.Reload()
            tags = ea.Tags().get('scalars', [])
            if not tags or len(ea.Scalars(tags[0])) < min_events:
                shutil.rmtree(path); deleted += 1
        except: pass
    sys.stdout.write("\n")
    if deleted > 0: print(f"{C_CYAN}🧹 CLEANUP: {deleted} Leichen entfernt.{C_RESET}\n")

def is_tensorboard_running(port=6006):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0
    except:
        return False

def start_tensorboard(log_dir, port=6006):
    try:
        subprocess.run(['pkill', '-f', 'tensorboard'], stderr=subprocess.DEVNULL)
        time.sleep(1)
        cmd = ['tensorboard', f'--logdir={log_dir}', f'--port={port}', '--bind_all', '--reload_interval=5']
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(10):
            time.sleep(0.5)
            if is_tensorboard_running(port):
                print(f"{C_GREEN}✓ TensorBoard started on http://localhost:{port}{C_RESET}")
                return True
        return True
    except Exception as e:
        print(f"{C_RED}✗ Failed to start TensorBoard: {e}{C_RESET}")
        return False

def get_visible_len(text): return len(ANSI_ESCAPE.sub('', text))

def make_bar(percent, width):
    width = max(5, width)
    filled = max(0, min(width, int((percent / 100.0) * width)))
    return f"{C_GREEN}{'█' * filled}{C_GRAY}{'░' * (width - filled)}{C_RESET}"

def format_time(seconds):
    d, h, m = int(seconds // 86400), int((seconds % 86400) // 3600), int((seconds % 3600) // 60)
    return f"{d}d {h}h {m}m" if d > 0 else f"{h}h {m}m"

def load_config():
    cfg = defaults.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f: cfg.update(json.load(f))
        except: pass
    return cfg

def save_config(cfg):
    with open(CONFIG_FILE, 'w') as f_out: json.dump(cfg, f_out, indent=4)

class HybridLoss(nn.Module):
    """100% self-trained loss: L1 + Multi-Scale + Gradient"""
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target, l1_weight=0.7, ms_weight=0.2, grad_weight=0.1):
        pred = torch.clamp(pred, 0.0, 1.0)
        l1_loss = self.l1(pred, target)
        
        pred_h = F.avg_pool2d(pred, 2)
        target_h = F.avg_pool2d(target, 2)
        ms_loss = 0.5 * self.l1(pred_h, target_h)
        
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        grad_loss = self.l1(pred_dx, target_dx) + self.l1(pred_dy, target_dy)
        
        total = l1_weight * l1_loss + ms_weight * ms_loss + grad_weight * grad_loss
        
        return {
            'l1': l1_loss.item(),
            'ms': ms_loss.item(), 
            'grad': grad_loss.item(),
            'total_tensor': total,
            'l1_tensor': l1_loss,
            'ms_tensor': ms_loss,
            'grad_tensor': grad_loss
        }

def calculate_trends(activities):
    trends = []
    for layer_id, current_val in enumerate(activities, 1):
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

def calculate_convergence_status(loss_history):
    if len(loss_history) < 100:
        return "Warming up..."
    
    recent = loss_history[-100:]
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent, 1)[0]
    
    if slope < -0.00005:
        return f"{C_GREEN}Converging ✓{C_RESET}"
    elif abs(slope) < 0.00005:
        return f"{C_CYAN}Plateauing ⚠{C_RESET}"
    else:
        return f"{C_RED}Diverging ✗{C_RESET}"

def get_activity_data(model):
    m = model.module if hasattr(model, 'module') else model
    
    if not hasattr(m, 'get_layer_activity'):
        return [(i+1, 0, 0, 0.0) for i in range(30)]

    activities_raw = m.get_layer_activity()
    trends = calculate_trends(activities_raw)
    max_val = max(activities_raw) if max(activities_raw) > 1e-12 else 1e-12
    
    activities = [(i+1, int((v / max_val) * 100), trends[i], v) 
                  for i, v in enumerate(activities_raw)]
    
    return activities

def get_adaptive_status_with_notification(adaptive_system):
    """Get adaptive status including last notification"""
    status = adaptive_system.get_status()
    status['last_notification'] = adaptive_system.get_last_notification()
    return status

def draw_ui(step, epoch, losses, it_time, activities, cfg, num_images, steps_per_epoch, current_epoch_step, adaptive_status=None, paused=False):
    global last_term_size, last_quality_metrics
    term_size = shutil.get_terminal_size()
    
    if term_size != last_term_size:
        print(ANSI_CLEAR)
        last_term_size = term_size
    
    print(ANSI_CLEAR + ANSI_HOME + "\033[?25l")
    
    ui_w = max(90, term_size.columns - 4)
    total_prog = (step / cfg["MAX_STEPS"]) * 100 if cfg["MAX_STEPS"] > 0 else 0
    total_eta = format_time((cfg["MAX_STEPS"] - step) * it_time) if not paused else "PAUSIERT"
    epoch_prog = (current_epoch_step / steps_per_epoch) * 100 if steps_per_epoch > 0 else 0
    epoch_eta = format_time((steps_per_epoch - current_epoch_step) * it_time) if not paused else "PAUSIERT"

    def print_line(content):
        padding = max(0, ui_w - get_visible_len(content) - 4)
        sys.stdout.write(f" ║ {content}{' ' * padding} ║\n")

    # Header
    sys.stdout.write(f" {C_GRAY}╔{'═'*(ui_w-2)}╗{C_RESET}\n")
    status = f"{C_RED}⏸ PAUSIERT (P){C_RESET}" if paused else f"{C_GREEN}▶ RUNNING{C_RESET}"
    print_line(f"{C_BOLD}MISSION CONTROL{C_RESET} │ {status} │ STEP: {step}/{cfg['MAX_STEPS']}")
    
    # Combined Progress Line: Overall (left) | Epoch (right)
    overall_bar_width = (ui_w - 80) // 2  # Shorter bars
    epoch_bar_width = (ui_w - 80) // 2
    overall_bar = make_bar(total_prog, overall_bar_width)
    epoch_bar = make_bar(epoch_prog, epoch_bar_width)
    print_line(f"TOTAL: {overall_bar} {total_prog:>4.1f}% ETA:{total_eta} │ EPOCH {epoch}: {epoch_bar} {epoch_prog:>4.1f}% ETA:{epoch_eta}")
    sys.stdout.write(f" {C_GRAY}╠{'═'*(ui_w-2)}╣{C_RESET}\n")
    
    # Adaptive Change Notification (if any)
    if adaptive_status:
        notification = adaptive_status.get('last_notification')
        if notification:
            notif_color = C_YELLOW if notification['type'] in ['plateau', 'divergence'] else C_CYAN
            step_info = f"@ Step {notification['step']}" if notification['step'] else ""
            print_line(f"{notif_color}{notification['message']}{C_RESET} {step_info}")
            if notification.get('details'):
                print_line(f"  {C_GRAY}{notification['details']}{C_RESET}")
            sys.stdout.write(f" {C_GRAY}╠{'─'*(ui_w-2)}╣{C_RESET}\n")
    
    # Adaptive Parameters Section
    if adaptive_status:
        lr = adaptive_status['lr']
        l1_w, ms_w, grad_w = adaptive_status['loss_weights']
        grad_clip = adaptive_status['grad_clip']
        best_loss = adaptive_status['best_loss']
        plateau_count = adaptive_status['plateau_counter']
        plateau_max = 300  # from AdaptiveLRScheduler patience
        
        print_line(f"{C_BOLD}🤖 ADAPTIVE PARAMETERS{C_RESET}")
        sys.stdout.write(f" {C_GRAY}╠{'─'*(ui_w-2)}╣{C_RESET}\n")
        print_line(f"LR: {C_CYAN}{lr:.2e}{C_RESET} │ Grad Clip: {C_CYAN}{grad_clip:.2f}{C_RESET} │ Best Loss: {C_CYAN}{best_loss:.6f}{C_RESET} │ Plateau: {plateau_count}/{plateau_max}")
        print_line(f"Loss Weights → L1: {C_CYAN}{l1_w:.3f}{C_RESET} │ MS: {C_CYAN}{ms_w:.3f}{C_RESET} │ Grad: {C_CYAN}{grad_w:.3f}{C_RESET}")
        sys.stdout.write(f" {C_GRAY}╠{'═'*(ui_w-2)}╣{C_RESET}\n")
    
    # Loss Info
    loss_str = f"L1: {C_CYAN}{losses.get('l1',0):.4f}{C_RESET} │ MS: {C_CYAN}{losses.get('ms',0):.4f}{C_RESET} │ Grad: {C_CYAN}{losses.get('grad',0):.4f}{C_RESET} │ Total: {C_BOLD}{C_GREEN}{losses.get('total',0):.4f}{C_RESET}"
    print_line(f"LOSS → {loss_str}")
    vram = f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB"
    print_line(f"DATASET: {num_images:<7} imgs │ VRAM: {vram} │ SPEED: {it_time:.2f}s/it │ {calculate_convergence_status(loss_history)}")
    sys.stdout.write(f" {C_GRAY}╠{'═'*(ui_w-2)}╣{C_RESET}\n")
    
    # QUALITY METRICS (last validation)
    lr_q = last_quality_metrics['lr_quality']
    ki_q = last_quality_metrics['ki_quality']
    imp = last_quality_metrics['improvement']
    
    print_line(f"{C_BOLD}📊 QUALITY METRICS{C_RESET} (last validation)")
    sys.stdout.write(f" {C_GRAY}╠{'─'*(ui_w-2)}╣{C_RESET}\n")
    
    # LR Quality (Baseline)
    lr_bar = make_bar(lr_q, ui_w-50)
    lr_color = C_RED if lr_q < 50 else (C_YELLOW if lr_q < 70 else C_GREEN)
    print_line(f"LR→GT:   Quality: {lr_bar} {lr_color}{lr_q:>5.1f}%{C_RESET}  (Baseline)")
    
    # KI Quality (Your Model)
    ki_bar = make_bar(ki_q, ui_w-50)
    ki_color = C_RED if ki_q < 60 else (C_YELLOW if ki_q < 80 else C_GREEN)
    print_line(f"KI→GT:   Quality: {ki_bar} {ki_color}{ki_q:>5.1f}%{C_RESET}  (Your Model)")
    
    # Improvement
    imp_bar = make_bar(min(100, imp), ui_w-50)
    imp_color = C_RED if imp < 10 else (C_YELLOW if imp < 25 else C_GREEN)
    imp_sign = "+" if imp >= 0 else ""
    print_line(f"Improve: {imp_bar} {imp_color}{imp_sign}{imp:>5.1f}%{C_RESET}  (Gain!)")
    
    sys.stdout.write(f" {C_GRAY}╠{'═'*(ui_w-2)}╣{C_RESET}\n")

    # LAYER DISPLAY - 4 MODI
    display_mode = cfg.get("DISPLAY_MODE", 0)
    available_lines = term_size.lines - 22
    
    mode_name = DISPLAY_MODE_NAMES[display_mode]
    print_line(f"{C_BOLD}📊 VIEW MODE: {mode_name}{C_RESET}")
    sys.stdout.write(f" {C_GRAY}╠{'─'*(ui_w-2)}╣{C_RESET}\n")
    
    if display_mode == 0:
        # MODE 0: Grouped by Trunk → Sorted by Position
        backward = activities[:15]
        forward = activities[15:30]
        fusion = activities[30:32] if len(activities) >= 32 else []
        backward_overall = int(np.mean([act for _, act, _, _ in backward]))
        forward_overall = int(np.mean([act for _, act, _, _ in forward]))
        fusion_overall = int(np.mean([act for _, act, _, _ in fusion])) if fusion else 0
        
        print_line(f"{C_BOLD}🔥 BACKWARD TRUNK (L1-L15){C_RESET} - Overall: {make_bar(backward_overall, 20)} {backward_overall}%")
        sys.stdout.write(f" {C_GRAY}╠{'─'*(ui_w-2)}╣{C_RESET}\n")
        
        if available_lines >= 32:
            for idx, act, trend, raw in backward:
                print_line(f"LAYER {idx:>2}: {make_bar(act, ui_w-25)} {act:>3}%")
        else:
            for row in range(0, 15, 2):
                left = backward[row]
                right = backward[row+1] if row+1 < 15 else None
                left_str = f"L{left[0]:02d}:{make_bar(left[1], (ui_w-30)//2)}{left[1]:3}%"
                right_str = f"L{right[0]:02d}:{make_bar(right[1], (ui_w-30)//2)}{right[1]:3}%" if right else ""
                print_line(f"{left_str} │ {right_str}")
        
        sys.stdout.write(f" {C_GRAY}╠{'═'*(ui_w-2)}╣{C_RESET}\n")
        print_line(f"{C_BOLD}⚡ FORWARD TRUNK (L16-L30){C_RESET} - Overall: {make_bar(forward_overall, 20)} {forward_overall}%")
        sys.stdout.write(f" {C_GRAY}╠{'─'*(ui_w-2)}╣{C_RESET}\n")
        
        if available_lines >= 32:
            for idx, act, trend, raw in forward:
                print_line(f"LAYER {idx:>2}: {make_bar(act, ui_w-25)} {act:>3}%")
        else:
            for row in range(0, 15, 2):
                left = forward[row]
                right = forward[row+1] if row+1 < 15 else None
                left_str = f"L{left[0]:02d}:{make_bar(left[1], (ui_w-30)//2)}{left[1]:3}%"
                right_str = f"L{right[0]:02d}:{make_bar(right[1], (ui_w-30)//2)}{right[1]:3}%" if right else ""
                print_line(f"{left_str} │ {right_str}")
        
        if fusion:
            sys.stdout.write(f" {C_GRAY}╠{'═'*(ui_w-2)}╣{C_RESET}\n")
            print_line(f"{C_BOLD}🔗 FUSION (L31-L32){C_RESET} - Overall: {make_bar(fusion_overall, 20)} {fusion_overall}%")
            sys.stdout.write(f" {C_GRAY}╠{'─'*(ui_w-2)}╣{C_RESET}\n")
            
            if available_lines >= 32:
                for idx, act, trend, raw in fusion:
                    print_line(f"LAYER {idx:>2}: {make_bar(act, ui_w-25)} {act:>3}%")
            else:
                left = fusion[0]
                right = fusion[1] if len(fusion) > 1 else None
                left_str = f"L{left[0]:02d}:{make_bar(left[1], (ui_w-30)//2)}{left[1]:3}%"
                right_str = f"L{right[0]:02d}:{make_bar(right[1], (ui_w-30)//2)}{right[1]:3}%" if right else ""
                print_line(f"{left_str} │ {right_str}")
    
    elif display_mode == 1:
        # MODE 1: Grouped by Trunk → Sorted by Activity
        backward = sorted(activities[:15], key=lambda x: x[1], reverse=True)
        forward = sorted(activities[15:30], key=lambda x: x[1], reverse=True)
        fusion = sorted(activities[30:32], key=lambda x: x[1], reverse=True) if len(activities) >= 32 else []
        backward_overall = int(np.mean([act for _, act, _, _ in backward]))
        forward_overall = int(np.mean([act for _, act, _, _ in forward]))
        fusion_overall = int(np.mean([act for _, act, _, _ in fusion])) if fusion else 0
        
        print_line(f"{C_BOLD}🔥 BACKWARD TRUNK (L1-L15 sorted){C_RESET} - Overall: {make_bar(backward_overall, 20)} {backward_overall}%")
        sys.stdout.write(f" {C_GRAY}╠{'─'*(ui_w-2)}╣{C_RESET}\n")
        
        if available_lines >= 32:
            for idx, act, trend, raw in backward:
                print_line(f"LAYER {idx:>2}: {make_bar(act, ui_w-25)} {act:>3}%")
        else:
            for row in range(0, 15, 2):
                left = backward[row]
                right = backward[row+1] if row+1 < 15 else None
                left_str = f"L{left[0]:02d}:{make_bar(left[1], (ui_w-30)//2)}{left[1]:3}%"
                right_str = f"L{right[0]:02d}:{make_bar(right[1], (ui_w-30)//2)}{right[1]:3}%" if right else ""
                print_line(f"{left_str} │ {right_str}")
        
        sys.stdout.write(f" {C_GRAY}╠{'═'*(ui_w-2)}╣{C_RESET}\n")
        print_line(f"{C_BOLD}⚡ FORWARD TRUNK (L16-L30 sorted){C_RESET} - Overall: {make_bar(forward_overall, 20)} {forward_overall}%")
        sys.stdout.write(f" {C_GRAY}╠{'─'*(ui_w-2)}╣{C_RESET}\n")
        
        if available_lines >= 32:
            for idx, act, trend, raw in forward:
                print_line(f"LAYER {idx:>2}: {make_bar(act, ui_w-25)} {act:>3}%")
        else:
            for row in range(0, 15, 2):
                left = forward[row]
                right = forward[row+1] if row+1 < 15 else None
                left_str = f"L{left[0]:02d}:{make_bar(left[1], (ui_w-30)//2)}{left[1]:3}%"
                right_str = f"L{right[0]:02d}:{make_bar(right[1], (ui_w-30)//2)}{right[1]:3}%" if right else ""
                print_line(f"{left_str} │ {right_str}")
        
        if fusion:
            sys.stdout.write(f" {C_GRAY}╠{'═'*(ui_w-2)}╣{C_RESET}\n")
            print_line(f"{C_BOLD}🔗 FUSION (L31-L32 sorted){C_RESET} - Overall: {make_bar(fusion_overall, 20)} {fusion_overall}%")
            sys.stdout.write(f" {C_GRAY}╠{'─'*(ui_w-2)}╣{C_RESET}\n")
            
            if available_lines >= 32:
                for idx, act, trend, raw in fusion:
                    print_line(f"LAYER {idx:>2}: {make_bar(act, ui_w-25)} {act:>3}%")
            else:
                left = fusion[0]
                right = fusion[1] if len(fusion) > 1 else None
                left_str = f"L{left[0]:02d}:{make_bar(left[1], (ui_w-30)//2)}{left[1]:3}%"
                right_str = f"L{right[0]:02d}:{make_bar(right[1], (ui_w-30)//2)}{right[1]:3}%" if right else ""
                print_line(f"{left_str} │ {right_str}")
    
    elif display_mode == 2:
        # MODE 2: Flat List → Sorted by Position
        num_layers = len(activities)
        if available_lines >= num_layers:
            for idx, act, trend, raw in activities:
                print_line(f"LAYER {idx:>2}: {make_bar(act, ui_w-25)} {act:>3}%")
        else:
            for row in range(0, num_layers, 2):
                left = activities[row]
                right = activities[row+1] if row+1 < num_layers else None
                left_str = f"L{left[0]:02d}:{make_bar(left[1], (ui_w-30)//2)}{left[1]:3}%"
                right_str = f"L{right[0]:02d}:{make_bar(right[1], (ui_w-30)//2)}{right[1]:3}%" if right else ""
                print_line(f"{left_str} │ {right_str}")
    
    else:  # display_mode == 3
        # MODE 3: Flat List → Sorted by Activity
        sorted_acts = sorted(activities, key=lambda x: x[1], reverse=True)
        num_layers = len(sorted_acts)
        
        if available_lines >= num_layers:
            for idx, act, trend, raw in sorted_acts:
                print_line(f"LAYER {idx:>2}: {make_bar(act, ui_w-25)} {act:>3}%")
        else:
            for row in range(0, num_layers, 2):
                left = sorted_acts[row]
                right = sorted_acts[row+1] if row+1 < num_layers else None
                left_str = f"L{left[0]:02d}:{make_bar(left[1], (ui_w-30)//2)}{left[1]:3}%"
                right_str = f"L{right[0]:02d}:{make_bar(right[1], (ui_w-30)//2)}{right[1]:3}%" if right else ""
                print_line(f"{left_str} │ {right_str}")

    # Footer
    sys.stdout.write(f" {C_GRAY}╠{'═'*(ui_w-2)}╣{C_RESET}\n")
    nv = cfg['VAL_STEP_EVERY']-(step%cfg['VAL_STEP_EVERY'])
    ns = cfg['SAVE_STEP_EVERY']-(step%cfg['SAVE_STEP_EVERY'])
    print_line(f"VAL IN: {nv:<5} │ SAVE IN: {ns:<5} │ BATCH: {BATCH_SIZE}x{cfg['ACCUMULATION_STEPS']}={BATCH_SIZE*cfg['ACCUMULATION_STEPS']} │ GRAD CLIP: {cfg.get('GRAD_CLIP', 1.0)}")
    sys.stdout.write(f" {C_GRAY}╚{'═'*(ui_w-2)}╝{C_RESET}\n")
    sys.stdout.write(f"{' ' * ((ui_w - 55) // 2)}{C_BOLD}( ENTER: Config | S: Next View | P: Pause | V: Val ){C_RESET}\n")
    sys.stdout.flush()

def live_menu(cfg, optimizer, old_settings, model, step, epoch, loss, it_t, ds_len, steps_ep, curr_ep_step):
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    print("\033[?25h" + ANSI_CLEAR + ANSI_HOME)
    while True:
        print(f"{C_BOLD}🛠️  LIVE CONFIG{C_RESET}\n" + "-"*45)
        keys = list(cfg.keys())
        for idx, k in enumerate(keys): 
            val_display = DISPLAY_MODE_NAMES[cfg[k]] if k == "DISPLAY_MODE" else cfg[k]
            print(f" {idx+1}. {k:<20}: {val_display}")
        print("-" * 45 + "\n 0. ZURÜCK")
        wahl = input("\n Auswahl: ").lower()
        if wahl == "0": break
        try:
            idx = int(wahl) - 1
            if 0 <= idx < len(keys):
                k_name = keys[idx]
                new_val = input(f"Neuer Wert für {k_name}: ")
                if k_name == "LR_EXPONENT":
                    val = int(new_val); cfg[k_name] = val
                    for pg in optimizer.param_groups: pg['lr'] = 10**val
                elif k_name == "GRAD_CLIP":
                    cfg[k_name] = float(new_val)
                elif k_name == "DISPLAY_MODE":
                    cfg[k_name] = int(new_val) % 4
                else:
                    cfg[k_name] = type(cfg[k_name])(new_val)
                save_config(cfg)
        except: pass
    tty.setcbreak(sys.stdin.fileno())
    print(ANSI_CLEAR)

class VSRDataset(Dataset):
    def __init__(self, root_dir, dataset_type='Patches'):
        self.dataset_type = dataset_type
        self.gt_dir = os.path.join(root_dir, dataset_type, "GT")
        self.lr_dir = os.path.join(root_dir, dataset_type, "LR")
        self.patch_lr_dir = os.path.join(root_dir, "Patches", "LR")
        raw_filenames = sorted([f for f in os.listdir(self.gt_dir) if f.endswith('.png')])
        self.filenames = []
        for f in raw_filenames:
            if dataset_type == 'Val':
                if os.path.exists(os.path.join(self.lr_dir, f)): 
                    self.filenames.append((f, self.lr_dir))
                elif os.path.exists(os.path.join(self.patch_lr_dir, f)): 
                    self.filenames.append((f, self.patch_lr_dir))
                else:
                    try: os.remove(os.path.join(self.gt_dir, f))
                    except: pass
            else: 
                self.filenames.append((f, self.lr_dir))
    
    def __len__(self): 
        return len(self.filenames)
    
    def __getitem__(self, idx):
        name, lr_folder = self.filenames[idx]
        gt = cv2.cvtColor(cv2.imread(os.path.join(self.gt_dir, name)), cv2.COLOR_BGR2RGB) / 255.0
        lr_stack = cv2.cvtColor(cv2.imread(os.path.join(lr_folder, name)), cv2.COLOR_BGR2RGB) / 255.0
        lrs = [lr_stack[i*PATCH_LR:(i+1)*PATCH_LR, :] for i in range(5)]
        
        if self.dataset_type == 'Patches' and random.random() > 0.5:
            lrs = [f[:, ::-1].copy() for f in lrs]
            gt = gt[:, ::-1].copy()
        
        if self.dataset_type == 'Patches' and random.random() > 0.5:
            lrs = [f[::-1, :].copy() for f in lrs]
            gt = gt[::-1, :].copy()
        
        if self.dataset_type == 'Patches':
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                lrs = [np.rot90(f, k).copy() for f in lrs]
                gt = np.rot90(gt, k).copy()
        
        return torch.stack([torch.from_numpy(f).permute(2, 0, 1) for f in lrs]).float(), torch.from_numpy(gt).permute(2, 0, 1).float(), name

def train(old_settings):
    global last_quality_metrics
    
    cfg = load_config()
    LOG_BASE, CHECKPOINT_DIR = os.path.join(DATA_ROOT, "logs"), os.path.join(DATA_ROOT, "checkpoints")
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    os.system('clear')
    choice = input("⚠️  [L]öschen oder [F]ortsetzen? (L/F): ").lower()
    if choice == 'l':
        if os.path.exists(LOG_BASE): shutil.rmtree(LOG_BASE)
        if os.path.exists(CHECKPOINT_DIR): shutil.rmtree(CHECKPOINT_DIR)
        if os.path.exists(CONFIG_FILE): os.remove(CONFIG_FILE)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True); save_config(defaults); cfg = defaults.copy()

    cleanup_logs(LOG_BASE)
    
    print(f"\n{C_CYAN}Checking TensorBoard...{C_RESET}")
    if not is_tensorboard_running():
        print(f"{C_YELLOW}Starting TensorBoard...{C_RESET}")
        start_tensorboard(LOG_BASE)
    else:
        print(f"{C_GREEN}✓ TensorBoard running{C_RESET}")
    
    tty.setcbreak(sys.stdin.fileno())
    writer = SummaryWriter(log_dir=os.path.join(LOG_BASE, "active_run"))

    device = torch.device('cuda')
    model = VSRTriplePlus_3x(n_blocks=N_BLOCKS).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=10**cfg["LR_EXPONENT"], weight_decay=cfg["WEIGHT_DECAY"])

    # Initialize Adaptive System
    adaptive_system = FullAdaptiveSystem(optimizer)

    scaler = GradScaler()
    hybrid_criterion = HybridLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["MAX_STEPS"], eta_min=1e-7)
    
    global_step = 0
    ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth")), key=os.path.getmtime)
    if ckpts and choice != 'l':
        ckpt = torch.load(ckpts[-1])
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        global_step = ckpt.get('step', 0)
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    s_time, s_step, paused, do_val = time.time(), global_step, False, False

    try:
        for epoch in range(1, 100000):
            train_ds = VSRDataset(DATASET_ROOT, "Patches")
            steps_per_epoch = max(1, len(train_ds) // (BATCH_SIZE * cfg["ACCUMULATION_STEPS"]))
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
            model.train()
            for i, (lrs, gt, _) in enumerate(train_loader):
                while paused:
                    draw_ui(global_step, epoch, {'l1': 0, 'ms': 0, 'grad': 0, 'total': 0}, 0.1, 
                            get_activity_data(model), cfg, 
                            len(train_ds), steps_per_epoch, (i+1)//cfg["ACCUMULATION_STEPS"], 
                            adaptive_status=get_adaptive_status_with_notification(adaptive_system), paused=True)
                    time.sleep(0.5)
                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        if sys.stdin.read(1).lower() == 'p': paused = False; s_time = time.time(); s_step = global_step
                
                with autocast():
                    output = model(lrs.to(device))
                    gt_gpu = gt.to(device)
                    
                    # Get adaptive loss weights before computing loss
                    current_loss = 0.0
                    if 'loss' in locals():
                        current_loss = loss.item()
                    l1_w, ms_w, grad_w = adaptive_system.on_train_step(
                        current_loss if current_loss > 0 else 1.0,
                        output,
                        gt_gpu,
                        global_step
                    )
                    
                    # Compute loss with dynamic weights
                    loss_dict = hybrid_criterion(output, gt_gpu, l1_w, ms_w, grad_w)
                    loss = loss_dict['total_tensor']
                
                scaler.scale(loss / cfg["ACCUMULATION_STEPS"]).backward()
                
                if (i + 1) % cfg["ACCUMULATION_STEPS"] == 0:
                    scaler.unscale_(optimizer)
                    grad_norm, clip_val = adaptive_system.on_backward(model)
                    
                    it_t = (time.time() - s_time) / max(1, global_step - s_step) if global_step > s_step else 0.1
                    curr_ep_step = (i + 1) // cfg["ACCUMULATION_STEPS"]
                    
                    if global_step % 5 == 0:
                        losses_dict = {
                            'l1': loss_dict['l1'],
                            'ms': loss_dict['ms'],
                            'grad': loss_dict['grad'],
                            'total': loss.item()
                        }
                        loss_history.append(loss.item())
                        if len(loss_history) > 500:
                            loss_history.pop(0)
                        
                        draw_ui(global_step, epoch, losses_dict, it_t, 
                                get_activity_data(model), 
                                cfg, len(train_ds), steps_per_epoch, curr_ep_step,
                                adaptive_status=get_adaptive_status_with_notification(adaptive_system))
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    
                    if global_step % cfg["LOG_TBOARD_EVERY"] == 0:
                        writer.add_scalar("Training/Loss_L1", loss_dict['l1'], global_step)
                        writer.add_scalar("Training/Loss_MultiScale", loss_dict['ms'], global_step)
                        writer.add_scalar("Training/Loss_Gradient", loss_dict['grad'], global_step)
                        writer.add_scalar("Training/Loss_Total", loss.item(), global_step)
                        writer.add_scalar("Training/LearningRate", scheduler.get_last_lr()[0], global_step)
                        
                        # Log adaptive parameters
                        status = adaptive_system.get_status()
                        writer.add_scalar("Adaptive/LearningRate", status['lr'], global_step)
                        writer.add_scalar("Adaptive/LossWeight_L1", status['loss_weights'][0], global_step)
                        writer.add_scalar("Adaptive/LossWeight_MS", status['loss_weights'][1], global_step)
                        writer.add_scalar("Adaptive/LossWeight_Grad", status['loss_weights'][2], global_step)
                        writer.add_scalar("Adaptive/GradientClip", status['grad_clip'], global_step)
                        writer.add_scalar("Adaptive/BestLoss", status['best_loss'], global_step)
                        writer.add_scalar("Adaptive/PlateauCounter", status['plateau_counter'], global_step)
                        
                        m = model.module if hasattr(model, 'module') else model
                        activities = m.get_layer_activity()
                        for idx, act in enumerate(activities, 1):
                            writer.add_scalar(f"Layers/Block_{idx:02d}", act, global_step)
                        
                        writer.flush()

                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    k = sys.stdin.read(1).lower()
                    if k == '\n':
                        it_t = (time.time() - s_time) / max(1, global_step - s_step) if global_step > s_step else 0.1
                        live_menu(cfg, optimizer, old_settings, model, global_step, epoch, loss.item(), it_t, len(train_ds), steps_per_epoch, (i+1)//cfg["ACCUMULATION_STEPS"])
                        s_time, s_step = time.time(), global_step
                    elif k == 'p': paused = True
                    elif k == 's':
                        cfg["DISPLAY_MODE"] = (cfg.get("DISPLAY_MODE", 0) + 1) % 4
                        save_config(cfg)
                        it_t = (time.time() - s_time) / max(1, global_step - s_step) if global_step > s_step else 0.1
                        draw_ui(global_step, epoch, losses_dict, it_t, get_activity_data(model), cfg, len(train_ds), steps_per_epoch, curr_ep_step,
                                adaptive_status=get_adaptive_status_with_notification(adaptive_system))
                    elif k == 'v': do_val = True

                if (global_step % cfg["VAL_STEP_EVERY"] == 0 and (i+1) % cfg["ACCUMULATION_STEPS"] == 0) or do_val:
                    model.eval()
                    
                    print(f"\n{C_CYAN}{'='*80}{C_RESET}")
                    print(f"{C_BOLD}{C_GREEN}⚡ VALIDATION + QUALITY CHECK{C_RESET}")
                    print(f"{C_CYAN}{'='*80}{C_RESET}\n")
                    
                    v_loss = 0
                    val_ds = VSRDataset(DATASET_ROOT, "Val")
                    val_loader = DataLoader(val_ds, batch_size=1)
                    val_total = len(val_loader)
                    
                    # Quality accumulators
                    lr_psnrs, lr_ssims = [], []
                    ki_psnrs, ki_ssims = [], []
                    
                    val_start = time.time()
                    
                    with torch.no_grad():
                        for v_idx, (v_lrs, v_gt, v_name) in enumerate(val_loader):
                            # Progress Bar
                            progress = (v_idx + 1) / val_total * 100
                            filled = int(50 * (v_idx + 1) / val_total)
                            bar = f"{C_GREEN}{'█' * filled}{C_GRAY}{'░' * (50 - filled)}{C_RESET}"
                            eta = ((time.time() - val_start) / (v_idx + 1)) * (val_total - v_idx - 1) if v_idx > 0 else 0
                            sys.stdout.write(f"\r{C_CYAN}Progress:{C_RESET} [{bar}] {v_idx+1}/{val_total} ({progress:.1f}%) | ETA: {eta:.1f}s")
                            sys.stdout.flush()
                            
                            v_gt_gpu = v_gt.to(device)
                            v_out = model(v_lrs.to(device))
                            
                            # Loss
                            v_loss_dict = hybrid_criterion(v_out, v_gt_gpu)
                            v_loss += v_loss_dict['total_tensor'].item()
                            
                            # Quality Metrics
                            # LR upscaled (Bilinear Baseline)
                            lr_up = F.interpolate(v_lrs[0, 2].unsqueeze(0), size=(PATCH_GT, PATCH_GT), mode='bilinear', align_corners=False).to(device)
                            
                            # Calculate PSNR & SSIM
                            lr_psnr = calculate_psnr(lr_up, v_gt_gpu)
                            lr_ssim = calculate_ssim(lr_up, v_gt_gpu)
                            ki_psnr = calculate_psnr(torch.clamp(v_out, 0, 1), v_gt_gpu)
                            ki_ssim = calculate_ssim(torch.clamp(v_out, 0, 1), v_gt_gpu)
                            
                            lr_psnrs.append(lr_psnr)
                            lr_ssims.append(lr_ssim)
                            ki_psnrs.append(ki_psnr)
                            ki_ssims.append(ki_ssim)
                            
                            # Save images with quality % (max 100 images)
                            if v_idx < min(100, len(val_loader)):
                                img_lr = (lr_up.cpu()[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
                                img_ki = (torch.clamp(v_out[0], 0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
                                img_gt = (v_gt[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
                                
                                # Calculate Quality % for THIS image
                                this_lr_quality = quality_to_percent(lr_psnr, lr_ssim)
                                this_ki_quality = quality_to_percent(ki_psnr, ki_ssim)
                                
                                # Add Text with Quality %
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.7
                                thickness_bg = 3
                                thickness_fg = 1
                                
                                # LR Image
                                cv2.putText(img_lr, "LR", (12, 35), font, font_scale, (0,0,0), thickness_bg, cv2.LINE_AA)
                                cv2.putText(img_lr, "LR", (12, 35), font, font_scale, (255,255,255), thickness_fg, cv2.LINE_AA)
                                quality_text_lr = f"Quality: {this_lr_quality:.1f}%"
                                cv2.putText(img_lr, quality_text_lr, (12, 65), font, font_scale, (0,0,0), thickness_bg, cv2.LINE_AA)
                                cv2.putText(img_lr, quality_text_lr, (12, 65), font, font_scale, (255,165,0), thickness_fg, cv2.LINE_AA)  # Orange
                                
                                # KI Image
                                cv2.putText(img_ki, "KI", (12, 35), font, font_scale, (0,0,0), thickness_bg, cv2.LINE_AA)
                                cv2.putText(img_ki, "KI", (12, 35), font, font_scale, (255,255,255), thickness_fg, cv2.LINE_AA)
                                quality_text_ki = f"Quality: {this_ki_quality:.1f}%"
                                cv2.putText(img_ki, quality_text_ki, (12, 65), font, font_scale, (0,0,0), thickness_bg, cv2.LINE_AA)
                                cv2.putText(img_ki, quality_text_ki, (12, 65), font, font_scale, (0,255,0), thickness_fg, cv2.LINE_AA)  # Green
                                
                                # GT Image
                                cv2.putText(img_gt, "GT", (12, 35), font, font_scale, (0,0,0), thickness_bg, cv2.LINE_AA)
                                cv2.putText(img_gt, "GT", (12, 35), font, font_scale, (255,255,255), thickness_fg, cv2.LINE_AA)
                                quality_text_gt = "Quality: 100.0%"
                                cv2.putText(img_gt, quality_text_gt, (12, 65), font, font_scale, (0,0,0), thickness_bg, cv2.LINE_AA)
                                cv2.putText(img_gt, quality_text_gt, (12, 65), font, font_scale, (0,255,255), thickness_fg, cv2.LINE_AA)  # Cyan
                                
                                spacer = np.ones((PATCH_GT, 2, 3), dtype=np.uint8) * 255
                                combined = np.hstack([img_lr, spacer, img_ki, spacer, img_gt])
                                writer.add_image(f"Val/{v_name[0]}", torch.from_numpy(combined).permute(2, 0, 1).float()/255.0, global_step)
                    
                    sys.stdout.write("\n")
                    
                    # Calculate average metrics
                    avg_lr_psnr = np.mean(lr_psnrs)
                    avg_lr_ssim = np.mean(lr_ssims)
                    avg_ki_psnr = np.mean(ki_psnrs)
                    avg_ki_ssim = np.mean(ki_ssims)
                    
                    # Convert to Quality %
                    lr_quality = quality_to_percent(avg_lr_psnr, avg_lr_ssim)
                    ki_quality = quality_to_percent(avg_ki_psnr, avg_ki_ssim)
                    improvement = ki_quality - lr_quality
                    
                    # Update global metrics
                    last_quality_metrics = {
                        'lr_quality': lr_quality,
                        'ki_quality': ki_quality,
                        'improvement': improvement
                    }
                    
                    # Log to TensorBoard
                    v_loss_avg = v_loss / len(val_loader) if len(val_loader) > 0 else 0
                    writer.add_scalar("Validation/Loss_Total", v_loss_avg, global_step)
                    writer.add_scalar("Quality/LR_Percent", lr_quality, global_step)
                    writer.add_scalar("Quality/KI_Percent", ki_quality, global_step)
                    writer.add_scalar("Quality/Improvement_Percent", improvement, global_step)
                    writer.add_scalar("Quality/LR_PSNR", avg_lr_psnr, global_step)
                    writer.add_scalar("Quality/KI_PSNR", avg_ki_psnr, global_step)
                    writer.add_scalar("Quality/LR_SSIM", avg_lr_ssim * 100, global_step)
                    writer.add_scalar("Quality/KI_SSIM", avg_ki_ssim * 100, global_step)
                    
                    val_duration = time.time() - val_start
                    print(f"\n{C_CYAN}{'='*80}{C_RESET}")
                    print(f"{C_BOLD}📊 VALIDATION RESULTS{C_RESET}")
                    print(f"{C_CYAN}{'-'*80}{C_RESET}")
                    print(f"  Samples:        {val_total}")
                    print(f"  Loss:           {C_GREEN}{v_loss_avg:.6f}{C_RESET}")
                    print(f"  Duration:       {val_duration:.2f}s")
                    print(f"{C_CYAN}{'-'*80}{C_RESET}")
                    print(f"  {C_BOLD}QUALITY SCORES:{C_RESET}")
                    print(f"  LR Quality:     {C_YELLOW}{lr_quality:.1f}%{C_RESET}  (PSNR: {avg_lr_psnr:.2f} dB, SSIM: {avg_lr_ssim*100:.1f}%)")
                    print(f"  KI Quality:     {C_GREEN}{ki_quality:.1f}%{C_RESET}  (PSNR: {avg_ki_psnr:.2f} dB, SSIM: {avg_ki_ssim*100:.1f}%)")
                    print(f"  Improvement:    {C_BOLD}{C_GREEN}+{improvement:.1f}%{C_RESET}")
                    print(f"{C_CYAN}{'='*80}{C_RESET}\n")
                    
                    if do_val:
                        # Auto-continue nach 10 Sekunden, oder sofort bei Enter
                        print(f"{C_YELLOW}Auto-continue in 10s (Press ENTER to skip)...{C_RESET}", end='', flush=True)
                        start_wait = time.time()
                        while time.time() - start_wait < 10.0:
                            if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                                sys.stdin.read(1)  # Enter pressed
                                break
                            remaining = int(10.0 - (time.time() - start_wait))
                            if remaining >= 0:
                                print(f"\r{C_YELLOW}Auto-continue in {remaining}s (Press ENTER to skip)...{C_RESET}", end='', flush=True)
                        print()  # Neue Zeile
                    
                    model.train()
                    do_val = False
                    
                    # UI NEU ZEICHNEN
                    draw_ui(global_step, epoch, losses_dict, it_t, get_activity_data(model), cfg, len(train_ds), steps_per_epoch, curr_ep_step,
                            adaptive_status=get_adaptive_status_with_notification(adaptive_system))
                    
                if global_step % cfg["SAVE_STEP_EVERY"] == 0 and (i+1) % cfg["ACCUMULATION_STEPS"] == 0:
                    print(f"\n{C_YELLOW}💾 SAVING CHECKPOINT...{C_RESET}")
                    
                    checkpoint = {
                        'step': global_step,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'config': cfg
                    }
                    
                    save_path = os.path.join(CHECKPOINT_DIR, "latest.pth")
                    torch.save(checkpoint, save_path)
                    print(f"{C_GREEN}✓ Saved: {save_path}{C_RESET}")
                    
                    if global_step % 25000 == 0:
                        milestone_path = os.path.join(CHECKPOINT_DIR, f"step_{global_step}.pth")
                        torch.save(checkpoint, milestone_path)
                        print(f"{C_GREEN}✓ Milestone: {milestone_path}{C_RESET}\n")
                    
                    draw_ui(global_step, epoch, losses_dict, it_t, get_activity_data(model), cfg, len(train_ds), steps_per_epoch, curr_ep_step,
                            adaptive_status=get_adaptive_status_with_notification(adaptive_system))
                    
    except KeyboardInterrupt:
        print("\033[?25h")
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print(f"\n{C_BOLD}{C_RED}⚠️  TRAINING UNTERBROCHEN{C_RESET}")
        save_choice = input("Checkpoint speichern? (y/n): ").lower()
        
        if save_choice == 'y':
            checkpoint = {
                'step': global_step,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': cfg,
                'training_phase': 'pretrain'
            }
            
            save_path = os.path.join(CHECKPOINT_DIR, f"vsr_INTERRUPT_{global_step}.pth")
            torch.save(checkpoint, save_path)
            print(f"{C_GREEN}✅ Checkpoint gespeichert: {save_path}{C_RESET}")
        else:
            print(f"{C_YELLOW}⚠️  Checkpoint NICHT gespeichert!{C_RESET}")
        
        sys.exit(0)

if __name__ == "__main__":
    old_settings = termios.tcgetattr(sys.stdin)
    try: train(old_settings)
    finally: print("\033[?25h"); termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)