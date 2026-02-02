import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import cv2, os, time, shutil, sys, glob, json, re, random
import numpy as np
from datetime import datetime
from model import BasicVSR_Foundational
import select
import termios
import tty

# ============================================================
# --- KONFIGURATION & PFADE ---
# ============================================================
DATA_ROOT      = "/mnt/data/training/Universal/Mastermodell/Learn"
CONFIG_FILE    = os.path.join(DATA_ROOT, "train_config.json")
DATASET_ROOT   = "/mnt/data/training/Dataset/Universal/Mastermodell"

N_BLOCKS       = 35
BATCH_SIZE     = 1
NUM_WORKERS    = 4
PATCH_GT, PATCH_LR = 540, 180

defaults = {
    "LR_EXPONENT": -5,
    "WEIGHT_DECAY": 0.01,
    "MAX_STEPS": 1000000,
    "VAL_STEP_EVERY": 250,
    "SAVE_STEP_EVERY": 5000,
    "LOG_TBOARD_EVERY": 10,
    "LOG_LAYERS_EVERY": 50,
    "HIST_STEP_EVERY": 500,
    "ACCUMULATION_STEPS": 3,
    "SORT_BY_ACTIVITY": True
}

activity_history = {i+1: [] for i in range(N_BLOCKS)}
WINDOW_SIZE = 50
ema_grads = {}
alpha = 0.2
last_term_size = (0, 0)
C_GREEN, C_GRAY, C_RESET, C_BOLD, C_CYAN, C_RED = "\033[92m", "\033[90m", "\033[0m", "\033[1m", "\033[96m", "\033[91m"
ANSI_HOME, ANSI_CLEAR = "\033[H", "\033[2J"
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

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

def get_activity_data(model, sort_by_activity=True):
    global ema_grads
    current_raw = {}
    for name, p in model.named_parameters():
        if "res" in name and "weight" in name and p.grad is not None:
            current_raw[name] = p.grad.abs().mean().item()
    if not current_raw: return [(i+1, 0) for i in range(N_BLOCKS)]
    for name, val in current_raw.items():
        ema_grads[name] = alpha * val + (1 - alpha) * ema_grads.get(name, 0.0)
    block_scores = []
    for i in range(N_BLOCKS):
        vals = [v for k, v in ema_grads.items() if f"res.{i}." in k]
        block_scores.append(sum(vals)/len(vals) if vals else 0.0)
    max_val = max(block_scores) if max(block_scores) > 1e-12 else 1e-12
    activities = [(i+1, int((v / max_val) * 100)) for i, v in enumerate(block_scores)]
    return sorted(activities, key=lambda x: x[1], reverse=True) if sort_by_activity else activities

def draw_ui(step, epoch, loss, it_time, activities, cfg, num_images, steps_per_epoch, current_epoch_step, paused=False):
    global last_term_size
    term_size = shutil.get_terminal_size()
    if term_size != last_term_size: print(ANSI_CLEAR); last_term_size = term_size
    print(ANSI_HOME + "\033[?25l")
    ui_w = max(90, term_size.columns - 4)
    total_prog = (step / cfg["MAX_STEPS"]) * 100 if cfg["MAX_STEPS"] > 0 else 0
    total_eta = format_time((cfg["MAX_STEPS"] - step) * it_time) if not paused else "PAUSIERT"
    epoch_prog = (current_epoch_step / steps_per_epoch) * 100
    epoch_eta = format_time((steps_per_epoch - current_epoch_step) * it_time) if not paused else "PAUSIERT"

    def print_line(content):
        padding = max(0, ui_w - get_visible_len(content) - 4)
        sys.stdout.write(f" ║ {content}{' ' * padding} ║\n")

    sys.stdout.write(f" {C_GRAY}╔{'═'*(ui_w-2)}╗{C_RESET}\n")
    status = f"{C_RED}⏸ PAUSIERT (P){C_RESET}" if paused else f"{C_GREEN}▶ RUNNING{C_RESET}"
    print_line(f"{C_BOLD}MISSION CONTROL{C_RESET} │ {status} │ STEP: {step}/{cfg['MAX_STEPS']} │ LR: 1e{cfg['LR_EXPONENT']}")
    print_line(f"TOTAL PROG: {make_bar(total_prog, ui_w-65)} {total_prog:>5.1f}% │ ETA: {total_eta}")
    sys.stdout.write(f" {C_GRAY}╠{'═'*(ui_w-2)}╣{C_RESET}\n")
    print_line(f"EPOCH: {epoch:<4} │ STEP: {current_epoch_step}/{steps_per_epoch} │ ETA: {epoch_eta}")
    print_line(f"EPOCH PROG: {make_bar(epoch_prog, ui_w-65)} {epoch_prog:>5.1f}%")
    sys.stdout.write(f" {C_GRAY}╠{'═'*(ui_w-2)}╣{C_RESET}\n")
    print_line(f"DATASET: {num_images:<7} imgs │ LOSS: {loss:.6f} │ SPEED: {it_time:.2f}s/it")
    sys.stdout.write(f" {C_GRAY}╠{'═'*(ui_w-2)}╣{C_RESET}\n")

    for idx, s in activities:
        print_line(f"LAYER {idx:>2}: {make_bar(s, ui_w-25)} {s:>3}%")

    sys.stdout.write(f" {C_GRAY}╠{'═'*(ui_w-2)}╣{C_RESET}\n")
    nv, ns = cfg['VAL_STEP_EVERY']-(step%cfg['VAL_STEP_EVERY']), cfg['SAVE_STEP_EVERY']-(step%cfg['SAVE_STEP_EVERY'])
    print_line(f"VAL IN: {nv:<5} │ SAVE IN: {ns:<5} │ EFF-BATCH: {BATCH_SIZE * cfg['ACCUMULATION_STEPS']}")
    sys.stdout.write(f" {C_GRAY}╚{'═'*(ui_w-2)}╝{C_RESET}\n")
    sys.stdout.write(f"{' ' * ((ui_w - 45) // 2)}{C_BOLD}( ENTER: Setup | S: Sort | P: Pause | V: Instant-Val ){C_RESET}\n")
    sys.stdout.flush()

def live_menu(cfg, optimizer, old_settings, model, step, epoch, loss, it_t, ds_len, steps_ep, curr_ep_step):
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    print("\033[?25h" + ANSI_CLEAR + ANSI_HOME)
    while True:
        print(f"{C_BOLD}🛠️  LIVE-SETUP{C_RESET}\n" + "-"*45)
        keys = list(cfg.keys())
        for idx, k in enumerate(keys): print(f" {idx+1}. {k:<20}: {cfg[k]}")
        print("-" * 45 + "\n 0. ZURÜCK │ S. SORTIERUNG")
        wahl = input("\n Auswahl: ").lower()
        if wahl == "0": break
        if wahl == "s": cfg["SORT_BY_ACTIVITY"] = not cfg["SORT_BY_ACTIVITY"]; save_config(cfg); break
        try:
            idx = int(wahl) - 1
            if 0 <= idx < len(keys):
                k_name = keys[idx]
                new_val = input(f"Neuer Wert für {k_name}: ")
                if k_name == "LR_EXPONENT":
                    val = int(new_val); cfg[k_name] = val
                    for pg in optimizer.param_groups: pg['lr'] = 10**val
                else: cfg[k_name] = type(cfg[k_name])(new_val)
                save_config(cfg)
        except: pass
    tty.setcbreak(sys.stdin.fileno())
    print(ANSI_CLEAR)
    draw_ui(step, epoch, loss, it_t, get_activity_data(model, cfg["SORT_BY_ACTIVITY"]), cfg, ds_len, steps_ep, curr_ep_step)

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
                if os.path.exists(os.path.join(self.lr_dir, f)): self.filenames.append((f, self.lr_dir))
                elif os.path.exists(os.path.join(self.patch_lr_dir, f)): self.filenames.append((f, self.patch_lr_dir))
                else:
                    try: os.remove(os.path.join(self.gt_dir, f))
                    except: pass
            else: self.filenames.append((f, self.lr_dir))
    def __len__(self): return len(self.filenames)
    def __getitem__(self, idx):
        name, lr_folder = self.filenames[idx]
        gt = cv2.cvtColor(cv2.imread(os.path.join(self.gt_dir, name)), cv2.COLOR_BGR2RGB) / 255.0
        lr_stack = cv2.cvtColor(cv2.imread(os.path.join(lr_folder, name)), cv2.COLOR_BGR2RGB) / 255.0
        lrs = [lr_stack[i*PATCH_LR:(i+1)*PATCH_LR, :] for i in range(5)]
        return torch.stack([torch.from_numpy(f).permute(2, 0, 1) for f in lrs]).float(), torch.from_numpy(gt).permute(2, 0, 1).float(), name

def train(old_settings):
    cfg = load_config()
    LOG_BASE, CHECKPOINT_DIR = os.path.join(DATA_ROOT, "logs"), os.path.join(DATA_ROOT, "checkpoints")
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    os.system('clear')
    choice = input("⚠️  [L]öschen oder [F]ortsetzen? (L/F): ").lower()
    if choice == 'l':
        if os.path.exists(LOG_BASE): shutil.rmtree(LOG_BASE)
        if os.path.exists(CHECKPOINT_DIR): shutil.rmtree(CHECKPOINT_DIR)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True); save_config(defaults); cfg = defaults.copy()

    cleanup_logs(LOG_BASE)
    tty.setcbreak(sys.stdin.fileno())
    writer = SummaryWriter(log_dir=os.path.join(LOG_BASE, "active_run"))

    layout = {"Training": {"Loss": ["Margin", ["Loss/Train", "Loss/Val"], "Value"], 
                           "LR": ["Margin", ["LR/Log10"], "Value"],
                           "Ratio": ["Margin", ["Ratio/Val_Train"], "Value"]}}
    try: writer.add_custom_scalars(layout)
    except: pass

    device = torch.device('cuda')
    model = BasicVSR_Foundational(n_blocks=N_BLOCKS).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=10**cfg["LR_EXPONENT"], weight_decay=cfg["WEIGHT_DECAY"])
    criterion = nn.L1Loss()
    global_step = 0
    ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth")), key=os.path.getmtime)
    if ckpts and choice != 'l':
        ckpt = torch.load(ckpts[-1]); model.load_state_dict(ckpt['model_state_dict']); optimizer.load_state_dict(ckpt['optimizer_state_dict']); global_step = ckpt.get('step', 0)

    s_time, s_step, paused, do_val = time.time(), global_step, False, False

    try:
        for epoch in range(1, 100000):
            train_ds = VSRDataset(DATASET_ROOT, "Patches")
            steps_per_epoch = max(1, len(train_ds) // (BATCH_SIZE * cfg["ACCUMULATION_STEPS"]))
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
            model.train()
            for i, (lrs, gt, _) in enumerate(train_loader):
                while paused:
                    draw_ui(global_step, epoch, 0, 0.1, get_activity_data(model, cfg["SORT_BY_ACTIVITY"]), cfg, len(train_ds), steps_per_epoch, (i+1)//cfg["ACCUMULATION_STEPS"], paused=True)
                    time.sleep(0.5)
                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        if sys.stdin.read(1).lower() == 'p': paused = False; s_time = time.time(); s_step = global_step
                output = model(lrs.to(device))
                loss = criterion(output, gt.to(device))
                (loss / cfg["ACCUMULATION_STEPS"]).backward()
                if (i + 1) % cfg["ACCUMULATION_STEPS"] == 0:
                    it_t = (time.time() - s_time) / max(1, global_step - s_step) if global_step > s_step else 0.1
                    curr_ep_step = (i + 1) // cfg["ACCUMULATION_STEPS"]
                    if global_step % 5 == 0: draw_ui(global_step, epoch, loss.item(), it_t, get_activity_data(model, cfg["SORT_BY_ACTIVITY"]), cfg, len(train_ds), steps_per_epoch, curr_ep_step)
                    optimizer.step(); optimizer.zero_grad(); global_step += 1
                    if global_step % cfg["LOG_TBOARD_EVERY"] == 0:
                        writer.add_scalar("Loss/Train", loss.item(), global_step)
                        writer.add_scalar("LR/Log10", cfg["LR_EXPONENT"], global_step)
                
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    k = sys.stdin.read(1).lower()
                    if k == '\n':
                        it_t = (time.time() - s_time) / max(1, global_step - s_step) if global_step > s_step else 0.1
                        live_menu(cfg, optimizer, old_settings, model, global_step, epoch, loss.item(), it_t, len(train_ds), steps_per_epoch, (i+1)//cfg["ACCUMULATION_STEPS"])
                        s_time, s_step = time.time(), global_step
                    elif k == 'p': paused = True
                    elif k == 's': cfg["SORT_BY_ACTIVITY"] = not cfg["SORT_BY_ACTIVITY"]; save_config(cfg)
                    elif k == 'v': do_val = True

                if (global_step % cfg["VAL_STEP_EVERY"] == 0 and (i+1) % cfg["ACCUMULATION_STEPS"] == 0) or do_val:
                    model.eval(); v_loss = 0; val_ds = VSRDataset(DATASET_ROOT, "Val"); val_loader = DataLoader(val_ds, batch_size=1)
                    with torch.no_grad():
                        for v_idx, (v_lrs, v_gt, v_name) in enumerate(val_loader):
                            v_out = model(v_lrs.to(device))
                            v_loss += criterion(v_out, v_gt.to(device)).item()
                            if v_idx < 20:
                                # 3er Bild mit Beschriftung und Spacer
                                lr_up = torch.nn.functional.interpolate(v_lrs[0, 2].unsqueeze(0), size=(PATCH_GT, PATCH_GT), mode='nearest').squeeze(0)
                                img_lr = (lr_up.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
                                img_ki = (v_out[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
                                img_gt = (v_gt[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
                                for img, text in [(img_lr, "LR"), (img_ki, "KI"), (img_gt, "GT")]:
                                    cv2.putText(img, text, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
                                    cv2.putText(img, text, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
                                spacer = np.ones((PATCH_GT, 2, 3), dtype=np.uint8) * 255
                                combined = np.hstack([img_lr, spacer, img_ki, spacer, img_gt])
                                writer.add_image(f"Val/{v_name[0]}", torch.from_numpy(combined).permute(2, 0, 1).float()/255.0, global_step)
                    v_loss_avg = v_loss / len(val_loader) if len(val_loader) > 0 else 0
                    writer.add_scalar("Loss/Val", v_loss_avg, global_step)
                    if loss.item() > 0:
                        writer.add_scalar("Ratio/Val_Train", v_loss_avg / loss.item(), global_step)
                    model.train(); do_val = False
                if global_step % cfg["SAVE_STEP_EVERY"] == 0 and (i+1) % cfg["ACCUMULATION_STEPS"] == 0:
                    torch.save({'step': global_step, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(CHECKPOINT_DIR, f"vsr_step_{global_step}.pth"))
    except KeyboardInterrupt:
        print("\033[?25h"); termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print(f"\n{C_BOLD}{C_RED}⚠️ TRAINING UNTERBROCHEN{C_RESET}")
        if input("Checkpoint speichern? (y/n): ").lower() == 'y':
            torch.save({'step': global_step, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(CHECKPOINT_DIR, f"vsr_INTERRUPT_{global_step}.pth"))
            print(f"{C_GREEN}✅ Gespeichert.{C_RESET}")
        sys.exit(0)

if __name__ == "__main__":
    old_settings = termios.tcgetattr(sys.stdin)
    try: train(old_settings)
    finally: print("\033[?25h"); termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)