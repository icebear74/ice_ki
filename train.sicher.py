import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import cv2, os, time, random, shutil, sys, glob, json
from datetime import datetime
from model import BasicVSR_Foundational
import select

# ============================================================
# --- KONFIGURATION & PFADE ---
# ============================================================
CONFIG_FILE    = "/mnt/data/training/Universal/Mastermodell/Learn/train_config.json"
GEN_STATUS     = "/mnt/data/training/Universal/Mastermodell/Learn/generator_status.json"
DATA_ROOT      = "/mnt/data/training/Universal/Mastermodell/Learn"
N_BLOCKS       = 20
BATCH_SIZE     = 2 
LEARNING_RATE  = 1e-4
NUM_WORKERS    = 4

PATCH_GT, PATCH_LR = 540, 180

defaults = {
    "MAX_STEPS": 100000,
    "VAL_STEP_EVERY": 20,
    "SAVE_STEP_EVERY": 5000,
    "LOG_TBOARD_EVERY": 10,
    "HIST_STEP_EVERY": 1000,
    "ACCUMULATION_STEPS": 2 
}

# ============================================================
# --- HILFSFUNKTIONEN ---
# ============================================================

def cleanup_lr_folder(dataset_type='Patches'):
    gt_path = os.path.join(DATA_ROOT, dataset_type, "GT")
    lr_path = os.path.join(DATA_ROOT, dataset_type, "LR")
    if not os.path.exists(gt_path) or not os.path.exists(lr_path): return
    gt_files = set([f for f in os.listdir(gt_path) if f.endswith('.png')])
    lr_files = [f for f in os.listdir(lr_path) if f.endswith('.png')]
    deleted_count = 0
    for f in lr_files:
        if f not in gt_files:
            try: os.remove(os.path.join(lr_path, f)); deleted_count += 1
            except: pass
    if deleted_count > 0: print(f"🧹 Bereinigung: {deleted_count} Waisen gelöscht.")

def load_config():
    cfg = defaults.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f: cfg.update(json.load(f))
        except: pass
    return cfg

def save_config(cfg):
    with open(CONFIG_FILE, 'w') as f_out:
        json.dump(cfg, f_out, indent=4)

def live_menu(cfg):
    while True:
        os.system('clear')
        print("🛠️  LIVE-SETUP (Änderungen werden sofort gespeichert)")
        print(f" 1. Max Steps          : {cfg['MAX_STEPS']}")
        print(f" 2. Val Intervall      : {cfg['VAL_STEP_EVERY']}")
        print(f" 3. Save Intervall     : {cfg['SAVE_STEP_EVERY']}")
        print(f" 4. Hist Intervall     : {cfg['HIST_STEP_EVERY']}")
        print(f" 5. Accumulation Steps : {cfg['ACCUMULATION_STEPS']}")
        print("-" * 45)
        print(" 0. ZURÜCK ZUM TRAINING")
        
        wahl = input("\n Auswahl: ")
        if wahl == "0": break
        try:
            if wahl == "1": cfg["MAX_STEPS"] = int(input("Wert Max Steps: "))
            elif wahl == "2": cfg["VAL_STEP_EVERY"] = int(input("Wert Val Intervall: "))
            elif wahl == "3": cfg["SAVE_STEP_EVERY"] = int(input("Wert Save Intervall: "))
            elif wahl == "4": cfg["HIST_STEP_EVERY"] = int(input("Wert Hist Intervall: "))
            elif wahl == "5": cfg["ACCUMULATION_STEPS"] = int(input("Wert Accumulation: "))
            save_config(cfg)
        except: print("⚠️ Ungültige Eingabe!")

def get_generator_info():
    try:
        with open(GEN_STATUS, 'r') as f: return json.load(f)
    except: return None

def wait_for_generator():
    while True:
        info = get_generator_info()
        if info and info.get("ready_for_training", False): break
        time.sleep(5)

def get_activity_data(model):
    activities = []
    for i in range(N_BLOCKS):
        try:
            p = model.pro_resblocks[i].res[0].weight
            if p.grad is not None:
                grad_val = p.grad.abs().mean().item()
                score = min(100, max(0, int((torch.log10(torch.tensor(grad_val + 1e-9)) + 5) * 25)))
                bar_val = int(score / 5)
                activities.append((bar_val, score))
            else: activities.append((0, 0))
        except: activities.append((0, 0))
    return activities

def draw_ui(step, epoch, loss, it_time, activities, cfg, num_images, new_images, gen_finished):
    os.system('clear')
    rem_steps = max(0, cfg["MAX_STEPS"] - step)
    total_sec = rem_steps * it_time
    h, m = int(total_sec // 3600), int((total_sec % 3600) // 60)
    prog = (step / cfg["MAX_STEPS"]) * 100 if cfg["MAX_STEPS"] > 0 else 0
    bar = "█" * int(prog / 2) + "░" * (50 - int(prog / 2))
    gen_str = "✅ FERTIG" if gen_finished else "🔄 LÄUFT"
    
    print(f"╔════════════════════════════════ MASTER-TRAINING MISSION CONTROL ══════════════════════════════╗")
    print(f"║ Progress: [{bar}] {prog:>5.1f}% | Step: {step}/{cfg['MAX_STEPS']} ║")
    print(f"║ Epoch: {epoch:<5} | Loss: {loss:.6f} | Speed: {it_time:.2f}s/it | ETA: {h}h {m}m ║")
    print(f"╠══════════════════════════════════════ DATASET STATUS ════════════════════════════════════════╣")
    print(f"║ Bilder: {num_images:<10} (+{new_images:<4}) | Generator: {gen_str:<25} ║")
    print(f"╠══════════════════════════════════════ LAYER ACTIVITY ════════════════════════════════════════╣")
    for i in range(0, N_BLOCKS, 2):
        (v1, p1), (v2, p2) = activities[i], activities[i+1]
        print(f"║ B{i+1:>2}: [{'█'*v1 + '░'*(20-v1)}] {p1:>3}% | B{i+2:>2}: [{'█'*v2 + '░'*(20-v2)}] {p2:>3}% ║")
    print(f"╠══════════════════════════════════════ NEXT SCHEDULE ═════════════════════════════════════════╣")
    nv, ns = cfg['VAL_STEP_EVERY']-(step%cfg['VAL_STEP_EVERY']), cfg['SAVE_STEP_EVERY']-(step%cfg['SAVE_STEP_EVERY'])
    print(f"║ Val in: {nv:>5} steps | Save in: {ns:>5} steps | Batch-Eff: {cfg['ACCUMULATION_STEPS']*BATCH_SIZE:<11} ║")
    print(f"╚═══════════════════════════════ ( ENTER für LIVE-SETUP ) ═════════════════════════════════════╝")

class VSRDataset(Dataset):
    def __init__(self, root_dir, dataset_type='Patches'):
        self.gt_path = os.path.join(root_dir, dataset_type, "GT")
        self.lr_path = os.path.join(root_dir, dataset_type, "LR")
        self.filenames = sorted([f for f in os.listdir(self.gt_path) if f.endswith('.png')])
    def __len__(self): return len(self.filenames)
    def __getitem__(self, idx):
        name = self.filenames[idx]
        gt = cv2.cvtColor(cv2.imread(os.path.join(self.gt_path, name)), cv2.COLOR_BGR2RGB) / 255.0
        lr_stack = cv2.cvtColor(cv2.imread(os.path.join(self.lr_path, name)), cv2.COLOR_BGR2RGB) / 255.0
        lrs = [lr_stack[i*PATCH_LR:(i+1)*PATCH_LR, :] for i in range(5)]
        return torch.stack([torch.from_numpy(f).permute(2, 0, 1) for f in lrs]).float(), \
               torch.from_numpy(gt).permute(2, 0, 1).float(), name

def train():
    cleanup_lr_folder('Patches'); cleanup_lr_folder('Val')
    wait_for_generator()
    current_cfg = load_config()
    LOG_BASE, CHECKPOINT_DIR = os.path.join(DATA_ROOT, "logs"), os.path.join(DATA_ROOT, "checkpoints")

    os.system('clear')
    choice = input("⚠️ [L]öschen oder [F]ortsetzen? (L/F): ").lower()
    if choice == 'l':
        for d in [LOG_BASE, CHECKPOINT_DIR]: 
            if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        current_cfg = defaults.copy(); save_config(current_cfg)
    
    writer = SummaryWriter(log_dir=os.path.join(LOG_BASE, datetime.now().strftime("%Y%m%d-%H%M%S")))
    device = torch.device('cuda')
    model = BasicVSR_Foundational(n_blocks=N_BLOCKS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()

    global_step = 0
    ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth")), key=os.path.getmtime)
    if ckpts and choice != 'l':
        ckpt = torch.load(ckpts[-1])
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        global_step = ckpt.get('step', 0)

    last_img_count, session_start_time, session_start_step = 0, time.time(), global_step

    try:
        for epoch in range(1, 100000):
            info = get_generator_info()
            gen_finished = (info.get("status") == "finished") if info else False
            train_ds = VSRDataset(DATA_ROOT, "Patches")
            num_images = len(train_ds)
            new_imgs, last_img_count = (num_images - last_img_count if last_img_count > 0 else 0), num_images
            
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
            val_loader = DataLoader(VSRDataset(DATA_ROOT, "Val"), batch_size=1, shuffle=False)
            
            model.train()
            optimizer.zero_grad()

            for i, (lrs, gt, _) in enumerate(train_loader):
                lrs, gt = lrs.to(device), gt.to(device)
                output = model(lrs)
                loss = criterion(output, gt)
                
                acc_steps = current_cfg.get("ACCUMULATION_STEPS", 2)
                (loss / acc_steps).backward()

                if (i + 1) % acc_steps == 0:
                    if global_step % 5 == 0:
                        act = get_activity_data(model)
                        it_t = (time.time() - session_start_time) / max(1, global_step - session_start_step)
                        draw_ui(global_step, epoch, loss.item(), it_t, act, current_cfg, num_images, new_imgs, gen_finished)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    if global_step % current_cfg["LOG_TBOARD_EVERY"] == 0:
                        writer.add_scalar("Loss/Train", loss.item(), global_step)

                    if global_step % current_cfg["HIST_STEP_EVERY"] == 0:
                        for n, p in model.named_parameters():
                            if "pro_resblocks" in n and "weight" in n:
                                writer.add_histogram(f"Weights/{n}", p, global_step)
                        writer.flush()

                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    sys.stdin.readline()
                    live_menu(current_cfg)
                    session_start_time, session_start_step = time.time(), global_step

                if global_step % current_cfg["VAL_STEP_EVERY"] == 0 and (i+1) % acc_steps == 0:
                    model.eval(); v_loss = 0
                    with torch.no_grad():
                        for v_idx, (v_lrs, v_gt, v_name) in enumerate(val_loader):
                            v_lrs, v_gt = v_lrs.to(device), v_gt.to(device)
                            v_out = model(v_lrs)
                            v_loss += criterion(v_out, v_gt).item()
                            if v_idx == 0:
                                lr_pre = torch.nn.functional.interpolate(v_lrs[:, 2], size=(PATCH_GT, PATCH_GT), mode='nearest')
                                # WIEDER NEBENEINANDER (dim=2) wie gewünscht
                                comp = torch.cat([lr_pre[0], v_out[0], v_gt[0]], dim=2)
                                writer.add_image(f"Visuals/{v_name[0]}", comp.clamp(0, 1), global_step)
                    writer.add_scalar("Loss/Validation", v_loss / max(1, len(val_loader)), global_step)
                    writer.flush()
                    model.train()

                if global_step % current_cfg["SAVE_STEP_EVERY"] == 0 and (i+1) % acc_steps == 0:
                    torch.save({'step': global_step, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                               os.path.join(CHECKPOINT_DIR, f"vsr_step_{global_step}.pth"))

    except KeyboardInterrupt:
        print("\n🛑 Training unterbrochen. Speichere Notfall-Checkpoint...")
        torch.save({'step': global_step, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(CHECKPOINT_DIR, "vsr_checkpoint_emergency.pth"))
        sys.exit(0)

if __name__ == "__main__":
    train()