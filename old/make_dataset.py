import os, cv2, subprocess, random, glob, json, shutil, sys, re, time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# --- KONFIGURATION ---
BASE_DIR = "/mnt/data/training/Universal/Mastermodell"
OUT_DIR = os.path.join(BASE_DIR, "Learn")
TEMP_DIR = os.path.join(OUT_DIR, "temp")
STATUS_FILE = os.path.join(OUT_DIR, "generator_status.json")

BASE_FRAME_LIMIT = 3000
MAX_WORKERS = 2 
VAL_PERCENT = 0.01 
MIN_FILE_SIZE = 10000 

FILTERS = {
    "s01e": 1.0, "beverly": 0.3, "unendliche": 0.2, "shrek": 0.1,
    "trek": 0.2, "wars": 0.2, "forrest": 0.2, "hai": 0.3,
    "thrones": 0.1, "westworld": 0.1, "strange": 0.2
}
DEFAULT_WEIGHT = 0.1

def get_video_info(v_path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'format=duration:stream=avg_frame_rate', '-of', 'json', v_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        data = json.loads(result.stdout)
        fps = eval(data['streams'][0]['avg_frame_rate'])
        dur = float(data['format']['duration'])
        return fps, dur
    except: return 23.976, 3600.0

def update_status(status, current, total, phase="", training_ready=False):
    with open(STATUS_FILE, "w") as f:
        json.dump({
            "status": status, "phase": phase, 
            "current": current, "total": total, 
            "ready_for_training": training_ready,
            "last_update": time.time()
        }, f)

def extract_patch_with_retry(args):
    v_path, start_ts, vn, f_idx, prefix, duration = args
    clean_vn = re.sub(r'[^a-zA-Z0-9]', '_', vn)
    tid = f"{prefix}_{clean_vn}_idx{f_idx}.png"
    gt_path, lr_path = os.path.join(OUT_DIR, prefix, "GT", tid), os.path.join(OUT_DIR, prefix, "LR", tid)
    
    if os.path.exists(gt_path): return True, 1, prefix
    
    current_ts = start_ts
    for att in range(1, 11):
        thread_temp = os.path.join(TEMP_DIR, f"job_{clean_vn}_{f_idx}_a{att}_{random.randint(100,999)}")
        os.makedirs(thread_temp, exist_ok=True)
        tonemap_vf = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=mobius,zscale=t=bt709:m=bt709,format=yuv420p,scale=1920:1080:flags=lanczos"
        cmd = ['nice', '-n', '19', 'ffmpeg', '-y', '-threads', '1', '-ss', str(round(current_ts, 3)), '-i', v_path, '-vf', tonemap_vf, '-vframes', '5', os.path.join(thread_temp, 'f_%d.png')]
        subprocess.run(cmd, capture_output=True, check=False)
        frames = []
        for i in range(1, 6):
            f_p = os.path.join(thread_temp, f"f_{i}.png")
            if os.path.exists(f_p) and os.path.getsize(f_p) > MIN_FILE_SIZE:
                img = cv2.imread(f_p)
                if img is not None: frames.append(img)
        if len(frames) == 5:
            if cv2.absdiff(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY), cv2.cvtColor(frames[4], cv2.COLOR_BGR2GRAY)).mean() < 45:
                y, x = (1080-540)//2, (1920-540)//2
                cv2.imwrite(gt_path, frames[2][y:y+540, x:x+540])
                cv2.imwrite(lr_path, cv2.vconcat([cv2.resize(f[y:y+540, x:x+540], (180, 180)) for f in frames]))
                shutil.rmtree(thread_temp); return True, att, prefix
        if os.path.exists(thread_temp): shutil.rmtree(thread_temp)
        current_ts = (current_ts + 60.0) % duration
    return False, 10, prefix

if __name__ == "__main__":
    print("\033[H\033[J🚀 Initialisierung Dataset-Generator...")
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    for p in ["Patches/LR", "Patches/GT", "Val/LR", "Val/GT", "temp"]: os.makedirs(os.path.join(OUT_DIR, p), exist_ok=True)

    raw_vids = sorted(glob.glob(os.path.join(BASE_DIR, "*.mkv")))
    vid_data = []
    for v in tqdm(raw_vids, desc="🔍 Video-Analyse"):
        vn = os.path.basename(v)
        w = next((val for p, val in FILTERS.items() if p in vn.lower()), DEFAULT_WEIGHT)
        fps, dur = get_video_info(v)
        limit = int(BASE_FRAME_LIMIT * w)
        v_limit = max(1, int(limit * VAL_PERCENT))
        vid_data.append({'path': v, 'name': vn, 'weight': w, 'limit': limit, 'v_limit': v_limit, 'dur': dur})

    vid_data.sort(key=lambda x: x['weight'], reverse=True)

    # TASKS VORBEREITEN
    val_tasks = []
    train_pool = {d['name']: [] for d in vid_data}
    for d in vid_data:
        # Val
        for i in range(d['v_limit']):
            ts = (d['dur'] * 0.2) + (i * (d['dur'] * 0.6 / d['v_limit']))
            val_tasks.append((d['path'], ts, d['name'], i, "Val", d['dur']))
        # Patches
        step = d['dur'] / d['limit']
        for i in range(d['limit'] - d['v_limit']):
            train_pool[d['name']].append((d['path'], i * step, d['name'], i, "Patches", d['dur']))

    # GEWICHTETES INTERLEAVING
    weighted_train_tasks = []
    active_vids = [d['name'] for d in vid_data]
    checkpoint_idx = 0
    
    # Simuliere ersten Durchlauf für Checkpoint
    temp_active = list(active_vids)
    for vn in temp_active:
        w = next(d['weight'] for d in vid_data if d['name'] == vn)
        chunk = max(1, int(w * 20))
        checkpoint_idx += min(chunk, len(train_pool[vn]))

    # Tatsächliches Interleaving
    while active_vids:
        for vn in list(active_vids):
            w = next(d['weight'] for d in vid_data if d['name'] == vn)
            chunk_size = max(1, int(w * 20))
            for _ in range(chunk_size):
                if train_pool[vn]: weighted_train_tasks.append(train_pool[vn].pop(0))
                else: 
                    if vn in active_vids: active_vids.remove(vn)
                    break

    # UI SETUP
    cols, _ = shutil.get_terminal_size()
    fmt = "{desc} |{bar:" + str(max(10, cols - 75)) + "}| {n_fmt}/{total_fmt} [{postfix}]"
    
    # Jeder Film hat ein total von limit (Val + Patches)
    v_bars = {d['name']: tqdm(total=d['limit'], desc=f"{d['name'][:27]:<27}..", position=i+1, bar_format=fmt) for i, d in enumerate(vid_data)}
    o_bar = tqdm(total=len(val_tasks) + len(weighted_train_tasks), desc=f"{'GESAMT':<30}", colour='green', position=0, bar_format=fmt)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        # Phase 1: Validierung
        print(f"\nPhase 1: Validierung ({len(val_tasks)} Bilder)")
        for i in range(0, len(val_tasks), MAX_WORKERS):
            batch = val_tasks[i:i+MAX_WORKERS]
            for idx, (res, att, prefix) in enumerate(exe.map(extract_patch_with_retry, batch)):
                vn = batch[idx][2]
                if res:
                    v_bars[vn].update(1)
                    v_bars[vn].set_postfix_str(f"VAL", refresh=True)
                    o_bar.update(1)
            update_status("running", o_bar.n, o_bar.total, "Validation", False)

        # Phase 2: Patches
        print(f"\nPhase 2: Training Patches ({len(weighted_train_tasks)} Bilder)")
        for i in range(0, len(weighted_train_tasks), MAX_WORKERS):
            batch = weighted_train_tasks[i:i+MAX_WORKERS]
            for idx, (res, att, prefix) in enumerate(exe.map(extract_patch_with_retry, batch)):
                vn = batch[idx][2]
                if res:
                    v_bars[vn].update(1)
                    v_bars[vn].set_postfix_str(f"TRAIN", refresh=True)
                    o_bar.update(1)
            
            # Checkpoint erreicht? (Alle Val + Erster gewichteter Durchlauf fertig)
            is_ready = (o_bar.n >= (len(val_tasks) + checkpoint_idx))
            update_status("running", o_bar.n, o_bar.total, "Training", is_ready)

    update_status("finished", o_bar.n, o_bar.total, "Done", True)
    print("\n" * (len(vid_data) + 1) + "✅ Dataset bereit.")