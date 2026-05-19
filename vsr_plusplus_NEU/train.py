#!/usr/bin/env python3
"""
VSR++ Training Entry Point

Orchestrates the complete training system:
- Manual configuration (edit config.py)
- Model creation
- Data loading
- Training loop
- Checkpoint management
- Logging
"""

import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import subprocess
import socket
import time
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add current directory to path for local config.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vsr_plusplus_NEU.core.model_7frame import VSRBidirectional_7frames_3x
from vsr_plusplus_NEU.core.loss import HybridLoss
from vsr_plusplus_NEU.core.dataset import VSRDataset
from vsr_plusplus_NEU.training.trainer import VSRTrainer
from vsr_plusplus_NEU.training.validator import VSRValidator
from vsr_plusplus_NEU.training.lr_scheduler import AdaptiveLRScheduler
from vsr_plusplus_NEU.systems.checkpoint_manager import CheckpointManager
from vsr_plusplus_NEU.systems.logger import TrainingLogger, TensorBoardLogger
from vsr_plusplus_NEU.systems.adaptive_system import AdaptiveSystem
from vsr_plusplus_NEU.systems.run_lock import save_run_lock, load_and_verify_run_lock

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG LOADING — ALWAYS FROM config.py (local, gitignored)
# ══════════════════════════════════════════════════════════════════════════════
#
#  ► The ONLY config file that is ever loaded at runtime is  config.py.
#  ► There is NO fallback to config.active.py — if config.py is missing,
#    Python raises an ImportError immediately.  This is intentional.
#  ► config.active.py is a VERSIONED TEMPLATE stored in git.  It is NEVER
#    imported or read by any part of the training code.
#
#  Typical cause of wrong/missing settings (e.g. ASYNC_VAL_GPU=None,
#  USE_SR_MODEL not taking effect):
#    → Your local config.py is stale and missing new parameters.
#  Fix:
#      cp vsr_plusplus_NEU/config.active.py config.py
#      # then re-apply your local edits (DATA_ROOT, GPU index, …)
#
#  config.py is listed in .gitignore and must NEVER be committed.
# ══════════════════════════════════════════════════════════════════════════════
import config as cfg  # ← ALWAYS config.py — NO fallback to config.active.py

# ANSI colors
C_GREEN = "\033[92m"
C_CYAN = "\033[96m"
C_RED = "\033[91m"
C_YELLOW = "\033[93m"
C_BOLD = "\033[1m"
C_RESET = "\033[0m"

# Canonical fallback list of size keys used when no dataset_architecture.json is found.
KNOWN_SIZE_KEYS = ['540', '720', '720_169']

# Default per-size batch and gradient accumulation configuration.
# Used as fallback when ADAPTIVE_BATCH_CONFIG is not present in config.py.
# PRIMARY SOURCE: config.py → ADAPTIVE_BATCH_CONFIG  (set after config is loaded in main())
#
#   720_169 (720×405) – 16:9 full frames:  BS=2, accum=4 → eff=8  (~5.14 GB)
#   540     (540×540) – 1080p crops:       BS=2, accum=3 → eff=6  (~5.15 GB)
#   720     (720×720) – 4K crops:          BS=1, accum=4 → eff=4  (~6.14 GB, BS=2 = OOM)
_DEFAULT_BATCH_CONFIG = {
    '720_169': {'batch': 2, 'accum': 4},
    '540':     {'batch': 2, 'accum': 3},
    '720':     {'batch': 1, 'accum': 4},
}
# Runtime batch config — overwritten in main() from config.ADAPTIVE_BATCH_CONFIG.
# Code outside main() that needs it should use get_batch_config() below.
FIXED_BATCH_CONFIG = _DEFAULT_BATCH_CONFIG  # backward-compat alias; prefer batch_config in main()


def select_gpu() -> torch.device:
    """Fragt beim Start, welche GPU verwendet werden soll, wenn mehrere vorhanden sind.

    Returns:
        torch.device: Ausgewähltes Gerät (z.B. 'cuda:0', 'cuda:1', 'cpu').
    """
    if not torch.cuda.is_available():
        print(f"{C_YELLOW}⚠ Kein CUDA-fähiges Gerät gefunden – Training läuft auf CPU.{C_RESET}")
        return torch.device('cpu')

    gpu_count = torch.cuda.device_count()

    if gpu_count == 1:
        name = torch.cuda.get_device_name(0)
        mem_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"{C_GREEN}✓ GPU erkannt: {name} ({mem_total:.1f} GB){C_RESET}")
        return torch.device('cuda:0')

    # Mehrere GPUs – Auswahl anbieten
    print(f"\n{C_CYAN}{'='*60}{C_RESET}")
    print(f"{C_CYAN}  Mehrere GPUs gefunden – bitte wähle eine aus:{C_RESET}")
    print(f"{C_CYAN}{'='*60}{C_RESET}")
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        mem_total = props.total_memory / (1024 ** 3)
        mem_free = (props.total_memory - torch.cuda.memory_allocated(i)) / (1024 ** 3)
        print(f"  [{i}] {props.name}  –  {mem_total:.1f} GB gesamt, ~{mem_free:.1f} GB frei")
    print(f"{C_CYAN}{'='*60}{C_RESET}")

    while True:
        try:
            raw = input(f"GPU-Index wählen [0–{gpu_count - 1}]: ").strip()
            idx = int(raw)
            if 0 <= idx < gpu_count:
                name = torch.cuda.get_device_name(idx)
                print(f"{C_GREEN}✓ Verwende GPU {idx}: {name}{C_RESET}\n")
                return torch.device(f'cuda:{idx}')
            else:
                print(f"{C_RED}Ungültige Eingabe. Bitte eine Zahl zwischen 0 und {gpu_count - 1} eingeben.{C_RESET}")
        except ValueError:
            print(f"{C_RED}Ungültige Eingabe. Bitte eine Ganzzahl eingeben.{C_RESET}")


def is_tensorboard_running(port=6006):
    """Check if TensorBoard is already running on the specified port"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0
    except:
        return False


def start_tensorboard(log_dir, port=6006):
    """Start TensorBoard subprocess"""
    try:
        # Kill any existing tensorboard processes
        subprocess.run(['pkill', '-f', 'tensorboard'], stderr=subprocess.DEVNULL)
        time.sleep(1)
        
        # Start new tensorboard - point to active_run subdirectory
        active_run_dir = os.path.join(log_dir, "active_run")
        cmd = ['tensorboard', f'--logdir={active_run_dir}', f'--port={port}', '--bind_all', '--reload_interval=5']
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait for it to start (max 5 seconds)
        for _ in range(10):
            time.sleep(0.5)
            if is_tensorboard_running(port):
                print(f"{C_GREEN}✓ TensorBoard started on http://localhost:{port}{C_RESET}")
                return True
        
        print(f"{C_YELLOW}⚠ TensorBoard started but not responding yet on port {port}{C_RESET}")
        return True
    except Exception as e:
        print(f"{C_RED}✗ Failed to start TensorBoard: {e}{C_RESET}")
        return False


def main():
    """Main training entry point"""
    
    # Load configuration from config.py
    # ► ALWAYS config.py — NO fallback to config.active.py (see import block above)
    config = cfg.get_config()

    # ── Stale-config safety net ───────────────────────────────────────────────
    # If the user's local config.py was copied from an older version of
    # config.active.py its get_config() may not include newer parameters
    # (e.g. ASYNC_VAL_GPU, USE_SR_MODEL).  Rather than silently using the
    # default (None / False), we detect missing keys and inject them from the
    # module-level attributes that the user DID set in their config.py.
    # A prominent warning is printed so the user knows to re-copy.
    _REQUIRED_KEYS = ['ASYNC_VAL_GPU', 'USE_SR_MODEL', 'SR_MODEL_PATH']
    _stale_keys = [k for k in _REQUIRED_KEYS if k not in config and hasattr(cfg, k)]
    if _stale_keys:
        print(f"\n{C_YELLOW}{'═'*60}{C_RESET}")
        print(f"{C_YELLOW}  ⚠  STALE config.py DETECTED{C_RESET}")
        print(f"{C_YELLOW}{'─'*60}{C_RESET}")
        print(f"  The following keys are set in config.py but missing from")
        print(f"  get_config() — your config.py needs to be updated:")
        for k in _stale_keys:
            injected = getattr(cfg, k)
            config[k] = injected
            print(f"    {k} = {injected!r}  ← injected from module attribute")
        print(f"{C_YELLOW}  Fix: cp vsr_plusplus_NEU/config.active.py config.py{C_RESET}")
        print(f"{C_YELLOW}{'═'*60}{C_RESET}\n")
    # ─────────────────────────────────────────────────────────────────────────

    # Resolve batch config: prefer ADAPTIVE_BATCH_CONFIG from config.py over the
    # module-level default so users can change batch/accum in config.py and have
    # those values actually take effect everywhere in this file.
    global FIXED_BATCH_CONFIG
    FIXED_BATCH_CONFIG = config.get('ADAPTIVE_BATCH_CONFIG', _DEFAULT_BATCH_CONFIG)

    
    # All paths come from config.py — no runtime_config.json involved
    DATA_ROOT    = config.get('DATA_ROOT',    "/mnt/data/training/Universal/Mastermodell/Learn")
    DATASET_ROOT = config.get('DATASET_ROOT', "/mnt/data/training/Dataset/Universal/Mastermodell")
    dataset_name = config.get('DEFAULT_DATASET_NAME', 'master')
    
    # Dataset-specific root for checkpoints and logs
    DATASET_SPECIFIC_ROOT = os.path.join(DATASET_ROOT, dataset_name)
    
    # Show path configuration
    print(f"\n{C_CYAN}{'='*80}{C_RESET}")
    print(f"{C_CYAN}PATH CONFIGURATION{C_RESET}")
    print(f"{C_CYAN}{'='*80}{C_RESET}")
    print(f"  DATASET_ROOT:          {DATASET_ROOT}")
    print(f"  Dataset Name:          {dataset_name}")
    print(f"  DATASET_SPECIFIC_ROOT: {DATASET_SPECIFIC_ROOT}")
    print(f"  Checkpoints:           {DATASET_SPECIFIC_ROOT}/checkpoint_*.pth")
    print(f"  Logs:                  {DATASET_SPECIFIC_ROOT}/logs/")
    print(f"{C_CYAN}{'='*80}{C_RESET}\n")
    
    print("\n" + "="*80)
    print("VSR++ Training System - Manual Configuration")
    print("="*80 + "\n")
    
    # Print current configuration
    cfg.print_config()
    
    # User choice: DELETE or RESUME
    choice = input("⚠️  [L]öschen oder [F]ortsetzen? (L/F): ").lower()
    
    start_step = 0
    selected_checkpoint_path = None
    is_fresh_start = False
    checkpoint_mgr = CheckpointManager(DATASET_SPECIFIC_ROOT)
    
    if choice == 'l':
        # Safety confirmation to prevent accidental data loss
        print(f"\n{C_RED}{C_BOLD}⚠️  WARNUNG: Alle Trainingsdaten werden gelöscht!{C_RESET}")
        print(f"{C_YELLOW}Checkpoints (.pth) werden als .BAK gesichert.{C_RESET}")
        confirm = input(f"\n{C_RED}Sind Sie sicher? (ja/nein): {C_RESET}").lower()
        
        if confirm != 'ja':
            # User canceled - offer to resume instead
            print(f"\n{C_GREEN}✓ Abbruch - Training wird fortgesetzt{C_RESET}\n")
            choice = 'f'  # Switch to resume mode
        else:
            # Proceed with deletion
            print(f"\n{C_CYAN}🗑️  Starting fresh training...{C_RESET}")
            print(f"{C_CYAN}Sichere .pth Dateien...{C_RESET}")
            
            # Cleanup everything for fresh start (now includes backup)
            log_dir = os.path.join(DATASET_SPECIFIC_ROOT, "logs")
            backed_up = checkpoint_mgr.cleanup_all_for_fresh_start(log_dir)
            
            if backed_up > 0:
                print(f"{C_GREEN}✓ {backed_up} .pth Dateien als .BAK gesichert{C_RESET}")
            
            # Remove old lock file so a new one is written below
            _old_lock = os.path.join(DATASET_SPECIFIC_ROOT, "training_run_locked.json")
            if os.path.exists(_old_lock):
                os.remove(_old_lock)
                print(f"{C_CYAN}🔓 Old run lock removed (fresh start){C_RESET}")
            
            print(f"{C_GREEN}✅ All checkpoints, logs, and TensorBoard events cleaned up{C_RESET}\n")
            is_fresh_start = True
    
    if choice != 'l' or choice == 'f':
        # Resume mode (either selected 'f' or canceled 'l')
        print("\n📂 Resuming training...\n")
        
        # DEBUG: Show where we're looking for checkpoints
        print(f"{C_CYAN}Searching for checkpoints in: {DATASET_SPECIFIC_ROOT}{C_RESET}")
        print(f"{C_CYAN}Looking for pattern: checkpoint_*.pth{C_RESET}\n")
        
        # Use shared checkpoint selector module
        from vsr_plusplus_NEU.utils.checkpoint_selector import select_checkpoint_interactive
        
        selected_ckpt = select_checkpoint_interactive(checkpoint_mgr, auto_select_latest=False)
        
        if selected_ckpt:
            start_step = selected_ckpt['step']
            selected_checkpoint_path = selected_ckpt['path']
        else:
            print("⚠️  No checkpoint found, starting fresh")
            is_fresh_start = True
    
    # Start TensorBoard with dataset-specific log directory
    log_dir = os.path.join(DATASET_SPECIFIC_ROOT, "logs")
    print(f"\n{C_CYAN}Checking TensorBoard...{C_RESET}")
    print(f"{C_CYAN}Log directory: {log_dir}{C_RESET}")
    if not is_tensorboard_running():
        print(f"{C_YELLOW}Starting TensorBoard...{C_RESET}")
        start_tensorboard(log_dir)
    else:
        print(f"{C_GREEN}✓ TensorBoard already running{C_RESET}")
    print()
    
    # Extract parameters from config
    n_feats = config['N_FEATS']
    n_blocks = config['N_BLOCKS']
    
    # Create model
    print(f"Creating {arch_n_frames}-frame model...")
    device = select_gpu()
    use_checkpointing = config.get('USE_CHECKPOINTING', True)
    use_amp = config.get('USE_AMP', False)
    model = VSRBidirectional_7frames_3x(
        n_feats=n_feats,
        n_blocks=n_blocks,
        use_checkpointing=use_checkpointing,
        n_frames=arch_n_frames,
    ).to(device)

    # Verify FP32: ensure model parameters are in the expected dtype.
    # With AMP enabled the master weights remain FP32 while forward/backward
    # passes run in FP16 on the P100's native FP16 hardware.
    model = model.float()
    param_dtypes = set(p.dtype for p in model.parameters())
    if param_dtypes != {torch.float32}:
        raise RuntimeError(f"Model parameters are NOT float32! Found: {param_dtypes}")
    amp_status = "ON (P100 native FP16)" if use_amp else "OFF"
    print(f"{C_GREEN}✅ Model dtype: float32 (FP32 master weights) — AMP/FP16 is {amp_status}{C_RESET}")
    
    if use_checkpointing:
        print("✅ Gradient checkpointing ENABLED - saves ~40% activation memory")
    else:
        print("⚠️  Gradient checkpointing disabled")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {total_params/1e6:.2f}M parameters\n")
    
    # Create loss function with configured weights
    loss_fn = HybridLoss(
        l1_weight=config['L1_WEIGHT'],
        ms_weight=config['MS_WEIGHT'],
        grad_weight=config['GRAD_WEIGHT'],
        perceptual_weight=config.get('PERCEPTUAL_WEIGHT', 0.0)
    ).to(device)  # FIXED: Move loss function to same device as model
    
    
    # Create optimizer with layer-wise learning rates
    # Give Final Fusion layer 20x higher learning rate to strongly activate it
    lr = 10 ** config['LR_EXPONENT']
    
    # Separate parameters into 3 groups with different learning rates:
    # - final_fusion_params (20×): all fusion.* (GatedFusionBlock incl. gate)
    # - align_fuse_params (5×): per-frame alignment + fusion branches
    # - other_params (1×): everything else
    final_fusion_params = []
    align_fuse_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if name.startswith('fusion.'):
            # Final GatedFusionBlock — all parameters including gate
            final_fusion_params.append(param)
        elif any(name.startswith(prefix) for prefix in [
            'backward_align.', 'forward_align.',
            'backward_fuse.', 'forward_fuse.'
        ]):
            # Per-frame alignment and fusion — moderate boost
            align_fuse_params.append(param)
        else:
            other_params.append(param)
    
    print(f"  Parameter groups:")
    print(f"    other_params:        {sum(p.numel() for p in other_params)/1e6:.2f}M params  (lr×1)")
    print(f"    final_fusion_params: {sum(p.numel() for p in final_fusion_params)/1e6:.2f}M params  (lr×20)")
    print(f"    align_fuse_params:   {sum(p.numel() for p in align_fuse_params)/1e6:.2f}M params  (lr×5)")
    
    # Create parameter groups with different learning rates
    param_groups = [
        {
            'params': other_params,
            'lr': lr,
            'weight_decay': config['WEIGHT_DECAY']
        },
        {
            'params': final_fusion_params,
            'lr': lr * 20,  # 20× for Final GatedFusionBlock (incl. gate)
            'weight_decay': 0.0  # No weight decay for final fusion
        },
        {
            'params': align_fuse_params,
            'lr': lr * 5,   # 5× for per-frame alignment + fusion
            'weight_decay': config['WEIGHT_DECAY']
        },
    ]
    
    optimizer = optim.AdamW(param_groups)
    
    # Create LR scheduler
    # Initial LR for warmup start (from config)
    initial_lr = 10 ** config['LR_EXPONENT']
    
    lr_scheduler = AdaptiveLRScheduler(
        optimizer,
        warmup_steps=config['WARMUP_STEPS'],
        max_steps=config['MAX_STEPS'],
        max_lr=config['MAX_LR'],
        min_lr=config['MIN_LR'],
        initial_lr=initial_lr
    )
    
    # Initialize LR for step 0 (warmup start)
    lr_scheduler.step(0)
    
    # Create adaptive system
    if config['ADAPTIVE_LOSS_WEIGHTS'] or config['ADAPTIVE_GRAD_CLIP']:
        adaptive_system = AdaptiveSystem(
            initial_l1=config['L1_WEIGHT'],
            initial_ms=config['MS_WEIGHT'],
            initial_grad=config['GRAD_WEIGHT'],
            initial_perceptual=config.get('PERCEPTUAL_WEIGHT', 0.0)  # NEW: Pass perceptual weight
        )
    else:
        # Use fixed weights if adaptive is disabled
        adaptive_system = AdaptiveSystem(
            initial_l1=config['L1_WEIGHT'],
            initial_ms=config['MS_WEIGHT'],
            initial_grad=config['GRAD_WEIGHT'],
            initial_perceptual=config.get('PERCEPTUAL_WEIGHT', 0.0)  # NEW: Pass perceptual weight
        )
    
    # Create GradScaler for mixed precision training if enabled
    # (use_amp was already read from config above, alongside use_checkpointing)
    scaler = GradScaler('cuda') if use_amp else None
    
    if use_amp:
        print(f"{C_GREEN}✅ Mixed Precision (AMP) enabled - Tesla P100 native FP16 hardware active{C_RESET}\n")
    else:
        print(f"{C_GREEN}✅ Mixed Precision (AMP) disabled - training in pure FP32{C_RESET}\n")
    
    # Create datasets
    print("Loading datasets...")

    # All paths come from config.py
    data_root    = DATASET_ROOT

    # ------------------------------------------------------------------
    # Load dataset_architecture.json to discover format keys, n_frames,
    # and output image format.  Falls back gracefully when not present.
    # ------------------------------------------------------------------
    from vsr_plusplus_NEU.utils.dataset_architecture import load_dataset_architecture
    arch = load_dataset_architecture(data_root)
    if arch is not None:
        print(f"{C_GREEN}✓ Loaded dataset_architecture.json: {arch}{C_RESET}")
        arch_n_frames = arch.n_frames
        arch_img_ext  = arch.img_ext          # e.g. ".bmp" or ".png"
        arch_lr_dir_name = arch.get_lr_dir_name()   # e.g. "LR_7frames"
        # Size keys from the architecture JSON take priority over KNOWN_SIZE_KEYS
        arch_size_keys = arch.get_templates_for_category(dataset_name)
        if arch_size_keys:
            print(f"{C_CYAN}  Templates for category '{dataset_name}': {', '.join(arch_size_keys)}{C_RESET}")
        else:
            print(f"{C_YELLOW}  ⚠ No templates found for category '{dataset_name}' in architecture file{C_RESET}")
            arch_size_keys = []
    else:
        print(f"{C_YELLOW}⚠ dataset_architecture.json not found at {data_root} — using defaults{C_RESET}")
        arch_n_frames    = 7
        arch_img_ext     = ".png"             # legacy default
        arch_lr_dir_name = "LR_7frames"
        arch_size_keys   = []

    # ------------------------------------------------------------------
    # Model constraint validation (early abort on unsupported n_frames).
    #
    # The model supports any odd frame count ≥ 3 (e.g. 3, 5, 7, 9).
    # Abort early if the architecture file specifies an invalid value so
    # that checkpoints remain compatible.
    # ------------------------------------------------------------------
    FIXED_SCALE = 3

    if arch_n_frames < 3 or arch_n_frames % 2 == 0:
        print(f"\n{C_RED}{'='*72}{C_RESET}")
        print(f"{C_RED}❌  INVALID N_FRAMES{C_RESET}")
        print(f"{C_RED}    dataset_architecture.json says n_frames={arch_n_frames}{C_RESET}")
        print(f"{C_RED}    Supported values: odd numbers ≥ 3 (e.g. 3, 5, 7, 9).{C_RESET}")
        print(f"{C_RED}    Training aborted — update the architecture file.{C_RESET}")
        print(f"{C_RED}{'='*72}{C_RESET}\n")
        return

    # scale is not stored in the architecture file but is implicit in the LR
    # directory dimensions; we simply document the expectation here.
    print(f"{C_GREEN}✅ Architecture validated: n_frames={arch_n_frames}, scale={FIXED_SCALE} (fixed){C_RESET}")

    train_gt_pattern = 'patches/{size_key}/GT'
    train_lr_pattern = f'patches/{{size_key}}/{arch_lr_dir_name}'

    # Size keys to probe: architecture JSON first, then KNOWN_SIZE_KEYS as fallback
    probe_size_keys = arch_size_keys if arch_size_keys else KNOWN_SIZE_KEYS

    # Detect which size directories exist and have image files
    available_sizes = []
    print(f"{C_CYAN}Checking for dataset sizes in: {os.path.join(data_root, dataset_name)}{C_RESET}")
    print(f"{C_CYAN}  Using path pattern: {train_gt_pattern}  (ext: {arch_img_ext}){C_RESET}")

    from vsr_plusplus_NEU.core.dataset import _collect_image_files
    for size_key in probe_size_keys:
        train_dir = os.path.join(data_root, dataset_name,
                                 train_gt_pattern.replace('{size_key}', size_key))
        print(f"{C_CYAN}  Checking {size_key}: {train_dir}{C_RESET}")
        if os.path.exists(train_dir):
            # Single scan: use arch_img_ext as hint but accept any supported ext
            # to handle datasets where the format differs from the architecture file.
            files = _collect_image_files(train_dir, arch_img_ext)
            if not files:
                files = _collect_image_files(train_dir, "")  # fallback: any supported ext
            if files:
                available_sizes.append(size_key)
                print(f"{C_GREEN}    ✓ Found {len(files)} files{C_RESET}")
            else:
                print(f"{C_YELLOW}    ⚠ Directory exists but no image files found{C_RESET}")
        else:
            print(f"{C_YELLOW}    ⚠ Directory does not exist{C_RESET}")

    use_multi_size = bool(available_sizes)
    if use_multi_size:
        if len(available_sizes) == 1:
            print(f"{C_CYAN}✓ Single-size detected ({available_sizes[0]}), using multi-size loader for consistency{C_RESET}")
        else:
            print(f"{C_CYAN}✓ Multi-size training enabled: {', '.join(available_sizes)}{C_RESET}")
    else:
        print(f"{C_YELLOW}⚠ No training data found in {os.path.join(data_root, dataset_name)}{C_RESET}")
        print(f"{C_YELLOW}  Falling back to single-size training (size_key=540){C_RESET}")

    # Graduated data/loss strategy scheduler (set to None unless multi-size)
    data_strategy_scheduler = None

    if use_multi_size:
        # Build sizes_config from FIXED_BATCH_CONFIG
        try:
            from vsr_plusplus_NEU.core.dataloader import create_train_loader

            sizes_config = {}
            for size_key in available_sizes:
                batch_cfg = FIXED_BATCH_CONFIG.get(size_key)
                if batch_cfg is None:
                    # Dynamic size keys from V2 architecture use a safe default
                    print(f"{C_YELLOW}⚠ size_key '{size_key}' not in FIXED_BATCH_CONFIG — using safe defaults (batch=1, accum=4){C_RESET}")
                    batch_cfg = {'batch': 1, 'accum': 4}
                sizes_config[size_key] = {
                    'enabled':    True,
                    'distribution': 1.0 / len(available_sizes),
                    'batch_size': batch_cfg['batch'],
                    'accum':      batch_cfg['accum'],
                }

            # Startup diagnostic: file counts per size (bucket-aware, ext-aware)
            import time as _time
            print(f"\n{C_CYAN}{'━'*56}")
            print(f"  📋  DATASET FILE COUNTS (pre-load diagnostic)")
            print(f"{'━'*56}{C_RESET}")
            for sk in available_sizes:
                gt_dir = os.path.join(data_root, dataset_name,
                                      train_gt_pattern.replace('{size_key}', sk))
                lr_dir = os.path.join(data_root, dataset_name,
                                      train_lr_pattern.replace('{size_key}', sk))
                gt_files = _collect_image_files(gt_dir, arch_img_ext) if os.path.isdir(gt_dir) else []
                lr_files = _collect_image_files(lr_dir, arch_img_ext) if os.path.isdir(lr_dir) else []
                # Match by basename only (bucket paths differ between GT and LR)
                gt_bases = {os.path.basename(f) for f in gt_files}
                lr_bases = {os.path.basename(f) for f in lr_files}
                match_count = len(gt_bases & lr_bases)
                ok = len(gt_files) > 0 and len(lr_files) > 0 and match_count > 0
                status = f"{C_GREEN}✓" if ok else f"{C_RED}✗"
                cfg_info = FIXED_BATCH_CONFIG.get(sk, {'batch': 1, 'accum': 4})
                # Width 12 accommodates long V2 template names like "1152_169"
                print(f"  {status}  {sk:12s}{C_RESET}  GT={len(gt_files):6,}  LR={len(lr_files):6,}  "
                      f"matched={match_count:6,}  BS={cfg_info.get('batch','?')} accum={cfg_info.get('accum','?')}")
                if not os.path.isdir(lr_dir):
                    print(f"           {C_RED}⚠  LR directory NOT FOUND: {lr_dir}{C_RESET}")
                elif len(lr_files) == 0:
                    print(f"           {C_YELLOW}⚠  LR directory is empty{C_RESET}")
                elif match_count == 0:
                    print(f"           {C_RED}⚠  No GT/LR filename matches!{C_RESET}")
            print(f"{C_CYAN}{'━'*56}{C_RESET}\n")

            loader_config = {
                'data_root':        data_root,
                'dataset_name':     dataset_name,
                'sizes':            sizes_config,
                'augment':          True,
                'shuffle':          True,
                'paths':            None,  # use default path patterns
                'n_frames':         arch_n_frames,
                'img_ext':          arch_img_ext,
                'prefetch_count':   config.get('PREFETCH_BATCHES',     10),
                'prefetch_workers': config.get('PREFETCH_WORKERS',      1),
                'pin_workers':      config.get('PREFETCH_PIN_WORKERS',  1),
            }
            train_loader = create_train_loader(loader_config)

            total_samples = sum(len(ds) for ds in train_loader.datasets_dict.values())
            print(f"✅ Multi-size training samples: {total_samples:,}")
            print(f"{C_CYAN}{'━'*47}{C_RESET}")
            print(f"{C_CYAN}📊 Dataset Sizes Loaded at Startup:{C_RESET}")
            for sk, ds in train_loader.datasets_dict.items():
                dist = len(ds) / total_samples if total_samples > 0 else 0.0
                cfg_info = FIXED_BATCH_CONFIG.get(sk, {})
                print(f"  • {sk}: {len(ds):,} samples ({dist*100:.1f}%)  "
                      f"BS={cfg_info.get('batch','?')} accum={cfg_info.get('accum','?')}")
            print(f"{C_CYAN}{'━'*47}{C_RESET}\n")

            # Graduated data/loss strategy scheduler
            # Build template_areas from arch metadata (needed to choose warmup template
            # dynamically as the one with the largest GT area).
            _template_areas = {}
            _arch_weights = {}
            if arch is not None:
                try:
                    for _sk in list(train_loader.datasets_dict.keys()):
                        _entry = arch.get_format_entry(dataset_name, _sk)
                        if _entry:
                            _gt = _entry.get('gt_size')  # [width, height]
                            if _gt and len(_gt) == 2:
                                _template_areas[_sk] = _gt[0] * _gt[1]
                            _wt = _entry.get('weight', 0.0)
                            if _wt:
                                _arch_weights[_sk] = float(_wt)
                except Exception:
                    pass  # Fallback: equal shares (handled by DataStrategyScheduler)

            from vsr_plusplus_NEU.core.dataloader import DataStrategyScheduler
            data_strategy_scheduler = DataStrategyScheduler(
                all_size_keys=list(train_loader.datasets_dict.keys()),
                template_areas=_template_areas or None,
                arch_weights=_arch_weights or None,
            )
            _warmup_tmpl = data_strategy_scheduler.warmup_template or 'unknown'
            print(f"{C_CYAN}📅 DataStrategyScheduler enabled:{C_RESET}")
            print(f"  • Phase 1 (steps 0–{DataStrategyScheduler.WARMUP_END}): "
                  f"100% {_warmup_tmpl} (largest GT area), perceptual=0.0")
            print(f"  • Phase 2 (steps {DataStrategyScheduler.WARMUP_END}–"
                  f"{DataStrategyScheduler.CROP_INTRO_END}): "
                  f"linear mix-in, perceptual 0.0→{DataStrategyScheduler.TARGET_PERCEPTUAL_WEIGHT}")
            print(f"  • Phase 3 (steps {DataStrategyScheduler.CROP_INTRO_END}+): "
                  f"natural file-count sampling, perceptual={DataStrategyScheduler.TARGET_PERCEPTUAL_WEIGHT}")
            print()

        except Exception as e:
            import traceback
            print(f"{C_RED}❌ Error creating multi-size dataloader: {e}{C_RESET}")
            traceback.print_exc()
            print(f"{C_YELLOW}Falling back to single-size training{C_RESET}")
            use_multi_size = False

    if not use_multi_size:
        # Single-size fallback: use first detected size or '540' as last resort
        size_key = available_sizes[0] if available_sizes else '540'
        batch_cfg = FIXED_BATCH_CONFIG.get(size_key, {'batch': 1, 'accum': 4})
        try:
            train_dataset = VSRDataset(
                root=data_root,
                dataset_name=dataset_name,
                size_key=size_key,
                mode='train',
                augment=True,
                n_frames=arch_n_frames,
                img_ext=arch_img_ext,
                paths_config=None,
            )
            print(f"✅ Training samples: {len(train_dataset):,}\n")
        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
            return

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_cfg['batch'],
            shuffle=True,
            num_workers=config['NUM_WORKERS'],
            pin_memory=config['PIN_MEMORY'],
        )
    
    # Load validation datasets — auto-detect from val/{size_key}/GT directories
    val_loaders = []  # List of (size_key, loader) tuples
    val_gt_pattern = 'val/{size_key}/GT'

    # Ensure validation GT subdirs exist for all known template keys.
    # Use arch_size_keys when available, otherwise KNOWN_SIZE_KEYS.
    val_template_keys = arch_size_keys if arch_size_keys else KNOWN_SIZE_KEYS
    for size_key in val_template_keys:
        os.makedirs(os.path.join(data_root, dataset_name, 'val', size_key, 'GT'), exist_ok=True)
    print(f"{C_GREEN}✅ Validation GT directories ready under "
          f"{os.path.join(data_root, dataset_name, 'val')}{C_RESET}")

    # Auto-detect validation sizes (check for any image files, BMP or PNG)
    val_sizes = []
    for sk in val_template_keys:
        val_dir = os.path.join(data_root, dataset_name, val_gt_pattern.replace('{size_key}', sk))
        if os.path.isdir(val_dir):
            files = _collect_image_files(val_dir, arch_img_ext)
            if not files:
                # Try any supported format (handles mixed or auto-detected layouts)
                files = _collect_image_files(val_dir, "")
            if files:
                val_sizes.append(sk)
    if not val_sizes:
        val_sizes = available_sizes[:1] if available_sizes else ['540']

    print(f"{C_CYAN}Creating validation datasets for sizes: {', '.join(val_sizes)}{C_RESET}")
    total_val_samples = 0

    for size_key in val_sizes:
        try:
            val_dataset = VSRDataset(
                root=data_root,
                dataset_name=dataset_name,
                size_key=size_key,
                mode='val',
                augment=False,
                n_frames=arch_n_frames,
                img_ext=arch_img_ext,
                paths_config=None,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.get('VAL_BATCH_SIZE', 1),
                shuffle=False,
                num_workers=2,
                pin_memory=False,
            )
            val_loaders.append((size_key, val_loader))
            total_val_samples += len(val_dataset)
            print(f"  ✓ {size_key}: {len(val_dataset):,} samples")
        except Exception as e:
            print(f"  ⚠️  Warning: Could not load validation for {size_key}: {e}")

    if not val_loaders:
        print(f"❌ Error: No validation datasets loaded!")
        return

    print(f"{C_GREEN}✅ Total validation samples: {total_val_samples:,} across {len(val_loaders)} sizes{C_RESET}\n")

    # First loader is used as the primary validator target
    val_loader = val_loaders[0][1]

    # ------------------------------------------------------------------
    # Locked run config — checkpoint compatibility guard
    #
    # On fresh start: writes training_run_locked.json with the key model
    # and dataset parameters.  On resume: loads the locked file and aborts
    # with a clear error if any critical parameter has changed.
    # ------------------------------------------------------------------
    _run_templates = sorted(available_sizes)
    if is_fresh_start:
        _lock_path = save_run_lock(
            run_dir=DATASET_SPECIFIC_ROOT,
            n_feats=n_feats,
            n_blocks=n_blocks,
            n_frames=arch_n_frames,
            scale=FIXED_SCALE,
            dataset_root=data_root,
            category=dataset_name,
            templates=_run_templates,
        )
        print(f"{C_GREEN}🔒 Run lock created: {_lock_path}{C_RESET}")
    else:
        # On resume: verify the lock is compatible with the current config.
        # load_and_verify_run_lock() calls sys.exit(1) on mismatch.
        load_and_verify_run_lock(
            run_dir=DATASET_SPECIFIC_ROOT,
            n_feats=n_feats,
            n_blocks=n_blocks,
            n_frames=arch_n_frames,
            scale=FIXED_SCALE,
            dataset_root=data_root,
            category=dataset_name,
            templates=_run_templates,
        )
        # If we reach here, the lock either didn't exist (first run after upgrade)
        # or verification passed.  In the former case, create the lock now.
        save_run_lock(
            run_dir=DATASET_SPECIFIC_ROOT,
            n_feats=n_feats,
            n_blocks=n_blocks,
            n_frames=arch_n_frames,
            scale=FIXED_SCALE,
            dataset_root=data_root,
            category=dataset_name,
            templates=_run_templates,
        )
    # ------------------------------------------------------------------

    # Create checkpoint manager
    checkpoint_mgr = CheckpointManager(DATASET_SPECIFIC_ROOT)
    
    # Create loggers (use dataset-specific paths)
    log_dir = os.path.join(DATASET_SPECIFIC_ROOT, "logs")
    train_logger = TrainingLogger(DATASET_SPECIFIC_ROOT)
    tb_logger = TensorBoardLogger(log_dir)
    
    # Create validator
    # Optionally load EDSR SR model for sync validation when USE_SR_MODEL=True
    _sync_sr_model = None
    if config.get('USE_SR_MODEL', False):
        from vsr_plusplus_NEU.core.sr_model import load_sr_model
        print(f"{C_CYAN}USE_SR_MODEL=True – loading SR model for sync validation...{C_RESET}")
        _sync_sr_model = load_sr_model(device)  # always returns a model (EDSR or Bicubic fallback)

    validator = VSRValidator(model, val_loader, loss_fn, device=device, sr_model=_sync_sr_model)
    
    # Load checkpoint if resuming
    if start_step > 0 and selected_checkpoint_path:
        print(f"Loading checkpoint from {selected_checkpoint_path}...")
        # Use weights_only=False for compatibility with PyTorch 2.6+
        # Our checkpoints contain custom classes (AdaptiveLRScheduler) which are safe to load
        checkpoint = torch.load(selected_checkpoint_path, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Try to load optimizer state, but handle parameter group mismatch
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✅ Optimizer state loaded")
        except ValueError as e:
            if "parameter groups" in str(e):
                print(f"{C_YELLOW}⚠ Optimizer state not loaded: parameter group mismatch{C_RESET}")
                print(f"{C_YELLOW}  Old checkpoint has different optimizer structure{C_RESET}")
                print(f"{C_YELLOW}  Continuing with fresh optimizer state (LR and momentum reset){C_RESET}")
            else:
                raise
        
        # Restore scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            # Note: We'd need to implement state_dict for our scheduler
            pass
        
        print(f"✅ Checkpoint loaded (step {start_step:,})\n")
    
    # Create trainer
    trainer = VSRTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        validator=validator,
        checkpoint_mgr=checkpoint_mgr,
        train_logger=train_logger,
        tb_logger=tb_logger,
        adaptive_system=adaptive_system,
        config=config,
        device=device,
        scaler=scaler,
        use_amp=use_amp
    )
    
    # Pass all validation loaders to trainer for multi-size validation
    trainer.val_loaders = val_loaders
    
    # Attach graduated data/loss strategy scheduler when multi-size training
    if data_strategy_scheduler is not None:
        trainer.data_strategy_scheduler = data_strategy_scheduler
    
    # Set start step
    trainer.set_start_step(start_step)
    
    # Initialize dataset file monitoring with current counts
    print(f"\n{C_CYAN}Initializing dataset file monitoring...{C_RESET}")
    trainer._check_dataset_files()
    print(f"{C_GREEN}✓ Dataset file counts initialized{C_RESET}\n")
    
    # ── Async validation setup ────────────────────────────────────────────────
    # Set ASYNC_VAL_GPU in your config.py to the GPU index that should run
    # validation (e.g. ASYNC_VAL_GPU = 1).  Use 'auto' to automatically pick
    # the GPU that is NOT used for training.  Leave unset (or set to None) to
    # use the default synchronous validation that runs inline on the training GPU.
    async_val_gpu = config.get('ASYNC_VAL_GPU', None)
    _async_val_cfg_value = async_val_gpu  # keep original config value for summary display

    # Resolve 'auto': pick the GPU that is NOT used by training
    if async_val_gpu == 'auto':
        # device.index is None for torch.device('cuda') without explicit index;
        # treat it as index 0 (PyTorch default) so the math below is always valid.
        training_gpu_idx = device.index if (device.type == 'cuda' and device.index is not None) else (0 if device.type == 'cuda' else None)
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"{C_CYAN}ASYNC_VAL_GPU='auto': {gpu_count} GPU(s) found, training on GPU {training_gpu_idx}{C_RESET}")
        if gpu_count >= 2 and training_gpu_idx is not None:
            async_val_gpu = 1 - training_gpu_idx  # works for 2-GPU setups
            print(f"{C_GREEN}✓ ASYNC_VAL_GPU='auto' → training GPU {training_gpu_idx}, "
                  f"validation GPU {async_val_gpu}{C_RESET}")
        else:
            print(f"{C_YELLOW}⚠ ASYNC_VAL_GPU='auto': only {gpu_count} GPU(s) found – "
                  f"disabling async validation (synchronous mode){C_RESET}")
            async_val_gpu = None

    async_val_proc = None
    _async_val_disable_reason = None  # set below when async val cannot start

    if async_val_gpu is not None:
        # Determine which size keys have validation data
        val_sizes_with_data = [sk for sk, _ in val_loaders]

        # Write a config snapshot JSON so the subprocess can reconstruct the model
        config_json_path = os.path.join(DATASET_SPECIFIC_ROOT, 'async_val_config.json')
        config_snapshot = {
            'N_FEATS':           config.get('N_FEATS', 72),
            'N_BLOCKS':          config.get('N_BLOCKS', 24),
            'USE_CHECKPOINTING': config.get('USE_CHECKPOINTING', False),
            'n_frames':          arch_n_frames,
            'L1_WEIGHT':         config.get('L1_WEIGHT', 0.60),
            'MS_WEIGHT':         config.get('MS_WEIGHT', 0.20),
            'GRAD_WEIGHT':       config.get('GRAD_WEIGHT', 0.20),
            'PERCEPTUAL_WEIGHT': config.get('PERCEPTUAL_WEIGHT', 0.0),
            # SR reference model (False = disabled)
            'USE_SR_MODEL':      config.get('USE_SR_MODEL', False),
        }
        try:
            with open(config_json_path, 'w') as f:
                json.dump(config_snapshot, f, indent=2)
        except Exception as e:
            print(f"{C_YELLOW}⚠ Could not write async_val_config.json: {e}{C_RESET}")
            config_json_path = None

        tb_log_dir = os.path.join(DATASET_SPECIFIC_ROOT, "logs")

        # Build the subprocess command.  -u = unbuffered so every print()
        # appears in async_val.log immediately (Python uses block buffering
        # when stdout is redirected to a file).
        async_val_cmd = [
            sys.executable, '-u', '-m', 'vsr_plusplus_NEU.training.async_validator',
            '--checkpoint-dir', DATASET_SPECIFIC_ROOT,
            '--data-root',      data_root,
            '--dataset-name',   dataset_name,
            '--log-dir',        tb_log_dir,
            '--gpu',            str(async_val_gpu),
        ]
        if config_json_path:
            async_val_cmd += ['--config-json', config_json_path]

        print(f"\n{C_CYAN}{'='*60}{C_RESET}")
        print(f"{C_GREEN}🔀 Starting async validation process on GPU {async_val_gpu}{C_RESET}")
        print(f"   Sizes: {val_sizes_with_data}")
        print(f"{C_CYAN}{'='*60}{C_RESET}\n")

        _async_val_log_path = os.path.join(DATASET_SPECIFIC_ROOT, 'async_val.log')
        try:
            _async_val_log_fh = open(_async_val_log_path, 'a')
            async_val_proc = subprocess.Popen(
                async_val_cmd,
                stdout=_async_val_log_fh,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
            # Close our copy of the file handle – the child process has its own.
            _async_val_log_fh.close()
            print(f"{C_GREEN}✓ Async validator PID {async_val_proc.pid}{C_RESET}")
            print(f"  Log: {_async_val_log_path}")
            # Brief grace period: if the process exits immediately it means the
            # subprocess crashed on startup (e.g. import error, GPU unavailable).
            import time as _time
            _time.sleep(2.0)
            if async_val_proc.poll() is not None:
                _exit_code = async_val_proc.returncode
                print(f"{C_YELLOW}⚠ Async validator crashed immediately (exit code {_exit_code}){C_RESET}")
                print(f"{C_YELLOW}  Check {_async_val_log_path} for details{C_RESET}")
                print(f"{C_YELLOW}  Falling back to synchronous validation.{C_RESET}\n")
                _async_val_disable_reason = f"subprocess crashed (exit {_exit_code}) – see async_val.log"
                async_val_proc = None
            else:
                print(f"{C_GREEN}✓ Async validator running\n{C_RESET}")
        except Exception as e:
            print(f"{C_YELLOW}⚠ Failed to start async validator: {e}{C_RESET}")
            print(f"{C_YELLOW}  Falling back to synchronous validation.{C_RESET}\n")
            _async_val_disable_reason = f"subprocess failed to start: {e}"
            async_val_proc = None

        if async_val_proc is not None:
            trainer.enable_async_validation(
                checkpoint_dir=DATASET_SPECIFIC_ROOT,
                val_sizes=val_sizes_with_data,
                log_dir=tb_log_dir,
                proc=async_val_proc,
                restart_cmd=async_val_cmd,
                log_path=_async_val_log_path,
            )
    # ── End async validation setup ────────────────────────────────────────────

    # ── Final setup summary + countdown ──────────────────────────────────────
    # All setup steps are complete.  Print a summary so the user can verify
    # everything before the terminal UI takes over the screen.
    # The same summary is written to training_startup.log for post-hoc debugging.

    # Build async val status line
    if async_val_proc is not None:
        _async_status = f"✅ running on GPU {async_val_gpu} (PID {async_val_proc.pid})"
    elif _async_val_disable_reason:
        _async_status = f"⚠ {_async_val_disable_reason}"
    elif _async_val_cfg_value == 'auto':
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        _async_status = f"disabled (auto: only {gpu_count} GPU found – synchronous mode)"
    elif _async_val_cfg_value is None:
        _async_status = "disabled (ASYNC_VAL_GPU=None in config.py)"
    else:
        _async_status = f"disabled (config value: {_async_val_cfg_value!r})"

    # Build SR model status line
    if _sync_sr_model is not None:
        _sr_status = f"loaded ✅ ({getattr(_sync_sr_model, 'name', type(_sync_sr_model).__name__)})"
    else:
        _sr_status = "disabled (USE_SR_MODEL=False in config.py)"

    _summary_lines = [
        f"{'═'*60}",
        f"  ✅  SETUP COMPLETE – ready to start training",
        f"{'─'*60}",
        f"  Training GPU : {device}",
        f"  SR model     : {_sr_status}",
        f"  Async val    : {_async_status}",
        f"  Steps        : {start_step:,} → {config.get('MAX_STEPS', 150000):,}",
        f"  Config file  : config.py (ASYNC_VAL_GPU={_async_val_cfg_value!r}, USE_SR_MODEL={config.get('USE_SR_MODEL', False)!r})",
        f"{'─'*60}",
    ]

    # Print to console (with ANSI colours)
    print(f"\n{C_CYAN}{'═'*60}{C_RESET}")
    print(f"{C_CYAN}  ✅  SETUP COMPLETE – ready to start training{C_RESET}")
    print(f"{C_CYAN}{'─'*60}{C_RESET}")
    print(f"  Training GPU : {device}")
    print(f"  SR model     : {_sr_status}")
    if async_val_proc is not None:
        print(f"  Async val    : {C_GREEN}{_async_status}{C_RESET}")
    elif _async_val_disable_reason or _async_val_cfg_value is not None:
        print(f"  Async val    : {C_YELLOW}{_async_status}{C_RESET}")
    else:
        print(f"  Async val    : {_async_status}")
    print(f"  Steps        : {start_step:,} → {config.get('MAX_STEPS', 150000):,}")
    print(f"  Config file  : config.py  (ASYNC_VAL_GPU={_async_val_cfg_value!r}, USE_SR_MODEL={config.get('USE_SR_MODEL', False)!r})")
    print(f"{C_CYAN}{'─'*60}{C_RESET}")

    # Write the same summary (no ANSI) to training_startup.log for debugging
    import datetime as _dt
    _startup_log_path = os.path.join(DATASET_SPECIFIC_ROOT, 'training_startup.log')
    try:
        with open(_startup_log_path, 'a', encoding='utf-8') as _slg:
            _slg.write(f"\n[{_dt.datetime.now().isoformat(timespec='seconds')}] Training startup\n")
            for _line in _summary_lines:
                _slg.write(_line + '\n')
            _slg.flush()
        print(f"  Startup log  : {_startup_log_path}")
    except Exception as _log_err:
        print(f"{C_YELLOW}  ⚠ Could not write startup log: {_log_err}{C_RESET}")

    print(f"{C_YELLOW}  ⏳  Starting in 10 seconds — press Ctrl+C to abort …{C_RESET}")
    for _i in range(10, 0, -1):
        print(f"      {_i} …", end='\r', flush=True)
        time.sleep(1)
    print(f"  {C_GREEN}▶  Starting …{C_RESET}                    ")
    print(f"{C_CYAN}{'═'*60}{C_RESET}\n")
    # ─────────────────────────────────────────────────────────────────────────

    # Start training
    print("="*80)
    print("🚀 Starting training...")
    print("="*80 + "\n")
    
    try:
        trainer.run()
    finally:
        # Signal the async validator to stop when training ends
        if async_val_proc is not None:
            stop_file = os.path.join(DATASET_SPECIFIC_ROOT, 'async_val_stop')
            try:
                open(stop_file, 'w').close()
                print(f"{C_CYAN}💬 Async validator stop signal sent.{C_RESET}")
                async_val_proc.wait(timeout=30)
                print(f"{C_GREEN}✓ Async validator exited.{C_RESET}")
            except Exception:
                async_val_proc.terminate()
    
    print("\n" + "="*80)
    print("✅ Training complete!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
