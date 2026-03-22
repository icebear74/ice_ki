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

# NOTE: config.py is a LOCAL configuration file that exists on each developer's machine.
# It is listed in .gitignore (line 58) and should NEVER be pushed to the repository!
# 
# To create your config.py:
#   cp config.py.example config.py
#   OR
#   cp config.py.active config.py  (if you have an active config)
# 
# Then edit config.py to match your local setup (paths, GPU settings, etc.)
import config as cfg

# ANSI colors
C_GREEN = "\033[92m"
C_CYAN = "\033[96m"
C_RED = "\033[91m"
C_YELLOW = "\033[93m"
C_BOLD = "\033[1m"
C_RESET = "\033[0m"

# Canonical list of all supported training/validation size keys
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
    config = cfg.get_config()

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
            
            print(f"{C_GREEN}✅ All checkpoints, logs, and TensorBoard events cleaned up{C_RESET}\n")
    
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
    
    # Create model - USING 7-FRAME MODEL (as intended by dataset_generator_v2)
    print("Creating 7-frame model...")
    device = select_gpu()
    use_checkpointing = config.get('USE_CHECKPOINTING', True)
    use_amp = config.get('USE_AMP', False)
    model = VSRBidirectional_7frames_3x(
        n_feats=n_feats, 
        n_blocks=n_blocks,
        use_checkpointing=use_checkpointing
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

    # All paths come from config.py — no runtime_config.json
    data_root        = DATASET_ROOT
    train_gt_pattern = 'patches/{size_key}/GT'
    train_lr_pattern = 'patches/{size_key}/LR_7frames'

    # Detect which size directories exist and have GT files
    available_sizes = []
    print(f"{C_CYAN}Checking for dataset sizes in: {os.path.join(data_root, dataset_name)}{C_RESET}")
    print(f"{C_CYAN}  Using path pattern: {train_gt_pattern}{C_RESET}")

    for size_key in KNOWN_SIZE_KEYS:
        train_dir = os.path.join(data_root, dataset_name,
                                 train_gt_pattern.replace('{size_key}', size_key))
        print(f"{C_CYAN}  Checking {size_key}: {train_dir}{C_RESET}")
        if os.path.exists(train_dir):
            files = [f for f in os.listdir(train_dir) if f.lower().endswith('.png')]
            if files:
                available_sizes.append(size_key)
                print(f"{C_GREEN}    ✓ Found {len(files)} files{C_RESET}")
            else:
                print(f"{C_YELLOW}    ⚠ Directory exists but no .png files found{C_RESET}")
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
                    print(f"{C_RED}❌ size_key '{size_key}' not in FIXED_BATCH_CONFIG — add it to train.py!{C_RESET}")
                    raise ValueError(f"Unknown size_key '{size_key}'")
                sizes_config[size_key] = {
                    'enabled':    True,
                    'distribution': 1.0 / len(available_sizes),
                    'batch_size': batch_cfg['batch'],
                    'accum':      batch_cfg['accum'],
                }

            # Startup diagnostic: file counts per size
            import time as _time
            print(f"\n{C_CYAN}{'━'*56}")
            print(f"  📋  DATASET FILE COUNTS (pre-load diagnostic)")
            print(f"{'━'*56}{C_RESET}")
            for sk in available_sizes:
                gt_dir = os.path.join(data_root, dataset_name,
                                      train_gt_pattern.replace('{size_key}', sk))
                lr_dir = os.path.join(data_root, dataset_name,
                                      train_lr_pattern.replace('{size_key}', sk))
                gt_files = sorted([f for f in os.listdir(gt_dir)
                                   if f.lower().endswith('.png')]) if os.path.isdir(gt_dir) else []
                lr_files = sorted([f for f in os.listdir(lr_dir)
                                   if f.lower().endswith('.png')]) if os.path.isdir(lr_dir) else []
                match_count = len(set(gt_files) & set(lr_files))
                ok = len(gt_files) > 0 and len(lr_files) > 0 and match_count > 0
                status = f"{C_GREEN}✓" if ok else f"{C_RED}✗"
                cfg_info = FIXED_BATCH_CONFIG.get(sk, {})
                print(f"  {status}  {sk:8s}{C_RESET}  GT={len(gt_files):6,}  LR={len(lr_files):6,}  "
                      f"matched={match_count:6,}  BS={cfg_info.get('batch','?')} accum={cfg_info.get('accum','?')}")
                if not os.path.isdir(lr_dir):
                    print(f"           {C_RED}⚠  LR directory NOT FOUND: {lr_dir}{C_RESET}")
                elif len(lr_files) == 0:
                    print(f"           {C_YELLOW}⚠  LR directory is empty{C_RESET}")
                elif match_count == 0:
                    print(f"           {C_RED}⚠  No GT/LR filename matches!{C_RESET}")
            print(f"{C_CYAN}{'━'*56}{C_RESET}")
            print(f"{C_YELLOW}  ⏳  Starting in 10 seconds — press Ctrl+C to abort …{C_RESET}")
            for _i in range(10, 0, -1):
                print(f"      {_i} …", end='\r', flush=True)
                _time.sleep(1)
            print(f"  {C_GREEN}▶  Continuing …{C_RESET}                    ")
            print(f"{C_CYAN}{'━'*56}{C_RESET}\n")

            loader_config = {
                'data_root':        data_root,
                'dataset_name':     dataset_name,
                'sizes':            sizes_config,
                'augment':          True,
                'shuffle':          True,
                'paths':            None,  # use default path patterns
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
            from vsr_plusplus_NEU.core.dataloader import DataStrategyScheduler
            data_strategy_scheduler = DataStrategyScheduler(
                all_size_keys=list(train_loader.datasets_dict.keys())
            )
            print(f"{C_CYAN}📅 DataStrategyScheduler enabled:{C_RESET}")
            print(f"  • Phase 1 (steps 0–{DataStrategyScheduler.WARMUP_END}): "
                  f"100% 720_169 only, perceptual=0.0")
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

    # Ensure validation GT subdirs exist so the user can copy images into them
    for size_key in KNOWN_SIZE_KEYS:
        os.makedirs(os.path.join(data_root, dataset_name, 'val', size_key, 'GT'), exist_ok=True)
    print(f"{C_GREEN}✅ Validation GT directories ready: "
          f"{os.path.join(data_root, dataset_name, 'val', '{size_key}', 'GT')}{C_RESET}")

    # Auto-detect validation sizes
    val_sizes = []
    for sk in KNOWN_SIZE_KEYS:
        val_dir = os.path.join(data_root, dataset_name, val_gt_pattern.replace('{size_key}', sk))
        if os.path.isdir(val_dir):
            files = [f for f in os.listdir(val_dir) if f.lower().endswith('.png')]
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

    # Create checkpoint manager
    checkpoint_mgr = CheckpointManager(DATASET_SPECIFIC_ROOT)
    
    # Create loggers (use dataset-specific paths)
    log_dir = os.path.join(DATASET_SPECIFIC_ROOT, "logs")
    train_logger = TrainingLogger(DATASET_SPECIFIC_ROOT)
    tb_logger = TensorBoardLogger(log_dir)
    
    # Create validator
    validator = VSRValidator(model, val_loader, loss_fn, device=device)
    
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
    
    # Start training
    print("="*80)
    print("🚀 Starting training...")
    print("="*80 + "\n")
    
    trainer.run()
    
    print("\n" + "="*80)
    print("✅ Training complete!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
