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
import subprocess
import socket
import time
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add current directory to path for local config.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vsr_plusplus_NEU.core.model import VSRBidirectional_3x
from vsr_plusplus_NEU.core.loss import HybridLoss
from vsr_plusplus_NEU.core.dataset import VSRDataset
from vsr_plusplus_NEU.training.trainer import VSRTrainer
from vsr_plusplus_NEU.training.validator import VSRValidator
from vsr_plusplus_NEU.training.lr_scheduler import AdaptiveLRScheduler
from vsr_plusplus_NEU.systems.checkpoint_manager import CheckpointManager
from vsr_plusplus_NEU.systems.logger import TrainingLogger, TensorBoardLogger
from vsr_plusplus_NEU.systems.adaptive_system import AdaptiveSystem
from vsr_plusplus_NEU.systems.runtime_config import RuntimeConfigManager

# Import manual configuration
# Import from local directory (config.py is not in repo due to .gitignore)
import config as cfg

# ANSI colors
C_GREEN = "\033[92m"
C_CYAN = "\033[96m"
C_RED = "\033[91m"
C_YELLOW = "\033[93m"
C_BOLD = "\033[1m"
C_RESET = "\033[0m"


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


def validate_startup_config(runtime_config_manager):
    """
    Validate startup configuration for 7-frame VSR training
    
    Args:
        runtime_config_manager: RuntimeConfigManager instance
        
    Returns:
        True if validation passes, False otherwise
    """
    print(f"\n{C_CYAN}{'='*80}{C_RESET}")
    print(f"{C_CYAN}Validating Startup Configuration{C_RESET}")
    print(f"{C_CYAN}{'='*80}{C_RESET}\n")
    
    # Validate configuration
    is_valid, errors = runtime_config_manager.validate()
    
    if not is_valid:
        print(f"{C_RED}{C_BOLD}❌ Configuration Validation Failed!{C_RESET}\n")
        for i, error in enumerate(errors, 1):
            print(f"{C_RED}  {i}. {error}{C_RESET}")
        print(f"\n{C_YELLOW}Please fix the configuration issues and try again.{C_RESET}\n")
        return False
    
    # Print validation success
    print(f"{C_GREEN}✅ Configuration validation passed!{C_RESET}\n")
    
    # Print key config values
    config = runtime_config_manager.get_all()
    
    if 'model' in config:
        print(f"{C_BOLD}Model Configuration:{C_RESET}")
        print(f"  • Frames: {config['model'].get('n_frames', 'N/A')}")
        print(f"  • Features: {config['model'].get('n_feats', 'N/A')}")
        print(f"  • Blocks: {config['model'].get('n_blocks', 'N/A')}")
        print(f"  • Precision: {config['model'].get('precision', 'N/A')}")
        print()
    
    if 'training' in config:
        print(f"{C_BOLD}Training Configuration:{C_RESET}")
        print(f"  • Effective Batch Size: {config['training'].get('effective_batch_size', 'N/A')}")
        print()
        
        if 'adaptive_batch' in config['training']:
            print(f"  {C_BOLD}Adaptive Batch Configs:{C_RESET}")
            for size, batch_config in config['training']['adaptive_batch'].items():
                batch = batch_config.get('batch', 'N/A')
                accum = batch_config.get('accum', 'N/A')
                print(f"    • {size}: batch={batch}, accum={accum}")
            print()
    
    if 'size_distribution' in config:
        print(f"{C_BOLD}Size Distribution:{C_RESET}")
        total = 0.0
        for size, percentage in config['size_distribution'].items():
            print(f"  • {size}: {percentage*100:.1f}%")
            total += percentage
        print(f"  • Total: {total*100:.1f}%")
        print()
    
    print(f"{C_CYAN}{'='*80}{C_RESET}\n")
    return True



def main():
    """Main training entry point"""
    
    # Load configuration from config.py
    config = cfg.get_config()
    
    # Override paths from config if they exist
    DATA_ROOT = config.get('DATA_ROOT', "/mnt/data/training/Universal/Mastermodell/Learn")
    DATASET_ROOT = config.get('DATASET_ROOT', "/mnt/data/training/Dataset/Universal/Mastermodell")
    
    # Load runtime_config early to get dataset-specific paths
    runtime_config_path = os.path.join(os.path.dirname(__file__), "runtime_config.json")
    rt_config = None
    dataset_name = 'master'  # Default
    
    if os.path.exists(runtime_config_path):
        try:
            with open(runtime_config_path, 'r') as f:
                rt_config = json.load(f)
            data_config = rt_config.get('data', {})
            DATASET_ROOT = data_config.get('root', DATASET_ROOT)
            dataset_name = data_config.get('dataset_name', 'master')
        except Exception as e:
            print(f"{C_YELLOW}⚠ Failed to load runtime_config.json: {e}{C_RESET}")
    
    # Use dataset-specific paths for checkpoints and logs
    # E.g., /mnt/data/training/datasetNeu/master/ (not DATA_ROOT)
    DATASET_SPECIFIC_ROOT = os.path.join(DATASET_ROOT, dataset_name)
    
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
        
        # Get all checkpoints
        all_checkpoints = checkpoint_mgr.list_checkpoints()
        
        if not all_checkpoints:
            print("⚠️  No checkpoint found, starting fresh")
        else:
            # Show detailed checkpoint selection menu
            print("=" * 100)
            print("AVAILABLE CHECKPOINTS (Last 10):")
            print("=" * 100)
            print(f"{'#':<4} {'Step':<12} {'Type':<12} {'Quality':<12} {'Loss':<10} {'Date':<18}")
            print("-" * 100)
            
            # Show last 10 checkpoints
            recent_checkpoints = all_checkpoints[-10:]
            for idx, ckpt in enumerate(recent_checkpoints, 1):
                step_display = f"{ckpt['step']:,}"
                type_display = ckpt['type']
                quality_display = f"{ckpt['quality']*100:.1f}%"
                loss_display = f"{ckpt['loss']:.4f}"
                date_display = ckpt['date_str']
                
                print(f"{idx:<4} {step_display:<12} {type_display:<12} {quality_display:<12} {loss_display:<10} {date_display:<18}")
            
            print("=" * 100)
            
            # User selection
            selection = input(f"\n{C_CYAN}Welchen Checkpoint laden? (Nummer 1-{len(recent_checkpoints)} oder Enter für neuesten): {C_RESET}").strip()
            
            if selection == "":
                # Use latest (last in list)
                selected_ckpt = all_checkpoints[-1]
                start_step = selected_ckpt['step']
                selected_checkpoint_path = selected_ckpt['path']
                print(f"{C_GREEN}✅ Using latest checkpoint: Step {start_step:,}{C_RESET}")
            else:
                try:
                    choice_idx = int(selection)
                    if 1 <= choice_idx <= len(recent_checkpoints):
                        selected_ckpt = recent_checkpoints[choice_idx - 1]
                        start_step = selected_ckpt['step']
                        selected_checkpoint_path = selected_ckpt['path']
                        print(f"{C_GREEN}✅ Selected checkpoint: Step {start_step:,} ({selected_ckpt['type']}){C_RESET}")
                    else:
                        print(f"{C_YELLOW}Invalid selection, using latest checkpoint{C_RESET}")
                        selected_ckpt = all_checkpoints[-1]
                        start_step = selected_ckpt['step']
                        selected_checkpoint_path = selected_ckpt['path']
                except ValueError:
                    print(f"{C_YELLOW}Invalid input, using latest checkpoint{C_RESET}")
                    selected_ckpt = all_checkpoints[-1]
                    start_step = selected_ckpt['step']
                    selected_checkpoint_path = selected_ckpt['path']
            
            print()
    
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
    batch_size = config['BATCH_SIZE']
    accumulation_steps = config['ACCUMULATION_STEPS']
    
    # Create model
    print("Creating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VSRBidirectional_3x(
        n_feats=n_feats, 
        n_blocks=n_blocks,
        use_checkpointing=config.get('USE_GRADIENT_CHECKPOINTING', True)
    ).to(device)
    
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
    # Give Final Fusion layer 10x higher learning rate to activate it
    lr = 10 ** config['LR_EXPONENT']
    
    # Separate Final Fusion parameters from other parameters
    final_fusion_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'fusion.conv' in name:  # Final fusion layer (TrackedConv2d wraps the conv)
            final_fusion_params.append(param)
        else:
            other_params.append(param)
    
    # Create parameter groups with different learning rates
    param_groups = [
        {
            'params': other_params,
            'lr': lr,
            'weight_decay': config['WEIGHT_DECAY']
        },
        {
            'params': final_fusion_params,
            'lr': lr * 10,  # 10x higher for Final Fusion
            'weight_decay': config['WEIGHT_DECAY'] * 0.5  # Less weight decay for aggressive learning
        }
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
    
    # Create datasets
    print("Loading datasets...")
    
    # Check for runtime_config.json to enable multi-size training
    # Look in vsr_plusplus_NEU directory first, then in DATA_ROOT
    runtime_config_path = os.path.join(os.path.dirname(__file__), "runtime_config.json")
    if not os.path.exists(runtime_config_path):
        runtime_config_path = os.path.join(DATA_ROOT, "runtime_config.json")
    
    use_multi_size = False
    rt_config = None
    
    if os.path.exists(runtime_config_path):
        try:
            with open(runtime_config_path, 'r') as f:
                rt_config = json.load(f)
            
            # New runtime_config.json structure (no longer uses size_distribution):
            # {
            #   "data": {"root": "...", "dataset_name": "master"},
            #   "training": {"adaptive_batch": {"540": {"batch": 1, "accum": 6}, ...}}
            # }
            
            # Auto-detect which sizes are available by checking for directories
            data_config = rt_config.get('data', {})
            data_root = data_config.get('root', DATASET_ROOT)
            dataset_name = data_config.get('dataset_name', 'master')
            
            # Get configurable paths (with defaults)
            paths_config = data_config.get('paths', {})
            train_gt_pattern = paths_config.get('train_gt', 'patches/{size_key}/GT')
            
            print(f"{C_CYAN}Checking for dataset sizes in: {os.path.join(data_root, dataset_name)}{C_RESET}")
            print(f"{C_CYAN}  Using path pattern: {train_gt_pattern}{C_RESET}")
            
            # Check which size directories exist and have files
            available_sizes = []
            for size_key in ['540', '720', '720_169']:
                # Build path using pattern
                train_path = train_gt_pattern.replace('{size_key}', size_key)
                train_dir = os.path.join(data_root, dataset_name, train_path)
                print(f"{C_CYAN}  Checking {size_key}: {train_dir}{C_RESET}")
                
                if os.path.exists(train_dir):
                    # Check for both .png and .PNG files (case-insensitive)
                    files = [f for f in os.listdir(train_dir) if f.lower().endswith('.png')]
                    if files:
                        available_sizes.append(size_key)
                        print(f"{C_GREEN}    ✓ Found {len(files)} files for size {size_key}{C_RESET}")
                    else:
                        print(f"{C_YELLOW}    ⚠ Directory exists but no .png files found{C_RESET}")
                else:
                    print(f"{C_YELLOW}    ⚠ Directory does not exist{C_RESET}")
            
            if len(available_sizes) > 1:
                use_multi_size = True
                print(f"{C_CYAN}✓ Multi-size training enabled: {', '.join(available_sizes)}{C_RESET}")
            elif len(available_sizes) == 1:
                print(f"{C_CYAN}✓ Single-size training: {available_sizes[0]}{C_RESET}")
            else:
                print(f"{C_YELLOW}⚠ No training data found, falling back to defaults{C_RESET}")
        except Exception as e:
            print(f"{C_YELLOW}⚠ Failed to load runtime_config.json: {e}{C_RESET}")
            print(f"{C_YELLOW}Using single-size training{C_RESET}")
    
    # Helper function to auto-detect available sizes
    def detect_available_sizes(data_root, dataset_name, train_gt_pattern='patches/{size_key}/GT'):
        """Detect which dataset sizes are available by checking directories.
        
        Args:
            data_root: Root directory
            dataset_name: Dataset name (e.g., 'master')
            train_gt_pattern: Path pattern with {size_key} placeholder
        """
        available = []
        print(f"{C_CYAN}Detecting available sizes in: {os.path.join(data_root, dataset_name)}{C_RESET}")
        print(f"{C_CYAN}  Using path pattern: {train_gt_pattern}{C_RESET}")
        
        for size_key in ['540', '720', '720_169']:
            # Build path using pattern
            train_path = train_gt_pattern.replace('{size_key}', size_key)
            train_dir = os.path.join(data_root, dataset_name, train_path)
            print(f"{C_CYAN}  Checking {size_key}: {train_dir}{C_RESET}")
            
            if os.path.exists(train_dir):
                # Check for both .png and .PNG files (case-insensitive)
                files = [f for f in os.listdir(train_dir) if f.lower().endswith('.png')]
                if files:
                    available.append((size_key, len(files)))
                    print(f"{C_GREEN}    ✓ Found {len(files)} files{C_RESET}")
                else:
                    print(f"{C_YELLOW}    ⚠ Directory exists but no .png files found{C_RESET}")
            else:
                print(f"{C_YELLOW}    ⚠ Directory does not exist{C_RESET}")
        
        return available
    
    if use_multi_size:
        # Use multi-size dataloader
        try:
            from vsr_plusplus_NEU.core.dataloader import create_train_loader
            
            # Extract data from new runtime_config.json structure
            data_config = rt_config.get('data', {})
            data_root = data_config.get('root', DATASET_ROOT)
            dataset_name = data_config.get('dataset_name', 'master')
            adaptive_batch = rt_config.get('training', {}).get('adaptive_batch', {})
            
            # Auto-detect available sizes and create config
            sizes_config = {}
            paths_config = data_config.get('paths', {})
            train_gt_pattern = paths_config.get('train_gt', 'patches/{size_key}/GT')
            available = detect_available_sizes(data_root, dataset_name, train_gt_pattern)
            
            for size_key, file_count in available:
                batch_info = adaptive_batch.get(size_key, {})
                sizes_config[size_key] = {
                    'enabled': True,  # All detected sizes are enabled
                    'distribution': 1.0 / len(available),  # Equal distribution (not used for weighting)
                    'batch_size': batch_info.get('batch', 1)
                }
            
            # Prepare config for multi-size loader
            loader_config = {
                'data_root': data_root,
                'dataset_name': dataset_name,
                'sizes': sizes_config,
                'augment': True,
                'shuffle': True,
                'paths': paths_config  # NEW: Pass paths config
            }
            
            train_loader = create_train_loader(loader_config)
            
            # Count total samples across all sizes
            total_samples = sum(len(ds) for ds in train_loader.datasets_dict.values())
            print(f"✅ Multi-size training samples: {total_samples:,}")
            print(f"{C_CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{C_RESET}")
            print(f"{C_CYAN}📊 Dataset Sizes Loaded at Startup:{C_RESET}")
            for size_key, dataset in train_loader.datasets_dict.items():
                # Calculate actual distribution from file counts
                dist = len(dataset) / total_samples if total_samples > 0 else 0.0
                print(f"  • {size_key}: {len(dataset):,} samples ({dist*100:.1f}%)")
            print(f"{C_CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{C_RESET}")
            print(f"{C_YELLOW}⚠️  To change dataset sizes, modify runtime_config.json and RESTART training{C_RESET}")
            print()
        except Exception as e:
            import traceback
            print(f"{C_RED}❌ Error creating multi-size dataloader: {e}{C_RESET}")
            traceback.print_exc()
            print(f"{C_YELLOW}Falling back to single-size training{C_RESET}")
            use_multi_size = False
    
    if not use_multi_size:
        # Use traditional single-size dataloader - get config from runtime_config if available
        if rt_config:
            data_config = rt_config.get('data', {})
            data_root = data_config.get('root', DATASET_ROOT)
            dataset_name = data_config.get('dataset_name', 'master')
            paths_config = data_config.get('paths', None)  # NEW: Get paths config
            # Auto-detect first available size
            train_gt_pattern = paths_config.get('train_gt', 'patches/{size_key}/GT') if paths_config else 'patches/{size_key}/GT'
            available = detect_available_sizes(data_root, dataset_name, train_gt_pattern)
            size_key = available[0][0] if available else '540'
        else:
            # Fallback to defaults
            data_root = DATASET_ROOT
            dataset_name = 'master'
            size_key = '540'
            paths_config = None
        
        try:
            train_dataset = VSRDataset(
                root=data_root,
                dataset_name=dataset_name,
                size_key=size_key,
                mode='train',
                augment=True,
                paths_config=paths_config  # NEW: Pass paths config
            )
            
            print(f"✅ Training samples: {len(train_dataset):,}\n")
        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
            return
        
        # Create single-size data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config['NUM_WORKERS'],
            pin_memory=config['PIN_MEMORY']
        )
    
    # Load validation datasets - use ALL configured sizes from runtime_config
    val_loaders = []  # List of (size_key, loader) tuples
    
    if rt_config:
        data_config = rt_config.get('data', {})
        data_root = data_config.get('root', DATASET_ROOT)
        dataset_name = data_config.get('dataset_name', 'master')
        paths_config = data_config.get('paths', None)  # NEW: Get paths config
        # Get validation sizes from config
        val_sizes = rt_config.get('validation', {}).get('sizes', [])
        if not val_sizes:
            # Fallback to auto-detected training sizes
            train_gt_pattern = paths_config.get('train_gt', 'patches/{size_key}/GT') if paths_config else 'patches/{size_key}/GT'
            available = detect_available_sizes(data_root, dataset_name, train_gt_pattern)
            val_sizes = [size_key for size_key, _ in available] if available else ['540']
    else:
        # Fallback to defaults
        data_root = DATASET_ROOT
        dataset_name = 'master'
        val_sizes = ['540']
        paths_config = None
    
    # Create validation dataset and loader for EACH size
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
                paths_config=paths_config  # NEW: Pass paths config
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.get('VAL_BATCH_SIZE', 1),
                shuffle=False,
                num_workers=2,
                pin_memory=False  # Disable for validation (saves VRAM)
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
    
    # For backward compatibility, use first loader as primary val_loader
    # (trainer will iterate through all in val_loaders list)
    val_loader = val_loaders[0][1]  # Primary loader for validator initialization
    
    # Create checkpoint manager (use dataset-specific root)
    checkpoint_mgr = CheckpointManager(DATASET_SPECIFIC_ROOT)
    
    # Create runtime config manager (reuse runtime_config_path defined earlier)
    runtime_config = RuntimeConfigManager(
        config_path=runtime_config_path,
        base_config=config
    )
    
    # Create loggers (use dataset-specific paths)
    log_dir = os.path.join(DATASET_SPECIFIC_ROOT, "logs")
    train_logger = TrainingLogger(DATASET_SPECIFIC_ROOT)
    tb_logger = TensorBoardLogger(log_dir)
    
    # Create validator
    validator = VSRValidator(model, val_loader, loss_fn, device=device)
    
    # Load checkpoint if resuming
    if start_step > 0 and selected_checkpoint_path:
        print(f"Loading checkpoint from {selected_checkpoint_path}...")
        checkpoint = torch.load(selected_checkpoint_path, map_location=device)
        
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
        runtime_config=runtime_config
    )
    
    # Pass all validation loaders to trainer for multi-size validation
    trainer.val_loaders = val_loaders
    
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
