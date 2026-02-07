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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsr_plus_plus.core.model import VSRBidirectional_3x
from vsr_plus_plus.core.loss import HybridLoss
from vsr_plus_plus.core.dataset import VSRDataset
from vsr_plus_plus.training.trainer import VSRTrainer
from vsr_plus_plus.training.validator import VSRValidator
from vsr_plus_plus.training.lr_scheduler import AdaptiveLRScheduler
from vsr_plus_plus.systems.checkpoint_manager import CheckpointManager
from vsr_plus_plus.systems.logger import TrainingLogger, TensorBoardLogger
from vsr_plus_plus.systems.adaptive_system import AdaptiveSystem
from vsr_plus_plus.systems.runtime_config import RuntimeConfigManager

# Import manual configuration
import vsr_plus_plus.config as cfg

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


def main():
    """Main training entry point"""
    
    # Load configuration from config.py
    config = cfg.get_config()
    
    # Override paths from config if they exist
    DATA_ROOT = config.get('DATA_ROOT', "/mnt/data/training/Universal/Mastermodell/Learn")
    DATASET_ROOT = config.get('DATASET_ROOT', "/mnt/data/training/Dataset/Universal/Mastermodell")
    
    print("\n" + "="*80)
    print("VSR++ Training System - Manual Configuration")
    print("="*80 + "\n")
    
    # Print current configuration
    cfg.print_config()
    
    # User choice: DELETE or RESUME
    choice = input("⚠️  [L]öschen oder [F]ortsetzen? (L/F): ").lower()
    
    start_step = 0
    selected_checkpoint_path = None
    checkpoint_mgr = CheckpointManager(DATA_ROOT)
    
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
            log_dir = os.path.join(DATA_ROOT, "logs")
            backed_up = checkpoint_mgr.cleanup_all_for_fresh_start(log_dir)
            
            if backed_up > 0:
                print(f"{C_GREEN}✓ {backed_up} .pth Dateien als .BAK gesichert{C_RESET}")
            
            print(f"{C_GREEN}✅ All checkpoints, logs, and TensorBoard events cleaned up{C_RESET}\n")
    
    if choice != 'l' or choice == 'f':
        # Resume mode (either selected 'f' or canceled 'l')
        print("\n📂 Resuming training...\n")
        
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
    
    # Start TensorBoard
    log_dir = os.path.join(DATA_ROOT, "logs")
    print(f"\n{C_CYAN}Checking TensorBoard...{C_RESET}")
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
    
    try:
        train_dataset = VSRDataset(DATASET_ROOT, mode='Patches', augment=True)
        val_dataset = VSRDataset(DATASET_ROOT, mode='Val', augment=False)
        
        print(f"✅ Training samples: {len(train_dataset):,}")
        print(f"✅ Validation samples: {len(val_dataset):,}\n")
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['NUM_WORKERS'],
        pin_memory=config['PIN_MEMORY']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('VAL_BATCH_SIZE', 1),
        shuffle=False,
        num_workers=2,
        pin_memory=False  # Disable for validation (saves VRAM)
    )
    
    # Create checkpoint manager
    checkpoint_mgr = CheckpointManager(DATA_ROOT)
    
    # Create runtime config manager
    runtime_config_path = os.path.join(DATA_ROOT, "runtime_config.json")
    runtime_config = RuntimeConfigManager(
        config_path=runtime_config_path,
        base_config=config
    )
    
    # Create loggers
    log_dir = os.path.join(DATA_ROOT, "logs")
    train_logger = TrainingLogger(DATA_ROOT)
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
    
    # Set start step
    trainer.set_start_step(start_step)
    
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
