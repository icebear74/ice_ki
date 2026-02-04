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

# Import manual configuration
import vsr_plus_plus.config as cfg


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
    checkpoint_mgr = CheckpointManager(DATA_ROOT)
    
    if choice == 'l':
        print("\n🗑️  Starting fresh training...\n")
        
        # Cleanup old checkpoints
        checkpoint_mgr.cleanup_old_checkpoints()
        print("✅ Old checkpoints cleaned up\n")
        
    else:
        print("\n📂 Resuming training...\n")
        
        # Show checkpoint info
        checkpoint_mgr.show_checkpoint_info()
        
        # Get latest checkpoint
        latest_path, latest_step = checkpoint_mgr.get_latest_checkpoint()
        
        if latest_path:
            print(f"✅ Found checkpoint at step {latest_step:,}")
            resume = input("Resume from this checkpoint? (Y/n): ").lower()
            
            if resume != 'n':
                start_step = latest_step
            else:
                print("Starting fresh (checkpoint will not be loaded)")
        else:
            print("⚠️  No checkpoint found, starting fresh")
    
    # Extract parameters from config
    n_feats = config['N_FEATS']
    n_blocks = config['N_BLOCKS']
    batch_size = config['BATCH_SIZE']
    accumulation_steps = config['ACCUMULATION_STEPS']
    
    # Create model
    print("Creating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VSRBidirectional_3x(n_feats=n_feats, n_blocks=n_blocks).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {total_params/1e6:.2f}M parameters\n")
    
    # Create loss function with configured weights
    loss_fn = HybridLoss(
        l1_weight=config['L1_WEIGHT'],
        ms_weight=config['MS_WEIGHT'],
        grad_weight=config['GRAD_WEIGHT']
    )
    
    # Create optimizer
    lr = 10 ** config['LR_EXPONENT']
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config['WEIGHT_DECAY']
    )
    
    # Create LR scheduler
    lr_scheduler = AdaptiveLRScheduler(
        optimizer,
        warmup_steps=config['WARMUP_STEPS'],
        max_steps=config['MAX_STEPS'],
        max_lr=config['MAX_LR'],
        min_lr=config['MIN_LR']
    )
    
    # Create adaptive system
    if config['ADAPTIVE_LOSS_WEIGHTS'] or config['ADAPTIVE_GRAD_CLIP']:
        adaptive_system = AdaptiveSystem(
            initial_l1=config['L1_WEIGHT'],
            initial_ms=config['MS_WEIGHT'],
            initial_grad=config['GRAD_WEIGHT']
        )
    else:
        # Use fixed weights if adaptive is disabled
        adaptive_system = AdaptiveSystem(
            initial_l1=config['L1_WEIGHT'],
            initial_ms=config['MS_WEIGHT'],
            initial_grad=config['GRAD_WEIGHT']
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
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=config['PIN_MEMORY']
    )
    
    # Create checkpoint manager
    checkpoint_mgr = CheckpointManager(DATA_ROOT)
    
    # Create loggers
    log_dir = os.path.join(DATA_ROOT, "logs")
    train_logger = TrainingLogger(DATA_ROOT)
    tb_logger = TensorBoardLogger(log_dir)
    
    # Create validator
    validator = VSRValidator(model, val_loader, loss_fn, device=device)
    
    # Load checkpoint if resuming
    if start_step > 0:
        latest_path, _ = checkpoint_mgr.get_latest_checkpoint()
        if latest_path:
            print(f"Loading checkpoint from {latest_path}...")
            checkpoint = torch.load(latest_path, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
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
        device=device
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
