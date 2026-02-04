#!/usr/bin/env python3
"""
VSR++ Training Entry Point

Orchestrates the complete training system:
- Auto-tuning (optional)
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
from vsr_plus_plus.systems.auto_tune import auto_tune_config
from vsr_plus_plus.systems.checkpoint_manager import CheckpointManager
from vsr_plus_plus.systems.logger import TrainingLogger, TensorBoardLogger
from vsr_plus_plus.systems.adaptive_system import AdaptiveSystem
from vsr_plus_plus.utils.config import load_config, save_config, get_default_config


def main():
    """Main training entry point"""
    
    # Paths
    DATA_ROOT = "/mnt/data/training/Universal/Mastermodell/Learn"
    DATASET_ROOT = "/mnt/data/training/Dataset/Universal/Mastermodell"
    CONFIG_FILE = os.path.join(DATA_ROOT, "train_config.json")
    
    print("\n" + "="*80)
    print("VSR++ Training System")
    print("="*80 + "\n")
    
    # User choice: DELETE or RESUME
    choice = input("⚠️  [L]öschen oder [F]ortsetzen? (L/F): ").lower()
    
    config = None
    start_step = 0
    
    if choice == 'l':
        print("\n🔧 Starting fresh training with auto-tuning...\n")
        
        # Run auto-tune
        print("Running auto-tune to find optimal configuration...")
        model_config = auto_tune_config(
            target_speed=4.0,
            max_vram_gb=6.0,
            min_effective_batch=4
        )
        
        # Create config
        config = get_default_config()
        config['AUTO_TUNED'] = True
        config['MODEL_CONFIG'] = model_config
        config['ACCUMULATION_STEPS'] = model_config['accumulation_steps']
        
        # Save config
        save_config(config, CONFIG_FILE)
        print(f"\n✅ Configuration saved to {CONFIG_FILE}\n")
        
        # Cleanup old checkpoints
        checkpoint_mgr = CheckpointManager(DATA_ROOT)
        checkpoint_mgr.cleanup_old_checkpoints()
        
    else:
        print("\n📂 Resuming training...\n")
        
        # Load config
        config = load_config(CONFIG_FILE)
        
        # Show checkpoint info
        checkpoint_mgr = CheckpointManager(DATA_ROOT)
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
    
    # Extract model config
    model_config = config.get('MODEL_CONFIG', {})
    n_feats = model_config.get('n_feats', 128)
    n_blocks = model_config.get('n_blocks', 32)
    batch_size = model_config.get('batch_size', 4)
    accumulation_steps = config.get('ACCUMULATION_STEPS', 1)
    
    print("\n" + "="*80)
    print("Model Configuration:")
    print("="*80)
    print(f"  Features: {n_feats}")
    print(f"  Blocks: {n_blocks}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Accumulation Steps: {accumulation_steps}")
    print(f"  Effective Batch: {batch_size * accumulation_steps}")
    print("="*80 + "\n")
    
    # Create model
    print("Creating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VSRBidirectional_3x(n_feats=n_feats, n_blocks=n_blocks).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {total_params/1e6:.2f}M parameters\n")
    
    # Create loss function
    loss_fn = HybridLoss(l1_weight=0.6, ms_weight=0.2, grad_weight=0.2)
    
    # Create optimizer
    lr = 10 ** config.get('LR_EXPONENT', -5)  # 1e-5 default
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config.get('WEIGHT_DECAY', 0.001)
    )
    
    # Create LR scheduler
    lr_scheduler = AdaptiveLRScheduler(
        optimizer,
        warmup_steps=config.get('WARMUP_STEPS', 1000),
        max_steps=config.get('MAX_STEPS', 100000),
        max_lr=1e-4,
        min_lr=1e-6
    )
    
    # Create adaptive system
    adaptive_system = AdaptiveSystem(
        initial_l1=0.6,
        initial_ms=0.2,
        initial_grad=0.2
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
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True
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
