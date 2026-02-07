#!/usr/bin/env python3
"""
VSR++ Unified Training Script

Supports:
- Multi-category datasets (General/Space/Toon)
- Dual LR versions (5 frames vs 7 frames)
- Multi-format training (different patch sizes)
- Gradient accumulation
- Fresh training from scratch
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsr_plus_plus.utils.yaml_config import load_yaml_config, validate_config, print_config
from vsr_plus_plus.data import MultiFormatMultiCategoryDataset, ValidationDataset
from vsr_plus_plus.core.model import VSRBidirectional_3x

# ANSI colors
C_GREEN = "\033[92m"
C_CYAN = "\033[96m"
C_RED = "\033[91m"
C_YELLOW = "\033[93m"
C_BOLD = "\033[1m"
C_RESET = "\033[0m"


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 variant)"""
    
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return torch.mean(loss)


def get_cosine_scheduler(optimizer, total_steps, warmup_steps, min_lr, base_lr):
    """
    Create cosine annealing scheduler with warmup
    
    Args:
        optimizer: PyTorch optimizer
        total_steps: Total training steps
        warmup_steps: Warmup steps
        min_lr: Minimum learning rate (absolute value)
        base_lr: Base learning rate (absolute value)
    
    Returns:
        Scheduler function
    """
    # Convert to fraction for lambda scheduler
    min_lr_fraction = min_lr / base_lr
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, warmup_steps))
        else:
            # Cosine annealing
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            # Scale from 1.0 to min_lr_fraction
            return min_lr_fraction + (1.0 - min_lr_fraction) * cosine_decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def custom_collate_fn(batch):
    """
    Custom collate to handle single samples with format info
    (since we use gradient accumulation instead of large batches)
    """
    return {
        'lr': batch[0]['lr'].unsqueeze(0),
        'gt': batch[0]['gt'].unsqueeze(0),
        'format': batch[0]['format']
    }


def calculate_psnr(pred, target):
    """Calculate PSNR between prediction and target"""
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))


def save_validation_image(pred, gt, step, idx, config):
    """Save validation result image"""
    import cv2
    
    save_dir = os.path.join(config.LOGGING.save_dir, "validation_images", f"step_{step:06d}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy
    pred_np = (pred.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    gt_np = (gt.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)
    gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR)
    
    # Save
    cv2.imwrite(os.path.join(save_dir, f"pred_{idx:03d}.png"), pred_bgr)
    cv2.imwrite(os.path.join(save_dir, f"gt_{idx:03d}.png"), gt_bgr)


def run_validation(model, val_dataset, step, config, device):
    """Run validation and save results"""
    model.eval()
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    psnr_list = []
    
    print(f"\n{C_CYAN}Running validation at step {step:,}...{C_RESET}")
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= config.LOGGING.val_save_images:
                break
            
            lr = batch['lr'].to(device)
            gt = batch['gt'].to(device)
            
            # Inference
            output = model(lr)
            
            # Metrics
            psnr = calculate_psnr(output, gt)
            psnr_list.append(psnr)
            
            # Save image
            save_validation_image(output[0], gt[0], step, i, config)
    
    avg_psnr = np.mean(psnr_list)
    print(f"{C_GREEN}📊 Validation Step {step:,}: PSNR = {avg_psnr:.2f} dB{C_RESET}\n")
    
    model.train()
    return avg_psnr


def save_checkpoint(model, optimizer, scheduler, step, config):
    """Save training checkpoint"""
    checkpoint_dir = os.path.join(config.LOGGING.save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step:06d}.pth")
    
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
        'config': config.to_dict()
    }, checkpoint_path)
    
    print(f"{C_GREEN}✅ Checkpoint saved: {checkpoint_path}{C_RESET}")


def train(config_path):
    """Main training function"""
    
    # Load and validate config
    print(f"\n{C_CYAN}Loading configuration from {config_path}...{C_RESET}")
    config = load_yaml_config(config_path)
    validate_config(config)
    
    print(f"\n{'='*80}")
    print(f"{C_BOLD}VSR++ Unified Training System{C_RESET}")
    print(f"{'='*80}\n")
    print_config(config)
    print(f"\n{'='*80}\n")
    
    # Setup device
    device = torch.device(f'cuda:{config.HARDWARE.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create datasets
    print(f"{C_CYAN}Creating datasets...{C_RESET}")
    train_dataset = MultiFormatMultiCategoryDataset(config)
    val_dataset = ValidationDataset(config)
    
    if len(train_dataset) == 0:
        print(f"{C_RED}Error: No training samples found!{C_RESET}")
        return
    
    # Create weighted sampler for format-based sampling
    format_weights = [
        config.DATA.format_weights[sample['format']] 
        for sample in train_dataset.image_pairs
    ]
    sampler = WeightedRandomSampler(format_weights, len(train_dataset), replacement=True)
    
    # DataLoader - batch_size=1 since we handle batching via gradient accumulation
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=config.HARDWARE.num_workers,
        pin_memory=config.HARDWARE.pin_memory,
        collate_fn=custom_collate_fn
    )
    
    # Determine number of frames from config
    num_frames = 7 if config.DATA.lr_version == "7frames" else 5
    
    # Create model
    print(f"{C_CYAN}Creating model...{C_RESET}")
    model = VSRBidirectional_3x(
        n_feats=config.MODEL.base_channels,
        n_blocks=config.MODEL.num_blocks,
        use_checkpointing=config.MODEL.use_checkpointing,
        num_frames=num_frames
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{C_GREEN}✅ Model created: {total_params/1e6:.2f}M parameters{C_RESET}")
    print(f"   Frames: {num_frames}")
    print(f"   Channels: {config.MODEL.base_channels}")
    print(f"   Blocks: {config.MODEL.num_blocks}\n")
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.TRAINING.learning_rate,
        betas=tuple(config.TRAINING.betas),
        weight_decay=config.TRAINING.weight_decay
    )
    
    # Create scheduler
    scheduler = get_cosine_scheduler(
        optimizer,
        config.TRAINING.total_steps,
        config.TRAINING.warmup_steps,
        config.TRAINING.min_lr,
        config.TRAINING.learning_rate
    )
    
    # Create loss function
    criterion = CharbonnierLoss(epsilon=config.LOSS.epsilon)
    
    # AMP scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=config.HARDWARE.mixed_precision)
    
    # Training state
    step = config.TRAINING.start_step
    accumulation_counter = 0
    
    # Create save directories
    os.makedirs(config.LOGGING.save_dir, exist_ok=True)
    os.makedirs(config.LOGGING.tensorboard_dir, exist_ok=True)
    
    # Training loop
    print(f"{'='*80}")
    print(f"{C_BOLD}🚀 Starting Training{C_RESET}")
    print(f"{'='*80}")
    print(f"Category: {config.DATA.category}")
    print(f"LR Version: {config.DATA.lr_version} ({num_frames} frames)")
    print(f"Total Steps: {config.TRAINING.total_steps:,}")
    print(f"Start Step: {step:,}")
    print(f"{'='*80}\n")
    
    pbar = tqdm(total=config.TRAINING.total_steps - step, desc="Training", unit="step")
    
    model.train()
    epoch = 0
    
    while step < config.TRAINING.total_steps:
        epoch += 1
        
        for batch in train_loader:
            lr = batch['lr'].to(device)
            gt = batch['gt'].to(device)
            format_name = batch['format']
            
            # Get format-specific settings
            accum_steps = config.TRAINING.gradient_accumulation[format_name]
            
            # Forward pass with AMP
            with torch.cuda.amp.autocast(enabled=config.HARDWARE.mixed_precision):
                output = model(lr)
                loss = criterion(output, gt) / accum_steps
            
            # Backward
            scaler.scale(loss).backward()
            
            accumulation_counter += 1
            
            # Update weights after accumulation
            if accumulation_counter >= accum_steps:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
                accumulation_counter = 0
                step += 1
                pbar.update(1)
                
                # Logging
                if step % config.LOGGING.log_frequency == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix({
                        'loss': f'{loss.item() * accum_steps:.4f}',
                        'lr': f'{current_lr:.2e}',
                        'format': format_name
                    })
                
                # Validation
                if step % config.DATA.val_frequency == 0:
                    psnr = run_validation(model, val_dataset, step, config, device)
                
                # Save checkpoint
                if step % config.LOGGING.save_frequency == 0:
                    save_checkpoint(model, optimizer, scheduler, step, config)
                
                if step >= config.TRAINING.total_steps:
                    break
    
    pbar.close()
    
    print(f"\n{'='*80}")
    print(f"{C_GREEN}{C_BOLD}✅ Training Complete!{C_RESET}")
    print(f"{'='*80}\n")
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, scheduler, step, config)


def main():
    parser = argparse.ArgumentParser(description='VSR++ Unified Training')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML config file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"{C_RED}Error: Config file not found: {args.config}{C_RESET}")
        return
    
    try:
        train(args.config)
    except KeyboardInterrupt:
        print(f"\n{C_YELLOW}Training interrupted by user{C_RESET}")
    except Exception as e:
        print(f"\n{C_RED}Error during training: {e}{C_RESET}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
