"""
VSRValidator - Validation logic for VSR training

Validates model on validation set and computes quality metrics
"""

import sys
import time
import torch
import torch.nn.functional as F
from vsr_plus_plus.utils.metrics import calculate_psnr, calculate_ssim, quality_to_percent
from vsr_plus_plus.utils.ui_terminal import C_GREEN, C_GRAY, C_CYAN, C_RESET


class VSRValidator:
    """
    Validation logic for VSR training
    
    Args:
        model: VSR model
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to run on
    """
    
    def __init__(self, model, val_loader, loss_fn, device='cuda'):
        self.model = model
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.device = device
    
    def validate(self, global_step):
        """
        Run validation
        
        Args:
            global_step: Current training step
            
        Returns:
            Dict with validation metrics:
            {
                'val_loss': float,
                'lr_quality': float (0-1),
                'ki_quality': float (0-1),
                'improvement': float (ki - lr),
                'lr_psnr': float,
                'lr_ssim': float,
                'ki_psnr': float,
                'ki_ssim': float
            }
        """
        self.model.eval()
        
        total_loss = 0.0
        total_lr_psnr = 0.0
        total_lr_ssim = 0.0
        total_ki_psnr = 0.0
        total_ki_ssim = 0.0
        
        num_samples = 0
        
        # For image logging (save first batch)
        first_lr = None
        first_ki = None
        first_gt = None
        
        val_total = len(self.val_loader)
        val_start = time.time()
        
        with torch.no_grad():
            for batch_idx, (lr_stack, gt) in enumerate(self.val_loader):
                # Progress Bar (like original)
                progress = (batch_idx + 1) / val_total * 100
                filled = int(50 * (batch_idx + 1) / val_total)
                bar = f"{C_GREEN}{'█' * filled}{C_GRAY}{'░' * (50 - filled)}{C_RESET}"
                eta = ((time.time() - val_start) / (batch_idx + 1)) * (val_total - batch_idx - 1) if batch_idx > 0 else 0
                sys.stdout.write(f"\r{C_CYAN}Progress:{C_RESET} [{bar}] {batch_idx+1}/{val_total} ({progress:.1f}%) | ETA: {eta:.1f}s")
                sys.stdout.flush()
                lr_stack = lr_stack.to(self.device)
                gt = gt.to(self.device)
                
                # Forward pass
                ki_output = self.model(lr_stack)
                
                # Compute loss
                loss_dict = self.loss_fn(ki_output, gt)
                total_loss += loss_dict['total'].item() if torch.is_tensor(loss_dict['total']) else loss_dict['total']
                
                # Get LR center frame (upscaled for comparison)
                lr_center = lr_stack[:, 2]  # Center frame
                lr_upscaled = F.interpolate(lr_center, scale_factor=3, mode='bilinear', align_corners=False)
                
                # Compute metrics for each sample in batch
                for i in range(lr_stack.size(0)):
                    # LR metrics
                    lr_psnr = calculate_psnr(lr_upscaled[i], gt[i])
                    lr_ssim = calculate_ssim(lr_upscaled[i], gt[i])
                    
                    # KI metrics
                    ki_psnr = calculate_psnr(ki_output[i], gt[i])
                    ki_ssim = calculate_ssim(ki_output[i], gt[i])
                    
                    total_lr_psnr += lr_psnr
                    total_lr_ssim += lr_ssim
                    total_ki_psnr += ki_psnr
                    total_ki_ssim += ki_ssim
                    
                    num_samples += 1
                
                # Save first batch for visualization
                if batch_idx == 0:
                    first_lr = lr_upscaled[0].cpu()
                    first_ki = ki_output[0].cpu()
                    first_gt = gt[0].cpu()
                
                # Limit validation samples
                if num_samples >= 100:
                    break
        
        # Clear progress line
        print()  # New line after progress bar
        
        self.model.train()
        
        # Compute averages
        avg_loss = total_loss / max(1, len(self.val_loader))
        avg_lr_psnr = total_lr_psnr / max(1, num_samples)
        avg_lr_ssim = total_lr_ssim / max(1, num_samples)
        avg_ki_psnr = total_ki_psnr / max(1, num_samples)
        avg_ki_ssim = total_ki_ssim / max(1, num_samples)
        
        # Compute quality scores
        lr_quality = quality_to_percent(avg_lr_psnr, avg_lr_ssim)
        ki_quality = quality_to_percent(avg_ki_psnr, avg_ki_ssim)
        improvement = ki_quality - lr_quality
        
        return {
            'val_loss': avg_loss,
            'lr_quality': lr_quality,
            'ki_quality': ki_quality,
            'improvement': improvement,
            'lr_psnr': avg_lr_psnr,
            'lr_ssim': avg_lr_ssim,
            'ki_psnr': avg_ki_psnr,
            'ki_ssim': avg_ki_ssim,
            'sample_lr': first_lr,
            'sample_ki': first_ki,
            'sample_gt': first_gt
        }
