"""
VSRValidator - Validation logic for VSR training

Validates model on validation set and computes quality metrics
"""

import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
import cv2
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
        total_improvement = 0.0  # Sum of per-image (KI - LR) improvements
        total_ki_to_gt = 0.0  # Sum of per-image (KI - GT) differences
        total_lr_to_gt = 0.0  # Sum of per-image (LR - GT) differences
        
        num_samples = 0
        
        # For image logging - process images immediately to save memory
        # Only store final labeled images, not intermediate lr/ki/gt separately
        labeled_images = []
        
        val_total = len(self.val_loader)
        val_start = time.time()
        
        with torch.no_grad():
            for batch_idx, (lr_stack, gt) in enumerate(self.val_loader):
                # Update sample count BEFORE displaying
                num_samples += lr_stack.size(0)
                
                # Progress Bar - show batches AND cumulative samples for clarity
                progress = (batch_idx + 1) / val_total * 100
                filled = int(30 * (batch_idx + 1) / val_total)
                bar = f"{C_GREEN}{'█' * filled}{C_GRAY}{'░' * (30 - filled)}{C_RESET}"
                
                # Calculate ETA more robustly
                if batch_idx > 0:
                    elapsed = time.time() - val_start
                    avg_time_per_batch = elapsed / (batch_idx + 1)
                    remaining_batches = val_total - (batch_idx + 1)
                    eta = avg_time_per_batch * remaining_batches
                else:
                    eta = 0
                
                # Show "Batch X/Y (N samples)" with percentage
                sys.stdout.write(f"\r{C_CYAN}Progress:{C_RESET} [{bar}] {progress:.1f}% | Batch {batch_idx+1}/{val_total} ({num_samples} samples) | ETA: {eta:.1f}s")
                sys.stdout.flush()
                lr_stack = lr_stack.to(self.device)
                gt = gt.to(self.device)
                
                # Forward pass
                ki_output = self.model(lr_stack)
                
                # Compute loss and immediately extract scalar (don't keep GPU tensor)
                loss_dict = self.loss_fn(ki_output, gt)
                total_loss += loss_dict['total'].item() if torch.is_tensor(loss_dict['total']) else loss_dict['total']
                del loss_dict  # Free loss tensors immediately
                
                # Get LR center frame (upscaled for comparison)
                lr_center = lr_stack[:, 2]  # Center frame
                lr_upscaled = F.interpolate(lr_center, scale_factor=3, mode='bilinear', align_corners=False)
                del lr_center  # Free immediately after use
                
                # Compute metrics for each sample in batch
                for i in range(lr_stack.size(0)):
                    # LR metrics - compute on GPU, extract scalar immediately
                    lr_psnr = calculate_psnr(lr_upscaled[i], gt[i])
                    lr_ssim = calculate_ssim(lr_upscaled[i], gt[i])
                    
                    # KI metrics - compute on GPU, extract scalar immediately
                    ki_psnr = calculate_psnr(ki_output[i], gt[i])
                    ki_ssim = calculate_ssim(ki_output[i], gt[i])
                    
                    total_lr_psnr += lr_psnr
                    total_lr_ssim += lr_ssim
                    total_ki_psnr += ki_psnr
                    total_ki_ssim += ki_ssim
                    
                    # Calculate quality percentages
                    lr_qual = quality_to_percent(lr_psnr, lr_ssim)
                    ki_qual = quality_to_percent(ki_psnr, ki_ssim)
                    gt_qual = 1.0  # GT is always 100% quality
                    
                    # Add per-image improvement (KI - LR)
                    total_improvement += (ki_qual - lr_qual)
                    
                    # Add per-image differences to GT
                    total_ki_to_gt += (ki_qual - gt_qual)
                    total_lr_to_gt += (lr_qual - gt_qual)
                    
                    # GPU MEMORY OPTIMIZATION: Move to CPU IMMEDIATELY after metrics computed
                    # Don't keep GPU tensors around - they take up valuable VRAM
                    lr_img = lr_upscaled[i].cpu().permute(1, 2, 0).numpy()
                    ki_img = ki_output[i].cpu().permute(1, 2, 0).numpy()
                    gt_img = gt[i].cpu().permute(1, 2, 0).numpy()
                    
                    # Clip and convert to 0-255 AND copy
                    # .copy() is CRITICAL - cv2.putText modifies in-place!
                    lr_img = np.clip(lr_img * 255, 0, 255).astype(np.uint8).copy()
                    ki_img = np.clip(ki_img * 255, 0, 255).astype(np.uint8).copy()
                    gt_img = np.clip(gt_img * 255, 0, 255).astype(np.uint8).copy()
                    
                    # Add text labels
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.5
                    thickness = 3
                    
                    # LR label
                    text = f"LR {lr_qual*100:.1f}%"
                    cv2.putText(lr_img, text, (10, 40), font, font_scale, (255, 255, 255), thickness)
                    cv2.putText(lr_img, text, (10, 40), font, font_scale, (0, 255, 0), thickness-1)
                    
                    # KI label
                    text = f"KI {ki_qual*100:.1f}%"
                    cv2.putText(ki_img, text, (10, 40), font, font_scale, (255, 255, 255), thickness)
                    cv2.putText(ki_img, text, (10, 40), font, font_scale, (0, 255, 255), thickness-1)
                    
                    # GT label
                    text = "GT 100.0%"
                    cv2.putText(gt_img, text, (10, 40), font, font_scale, (255, 255, 255), thickness)
                    cv2.putText(gt_img, text, (10, 40), font, font_scale, (255, 0, 0), thickness-1)
                    
                    # Add 3-pixel black borders between images
                    border_width = 3
                    border_color = (0, 0, 0)  # Black
                    
                    # Add border to LR image (right side)
                    lr_bordered = cv2.copyMakeBorder(lr_img, 0, 0, 0, border_width,
                                                      cv2.BORDER_CONSTANT, value=border_color)
                    # Add border to KI image (right side)
                    ki_bordered = cv2.copyMakeBorder(ki_img, 0, 0, 0, border_width,
                                                      cv2.BORDER_CONSTANT, value=border_color)
                    # No border on GT (last image)
                    
                    # Concatenate side by side: LR | KI | GT
                    combined = np.concatenate([lr_bordered, ki_bordered, gt_img], axis=1)
                    
                    # Convert back to tensor (CHW format for TensorBoard)
                    combined_tensor = torch.from_numpy(combined).permute(2, 0, 1)
                    combined_tensor = combined_tensor.float() / 255.0
                    combined_tensor = combined_tensor.contiguous()
                    
                    # Store only the final labeled image
                    labeled_images.append(combined_tensor)
                
                # GPU MEMORY CRITICAL: Free GPU tensors IMMEDIATELY after batch processing
                # This is the key to reducing VRAM usage during validation
                del lr_stack, gt, ki_output, lr_upscaled
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Force GPU to release memory NOW
        
        # Clear progress line
        print()  # New line after progress bar
        
        # CLEANUP - Force release GPU memory
        torch.cuda.empty_cache()
        
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
        
        # Use SUM of per-image improvements (not average)
        # This shows total improvement across all validation images
        improvement = total_improvement
        ki_to_gt = total_ki_to_gt
        lr_to_gt = total_lr_to_gt
        
        return {
            'val_loss': avg_loss,
            'lr_quality': lr_quality,
            'ki_quality': ki_quality,
            'improvement': improvement,
            'ki_to_gt': ki_to_gt,  # Total difference KI to GT
            'lr_to_gt': lr_to_gt,  # Total difference LR to GT
            'lr_psnr': avg_lr_psnr,
            'lr_ssim': avg_lr_ssim,
            'ki_psnr': avg_ki_psnr,
            'ki_ssim': avg_ki_ssim,
            'labeled_images': labeled_images  # Already labeled and ready for TensorBoard
        }
