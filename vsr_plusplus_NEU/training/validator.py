"""
VSRValidator - Validation logic for VSR training

Validates model on validation set and computes quality metrics
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from vsr_plusplus_NEU.utils.metrics import calculate_psnr, calculate_ssim, quality_to_percent
from vsr_plusplus_NEU.utils.ui_terminal import C_GREEN, C_GRAY, C_CYAN, C_RESET


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
    
    def validate(self, global_step, progress_callback=None):
        """
        Run validation

        Args:
            global_step: Current training step
            progress_callback: Optional callable(done, total, size_key) called
                               after each batch so callers can publish live
                               progress to the WebUI.
            
        Returns:
            Dict with validation metrics:
            {
                'val_loss': float,
                'lr_quality': float (0-1, SSIM-based),
                'bicubic_quality': float (0-1, SSIM-based),
                'ki_quality': float (0-1, SSIM-based),
                'improvement': float (avg per-sample ki_quality - lr_quality),
                'ki_to_gt': float (avg per-sample ki_quality - 1.0),
                'lr_to_gt': float (avg per-sample lr_quality - 1.0),
                'lr_psnr': float,
                'lr_ssim': float,
                'bicubic_psnr': float,
                'bicubic_ssim': float,
                'ki_psnr': float,
                'ki_ssim': float
            }
        """
        self.model.eval()
        
        total_loss = 0.0
        total_lr_psnr = 0.0
        total_lr_ssim = 0.0
        total_bicubic_psnr = 0.0
        total_bicubic_ssim = 0.0
        total_ki_psnr = 0.0
        total_ki_ssim = 0.0
        total_improvement = 0.0   # Sum of per-image (KI_quality - LR_quality)
        total_ki_to_gt = 0.0      # Sum of per-image (KI_quality - GT_quality)
        total_lr_to_gt = 0.0      # Sum of per-image (LR_quality - GT_quality)
        
        num_samples = 0
        
        # For image logging - process images immediately to save memory
        # Only store final labeled images, not intermediate lr/ki/gt separately
        labeled_images = []
        
        val_total = len(self.val_loader)
        val_start = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                lr_stack, gt, filenames = batch
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

                # ── Center-frame extraction ──────────────────────────────────
                # lr_stack shape: [B, 7, 3, H_lr, W_lr] — 7 LR frames, RGB.
                # The VSR model uses all 7 frames but produces the upscaled
                # CENTER frame as output (frame index n_frames//2 = 3).
                # LR, bicubic, and SR baselines MUST use the same center frame
                # for a fair comparison against the GT.
                center_idx      = lr_stack.size(1) // 2   # = 3 for 7-frame model
                lr_center       = lr_stack[:, center_idx]  # [B, 3, H_lr, W_lr]
                # Bilinear upscale (matches original pre-model input preprocessing)
                lr_upscaled     = F.interpolate(lr_center, scale_factor=3, mode='bilinear', align_corners=False)
                # Bicubic upscale (clean baseline — always computed)
                bicubic_upscaled = F.interpolate(lr_center, scale_factor=3, mode='bicubic', align_corners=False)
                bicubic_upscaled = torch.clamp(bicubic_upscaled, 0.0, 1.0)
                del lr_center  # Free immediately after use
                
                # Compute metrics for each sample in batch
                for i in range(lr_stack.size(0)):
                    # LR (bilinear) metrics
                    lr_psnr = calculate_psnr(lr_upscaled[i], gt[i])
                    lr_ssim = calculate_ssim(lr_upscaled[i], gt[i])

                    # Bicubic metrics
                    bic_psnr = calculate_psnr(bicubic_upscaled[i], gt[i])
                    bic_ssim = calculate_ssim(bicubic_upscaled[i], gt[i])

                    # KI (VSR) metrics
                    ki_psnr = calculate_psnr(ki_output[i], gt[i])
                    ki_ssim = calculate_ssim(ki_output[i], gt[i])
                    
                    total_lr_psnr      += lr_psnr
                    total_lr_ssim      += lr_ssim
                    total_bicubic_psnr += bic_psnr
                    total_bicubic_ssim += bic_ssim
                    total_ki_psnr      += ki_psnr
                    total_ki_ssim      += ki_ssim
                    
                    # Quality scores (SSIM-based, 0-1)
                    lr_qual      = quality_to_percent(lr_psnr,  lr_ssim)
                    bic_qual     = quality_to_percent(bic_psnr, bic_ssim)
                    ki_qual      = quality_to_percent(ki_psnr,  ki_ssim)
                    gt_qual      = 1.0  # GT is always 100% quality
                    
                    # Per-image deltas (summed, divided later to produce averages)
                    total_improvement += (ki_qual - lr_qual)
                    total_ki_to_gt    += (ki_qual - gt_qual)
                    total_lr_to_gt    += (lr_qual - gt_qual)
                    
                    # ── Build 4-panel comparison: LR | Bicubic | VSR | GT ──
                    # GPU MEMORY OPTIMIZATION: Move to CPU IMMEDIATELY after metrics computed
                    lr_img      = lr_upscaled[i].cpu().permute(1, 2, 0).numpy()
                    bic_img     = bicubic_upscaled[i].cpu().permute(1, 2, 0).numpy()
                    ki_img      = ki_output[i].cpu().permute(1, 2, 0).numpy()
                    gt_img      = gt[i].cpu().permute(1, 2, 0).numpy()
                    
                    # Clip and convert to uint8 (copy() required: cv2 modifies in-place)
                    lr_img  = np.clip(lr_img  * 255, 0, 255).astype(np.uint8).copy()
                    bic_img = np.clip(bic_img * 255, 0, 255).astype(np.uint8).copy()
                    ki_img  = np.clip(ki_img  * 255, 0, 255).astype(np.uint8).copy()
                    gt_img  = np.clip(gt_img  * 255, 0, 255).astype(np.uint8).copy()
                    
                    font       = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.5
                    thickness  = 3

                    def _label(img, text, color_fg):
                        cv2.putText(img, text, (10, 40), font, font_scale, (255, 255, 255), thickness)
                        cv2.putText(img, text, (10, 40), font, font_scale, color_fg, thickness - 1)

                    _label(lr_img,  f"LR {lr_qual*100:.1f}%",      (0, 255, 0))
                    _label(bic_img, f"Bicubic {bic_qual*100:.1f}%", (0, 200, 255))
                    _label(ki_img,  f"VSR {ki_qual*100:.1f}%",     (0, 255, 255))
                    _label(gt_img,  "GT 100.0%",                    (255, 0, 0))
                    
                    # Black separators between panels
                    border_width = 3
                    border_color = (0, 0, 0)
                    lr_b  = cv2.copyMakeBorder(lr_img,  0, 0, 0, border_width, cv2.BORDER_CONSTANT, value=border_color)
                    bic_b = cv2.copyMakeBorder(bic_img, 0, 0, 0, border_width, cv2.BORDER_CONSTANT, value=border_color)
                    ki_b  = cv2.copyMakeBorder(ki_img,  0, 0, 0, border_width, cv2.BORDER_CONSTANT, value=border_color)
                    combined = np.concatenate([lr_b, bic_b, ki_b, gt_img], axis=1)
                    
                    combined_tensor = torch.from_numpy(combined).permute(2, 0, 1).float() / 255.0
                    combined_tensor = combined_tensor.contiguous()
                    
                    # Store only the final labeled image, keyed by filename stem
                    name = os.path.splitext(os.path.basename(filenames[i]))[0]
                    labeled_images.append((name, combined_tensor))
                
                # GPU MEMORY CRITICAL: Free GPU tensors IMMEDIATELY after batch processing
                del lr_stack, gt, ki_output, lr_upscaled, bicubic_upscaled
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Force GPU to release memory NOW

                # Progress callback for live WebUI updates
                if progress_callback is not None:
                    try:
                        progress_callback(batch_idx + 1, val_total)
                    except Exception:
                        pass
        
        # Clear progress line
        print()  # New line after progress bar
        
        # CLEANUP - Force release GPU memory
        torch.cuda.empty_cache()
        
        self.model.train()
        
        # Compute averages
        n = max(1, num_samples)
        avg_loss          = total_loss          / max(1, len(self.val_loader))
        avg_lr_psnr       = total_lr_psnr       / n
        avg_lr_ssim       = total_lr_ssim       / n
        avg_bicubic_psnr  = total_bicubic_psnr  / n
        avg_bicubic_ssim  = total_bicubic_ssim  / n
        avg_ki_psnr       = total_ki_psnr       / n
        avg_ki_ssim       = total_ki_ssim       / n
        
        # Compute quality scores (SSIM-based)
        lr_quality      = quality_to_percent(avg_lr_psnr,      avg_lr_ssim)
        bicubic_quality = quality_to_percent(avg_bicubic_psnr, avg_bicubic_ssim)
        ki_quality      = quality_to_percent(avg_ki_psnr,      avg_ki_ssim)
        
        # Per-sample averages (comparable across runs with different dataset sizes)
        improvement = total_improvement / n
        ki_to_gt    = total_ki_to_gt    / n
        lr_to_gt    = total_lr_to_gt    / n
        
        return {
            'val_loss':       avg_loss,
            'lr_quality':     lr_quality,
            'bicubic_quality': bicubic_quality,
            'ki_quality':     ki_quality,
            'improvement':    improvement,
            'ki_to_gt':       ki_to_gt,
            'lr_to_gt':       lr_to_gt,
            'lr_psnr':        avg_lr_psnr,
            'lr_ssim':        avg_lr_ssim,
            'bicubic_psnr':   avg_bicubic_psnr,
            'bicubic_ssim':   avg_bicubic_ssim,
            'ki_psnr':        avg_ki_psnr,
            'ki_ssim':        avg_ki_ssim,
            'labeled_images': labeled_images  # Already labeled and ready for TensorBoard
        }

