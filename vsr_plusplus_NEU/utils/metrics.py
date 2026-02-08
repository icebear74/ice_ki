"""
Metrics - PSNR, SSIM, and Quality Calculation
"""

import torch
import torch.nn.functional as F


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Calculate PSNR between two images (0-1 range)
    
    Args:
        img1: First image tensor [C, H, W] or [B, C, H, W]
        img2: Second image tensor [C, H, W] or [B, C, H, W]
        
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    
    if mse < 1e-10:
        return 50.0  # Perfect match
    
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Calculate SSIM between two images (simplified version, 0-1 range)
    
    Args:
        img1: First image tensor [C, H, W] or [B, C, H, W]
        img2: Second image tensor [C, H, W] or [B, C, H, W]
        
    Returns:
        SSIM value (0-1)
    """
    # Ensure 4D tensor [B, C, H, W]
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Calculate means
    mu1 = F.avg_pool2d(img1, 11, stride=1, padding=5)
    mu2 = F.avg_pool2d(img2, 11, stride=1, padding=5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = F.avg_pool2d(img1 ** 2, 11, stride=1, padding=5) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, 11, stride=1, padding=5) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, 11, stride=1, padding=5) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def quality_to_percent(psnr: float, ssim: float) -> float:
    """
    Convert PSNR/SSIM to quality percentage (0-1)
    
    Args:
        psnr: PSNR value in dB
        ssim: SSIM value (0-1)
        
    Returns:
        Quality score (0-1)
    """
    # Normalize PSNR to 0-1 (assume 20-50 dB range)
    psnr_score = (psnr - 20) / 30
    psnr_score = max(0, min(1, psnr_score))
    
    # Combine PSNR and SSIM (70% PSNR, 30% SSIM)
    combined = 0.7 * psnr_score + 0.3 * ssim
    
    return max(0, min(1, combined))
