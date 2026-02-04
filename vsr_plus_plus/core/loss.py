"""
HybridLoss - Multi-component loss function

Combines:
- L1 loss (pixel-wise difference)
- Multi-scale loss (downsampled comparison)
- Gradient loss (spatial gradients)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridLoss(nn.Module):
    """
    Hybrid loss combining L1, multi-scale, and gradient components
    
    Args:
        l1_weight: Weight for L1 loss component
        ms_weight: Weight for multi-scale loss component
        grad_weight: Weight for gradient loss component
    """
    
    def __init__(self, l1_weight=0.6, ms_weight=0.2, grad_weight=0.2):
        super().__init__()
        self.l1_weight = l1_weight
        self.ms_weight = ms_weight
        self.grad_weight = grad_weight
    
    def forward(self, pred, target, l1_w=None, ms_w=None, grad_w=None):
        """
        Compute hybrid loss
        
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
            l1_w: Optional L1 weight override (for adaptive training)
            ms_w: Optional MS weight override
            grad_w: Optional Grad weight override
            
        Returns:
            Dict with 'l1', 'ms', 'grad', and 'total' loss values
        """
        # Use provided weights or defaults
        l1_w = l1_w if l1_w is not None else self.l1_weight
        ms_w = ms_w if ms_w is not None else self.ms_weight
        grad_w = grad_w if grad_w is not None else self.grad_weight
        
        # 1. L1 Loss
        l1_loss = F.l1_loss(pred, target)
        
        # 2. Multi-Scale Loss (downsample 2x and compare)
        pred_down = F.avg_pool2d(pred, kernel_size=2, stride=2)
        target_down = F.avg_pool2d(target, kernel_size=2, stride=2)
        ms_loss = F.l1_loss(pred_down, target_down)
        
        # 3. Gradient Loss (spatial gradients)
        # Horizontal gradients
        pred_grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        target_grad_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        
        # Vertical gradients
        pred_grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        target_grad_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        
        grad_loss = (F.l1_loss(pred_grad_x, target_grad_x) + 
                    F.l1_loss(pred_grad_y, target_grad_y)) / 2
        
        # 4. Weighted combination
        total_loss = l1_w * l1_loss + ms_w * ms_loss + grad_w * grad_loss
        
        return {
            'l1': l1_loss.item(),
            'ms': ms_loss.item(),
            'grad': grad_loss.item(),
            'total': total_loss
        }
