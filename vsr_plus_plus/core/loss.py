"""
HybridLoss - Multi-component loss function

Combines:
- L1 loss (pixel-wise difference)
- Multi-scale loss (downsampled comparison)
- Gradient loss (spatial gradients)
- Perceptual loss (Custom self-learned feature matching)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomFeatureExtractor(nn.Module):
    """
    Lightweight CNN for extracting multi-scale features
    NO PRETRAINED WEIGHTS - learns from scratch!
    
    Architecture:
    - 3 stages with progressive downsampling
    - Each stage: 2 conv layers + pooling
    - Feature extraction at multiple scales
    - Designed to be efficient and lightweight
    """
    
    def __init__(self, n_feats=32):
        super().__init__()
        
        # Stage 1: 3 → 32 (full resolution)
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Stage 2: 32 → 64 (1/2 resolution)
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(n_feats, n_feats*2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feats*2, n_feats*2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Stage 3: 64 → 64 (1/4 resolution)
        self.stage3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(n_feats*2, n_feats*2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feats*2, n_feats*2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x):
        """Extract multi-scale features"""
        feat1 = self.stage1(x)      # Full res
        feat2 = self.stage2(feat1)  # 1/2 res
        feat3 = self.stage3(feat2)  # 1/4 res
        return [feat1, feat2, feat3]


class CustomPerceptualLoss(nn.Module):
    """
    Self-learned perceptual loss using custom feature extractor
    
    - Extracts multi-scale features from pred and target
    - Computes L1 distance at each scale
    - Features learn to be discriminative during training
    - NO pretrained knowledge!
    """
    
    def __init__(self, n_feats=32):
        super().__init__()
        self.feature_extractor = CustomFeatureExtractor(n_feats)
    
    def forward(self, pred, target):
        """
        Compute perceptual loss
        
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
            
        Returns:
            Perceptual loss (scalar)
        """
        # Extract features at multiple scales
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        
        # Compute L1 loss at each scale
        loss = 0.0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += F.l1_loss(pred_feat, target_feat)
        
        # Average over scales
        return loss / len(pred_features)


class HybridLoss(nn.Module):
    """
    Hybrid loss combining L1, multi-scale, gradient, and perceptual components
    
    Args:
        l1_weight: Weight for L1 loss component
        ms_weight: Weight for multi-scale loss component
        grad_weight: Weight for gradient loss component
        perceptual_weight: Weight for perceptual loss component (0 to disable)
    """
    
    def __init__(self, l1_weight=0.6, ms_weight=0.2, grad_weight=0.2, perceptual_weight=0.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.ms_weight = ms_weight
        self.grad_weight = grad_weight
        self.perceptual_weight = perceptual_weight
        
        # Create perceptual loss module if weight > 0
        if perceptual_weight > 0:
            self.perceptual_loss = CustomPerceptualLoss(n_feats=32)
        else:
            self.perceptual_loss = None
    
    def forward(self, pred, target, l1_w=None, ms_w=None, grad_w=None, perceptual_w=None):
        """
        Compute hybrid loss
        
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
            l1_w: Optional L1 weight override (for adaptive training)
            ms_w: Optional MS weight override
            grad_w: Optional Grad weight override
            perceptual_w: Optional Perceptual weight override
            
        Returns:
            Dict with 'l1', 'ms', 'grad', 'perceptual', and 'total' loss values
        """
        # Use provided weights or defaults
        l1_w = l1_w if l1_w is not None else self.l1_weight
        ms_w = ms_w if ms_w is not None else self.ms_weight
        grad_w = grad_w if grad_w is not None else self.grad_weight
        perceptual_w = perceptual_w if perceptual_w is not None else self.perceptual_weight
        
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
        
        # 4. Perceptual Loss (if enabled)
        if self.perceptual_loss is not None and perceptual_w > 0:
            perceptual_loss = self.perceptual_loss(pred, target)
        else:
            perceptual_loss = torch.tensor(0.0, device=pred.device)
        
        # 5. Weighted combination
        total_loss = (l1_w * l1_loss + 
                     ms_w * ms_loss + 
                     grad_w * grad_loss + 
                     perceptual_w * perceptual_loss)
        
        return {
            'l1': l1_loss.item(),
            'ms': ms_loss.item(),
            'grad': grad_loss.item(),
            'perceptual': perceptual_loss.item() if isinstance(perceptual_loss, torch.Tensor) else 0.0,
            'total': total_loss
        }
