"""
HybridLoss - Multi-component loss function

Combines:
- L1 loss (pixel-wise difference)
- Multi-scale loss (downsampled comparison)
- Gradient loss (spatial gradients)
- Perceptual loss (VGG-based feature matching with ImageNet weights)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss using ImageNet pretrained weights
    
    - Uses VGG16 features from multiple layers
    - Pretrained on ImageNet for robust feature extraction
    - Frozen weights (no training) for stable gradients
    - Provides real perceptual feedback for sharpness
    """
    
    def __init__(self):
        super().__init__()
        # Load VGG16 with ImageNet weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
        # Extract feature layers (relu1_2, relu2_2, relu3_3, relu4_3)
        self.features = nn.ModuleList([
            vgg.features[:4],   # relu1_2
            vgg.features[:9],   # relu2_2
            vgg.features[:16],  # relu3_3
            vgg.features[:23],  # relu4_3
        ])
        
        # Freeze all VGG parameters
        for feature_layer in self.features:
            for param in feature_layer.parameters():
                param.requires_grad = False
        
        # Set to eval mode
        self.eval()
        
        # VGG normalization constants (ImageNet)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        """Normalize input to VGG's expected range"""
        return (x - self.mean) / self.std
    
    def forward(self, pred, target):
        """
        Compute perceptual loss
        
        Args:
            pred: Predicted image [B, 3, H, W] in range [0, 1]
            target: Target image [B, 3, H, W] in range [0, 1]
            
        Returns:
            Perceptual loss (scalar)
        """
        # Normalize inputs
        pred = self.normalize(pred)
        target = self.normalize(target)
        
        # Compute L1 loss at each feature layer
        loss = 0.0
        for feature_layer in self.features:
            pred_feat = feature_layer(pred)
            target_feat = feature_layer(target)
            loss += F.l1_loss(pred_feat, target_feat)
        
        # Average over layers
        return loss / len(self.features)


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
            self.perceptual_loss = PerceptualLoss()
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
