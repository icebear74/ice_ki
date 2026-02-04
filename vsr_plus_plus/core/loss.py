"""
HybridLoss - Multi-component loss function

Combines:
- L1 loss (pixel-wise difference)
- Multi-scale loss (downsampled comparison)
- Gradient loss (spatial gradients)
- Perceptual loss (VGG16 feature matching)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PerceptualLoss(nn.Module):
    """
    VGG16-based perceptual loss to prevent averaging and force GT feature learning.
    
    Extracts features from multiple VGG16 layers and computes L1 distance
    between predicted and target features.
    
    Args:
        feature_layers: List of VGG layer indices to extract features from
                       Default: [3, 8, 15, 22] for relu1_2, relu2_2, relu3_3, relu4_3
    """
    
    def __init__(self, feature_layers=None):
        super().__init__()
        
        if feature_layers is None:
            # Default layers: relu1_2, relu2_2, relu3_3, relu4_3
            feature_layers = [3, 8, 15, 22]
        
        self.feature_layers = feature_layers
        
        # Load pretrained VGG16
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        
        # Freeze VGG weights - we only use it for feature extraction
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Set to eval mode
        self.features.eval()
        
        # Normalization values for VGG (ImageNet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        """Normalize input to VGG ImageNet range"""
        return (x - self.mean) / self.std
    
    def extract_features(self, x):
        """Extract features from specified VGG layers"""
        features = []
        h = x
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in self.feature_layers:
                features.append(h)
        return features
    
    def forward(self, pred, target):
        """
        Compute perceptual loss
        
        Args:
            pred: Predicted image [B, 3, H, W] in range [0, 1]
            target: Target image [B, 3, H, W] in range [0, 1]
            
        Returns:
            Perceptual loss (scalar tensor)
        """
        # Normalize inputs to VGG range
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)
        
        # Extract features
        pred_features = self.extract_features(pred_norm)
        target_features = self.extract_features(target_norm)
        
        # Compute L1 loss between features at each layer
        loss = 0.0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += F.l1_loss(pred_feat, target_feat)
        
        # Average over all layers
        loss = loss / len(self.feature_layers)
        
        return loss


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
