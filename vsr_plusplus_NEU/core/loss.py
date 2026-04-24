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
    - Single forward pass: layers evaluated once, outputs extracted at
      relu1_2 (idx 3), relu2_2 (idx 8), relu3_3 (idx 15), relu4_3 (idx 22)
    """
    
    def __init__(self):
        super().__init__()
        # Load VGG16 with ImageNet weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
        # Store all layers up to relu4_3 as a single sequence for one forward pass
        self.features = vgg.features[:23]  # layers 0–22 (relu4_3 inclusive)
        
        # Indices at which to extract intermediate feature maps:
        # relu1_2=3, relu2_2=8, relu3_3=15, relu4_3=22
        self._extract_indices = {3, 8, 15, 22}
        
        # Freeze all VGG parameters
        for param in self.features.parameters():
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
        Compute perceptual loss using a single forward pass through VGG.
        
        Args:
            pred: Predicted image [B, 3, H, W] in range [0, 1]
            target: Target image [B, 3, H, W] in range [0, 1]
            
        Returns:
            Perceptual loss (scalar)
        """
        # Normalize inputs
        pred = self.normalize(pred)
        target = self.normalize(target)
        
        # Single pass: accumulate L1 loss at each target layer index
        loss = 0.0
        num_extracted = 0
        x_pred = pred
        x_target = target
        for i, layer in enumerate(self.features):
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            if i in self._extract_indices:
                loss += F.l1_loss(x_pred, x_target)
                num_extracted += 1
        
        # Average over extracted layers (num_extracted is always 4 but guard defensively)
        return loss / max(num_extracted, 1)


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
        
        # 2. Multi-Scale Loss (2x and 4x downsampling, averaged)
        pred_down2 = F.avg_pool2d(pred, kernel_size=2, stride=2)
        target_down2 = F.avg_pool2d(target, kernel_size=2, stride=2)
        pred_down4 = F.avg_pool2d(pred_down2, kernel_size=2, stride=2)
        target_down4 = F.avg_pool2d(target_down2, kernel_size=2, stride=2)
        ms_loss = (F.l1_loss(pred_down2, target_down2) +
                   F.l1_loss(pred_down4, target_down4)) / 2
        
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
        # Lazily initialize PerceptualLoss so that a scheduled weight > 0 works
        # even when the initial perceptual_weight was set to 0.0 at construction.
        if perceptual_w > 0:
            if self.perceptual_loss is None:
                self.perceptual_loss = PerceptualLoss().to(pred.device)
            # VGG16 always runs in float32 (for stability and because it is never
            # converted to half()).  Cast pred/target to float32 here so that the
            # call works correctly when the SR model itself is in float16.
            perceptual_loss = self.perceptual_loss(pred.float(), target.float())
        else:
            perceptual_loss = torch.tensor(0.0, device=pred.device)
        
        # 5. Weighted combination, normalized by total weight so gradient
        #    magnitude stays constant as perceptual weight ramps up.
        total_w = l1_w + ms_w + grad_w + perceptual_w
        total_w = max(total_w, 1e-8)  # guard against all-zero weights
        total_loss = (l1_w * l1_loss + 
                     ms_w * ms_loss + 
                     grad_w * grad_loss + 
                     perceptual_w * perceptual_loss) / total_w
        
        return {
            'l1': l1_loss.item(),
            'ms': ms_loss.item(),
            'grad': grad_loss.item(),
            'perceptual': perceptual_loss.item() if isinstance(perceptual_loss, torch.Tensor) else 0.0,
            'total': total_loss
        }
