"""
VSRBidirectional_3x - Bidirectional Video Super-Resolution Model

Key features:
- Frame-3 initialization (NOT zeros!)
- Bidirectional propagation (backward F3→F4→F5, forward F3→F2→F1)
- 3x upscaling (180x180 → 540x540)
- Activity monitoring for all blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ResidualBlock(nn.Module):
    """Residual block with activity monitoring and gradient checkpointing support"""
    
    def __init__(self, n_feats, use_checkpointing=False):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.1, inplace=False)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.last_activity = 0.0
    
    def _forward_impl(self, x):
        """Actual forward pass implementation"""
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = residual + out
        
        # Track activity
        self.last_activity = out.detach().abs().mean().item()
        
        return out
    
    def forward(self, x):
        """Forward with optional gradient checkpointing"""
        if self.use_checkpointing and self.training:
            # Use gradient checkpointing during training
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            # Normal forward pass during validation/inference
            return self._forward_impl(x)


class TrackedConv2d(nn.Module):
    """Conv2d wrapper with activity tracking (for fusion layers)"""
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        self.last_activity = 0.0
    
    def forward(self, x):
        out = self.conv(x)
        # Track activity
        self.last_activity = out.detach().abs().mean().item()
        return out


class VSRBidirectional_3x(nn.Module):
    """
    Bidirectional VSR model with Frame-3 initialization
    
    Input: [B, 5, 3, 180, 180] (5 frames, 180x180 LR)
    Output: [B, 3, 540, 540] (1 frame, 540x540 HR, 3x upscale)
    
    Args:
        n_feats: Number of feature channels (64-256, auto-tuned)
        n_blocks: Total number of ResBlocks (split between trunks)
        use_checkpointing: Enable gradient checkpointing to save VRAM
    """
    
    def __init__(self, n_feats=128, n_blocks=32, use_checkpointing=False):
        super().__init__()
        self.n_feats = n_feats
        self.n_blocks = n_blocks
        self.use_checkpointing = use_checkpointing
        
        half_blocks = max(1, n_blocks // 2)
        
        # 1. Feature Extraction
        self.feat_extract = nn.Conv2d(3, n_feats, 3, 1, 1)
        
        # 2. Fusion layers for combining features (WITH TRACKING)
        self.backward_fuse = TrackedConv2d(n_feats * 2, n_feats, 1)
        self.forward_fuse = TrackedConv2d(n_feats * 2, n_feats, 1)
        
        # 3. Propagation Trunks
        self.backward_trunk = nn.ModuleList([
            ResidualBlock(n_feats, use_checkpointing=use_checkpointing) for _ in range(half_blocks)
        ])
        self.forward_trunk = nn.ModuleList([
            ResidualBlock(n_feats, use_checkpointing=use_checkpointing) for _ in range(half_blocks)
        ])
        
        # 4. Final Fusion (WITH TRACKING)
        self.fusion = TrackedConv2d(n_feats * 2, n_feats, 1)
        
        # 5. Upsampling (3x with PixelShuffle)
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 9, 3, 1, 1),
            nn.PixelShuffle(3),
            nn.Conv2d(n_feats, 3, 3, 1, 1)
        )
    
    def forward(self, x):
        """
        Forward pass with bidirectional propagation
        
        Args:
            x: Input tensor [B, 5, 3, 180, 180]
            
        Returns:
            Output tensor [B, 3, 540, 540]
        """
        B, T, C, H, W = x.size()
        
        # Extract features from all 5 frames
        feats = self.feat_extract(x.view(-1, C, H, W))
        feats = feats.view(B, T, self.n_feats, H, W)
        
        # CRITICAL: Initialize with Frame 3 (center frame), NOT zeros!
        center_feat = feats[:, 2].clone()  # Frame 3 (index 2)
        
        # Backward propagation: F3 → F4 → F5
        back_prop = center_feat
        for i in [3, 4]:
            # Fuse current propagation with next frame
            fused = self.backward_fuse(torch.cat([back_prop, feats[:, i]], dim=1))
            # Process through backward trunk
            for block in self.backward_trunk:
                fused = block(fused)
            back_prop = fused
        
        # Forward propagation: F3 → F2 → F1
        forw_prop = center_feat
        for i in [1, 0]:
            # Fuse current propagation with previous frame
            fused = self.forward_fuse(torch.cat([forw_prop, feats[:, i]], dim=1))
            # Process through forward trunk
            for block in self.forward_trunk:
                fused = block(fused)
            forw_prop = fused
        
        # Fuse bidirectional features
        fused = self.fusion(torch.cat([back_prop, forw_prop], dim=1))
        
        # Upsampling with residual connection
        base = F.interpolate(x[:, 2], scale_factor=3, mode='bilinear', align_corners=False)
        upsampled = self.upsample(fused)
        
        return upsampled + base
    
    def get_layer_activity(self):
        """
        Returns activity levels for all blocks including fusion layers
        
        Returns:
            Dict with activities:
            {
                'backward_trunk': [list of ResidualBlock activities],
                'backward_fuse': float (fusion layer activity),
                'forward_trunk': [list of ResidualBlock activities],
                'forward_fuse': float (fusion layer activity),
                'fusion': float (final fusion layer activity)
            }
        """
        backward_activities = []
        for block in self.backward_trunk:
            backward_activities.append(block.last_activity)
        
        forward_activities = []
        for block in self.forward_trunk:
            forward_activities.append(block.last_activity)
        
        return {
            'backward_trunk': backward_activities,
            'backward_fuse': self.backward_fuse.last_activity,
            'forward_trunk': forward_activities,
            'forward_fuse': self.forward_fuse.last_activity,
            'fusion': self.fusion.last_activity
        }
