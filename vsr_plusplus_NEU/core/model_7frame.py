"""
7-Frame Bidirectional VSR Model
MATCHES original VSRBidirectional_3x architecture exactly for realistic memory testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class ResidualBlock(nn.Module):
    """Residual block matching original architecture."""
    def __init__(self, n_feats, use_checkpointing=False):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.1, inplace=False)  # LeakyReLU like original
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.last_activity = 0.0
        self.use_checkpointing = use_checkpointing
        
    def _forward_impl(self, x):
        """Internal forward computation, separated for gradient checkpointing support."""
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = residual + out
        return out
        
    def forward(self, x):
        if self.use_checkpointing and self.training:
            out = checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            out = self._forward_impl(x)
        
        # Track activity
        self.last_activity = out.detach().abs().mean().item()
        
        return out

class FusionBlock(nn.Module):
    """
    FusionBlock v2: Mit Residual Skip-Connection (1x1).

    Der Skip-Pfad (1x1 Conv, kein Bias) projiziert in_feats → out_feats direkt.
    Der Haupt-Pfad (3x3 → ReLU → 1x1) lernt nur noch die Korrektur.
    Bewährtes ResNet-Muster: stabilere, schnellere Konvergenz.

    Trackt Aktivität aller drei Conv-Layer getrennt für WebUI-Visualisierung.
    """
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_feats, out_feats, 3, 1, 1)
        self.relu    = nn.LeakyReLU(0.1, inplace=False)
        self.conv1x1 = nn.Conv2d(out_feats, out_feats, 1)
        # Skip: projiziert in_feats → out_feats (kein Bias, kein räumlicher Kontext)
        # Entspricht dem Dimensions-Anpassungs-Skip aus ResNet (Option B)
        self.skip    = nn.Conv2d(in_feats, out_feats, 1, bias=False)
        self.last_activity_3x3  = 0.0
        self.last_activity_1x1  = 0.0
        self.last_activity_skip = 0.0

    def forward(self, x):
        # Skip-Pfad: direkte Projektion (günstig, stabil)
        identity = self.skip(x)
        self.last_activity_skip = identity.detach().abs().mean().item()

        # Haupt-Pfad: lernt nur noch die Korrektur auf identity
        out = self.conv3x3(x)
        self.last_activity_3x3 = out.detach().abs().mean().item()
        out = self.relu(out)
        out = self.conv1x1(out)
        self.last_activity_1x1 = out.detach().abs().mean().item()

        return out + identity

class VSRBidirectional_7frames_3x(nn.Module):
    """
    7-Frame Bidirectional VSR Model - EXACT MATCH to original training architecture
    
    Input: [B, 7, 3, H, W] (7 frames)
    Output: [B, 3, H*3, W*3] (upscaled center frame)
    
    Architecture matches VSRBidirectional_3x for realistic memory measurements.
    """
    def __init__(self, n_feats=72, n_blocks=26, use_checkpointing=False):
        super().__init__()
        self.n_feats = n_feats
        self.n_blocks = n_blocks
        
        half_blocks = max(1, n_blocks // 2)
        
        # 1. Feature Extraction
        self.feat_extract = nn.Conv2d(3, n_feats, 3, 1, 1)
        
        # 2. Fusion layers for combining features (CRITICAL for memory)
        self.backward_fuse = FusionBlock(n_feats * 2, n_feats)
        self.forward_fuse = FusionBlock(n_feats * 2, n_feats)
        
        # 3. Propagation Trunks
        self.backward_trunk = nn.ModuleList([
            ResidualBlock(n_feats, use_checkpointing=use_checkpointing) for _ in range(half_blocks)
        ])
        self.forward_trunk = nn.ModuleList([
            ResidualBlock(n_feats, use_checkpointing=use_checkpointing) for _ in range(half_blocks)
        ])
        
        # 4. Final Fusion
        self.fusion = FusionBlock(n_feats * 2, n_feats)
        
        # 5. Upsampling (3x with PixelShuffle)
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 9, 3, 1, 1),
            nn.PixelShuffle(3),
            nn.Conv2d(n_feats, 3, 3, 1, 1)
        )
        
    def forward(self, x):
        """
        Forward pass matching original architecture
        
        Args:
            x: Input tensor [B, 7, 3, H, W]
            
        Returns:
            Output tensor [B, 3, H*3, W*3]
        """
        B, T, C, H, W = x.size()
        
        # Extract features from all 7 frames
        feats = self.feat_extract(x.view(-1, C, H, W))
        feats = feats.view(B, T, self.n_feats, H, W)
        
        # Initialize with Frame 4 (center frame, index 3)
        center_feat = feats[:, 3].clone()
        
        # Backward propagation: F4 → F5 → F6
        back_prop = center_feat
        for i in [4, 5, 6]:
            # Fuse THEN process through trunk (like original)
            fused = self.backward_fuse(torch.cat([back_prop, feats[:, i]], dim=1))
            for block in self.backward_trunk:
                fused = block(fused)
            back_prop = fused
        
        # Forward propagation: F4 → F3 → F2 → F1
        forw_prop = center_feat
        for i in [2, 1, 0]:
            # Fuse THEN process through trunk (like original)
            fused = self.forward_fuse(torch.cat([forw_prop, feats[:, i]], dim=1))
            for block in self.forward_trunk:
                fused = block(fused)
            forw_prop = fused
        
        # Fuse bidirectional features
        fused = self.fusion(torch.cat([back_prop, forw_prop], dim=1))
        
        # Upsampling with residual connection (CRITICAL - like original!)
        base = F.interpolate(x[:, 3], scale_factor=3, mode='bilinear', align_corners=False)
        upsampled = self.upsample(fused)
        
        return upsampled + base
    
    def get_layer_activity(self):
        """
        Returns activity levels for all blocks including fusion layers

        Returns:
            Dict with activities:
            {
                'backward_trunk': [list of ResidualBlock activities],
                'backward_fuse': [3x3 activity, 1x1 activity, skip activity],
                'forward_trunk': [list of ResidualBlock activities],
                'forward_fuse': [3x3 activity, 1x1 activity, skip activity],
                'fusion': [3x3 activity, 1x1 activity, skip activity]
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
            'backward_fuse': [self.backward_fuse.last_activity_3x3, self.backward_fuse.last_activity_1x1, self.backward_fuse.last_activity_skip],
            'forward_trunk': forward_activities,
            'forward_fuse': [self.forward_fuse.last_activity_3x3, self.forward_fuse.last_activity_1x1, self.forward_fuse.last_activity_skip],
            'fusion': [self.fusion.last_activity_3x3, self.fusion.last_activity_1x1, self.fusion.last_activity_skip]
        }

