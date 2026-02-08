"""
7-Frame Bidirectional VSR Model
MATCHES original VSRBidirectional_3x architecture exactly for realistic memory testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block matching original architecture."""
    def __init__(self, n_feats):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.1, inplace=False)  # LeakyReLU like original
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return residual + out

class VSRBidirectional_7frames_3x(nn.Module):
    """
    7-Frame Bidirectional VSR Model - EXACT MATCH to original training architecture
    
    Input: [B, 7, 3, H, W] (7 frames)
    Output: [B, 3, H*3, W*3] (upscaled center frame)
    
    Architecture matches VSRBidirectional_3x for realistic memory measurements.
    """
    def __init__(self, n_feats=64, n_blocks=24):
        super().__init__()
        self.n_feats = n_feats
        self.n_blocks = n_blocks
        
        half_blocks = max(1, n_blocks // 2)
        
        # 1. Feature Extraction
        self.feat_extract = nn.Conv2d(3, n_feats, 3, 1, 1)
        
        # 2. Fusion layers for combining features (CRITICAL for memory)
        self.backward_fuse = nn.Conv2d(n_feats * 2, n_feats, 1)
        self.forward_fuse = nn.Conv2d(n_feats * 2, n_feats, 1)
        
        # 3. Propagation Trunks
        self.backward_trunk = nn.ModuleList([
            ResidualBlock(n_feats) for _ in range(half_blocks)
        ])
        self.forward_trunk = nn.ModuleList([
            ResidualBlock(n_feats) for _ in range(half_blocks)
        ])
        
        # 4. Final Fusion
        self.fusion = nn.Conv2d(n_feats * 2, n_feats, 1)
        
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
