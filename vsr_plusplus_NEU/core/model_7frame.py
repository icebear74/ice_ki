import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Residual block with two conv layers."""
    def __init__(self, n_feats, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2)
        
    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res

class VSRBidirectional_7frames_3x(nn.Module):
    """
    7-Frame Bidirectional VSR Model
    Input: 7 frames stacked (B, 3, H, W*7)
    Output: Upscaled center frame (B, 3, H*3, W*3)
    """
    def __init__(self, n_feats=64, n_blocks=24):
        super(VSRBidirectional_7frames_3x, self).__init__()
        
        self.n_feats = n_feats
        self.n_blocks = n_blocks
        
        # Feature extraction
        self.fea_extract = nn.Conv2d(3, n_feats, 3, 1, 1)
        
        # Bidirectional trunks
        half_blocks = n_blocks // 2
        self.backward_trunk = nn.Sequential(*[ResidualBlock(n_feats) for _ in range(half_blocks)])
        self.forward_trunk = nn.Sequential(*[ResidualBlock(n_feats) for _ in range(half_blocks)])
        
        # Fusion
        self.fusion = nn.Conv2d(n_feats * 2, n_feats, 1, 1, 0)
        
        # Upsampling (3x with PixelShuffle)
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 9, 3, 1, 1),
            nn.PixelShuffle(3),
            nn.Conv2d(n_feats, 3, 3, 1, 1)
        )
        
    def forward(self, x):
        """
        x: (B, 3, H, W*7) - 7 frames horizontally stacked
        Returns: (B, 3, H*3, W*3) - upscaled center frame
        """
        B, C, H, W_total = x.shape
        W = W_total // 7
        
        # Split into 7 frames
        frames = torch.split(x, W, dim=3)  # List of 7 tensors (B, 3, H, W)
        
        # Extract features for each frame
        feats = [self.fea_extract(f) for f in frames]  # 7x (B, n_feats, H, W)
        
        # Initialize propagation from CENTER frame (frame 3, index 3)
        center_feat = feats[3].clone()
        
        # Backward propagation: F3 → F4 → F5 → F6
        back_prop = center_feat
        back_feats = [center_feat]
        for i in [4, 5, 6]:
            back_prop = self.backward_trunk(back_prop + feats[i])
            back_feats.append(back_prop)
        
        # Forward propagation: F3 → F2 → F1 → F0
        forw_prop = center_feat
        forw_feats = [center_feat]
        for i in [2, 1, 0]:
            forw_prop = self.forward_trunk(forw_prop + feats[i])
            forw_feats.insert(0, forw_prop)
        
        # Fuse bidirectional features at center frame
        fused = torch.cat([back_feats[0], forw_feats[3]], dim=1)  # (B, n_feats*2, H, W)
        fused = self.fusion(fused)  # (B, n_feats, H, W)
        
        # Upsample
        out = self.upsample(fused)  # (B, 3, H*3, W*3)
        
        return out
