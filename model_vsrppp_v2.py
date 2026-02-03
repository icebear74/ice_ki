import torch
import torch.nn as nn
import torch.nn.functional as F

class HeavyBlock(nn.Module):
    def __init__(self, n_feats=96):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )
        self.last_activity = 0.0

    def forward(self, x):
        res = self.conv(x)
        out = x + res * 0.1
        self.last_activity = out.detach().abs().mean().item()
        return out

class VSRTriplePlus_3x(nn.Module):
    def __init__(self, n_feats=96, n_blocks=30):
        super().__init__()
        self.n_feats = n_feats
        half_blocks = max(1, n_blocks // 2)

        # 1. Feature Extraction
        self.feat_extract = nn.Conv2d(3, n_feats, 3, 1, 1)

        # 2. Adaptive Fusion (instead of naive addition)
        self.back_fusion = nn.Conv2d(n_feats * 2, n_feats, 1)
        self.forw_fusion = nn.Conv2d(n_feats * 2, n_feats, 1)
        
        # 3. Propagation Trunks
        self.backward_trunk = nn.Sequential(*[HeavyBlock(n_feats) for _ in range(half_blocks)])
        self.forward_trunk  = nn.Sequential(*[HeavyBlock(n_feats) for _ in range(half_blocks)])
        
        # 4. Learnable Temporal Weights
        self.temp_weight = nn.Parameter(torch.tensor([0.3, 1.0, 0.3]))
        
        # 5. Fusion
        self.fusion = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

        # 6. Upsampling (Factor 3)
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 9, 3, 1, 1),
            nn.PixelShuffle(3),
            nn.Conv2d(n_feats, 3, 3, 1, 1)
        )

    def forward(self, x):
        b, t, c, h, w = x.size()
        
        base = F.interpolate(x[:, 2], scale_factor=3, mode='bilinear', align_corners=False)

        f = self.feat_extract(x.view(-1, c, h, w)).view(b, t, -1, h, w)

        # Adaptive fusion instead of addition
        back = self.backward_trunk(self.back_fusion(torch.cat([f[:, 3], f[:, 4]], dim=1)))
        forw = self.forward_trunk(self.forw_fusion(torch.cat([f[:, 0], f[:, 1]], dim=1)))

        # Learnable temporal weights with softmax
        w = F.softmax(self.temp_weight, dim=0)
        out = f[:, 2] * w[1] + back * w[0] + forw * w[2]
        out = self.fusion(out)

        return self.upsample(out)
    
    def get_layer_activity(self):
        """Returns activity levels for all blocks"""
        acts = []
        for m in self.backward_trunk:
            if isinstance(m, HeavyBlock):
                acts.append(m.last_activity)
        for m in self.forward_trunk:
            if isinstance(m, HeavyBlock):
                acts.append(m.last_activity)
        return acts if acts else [0.0] * 30
    
    @property
    def frame_stats(self):
        """Dummy stats for train.py compatibility"""
        return {"F1": 0.0, "F2": 0.0, "F4": 0.0, "F5": 0.0}