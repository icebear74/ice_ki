import torch
import torch.nn as nn
import torch.nn.functional as F

class HeavyBlock(nn.Module):
    def __init__(self, n_feats=96):
        super().__init__()
        # Stabileres Design mit 3x3 Convs
        self.conv = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )
        self.last_activity = 0.0

    def forward(self, x):
        res = self.conv(x)
        # Residual Scaling (0.1) verhindert das "Ausflippen" der Pixel
        out = x + res * 0.1
        self.last_activity = out.detach().abs().mean().item()
        return out

class VSRTriplePlus_3x(nn.Module):
    def __init__(self, n_feats=96, n_blocks=30): # n_blocks wieder drin
        super().__init__()
        self.n_feats = n_feats
        # Wir teilen n_blocks auf die zwei Zweige auf (z.B. 15/15)
        half_blocks = max(1, n_blocks // 2)

        # 1. Feature Extraction
        self.feat_extract = nn.Conv2d(3, n_feats, 3, 1, 1)

        # 2. Propagation Trunks (Stabilisierung durch Aufteilung)
        self.backward_trunk = nn.Sequential(*[HeavyBlock(n_feats) for _ in range(half_blocks)])
        self.forward_trunk  = nn.Sequential(*[HeavyBlock(n_feats) for _ in range(half_blocks)])
        
        # 3. Sanfte Fusion
        self.fusion = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

        # 4. Upsampling (Faktor 3)
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 9, 3, 1, 1),
            nn.PixelShuffle(3),
            nn.Conv2d(n_feats, 3, 3, 1, 1)
        )

    def forward(self, x):
        b, t, c, h, w = x.size()
        
        # Bilinearer Anker für Farbstabilität (Frame 2 ist die Mitte von 0,1,2,3,4)
        base = F.interpolate(x[:, 2], scale_factor=3, mode='bilinear', align_corners=False)

        # Features extrahieren
        f = self.feat_extract(x.view(-1, c, h, w)).view(b, t, -1, h, w)

        # Zukunft (4,3) und Vergangenheit (0,1) sanft addieren
        back = self.backward_trunk(f[:, 3] + f[:, 4])
        forw = self.forward_trunk(f[:, 0] + f[:, 1])

        # Finale Fusion: Ziel-Frame + Kontext
        out = f[:, 2] + (back * 0.5) + (forw * 0.5)
        out = self.fusion(out)

        # Rekonstruktion + Basis-Upscale
        return self.upsample(out) + base