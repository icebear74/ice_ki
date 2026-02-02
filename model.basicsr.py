import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, n_feats=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )
    def forward(self, x):
        return x + self.conv(x)

class BasicVSR_Foundational(nn.Module):
    def __init__(self, n_feats=64, n_blocks=20): # Hier sind deine 20 Layer
        super().__init__()
        self.n_feats = n_feats
        self.feat_extract = nn.Conv2d(3, n_feats, 3, 1, 1)
        
        # Propagation-Zweige (Gedächtnis vorwärts/rückwärts)
        self.backward_res = nn.Sequential(*[ResidualBlock(n_feats) for _ in range(n_blocks)])
        self.forward_res = nn.Sequential(*[ResidualBlock(n_feats) for _ in range(n_blocks)])
        
        # Upsampling (Faktor 3)
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 9, 3, 1, 1),
            nn.PixelShuffle(3),
            nn.Conv2d(n_feats, 3, 3, 1, 1)
        )

    def forward(self, x):
        # x: [Batch, Time(5), Channels(3), H(180), W(180)]
        b, t, c, h, w = x.size()
        
        # 1. Feature Extraction
        feats = self.feat_extract(x.view(-1, c, h, w)).view(b, t, self.n_feats, h, w)
        
        # 2. Backward Pass (von Frame 5 zu 1)
        outputs_back = []
        prop = torch.zeros(b, self.n_feats, h, w).to(x.device)
        for i in range(t-1, -1, -1):
            prop = self.backward_res(feats[:, i] + prop)
            outputs_back.append(prop)
        outputs_back = outputs_back[::-1]
        
        # 3. Forward Pass & Fusion (von Frame 1 zu 5)
        # Wir geben am Ende nur den mittleren Frame (Index 2) als Ergebnis aus,
        # da dieser die Infos von 2 davor und 2 danach am besten nutzt.
        prop = torch.zeros(b, self.n_feats, h, w).to(x.device)
        for i in range(t):
            prop = self.forward_res(feats[:, i] + prop + outputs_back[i])
            if i == 2: # Der Zielframe (Mitte der 5er Sequenz)
                return self.upsample(prop)
                