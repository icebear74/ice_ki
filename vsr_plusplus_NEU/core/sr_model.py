"""
BasicSRModel - leichtgewichtiges Single-Image Super-Resolution Modell (SRCNN-Stil)

Aufbau:
  1. Bicubic-Upsample des LR-Eingangsbildes auf Zielgröße (3×)
  2. Drei Conv-Schichten mit ReLU zur Verfeinerung der Struktur

Das Modell verwendet nur das Mittelbild (Frame-Index 3) aus dem 7-Frame-Stack,
da es keinerlei Temporal-Information verarbeitet – so ist der Vergleich
mit dem VSR-Modell fair.

Verwendung:
    model = BasicSRModel()
    # Für gespeicherte Gewichte:
    model = BasicSRModel.from_checkpoint('/pfad/zur/datei.pth', device)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicSRModel(nn.Module):
    """
    SRCNN-inspiriertes SR-Modell für den Vergleich mit dem VSR-Modell.

    Architektur:
      Bicubic-Upsample → Conv(3,64,9) → ReLU → Conv(64,32,1) → ReLU → Conv(32,3,5)

    Args:
        scale:    Upscale-Faktor (Standard: 3, passend zum VSR-Modell)
        n_mid:    Anzahl mittlere Feature-Maps der zweiten Conv-Schicht
        n_feats:  Anzahl Feature-Maps der ersten Conv-Schicht
    """

    def __init__(self, scale: int = 3, n_feats: int = 64, n_mid: int = 32):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(3, n_feats, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(n_feats, n_mid,  kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(n_mid,   3,       kernel_size=5, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor der Form [B, C, H, W] (einzelnes Bild, bereits auf Zielgröße
               hochskaliert ODER noch auf LR-Größe — wird intern upgesamplt).

        Returns:
            Tensor [B, C, H*scale, W*scale] wenn x auf LR-Größe, sonst [B, C, H, W]
        """
        # Bicubic-Upsample (kein Gradienten-Overhead außerhalb des Trainings)
        up = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        out = F.relu(self.conv1(up), inplace=True)
        out = F.relu(self.conv2(out), inplace=True)
        out = self.conv3(out)
        # Residual: SR-Korrektur + Bicubic-Basis
        return torch.clamp(up + out, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, path: str, device: torch.device,
                        scale: int = 3, n_feats: int = 64, n_mid: int = 32):
        """
        Erstellt ein BasicSRModel und lädt Gewichte aus einer .pth-Datei.

        Die Datei kann entweder ein reines state_dict oder ein dict mit dem
        Schlüssel 'model_state_dict' sein.

        Args:
            path:    Pfad zur Checkpoint-Datei
            device:  Ziel-Gerät
            scale:   Upscale-Faktor (muss mit gespeichertem Modell übereinstimmen)
            n_feats: Feature-Maps (muss mit gespeichertem Modell übereinstimmen)
            n_mid:   Mittlere Feature-Maps

        Returns:
            BasicSRModel im eval()-Modus auf *device*
        """
        model = cls(scale=scale, n_feats=n_feats, n_mid=n_mid)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model
