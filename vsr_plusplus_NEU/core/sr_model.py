"""
SR Reference Model - EDSR (Enhanced Deep Super-Resolution) via torch.hub

Lädt automatisch ein vortrainiertes EDSR x3 Modell aus dem offiziellen
Repository (sanghyun-son/EDSR-PyTorch) und stellt es als PyTorch-Modul
bereit, das [0,1]-normierte Tensoren erwartet und zurückgibt.

Verwendung (intern im async_validator):
    from vsr_plusplus_NEU.core.sr_model import load_sr_model
    sr_model = load_sr_model(device)  # None wenn Download fehlschlägt
"""

import torch
import torch.nn as nn


class _EDSRWrapper(nn.Module):
    """
    Schlanker Wrapper um das torch.hub-EDSR-Modell.

    Passt die Normierung an:
      - Eingang : [B, C, H, W]  im Bereich [0, 1]
      - EDSR    : erwartet und liefert Pixelwerte im Bereich [0, 255]
      - Ausgang : [B, C, H*scale, W*scale]  im Bereich [0, 1]
    """

    def __init__(self, hub_model):
        super().__init__()
        self.hub_model = hub_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.hub_model(x * 255.0)
        return torch.clamp(out / 255.0, 0.0, 1.0)


def load_sr_model(device: torch.device) -> '_EDSRWrapper | None':
    """
    Lädt das vortrainierte EDSR x3 Modell via torch.hub.

    Gibt ``None`` zurück wenn der Download fehlschlägt, damit das Training
    ohne SR-Referenz weiterläuft.

    Args:
        device: Ziel-Gerät (z.B. ``torch.device('cuda:1')``)

    Returns:
        _EDSRWrapper im eval()-Modus auf *device*, oder None bei Fehler.
    """
    try:
        print("[SR] Lade EDSR x3 via torch.hub …", flush=True)
        hub_model = torch.hub.load(
            'sanghyun-son/EDSR-PyTorch',
            'edsr',
            pretrained=True,
            scale=3,
            verbose=False,
        )
        hub_model = hub_model.to(device)
        hub_model.eval()
        wrapper = _EDSRWrapper(hub_model).to(device)
        wrapper.eval()
        total_params = sum(p.numel() for p in wrapper.parameters()) / 1e6
        print(f"[SR] ✅ EDSR x3 geladen ({total_params:.1f}M Parameter)", flush=True)
        return wrapper
    except Exception as e:
        print(f"[SR] ⚠ EDSR konnte nicht geladen werden: {e}", flush=True)
        print("[SR]   Validierung läuft ohne SR-Referenz weiter.", flush=True)
        return None
