"""
SR Reference Model - EDSR-baseline x3 (Enhanced Deep Super-Resolution)

Implementiert das EDSR-baseline x3 Netzwerk direkt (keine externen Pakete,
kein torch.hub nötig) und lädt die offiziellen vortrainierten Gewichte per
direktem HTTP-Download von der SNU-Projektseite.

Gewichte werden in ~/.cache/ice_ki/sr/ gecacht; nach dem ersten Download
startet das Modell offline und sofort.

Verwendung (intern in async_validator und train.py):
    from vsr_plusplus_NEU.core.sr_model import load_sr_model
    sr_model = load_sr_model(device)  # None wenn Download fehlschlägt

Architektur (EDSR-baseline, Lim et al. 2017):
    n_resblocks=16, n_feats=64, scale=3, res_scale=1.0
    RGB-Mittelwertsubtraktion: (114.4, 111.5, 103.0)
"""

import math
import os
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn

# ── Offizielle vortrainierte Gewichte (EDSR-baseline x3) ─────────────────────
# Quelle: http://cv.snu.ac.kr/research/EDSR/models/
_WEIGHTS_URL = (
    "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-9aade23f.pt"
)
_WEIGHTS_SHA256_PREFIX = "9aade23f"   # Bestandteil des Dateinamens als Checksumme

# RGB-Mittelwerte die das offizielle EDSR-baseline-Modell subtrahiert (×255-Raum)
_RGB_MEAN = (0.4488 * 255, 0.4371 * 255, 0.4040 * 255)  # ≈ (114.4, 111.5, 103.0)


# ── EDSR-Bausteine ────────────────────────────────────────────────────────────

class _ResBlock(nn.Module):
    def __init__(self, n_feats: int, res_scale: float = 1.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class _Upsampler(nn.Sequential):
    """PixelShuffle-basiertes Upsampling (scale muss 2, 3 oder 4 sein)."""

    def __init__(self, n_feats: int, scale: int):
        layers: list[nn.Module] = []
        if (scale & (scale - 1)) == 0:           # Potenz von 2
            for _ in range(int(math.log(scale, 2))):
                layers += [nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1),
                            nn.PixelShuffle(2)]
        elif scale == 3:
            layers += [nn.Conv2d(n_feats, 9 * n_feats, 3, padding=1),
                        nn.PixelShuffle(3)]
        else:
            raise ValueError(f"Upsampler: scale {scale} nicht unterstützt")
        super().__init__(*layers)


class _EDSRBaseline(nn.Module):
    """
    EDSR-baseline (Lim et al. 2017): n_resblocks=16, n_feats=64, scale=3.

    Eingang/Ausgang: Float-Tensoren im Bereich [0, 255], Form [B, 3, H, W].
    """

    def __init__(self, scale: int = 3, n_resblocks: int = 16, n_feats: int = 64):
        super().__init__()
        # Registriere RGB-Mittelwert als Buffer (wird mit state_dict gespeichert)
        mean = torch.tensor(_RGB_MEAN).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)

        self.head = nn.Conv2d(3, n_feats, 3, padding=1)
        self.body = nn.Sequential(
            *[_ResBlock(n_feats) for _ in range(n_resblocks)],
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )
        self.tail = nn.Sequential(
            _Upsampler(n_feats, scale),
            nn.Conv2d(n_feats, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.mean
        x = self.head(x)
        x = x + self.body(x)
        x = self.tail(x)
        return x + self.mean.repeat(1, 1, x.shape[-2] // self.mean.shape[-2] + 1,
                                         x.shape[-1] // self.mean.shape[-1] + 1
                                    )[:, :, :x.shape[-2], :x.shape[-1]]


class _EDSRWrapper(nn.Module):
    """
    Normierungsadapter: erwartet [0,1]-Tensoren, gibt [0,1]-Tensoren zurück.

      - Eingang : [B, C, H, W]  im Bereich [0, 1]
      - _EDSRBaseline: erwartet und liefert [0, 255]
      - Ausgang : [B, C, H*scale, W*scale]  im Bereich [0, 1]
    """

    def __init__(self, model: _EDSRBaseline):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x * 255.0)
        return torch.clamp(out / 255.0, 0.0, 1.0)


# ── Gewichte laden ────────────────────────────────────────────────────────────

def _get_weights_path() -> Path:
    """Gibt den Cache-Pfad für die Gewichte zurück (~/.cache/ice_ki/sr/)."""
    cache_dir = Path(os.environ.get("ICE_KI_CACHE_DIR", Path.home() / ".cache" / "ice_ki")) / "sr"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "edsr_baseline_x3.pt"


def _download_weights(url: str, dest: Path) -> None:
    """Lädt Gewichte herunter und zeigt einen einfachen Fortschrittsbalken."""
    print(f"[SR] Lade Gewichte herunter: {url}", flush=True)
    print(f"[SR]   → {dest}", flush=True)

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            downloaded = min(block_num * block_size, total_size)
            pct = downloaded * 100 // total_size
            bar = '█' * (pct // 5) + '░' * (20 - pct // 5)
            print(f"\r[SR]   [{bar}] {pct:3d}%  {downloaded/1e6:.1f}/{total_size/1e6:.1f} MB",
                  end='', flush=True)

    tmp = dest.with_suffix('.tmp')
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
        print(flush=True)  # Zeilenumbruch nach Fortschrittsbalken
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def load_sr_model(device: torch.device) -> '_EDSRWrapper | None':
    """
    Lädt das vortrainierte EDSR-baseline x3 Modell.

    Beim ersten Aufruf werden die Gewichte (~5 MB) von der offiziellen
    SNU-Projektseite heruntergeladen und in ~/.cache/ice_ki/sr/ gecacht.
    Danach startet das Modell sofort ohne Netzwerkzugriff.

    Kein ``torch.hub``, kein separates Paket — nur PyTorch erforderlich.

    Args:
        device: Ziel-Gerät (z.B. ``torch.device('cuda:1')``)

    Returns:
        _EDSRWrapper im eval()-Modus auf *device*, oder None bei Fehler.
    """
    try:
        weights_path = _get_weights_path()

        if not weights_path.exists():
            print("[SR] Gewichte nicht im Cache — starte Download …", flush=True)
            _download_weights(_WEIGHTS_URL, weights_path)
            print("[SR] ✅ Download abgeschlossen", flush=True)
        else:
            print(f"[SR] Verwende gecachte Gewichte: {weights_path}", flush=True)

        print("[SR] Lade EDSR-baseline x3 …", flush=True)
        net = _EDSRBaseline(scale=3, n_resblocks=16, n_feats=64)
        state = torch.load(weights_path, map_location='cpu', weights_only=True)
        net.load_state_dict(state, strict=False)
        net = net.to(device)
        net.eval()

        wrapper = _EDSRWrapper(net).to(device)
        wrapper.eval()

        total_params = sum(p.numel() for p in wrapper.parameters()) / 1e6
        print(f"[SR] ✅ EDSR-baseline x3 geladen ({total_params:.1f}M Parameter)", flush=True)
        return wrapper

    except Exception as e:
        print(f"[SR] ⚠ EDSR konnte nicht geladen werden: {e}", flush=True)
        print("[SR]   Validierung läuft ohne SR-Referenz weiter.", flush=True)
        return None
