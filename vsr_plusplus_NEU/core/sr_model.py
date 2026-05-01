"""
SR Reference Model – EDSR-baseline x3 mit Bicubic-Fallback

Strategie (in dieser Reihenfolge):
  1. Versucht, vortrainierte EDSR-baseline-x3-Gewichte von mehreren Quellen
     herunterzuladen (SNU-Projektseite und GitHub-Mirror) und in
     ~/.cache/ice_ki/sr/ zu cachen.  Danach startet das Modell sofort offline.
  2. Schlägt jeder Download fehl, wird automatisch ein eingebautes
     Bicubic-Upsampling-Modell als SR-Referenz genutzt.

Bicubic ist der standardmäßige SR-Vergleichs-Baseline (alle SR-Paper
vergleichen gegen Bicubic), benötigt keinerlei Download und liefert direkt
einen brauchbaren 5-Panel-Vergleich in TensorBoard.

``load_sr_model()`` gibt niemals ``None`` zurück, solange PyTorch funktioniert.

Verwendung (intern in async_validator und train.py):
    from vsr_plusplus_NEU.core.sr_model import load_sr_model
    sr_model = load_sr_model(device)  # Bicubic-Fallback wenn Download fehlschlägt

Architektur (EDSR-baseline, Lim et al. 2017):
    n_resblocks=16, n_feats=64, scale=3, res_scale=1.0
    RGB-Mittelwertsubtraktion: (114.4, 111.5, 103.0)
"""

import math
import os
import socket
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Download-URLs (werden der Reihe nach versucht) ───────────────────────────
# Erste URL: offizielle SNU-Projektseite
# Zweite URL: GitHub-Release-Mirror des gleichen Checkpoints
_WEIGHTS_URLS = [
    "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-9aade23f.pt",
    "https://github.com/sanghyun-son/EDSR-PyTorch/releases/download/v1.0.0/edsr_baseline_x3-9aade23f.pt",
]
_DOWNLOAD_TIMEOUT_S = 15  # Sekunden pro Versuch

# RGB-Mittelwerte die das offizielle EDSR-baseline-Modell subtrahiert (×255-Raum)
_RGB_MEAN = (0.4488 * 255, 0.4371 * 255, 0.4040 * 255)  # ≈ (114.4, 111.5, 103.0)


# ── EDSR-Bausteine ────────────────────────────────────────────────────────────

class _ResBlock(nn.Module):
    def __init__(self, n_feats: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


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
    Normierungsadapter für EDSR: erwartet [0,1]-Tensoren, gibt [0,1] zurück.
    """

    name = "EDSR-baseline x3"

    def __init__(self, model: _EDSRBaseline):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x * 255.0)
        return torch.clamp(out / 255.0, 0.0, 1.0)


class _BicubicSR(nn.Module):
    """
    Bicubic-Upsampling als SR-Fallback-Referenz.

    Bicubic ×3 ist der Standard-Baseline in der SR-Literatur und benötigt
    keinerlei Download.  Eingang/Ausgang: [B, C, H, W] im Bereich [0, 1].
    """

    name = "Bicubic x3 (Fallback)"

    def __init__(self, scale: int = 3):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale,
                             mode='bicubic', align_corners=False).clamp(0.0, 1.0)


# ── Gewichte laden ────────────────────────────────────────────────────────────

def _get_weights_path() -> Path:
    """Gibt den Cache-Pfad für die Gewichte zurück (~/.cache/ice_ki/sr/)."""
    cache_dir = Path(os.environ.get("ICE_KI_CACHE_DIR", Path.home() / ".cache" / "ice_ki")) / "sr"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "edsr_baseline_x3.pt"


def _download_weights(url: str, dest: Path, timeout: int) -> None:
    """Lädt Gewichte von *url* nach *dest* mit Fortschrittsbalken."""
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
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout)
        try:
            urllib.request.urlretrieve(url, tmp, reporthook=_progress)
        finally:
            socket.setdefaulttimeout(old_timeout)
        print(flush=True)
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _try_load_edsr(device: torch.device) -> '_EDSRWrapper | None':
    """
    Versucht EDSR-Gewichte zu laden (Cache → Download mehrerer URLs).
    Gibt ``None`` zurück wenn alle Versuche fehlschlagen.
    """
    weights_path = _get_weights_path()

    if not weights_path.exists():
        last_err: Exception | None = None
        for url in _WEIGHTS_URLS:
            try:
                print(f"[SR] Versuche Download: {url.split('/')[-1]} …", flush=True)
                _download_weights(url, weights_path, _DOWNLOAD_TIMEOUT_S)
                print("[SR] ✅ Download abgeschlossen", flush=True)
                last_err = None
                break
            except Exception as e:
                print(f"[SR]   ✗ {e}", flush=True)
                last_err = e
        if last_err is not None:
            return None

    print("[SR] Lade EDSR-baseline x3 …", flush=True)
    net = _EDSRBaseline(scale=3, n_resblocks=16, n_feats=64)
    state = torch.load(weights_path, map_location='cpu', weights_only=True)
    net.load_state_dict(state, strict=False)
    net = net.to(device).eval()
    return _EDSRWrapper(net).to(device).eval()


def load_sr_model(device: torch.device) -> '_EDSRWrapper | _BicubicSR':
    """
    Lädt das SR-Referenzmodell für den Validator.

    Reihenfolge:
      1. Gecachte EDSR-Gewichte in ~/.cache/ice_ki/sr/ (sofortiger Start)
      2. Download der EDSR-Gewichte (mehrere URLs, je {timeout}s Timeout)
      3. Bicubic ×3 als eingebauter Fallback (kein Download nötig)

    Gibt **immer** ein lauffähiges nn.Module zurück (niemals None), damit
    der 5-Panel-TensorBoard-Vergleich in jedem Fall verfügbar ist.

    Das zurückgegebene Modell hat ein ``.name``-Attribut das angibt, welcher
    Typ geladen wurde (z.B. ``"EDSR-baseline x3"`` oder ``"Bicubic x3 (Fallback)"``).

    Args:
        device: Ziel-Gerät (z.B. ``torch.device('cuda:1')``)

    Returns:
        _EDSRWrapper oder _BicubicSR im eval()-Modus auf *device*.
    """.format(timeout=_DOWNLOAD_TIMEOUT_S)
    try:
        weights_path = _get_weights_path()
        if weights_path.exists():
            print(f"[SR] Verwende gecachte Gewichte: {weights_path}", flush=True)
        wrapper = _try_load_edsr(device)
        if wrapper is not None:
            total_params = sum(p.numel() for p in wrapper.parameters()) / 1e6
            print(f"[SR] ✅ EDSR-baseline x3 geladen ({total_params:.1f}M Parameter)", flush=True)
            return wrapper
    except Exception as e:
        print(f"[SR] ⚠ EDSR-Ladefehler: {e}", flush=True)

    # ── Bicubic-Fallback ──────────────────────────────────────────────────────
    print("[SR] ↩ Bicubic-Fallback aktiv (kein Download möglich)", flush=True)
    print("[SR]   Bicubic ×3 ist der Standard-SR-Baseline — Vergleich bleibt sinnvoll.", flush=True)
    bicubic = _BicubicSR(scale=3).to(device).eval()
    return bicubic
