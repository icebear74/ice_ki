"""
SR Reference Model – EDSR-baseline x3 / SwinIR-M x3 / Bicubic-Fallback

Ladevorgehen (in dieser Reihenfolge):
  1. Gecachte EDSR-Gewichte in ~/.cache/ice_ki/sr/  → sofortiger Start
  2. Download der EDSR-Gewichte von der SNU-Projektseite (cv.snu.ac.kr)
  3. Download der SwinIR-M x3 Gewichte von GitHub Releases
     (JingyunLiang/SwinIR, immer erreichbar – keine externen Pakete nötig)
  4. Bicubic ×3 als eingebauter Fallback (kein Download, immer verfügbar)

SwinIR-M x3 (Liang et al. 2021, https://arxiv.org/abs/2108.10257):
    embed_dim=180, depths=[6,6,6,6,6,6], num_heads=[6,6,6,6,6,6],
    window_size=8, upscale=3, img_range=1.
    Gewichte: ~60 MB von GitHub Releases (keine timm-Abhängigkeit)

EDSR-baseline x3 (Lim et al. 2017):
    n_resblocks=16, n_feats=64, scale=3
    Gewichte: ~5 MB von cv.snu.ac.kr (wenn erreichbar)

``load_sr_model()`` gibt niemals ``None`` zurück.

Verwendung (intern in async_validator und train.py):
    from vsr_plusplus_NEU.core.sr_model import load_sr_model
    sr_model = load_sr_model(device)
"""

import math
import os
import socket
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── EDSR-Download-URLs (werden der Reihe nach versucht) ──────────────────────
_EDSR_URLS = [
    "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-9aade23f.pt",
]

# ── SwinIR-M x3 Gewichte (GitHub Releases — immer erreichbar) ────────────────
_SWINIR_URL = (
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
    "001_classicalSR_DIV2K_s48w8_SwinIR-M_x3.pth"
)

_DOWNLOAD_TIMEOUT_S = 20  # Sekunden pro Versuch

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


class _SwinIRWrapper(nn.Module):
    """
    Adapter für SwinIR-M x3: erwartet [0,1]-Tensoren, gibt [0,1] zurück.

    SwinIR verarbeitet [0,1]-Eingaben intern bereits selbst (img_range=1.),
    daher ist hier nur ein clamp nötig.
    """

    name = "SwinIR-M x3"

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).clamp(0.0, 1.0)


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

def _cache_dir() -> Path:
    """Gibt das Cache-Verzeichnis für SR-Gewichte zurück (~/.cache/ice_ki/sr/)."""
    d = Path(os.environ.get("ICE_KI_CACHE_DIR", Path.home() / ".cache" / "ice_ki")) / "sr"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_edsr_weights_path() -> Path:
    return _cache_dir() / "edsr_baseline_x3.pt"


def _get_swinir_weights_path() -> Path:
    return _cache_dir() / "swinir_m_x3_div2k.pth"


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
    Versucht EDSR-Gewichte zu laden (Cache → Download von SNU).
    Gibt ``None`` zurück wenn alle Versuche fehlschlagen.
    """
    weights_path = _get_edsr_weights_path()

    if not weights_path.exists():
        last_err: Exception | None = None
        for url in _EDSR_URLS:
            try:
                print(f"[SR] Versuche EDSR-Download …", flush=True)
                _download_weights(url, weights_path, _DOWNLOAD_TIMEOUT_S)
                print("[SR] ✅ EDSR-Download abgeschlossen", flush=True)
                last_err = None
                break
            except Exception as e:
                print(f"[SR]   ✗ EDSR: {e}", flush=True)
                last_err = e
        if last_err is not None:
            return None

    print("[SR] Lade EDSR-baseline x3 …", flush=True)
    net = _EDSRBaseline(scale=3, n_resblocks=16, n_feats=64)
    state = torch.load(weights_path, map_location='cpu', weights_only=True)
    net.load_state_dict(state, strict=False)
    net = net.to(device).eval()
    return _EDSRWrapper(net).to(device).eval()


def _try_load_swinir(device: torch.device) -> '_SwinIRWrapper | None':
    """
    Versucht SwinIR-M x3 Gewichte zu laden (Cache → Download von GitHub Releases).
    Gibt ``None`` zurück wenn der Download fehlschlägt.
    """
    from vsr_plusplus_NEU.core._swinir import SwinIR  # lokaler Import vermeidet zirkuläre Abhängigkeiten

    weights_path = _get_swinir_weights_path()

    if not weights_path.exists():
        try:
            print(f"[SR] Versuche SwinIR-Download von GitHub Releases (~60 MB) …", flush=True)
            _download_weights(_SWINIR_URL, weights_path, _DOWNLOAD_TIMEOUT_S)
            print("[SR] ✅ SwinIR-Download abgeschlossen", flush=True)
        except Exception as e:
            print(f"[SR]   ✗ SwinIR: {e}", flush=True)
            return None

    print("[SR] Lade SwinIR-M x3 …", flush=True)
    net = SwinIR(
        upscale=3, img_size=48, window_size=8, img_range=1.,
        depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2.,
        upsampler='pixelshuffle', resi_connection='1conv')

    raw = torch.load(weights_path, map_location='cpu', weights_only=True)
    # Official SwinIR checkpoints store weights under 'params' or 'params_ema'
    state = raw.get('params_ema') or raw.get('params') or raw
    net.load_state_dict(state, strict=True)
    net = net.to(device).eval()
    return _SwinIRWrapper(net).to(device).eval()


def load_sr_model(device: torch.device) -> '_EDSRWrapper | _SwinIRWrapper | _BicubicSR':
    """
    Lädt das SR-Referenzmodell für den Validator.

    Reihenfolge:
      1. Gecachte EDSR-Gewichte (~/.cache/ice_ki/sr/edsr_baseline_x3.pt)
      2. Download der EDSR-Gewichte von cv.snu.ac.kr
      3. SwinIR-M x3 von GitHub Releases (JingyunLiang/SwinIR)
      4. Bicubic ×3 als eingebauter Fallback

    Gibt **immer** ein lauffähiges nn.Module zurück (niemals None), damit
    der 5-Panel-TensorBoard-Vergleich in jedem Fall verfügbar ist.

    Das zurückgegebene Modell hat ein ``.name``-Attribut das angibt, welcher
    Typ geladen wurde.

    Args:
        device: Ziel-Gerät (z.B. ``torch.device('cuda:1')``)

    Returns:
        _EDSRWrapper, _SwinIRWrapper oder _BicubicSR im eval()-Modus auf *device*.
    """
    # ── 1+2: EDSR (Cache oder SNU-Download) ──────────────────────────────────
    edsr_path = _get_edsr_weights_path()
    if edsr_path.exists():
        print(f"[SR] Verwende gecachte EDSR-Gewichte: {edsr_path}", flush=True)
    try:
        wrapper = _try_load_edsr(device)
        if wrapper is not None:
            total_params = sum(p.numel() for p in wrapper.parameters()) / 1e6
            print(f"[SR] ✅ EDSR-baseline x3 geladen ({total_params:.1f}M Parameter)", flush=True)
            return wrapper
    except Exception as e:
        print(f"[SR] ⚠ EDSR-Ladefehler: {e}", flush=True)

    # ── 3: SwinIR-M x3 (GitHub Releases) ─────────────────────────────────────
    swinir_path = _get_swinir_weights_path()
    if swinir_path.exists():
        print(f"[SR] Verwende gecachte SwinIR-Gewichte: {swinir_path}", flush=True)
    try:
        wrapper = _try_load_swinir(device)
        if wrapper is not None:
            total_params = sum(p.numel() for p in wrapper.parameters()) / 1e6
            print(f"[SR] ✅ SwinIR-M x3 geladen ({total_params:.1f}M Parameter)", flush=True)
            return wrapper
    except Exception as e:
        print(f"[SR] ⚠ SwinIR-Ladefehler: {e}", flush=True)

    # ── 4: Bicubic-Fallback ───────────────────────────────────────────────────
    print("[SR] ↩ Bicubic-Fallback aktiv (kein SR-Modell heruntergeladen)", flush=True)
    print("[SR]   Bicubic ×3 ist der Standard-SR-Baseline — Vergleich bleibt sinnvoll.", flush=True)
    return _BicubicSR(scale=3).to(device).eval()
