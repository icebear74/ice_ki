#!/usr/bin/env python3
"""
Complete Video Processing Pipeline with VSR Comparison
- Detects interlacing, frame redundancy (YADIF artifacts)
- Works with both 50fps and 25fps videos
- Auto-corrects via YADIF deinterlacing or fps=25 deduplication
- Generates split-screen VSR comparison
- 7-frame VSR model support
"""

import cv2
import subprocess
import json
import numpy as np
from pathlib import Path
import sys
import shutil
import os
import gc
import warnings
import torch
from collections import defaultdict
import time as time_module

# ANSI colors
C_GREEN  = "\033[92m"
C_CYAN   = "\033[96m"
C_RED    = "\033[91m"
C_YELLOW = "\033[93m"
C_RESET  = "\033[0m"


# ═══════════════════════════════════════════════════════════════════════════
# TensorRT: Engine-Cache, Auto-Build (via optimize_checkpoint.py), Inferenz
# ═══════════════════════════════════════════════════════════════════════════

def _engine_path_for(checkpoint_path: str, width: int, height: int,
                     precision: str = 'fp16') -> Path:
    """
    Gibt den Pfad der gecachten TRT Engine zurück.
    Format: <checkpoint_dir>/<checkpoint_stem>_trt_<precision>_<W>x<H>.engine
    Wiederverwendbar: gleicher Checkpoint + gleiche Auflösung → gleiche Engine.
    """
    p = Path(checkpoint_path)
    return p.parent / f"{p.stem}_trt_{precision}_{width}x{height}.engine"


def _build_trt_for_vergleich(checkpoint_path: str, width: int, height: int,
                              engine_path: Path, precision: str = 'fp16') -> bool:
    """
    Baut eine TRT Engine via optimize_checkpoint.py als Subprozess.
    Output läuft direkt auf das Terminal — der User sieht den vollen Build-Fortschritt.
    Bei Änderungen an optimize_checkpoint.py wirken diese automatisch.
    """
    optimize_script = Path(__file__).resolve().parent / 'optimize_checkpoint.py'
    if not optimize_script.exists():
        print(f"  ❌ optimize_checkpoint.py nicht gefunden: {optimize_script}")
        return False

    print(f"\n  🏗️  TRT Engine wird gebaut (einmalig, dann gecacht):")
    print(f"     Checkpoint : {checkpoint_path}")
    print(f"     Engine     : {engine_path}")
    print(f"     Input      : {width}×{height}  →  SR {width * 3}×{height * 3}")
    print(f"     ⏳ Bitte warten — typisch ~5 min, maximal 15 min...\n")

    cmd = [
        sys.executable,
        str(optimize_script),
        '--checkpoint', checkpoint_path,
        '--output',     str(engine_path),
        '--format',     'tensorrt',
        '--precision',  precision,
        '--width',      str(width),
        '--height',     str(height),
        '--workspace-gb', '2',
        '--device',     'cuda',
    ]

    _TIMEOUT = 900  # 15 Minuten
    try:
        # Kein capture_output — subprocess-Output erscheint direkt im Terminal
        result = subprocess.run(cmd, timeout=_TIMEOUT)
        if result.returncode == 0:
            print(f"\n  ✅ TRT Engine gespeichert: {engine_path}")
            return True
        print(f"\n  ❌ optimize_checkpoint.py fehlgeschlagen (returncode={result.returncode})")
        return False
    except subprocess.TimeoutExpired:
        print(f"\n  ❌ TRT Build Timeout ({_TIMEOUT // 60} min)")
        return False
    except Exception as e:
        print(f"\n  ❌ TRT Build fehlgeschlagen: {e}")
        return False


class _TRTSession:
    """
    Schlanke TensorRT-Inferenz-Session ohne pycuda.

    Verwendet torch CUDA-Pointer (Tensor.data_ptr()) mit execute_v2 —
    funktioniert mit TRT 8.x und 9.x ohne zusätzliche Abhängigkeiten.
    I/O-Typ: float32 (entspricht dem ONNX-Export des Modells).
    """

    def __init__(self, engine_path: str, input_shape: tuple):
        import tensorrt as trt
        _logger  = trt.Logger(trt.Logger.WARNING)
        _runtime = trt.Runtime(_logger)
        with open(engine_path, 'rb') as f:
            self._engine = _runtime.deserialize_cuda_engine(f.read())
        self._context    = self._engine.create_execution_context()
        self.input_shape = input_shape          # (1, 7, 3, H, W)
        B, T, C, H, W   = input_shape
        self.output_shape = (B, C, H * 3, W * 3)

    def infer(self, frames_tensor: torch.Tensor) -> torch.Tensor:
        """
        frames_tensor : (1, 7, 3, H, W)  float32  contiguous  auf CUDA
        Rückgabe      : (1, 3, H*3, W*3) float32  auf CUDA
        Kein pycuda — nutzt rohe CUDA-Device-Pointer aus torch.Tensor.
        """
        inp = frames_tensor.contiguous().float()
        out = torch.empty(self.output_shape, dtype=torch.float32, device=inp.device)
        self._context.execute_v2([inp.data_ptr(), out.data_ptr()])
        return out


def select_gpu(gpu_index: int = None) -> torch.device:
    """Zeigt alle verfügbaren GPUs an und lässt den Nutzer eine auswählen.

    Args:
        gpu_index: Vorgegebener GPU-Index (z.B. über --gpu auf der CLI).
                   Ist er angegeben, wird keine interaktive Auswahl gestartet.

    Returns:
        torch.device: Ausgewähltes Gerät (z.B. 'cuda:0', 'cuda:1', 'cpu').
    """
    if not torch.cuda.is_available():
        print(f"{C_YELLOW}⚠ Kein CUDA-fähiges Gerät gefunden – läuft auf CPU.{C_RESET}")
        return torch.device('cpu')

    gpu_count = torch.cuda.device_count()

    # Immer alle GPUs anzeigen
    print(f"\n{C_CYAN}{'='*60}{C_RESET}")
    print(f"{C_CYAN}  Verfügbare GPUs:{C_RESET}")
    print(f"{C_CYAN}{'='*60}{C_RESET}")
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        mem_total = props.total_memory / (1024 ** 3)
        mem_free  = (props.total_memory - torch.cuda.memory_allocated(i)) / (1024 ** 3)
        cc = f"CC {props.major}.{props.minor}"
        print(f"  [{i}] {props.name}  –  {mem_total:.1f} GB gesamt, ~{mem_free:.1f} GB frei  ({cc})")
    print(f"{C_CYAN}{'='*60}{C_RESET}")

    # Vorgegebener Index (--gpu)
    if gpu_index is not None:
        if 0 <= gpu_index < gpu_count:
            name = torch.cuda.get_device_name(gpu_index)
            print(f"{C_GREEN}✓ Verwende GPU {gpu_index}: {name}  (via --gpu){C_RESET}\n")
            return torch.device(f'cuda:{gpu_index}')
        else:
            print(f"{C_RED}⚠ --gpu {gpu_index} ungültig (nur 0–{gpu_count - 1} verfügbar) – frage interaktiv.{C_RESET}")

    # Nur eine GPU → automatisch
    if gpu_count == 1:
        name = torch.cuda.get_device_name(0)
        print(f"{C_GREEN}✓ GPU erkannt: {name} – wird automatisch verwendet.{C_RESET}\n")
        return torch.device('cuda:0')

    # Interaktive Auswahl
    while True:
        try:
            raw = input(f"GPU-Index wählen [0–{gpu_count - 1}]: ").strip()
            idx = int(raw)
            if 0 <= idx < gpu_count:
                name = torch.cuda.get_device_name(idx)
                print(f"{C_GREEN}✓ Verwende GPU {idx}: {name}{C_RESET}\n")
                return torch.device(f'cuda:{idx}')
            print(f"{C_RED}Ungültige Eingabe. Bitte eine Zahl zwischen 0 und {gpu_count - 1} eingeben.{C_RESET}")
        except ValueError:
            print(f"{C_RED}Ungültige Eingabe. Bitte eine Ganzzahl eingeben.{C_RESET}")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: VIDEO ANALYSIS
# ═══════════════��═══════════════════════════════════════════════════════════

def get_video_info(video_path: str) -> dict:
    """Get video metadata using ffprobe"""
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'json',
        '-show_format', '-show_streams',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return None
            
        data = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    
    if 'streams' not in data:
        return None
    
    video_stream = next(
        (s for s in data['streams'] if s['codec_type'] == 'video'),
        None
    )
    
    if not video_stream:
        return None
    
    is_interlaced = (
        video_stream.get('field_order', 'progressive') != 'progressive' or
        video_stream.get('interlaced_frame', 0) > 0
    )
    
    width = video_stream.get('width', 0)
    height = video_stream.get('height', 0)
    
    return {
        'fps': float(video_stream.get('r_frame_rate', '25').split('/')[0]) / 
               float(video_stream.get('r_frame_rate', '25').split('/')[1] or 1),
        'width': width,
        'height': height,
        'is_interlaced': is_interlaced,
        'codec': video_stream.get('codec_name', 'unknown'),
        'frames': int(video_stream.get('nb_frames', 0)),
        'is_dvd_res': width == 720 and height == 576
    }


def extract_frames(video_path: str, duration_sec: int = 2) -> list:
    """Extract frames for analysis"""
    temp_dir = Path("frames_temp")
    
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    temp_dir.mkdir(exist_ok=True)
    
    cmd = [
        'ffmpeg', '-loglevel', 'quiet',
        '-hwaccel', 'cuda',
        '-ss', '300',           # Start at 5 minutes
        '-t', str(duration_sec),
        '-i', video_path,
        f'{temp_dir}/frame_%04d.png'
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=30)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return []
    
    frames = sorted(temp_dir.glob("frame_*.png"))
    return frames


def frame_similarity(frame1, frame2) -> float:
    """Calculate frame similarity 0-1"""
    if frame1 is None or frame2 is None:
        return 0.0
    
    hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8],
                         [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8],
                         [0, 256, 0, 256, 0, 256])
    
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return 1.0 - similarity


def analyze_video(video_path: str) -> dict:
    """Complete video analysis - works for both 50fps and 25fps"""
    
    info = get_video_info(video_path)
    if not info:
        return {
            'status': 'error',
            'reason': 'Could not read video metadata'
        }
    
    # Check 1: Interlaced?
    if info['is_interlaced']:
        return {
            'status': 'interlaced',
            'fps': info['fps'],
            'resolution': f"{info['width']}×{info['height']}",
            'action_required': 'deinterlace',
            'filter': 'yadif=mode=0'
        }
    
    # Check 2: DVD resolution?
    if not info['is_dvd_res']:
        return {
            'status': 'non_dvd',
            'fps': info['fps'],
            'resolution': f"{info['width']}×{info['height']}",
            'action_required': None
        }
    
    # Check 3: Frame redundancy (works for ANY fps - 50fps or 25fps!)
    print(f"  Analyzing frame redundancy...")
    frames = extract_frames(video_path, duration_sec=2)
    
    if len(frames) < 2:
        return {
            'status': 'error',
            'reason': 'Could not extract frames',
            'fps': info['fps'],
            'resolution': f"{info['width']}×{info['height']}"
        }
    
    # Compare EVERY consecutive frame pair
    similarities = []
    for i in range(0, len(frames) - 1, 1):
        f1 = cv2.imread(str(frames[i]))
        f2 = cv2.imread(str(frames[i + 1]))
        sim = frame_similarity(f1, f2)
        similarities.append(sim)
    
    avg_sim = np.mean(similarities)
    dup_count = sum(1 for s in similarities if s > 0.98)
    dup_ratio = dup_count / len(similarities) * 100 if similarities else 0
    
    shutil.rmtree("frames_temp", ignore_errors=True)
    
    # Determine status based on redundancy ratio
    if dup_ratio >= 90:
        status = 'yadif_broken'
        action = 'deduplicate'
        filter_str = 'fps=25,setpts=N/FRAME_RATE/TB'
    elif dup_ratio >= 70:
        status = 'yadif_partial'
        action = 'deduplicate'
        filter_str = 'fps=25,setpts=N/FRAME_RATE/TB'
    else:
        status = 'yadif_good'
        action = None
        filter_str = None
    
    return {
        'status': status,
        'fps': info['fps'],
        'resolution': f"{info['width']}×{info['height']}",
        'frames_extracted': len(frames),
        'redundancy_ratio': dup_ratio,
        'avg_similarity': avg_sim,
        'action_required': action,
        'filter': filter_str
    }


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: VIDEO CORRECTION
# ═══════════════════════════════════════════════════════════════════════════

def correct_video(input_path: str, output_path: str, correction_type: str) -> bool:
    """Apply video corrections"""
    
    print(f"\n  🔧 Correcting: {correction_type}")
    
    if correction_type == 'deinterlace':
        cmd = [
            'ffmpeg', '-hwaccel', 'cuda',
            '-i', input_path,
            '-vf', 'yadif=mode=0',
            '-c:v', 'hevc_nvenc', '-preset', 'slow', '-crf', '18',
            '-c:a', 'copy',
            '-map', '0',
            output_path
        ]
    
    elif correction_type == 'deduplicate':
        cmd = [
            'ffmpeg', '-hwaccel', 'cuda',
            '-i', input_path,
            '-vf', 'fps=25,setpts=N/FRAME_RATE/TB',
            '-c:v', 'hevc_nvenc', '-preset', 'slow', '-crf', '18',
            '-c:a', 'copy',
            '-map', '0',
            output_path
        ]
    
    else:
        raise ValueError(f"Unknown correction type: {correction_type}")
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        print(f"  ✅ Corrected: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Correction failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: VSR COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def _infer_arch_from_checkpoint(ckpt: dict):
    """
    Liest n_feats und n_blocks aus einem geladenen Checkpoint.

    Strategie (Priorität):
      1. ckpt['model_config']  — explizit gespeichert (neue Checkpoints)
      2. Ableitung aus state_dict-Tensor-Shapes  — funktioniert für alle Checkpoints
      3. Fallback config.py (lokale Datei neben den Skripten)
      4. Hard-kodierte Defaults (72 / 28)

    Gibt (n_feats, n_blocks, quelle) zurück.
    """
    _DEFAULTS = (72, 28)

    # 1. Explizit gespeicherter model_config (neue Checkpoints)
    mc = ckpt.get("model_config", {})
    if mc.get("N_FEATS") and mc.get("N_BLOCKS"):
        return int(mc["N_FEATS"]), int(mc["N_BLOCKS"]), "model_config (checkpoint)"

    # 2. Ableitung aus state_dict
    state = ckpt.get("model_state_dict", ckpt)
    try:
        n_feats  = int(state["feat_extract.weight"].shape[0])
        half     = len({k.split(".")[1] for k in state if k.startswith("backward_trunk.")})
        n_blocks = half * 2
        return n_feats, n_blocks, "state_dict (abgeleitet)"
    except Exception:
        pass

    # 3. config.py
    try:
        import config as cfg
        c = cfg.get_config()
        nf = int(c.get("N_FEATS",  _DEFAULTS[0]))
        nb = int(c.get("N_BLOCKS", _DEFAULTS[1]))
        return nf, nb, "config.py"
    except Exception:
        pass

    # 4. Defaults
    return _DEFAULTS[0], _DEFAULTS[1], "Defaults"


class VSRComparator:
    def __init__(self, checkpoint_path: str, device: str = 'cuda:0'):
        """Initialize VSR model — architecture is read from the checkpoint"""
        self.device          = torch.device(device)
        self.available       = False
        self.use_fp16        = False
        self.checkpoint_path = checkpoint_path   # für TRT Engine-Cache

        try:
            from vsr_plusplus_NEU.core.model_7frame import VSRBidirectional_7frames_3x

            # Load checkpoint to CPU first — avoids doubling GPU peak memory during load
            # and lets us infer n_feats/n_blocks before constructing the model.
            print(f"  Loading checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            # Architektur-Parameter direkt aus dem Checkpoint lesen
            n_feats, n_blocks, source = _infer_arch_from_checkpoint(ckpt)
            print(f"  Architektur ({source}): n_feats={n_feats}, n_blocks={n_blocks}")

            # Let cuDNN auto-select the fastest conv kernels for the fixed input size
            torch.backends.cudnn.benchmark = True

            # Persistieren für TRT Engine-Cache
            self.n_feats  = n_feats
            self.n_blocks = n_blocks

            print(f"  Loading VSR model (n_blocks={n_blocks}, n_feats={n_feats})...")
            self.model = VSRBidirectional_7frames_3x(n_feats=n_feats, n_blocks=n_blocks)

            state = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state)
            del ckpt  # free CPU RAM immediately

            # Move to GPU, switch to fp16 (2× throughput on CUDA), set eval mode
            self.model = self.model.to(self.device).half().eval()
            self.use_fp16 = True

            # torch.compile: fuses ops and removes Python overhead (PyTorch >= 2.0)
            # Triton (the compile backend) requires CUDA Capability >= 7.0.
            # Tesla P100 is CC 6.0 — skip compile silently on old GPUs.
            _cc = torch.cuda.get_device_capability(self.device)
            if _cc[0] >= 7:
                try:
                    self.model = torch.compile(self.model, mode='reduce-overhead')
                    print(f"  ✅ torch.compile enabled (reduce-overhead, CC {_cc[0]}.{_cc[1]})")
                except Exception:
                    pass  # graceful fallback on older PyTorch
            else:
                # CC < 7.0: torch.compile/Triton not supported.
                # Fallback: torch.jit.trace + freeze applied lazily on first inference
                # (shape-dependent; traced once per unique input resolution).
                self._jit_traced_shape = None  # (H, W) of currently frozen trace; None = not yet traced
                self._jit_model = None
                print(f"  ℹ️  torch.compile skipped (CC {_cc[0]}.{_cc[1]} < 7.0) — torch.jit.trace fallback will be used")

            self.available = True
            print(f"  ✅ VSR model loaded (fp16, cudnn.benchmark=True)")
        except Exception as e:
            print(f"  ⚠️  VSR model failed to load: {e}")
            import traceback
            traceback.print_exc()
            self.available = False
    
    def create_comparison(self, input_video: str, output_video: str) -> bool:
        """
        Erstellt Split-Screen-Vergleich (FFmpeg-Upscale links | VSR rechts).

        Ablauf:
          [1/3] Frames extrahieren (1 Minute ab 5:00)
          [2/3] FFmpeg 3×-Upscale der extrahierten Frames
          [3/3] VSR-Inferenz + direktes Streaming an ffmpeg-Combine (kein .raw-Temp-File)

        Inferenz-Backend (automatische Auswahl):
          • TRT  (bevorzugt): Engine wird beim ersten Lauf neben dem Checkpoint gebaut
                              und bei gleicher Auflösung wiederverwendet.
          • PyTorch FP16 (Fallback): chunked mini-batch, JIT-trace auf CC < 7.0
        """
        if not self.available:
            print(f"  ⚠️  Skipping VSR comparison (model unavailable)")
            return False

        print(f"\n🎨 VSR COMPARISON:")
        temp_dir    = Path("/tmp/vsr_compare")
        combine_proc = None          # Popen-Handle; wird in finally geschlossen

        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(exist_ok=True)

        # Konfigurierbare Zeitparameter
        _EXTRACT_OFFSET  = 300   # Start bei 5:00 min im Video
        _EXTRACT_DURATION = 60   # 1 Minute extrahieren
        _EXTRACT_TIMEOUT = max(300, _EXTRACT_DURATION * 5)  # Timeout: 5× Videodauer

        try:
            # ────────────────────────────────────────────────────────────
            # Phase 1: Frames extrahieren (1 Minute)
            # ────────────────────────────────────────────────────────────
            print(f"  [1/3] Extracting frames ({_EXTRACT_DURATION}s at {_EXTRACT_OFFSET//60}:{_EXTRACT_OFFSET%60:02d})...")
            cmd = [
                'ffmpeg', '-hwaccel', 'cuda', '-loglevel', 'quiet',
                '-ss', str(_EXTRACT_OFFSET), '-t', str(_EXTRACT_DURATION),
                '-i', input_video,
                f'{temp_dir}/frame_%06d.png'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=_EXTRACT_TIMEOUT)
            if result.returncode != 0:
                print(f"       ❌ Failed to extract frames")
                return False

            frames = sorted(temp_dir.glob("frame_*.png"))
            if len(frames) < 7:
                print(f"       ⚠️  Not enough frames: {len(frames)} (need 7+)")
                return False

            print(f"       ✓ {len(frames)} frames extracted")

            # Eingabe-Dimensionen aus erstem Frame ableiten
            _probe      = cv2.imread(str(frames[0]))
            H_in, W_in  = _probe.shape[:2]
            del _probe
            H_out, W_out = H_in * 3, W_in * 3
            half_w       = W_out // 2
            print(f"       Input: {W_in}×{H_in} → Output: {W_out}×{H_out}")

            # ────────────────────────────────────────────────────────────
            # TRT Engine: auto-bauen falls nicht vorhanden, dann laden
            # Engine-Datei liegt neben dem Checkpoint und wird wiederverwendet
            # solange Checkpoint und Auflösung gleich bleiben.
            # ────────────────────────────────────────────────────────────
            trt_session  = None
            engine_path  = _engine_path_for(self.checkpoint_path, W_in, H_in)
            input_shape  = (1, 7, 3, H_in, W_in)

            try:
                import tensorrt  # noqa: F401 — nur Verfügbarkeit prüfen
                if not engine_path.exists():
                    built = _build_trt_for_vergleich(
                        self.checkpoint_path, W_in, H_in, engine_path
                    )
                    if not built:
                        print(f"  ⚠️  TRT Build fehlgeschlagen — PyTorch-Fallback wird verwendet")
                if engine_path.exists():
                    print(f"  🚀 Lade TRT Engine: {engine_path.name}")
                    trt_session = _TRTSession(str(engine_path), input_shape)
                    print(f"     ✅ TRT Session bereit")
            except ImportError:
                print(f"  ℹ️  tensorrt nicht verfügbar — verwende PyTorch fp16")

            # ────────────────────────────────────────────────────────────
            # Phase 2: FFmpeg Referenz-Upscale (Lanczos 3×)
            # ────────────────────────────────────────────────────────────
            ffmpeg_upscale_path = str(temp_dir / 'ffmpeg_upscale.mkv')
            print(f"  [2/3] FFmpeg upscaling ({len(frames)} frames → {W_out}×{H_out})...")
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-hwaccel', 'cuda', '-loglevel', 'error',
                '-framerate', '25',
                '-pattern_type', 'glob', '-i', f'{temp_dir}/frame_*.png',
                '-vf', f'scale={W_out}:{H_out}:flags=lanczos',
                '-c:v', 'hevc_nvenc', '-preset', 'medium', '-crf', '18',
                ffmpeg_upscale_path,
            ]
            t0 = time_module.time()
            r  = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                print(f"       ❌ FFmpeg failed: {r.stderr}")
                return False
            print(f"       ✓ done ({time_module.time() - t0:.1f}s)")

            # ────────────────────────────────────────────────────────────
            # Phase 3: VSR-Inferenz + Combine via Stdin-Pipe
            # Kein temporäres .raw-File — Frames werden direkt gestreamt.
            # ffmpeg liest VSR-Frames aus stdin (pipe:0) während wir inferieren.
            # ────────────────────────────────────────────────────────────
            N       = len(frames)
            OFFSETS = [-3, -2, -1, 0, 1, 2, 3]
            all_indices = [
                [max(0, min(N - 1, i + o)) for o in OFFSETS]
                for i in range(N)
            ]

            # Alle PNG-Frames in CPU-RAM laden (random access für 7-frame-Fenster)
            print(f"  [3/3] VSR inference → ffmpeg combine (streaming, {N} frames)...")
            print(f"       Loading {N} frames to RAM...")
            cpu_frames = []
            for fp in frames:
                bgr = cv2.imread(str(fp))
                cpu_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            print(f"       {N} frames loaded")

            # JIT-trace Fallback für CC < 7.0 (kein torch.compile), wenn kein TRT
            dtype       = torch.float16 if self.use_fp16 else torch.float32
            infer_model = None
            if trt_session is None:
                if hasattr(self, '_jit_traced_shape'):
                    if self._jit_traced_shape != (H_in, W_in):
                        print(f"       🔧 Applying torch.jit.trace+freeze ({H_in}×{W_in})...")
                        dummy = torch.zeros(1, 7, 3, H_in, W_in,
                                            dtype=dtype, device=self.device)
                        try:
                            with torch.no_grad(), warnings.catch_warnings():
                                warnings.simplefilter("ignore", torch.jit.TracerWarning)
                                traced = torch.jit.trace(self.model, dummy)
                                self._jit_model = torch.jit.freeze(traced)
                            self._jit_traced_shape = (H_in, W_in)
                            print(f"       ✅ torch.jit.trace+freeze ready")
                        except Exception as _jit_ex:
                            print(f"       ⚠️  JIT trace failed ({_jit_ex}), using eager")
                            self._jit_model = self.model
                            self._jit_traced_shape = (H_in, W_in)
                    infer_model = self._jit_model
                else:
                    infer_model = self.model

            # ffmpeg combine-Prozess starten — liest VSR-Frames aus stdin
            filter_complex = (
                f"[0:v]crop={half_w}:{H_out}:0:0[left];"
                f"[1:v]crop={half_w}:{H_out}:{half_w}:0[right];"
                f"[left][right]hstack[combined];"
                f"[combined]"
                f"drawbox=x={half_w - 2}:y=0:w=4:h={H_out}:color=white:t=fill,"
                f"drawtext=text='FFmpeg Upscale (x3)':fontsize=60:fontcolor=white"
                f":x=50:y=50:box=1:boxcolor=black@0.5,"
                f"drawtext=text='VSR Model (x3)':fontsize=60:fontcolor=white"
                f":x={half_w + 30}:y=50:box=1:boxcolor=black@0.5[out]"
            )
            combine_cmd = [
                'ffmpeg', '-y', '-loglevel', 'error',
                '-i', ffmpeg_upscale_path,
                '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                '-s', f'{W_out}x{H_out}', '-r', '25',
                '-i', 'pipe:0',
                '-filter_complex', filter_complex,
                '-map', '[out]',
                '-c:v', 'hevc_nvenc', '-preset', 'medium', '-crf', '18',
                output_video,
            ]
            combine_proc = subprocess.Popen(combine_cmd,
                                            stdin=subprocess.PIPE,
                                            stderr=subprocess.PIPE)

            # ── Inferenz-Schleife ─────────────────────────────────────
            mode       = "TRT" if trt_session else "PyTorch"
            CHUNK      = 1 if trt_session else 8
            t_start    = time_module.time()
            n_done     = 0

            with torch.no_grad():
                for chunk_start in range(0, N, CHUNK):
                    chunk_end     = min(chunk_start + CHUNK, N)
                    chunk_indices = all_indices[chunk_start:chunk_end]

                    if trt_session:
                        # TRT: batch=1, float32
                        window_idx = chunk_indices[0]
                        window = torch.stack([
                            torch.from_numpy(cpu_frames[j])
                                 .permute(2, 0, 1)
                                 .to(dtype=torch.float32, device=self.device) / 255.0
                            for j in window_idx
                        ]).unsqueeze(0)                        # (1, 7, 3, H, W)
                        out_batch = trt_session.infer(window)  # (1, 3, H*3, W*3)
                        del window
                    else:
                        # PyTorch: chunked mini-batch
                        needed    = sorted({idx for w in chunk_indices for idx in w})
                        idx_map   = {g: l for l, g in enumerate(needed)}
                        src = torch.stack([
                            torch.from_numpy(cpu_frames[j])
                                 .permute(2, 0, 1)
                                 .to(dtype=dtype, device=self.device) / 255.0
                            for j in needed
                        ])
                        local_idx = torch.tensor(
                            [[idx_map[g] for g in w] for w in chunk_indices],
                            device=self.device,
                        )
                        out_batch = infer_model(src[local_idx])  # (CHUNK, 3, H*3, W*3)
                        del src, local_idx

                    # uint8-Konvertierung auf GPU, dann als BGR-Bytes in Pipe schreiben
                    for k in range(out_batch.shape[0]):
                        rgb = (
                            out_batch[k].permute(1, 2, 0)
                                        .clamp(0.0, 1.0).mul(255)
                                        .to(torch.uint8).cpu().numpy()
                        )
                        combine_proc.stdin.write(
                            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).tobytes()
                        )
                    del out_batch

                    n_done = chunk_end
                    if n_done % 50 == 0 or n_done == N:
                        elapsed = time_module.time() - t_start
                        fps     = n_done / elapsed if elapsed > 0 else 0
                        remain  = (N - n_done) / fps if fps > 0 else 0
                        remain_str = (f"~{remain:.0f}s" if remain < 60
                                      else f"~{remain / 60:.1f} min")
                        print(f"       [{n_done:4d}/{N}]  {mode}  {fps:.1f} fps"
                              f"  ⏱ noch {remain_str}")

            # Pipe schließen → ffmpeg beendet die Kodierung
            combine_proc.stdin.close()
            _, stderr_bytes = combine_proc.communicate()
            retcode = combine_proc.returncode
            combine_proc = None

            if retcode != 0:
                stderr_msg = stderr_bytes.decode(errors='replace').strip() if stderr_bytes else ''
                print(f"       ❌ ffmpeg combine fehlgeschlagen (returncode={retcode})")
                if stderr_msg:
                    print(f"          {stderr_msg}")
                return False

            total_sec  = time_module.time() - t_start
            avg_fps    = N / total_sec if total_sec > 0 else 0
            output_mb  = os.path.getsize(output_video) / (1024 * 1024)

            print(f"\n  ✅ VSR comparison complete!")
            print(f"     {N} Frames  |  ⌀ {avg_fps:.1f} fps ({mode})"
                  f"  |  Gesamt {total_sec / 60:.1f} min")
            print(f"     Output: {output_video}  ({output_mb:.1f} MB)")
            print(f"     Auflösung: {W_out}×{H_out}  (FFmpeg links | VSR rechts)")
            return True

        except KeyboardInterrupt:
            print(f"\n  ⚠️  Abgebrochen")
            return False
        except Exception as e:
            print(f"  ❌ VSR comparison fehlgeschlagen: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Pipe sauber schließen falls noch offen (z.B. nach Exception)
            if combine_proc is not None:
                try:
                    combine_proc.stdin.close()
                except Exception:
                    pass
                combine_proc.terminate()
            # Temp-Verzeichnis (PNGs + ffmpeg_upscale.mkv) aufräumen
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════

def process_video(video_path: str, output_video: str = None, apply_corrections: bool = True,
                  use_vsr: bool = False, vsr_checkpoint: str = None, device: str = 'cuda:0'):
    """Complete workflow: Analyze → Correct → Compare"""
    
    print(f"\n{'='*80}")
    print(f"Processing: {Path(video_path).name}")
    print(f"{'='*80}\n")
    
    # Phase 1: Analyze
    print("📊 ANALYSIS:")
    analysis = analyze_video(video_path)
    
    print(f"  Status: {analysis['status']}")
    print(f"  FPS: {analysis.get('fps', 'N/A')}")
    print(f"  Resolution: {analysis.get('resolution', 'N/A')}")
    
    if analysis['status'] == 'error':
        print(f"  ❌ Error: {analysis.get('reason', 'Unknown')}\n")
        return False
    
    if analysis['status'] == 'non_dvd':
        print(f"  ⏭️  Skipped: Not DVD resolution\n")
        return False
    
    if analysis['status'] == 'yadif_good':
        print(f"  ✅ Unique frames ({100 - analysis['redundancy_ratio']:.1f}%)")
        print(f"  ⏭️  No correction needed\n")
        
        # Still do VSR comparison if requested (for reference)
        if use_vsr and vsr_checkpoint:
            print("🎨 VSR COMPARISON (reference):")
            comparator = VSRComparator(vsr_checkpoint, device=device)
            if not output_video:
                output_video = str(video_path).replace('.mkv', '_VSR_COMPARISON.mkv')
            return comparator.create_comparison(video_path, output_video)
        return True
    
    if analysis['status'] == 'interlaced':
        print(f"  ⚠️  INTERLACED SOURCE")
        correction = 'deinterlace'
    else:
        print(f"  ⚠️  {analysis['status'].upper()}")
        print(f"  Redundancy: {analysis['redundancy_ratio']:.1f}%")
        print(f"  Avg Similarity: {analysis['avg_similarity']*100:.1f}%")
        correction = 'deduplicate'
    
    # Phase 2: Correct
    if not apply_corrections:
        print(f"  ℹ️  Correction disabled (dry-run)\n")
        return True
    
    if not output_video:
        output_video = str(video_path).replace('.mkv', '_CORRECTED.mkv')
    
    if not correct_video(video_path, output_video, correction):
        return False
    
    # Phase 3: VSR Comparison (optional)
    if use_vsr and vsr_checkpoint and Path(vsr_checkpoint).exists():
        comparison_video = str(video_path).replace('.mkv', '_COMPARISON_SPLIT.mkv')
        comparator = VSRComparator(vsr_checkpoint, device=device)
        comparator.create_comparison(output_video, comparison_video)
    
    print()
    return True


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print(f"""
Usage: python vergleich.py <video_path> [output_path] [options]

Options:
  --no-correct          Dry-run (analyze only)
  --vsr <checkpoint>    Enable VSR comparison (requires checkpoint.pth)
  --gpu <index>         GPU device index (default: interaktive Auswahl)

Examples:
  # Analyze and correct, save to custom output
  python vergleich.py input.mkv output.mkv

  # Analyze only (dry-run)
  python vergleich.py input.mkv --no-correct

  # With VSR comparison
  python vergleich.py input.mkv output.mkv --vsr checkpoint_best.pth

  # VSR comparison, GPU 1 direkt angeben (ohne interaktive Auswahl)
  python vergleich.py input.mkv output.mkv --vsr checkpoint_best.pth --gpu 1
        """)
        sys.exit(1)

    video_path = sys.argv[1]
    output_video = None
    apply_corrections = '--no-correct' not in sys.argv
    # --checkpoint is accepted as an alias for --vsr
    use_vsr = '--vsr' in sys.argv or '--checkpoint' in sys.argv
    vsr_checkpoint = None

    # Parse output path (if provided and not an option)
    if len(sys.argv) > 2 and not sys.argv[2].startswith('--'):
        output_video = sys.argv[2]

    if use_vsr:
        flag = '--vsr' if '--vsr' in sys.argv else '--checkpoint'
        try:
            idx = sys.argv.index(flag)
            vsr_checkpoint = sys.argv[idx + 1]
        except (IndexError, ValueError):
            print(f"❌ {flag} requires checkpoint path")
            sys.exit(1)

    # Parse --gpu <index>
    gpu_index = None
    if '--gpu' in sys.argv:
        try:
            gpu_index = int(sys.argv[sys.argv.index('--gpu') + 1])
        except (IndexError, ValueError):
            print("❌ --gpu requires an integer index")
            sys.exit(1)

    if not Path(video_path).exists():
        print(f"❌ File not found: {video_path}")
        sys.exit(1)

    # GPU-Auswahl (interaktiv oder via --gpu)
    device = select_gpu(gpu_index)
    device_str = str(device)  # z.B. 'cuda:0'

    # Wiederholungsbefehl ausgeben
    args = list(sys.argv)
    # Alias --checkpoint → --vsr normalisieren, damit der Hinweis korrekt ist
    if '--checkpoint' in args:
        args[args.index('--checkpoint')] = '--vsr'
    # --gpu bereits vorhanden? Wert aktualisieren; sonst anhängen
    if '--gpu' in args:
        args[args.index('--gpu') + 1] = str(device.index if device.type == 'cuda' else 0)
    elif device.type == 'cuda':
        args.extend(['--gpu', str(device.index if device.index is not None else 0)])
    print(f"{C_CYAN}💡 Nächstes Mal ohne Auswahl:{C_RESET}")
    # Pfade mit Leerzeichen oder Sonderzeichen in Anführungszeichen
    quoted = []
    for a in args:
        quoted.append(f'"{a}"' if (' ' in a or '(' in a or ')' in a) else a)
    print(f"   {' '.join(quoted)}\n")

    success = process_video(
        video_path,
        output_video=output_video,
        apply_corrections=apply_corrections,
        use_vsr=use_vsr,
        vsr_checkpoint=vsr_checkpoint,
        device=device_str,
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
