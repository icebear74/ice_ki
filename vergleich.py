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

class VSRComparator:
    def __init__(self, checkpoint_path: str, device: str = 'cuda:0'):
        """Initialize VSR model with 24 blocks — optimized for inference"""
        self.device = torch.device(device)
        self.available = False
        self.use_fp16 = False

        try:
            from vsr_plusplus_NEU.core.model_7frame import VSRBidirectional_7frames_3x

            # Model architecture — read from config.py if available, else use defaults
            n_blocks = 24
            n_feats  = 72
            n_frames = 7
            try:
                import config as _cfg
                _c = _cfg.get_config()
                n_feats  = _c.get("N_FEATS",  n_feats)
                n_blocks = _c.get("N_BLOCKS", n_blocks)
                n_frames = _c.get("n_frames", n_frames)
            except Exception:
                pass

            # Let cuDNN auto-select the fastest conv kernels for the fixed input size
            torch.backends.cudnn.benchmark = True

            print(f"  Loading VSR model (n_blocks={n_blocks}, n_feats={n_feats}, n_frames={n_frames})...")
            self.model = VSRBidirectional_7frames_3x(n_feats=n_feats, n_blocks=n_blocks,
                                                     n_frames=n_frames)

            # Load checkpoint to CPU first — avoids doubling GPU peak memory during load
            print(f"  Loading checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
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
        """Create split-screen comparison with 7-frame model support"""
        
        if not self.available:
            print(f"  ⚠️  Skipping VSR comparison (model unavailable)")
            return False
        
        print(f"\n🎨 VSR COMPARISON:")
        temp_dir = Path("/tmp/vsr_compare")
        
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # ────────────────────────────────────────────────────────────
            # Phase 1: Extract frames
            # ────────────────────────────────────────────────────────────
            print(f"  [1/5] Extracting frames...")
            cmd = [
                'ffmpeg', '-hwaccel', 'cuda', '-loglevel', 'quiet',
                '-ss', '300', '-t', '2',
                '-i', input_video,
                f'{temp_dir}/frame_%06d.png'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"       ❌ Failed to extract frames")
                return False
            
            frames = sorted(temp_dir.glob("frame_*.png"))
            if len(frames) < 7:
                print(f"       ⚠️  Not enough frames for 7-frame model: {len(frames)} (need 7+)")
                return False
            
            print(f"       ✓ Extracted {len(frames)} frames")

            # Detect input dimensions from first frame to derive 3× output size
            _probe = cv2.imread(str(frames[0]))
            H_in, W_in = _probe.shape[:2]
            del _probe
            H_out, W_out = H_in * 3, W_in * 3
            half_w = W_out // 2
            print(f"       Input: {W_in}×{H_in} → Output: {W_out}×{H_out}")

            # ────────────────────────────────────────────────────────────
            # Phase 2: FFmpeg upscale - ONLY the extracted frames segment
            # ────────────────────────────────────────────────────────────
            print(f"  [2/5] FFmpeg upscaling (3x to {W_out}×{H_out})...")
            print(f"       Processing {len(frames)}-frame segment...")
            
            ffmpeg_cmd = [
                'ffmpeg', '-hwaccel', 'cuda', '-loglevel', 'error',
                '-framerate', '25',
                '-pattern_type', 'glob', '-i', f'{temp_dir}/frame_*.png',
                '-vf', f'scale={W_out}:{H_out}:flags=lanczos',
                '-c:v', 'hevc_nvenc', '-preset', 'medium', '-crf', '18',
                f'{temp_dir}/ffmpeg_upscale.mkv'
            ]
            
            print(f"       FFmpeg input: PNG frames (not re-decoding the full video!)")
            print(f"       Encoding: hevc_nvenc, preset=medium")
            
            start_time = time_module.time()
            try:
                result = subprocess.run(
                    ffmpeg_cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                elapsed = time_module.time() - start_time
                
                if result.returncode != 0:
                    print(f"       ❌ FFmpeg failed: {result.stderr}")
                    return False
                
                print(f"       ✓ FFmpeg done ({elapsed:.1f}s)")
                
            except subprocess.TimeoutExpired:
                elapsed = time_module.time() - start_time
                print(f"       ❌ FFmpeg timeout after {elapsed:.1f}s")
                return False
            
            # ────────────────────────────────────────────────────────────
            # Phase 3: VSR upscale - 7-frame model, fully GPU-batched
            # ────────────────────────────────────────────────────────────
            print(f"  [3/5] VSR upscaling (7-frame model, {len(frames)} frames)...")

            # --- Optimization 1: load all frames to CPU RAM as numpy (cheap) ---
            # GPU memory stays O(chunk), not O(N*7) — no OOM on large videos
            print(f"       Loading {len(frames)} frames to CPU RAM...")
            cpu_frames = []
            for fp in frames:
                bgr = cv2.imread(str(fp))
                cpu_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))  # HWC uint8
            N = len(cpu_frames)
            print(f"       {N} frames loaded to RAM")

            # --- Optimization 2: pre-compute all 7-frame window indices (CPU, tiny) ---
            OFFSETS = [-3, -2, -1, 0, 1, 2, 3]
            all_indices = [
                [max(0, min(N - 1, i + o)) for o in OFFSETS]
                for i in range(N)
            ]  # list[N][7]  — plain ints, no GPU needed yet

            # --- Optimization 3 & 4: streaming mini-batch inference ---
            # Per chunk: only ≤14 unique source frames are transferred to GPU (fp16).
            # GPU memory usage is constant O(chunk) regardless of video length.
            # OOM failsafe: on OutOfMemoryError the chunk size is halved and the
            # whole pass is restarted until CHUNK=1; only then we give up.
            CHUNK = 8
            MIN_CHUNK = 1
            dtype = torch.float16 if self.use_fp16 else torch.float32

            # ── torch.jit.trace fallback for CC < 7.0 (torch.compile unavailable) ──
            # Traced once per unique input resolution; retraced if resolution changes.
            # self.model (eager) is always kept intact so retracing is always possible.
            if hasattr(self, '_jit_traced_shape'):
                H_in, W_in = cpu_frames[0].shape[:2]
                if self._jit_traced_shape != (H_in, W_in):
                    print(f"       🔧 Applying torch.jit.trace+freeze (input {H_in}×{W_in})...")
                    dummy = torch.zeros(1, 7, 3, H_in, W_in, dtype=dtype, device=self.device)
                    try:
                        # Suppress TracerWarning for self.last_* diagnostic .item() calls.
                        # These attributes are pure monitoring side-effects and do not
                        # affect the output tensor, so the trace is correct.
                        with torch.no_grad(), warnings.catch_warnings():
                            warnings.simplefilter("ignore", torch.jit.TracerWarning)
                            traced = torch.jit.trace(self.model, dummy)
                            self._jit_model = torch.jit.freeze(traced)
                        self._jit_traced_shape = (H_in, W_in)
                        print(f"       ✅ torch.jit.trace+freeze ready")
                    except Exception as _jit_ex:
                        print(f"       ⚠️  JIT trace failed ({_jit_ex}), falling back to eager mode")
                        self._jit_model = self.model
                        self._jit_traced_shape = (H_in, W_in)  # don't retry
                infer_model = self._jit_model
            else:
                infer_model = self.model

            vsr_frames = []
            while CHUNK >= MIN_CHUNK:
                vsr_frames = []
                oom_hit = False
                try:
                    with torch.no_grad():
                        for start in range(0, N, CHUNK):
                            end = min(start + CHUNK, N)
                            batch_indices = all_indices[start:end]

                            # Collect only the unique source frames this chunk needs
                            needed = sorted({idx for window in batch_indices for idx in window})
                            idx_map = {g: l for l, g in enumerate(needed)}

                            # Transfer ≤(CHUNK+6) unique frames to GPU as fp16
                            src = torch.stack([
                                torch.from_numpy(cpu_frames[i])
                                     .permute(2, 0, 1)
                                     .to(dtype=dtype, device=self.device) / 255.0
                                for i in needed
                            ])

                            # Build windows: [CHUNK, 7, 3, H, W]
                            local_idx = torch.tensor(
                                [[idx_map[g] for g in w] for w in batch_indices],
                                device=self.device
                            )
                            batch_windows = src[local_idx]
                            del src

                            out = infer_model(batch_windows)   # [CHUNK, 3, H*3, W*3]

                            # uint8 conversion on GPU (no .astype)
                            out_np = (
                                out.permute(0, 2, 3, 1)
                                   .clamp(0.0, 1.0)
                                   .mul(255)
                                   .to(torch.uint8)
                                   .cpu()
                                   .numpy()
                            )   # [CHUNK, H*3, W*3, 3]  RGB uint8
                            del out, batch_windows, local_idx
                            vsr_frames.extend(out_np)
                            print(f"       [{end}/{N}] chunk={CHUNK} frames processed...")

                    break  # success – leave retry loop

                except Exception as e:
                    is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or (
                        isinstance(e, RuntimeError) and 'out of memory' in str(e).lower()
                    )
                    if is_oom:
                        old_chunk = CHUNK
                        CHUNK = CHUNK // 2
                        torch.cuda.empty_cache()
                        gc.collect()
                        if CHUNK < MIN_CHUNK:
                            print(f"       ❌ GPU OOM at CHUNK=1 – not enough VRAM for this video.")
                            print(f"          Try a shorter clip or a GPU with more VRAM.")
                            return False
                        print(f"       {C_YELLOW}⚠ GPU OOM (CHUNK={old_chunk}) → "
                              f"retrying with CHUNK={CHUNK}...{C_RESET}")
                        oom_hit = True
                    else:
                        print(f"       ❌ VSR inference failed:")
                        print(f"          Error: {e}")
                        import traceback
                        traceback.print_exc()
                        return False

            print(f"       ✓ VSR done ({len(vsr_frames)} frames, chunk={CHUNK})")
            
            # ────────────────────────────────────────────────────────────
            # Phase 4: Write VSR raw video
            # ────────────────────────────────────────────────────────────
            print(f"  [4/5] Writing VSR raw video...")
            vsr_raw = f'{temp_dir}/vsr_upscale.raw'
            
            try:
                with open(vsr_raw, 'wb') as f:
                    for i, frame in enumerate(vsr_frames):
                        if i % max(1, len(vsr_frames)//4) == 0 or i == len(vsr_frames) - 1:
                            pass  # Silent progress
                        f.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).tobytes())
                
                file_size_mb = os.path.getsize(vsr_raw) / (1024 * 1024)
                print(f"       ✓ Raw file: {file_size_mb:.1f} MB")
            except Exception as e:
                print(f"       ❌ Failed to write raw frames: {e}")
                return False
            
            # ────────────────────────────────────────────────────────────
            # Phase 5: Combine split-screen
            # ────────────────────────────────────────────────────────────
            # Release all GPU tensors / cached allocations before starting
            # FFmpeg so it can create a fresh CUDA context for nvenc encode
            # without hitting CUDA_ERROR_OUT_OF_MEMORY.
            torch.cuda.empty_cache()
            gc.collect()

            print(f"  [5/5] Creating split-screen comparison...")
            # Left half  = left portion of FFmpeg upscale (input 0, x=0)
            # Right half = right portion of VSR upscale   (input 1, x=half_w)
            # White divider line at the seam + labels per side
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
                'ffmpeg', '-y',
                '-i', f'{temp_dir}/ffmpeg_upscale.mkv',
                '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                '-s', f'{W_out}x{H_out}', '-r', '25',
                '-i', vsr_raw,
                '-filter_complex', filter_complex,
                '-map', '[out]',
                '-c:v', 'hevc_nvenc', '-preset', 'medium', '-crf', '18',
                output_video
            ]
            print(f"       CMD: {' '.join(combine_cmd)}")
            
            start_time = time_module.time()
            try:
                result = subprocess.run(
                    combine_cmd,
                    timeout=600
                )
                elapsed = time_module.time() - start_time
                
                if result.returncode != 0:
                    print(f"       ❌ Combine failed (returncode={result.returncode})")
                    return False
                
                output_size_mb = os.path.getsize(output_video) / (1024 * 1024)
                print(f"       ✓ Done ({elapsed:.1f}s, {output_size_mb:.1f} MB)")
                
            except subprocess.TimeoutExpired:
                elapsed = time_module.time() - start_time
                print(f"       ❌ Timeout after {elapsed:.1f}s")
                return False
            
            print(f"  ✅ VSR comparison complete!")
            print(f"     Output: {output_video}")
            print(f"     Resolution: {W_out}×{H_out} (split-screen: FFmpeg left | VSR right)")
            
            return True
        
        except KeyboardInterrupt:
            print(f"\n  ⚠️  User interrupted")
            return False
        except Exception as e:
            print(f"  ❌ VSR comparison failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # Cleanup
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
