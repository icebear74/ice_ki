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
import torch
from collections import defaultdict
import time as time_module


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

            # Model uses 24 blocks (12 backward + 12 forward)
            n_blocks = 24
            n_feats = 72

            # Let cuDNN auto-select the fastest conv kernels for the fixed input size
            torch.backends.cudnn.benchmark = True

            print(f"  Loading VSR model (n_blocks={n_blocks}, n_feats={n_feats})...")
            self.model = VSRBidirectional_7frames_3x(n_feats=n_feats, n_blocks=n_blocks)

            # Load checkpoint to CPU first — avoids doubling GPU peak memory during load
            print(f"  Loading checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            del ckpt  # free CPU RAM immediately

            # Move to GPU, switch to fp16 (2× throughput on CUDA), set eval mode
            self.model = self.model.to(self.device).half().eval()
            self.use_fp16 = True

            # torch.compile: fuses ops and removes Python overhead (PyTorch >= 2.0)
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print(f"  ✅ torch.compile enabled (reduce-overhead)")
            except Exception:
                pass  # graceful fallback on older PyTorch

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
            
            # ────────────────────────────────────────────────────────────
            # Phase 2: FFmpeg upscale - ONLY the extracted frames segment
            # ────────────────────────────────────────────────────────────
            print(f"  [2/5] FFmpeg upscaling (3x to 2160×1728)...")
            print(f"       Processing {len(frames)}-frame segment...")
            
            ffmpeg_cmd = [
                'ffmpeg', '-hwaccel', 'cuda', '-loglevel', 'error',
                '-framerate', '25',
                '-pattern_type', 'glob', '-i', f'{temp_dir}/frame_*.png',
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
            # Phase 3: VSR upscale - 7-frame model with sliding window
            # ────────────────────────────────────────────────────────────
            print(f"  [3/5] VSR upscaling (7-frame model, {len(frames)} frames)...")
            print(f"       Loading frames into 7-frame buffer...")
            
            vsr_frames = []
            
            # Load all frames first
            all_frames = []
            for frame_path in frames:
                frame = cv2.imread(str(frame_path))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                all_frames.append(frame)
            
            print(f"       Loaded {len(all_frames)} frames")
            
            # Process with 7-frame sliding window
            with torch.no_grad():
                for i in range(len(all_frames)):
                    # Build 7-frame window: [center-3, center-2, center-1, center, center+1, center+2, center+3]
                    center_idx = i
                    
                    frame_indices = []
                    for offset in [-3, -2, -1, 0, 1, 2, 3]:
                        idx = center_idx + offset
                        # Clamp to valid range
                        if idx < 0:
                            idx = 0
                        elif idx >= len(all_frames):
                            idx = len(all_frames) - 1
                        frame_indices.append(idx)
                    
                    # Stack 7 frames: shape [1, 7, 3, H, W] as expected by model.forward
                    frames_tensors = [
                        torch.from_numpy(all_frames[idx]).float().permute(2, 0, 1) / 255.0
                        for idx in frame_indices
                    ]
                    frame_t = torch.stack(frames_tensors, dim=0).unsqueeze(0)  # (1, 7, 3, H, W)
                    
                    if i % max(1, len(all_frames)//4) == 0 or i == len(all_frames) - 1:
                        print(f"       [{i+1}/{len(all_frames)}] Processing frame {i+1}...")
                    
                    try:
                        # Model input: (1, 21, H, W) - 7 frames * 3 channels
                        vsr_output = self.model(frame_t.to(self.device))
                        # Output: (1, 3, H*3, W*3)
                        vsr_output = (vsr_output.squeeze(0).permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
                        vsr_frames.append(vsr_output)
                    except Exception as e:
                        print(f"       ❌ VSR inference failed on frame {i}:")
                        print(f"          Input shape: {frame_t.shape} (expected: 1, 7, 3, H, W)")
                        print(f"          Error: {e}")
                        import traceback
                        traceback.print_exc()
                        return False
            
            print(f"       ✓ VSR done ({len(vsr_frames)} frames processed)")
            
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
            print(f"  [5/5] Creating split-screen comparison...")
            combine_cmd = [
                'ffmpeg', '-hwaccel', 'cuda', '-loglevel', 'error',
                '-i', f'{temp_dir}/ffmpeg_upscale.mkv',
                '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '2160x1728', '-r', '25',
                '-i', vsr_raw,
                '-filter_complex', '''
                    [0:v]crop=1080:1728:0:0[left];
                    [1:v]crop=1080:1728:1080:0[right];
                    [left][right]hstack[combined];
                    [combined]drawtext=text='FFmpeg Upscale (x3)':fontsize=60:fontcolor=white:x=50:y=50:box=1:boxcolor=black@0.5,
                    drawtext=text='VSR Model (x3)':fontsize=60:fontcolor=white:x=1130:y=50:box=1:boxcolor=black@0.5[out]
                ''',
                '-map', '[out]',
                '-c:v', 'hevc_nvenc', '-preset', 'medium', '-crf', '18',
                output_video
            ]
            
            start_time = time_module.time()
            try:
                result = subprocess.run(
                    combine_cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                elapsed = time_module.time() - start_time
                
                if result.returncode != 0:
                    print(f"       ❌ Combine failed: {result.stderr}")
                    return False
                
                output_size_mb = os.path.getsize(output_video) / (1024 * 1024)
                print(f"       ✓ Done ({elapsed:.1f}s, {output_size_mb:.1f} MB)")
                
            except subprocess.TimeoutExpired:
                elapsed = time_module.time() - start_time
                print(f"       ❌ Timeout after {elapsed:.1f}s")
                return False
            
            print(f"  ✅ VSR comparison complete!")
            print(f"     Output: {output_video}")
            print(f"     Resolution: 2880×1728 (split-screen)")
            
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
                  use_vsr: bool = False, vsr_checkpoint: str = None):
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
            comparator = VSRComparator(vsr_checkpoint)
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
        comparator = VSRComparator(vsr_checkpoint)
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
  --gpu <index>         GPU device (default: 0)

Examples:
  # Analyze and correct, save to custom output
  python vergleich.py input.mkv output.mkv
  
  # Analyze only (dry-run)
  python vergleich.py input.mkv --no-correct
  
  # With VSR comparison
  python vergleich.py input.mkv output.mkv --vsr checkpoint_best.pth
  
  # VSR comparison without correction
  python vergleich.py input.mkv output_comparison.mkv --no-correct --vsr checkpoint_best.pth
        """)
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_video = None
    apply_corrections = '--no-correct' not in sys.argv
    use_vsr = '--vsr' in sys.argv
    vsr_checkpoint = None
    
    # Parse output path (if provided and not an option)
    if len(sys.argv) > 2 and not sys.argv[2].startswith('--'):
        output_video = sys.argv[2]
    
    if use_vsr:
        try:
            idx = sys.argv.index('--vsr')
            vsr_checkpoint = sys.argv[idx + 1]
        except (IndexError, ValueError):
            print("❌ --vsr requires checkpoint path")
            sys.exit(1)
    
    if not Path(video_path).exists():
        print(f"❌ File not found: {video_path}")
        sys.exit(1)
    
    success = process_video(
        video_path,
        output_video=output_video,
        apply_corrections=apply_corrections,
        use_vsr=use_vsr,
        vsr_checkpoint=vsr_checkpoint
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
