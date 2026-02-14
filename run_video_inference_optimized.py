#!/usr/bin/env python3
"""
Optimized Video Inference Script - Video-Verarbeitung mit optimiertem Modell

Unterstützt:
- TensorRT Engines (FP16/FP32)
- TorchScript Modelle
- ONNX Modelle
- Original PyTorch Checkpoints

Verwendung:
    # Mit TensorRT Engine
    python run_video_inference_optimized.py --model model_trt_fp16.engine --input video.mkv --output result.mkv
    
    # Mit TorchScript
    python run_video_inference_optimized.py --model model_scripted.pt --input video.mkv --output result.mkv
    
    # Mit ONNX
    python run_video_inference_optimized.py --model model.onnx --input video.mkv --output result.mkv
    
    # Mit original Checkpoint (automatische Format-Erkennung)
    python run_video_inference_optimized.py --model model.pth --input video.mkv --output result.mkv
"""

import argparse
import os
import sys
import subprocess
import tempfile
import time
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add vsr_plusplus_NEU to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vsr_plusplus_NEU'))


def detect_model_format(model_path):
    """
    Erkennt das Modell-Format anhand der Dateiendung
    
    Returns:
        format: 'tensorrt', 'torchscript', 'onnx', oder 'pytorch'
    """
    ext = os.path.splitext(model_path)[1].lower()
    
    if ext in ['.engine', '.trt']:
        return 'tensorrt'
    elif ext == '.pt' or ext == '.pts':
        # Könnte TorchScript oder PyTorch sein
        # Versuche zu laden
        try:
            torch.jit.load(model_path)
            return 'torchscript'
        except:
            return 'pytorch'
    elif ext == '.onnx':
        return 'onnx'
    elif ext == '.pth':
        return 'pytorch'
    else:
        raise ValueError(f"Unbekanntes Modell-Format: {ext}")


def load_tensorrt_model(model_path, device='cuda'):
    """
    Lädt TensorRT Engine
    """
    try:
        from torch2trt import TRTModule
    except ImportError:
        raise ImportError("torch2trt nicht installiert! Installieren mit: pip install torch2trt")
    
    print(f"📦 Lade TensorRT Engine: {model_path}")
    
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(model_path))
    
    print(f"✅ TensorRT Engine geladen")
    
    return model_trt, {'format': 'tensorrt'}


def load_torchscript_model(model_path, device='cuda'):
    """
    Lädt TorchScript Modell
    """
    print(f"📦 Lade TorchScript Modell: {model_path}")
    
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    
    print(f"✅ TorchScript Modell geladen")
    
    return model, {'format': 'torchscript'}


def load_onnx_model(model_path, device='cuda'):
    """
    Lädt ONNX Modell mit ONNXRuntime
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime nicht installiert! Installieren mit: pip install onnxruntime-gpu")
    
    print(f"📦 Lade ONNX Modell: {model_path}")
    
    # ONNX Runtime Session erstellen
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    
    session = ort.InferenceSession(model_path, providers=providers)
    
    print(f"✅ ONNX Modell geladen")
    print(f"   Provider: {session.get_providers()[0]}")
    
    return session, {'format': 'onnx'}


def load_pytorch_model(model_path, device='cuda'):
    """
    Lädt Original PyTorch Checkpoint
    """
    print(f"📦 Lade PyTorch Checkpoint: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Modell-Konfiguration
    n_feats = 72
    n_blocks = 28
    
    try:
        import config as cfg
        config = cfg.get_config()
        n_feats = config.get('N_FEATS', 72)
        n_blocks = config.get('N_BLOCKS', 28)
    except:
        pass
    
    from vsr_plusplus_NEU.core.model_7frame import VSRBidirectional_7frames_3x
    model = VSRBidirectional_7frames_3x(
        n_feats=n_feats,
        n_blocks=n_blocks
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    checkpoint_info = {
        'format': 'pytorch',
        'step': checkpoint.get('step', 'unknown'),
        'epoch': checkpoint.get('epoch', 'unknown')
    }
    
    print(f"✅ PyTorch Modell geladen (Step: {checkpoint_info['step']}, Epoch: {checkpoint_info['epoch']})")
    
    return model, checkpoint_info


def load_model(model_path, device='cuda'):
    """
    Lädt Modell basierend auf automatischer Format-Erkennung
    """
    model_format = detect_model_format(model_path)
    
    print(f"🔍 Erkanntes Format: {model_format}")
    
    if model_format == 'tensorrt':
        return load_tensorrt_model(model_path, device)
    elif model_format == 'torchscript':
        return load_torchscript_model(model_path, device)
    elif model_format == 'onnx':
        return load_onnx_model(model_path, device)
    elif model_format == 'pytorch':
        return load_pytorch_model(model_path, device)
    else:
        raise ValueError(f"Unbekanntes Format: {model_format}")


def get_video_resolution(video_path):
    """Ermittelt Video-Auflösung"""
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=p=0',
        video_path
    ]
    
    try:
        output = subprocess.check_output(probe_cmd, stderr=subprocess.DEVNULL).decode().strip()
        width, height = map(int, output.split(','))
        return width, height
    except Exception as e:
        raise ValueError(f"Konnte Video-Auflösung nicht ermitteln: {e}")


def extract_frames_from_video(video_path, output_dir, scale_factor=None):
    """Extrahiert Frames aus Video"""
    print(f"📹 Extrahiere Frames aus Video...")
    
    orig_width, orig_height = get_video_resolution(video_path)
    print(f"   Original-Auflösung: {orig_width}×{orig_height}")
    
    if scale_factor is not None:
        target_width = int(orig_width * scale_factor)
        target_height = int(orig_height * scale_factor)
        print(f"   Skaliere auf: {target_width}×{target_height}")
    else:
        target_width = orig_width
        target_height = orig_height
        print(f"   Behalte Original-Auflösung bei")
    
    # FPS auslesen
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    
    try:
        fps_str = subprocess.check_output(probe_cmd, stderr=subprocess.DEVNULL).decode().strip()
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)
        print(f"   Video FPS: {fps:.2f}")
    except:
        fps = 24.0
        print(f"   ⚠️  Konnte FPS nicht auslesen, verwende {fps}")
    
    # Frames extrahieren
    if scale_factor is not None:
        extract_cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', f'scale={target_width}:{target_height}:flags=lanczos',
            '-q:v', '1',
            os.path.join(output_dir, 'frame_%06d.png')
        ]
    else:
        extract_cmd = [
            'ffmpeg', '-i', video_path,
            '-q:v', '1',
            os.path.join(output_dir, 'frame_%06d.png')
        ]
    
    try:
        subprocess.run(extract_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg Fehler: {e.stderr.decode()}")
        raise
    
    frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
    print(f"✅ {len(frame_files)} Frames extrahiert")
    
    return frame_files, fps, (target_width, target_height)


def process_frame_pytorch(model, frames_tensor, device):
    """Verarbeitet Frame mit PyTorch/TorchScript Modell"""
    with torch.no_grad():
        frames_tensor = frames_tensor.to(device)
        output = model(frames_tensor)
    return output


def process_frame_onnx(session, frames_tensor):
    """Verarbeitet Frame mit ONNX Runtime"""
    # ONNX Runtime erwartet numpy array
    input_numpy = frames_tensor.cpu().numpy()
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    result = session.run([output_name], {input_name: input_numpy})
    
    # Zurück zu torch tensor
    output = torch.from_numpy(result[0])
    return output


def process_frames_with_model(model, model_info, frames_dir, frame_files, output_dir, device='cuda'):
    """
    Verarbeitet Frames mit optimiertem Modell
    """
    print(f"🔄 Verarbeite Frames mit optimiertem Modell...")
    print(f"   Format: {model_info['format']}")
    
    total_frames = len(frame_files)
    
    if total_frames < 7:
        raise ValueError(f"Zu wenige Frames ({total_frames}). Mindestens 7 Frames benötigt.")
    
    processed_count = 0
    total_time = 0
    
    # Format-spezifische Verarbeitung
    is_onnx = (model_info['format'] == 'onnx')
    
    for i in tqdm(range(3, total_frames - 3), desc="Processing frames"):
        start_time = time.time()
        
        # Lade 7 Frames
        window_frames = []
        
        for offset in range(-3, 4):
            frame_path = os.path.join(frames_dir, frame_files[i + offset])
            frame = cv2.imread(frame_path)
            
            if frame is None:
                raise ValueError(f"Konnte Frame nicht laden: {frame_files[i + offset]}")
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            window_frames.append(frame)
        
        # Stack frames
        frames_tensor = torch.from_numpy(np.stack(window_frames))
        frames_tensor = frames_tensor.permute(0, 3, 1, 2).unsqueeze(0)
        
        # Durch Modell laufen lassen
        if is_onnx:
            output = process_frame_onnx(model, frames_tensor)
        else:
            output = process_frame_pytorch(model, frames_tensor, device)
        
        # Output zu Bild konvertieren
        output_img = output[0].cpu().permute(1, 2, 0).numpy()
        output_img = np.clip(output_img * 255.0, 0, 255).astype(np.uint8)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        
        # Speichern
        output_path = os.path.join(output_dir, f'frame_{i-3:06d}.png')
        cv2.imwrite(output_path, output_img)
        
        processed_count += 1
        total_time += time.time() - start_time
    
    avg_time = total_time / processed_count * 1000  # ms
    fps = 1000 / avg_time
    
    print(f"✅ {processed_count} Frames verarbeitet")
    print(f"⏱️  Durchschnitt: {avg_time:.2f} ms/Frame ({fps:.2f} FPS)")
    
    return processed_count


def create_video_from_frames(frames_dir, output_path, input_video_path, fps=24):
    """Erstellt Video aus Frames"""
    print(f"🎬 Erstelle Video aus Frames...")
    
    import tempfile
    temp_video = tempfile.mktemp(suffix='.mkv')
    
    try:
        # Video ohne Audio erstellen
        create_cmd = [
            'ffmpeg',
            '-framerate', str(fps),
            '-i', os.path.join(frames_dir, 'frame_%06d.png'),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-y', temp_video
        ]
        
        subprocess.run(create_cmd, check=True, capture_output=True)
        print(f"   ✅ Video ohne Audio erstellt")
        
        # Audio mergen
        merge_cmd = [
            'ffmpeg',
            '-i', temp_video,
            '-i', input_video_path,
            '-map', '0:v:0',
            '-map', '1:a?',
            '-c:v', 'copy',
            '-c:a', 'copy',
            '-y', output_path
        ]
        
        try:
            subprocess.run(merge_cmd, check=True, capture_output=True)
            print(f"   ✅ Audio gemerged")
        except subprocess.CalledProcessError:
            print(f"   ⚠️  Konnte Audio nicht mergen, speichere Video ohne Audio")
            import shutil
            shutil.copy(temp_video, output_path)
        
    finally:
        if os.path.exists(temp_video):
            os.remove(temp_video)
    
    print(f"✅ Video erstellt: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Video Inference mit optimiertem Modell',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Mit TensorRT Engine
  python run_video_inference_optimized.py --model model_trt_fp16.engine --input video.mkv --output result.mkv
  
  # Mit TorchScript
  python run_video_inference_optimized.py --model model_scripted.pt --input video.mkv --output result.mkv
  
  # Mit ONNX
  python run_video_inference_optimized.py --model model.onnx --input video.mkv --output result.mkv
        """
    )
    
    parser.add_argument('--model', '-m', required=True,
                        help='Pfad zum optimierten Modell')
    parser.add_argument('--input', '-i', required=True,
                        help='Pfad zum Input-Video')
    parser.add_argument('--output', '-o', required=True,
                        help='Pfad zum Output-Video')
    parser.add_argument('--device', '-d', choices=['cuda', 'cpu'],
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (Standard: cuda falls verfügbar)')
    parser.add_argument('--framerate', '-f', type=float,
                        help='Output FPS (Standard: wie Input-Video)')
    
    args = parser.parse_args()
    
    # Header
    print("=" * 70)
    print("🚀 Optimized Video Inference")
    print("=" * 70)
    print(f"Modell: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print("=" * 70)
    
    # Validierung
    if not os.path.exists(args.model):
        print(f"❌ Modell nicht gefunden: {args.model}")
        return 1
    
    if not os.path.exists(args.input):
        print(f"❌ Input-Video nicht gefunden: {args.input}")
        return 1
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA nicht verfügbar, verwende CPU")
        args.device = 'cpu'
    
    # Temporäres Verzeichnis
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\n📂 Arbeitsverzeichnis: {temp_dir}\n")
        
        frames_dir = os.path.join(temp_dir, 'input_frames')
        output_frames_dir = os.path.join(temp_dir, 'output_frames')
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(output_frames_dir, exist_ok=True)
        
        try:
            # Schritt 1: Modell laden
            model, model_info = load_model(args.model, args.device)
            print()
            
            # Schritt 2: Frames extrahieren
            frame_files, video_fps, input_resolution = extract_frames_from_video(args.input, frames_dir)
            print(f"   LR Input: {input_resolution[0]}×{input_resolution[1]}")
            print(f"   HR Output wird: {input_resolution[0]*3}×{input_resolution[1]*3} (3x Upscaling)")
            print()
            
            # Output FPS
            output_fps = args.framerate if args.framerate is not None else video_fps
            print(f"🎯 Output FPS: {output_fps:.2f}\n")
            
            # Schritt 3: Frames verarbeiten
            processed_count = process_frames_with_model(
                model, model_info, frames_dir, frame_files, output_frames_dir,
                device=args.device
            )
            print()
            
            # Schritt 4: Video erstellen
            create_video_from_frames(output_frames_dir, args.output, args.input, output_fps)
            print()
            
            # Erfolg
            print("=" * 70)
            print("✅ Fertig!")
            print("=" * 70)
            print(f"📊 Statistik:")
            print(f"   Modell-Format: {model_info['format']}")
            print(f"   Frames verarbeitet: {processed_count}")
            print(f"   Output: {args.output}")
            print("=" * 70)
            
            return 0
            
        except Exception as e:
            print(f"\n❌ Fehler: {e}")
            import traceback
            traceback.print_exc()
            return 1


if __name__ == '__main__':
    sys.exit(main())
