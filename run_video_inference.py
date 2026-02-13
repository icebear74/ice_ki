#!/usr/bin/env python3
"""
Video Inference Script - Manuelle Video-Verarbeitung mit VSR++ Modell (7 Frames)

Lädt ein gespeichertes Checkpoint und verarbeitet ein Video mit dem trainierten 7-Frame Modell.

Verwendung:
    # Interaktive Checkpoint-Auswahl (nutzt Training-Pfade)
    python run_video_inference.py --input video.mkv --output result.mkv
    
    # Oder spezifischen Checkpoint angeben
    python run_video_inference.py --checkpoint path/to/checkpoint.pth --input video.mkv --output result.mkv
    
Optionale Parameter:
    --device cuda/cpu       (Standard: cuda falls verfügbar)
    --framerate N           (Standard: wie Input-Video)
"""

import argparse
import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

# Add vsr_plusplus_NEU to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vsr_plusplus_NEU'))


def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """
    Lädt das 7-Frame Modell aus einem Checkpoint
    
    Args:
        checkpoint_path: Pfad zum Checkpoint (.pth Datei)
        device: Device für Inferenz (cuda/cpu)
    
    Returns:
        model: Geladenes Modell im eval() Modus
        checkpoint_info: Dictionary mit Checkpoint-Informationen
    """
    # Import here to allow --help to work without torch installed
    import torch
    
    print(f"📦 Lade Checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {checkpoint_path}")
    
    # Checkpoint laden
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Modell-Konfiguration aus Checkpoint extrahieren
    model_config = checkpoint.get('model_config', {})
    n_feats = model_config.get('n_feats', 128)
    n_blocks = model_config.get('n_blocks', 32)
    
    print(f"   Modell-Konfiguration: n_feats={n_feats}, n_blocks={n_blocks}")
    
    # 7-Frame Modell erstellen
    from vsr_plusplus_NEU.core.model_7frame import VSRBidirectional_7frames_3x
    model = VSRBidirectional_7frames_3x(
        n_feats=n_feats,
        n_blocks=n_blocks
    ).to(device)
    
    # State Dict laden
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Checkpoint-Informationen
    checkpoint_info = {
        'step': checkpoint.get('step', 'unknown'),
        'epoch': checkpoint.get('epoch', 'unknown'),
        'loss': checkpoint.get('loss', 'unknown'),
        'n_feats': n_feats,
        'n_blocks': n_blocks
    }
    
    print(f"✅ Modell geladen (Step: {checkpoint_info['step']}, Epoch: {checkpoint_info['epoch']})")
    
    return model, checkpoint_info


def extract_frames_from_video(video_path, output_dir, target_size=180):
    """
    Extrahiert Frames aus einem Video mit FFmpeg
    
    Args:
        video_path: Pfad zum Input-Video
        output_dir: Verzeichnis für extrahierte Frames
        target_size: Zielgröße für LR-Frames (Standard: 180x180)
    
    Returns:
        frame_files: Liste der extrahierten Frame-Dateien
        fps: Framerate des Videos
    """
    print(f"📹 Extrahiere Frames aus Video...")
    
    # Erst die Framerate auslesen
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    
    try:
        fps_str = subprocess.check_output(probe_cmd, stderr=subprocess.DEVNULL).decode().strip()
        # Parse Framerate (z.B. "24/1" oder "30000/1001")
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)
        print(f"   Video FPS: {fps:.2f}")
    except:
        fps = 24.0  # Fallback
        print(f"   ⚠️  Konnte FPS nicht auslesen, verwende {fps}")
    
    # Frames extrahieren
    extract_cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', f'scale={target_size}:{target_size}:flags=lanczos',
        '-q:v', '1',  # Hohe Qualität
        os.path.join(output_dir, 'frame_%06d.png')
    ]
    
    try:
        subprocess.run(extract_cmd, check=True, capture_output=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg Fehler: {e.stderr.decode()}")
        raise
    
    # Liste der extrahierten Frames
    frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
    print(f"✅ {len(frame_files)} Frames extrahiert")
    
    return frame_files, fps


def process_frames_with_model(model, frames_dir, frame_files, output_dir, device='cuda', batch_size=1):
    """
    Verarbeitet Frames mit dem 7-Frame Modell (Sliding Window)
    
    Args:
        model: VSR 7-Frame Modell
        frames_dir: Verzeichnis mit Input-Frames
        frame_files: Liste der Frame-Dateien
        output_dir: Verzeichnis für Output-Frames
        device: Device für Inferenz
        batch_size: Anzahl Frames parallel zu verarbeiten (aktuell nicht verwendet)
    
    Returns:
        processed_count: Anzahl verarbeiteter Frames
    """
    # Import here to allow --help to work without dependencies
    import torch
    import cv2
    import numpy as np
    from tqdm import tqdm
    
    print(f"🔄 Verarbeite Frames mit 7-Frame Modell...")
    
    total_frames = len(frame_files)
    
    if total_frames < 7:
        raise ValueError(f"Zu wenige Frames ({total_frames}). Mindestens 7 Frames benötigt.")
    
    processed_count = 0
    
    with torch.no_grad():
        # Wir brauchen 7 Frames für das Modell (mit Frame 3 als Center, Index 3)
        # Verarbeite von Frame 3 bis Frame (total-3) um immer Context zu haben
        for i in tqdm(range(3, total_frames - 3), desc="Processing frames"):
            # Lade 7 aufeinanderfolgende Frames (i-3 bis i+3, mit i als Center)
            window_frames = []
            
            for offset in range(-3, 4):  # -3, -2, -1, 0, 1, 2, 3
                frame_path = os.path.join(frames_dir, frame_files[i + offset])
                frame = cv2.imread(frame_path)
                
                if frame is None:
                    raise ValueError(f"Konnte Frame nicht laden: {frame_files[i + offset]}")
                
                # BGR zu RGB und normalisieren zu [0, 1]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                window_frames.append(frame)
            
            # Stack frames: [7, H, W, 3] -> [1, 7, 3, H, W]
            frames_tensor = torch.from_numpy(np.stack(window_frames))
            frames_tensor = frames_tensor.permute(0, 3, 1, 2).unsqueeze(0)  # [1, 7, 3, H, W]
            frames_tensor = frames_tensor.to(device)
            
            # Durch Modell laufen lassen
            output = model(frames_tensor)  # [1, 3, 540, 540]
            
            # Output zu Bild konvertieren
            output_img = output[0].cpu().permute(1, 2, 0).numpy()
            output_img = np.clip(output_img * 255.0, 0, 255).astype(np.uint8)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
            
            # Output-Frame speichern (first iteration at i=3 produces frame_000000.png)
            output_path = os.path.join(output_dir, f'frame_{i-3:06d}.png')
            cv2.imwrite(output_path, output_img)
            
            processed_count += 1
    
    print(f"✅ {processed_count} Frames verarbeitet")
    return processed_count


def create_video_from_frames(frames_dir, output_path, input_video_path, fps=24):
    """
    Erstellt Video aus Frames und merged Audio vom Original
    
    Args:
        frames_dir: Verzeichnis mit verarbeiteten Frames
        output_path: Pfad für Output-Video
        input_video_path: Pfad zum Original-Video (für Audio)
        fps: Framerate für Output-Video
    """
    print(f"🎞️  Erstelle Output-Video...")
    
    # Temporäres Video ohne Audio erstellen
    with tempfile.NamedTemporaryFile(suffix='.mkv', delete=False) as tmp_file:
        temp_video = tmp_file.name
    
    try:
        # Video aus Frames erstellen
        create_cmd = [
            'ffmpeg', '-framerate', str(fps),
            '-i', os.path.join(frames_dir, 'frame_%06d.png'),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',  # Hohe Qualität
            '-pix_fmt', 'yuv420p',
            '-y', temp_video
        ]
        
        subprocess.run(create_cmd, check=True, capture_output=True, stderr=subprocess.PIPE)
        print(f"   ✅ Video ohne Audio erstellt")
        
        # Audio vom Original mergen
        merge_cmd = [
            'ffmpeg',
            '-i', temp_video,
            '-i', input_video_path,
            '-map', '0:v:0',  # Video vom verarbeiteten
            '-map', '1:a?',   # Audio vom Original (falls vorhanden)
            '-c:v', 'copy',   # Video-Codec kopieren
            '-c:a', 'copy',   # Audio-Codec kopieren
            '-y', output_path
        ]
        
        try:
            subprocess.run(merge_cmd, check=True, capture_output=True, stderr=subprocess.PIPE)
            print(f"   ✅ Audio gemerged")
        except subprocess.CalledProcessError:
            # Falls Audio-Merge fehlschlägt, Video ohne Audio speichern
            print(f"   ⚠️  Konnte Audio nicht mergen, speichere Video ohne Audio")
            shutil.copy(temp_video, output_path)
        
    finally:
        # Temporäres Video löschen
        if os.path.exists(temp_video):
            os.unlink(temp_video)
    
    print(f"✅ Video gespeichert: {output_path}")


def main():
    # Import torch here to check if available
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False
    
    parser = argparse.ArgumentParser(
        description='Video Inference mit VSR++ 7-Frame Modell',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Interaktive Checkpoint-Auswahl (nutzt Training-Pfade aus config.py)
  python run_video_inference.py --input video.mkv --output result.mkv
  
  # Spezifischen Checkpoint angeben
  python run_video_inference.py --checkpoint checkpoints/best.pth --input video.mkv --output result.mkv
  
  # Mit CPU statt GPU
  python run_video_inference.py --input video.mkv --output result.mkv --device cpu
  
  # Mit 30 FPS Output
  python run_video_inference.py --input video.mkv --output result.mkv --framerate 30
        """
    )
    
    parser.add_argument('--checkpoint', '-c', default=None, help='Pfad zum Checkpoint (.pth Datei). Wenn nicht angegeben, interaktive Auswahl.')
    parser.add_argument('--input', '-i', required=True, help='Pfad zum Input-Video')
    parser.add_argument('--output', '-o', required=True, help='Pfad zum Output-Video')
    parser.add_argument('--device', '-d', default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Device für Inferenz (Standard: auto - nutzt CUDA falls verfügbar)')
    parser.add_argument('--framerate', '-f', type=float, default=None,
                        help='FPS für Output-Video (Standard: wie Input-Video)')
    
    args = parser.parse_args()
    
    # Check if torch is available
    if not torch_available:
        print("❌ PyTorch ist nicht installiert!")
        print("   Bitte installieren Sie PyTorch:")
        print("   pip install torch torchvision")
        return 1
    
    import torch
    
    # Checkpoint-Auswahl
    checkpoint_path = args.checkpoint
    
    if checkpoint_path is None:
        # Interaktive Auswahl
        print("\n" + "="*70)
        print("Interaktive Checkpoint-Auswahl")
        print("="*70 + "\n")
        
        try:
            from vsr_plusplus_NEU.utils.checkpoint_selector import select_checkpoint_interactive, get_checkpoint_dir_from_config
            from vsr_plusplus_NEU.systems.checkpoint_manager import CheckpointManager
            
            # Get checkpoint directory from config
            checkpoint_dir = get_checkpoint_dir_from_config()
            print(f"📁 Checkpoint-Verzeichnis: {checkpoint_dir}\n")
            
            # Create checkpoint manager
            checkpoint_mgr = CheckpointManager(checkpoint_dir)
            
            # Interactive selection
            selected_ckpt = select_checkpoint_interactive(checkpoint_mgr, auto_select_latest=False)
            
            if selected_ckpt is None:
                print("❌ Keine Checkpoints gefunden!")
                print(f"   Überprüfen Sie das Verzeichnis: {checkpoint_dir}")
                return 1
            
            checkpoint_path = selected_ckpt['path']
        
        except (ImportError, FileNotFoundError, KeyError, AttributeError) as e:
            print(f"❌ Fehler bei Checkpoint-Auswahl: {e}")
            print("   Bitte geben Sie den Checkpoint-Pfad manuell an mit --checkpoint")
            import traceback
            traceback.print_exc()
            return 1
    
    # Device auswählen
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("=" * 70)
    print("🎬 VSR++ Video Inference (7 Frames)")
    print("=" * 70)
    print(f"📁 Checkpoint: {checkpoint_path}")
    print(f"📹 Input:      {args.input}")
    print(f"💾 Output:     {args.output}")
    print(f"🖥️  Device:     {device}")
    print("=" * 70)
    
    # Validierung
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint nicht gefunden: {checkpoint_path}")
        return 1
    
    if not os.path.exists(args.input):
        print(f"❌ Input-Video nicht gefunden: {args.input}")
        return 1
    
    # Temporäres Verzeichnis für Frames
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\n📂 Arbeitsverzeichnis: {temp_dir}\n")
        
        frames_dir = os.path.join(temp_dir, 'input_frames')
        output_frames_dir = os.path.join(temp_dir, 'output_frames')
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(output_frames_dir, exist_ok=True)
        
        try:
            # Schritt 1: Modell laden
            model, checkpoint_info = load_model_from_checkpoint(checkpoint_path, device)
            print()
            
            # Schritt 2: Frames extrahieren
            frame_files, video_fps = extract_frames_from_video(args.input, frames_dir)
            print()
            
            # Framerate für Output bestimmen
            output_fps = args.framerate if args.framerate is not None else video_fps
            print(f"🎯 Output FPS: {output_fps:.2f}\n")
            
            # Schritt 3: Frames verarbeiten (7-Frame Modell)
            processed_count = process_frames_with_model(
                model, frames_dir, frame_files, output_frames_dir, 
                device=device
            )
            print()
            
            # Schritt 4: Video erstellen
            create_video_from_frames(output_frames_dir, args.output, args.input, output_fps)
            print()
            
            # Erfolg!
            print("=" * 70)
            print("✅ Fertig!")
            print("=" * 70)
            print(f"📊 Statistik:")
            print(f"   Checkpoint: Step {checkpoint_info['step']}, Epoch {checkpoint_info['epoch']}")
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
