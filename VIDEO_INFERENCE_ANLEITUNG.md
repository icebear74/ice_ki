# Video Inference Script - Anleitung

## Beschreibung

Mit `run_video_inference.py` können Sie manuell ein Video durch das trainierte VSR++ Modell laufen lassen, indem Sie einen gespeicherten Checkpoint verwenden.

## Voraussetzungen

- Python 3.8+
- PyTorch
- OpenCV (cv2)
- FFmpeg (installiert und im PATH)
- tqdm

Installation der Python-Pakete:
```bash
pip install torch torchvision opencv-python tqdm
```

## Verwendung

### Basis-Verwendung

```bash
python run_video_inference.py \
    --checkpoint /pfad/zum/checkpoint.pth \
    --input video.mkv \
    --output ergebnis.mkv
```

### Alle Parameter

```bash
python run_video_inference.py \
    --checkpoint PFAD     # Pfad zum Checkpoint (.pth Datei) [ERFORDERLICH]
    --input PFAD          # Pfad zum Input-Video [ERFORDERLICH]
    --output PFAD         # Pfad zum Output-Video [ERFORDERLICH]
    --device auto|cuda|cpu # Device (Standard: auto - CUDA falls verfügbar)
    --batch-size N        # Batch Size (Standard: 1)
    --framerate FPS       # FPS für Output (Standard: wie Input)
```

## Beispiele

### 1. Einfache Video-Verarbeitung

```bash
python run_video_inference.py \
    --checkpoint checkpoints/checkpoint_step_50000.pth \
    --input testvideo.mkv \
    --output testvideo_upscaled.mkv
```

### 2. Mit CPU statt GPU

Wenn Sie keine CUDA-GPU haben oder CPU verwenden möchten:

```bash
python run_video_inference.py \
    --checkpoint checkpoints/best.pth \
    --input video.mp4 \
    --output video_upscaled.mp4 \
    --device cpu
```

### 3. Mit anderer Framerate

Output mit 30 FPS statt der Original-Framerate:

```bash
python run_video_inference.py \
    --checkpoint checkpoints/checkpoint_step_100000.pth \
    --input video.mkv \
    --output video_30fps.mkv \
    --framerate 30
```

### 4. Mit DATA_ROOT Pfad

Wenn Ihr Checkpoint im Learn-Verzeichnis liegt:

```bash
python run_video_inference.py \
    --checkpoint /mnt/data/training/Universal/Mastermodell/Learn/checkpoints/checkpoint_step_75000.pth \
    --input /mnt/data/training/Universal/Mastermodell/Learn/testvideo.mkv \
    --output /mnt/data/training/Universal/Mastermodell/Learn/testvideo_upscaled.mkv
```

## Funktionsweise

Das Script:

1. **Lädt den Checkpoint** und erstellt das Modell mit der richtigen Konfiguration
2. **Extrahiert Frames** aus dem Input-Video mit FFmpeg (skaliert auf 180x180 für LR)
3. **Verarbeitet Frames** mit Sliding Window (5 Frames → 1 Frame, 3x Upscaling zu 540x540)
4. **Erstellt das Output-Video** aus den verarbeiteten Frames
5. **Merged Audio und Metadata** vom Original-Video in das Output-Video

## Technische Details

### Sliding Window
- Das Modell benötigt 5 aufeinanderfolgende Frames
- Der mittlere Frame (Index 2) wird hochskaliert
- Die ersten 2 und letzten 2 Frames des Videos werden übersprungen

### Upscaling
- Input: 180x180 Pixel (LR)
- Output: 540x540 Pixel (HR)
- Faktor: 3x

### Video-Formate
- Unterstützt alle von FFmpeg unterstützten Formate (mkv, mp4, avi, etc.)
- Output wird als H.264 mit hoher Qualität (CRF 18) kodiert

## Output

Das Script gibt detaillierte Informationen aus:

```
======================================================================
🎬 VSR++ Video Inference
======================================================================
📁 Checkpoint: checkpoints/checkpoint_step_50000.pth
📹 Input:      testvideo.mkv
💾 Output:     testvideo_upscaled.mkv
🖥️  Device:     cuda
======================================================================

📂 Arbeitsverzeichnis: /tmp/tmpXXXXXX

📦 Lade Checkpoint: checkpoints/checkpoint_step_50000.pth
   Modell-Konfiguration: n_feats=128, n_blocks=32
✅ Modell geladen (Step: 50000, Epoch: 10)

📹 Extrahiere Frames aus Video...
   Video FPS: 24.00
✅ 240 Frames extrahiert

🎯 Output FPS: 24.00

🔄 Verarbeite Frames mit Modell...
Processing frames: 100%|████████████████| 236/236 [00:45<00:00,  5.18it/s]
✅ 236 Frames verarbeitet

🎞️  Erstelle Output-Video...
   ✅ Video ohne Audio erstellt
   ✅ Audio gemerged
✅ Video gespeichert: testvideo_upscaled.mkv

======================================================================
✅ Fertig!
======================================================================
📊 Statistik:
   Checkpoint: Step 50000, Epoch 10
   Frames verarbeitet: 236
   Output: testvideo_upscaled.mkv
======================================================================
```

## Tipps

### Checkpoint finden

Checkpoints werden normalerweise hier gespeichert:
```bash
ls -lh /mnt/data/training/Universal/Mastermodell/Learn/checkpoints/
```

Oder im Projekt-Verzeichnis:
```bash
ls -lh ./Learn/checkpoints/
```

### Bester Checkpoint

Der beste Checkpoint ist meist:
- `best_checkpoint.pth` (automatisch gespeicherter bester Checkpoint)
- oder der mit dem höchsten Step-Count

### Performance

- **GPU**: ~5-10 Frames/Sekunde (abhängig von GPU)
- **CPU**: ~0.5-2 Frames/Sekunde (sehr langsam!)

→ GPU wird stark empfohlen!

### VRAM-Probleme

Falls Sie CUDA Out-of-Memory Fehler bekommen:
- Verwenden Sie `--device cpu`
- Oder verarbeiten Sie ein kürzeres Video

## Fehlerbehandlung

### "Checkpoint nicht gefunden"
```bash
# Überprüfen Sie den Pfad
ls -lh /pfad/zum/checkpoint.pth
```

### "FFmpeg not found"
```bash
# FFmpeg installieren
sudo apt-get install ffmpeg  # Ubuntu/Debian
brew install ffmpeg          # macOS
```

### "No module named 'torch'"
```bash
# PyTorch installieren
pip install torch torchvision
```

### "CUDA out of memory"
```bash
# CPU verwenden statt GPU
python run_video_inference.py ... --device cpu
```

## Integration mit Web UI

Das Web UI Feature nutzt intern eine ähnliche Logik. Mit diesem Script können Sie:
- Videos offline verarbeiten
- Verschiedene Checkpoints testen
- Batch-Verarbeitung durchführen

## Support

Bei Problemen:
1. Überprüfen Sie, dass alle Voraussetzungen installiert sind
2. Testen Sie mit einem kurzen Video (< 10 Sekunden)
3. Überprüfen Sie die Checkpoint-Datei
4. Verwenden Sie `--device cpu` zum Debuggen
