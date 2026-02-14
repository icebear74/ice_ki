# Video Inference Script - Anleitung

## Beschreibung

Mit `run_video_inference.py` können Sie manuell ein Video durch das trainierte VSR++ **7-Frame Modell** laufen lassen, indem Sie einen gespeicherten Checkpoint verwenden.

**NEU:** Das Script nutzt nun:
- ✅ **7 Frames** (korrekt für VSR++ Modell)
- ✅ **Interaktive Checkpoint-Auswahl** (wie im Training)
- ✅ **Training-Pfade aus config.py** (DATASET_ROOT, DATA_ROOT)
- ✅ **Gemeinsames Checkpoint-Selector-Modul**

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

### Interaktive Checkpoint-Auswahl (EMPFOHLEN)

```bash
# Script nutzt Training-Pfade und zeigt alle verfügbaren Checkpoints
python run_video_inference.py --input video.mkv --output ergebnis.mkv
```

Das Script zeigt dann:
```
================================================================================
AVAILABLE CHECKPOINTS (Last 10):
================================================================================
#    Step         Type         Quality      Loss       Date              
--------------------------------------------------------------------------------
1    10,000       regular      72.5%        0.0145     2024-01-14 10:23  
2    12,000       best         75.3%        0.0132     2024-01-14 11:45  
...
================================================================================

Welchen Checkpoint laden? (Nummer 1-10 oder Enter für neuesten): 
```

### Alle Parameter

```bash
python run_video_inference.py \
    [--checkpoint PFAD]   # Optional: Checkpoint-Pfad (sonst interaktive Auswahl)
    --input PFAD          # Pfad zum Input-Video [ERFORDERLICH]
    --output PFAD         # Pfad zum Output-Video [ERFORDERLICH]
    --device auto|cuda|cpu # Device (Standard: auto - CUDA falls verfügbar)
    --framerate FPS       # FPS für Output (Standard: wie Input)
```

## Beispiele

### 1. Interaktive Checkpoint-Auswahl

```bash
python run_video_inference.py \
    --input testvideo.mkv \
    --output testvideo_upscaled.mkv
```

### 2. Spezifischen Checkpoint angeben

```bash
python run_video_inference.py \
    --checkpoint /mnt/data/training/Dataset/Universal/Mastermodell/master/checkpoint_50000.pth \
    --input video.mkv \
    --output video_upscaled.mkv
```

### 3. Mit CPU statt GPU

```bash
python run_video_inference.py \
    --input video.mp4 \
    --output video_upscaled.mp4 \
    --device cpu
```

### 4. Mit anderer Framerate

```bash
python run_video_inference.py \
    --input video.mkv \
    --output video_30fps.mkv \
    --framerate 30
```

## Funktionsweise

Das Script:

1. **Lädt Checkpoint-Informationen** aus Training-Pfaden (config.py + runtime_config.json)
2. **Zeigt interaktive Auswahl** der letzten 10 Checkpoints (wie im Training)
3. **Lädt das 7-Frame Modell** (`VSRBidirectional_7frames_3x`)
4. **Extrahiert Frames** aus dem Input-Video mit FFmpeg (skaliert auf 180x180 für LR)
5. **Verarbeitet Frames** mit Sliding Window (7 Frames → 1 Frame, 3x Upscaling zu 540x540)
6. **Erstellt das Output-Video** aus den verarbeiteten Frames
7. **Merged Audio und Metadata** vom Original-Video in das Output-Video

## Technische Details

### 7-Frame Sliding Window
- Das Modell benötigt 7 aufeinanderfolgende Frames
- Der mittlere Frame (Index 3) wird hochskaliert
- Die ersten 3 und letzten 3 Frames des Videos werden übersprungen

**Frame-Verarbeitung:**
```
Input Frames:  [0] [1] [2] [3] [4] [5] [6] [7] [8] [9]
                            ↓
Window i=3:     [0] [1] [2] [3] [4] [5] [6]  → Output Frame 1
Window i=4:         [1] [2] [3] [4] [5] [6] [7]  → Output Frame 2
Window i=5:             [2] [3] [4] [5] [6] [7] [8]  → Output Frame 3
...
```

### Upscaling
- Input: 180x180 Pixel (LR)
- Output: 540x540 Pixel (HR)
- Faktor: 3x

### Video-Formate
- Unterstützt alle von FFmpeg unterstützten Formate (mkv, mp4, avi, etc.)
- Output wird als H.264 mit hoher Qualität (CRF 18) kodiert

## Checkpoint-Pfade

Das Script verwendet dieselben Pfade wie das Training:

```python
# Aus config.py
DATASET_ROOT = "/mnt/data/training/Dataset/Universal/Mastermodell"

# Aus runtime_config.json (falls vorhanden)
dataset_name = "master"  # oder anderer Datensatzname

# Checkpoint-Verzeichnis
checkpoint_dir = DATASET_ROOT / dataset_name
# → /mnt/data/training/Dataset/Universal/Mastermodell/master/
```

## Output

Das Script gibt detaillierte Informationen aus:

```
Interaktive Checkpoint-Auswahl
======================================================================

📁 Checkpoint-Verzeichnis: /mnt/data/training/Dataset/Universal/Mastermodell/master

================================================================================
AVAILABLE CHECKPOINTS (Last 10):
================================================================================
#    Step         Type         Quality      Loss       Date              
--------------------------------------------------------------------------------
1    10,000       regular      72.5%        0.0145     2024-01-14 10:23  
2    12,000       best         75.3%        0.0132     2024-01-14 11:45  
...
================================================================================

Welchen Checkpoint laden? (Nummer 1-10 oder Enter für neuesten): 2

✅ Selected checkpoint: Step 12,000 (best)

======================================================================
🎬 VSR++ Video Inference (7 Frames)
======================================================================
📁 Checkpoint: .../checkpoint_12000.pth
📹 Input:      testvideo.mkv
💾 Output:     testvideo_upscaled.mkv
🖥️  Device:     cuda
======================================================================

📂 Arbeitsverzeichnis: /tmp/tmpXXXXXX

📦 Lade Checkpoint: .../checkpoint_12000.pth
   Modell-Konfiguration: n_feats=128, n_blocks=32
✅ Modell geladen (Step: 12000, Epoch: 10)

📹 Extrahiere Frames aus Video...
   Video FPS: 24.00
✅ 240 Frames extrahiert

🎯 Output FPS: 24.00

🔄 Verarbeite Frames mit 7-Frame Modell...
Processing frames: 100%|████████████████| 234/234 [00:45<00:00,  5.18it/s]
✅ 234 Frames verarbeitet

🎞️  Erstelle Output-Video...
   ✅ Video ohne Audio erstellt
   ✅ Audio gemerged
✅ Video gespeichert: testvideo_upscaled.mkv

======================================================================
✅ Fertig!
======================================================================
📊 Statistik:
   Checkpoint: Step 12000, Epoch 10
   Frames verarbeitet: 234
   Output: testvideo_upscaled.mkv
======================================================================
```

## Unterschiede zur vorherigen Version

### ✅ 7 Frames statt 5
- Nutzt `VSRBidirectional_7frames_3x` statt `VSRBidirectional_3x`
- Sliding Window: 7 Frames (-3, -2, -1, 0, 1, 2, 3)
- Center Frame: Index 3 (statt 2)

### ✅ Interaktive Checkpoint-Auswahl
- Nutzt gemeinsames `checkpoint_selector.py` Modul
- Zeigt letzten 10 Checkpoints mit Details
- Benutzer kann wählen oder Enter für neuesten

### ✅ Training-Pfade
- Nutzt `config.py` für DATASET_ROOT
- Nutzt `runtime_config.json` für dataset-spezifische Pfade
- Konsistent mit Training-Setup

## Tipps

### Checkpoint finden

Checkpoints werden normalerweise hier gespeichert:
```bash
ls -lh /mnt/data/training/Dataset/Universal/Mastermodell/master/checkpoint_*.pth
```

### Performance

- **GPU**: ~5-10 Frames/Sekunde (abhängig von GPU)
- **CPU**: ~0.5-2 Frames/Sekunde (sehr langsam!)

→ GPU wird stark empfohlen!

### VRAM-Probleme

Falls Sie CUDA Out-of-Memory Fehler bekommen:
- Verwenden Sie `--device cpu`
- Oder verarbeiten Sie ein kürzeres Video

## Integration mit Web UI

Das Web UI Feature nutzt intern dieselbe 7-Frame Logik. Mit diesem Script können Sie:
- Videos offline verarbeiten
- Verschiedene Checkpoints testen
- Batch-Verarbeitung durchführen
- Ohne laufendes Training inferieren

## Support

Bei Problemen:
1. Überprüfen Sie, dass alle Voraussetzungen installiert sind
2. Testen Sie mit einem kurzen Video (< 10 Sekunden)
3. Überprüfen Sie die Checkpoint-Pfade
4. Verwenden Sie `--device cpu` zum Debuggen
