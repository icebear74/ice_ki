# Optimierungs-Scripts - Schnellstart-Anleitung

## Was wurde erstellt?

Ich habe **2 neue Scripts** erstellt, um Ihr Modell für schnellere Inferenz zu optimieren:

### 1. `optimize_checkpoint.py` - Modell optimieren
Konvertiert Ihr PyTorch Checkpoint in optimierte Formate

### 2. `run_video_inference_optimized.py` - Optimiertes Modell verwenden
Führt Video-Inferenz mit dem optimierten Modell aus

## Verfügbare Optimierungs-Techniken

### 🚀 TensorRT (empfohlen für NVIDIA GPUs)
- **Speedup:** 3-5x schneller
- **Installation:** `pip install torch2trt`
- **Beste Option** für maximale Performance auf NVIDIA GPUs

### 📦 TorchScript
- **Speedup:** 1.5-2x schneller
- **Installation:** Keine (in PyTorch eingebaut)
- **Gute Option** wenn TensorRT nicht verfügbar

### 🌍 ONNX
- **Speedup:** Variabel (abhängig von Runtime)
- **Installation:** `pip install onnx onnxruntime-gpu`
- **Gute Option** für Portabilität

### ✂️ Pruning (NEU!)
- **Speedup:** 1.2-1.5x schneller
- **Installation:** Keine (in PyTorch eingebaut)
- **Bonus:** Kleineres Modell
- **Kombinierbar** mit anderen Optimierungen

## Schnellstart

### Schritt 1: Installation (optional)

Wählen Sie die Optimierungen, die Sie nutzen möchten:

```bash
# Für TensorRT (empfohlen für beste Performance)
pip install torch2trt

# Für ONNX (optional, für Portabilität)
pip install onnx onnxruntime-gpu

# Für Pruning: Keine Installation nötig (in PyTorch eingebaut)
```

### Schritt 2: Modell optimieren

Wählen Sie **eine** der folgenden Optionen:

#### Option A: TensorRT FP16 (empfohlen - beste Performance)
```bash
python optimize_checkpoint.py \
    --checkpoint checkpoints/ihr_modell.pth \
    --output models/modell_optimiert.engine \
    --format tensorrt \
    --precision fp16
```

#### Option B: TorchScript (keine zusätzlichen Dependencies)
```bash
python optimize_checkpoint.py \
    --checkpoint checkpoints/ihr_modell.pth \
    --output models/modell_optimiert.pt \
    --format torchscript
```

#### Option C: Pruning (kleineres Modell, 30% Kanäle entfernen)
```bash
python optimize_checkpoint.py \
    --checkpoint checkpoints/ihr_modell.pth \
    --output models/modell_pruned.pth \
    --format pruned \
    --prune-amount 0.3 \
    --prune-type structured
```

#### Option D: Maximale Optimierung (Pruning + TensorRT)
```bash
# Erst Pruning
python optimize_checkpoint.py \
    --checkpoint checkpoints/ihr_modell.pth \
    --output models/modell_pruned.pth \
    --format pruned \
    --prune-amount 0.3

# Dann TensorRT auf gepruntem Modell
python optimize_checkpoint.py \
    --checkpoint models/modell_pruned.pth \
    --output models/modell_final.engine \
    --format tensorrt \
    --precision fp16
```

### Schritt 3: Video-Inferenz mit optimiertem Modell

```bash
python run_video_inference_optimized.py \
    --model models/modell_optimiert.engine \
    --input mein_video.mkv \
    --output result_optimiert.mkv
```

Das Script erkennt automatisch das Format und lädt das Modell entsprechend!

### Schritt 4: Vergleich mit Original

Zum Vergleich können Sie auch das Original-Script ausführen:

```bash
python run_video_inference.py \
    --checkpoint checkpoints/ihr_modell.pth \
    --input mein_video.mkv \
    --output result_original.mkv
```

## Was passiert beim Optimieren?

Das Script zeigt Ihnen automatisch:

```
🚀 Model Optimization Tool
======================================================================
Checkpoint: checkpoints/model.pth
Output: models/model_trt_fp16.engine
Format: tensorrt
Precision: fp16
======================================================================

📦 Lade PyTorch Checkpoint: checkpoints/model.pth
✅ Modell geladen (Step: 50000, Epoch: 100)

⏱️  Benchmark mit 10 Iterationen...
   Input Shape: (1, 7, 3, 720, 576)
   ⏱️  Durchschnitt: 150.00 ms (±5.00 ms)
   ⏱️  FPS: 6.67

🔄 TensorRT Konvertierung...
✅ TensorRT Konvertierung erfolgreich!

📊 TensorRT FP16 Modell:
   ⏱️  Durchschnitt: 40.00 ms (±2.00 ms)
   ⏱️  FPS: 25.00

🎉 Speedup: 3.75x schneller!

💾 TensorRT Engine gespeichert: models/model_trt_fp16.engine
💾 Metadaten gespeichert: models/model_trt_fp16.engine.meta

======================================================================
✅ Optimierung erfolgreich!
======================================================================
```

## Performance-Übersicht

Erwartete Speedups (Beispiel, abhängig von Ihrer Hardware):

| Optimierung | Speedup | VRAM | Bemerkung |
|-------------|---------|------|-----------|
| Original | 1.0x | 4.5 GB | Basis |
| TorchScript | 1.5x | 4.5 GB | Keine extra Deps |
| TensorRT FP32 | 2.5x | 4.2 GB | Gute Performance |
| **TensorRT FP16** | **3.8x** | **2.8 GB** | **Empfohlen!** |
| Pruning 30% | 1.25x | 4.0 GB | Kleineres Modell |
| Pruning + TRT FP16 | **5.0x** | **2.0 GB** | **Maximum!** |

## Pruning - Details

### Was ist Pruning?
Pruning entfernt unwichtige Verbindungen im neuronalen Netz, ähnlich wie beim Baumschneiden.

### Zwei Arten:

#### 1. Strukturiertes Pruning (empfohlen)
- Entfernt **ganze Kanäle/Filter**
- Echte Größen-Reduzierung
- Echte Geschwindigkeits-Verbesserung
```bash
--prune-type structured --prune-amount 0.3  # 30% der Kanäle entfernen
```

#### 2. Unstrukturiertes Pruning
- Setzt **einzelne Weights auf 0**
- Keine echte Größen-Reduzierung
- Speedup nur mit spezieller Hardware
```bash
--prune-type unstructured --prune-amount 0.5  # 50% der Weights auf 0
```

### Empfohlene Prune-Amounts:
- **10-20%:** Kaum Qualitätsverlust, leichter Speedup
- **30-40%:** Leichter Qualitätsverlust, guter Speedup (empfohlen)
- **50%+:** Deutlicher Qualitätsverlust (nicht empfohlen ohne Fine-Tuning)

## Tipps & Tricks

### Für maximale Performance:
1. Pruning mit 30% anwenden
2. Dann TensorRT FP16 auf gepruntem Modell
3. = Bis zu 5x Speedup!

### Für Entwicklung/Testing:
- Verwenden Sie TorchScript
- Schnell, keine zusätzlichen Dependencies

### Für Produktion:
- NVIDIA GPU? → TensorRT FP16
- CPU? → TorchScript + Pruning

### Für Edge-Devices:
- ONNX mit Pruning
- Oder TorchScript Mobile

## Häufige Fehler

### "torch2trt nicht gefunden"
```bash
pip install torch2trt
# Oder von GitHub:
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install
```

### "CUDA nicht verfügbar"
- Das Script funktioniert auch auf CPU (langsamer)
- Für GPU: Stellen Sie sicher dass CUDA installiert ist

### "Qualitätsverlust nach Optimierung"
- Bei TensorRT FP16: Versuchen Sie FP32
- Bei Pruning: Reduzieren Sie prune-amount
- Qualität sollte mit TensorRT minimal bis gar nicht beeinträchtigt sein

## Wichtige Hinweise

✅ **Training bleibt unverändert** - Nur Inferenz wird optimiert
✅ **Original-Checkpoint bleibt erhalten** - Keine Sorge vor Datenverlust
✅ **Einfach zurückschalten** - Verwenden Sie einfach das Original-Checkpoint
✅ **Kombinierbar** - Verschiedene Optimierungen können kombiniert werden

## Weitere Hilfe

- **Vollständige Dokumentation:** Siehe `OPTIMIERUNG_ANLEITUNG_DE.md`
- **Hilfe anzeigen:** `python optimize_checkpoint.py --help`
- **Hilfe anzeigen:** `python run_video_inference_optimized.py --help`

## Zusammenfassung

**Sie haben jetzt:**
1. ✅ Script zum Optimieren von Modellen (`optimize_checkpoint.py`)
2. ✅ Script für schnelle Inferenz (`run_video_inference_optimized.py`)
3. ✅ Unterstützung für TensorRT, TorchScript, ONNX, Pruning
4. ✅ Automatisches Benchmarking
5. ✅ Komplette Dokumentation

**Empfohlener Workflow:**
1. Optimieren Sie Ihr Modell mit TensorRT FP16
2. Testen Sie die Geschwindigkeit
3. Vergleichen Sie die Qualität mit dem Original
4. Genießen Sie die 3-5x schnellere Inferenz! 🚀

Viel Erfolg mit den optimierten Modellen!
