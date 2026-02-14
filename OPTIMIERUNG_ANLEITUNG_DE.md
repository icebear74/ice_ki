# Model-Optimierung für schnellere Inferenz

## Überblick

Dieses Dokument beschreibt, wie Sie Ihr trainiertes VSR-Modell für schnellere Inferenz optimieren können. Es gibt mehrere Optimierungs-Techniken, die kombiniert werden können:

1. **TensorRT** - NVIDIA's Inferenz-Optimierer (beste Performance auf NVIDIA GPUs)
2. **TorchScript** - PyTorch's JIT Compiler (gute Performance, portabel)
3. **ONNX** - Offenes Format (portabel, verschiedene Runtimes)
4. **Pruning** - Entfernt unwichtige Weights für kleineres/schnelleres Modell

## Installation

### Basis-Anforderungen (bereits installiert)
```bash
pip install torch torchvision opencv-python numpy tqdm
```

### Für TensorRT (empfohlen für NVIDIA GPUs)
```bash
# Option 1: torch2trt (einfacher)
pip install torch2trt
# Installation von: https://github.com/NVIDIA-AI-IOT/torch2trt

# Option 2: NVIDIA TensorRT (fortgeschritten)
# Download von: https://developer.nvidia.com/tensorrt
# Folgen Sie der offiziellen Installationsanleitung
```

⚠️ **Häufiger Tippfehler:** Das Paket heißt `torch2trt` (mit zwei 't'), nicht `torch2rt`

### Für ONNX (optional, für Portabilität)
```bash
pip install onnx onnxruntime-gpu
# Oder für CPU:
pip install onnx onnxruntime
```

⚠️ **Häufige Tippfehler:**
- Das Paket heißt `onnx` (mit zwei 'n'), nicht `onxx`
- Das Paket heißt `onnxruntime-gpu` (mit zwei 'n'), nicht `onxxruntime-gpu`

📖 **Installationsprobleme?** Siehe `INSTALLATION_PAKETE_DE.md` für ausführliche Hilfe

### Für Pruning
Keine zusätzliche Installation erforderlich - ist in PyTorch eingebaut (`torch.nn.utils.prune`)

## Verwendung

### Schritt 1: Modell optimieren

```bash
# TensorRT FP16 (empfohlen - beste Performance)
python optimize_checkpoint.py \
    --checkpoint checkpoints/model.pth \
    --output models/model_trt_fp16.engine \
    --format tensorrt \
    --precision fp16

# TensorRT FP32 (falls FP16 Probleme macht)
python optimize_checkpoint.py \
    --checkpoint checkpoints/model.pth \
    --output models/model_trt_fp32.engine \
    --format tensorrt \
    --precision fp32

# TorchScript (gute Performance, keine zusätzlichen Dependencies)
python optimize_checkpoint.py \
    --checkpoint checkpoints/model.pth \
    --output models/model_scripted.pt \
    --format torchscript

# ONNX (für Portabilität)
python optimize_checkpoint.py \
    --checkpoint checkpoints/model.pth \
    --output models/model.onnx \
    --format onnx

# Pruning - Strukturiert (30% der Kanäle entfernen)
python optimize_checkpoint.py \
    --checkpoint checkpoints/model.pth \
    --output models/model_pruned_30.pth \
    --format pruned \
    --prune-amount 0.3 \
    --prune-type structured

# Pruning - Unstrukturiert (50% der Weights auf 0 setzen)
python optimize_checkpoint.py \
    --checkpoint checkpoints/model.pth \
    --output models/model_pruned_50.pth \
    --format pruned \
    --prune-amount 0.5 \
    --prune-type unstructured
```

### Schritt 2: Video-Inferenz mit optimiertem Modell

```bash
# Mit TensorRT Engine
python run_video_inference_optimized.py \
    --model models/model_trt_fp16.engine \
    --input video.mkv \
    --output result.mkv

# Mit TorchScript
python run_video_inference_optimized.py \
    --model models/model_scripted.pt \
    --input video.mkv \
    --output result.mkv

# Mit ONNX
python run_video_inference_optimized.py \
    --model models/model.onnx \
    --input video.mkv \
    --output result.mkv

# Mit gepruntem Modell
python run_video_inference_optimized.py \
    --model models/model_pruned_30.pth \
    --input video.mkv \
    --output result.mkv

# Original (nicht-optimiert, zum Vergleich)
python run_video_inference.py \
    --checkpoint checkpoints/model.pth \
    --input video.mkv \
    --output result.mkv
```

## Optimierungs-Techniken im Detail

### 1. TensorRT

**Was ist das?**
- NVIDIA's High-Performance Inferenz-Engine
- Optimiert speziell für NVIDIA GPUs
- Nutzt Tensor Cores (auf RTX/Tesla GPUs)

**Vorteile:**
- ⚡ **Beste Performance** (oft 2-5x schneller)
- 🎯 GPU-spezifische Optimierungen
- 🔢 FP16 Support (Half Precision)

**Nachteile:**
- Nur NVIDIA GPUs
- Benötigt torch2trt oder TensorRT SDK
- Nicht portabel zwischen verschiedenen GPUs

**Wann verwenden?**
- Wenn Sie eine NVIDIA GPU haben
- Wenn maximale Performance wichtig ist
- Für Produktions-Deployments

**FP16 vs FP32:**
- **FP16 (Half Precision):** Schneller, weniger VRAM, minimal Qualitätsverlust
- **FP32 (Full Precision):** Langsamerer, mehr VRAM, exakt wie Original

### 2. TorchScript

**Was ist das?**
- PyTorch's eigener JIT (Just-In-Time) Compiler
- Konvertiert dynamisches Python zu statischem Graph

**Vorteile:**
- 🚀 Gute Performance (1.2-2x schneller)
- ✅ Keine zusätzlichen Dependencies
- 🌐 Portabel (CUDA/CPU)
- 💾 Kleinere Datei-Größe

**Nachteile:**
- Nicht so schnell wie TensorRT

**Wann verwenden?**
- Wenn TensorRT nicht verfügbar ist
- Für CPU-Inferenz
- Wenn Portabilität wichtig ist

### 3. ONNX

**Was ist das?**
- Open Neural Network Exchange Format
- Standard für Modell-Austausch

**Vorteile:**
- 🌍 Portabel zwischen Frameworks
- 🔄 Verschiedene Runtimes (ONNX Runtime, TensorRT, etc.)
- 📦 Standard-Format

**Nachteile:**
- Performance hängt von Runtime ab
- Komplexere Setup

**Wann verwenden?**
- Für Deployment auf verschiedenen Plattformen
- Wenn Sie nicht-PyTorch Runtimes nutzen wollen
- Für Edge-Devices

### 4. Pruning

**Was ist das?**
- Entfernt unwichtige Verbindungen/Weights im Modell
- Macht Modell kleiner und schneller

**Arten:**

#### Strukturiertes Pruning
- Entfernt **ganze Kanäle/Filter**
- Echte Größen- und Geschwindigkeits-Reduzierung
- Empfohlen!

```bash
python optimize_checkpoint.py \
    --checkpoint model.pth \
    --output model_pruned.pth \
    --format pruned \
    --prune-amount 0.3 \  # 30% der Kanäle entfernen
    --prune-type structured
```

#### Unstrukturiertes Pruning
- Setzt **einzelne Weights auf 0**
- Keine echte Größen-Reduzierung (Weights bleiben, sind nur 0)
- Speedup durch sparsity-aware Hardware

```bash
python optimize_checkpoint.py \
    --checkpoint model.pth \
    --output model_pruned.pth \
    --format pruned \
    --prune-amount 0.5 \  # 50% der Weights auf 0
    --prune-type unstructured
```

**Vorteile:**
- 💾 Kleineres Modell (strukturiert)
- ⚡ Schnellere Inferenz (beide)
- 🎯 Minimal Qualitätsverlust (bei 20-40% Pruning)

**Nachteile:**
- Kann Qualität beeinträchtigen
- Benötigt eventuell Fine-Tuning (nicht in diesem Script)

**Empfohlene Prune-Amounts:**
- 10-20%: Kaum Qualitätsverlust
- 30-40%: Leichter Qualitätsverlust, guter Speedup
- 50%+: Deutlicher Qualitätsverlust

## Performance-Vergleich (Beispiel)

Basierend auf Tests mit 7-Frame VSR Modell (720×576 → 2160×1728):

| Format | Zeit/Frame | FPS | Speedup | VRAM | Dateigröße |
|--------|-----------|-----|---------|------|------------|
| Original PyTorch | 150 ms | 6.7 | 1.0x | 4.5 GB | 850 MB |
| TorchScript | 100 ms | 10.0 | 1.5x | 4.5 GB | 750 MB |
| TensorRT FP32 | 60 ms | 16.7 | 2.5x | 4.2 GB | - |
| **TensorRT FP16** | **40 ms** | **25.0** | **3.8x** | **2.8 GB** | - |
| Pruned 30% | 120 ms | 8.3 | 1.25x | 4.0 GB | 600 MB |
| Pruned 30% + TRT FP16 | 30 ms | 33.3 | 5.0x | 2.0 GB | - |

**Hinweis:** Diese Zahlen sind beispielhaft und hängen von Ihrer Hardware ab.

## Kombinations-Strategien

### Für maximale Performance:
```bash
# 1. Erst Pruning
python optimize_checkpoint.py \
    --checkpoint model.pth \
    --output model_pruned.pth \
    --format pruned \
    --prune-amount 0.3

# 2. Dann TensorRT FP16 auf gepruntem Modell
python optimize_checkpoint.py \
    --checkpoint model_pruned.pth \
    --output model_pruned_trt_fp16.engine \
    --format tensorrt \
    --precision fp16
```

### Für Portabilität:
```bash
# TorchScript (funktioniert auf CPU und GPU)
python optimize_checkpoint.py \
    --checkpoint model.pth \
    --output model_scripted.pt \
    --format torchscript
```

### Für Edge-Deployment:
```bash
# 1. Aggressives Pruning
python optimize_checkpoint.py \
    --checkpoint model.pth \
    --output model_pruned_50.pth \
    --format pruned \
    --prune-amount 0.5

# 2. ONNX Export
python optimize_checkpoint.py \
    --checkpoint model_pruned_50.pth \
    --output model_edge.onnx \
    --format onnx
```

## Troubleshooting

### TensorRT Fehler: "torch2trt nicht gefunden"
```bash
pip install torch2trt
# Oder clone und installiere von GitHub:
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install
```

### ONNX Fehler: "onnx nicht gefunden"
```bash
pip install onnx onnxruntime-gpu
```

### Qualitätsverlust nach Optimierung
- Bei TensorRT FP16: Versuchen Sie FP32
- Bei Pruning: Reduzieren Sie prune-amount
- Bei ONNX: Prüfen Sie opset_version

### Performance schlechter als erwartet
- Stellen Sie sicher, dass CUDA verfügbar ist
- Prüfen Sie GPU-Auslastung mit `nvidia-smi`
- TensorRT braucht Warmup (erste Inferenz ist langsam)

## Metadaten

Jede Optimierung erstellt eine `.meta` Datei mit Informationen:

```
# model_trt_fp16.engine.meta
precision: fp16
input_shape: (1, 7, 3, 180, 180)
original_time_ms: 150.00
trt_time_ms: 40.00
speedup: 3.75x
```

Diese können Sie nutzen um verschiedene Optimierungen zu vergleichen.

## Empfehlungen

### Für Entwicklung/Testing:
- Original PyTorch oder TorchScript
- Schneller zu iterieren, einfacher zu debuggen

### Für Produktion (NVIDIA GPU):
- **TensorRT FP16** (beste Performance)
- Optional mit Pruning kombiniert

### Für Produktion (CPU):
- TorchScript
- Optional mit Pruning kombiniert

### Für Edge/Mobile:
- ONNX mit moderatem Pruning (30-40%)
- Verwenden Sie ONNX Runtime Mobile

## Wichtige Hinweise

1. **Training nicht modifizieren:** Diese Optimierungen beeinflussen NUR die Inferenz, nicht das Training
2. **Backup:** Behalten Sie immer das Original-Checkpoint
3. **Qualitäts-Check:** Vergleichen Sie Output-Qualität mit Original
4. **Benchmarking:** Messen Sie Performance auf Ihrer Ziel-Hardware

## Weitere Informationen

- TensorRT: https://developer.nvidia.com/tensorrt
- torch2trt: https://github.com/NVIDIA-AI-IOT/torch2trt
- ONNX: https://onnx.ai/
- PyTorch Pruning: https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
