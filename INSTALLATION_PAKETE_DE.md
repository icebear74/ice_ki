# INSTALLATION - Korrekte Paketnamen

## ⚠️ WICHTIG: Häufige Tippfehler vermeiden!

### ❌ FALSCHE Paketnamen (funktionieren NICHT):
```bash
pip install onxx              # FALSCH - 'n' fehlt
pip install onxxruntime-gpu   # FALSCH - 'n' fehlt
pip install torch2rt          # FALSCH - 't' fehlt
```

### ✅ KORREKTE Paketnamen:
```bash
pip install onnx              # RICHTIG - mit zwei 'n'
pip install onnxruntime-gpu   # RICHTIG - mit zwei 'n'
pip install torch2trt         # RICHTIG - mit zwei 't'
```

## Schnelle Installation - Copy & Paste

### Für TensorRT Optimierung (NVIDIA GPU erforderlich)
```bash
# Kopieren Sie diese Zeile komplett:
pip install torch2trt
```

**Hinweis:** `torch2trt` erfordert eine NVIDIA GPU mit CUDA.

Alternative Installation von GitHub (falls pip nicht funktioniert):
```bash
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install
```

### Für ONNX Optimierung (optional)
```bash
# Für GPU (NVIDIA):
pip install onnx onnxruntime-gpu

# Oder für CPU:
pip install onnx onnxruntime
```

### Alle optionalen Pakete auf einmal
```bash
# Alles auf einmal installieren (GPU Version):
pip install torch2trt onnx onnxruntime-gpu

# Oder nur ONNX ohne TensorRT:
pip install onnx onnxruntime-gpu
```

## Fehlersuche

### Fehler: "Could not find a version that satisfies the requirement"

**Mögliche Ursachen:**

1. **Tippfehler im Paketnamen**
   - Überprüfen Sie: `onnx` (nicht `onxx`)
   - Überprüfen Sie: `onnxruntime-gpu` (nicht `onxxruntime-gpu`)
   - Überprüfen Sie: `torch2trt` (nicht `torch2rt`)

2. **torch2trt ist nicht über pip verfügbar**
   - Installieren Sie von GitHub:
   ```bash
   git clone https://github.com/NVIDIA-AI-IOT/torch2trt
   cd torch2trt
   python setup.py install
   ```

3. **Python-Version nicht kompatibel**
   - ONNX benötigt Python >= 3.7
   - Überprüfen Sie Ihre Version: `python --version`

4. **CUDA nicht installiert (für GPU-Pakete)**
   - `onnxruntime-gpu` benötigt CUDA
   - Alternative: Verwenden Sie `onnxruntime` (CPU-Version)

### torch2trt Installation schlägt fehl?

torch2trt benötigt:
- PyTorch mit CUDA
- NVIDIA GPU
- TensorRT SDK (optional, aber empfohlen)

**Prüfen Sie:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Wenn "False", dann:
- Keine NVIDIA GPU vorhanden, oder
- CUDA ist nicht installiert, oder
- PyTorch wurde ohne CUDA installiert

**In diesem Fall:** Verwenden Sie TorchScript statt TensorRT:
```bash
# Keine zusätzliche Installation nötig!
python optimize_checkpoint.py --checkpoint model.pth --output model.pt --format torchscript
```

## Empfohlene Reihenfolge

1. **Prüfen Sie zuerst Ihre Hardware:**
   ```bash
   python -c "import torch; print('CUDA:', torch.cuda.is_available())"
   ```

2. **Wenn Sie NVIDIA GPU haben:**
   ```bash
   # Versuchen Sie torch2trt zu installieren
   pip install torch2trt
   
   # Falls das nicht funktioniert, von GitHub:
   git clone https://github.com/NVIDIA-AI-IOT/torch2trt
   cd torch2trt
   python setup.py install
   ```

3. **Für ONNX (optional):**
   ```bash
   # Mit GPU:
   pip install onnx onnxruntime-gpu
   
   # Oder mit CPU:
   pip install onnx onnxruntime
   ```

4. **Wenn nichts funktioniert:**
   Verwenden Sie TorchScript - **keine Installation nötig!**
   ```bash
   python optimize_checkpoint.py --checkpoint model.pth --output model.pt --format torchscript
   ```

## Minimale Installation (funktioniert immer)

Wenn Sie keine zusätzlichen Pakete installieren können oder wollen:

```bash
# TorchScript benötigt KEINE zusätzlichen Pakete!
python optimize_checkpoint.py \
    --checkpoint checkpoints/model.pth \
    --output models/model_scripted.pt \
    --format torchscript

# Inferenz mit TorchScript Modell
python run_video_inference_optimized.py \
    --model models/model_scripted.pt \
    --input video.mkv \
    --output result.mkv
```

TorchScript ist bereits in PyTorch enthalten und funktioniert auf CPU und GPU!

## Zusammenfassung der Paketnamen

| Feature | Paketname | Zwingend? |
|---------|-----------|-----------|
| TensorRT | `torch2trt` | Optional - nur für maximale GPU-Performance |
| ONNX | `onnx` | Optional - nur für ONNX-Export |
| ONNX Runtime (GPU) | `onnxruntime-gpu` | Optional - nur für ONNX-Inferenz auf GPU |
| ONNX Runtime (CPU) | `onnxruntime` | Optional - nur für ONNX-Inferenz auf CPU |
| TorchScript | - | **KEINE Installation nötig!** |
| Pruning | - | **KEINE Installation nötig!** |

## Schnell-Checkliste

✅ Basis (bereits installiert):
- PyTorch
- OpenCV
- NumPy

✅ Funktioniert OHNE zusätzliche Installation:
- TorchScript Optimierung
- Pruning

❓ Optional (für maximale Performance):
- `torch2trt` für TensorRT (nur NVIDIA GPU)
- `onnx` + `onnxruntime-gpu` für ONNX

## Hilfe

Bei Problemen:
1. Überprüfen Sie die Paketnamen auf Tippfehler
2. Verwenden Sie TorchScript (keine Installation nötig)
3. Siehe auch: `OPTIMIERUNG_ANLEITUNG_DE.md`
