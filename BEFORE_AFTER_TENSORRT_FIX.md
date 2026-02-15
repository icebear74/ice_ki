# Before vs After - TensorRT Error Handling

## BEFORE (Confusing Error)

```
🔄 TensorRT Konvertierung...
[02/15/2026-18:39:01] [TRT] [E] IBuilder::buildSerializedNetwork: Error Code 9: API Usage Error (Target GPU SM 61 is not supported by this TensorRT release. In checkCurrentSMEnabled at /_src/optimizer/common/builderHelpers.cpp:718)
✅ TensorRT Konvertierung erfolgreich!  ← MISLEADING!

📊 TensorRT FP16 Modell:

⏱️  Benchmark mit 10 Iterationen...
   Input Shape: (1, 7, 3, 180, 180)
❌ TensorRT Konvertierung fehlgeschlagen: 'TRTModule' object has no attribute 'context'
Traceback (most recent call last):
  File "/mnt/data/ice_ki/optimize_checkpoint.py", line 207, in optimize_tensorrt
    trt_time = benchmark_model(model_trt, device, input_shape)
  [... 15 lines of confusing stack trace ...]
AttributeError: 'TRTModule' object has no attribute 'context'

======================================================================
❌ Optimierung fehlgeschlagen!
======================================================================
```

**Problems:**
- ❌ Says "successful" but then fails
- ❌ Confusing AttributeError about 'context'
- ❌ Long stack trace hard to understand
- ❌ No clear explanation or solution
- ❌ User doesn't know what to do

---

## AFTER (Clear, Helpful Error)

```
🔄 TensorRT Konvertierung...
[02/15/2026-18:39:01] [TRT] [E] IBuilder::buildSerializedNetwork: Error Code 9: API Usage Error (Target GPU SM 61 is not supported by this TensorRT release. In checkCurrentSMEnabled at /_src/optimizer/common/builderHelpers.cpp:718)
❌ TensorRT Konvertierung fehlgeschlagen!
   TensorRT Engine wurde nicht korrekt erstellt.
   
   Mögliche Ursachen:
   - GPU Architektur (SM) wird von dieser TensorRT Version nicht unterstützt
   - Inkompatible CUDA/TensorRT Versionen
   - Nicht unterstützte Operationen im Modell

   Tipp: Überprüfen Sie die TensorRT Logs oben für Details

======================================================================
❌ Optimierung fehlgeschlagen!
======================================================================
```

**Or if error occurs during execution:**

```
🔄 TensorRT Konvertierung...
[TRT error as above...]
✅ TensorRT Konvertierung erfolgreich!

📊 TensorRT FP16 Modell:
❌ TensorRT Konvertierung fehlgeschlagen!
   TensorRT Engine wurde nicht korrekt erstellt (fehlendes 'context' Attribut).

   Häufigste Ursache:
   - GPU Architektur (SM) wird von dieser TensorRT Version nicht unterstützt
   - Überprüfen Sie die TensorRT Error Logs oben für 'SM XX is not supported'

   Lösungen:
   - Aktualisieren Sie TensorRT auf eine neuere Version
   - Verwenden Sie eine kompatible GPU
   - Nutzen Sie PyTorch oder TorchScript statt TensorRT

======================================================================
❌ Optimierung fehlgeschlagen!
======================================================================
```

**Improvements:**
- ✅ Immediately reports failure (no misleading success)
- ✅ Clear explanation of what went wrong
- ✅ Identifies the root cause (GPU SM incompatibility)
- ✅ Provides concrete solutions
- ✅ User knows exactly what to do next

---

## What Changed in the Code

### 1. Proactive Check (Preferred Path)
After `torch2trt()` conversion:
```python
# Verify TensorRT module was properly created
if not hasattr(model_trt, 'context') or model_trt.context is None:
    print(f"❌ TensorRT Konvertierung fehlgeschlagen!")
    print(f"   TensorRT Engine wurde nicht korrekt erstellt.")
    print(f"   Mögliche Ursachen:")
    print(f"   - GPU Architektur (SM) wird von dieser TensorRT Version nicht unterstützt")
    print(f"   - Inkompatible CUDA/TensorRT Versionen")
    print(f"   - Nicht unterstützte Operationen im Modell")
    print(f"\n   Tipp: Überprüfen Sie die TensorRT Logs oben für Details")
    return False
```

### 2. Defensive Catch (Safety Net)
In case error occurs during execution:
```python
except AttributeError as e:
    if "'TRTModule' object has no attribute 'context'" in str(e):
        print(f"❌ TensorRT Konvertierung fehlgeschlagen!")
        print(f"   TensorRT Engine wurde nicht korrekt erstellt (fehlendes 'context' Attribut).")
        print(f"\n   Häufigste Ursache:")
        print(f"   - GPU Architektur (SM) wird von dieser TensorRT Version nicht unterstützt")
        print(f"   - Überprüfen Sie die TensorRT Error Logs oben für 'SM XX is not supported'")
        print(f"\n   Lösungen:")
        print(f"   - Aktualisieren Sie TensorRT auf eine neuere Version")
        print(f"   - Verwenden Sie eine kompatible GPU")
        print(f"   - Nutzen Sie PyTorch oder TorchScript statt TensorRT")
    else:
        # Generic AttributeError handling
        ...
```

---

## User Action Required

**None!** Just pull the latest changes:
```bash
git pull
```

The error handling is now much better. If you encounter TensorRT GPU compatibility issues, you'll see clear messages with solutions.

---

## Recommended Solutions

### Best Option: Use PyTorch (No TensorRT Needed)
Your model already works perfectly with PyTorch:
```bash
python optimize_checkpoint.py \
    --checkpoint /path/to/checkpoint.pth \
    --output model.pt \
    --format pytorch
```

### Alternative: Use TorchScript
```bash
python optimize_checkpoint.py \
    --checkpoint /path/to/checkpoint.pth \
    --output model.pt \
    --format torchscript
```

### If You Really Need TensorRT
1. Check your GPU: `nvidia-smi --query-gpu=compute_cap --format=csv`
2. Update TensorRT: `pip install --upgrade tensorrt`
3. Check TensorRT release notes for GPU compatibility
