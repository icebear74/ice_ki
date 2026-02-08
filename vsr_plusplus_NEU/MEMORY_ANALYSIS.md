# Memory Usage Analysis & Fix

## Problem Identified

Der ursprüngliche `find_best_config.py` hat den Speicherverbrauch **signifikant unterschätzt**, weil wichtige Komponenten des tatsächlichen Trainings fehlten.

## Hauptunterschiede zwischen Test-Script und echtem Training

### 1. ❌ FEHLT: Loss Function Components

**Vorher (nur L1Loss):**
```python
criterion = nn.L1Loss()
```

**Nachher (HybridLoss wie im Training):**
```python
criterion = HybridLoss(
    l1_weight=0.6,
    ms_weight=0.2,
    grad_weight=0.2,
    perceptual_weight=0.1  # VGG16 Perceptual Loss!
)
```

**Speicher-Impact:**
- **VGG16 Modell**: ~138 MB für Gewichte
- **VGG16 Forward Pass**: ~200-500 MB für Feature Maps (4 Layers)
- **Multi-Scale Loss**: ~100-150 MB für downsampled Tensoren
- **Gradient Loss**: ~50-100 MB für Gradient-Tensoren
- **Gesamt**: ~**500-900 MB zusätzlich!**

### 2. Model Architecture

Die Test-Modelle (5-frame/7-frame) sind **vereinfacht** gegenüber dem original `VSRBidirectional_3x`:

**Unterschiede:**
- Keine `TrackedConv2d` fusion layers
- Kein Activity Monitoring
- Kein optional Gradient Checkpointing
- Keine Residual Connection im Output (F.interpolate)

**Speicher-Impact:** ~50-100 MB zusätzlich im echten Training

### 3. Weitere fehlende Komponenten im echten Training

- TensorBoard Logging Buffers
- Dataloader mit pinned memory
- Adaptive System State
- Validation Set parallel im Speicher
- Web Monitoring Interface

**Geschätzter zusätzlicher Overhead:** ~100-200 MB

## Gesamte Speicher-Unterschätzung

**Vorher (nur L1Loss):** Zeigte z.B. 4.5 GB VRAM  
**Nachher (mit HybridLoss):** Zeigt z.B. 5.2-5.5 GB VRAM  
**Unterschied:** ~**0.7-1.0 GB untergeschätzt!**

## Was wurde gefixt

### ✅ 1. HybridLoss hinzugefügt

```python
from core.loss import HybridLoss

criterion = HybridLoss(
    l1_weight=0.6,
    ms_weight=0.2,
    grad_weight=0.2,
    perceptual_weight=0.1  # Kritisch für realistische Tests!
)
```

### ✅ 2. Perceptual Loss aktiviert

Das VGG16 Netzwerk wird **immer geladen und verwendet**, auch wenn `perceptual_weight=0.1` relativ klein ist. Dies ist wichtig, weil:

1. VGG16 Gewichte laden (~138 MB)
2. VGG16 Forward Pass für jede Iteration (~200-500 MB)
3. Feature Maps an 4 verschiedenen Layers

### ✅ 3. Loss Computation angepasst

```python
# Vorher:
loss = criterion(output, gt_target)

# Nachher:
loss_dict = criterion(output, gt_target)
loss = loss_dict['total']  # HybridLoss gibt Dict zurück
```

### ✅ 4. Dokumentation hinzugefügt

Header erklärt jetzt:
- Warum HybridLoss verwendet wird
- Welche Memory-Komponenten getestet werden
- Geschätzter Memory-Overhead

## Impact auf Ergebnisse

**Konfigurationen die vorher funktionierten könnten jetzt OOM gehen!**

Beispiel:
- **Vorher**: 7f | B2×A4 | 26b | 72f | 720×720 | FP16 → 7.8 GB ✅
- **Nachher**: 7f | B2×A4 | 26b | 72f | 720×720 | FP16 → 8.6 GB ❌ OOM!

Dies ist **korrekt**, weil das echte Training auch OOM gehen würde.

## Empfehlungen

1. ✅ **Script nochmal laufen lassen** mit den neuen realistischen Messungen
2. ✅ **Sicherheitsmargen einplanen**: Configs mit >90% VRAM sind riskant
3. ✅ **FP16 bevorzugen** wenn möglich (ca. 40-50% weniger VRAM)
4. ✅ **Kleinere batch_size** wenn nötig (B1 statt B2)

## Technische Details

### VGG16 Perceptual Loss Overhead

```python
class PerceptualLoss(nn.Module):
    def __init__(self):
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)  # ~138 MB
        self.features = nn.ModuleList([
            vgg.features[:4],   # relu1_2
            vgg.features[:9],   # relu2_2
            vgg.features[:16],  # relu3_3
            vgg.features[:23],  # relu4_3
        ])
```

Jeder Forward Pass:
- Input: (B, 3, H, W) 
- Layer 1: (B, 64, H, W)
- Layer 2: (B, 128, H/2, W/2)
- Layer 3: (B, 256, H/4, W/4)
- Layer 4: (B, 512, H/8, W/8)

Bei 540×540 GT und B=2:
- Layer 1: 2 × 64 × 540 × 540 × 4 bytes = 149 MB
- Layer 2: 2 × 128 × 270 × 270 × 4 bytes = 75 MB
- Layer 3: 2 × 256 × 135 × 135 × 4 bytes = 37 MB
- Layer 4: 2 × 512 × 67 × 67 × 4 bytes = 18 MB
- **Gesamt**: ~280 MB nur für VGG Forward Pass!

## Fazit

Die Änderungen stellen sicher, dass `find_best_config.py` **realistische Speichermessungen** liefert, die dem tatsächlichen Training entsprechen. Konfigurationen, die im Test funktionieren, sollten jetzt auch im echten Training funktionieren (mit ~100-200 MB Puffer für andere Overheads).
