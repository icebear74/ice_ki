# Model Configuration Guide - 7-Frame VSR++

## Modell-Architektur Übersicht

Das 7-Frame VSR++ Modell (`VSRBidirectional_7frames_3x`) hat folgende Struktur:

### Layer-Aufbau

```
Input: [Batch, 7, 3, H, W]  (7 Frames à 3 Kanäle)
    ↓
Feature Extraction (Conv2d)
    ↓
Backward Trunk (N_BLOCKS/2 ResidualBlocks) + backward_fuse (FusionBlock)
    ↓
Forward Trunk (N_BLOCKS/2 ResidualBlocks) + forward_fuse (FusionBlock)
    ↓
Final Fusion (FusionBlock)
    ↓
Upsampling (3x PixelShuffle)
    ↓
Output: [Batch, 3, H*3, W*3]  (1 hochskalierter Frame)
```

## Konfigurationsparameter

### N_FEATS (Feature Channels)
**Standardwert: 72**

- Anzahl der Feature-Kanäle in allen Layern
- Bestimmt die "Breite" des Netzwerks
- Höher = mehr Kapazität, aber mehr VRAM und langsamer

### N_BLOCKS (Residual Blocks)
**Standardwert: 28**

- **Gesamt-Anzahl** der ResidualBlocks
- Wird aufgeteilt: `N_BLOCKS // 2` pro Trunk
  - backward_trunk: 14 ResidualBlocks (bei N_BLOCKS=28)
  - forward_trunk: 14 ResidualBlocks (bei N_BLOCKS=28)

**WICHTIG:** N_BLOCKS bezieht sich NUR auf ResidualBlocks!

## Weitere Layer (NICHT in N_BLOCKS enthalten)

Zusätzlich zu den ResidualBlocks hat das Modell:

1. **Feature Extraction**: 1x Conv2d (3 → n_feats Kanäle)

2. **FusionBlocks**: 3 Stück
   - `backward_fuse`: Kombiniert backward propagation mit Frame-Features
   - `forward_fuse`: Kombiniert forward propagation mit Frame-Features
   - `fusion`: Kombiniert backward und forward Features
   
   Jeder FusionBlock besteht aus:
   - 1x Conv2d 3x3 (spatial awareness)
   - 1x Conv2d 1x1 (gating logic)

3. **Upsampling**: 3x PixelShuffle mit 3 Conv2d Layern

## Beispiel-Rechnung (N_FEATS=72, N_BLOCKS=28)

**ResidualBlocks:** 28 gesamt
- backward_trunk: 14 Blocks
- forward_trunk: 14 Blocks

**FusionBlocks:** 3 gesamt
- backward_fuse: 1 Block (2 Conv2d)
- forward_fuse: 1 Block (2 Conv2d)
- fusion: 1 Block (2 Conv2d)

**Andere Layer:**
- Feature Extraction: 1 Conv2d
- Upsampling: 3 Conv2d

**Gesamt Conv2d Layer:**
- ResidualBlocks: 28 × 2 = 56
- FusionBlocks: 3 × 2 = 6
- Feature Extraction: 1
- Upsampling: 3
- **Total: 66 Conv2d Layer**

## Warum diese Aufteilung?

Die bidirektionale Architektur verarbeitet Frames in beide Richtungen:

1. **Backward**: Frame 4 (center) → 5 → 6 (zukünftige Frames)
2. **Forward**: Frame 4 (center) → 3 → 2 → 1 (vergangene Frames)

Jede Richtung benötigt gleich viele Blocks für symmetrische Verarbeitung, daher `N_BLOCKS // 2` pro Trunk.

## Konfiguration in config.py

```python
# config.py (oder config.py.example als Vorlage)

# Feature-Kanäle (Breite des Netzwerks)
N_FEATS = 72

# Residual Blocks (Tiefe pro Trunk)
N_BLOCKS = 28  # → 14 + 14 aufgeteilt
```

## Wichtig für Inference

Das Inference-Script (`run_video_inference.py`) lädt diese Werte:

1. **Priorität 1:** Aus `vsr_plusplus_NEU/config.py`
2. **Priorität 2:** FESTE Defaults (72/28) aus config.py.example

**NIEMALS aus dem Checkpoint!** Der Checkpoint könnte alte Werte haben (z.B. 128/32 von altem Training).

## VRAM-Anforderungen (ungefähr)

| N_FEATS | N_BLOCKS | Batch=1 VRAM | Empfehlung |
|---------|----------|--------------|------------|
| 64      | 24       | ~3.0 GB      | Schnell    |
| **72**  | **28**   | **~3.8 GB**  | **Standard** |
| 80      | 32       | ~4.5 GB      | Qualität   |

Die Werte sind für 7-Frame Training mit 180x180 LR Input.

## Zusammenfassung

- **N_FEATS = 72**: Feature-Kanäle in allen Layern
- **N_BLOCKS = 28**: ResidualBlocks gesamt (14 + 14)
- **Fusion-Layer**: Zusätzlich 3 FusionBlocks (nicht in N_BLOCKS)
- **Gesamt**: ~66 Conv2d Layer im kompletten Modell
- **Quelle**: Immer config.py, niemals Checkpoint!
