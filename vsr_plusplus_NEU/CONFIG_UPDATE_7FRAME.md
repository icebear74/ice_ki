# config_p4_optimized.py - 7-Frame VSR Configuration

## Änderungen / Changes

### Modell-Parameter / Model Parameters
- **N_FEATS**: 64 → **72** (für 7-Frame-Modell optimiert)
- **N_BLOCKS**: 24 → **26** (erhöhte Kapazität für bessere Qualität)

### Batch-Konfiguration / Batch Configuration
- **BATCH_SIZE**: 2 → **1** (VRAM-sicher, getestet bei ~3.77 GB)
- **ACCUMULATION_STEPS**: 8 → **6** (effektive Batch-Größe: 6)

### Dataset-Pfade / Dataset Paths
Die Pfade wurden aktualisiert, um mit der `dataset_generator_v2` Konfiguration übereinzustimmen:

**Alt / Old:**
```python
DATA_ROOT = "/mnt/data/training/Universal/Mastermodell/Learn"
DATASET_ROOT = "/mnt/data/training/Dataset/Universal/Mastermodell"
```

**Neu / New:**
```python
DATA_ROOT = "/mnt/data/training/datasetNeu/master"
DATASET_ROOT = "/mnt/data/training/datasetNeu"
```

## Dataset-Struktur / Dataset Structure

Die erwartete Verzeichnisstruktur entspricht der Ausgabe von `dataset_generator_v2`:

```
/mnt/data/training/datasetNeu/
└── master/
    ├── train/
    │   ├── 5frames/
    │   │   ├── small_540/    # 540×540 GT, 180×180 LR
    │   │   ├── medium_169/   # 720×405 GT, 240×135 LR
    │   │   └── large_720/    # 720×720 GT, 240×240 LR
    │   └── 7frames/
    │       ├── small_540/    # 540×540 GT, 180×180 LR
    │       ├── medium_169/   # 720×405 GT, 240×135 LR
    │       └── large_720/    # 720×720 GT, 240×240 LR
    └── val/
        └── 7frames/
            └── large_720/    # 720×720 für Validierung
```

## Verwendung / Usage

### 1. Konfiguration anzeigen / Display Configuration
```bash
python3 vsr_plusplus_NEU/config_p4_optimized.py
```

### 2. In Training verwenden / Use in Training
```python
import vsr_plusplus_NEU.config_p4_optimized as cfg

config = cfg.get_config()
# config enthält alle Parameter
```

### 3. Konfiguration verifizieren / Verify Configuration
```python
from vsr_plusplus_NEU import config_p4_optimized as cfg

# Modell-Parameter prüfen
print(f"Features: {cfg.N_FEATS}")  # Sollte 72 sein
print(f"Blocks: {cfg.N_BLOCKS}")   # Sollte 26 sein

# Dataset-Pfade prüfen
print(f"Data Root: {cfg.DATA_ROOT}")
print(f"Dataset Root: {cfg.DATASET_ROOT}")
```

## VRAM-Anforderungen / VRAM Requirements

### Getestet / Tested:
- **Batch Size 1**: ~3.77 GB ✅ (sicher unter 6.5 GB Limit)
- **7-Frame Modell**: 72 Features, 26 Blocks
- **Image Sizes**: 
  - 540×540: ~3.77 GB
  - 720×405: ~3.77 GB
  - 720×720: ~3.77 GB (mit Akkumulation)

### Empfohlene Einstellungen / Recommended Settings:
- **BATCH_SIZE**: 1 (für alle Größen)
- **ACCUMULATION_STEPS**: 6
- **Effektive Batch-Größe**: 6
- **VRAM-Sicherheitsmarge**: ~2.73 GB

## Kompatibilität / Compatibility

### Dataset Generator V2
Diese Konfiguration ist vollständig kompatibel mit der Ausgabe von `dataset_generator_v2`:

**generator_config.json:**
```json
{
  "base_settings": {
    "output_base_dir": "/mnt/data/training/datasetNeu",
    "lr_versions": ["5frames", "7frames"]
  },
  "category_targets": {
    "master": 300000
  }
}
```

### 7-Frame VSR System
Diese Konfiguration ist abgestimmt auf das 7-Frame VSR Training System:
- Adaptive Batch Management
- Runtime Configuration
- Size Tracking
- Web GUI
- Terminal GUI

## Wichtige Hinweise / Important Notes

1. **VRAM-Limit**: Die Konfiguration bleibt unter 6.5 GB VRAM
2. **Batch=1**: Sicherste Einstellung für alle Bildgrößen
3. **Gradient Accumulation**: Ermöglicht effektive Batch-Größe von 6
4. **Dataset-Pfade**: Müssen mit dataset_generator_v2 übereinstimmen
5. **Mixed Precision (AMP)**: Aktiviert für schnelleres Training

## Fehlerbehebung / Troubleshooting

### Dataset nicht gefunden / Dataset not found
Überprüfen Sie, ob die Verzeichnisse existieren:
```bash
ls -la /mnt/data/training/datasetNeu/master/train/7frames/
```

### VRAM-Fehler / VRAM errors
- Verwenden Sie BATCH_SIZE = 1
- Reduzieren Sie bei Bedarf N_FEATS oder N_BLOCKS
- Überprüfen Sie, ob andere Prozesse VRAM verwenden

### Pfad-Fehler / Path errors
Stellen Sie sicher, dass DATA_ROOT auf das `master` Verzeichnis zeigt:
```python
DATA_ROOT = "/mnt/data/training/datasetNeu/master"
```

## Vergleich Alt vs. Neu / Comparison Old vs. New

| Parameter | Alt / Old | Neu / New | Grund / Reason |
|-----------|-----------|-----------|----------------|
| N_FEATS | 64 | 72 | 7-Frame-Modell Optimierung |
| N_BLOCKS | 24 | 26 | Bessere Qualität |
| BATCH_SIZE | 2 | 1 | VRAM-Sicherheit |
| ACCUMULATION_STEPS | 8 | 6 | Effektive Batch 6 |
| DATA_ROOT | Universal/Mastermodell/Learn | datasetNeu/master | Generator V2 |
| DATASET_ROOT | Dataset/Universal/Mastermodell | datasetNeu | Generator V2 |

## Weitere Informationen / More Information

Siehe auch / See also:
- `README_7FRAME.md` - Vollständige 7-Frame System Dokumentation
- `IMPLEMENTATION_COMPLETE.md` - Implementierungsdetails
- `dataset_generator_v2/README.md` - Dataset Generator Dokumentation
