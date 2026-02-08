# Dataset Struktur - Generator V2 und Training System

## Problem gelöst: Dataset Pfade korrigiert ✅

### Dataset Generator V2 Ausgabe-Struktur

Der `dataset_generator_v2` erstellt folgende Struktur:

```
/mnt/data/training/datasetNeu/          ← output_base_dir (generator_config.json)
└── Master/                              ← Kategorie
    └── MasterModel/
        └── Learn/
            ├── Patches/                 ← small_540 Format
            │   ├── GT/                  ← 540×540 Ground Truth
            │   ├── LR/                  ← 180×900 (5 frames gestackt)
            │   └── LR_7frames/          ← 180×1260 (7 frames gestackt)
            │
            ├── Patches_Medium169/       ← medium_169 Format
            │   ├── GT/                  ← 720×405 Ground Truth
            │   ├── LR/                  ← 240×135 (5 frames)
            │   └── LR_7frames/          ← 240×135 (7 frames)
            │
            ├── Patches_Large/           ← large_720 Format
            │   ├── GT/                  ← 720×720 Ground Truth
            │   ├── LR/                  ← 240×240 (5 frames)
            │   └── LR_7frames/          ← 240×240 (7 frames)
            │
            └── Val/                     ← Validation
                ├── GT/
                └── LR/
```

### Andere Kategorien

Generator V2 unterstützt mehrere Kategorien (siehe `utils/format_definitions.py`):

- **master**: `Master/MasterModel/Learn/`
- **universal**: `Universal/UniversalModel/Learn/`
- **space**: `Space/SpaceModel/Learn/`
- **toon**: `Toon/ToonModel/Learn/`

### VSR Training System Erwartung

Das `VSRDataset` (in `vsr_plus_plus/core/dataset.py`) erwartet:

```python
dataset_root/              # = DATA_ROOT in config
├── Patches/
│   ├── GT/               # Training Ground Truth
│   └── LR/               # Training Low Resolution (5-frame stack)
└── Val/
    ├── GT/               # Validation Ground Truth
    └── LR/               # Validation Low Resolution
```

### Richtige Konfiguration

In `config_p4_optimized.py` (und Ihrer lokalen `config.py`):

```python
# RICHTIG ✅
DATA_ROOT = "/mnt/data/training/datasetNeu/Master/MasterModel/Learn"
DATASET_ROOT = "/mnt/data/training/datasetNeu"

# FALSCH ❌
# DATA_ROOT = "/mnt/data/training/datasetNeu/master"
```

### Warum dieser Pfad?

1. **Generator V2** erstellt: `datasetNeu/Master/MasterModel/Learn/Patches/GT/`
2. **VSRDataset** sucht nach: `DATA_ROOT/Patches/GT/`
3. **Lösung**: `DATA_ROOT` muss auf `.../Master/MasterModel/Learn` zeigen

### Verifikation

Überprüfen Sie die Struktur:

```bash
# Sollte Dateien zeigen:
ls /mnt/data/training/datasetNeu/Master/MasterModel/Learn/Patches/GT/

# Sollte Dateien zeigen:
ls /mnt/data/training/datasetNeu/Master/MasterModel/Learn/Patches/LR/
```

### Generator Konfiguration

In `dataset_generator_v2/generator_config.json`:

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

### Format-Verteilung

Für die **master** Kategorie (siehe `utils/format_definitions.py`):

- **small_540**: 50% (540×540 Patches)
- **medium_169**: 35% (720×405 Patches)
- **large_720**: 15% (720×720 Patches)

### Training Workflow

1. **Generator ausführen**:
   ```bash
   cd dataset_generator_v2
   python3 make_dataset_multi.py
   ```

2. **Config kopieren** (falls noch nicht vorhanden):
   ```bash
   cd vsr_plusplus_NEU
   cp config_p4_optimized.py config.py
   ```

3. **Training starten**:
   ```bash
   cd vsr_plusplus_NEU
   python3 train.py
   ```

### Wichtige Hinweise

- **5-frame vs 7-frame**: Standard Training nutzt 5-frame (`LR/`), 7-frame ist optional (`LR_7frames/`)
- **Validation**: Generator erstellt Validation-Daten in `Val/GT/` und `Val/LR/`
- **Mehrere Formate**: Generator kann gleichzeitig verschiedene Patch-Größen erstellen

### Fehlersuche

**Problem**: `No PNG files found in .../Patches/GT`

**Lösung**: 
1. Prüfen Sie, ob Generator gelaufen ist
2. Überprüfen Sie `DATA_ROOT` Pfad in config.py
3. Stellen Sie sicher, dass der Pfad `Master/MasterModel/Learn` enthält

**Problem**: `No valid GT-LR pairs found`

**Lösung**:
1. Prüfen Sie, ob sowohl GT als auch LR Verzeichnisse Dateien enthalten
2. Dateien müssen gleichen Namen haben (z.B. `frame_0001.png`)
3. LR-Dateien müssen gestackte Frames sein (180×900 für small_540)

## Referenzen

- Generator Konfiguration: `dataset_generator_v2/generator_config.json`
- Format Definitionen: `dataset_generator_v2/utils/format_definitions.py`
- Dataset Loader: `vsr_plus_plus/core/dataset.py`
- Training Config: `vsr_plusplus_NEU/config_p4_optimized.py`
