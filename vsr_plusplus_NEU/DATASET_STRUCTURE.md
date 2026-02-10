# Dataset Struktur - Generator V2 und Training System

## Problem gelöst: Dataset Pfade korrigiert ✅

### Dataset Generator V2 Ausgabe-Struktur

Der `dataset_generator_v2` erstellt folgende Struktur (nur 7-frame Version):

```
/mnt/data/training/datasetNeu/          ← output_base_dir (generator_config.json)
└── master/                              ← Kategorie (flat lowercase)
    ├── patches/540/                     ← small_540 Format (size_key: 540)
    │   ├── GT/                          ← 540×540 Ground Truth
    │   └── LR_7frames/                  ← 180×1260 (7 frames gestackt vertikal)
    │
    ├── patches/720_169/                 ← medium_169 Format (size_key: 720_169)
    │   ├── GT/                          ← 405×720 Ground Truth (16:9)
    │   └── LR_7frames/                  ← 135×2352 (7 frames gestackt vertikal)
    │
    ├── patches/720/                     ← large_720 Format (size_key: 720)
    │   ├── GT/                          ← 720×720 Ground Truth
    │   └── LR_7frames/                  ← 240×1680 (7 frames gestackt vertikal)
    │
    └── Val/                             ← Validation (flat, vom Generator erstellt)
        └── GT/                          ← Mixed sizes (nicht genutzt)
```

**Wichtig**: 
- Alle LR-Daten sind 7-frame Versionen (vertikal gestackt)
- Der Generator erstellt `Val/GT/` (flat, mit großem V), aber das Training nutzt eine andere Struktur
- Validation-Dateien werden **manuell** in die korrekte Struktur kopiert (siehe unten)

### Andere Kategorien

Generator V2 unterstützt mehrere Kategorien (siehe `utils/format_definitions.py`):

- **master**: `master/` (flat, lowercase)
- **universal**: `universal/` (flat, lowercase)
- **space**: `space/` (flat, lowercase)
- **toon**: `toon/` (flat, lowercase)

### VSR Training System Erwartung

Das `VSRDataset` (in `vsr_plus_plus/core/dataset.py`) erwartet:

```python
dataset_root/              # = root Parameter in VSRDataset
└── master/                # = dataset_name Parameter (lowercase)
    ├── patches/540/       # = size_key Parameter (z.B. '540', '720', '720_169')
    │   ├── GT/            # Training Ground Truth
    │   └── LR_7frames/    # Training Low Resolution (7-frame stack)
    │
    └── val/540/           # Validation für size_key '540'
        ├── GT/            # Validation Ground Truth
        └── LR_7frames/    # Optional: Validation LR (falls back to patches/540/LR_7frames)
```

**Wichtig**: Die Validation-Struktur muss jetzt **size-spezifisch** sein:
- `val/540/GT/` für 540×540 Patches (mit `val/540/LR_7frames/` optional)
- `val/720/GT/` für 720×720 Patches (mit `val/720/LR_7frames/` optional)
- `val/720_169/GT/` für 720×405 (16:9) Patches (mit `val/720_169/LR_7frames/` optional)

### Richtige Konfiguration

In `runtime_config.json` oder beim Initialisieren von VSRDataset:

```python
# RICHTIG ✅
dataset = VSRDataset(
    root="/mnt/data/training/datasetNeu",
    dataset_name="master",      # lowercase
    size_key="540",             # oder '720', '720_169'
    mode="train"
)

val_dataset = VSRDataset(
    root="/mnt/data/training/datasetNeu",
    dataset_name="master",
    size_key="540",
    mode="val"
)

# Dies erwartet folgende Struktur:
# /mnt/data/training/datasetNeu/master/patches/540/GT/
# /mnt/data/training/datasetNeu/master/patches/540/LR_7frames/
# /mnt/data/training/datasetNeu/master/val/540/GT/
# /mnt/data/training/datasetNeu/master/val/540/LR_7frames/ (optional)
```

### VAL Datenstruktur Übersicht

Die **Validation (VAL)** Daten müssen wie folgt strukturiert sein:

```
/mnt/data/training/datasetNeu/master/val/
├── 540/
│   ├── GT/              ← Hier Ground Truth Bilder für 540×540 reinlegen
│   └── LR_7frames/      ← Optional: LR Versionen (sonst wird patches/540/LR_7frames genutzt)
├── 720/
│   ├── GT/              ← Hier Ground Truth Bilder für 720×720 reinlegen
│   └── LR_7frames/      ← Optional
└── 720_169/
    ├── GT/              ← Hier Ground Truth Bilder für 720×405 (16:9) reinlegen
    └── LR_7frames/      ← Optional
```

**Beispiel für val/540/GT:**
```
val/540/GT/
├── val_image_001.png    (540×540 Pixel)
├── val_image_002.png    (540×540 Pixel)
├── val_image_003.png    (540×540 Pixel)
└── ...
```

**Wichtig:**
- Jede size_key hat ihr eigenes Validierungs-Verzeichnis
- GT-Bilder müssen die korrekte Größe haben (540×540 für '540', 720×720 für '720', etc.)
- LR_7frames ist optional - wenn nicht vorhanden, wird automatisch auf `patches/{size_key}/LR_7frames` zurückgegriffen

### Warum dieser Pfad?

1. **Generator V2** erstellt: `datasetNeu/master/patches/540/GT/` und `datasetNeu/master/Val/GT/` (flat)
2. **VSRDataset** erwartet: `root/dataset_name/patches/size_key/GT/` und `root/dataset_name/val/size_key/GT/`
3. **Lösung**: Validation-Dateien müssen manuell in die size-spezifischen Verzeichnisse kopiert werden

### Verifikation

Überprüfen Sie die Struktur:

```bash
# Training-Daten (vom Generator erstellt):
ls /mnt/data/training/datasetNeu/master/patches/540/GT/
ls /mnt/data/training/datasetNeu/master/patches/540/LR_7frames/

# Validation-Daten (manuell erstellt):
ls /mnt/data/training/datasetNeu/master/val/540/GT/
```

### Generator Konfiguration

In `dataset_generator_v2/generator_config.json`:

```json
{
  "base_settings": {
    "output_base_dir": "/mnt/data/training/datasetNeu",
    "lr_versions": ["7frames"]
  },
  "category_targets": {
    "master": 300000
  }
}
```

**Hinweis**: Der Generator erstellt nur `Val/GT/` (flat), aber das Training benötigt `val/{size_key}/GT/`.
Die Validation-Dateien müssen manuell in die korrekte Struktur kopiert werden.

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
