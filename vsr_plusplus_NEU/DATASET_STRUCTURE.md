# Dataset Struktur - Generator V2 und Training System

## Problem gelöst: Dataset Pfade korrigiert ✅

### Dataset Generator V2 Ausgabe-Struktur

Der `dataset_generator_v2` erstellt folgende Struktur (nur 7-frame Version):

```
/mnt/data/training/datasetNeu/          ← output_base_dir (generator_config.json)
└── master/                              ← Kategorie (flat lowercase)
    ├── patches/540/                     ← small_540 Format (size_key: 540)
    │   ├── GT/                          ← 540×540 Ground Truth
    │   └── LR_7frames/                  ← 1260×180 (7 frames gestackt vertikal)
    │
    ├── patches/720_169/                 ← medium_169 Format (size_key: 720_169)
    │   ├── GT/                          ← 405×720 Ground Truth (16:9)
    │   └── LR_7frames/                  ← 945×240 (7 frames gestackt vertikal)
    │
    ├── patches/720/                     ← large_720 Format (size_key: 720)
    │   ├── GT/                          ← 720×720 Ground Truth
    │   └── LR_7frames/                  ← 1680×240 (7 frames gestackt vertikal)
    │
    └── Val/                             ← Validation (flat, vom Generator erstellt)
        └── GT/                          ← Mixed sizes (nicht genutzt)
```

**Wichtig**: 
- Alle LR-Daten sind 7-frame Versionen (vertikal gestackt: Höhe = einzelne_Höhe × 7)
- 540×540 GT → 1260×180 LR_7frames (Höhe: (540/3)×7=180×7=1260, Breite: 540/3=180)
- 720×720 GT → 1680×240 LR_7frames (Höhe: (720/3)×7=240×7=1680, Breite: 720/3=240)
- 405×720 GT → 945×240 LR_7frames (Höhe: (405/3)×7=135×7=945, Breite: 720/3=240)
- Der Generator erstellt `Val/GT/` (mit großem V), aber das Training erwartet `val/{size_key}/GT/` (lowercase)
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
    └── val/               # Validation
        └── GT/            # Validation Ground Truth (organisiert nach size)
            ├── 540/       # GT für 540×540 Patches
            ├── 720/       # GT für 720×720 Patches
            └── 720_169/   # GT für 720×405 (16:9) Patches
```

**Wichtig**: Die neue Validation-Struktur organisiert GT-Bilder unter `val/GT/{size_key}/`:
- `val/GT/540/` für 540×540 Validation Patches
- `val/GT/720/` für 720×720 Validation Patches
- `val/GT/720_169/` für 720×405 (16:9) Validation Patches
- LR-Bilder werden **immer** aus `patches/{size_key}/LR_7frames/` geladen

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
# /mnt/data/training/datasetNeu/master/val/GT/540/
# (LR wird automatisch aus patches/540/LR_7frames/ geladen)
```

### VAL Datenstruktur Übersicht

Die **Validation (VAL)** Daten müssen wie folgt strukturiert sein:

```
/mnt/data/training/datasetNeu/master/val/
└── GT/                    ← Validation Ground Truth Verzeichnis
    ├── 540/               ← Hier Ground Truth Bilder für 540×540 reinlegen
    ├── 720/               ← Hier Ground Truth Bilder für 720×720 reinlegen
    └── 720_169/           ← Hier Ground Truth Bilder für 720×405 (16:9) reinlegen
```

**Workflow für Validation-Daten:**
1. **GT kopieren**: Validation Ground Truth Bilder manuell nach `val/GT/{size_key}/` kopieren
2. **LR automatisch**: Das Training findet automatisch die entsprechenden LR Bilder in `patches/{size_key}/LR_7frames/`

**Beispiel:**
```bash
# Sie kopieren nur GT:
cp some_image.png /mnt/data/training/datasetNeu/master/val/GT/540/

# Training findet automatisch das LR hier:
# /mnt/data/training/datasetNeu/master/patches/540/LR_7frames/some_image.png
```

**Wichtig:**
- Sie müssen **nur GT-Bilder** nach `val/GT/{size_key}/` kopieren
- LR-Bilder werden automatisch aus `patches/{size_key}/LR_7frames/` geladen
- Die GT- und LR-Dateinamen müssen **identisch** sein (z.B. beide `image001.png`)
- Alle size_keys sind unter `val/GT/` organisiert

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

**Validation Setup (Beispiel):**
```bash
# 1. Erstellen Sie die Verzeichnisse
mkdir -p /mnt/data/training/datasetNeu/master/val/540/GT
mkdir -p /mnt/data/training/datasetNeu/master/val/720/GT
mkdir -p /mnt/data/training/datasetNeu/master/val/720_169/GT

# 2. Kopieren Sie Validation GT-Bilder aus den Training-Patches
# (wählen Sie gute, repräsentative Bilder aus)
cp /mnt/data/training/datasetNeu/master/patches/540/GT/some_good_image.png \
   /mnt/data/training/datasetNeu/master/val/540/GT/

# 3. LR-Bilder werden automatisch gefunden!
# Training findet automatisch:
# /mnt/data/training/datasetNeu/master/patches/540/LR_7frames/some_good_image.png
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
