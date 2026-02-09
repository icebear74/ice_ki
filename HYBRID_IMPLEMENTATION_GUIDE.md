# Dataset Generator V2 - Hybrid Implementation Guide

## Überblick (Overview)

Dieses Dokument beschreibt die **Hybrid-Implementierung**, die alle Features des Originals mit den UHD-Qualitätsverbesserungen kombiniert.

This document describes the **hybrid implementation** that combines all original features with UHD quality improvements.

## Implementierungen (Implementations)

### 1. `make_dataset_v2_clean.py` - Simplified Version
**Zweck**: Einfache, saubere Implementierung für neue Projekte
- Vereinfachte Konfiguration (generator_config_v2.json)
- State Management mit Resume
- UHD-Qualität
- 7-Frame nur

**Use when**: Starting new dataset from scratch

### 2. `make_dataset_v2_uhd.py` - Hybrid Version ⭐ **EMPFOHLEN**
**Zweck**: Vollständige Features + UHD-Qualität
- **Alle Original-Features:**
  - GUI mit Rich Display
  - Prioritätssystem (0-255)
  - Multi-Kategorie (master, universal, space, toon)
  - Komplette Videoliste (100+ Videos)
  - Progress Tracking & Persistence
  - 5-Frame UND 7-Frame Support
- **Plus UHD-Verbesserungen:**
  - Tonemap ohne Downscaling
  - Random Cropping von vollem UHD
  - DVD-realistisches LR (INTER_AREA)

**Use when**: Using original generator_config.json with complete video set

### 3. `make_dataset_multi.py` - Original
**Zweck**: Original-Implementierung (Referenz)
- Alle Features
- ABER: Qualitätsverlust durch HD-Downscaling

**Use when**: Reference only (deprecated for new datasets)

## Konfigurationsdateien (Configuration Files)

### Original: `generator_config.json` (3919 Zeilen)
```json
{
  "base_settings": {
    "base_frame_limit": 3000,
    "max_workers": 4,
    "output_base_dir": "/mnt/data/training/datasetNeu",
    "lr_versions": ["5frames", "7frames"]
  },
  "category_targets": {
    "master": 300000,
    "universal": 350000,
    "space": 160000,
    "toon": 90000
  },
  "format_config": {
    "master": {
      "small_540": {...},
      "medium_169": {...},  // 16:9 = 720×405
      "large_720": {...}
    },
    ...
  },
  "videos": [
    {
      "name": "Avatar",
      "path": "/mnt/data/video/filme/UHD/Avatar.mkv",
      "categories": {
        "master": 0.2,
        "universal": 0.2,
        "space": 0.3,
        "toon": 0.3
      },
      "priority": 0  // Optional
    },
    ...  // 100+ videos
  ]
}
```

### Neu (Simplified): `generator_config_v2.json` (70 Zeilen)
```json
{
  "dataset_name": "master",
  "root_path": "/mnt/data/training/datasetNeu",
  "source": {
    "categories": {
      "master": {"video_dir": "/path/to/UHD"},
      "universal": {"video_dir": "/path/to/HD"}
    },
    "category_weights": {
      "master": 0.25,
      "universal": 0.75
    }
  },
  "processing": {
    "total_patches": 100000,
    "n_frames": 7
  }
}
```

## Feature-Vergleich (Feature Comparison)

| Feature | Original | Simplified | Hybrid (UHD) |
|---------|----------|------------|--------------|
| **Video-Liste** | ✅ 100+ | ❌ Auto-Scan | ✅ 100+ |
| **Prioritäten** | ✅ 0-255 | ❌ | ✅ 0-255 |
| **Multi-Kategorie** | ✅ 4 (master/universal/space/toon) | ✅ 2 (master/universal) | ✅ 4 |
| **Category Targets** | ✅ Pro Kategorie | ✅ Total | ✅ Pro Kategorie |
| **GUI (Rich)** | ✅ | ✅ Basic | ✅ Full |
| **Progress Tracking** | ✅ | ✅ | ✅ |
| **Resume** | ✅ | ✅ | ✅ |
| **5-Frame Support** | ✅ | ❌ | ✅ |
| **7-Frame Support** | ✅ | ✅ | ✅ |
| **UHD Quality** | ❌ (HD resize) | ✅ | ✅ |
| **Random Crop** | ❌ Center | ✅ Anywhere | ✅ Anywhere |
| **LR Quality** | INTER_CUBIC | INTER_AREA | INTER_AREA |

## 16:9 Aspektverhältnis Überprüfung

**medium_169 Format:**
```python
GT: 720 × 405
LR: 240 × 135

# Verify
720 / 405 = 1.777... = 16/9 ✅
240 / 135 = 1.777... = 16/9 ✅
```

**Korrekt in allen Implementierungen!**

## Empfohlene Verwendung (Recommended Usage)

### Für Bestehende Projekte (For Existing Projects)
**Mit kompletter Videoliste:**
```bash
cd dataset_generator_v2
python make_dataset_v2_uhd.py ../generator_config.json
```

### Für Neue Projekte (For New Projects)
**Vereinfachte Konfiguration:**
```bash
cd dataset_generator_v2
python make_dataset_v2_clean.py generator_config_v2.json
```

## Verzeichnisstruktur (Directory Structure)

### Original/Hybrid (make_dataset_v2_uhd.py):
```
/mnt/data/training/datasetNeu/
├── master/
│   ├── train/
│   │   ├── 5frames/
│   │   │   ├── small_540/ {GT, LR}
│   │   │   ├── medium_169/ {GT, LR}
│   │   │   └── large_720/ {GT, LR}
│   │   └── 7frames/
│   │       ├── small_540/ {GT, LR}
│   │       ├── medium_169/ {GT, LR}
│   │       └── large_720/ {GT, LR}
├── universal/
│   └── train/ ...
├── space/
│   └── train/ ...
└── toon/
    └── train/ ...
```

### Simplified (make_dataset_v2_clean.py):
```
/mnt/data/training/datasetNeu/master/
├── patches/
│   ├── 720/ {GT, LR}
│   ├── 540/ {GT, LR}
│   └── 720_169/ {GT, LR}
└── val/
    └── ...
```

## Migration von Original zu Hybrid

**Keine Migration nötig!** Die Hybrid-Version verwendet die **gleiche Konfiguration** wie das Original.

Simply use:
```bash
python make_dataset_v2_uhd.py generator_config.json
```

## Qualitätsverbesserungen (Quality Improvements)

### Vorher (Before - Original):
```
UHD (3840×2160) 
  → HD Resize (1920×1080)  ❌ 75% Detailverlust
  → Center Crop (720×720)
  → LR (240×240)
```

### Nachher (After - Hybrid):
```
UHD (3840×2160)
  → Random Crop (720×720)  ✅ Volle UHD-Details
  → LR (240×240) mit INTER_AREA
```

## Testen (Testing)

### Hybrid-Version testen:
```bash
# Mit Original-Config
python make_dataset_v2_uhd.py ../generator_config.json

# Überprüfen:
# - Laden alle Videos?
# - Werden Prioritäten respektiert?
# - Funktioniert GUI?
# - Werden alle 4 Kategorien erstellt?
# - Ist 16:9 korrekt (720×405)?
```

## Zusammenfassung (Summary)

**Empfehlung**: Verwenden Sie `make_dataset_v2_uhd.py` mit der originalen `generator_config.json`

**Warum?**
- ✅ Alle Original-Features (100+ Videos, Prioritäten, 4 Kategorien)
- ✅ UHD-Qualität (kein Qualitätsverlust)
- ✅ Kompatibel mit bestehender Konfiguration
- ✅ Einfacher Wechsel vom Original
- ✅ 16:9 korrekt (720×405)

**Für neue, vereinfachte Projekte**: `make_dataset_v2_clean.py`
