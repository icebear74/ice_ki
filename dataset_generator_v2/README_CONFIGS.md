# Konfigurations-Dateien / Configuration Files

## Übersicht / Overview

Es gibt **2 verschiedene Config-Dateien** für verschiedene Anwendungsfälle:

There are **2 different config files** for different use cases:

---

## 1. `generator_config.json` (120 KB) ⭐ **EMPFOHLEN / RECOMMENDED**

### Für / For: `make_dataset_v2_uhd.py`

**Deutsch:**
- **Komplette Videoliste**: 467 Videos mit allen Pfaden
- **4 Kategorien**: master, universal, space, toon
- **Prioritätssystem**: Videos mit Priorität 0-255
- **Kategorie-Targets**: master: 300k, universal: 350k, space: 160k, toon: 90k
- **Format-Konfigurationen**: 3 Formate pro Kategorie (small_540, medium_169, large_720)
- **5-Frame UND 7-Frame** Support

**English:**
- **Complete video list**: 467 videos with all paths
- **4 categories**: master, universal, space, toon
- **Priority system**: Videos with priority 0-255
- **Category targets**: master: 300k, universal: 350k, space: 160k, toon: 90k
- **Format configurations**: 3 formats per category (small_540, medium_169, large_720)
- **5-frame AND 7-frame** support

### Verwendung / Usage:

```bash
cd dataset_generator_v2
python make_dataset_v2_uhd.py ../generator_config.json
```

**ODER / OR** (wenn im Root-Verzeichnis / if in root directory):

```bash
cd /home/runner/work/ice_ki/ice_ki
python dataset_generator_v2/make_dataset_v2_uhd.py generator_config.json
```

### Struktur / Structure:

```json
{
  "base_settings": {
    "output_base_dir": "/mnt/data/training/datasetNeu",
    "max_workers": 4,
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
      "medium_169": {...},
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
      "priority": 0
    },
    ... // 467 videos total
  ]
}
```

---

## 2. `generator_config_v2.json` (1.5 KB)

### Für / For: `make_dataset_v2_clean.py`

**Deutsch:**
- **Vereinfachte Konfiguration** für neue Projekte
- **2 Kategorien**: master, universal
- **Automatische Video-Suche** in Verzeichnissen
- **Nur 7-Frame** Support
- **Flache Verzeichnisstruktur**: patches/720/, patches/540/, etc.

**English:**
- **Simplified configuration** for new projects
- **2 categories**: master, universal
- **Automatic video scanning** in directories
- **7-frame only** support
- **Flat directory structure**: patches/720/, patches/540/, etc.

### Verwendung / Usage:

```bash
cd dataset_generator_v2
python make_dataset_v2_clean.py generator_config_v2.json
```

### Struktur / Structure:

```json
{
  "dataset_name": "master",
  "root_path": "/mnt/data/training/datasetNeu",
  "source": {
    "categories": {
      "master": {
        "video_dir": "/mnt/data/video/SerieUHD/Planet Earth 2",
        "extensions": [".mkv", ".mp4", ".avi"]
      },
      "universal": {
        "video_dir": "/mnt/data/video/Serie",
        "extensions": [".mkv", ".mp4", ".avi"]
      }
    },
    "category_weights": {
      "master": 0.25,
      "universal": 0.75
    }
  },
  "output_patches": {
    "720": {"gt_size": [720, 720], ...},
    "540": {"gt_size": [540, 540], ...},
    "720_169": {"gt_size": [720, 405], ...}
  },
  "processing": {
    "total_patches": 100000,
    "n_frames": 7
  }
}
```

---

## Welche Datei verwenden? / Which File to Use?

### ✅ Verwende `generator_config.json` wenn / Use `generator_config.json` if:

- Du die **komplette Videoliste** (467 Videos) verwenden willst
- Du **4 Kategorien** brauchst (master, universal, space, toon)
- Du **Prioritäten** für Videos setzen willst
- Du **5-Frame UND 7-Frame** Support brauchst
- Du die **original Verzeichnisstruktur** willst

**→ Für Produktion empfohlen! / Recommended for production!**

### ✅ Verwende `generator_config_v2.json` wenn / Use `generator_config_v2.json` if:

- Du ein **neues, einfaches Projekt** startest
- Du nur **2 Kategorien** brauchst
- Du Videos **automatisch scannen** lassen willst (keine Liste pflegen)
- Du nur **7-Frame** brauchst
- Du die **neue, flache Verzeichnisstruktur** willst

**→ Für neue, einfache Projekte! / For new, simple projects!**

---

## Vergleichstabelle / Comparison Table

| Feature | generator_config.json | generator_config_v2.json |
|---------|----------------------|--------------------------|
| **Dateigröße** | 120 KB | 1.5 KB |
| **Videos** | 467 (Liste) | Auto-Scan |
| **Kategorien** | 4 | 2 |
| **Prioritäten** | ✅ Ja | ❌ Nein |
| **5-Frame** | ✅ Ja | ❌ Nein |
| **7-Frame** | ✅ Ja | ✅ Ja |
| **Format** | small/medium/large | 720/540/720_169 |
| **Verzeichnis** | train/5frames/, train/7frames/ | patches/ |
| **Verwendung mit** | make_dataset_v2_uhd.py | make_dataset_v2_clean.py |

---

## Häufige Fehler / Common Errors

### ❌ Fehler: "AttributeError: 'DatasetGeneratorV2UHD' object has no attribute 'base_settings'"

**Ursache / Cause:** Falsche Config-Datei verwendet

**Lösung / Solution:** 
- `make_dataset_v2_uhd.py` braucht `generator_config.json`
- `make_dataset_v2_clean.py` braucht `generator_config_v2.json`

### ❌ Fehler: "KeyError: 'videos'"

**Ursache / Cause:** `generator_config_v2.json` mit `make_dataset_v2_uhd.py` verwendet

**Lösung / Solution:** Verwende `generator_config.json` stattdessen

### ❌ Fehler: "KeyError: 'base_settings'"

**Ursache / Cause:** `generator_config.json` mit `make_dataset_v2_clean.py` verwendet

**Lösung / Solution:** Verwende `generator_config_v2.json` stattdessen

---

## Schnellreferenz / Quick Reference

```bash
# PRODUCTION (467 Videos, 4 Kategorien, Prioritäten)
cd dataset_generator_v2
python make_dataset_v2_uhd.py ../generator_config.json

# SIMPLIFIED (Neue Projekte, 2 Kategorien, Auto-Scan)
cd dataset_generator_v2
python make_dataset_v2_clean.py generator_config_v2.json
```

---

## Dateien / Files

- ✅ `/generator_config.json` - Root-Verzeichnis (ORIGINAL)
- ✅ `/dataset_generator_v2/generator_config.json` - Symlink zum Original
- ✅ `/dataset_generator_v2/generator_config_v2.json` - Vereinfachte Version

**Hinweis:** Die Datei `dataset_generator_v2/generator_config.json` ist ein Symlink/Kopie 
zur Root-Datei `generator_config.json` für einfacheren Zugriff.

**Note:** The file `dataset_generator_v2/generator_config.json` is a symlink/copy 
of the root file `generator_config.json` for easier access.
