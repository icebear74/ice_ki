# VSR++ Training – Vollständige Analyse
## Dataset-Generator V2 (volldynamisch) vs. Trainingsprojekt `vsr_plusplus_NEU`

> **Stand:** 2026-04-27  
> **Analysierte Repos:** `icebear74/ice_ki`  
> **Betroffene Verzeichnisse:** `dataset_generator_v2/`, `vsr_plusplus_NEU/`

---

## ⚠️ Wichtigster Befund vorab: Generator ist jetzt VOLLSTÄNDIG DYNAMISCH

Der Generator kennt **keine hardgecodeten Format-Namen, Kategorie-Namen oder Verzeichnisstrukturen mehr**. Alle Entscheidungen werden zur Laufzeit aus zwei Konfigurationsdateien gelesen:

| Datei | Rolle |
|---|---|
| `dataset_generator_v2/templates.json` | Format-Templates (gt_size, lr_size, scale, aspect_ratio) + Degradation-Templates |
| `dataset_generator_v2/generator_config.json` | Machine-lokale Laufzeitkonfiguration: Kategorien, Formate, Weights, source_mode (in `.gitignore`, nicht committed) |

Der Trainer `vsr_plusplus_NEU` hingegen **hat noch hardcodierte Annahmen** über Format-Namen, Dateiendungen und Verzeichnisstrukturen. Das ist das zentrale Problem.

---

## 1. Datenlayout

### 1.1 Was der Generator V2 schreibt

**Pfad-Schema** (definiert in `utils/format_definitions.py`, Zeilen 86–108):

```
{root_path}/
└── {category}/                          ← aus config["categories"]
    ├── patches/{template_name}/
    │   ├── GT/
    │   │   ├── 0000/                    ← Bucket-Subdirectory (bis 10.000 Dateien)
    │   │   ├── 0001/
    │   │   └── ...
    │   └── LR_{n}frames/                ← z.B. LR_7frames (n aus config["processing"]["n_frames"])
    │       ├── 0000/                    ← GLEICHER Bucket-Name wie GT (immer synchron)
    │       ├── 0001/
    │       └── ...
    └── val/{template_name}/
        ├── GT/
        └── LR_{n}frames/
```

**Schlüsselpunkte:**
- `{template_name}` kommt aus `templates.json["format_templates"]` und dem `generator_config.json` – **frei wählbar**, z.B. `1152_169`, `960_43`, `960_169` oder jeder andere Name
- Bucket-Layout: `BUCKET_SIZE = 10.000` Dateien pro Bucket-Verzeichnis (4-stellige, nullgefüllte Namen: `0000`, `0001`, …)
- GT und LR teilen immer denselben Bucket-Namen (garantiert durch `get_synced_bucket_dirs()`, `format_definitions.py` Z. 33–83)
- **Dateiformat:** `bmp` (Standard) oder `png` – aus `config["output_format"]`, frei konfigurierbar
- Der Generator schreibt nach Abschluss eine **`dataset_architecture.json`** in `{root_path}/` (`make_dataset_v2_uhd.py`, Z. 1134–1208)

### 1.2 `dataset_architecture.json` – Der Selbstbeschreibungs-Mechanismus

Der Generator schreibt beim Start automatisch:

```
{root_path}/dataset_architecture.json
```

Inhalt (Beispiel mit Default-Templates):

```json
{
  "generated_at": "2026-04-27T10:00:00Z",
  "generator_version": "dataset_generator_v2",
  "root_path": "/mnt/data/training/datasetNeu4kNeu",
  "n_frames": 7,
  "output_format": "bmp",
  "category_targets": {"master": 100000},
  "categories": {
    "master": {
      "target_total": 100000,
      "formats": [
        {
          "template":      "1152_169",
          "weight":        60,
          "source_mode":   "resize",
          "gt_size":       [1152, 648],
          "lr_size":       [384, 216],
          "scale":         3,
          "aspect_ratio":  "16:9",
          "degradation_mix": {"dvd_film_balanced": 50, "classic_sitcom_sd": 50}
        },
        {
          "template":      "960_43",
          "weight":        40,
          "source_mode":   "crop",
          "gt_size":       [960, 720],
          "lr_size":       [320, 240],
          "scale":         3,
          "aspect_ratio":  "4:3",
          "degradation_mix": {"toon_sd": 100}
        }
      ]
    }
  },
  "format_templates": { ... },
  "degradation_templates": { ... }
}
```

**➡ Der Trainer SOLL diese Datei lesen**, um dynamisch zu wissen:
- welche Format-Namen existieren (→ Verzeichnisnamen)
- welche gt_size / lr_size / scale-Faktoren gelten
- wie viele Frames (n_frames)
- welche Dateiendung (output_format)

**Das tut er bisher NICHT.**

### 1.3 Aktuelle Default-Templates in `templates.json`

| Template-Name | GT-Size (W×H) | LR-Size (W×H) | Scale | Modus |
|---|---|---|---|---|
| `1152_169` | 1152 × 648 | 384 × 216 | 3 | resize (UHD→HD) |
| `960_169` | 960 × 540 | 320 × 180 | 3 | resize/crop |
| `960_43` | 960 × 720 | 320 × 240 | 3 | crop (4:3 Classic-TV) |

**Beliebige weitere Templates können in `templates.json` definiert werden** (`config_io.py` Z. 74–98 und Validator Z. 262–351). Aspekt-Ratios: `16:9`, `4:3`, `1:1`; Scale und `base_x` frei wählbar.

### 1.4 LR-Stack-Format (unveränderlich kompatibel)

Der Generator (`streaming_extractor.py`, `create_patch_pair()` Z. 1778+) stackt LR-Frames **vertikal** (`np.vstack`):

```
LR-Stack-Shape: (H_lr * n_frames, W_lr, 3)
```

Beispiele für 7 Frames und Scale 3:
| Template `gt_size` | LR-Frame | LR-Stack |
|---|---|---|
| 1152 × 648 | 384 × 216 | 1512 × 384 |
| 960 × 720 | 320 × 240 | 1680 × 320 |
| 960 × 540 | 320 × 180 | 1260 × 320 |

---

## 2. Gap-Analyse: Generator-Output vs. Trainer-Erwartung

### 🔴 GAP 1 – Format-Namen sind hardcodiert im Trainer (KRITISCH)

**Trainer erwartet:** `patches/540/GT/`, `patches/720/GT/`, `patches/720_169/GT/`  
**Generator schreibt (Defaults):** `patches/1152_169/GT/`, `patches/960_169/GT/`, `patches/960_43/GT/`  
**→ Verzeichnisse werden nie gefunden.**

Hardgecodete Stellen:

| Datei | Zeile | Code |
|---|---|---|
| `config.active.py` | 51–55 | `ADAPTIVE_BATCH_CONFIG = {'720_169': ..., '540': ..., '720': ...}` |
| `train.py` | 60 | `size_keys = ['720_169', '540', '720']` |
| `training/trainer.py` | 392, 521, 561, 577 | `display_fps = {'720': 0, '540': 0, '720_169': 0}` |
| `core/data_strategy.py` | 77–88 | `CROP_INTRO_END_DISTRIBUTION = {'720_169': 0.40, '540': 0.20, '720': 0.40}` |
| `DATASET_STRUCTURE.md` | überall | Dokumentation mit festen size_keys |

**Lösung:** Der Trainer muss `dataset_architecture.json` lesen und Format-Namen (size_keys) sowie gt_size/lr_size daraus beziehen – statt aus `config.active.py`.

---

### 🔴 GAP 2 – Bucket-Subdirectory-Layout (BLOCKIERT TRAINING)

**Dokumentiert in:** `LOADER_UPDATE_REQUIRED.md` (vollständige Migration-Anleitung bereits vorhanden)

Der Loader macht `os.listdir(self.gt_dir)` → sieht nur `['0000', '0001', ...]` → `.endswith('.png')`-Filter → **leere Liste → sofortiger Crash**.

**5 betroffene Stellen in `core/dataset.py`:**

| Zeile | Funktion |
|---|---|
| 150 | `__init__` – Initialer Scan |
| 574 | `check_for_new_files()` |
| 630 | `reload_files()` |
| 471 | `_load_index()` – Cache-Invalidierung |
| 506 | `_save_index()` – Cache speichern |

**Fertige Hilfsfunktion** (aus `LOADER_UPDATE_REQUIRED.md`):

```python
def _collect_image_files(base_dir: str) -> list[str]:
    """Unterstützt flaches Layout (alt) und Bucket-Layout (neu)."""
    SUPPORTED_EXTS = ('.png', '.bmp')
    files = []
    if not os.path.isdir(base_dir):
        return files
    entries = os.listdir(base_dir)
    bucket_dirs = sorted(
        e for e in entries
        if len(e) == 4 and e.isdigit() and os.path.isdir(os.path.join(base_dir, e))
    )
    if bucket_dirs:
        for bucket in bucket_dirs:
            bucket_path = os.path.join(base_dir, bucket)
            for f in sorted(os.listdir(bucket_path)):
                if f.lower().endswith(SUPPORTED_EXTS):
                    files.append(os.path.join(bucket, f))  # "0000/foo.bmp"
    else:
        for f in sorted(entries):
            if f.lower().endswith(SUPPORTED_EXTS):
                files.append(f)
    return files
```

---

### 🔴 GAP 3 – Dateiendung: BMP vs. PNG hardcodiert (BLOCKIERT TRAINING)

**Generator-Default:** `output_format: "bmp"` (aus `config_io.py` Z. 197, `make_dataset_v2_uhd.py` Z. 397)  
**Trainer:** `.endswith('.png')` hardcodiert an 6+ Stellen.  
**Beide Werte stehen in `dataset_architecture.json["output_format"]`.**

Betroffene Stellen:

| Datei | Zeile |
|---|---|
| `core/dataset.py` | 150, 574, 630 |
| `train.py` | 413 |
| Hilfsfunktion (s.o.) | zu erweitern |

---

### 🟡 GAP 4 – `ADAPTIVE_BATCH_CONFIG` ist format-spezifisch hardcodiert

```python
# config.active.py Z. 51–55
ADAPTIVE_BATCH_CONFIG = {
    '720_169': {'batch': 8, 'accum': 1},
    '540':     {'batch': 8, 'accum': 2},
    '720':     {'batch': 6, 'accum': 1},
}
```

Diese Keys müssen mit den tatsächlich verwendeten Format-Namen übereinstimmen. Mit dem dynamischen Generator müssen sie entweder:
- **Aus `dataset_architecture.json` generiert werden** (keine feste Zuordnung möglich)
- **Oder als generische Regel** über gt_size oder VRAM-Messung gesetzt werden (z.B.: `height * width > X → batch=4`)

---

### 🟡 GAP 5 – Modell-Architektur: PixelShuffle(3) hardcodiert auf Scale=3

**`core/model_7frame.py`, Z. 359:**
```python
self.upsample = nn.Sequential(
    nn.Conv2d(n_feats, n_feats * (3**2), 3, padding=1),
    nn.PixelShuffle(3),           # ← hardcoded scale=3
    nn.Conv2d(n_feats, 3, 3, padding=1)
)
```

Alle aktuellen Templates verwenden `"scale": 3` → **derzeit kein Problem**. Sollte ein Template mit anderen Scale-Faktoren kommen, muss das Modell angepasst werden.  
**Empfehlung:** Scale-Faktor aus `dataset_architecture.json["categories"][cat]["formats"][*]["scale"]` lesen und in den Modell-Konstruktor übergeben.

---

### ✅ KOMPATIBEL – LR-Stacking-Format

- Generator stackt 7 LR-Frames vertikal: `(H_lr*7, W_lr, 3)` ✓
- Trainer splittet: `lr[i*h_per_frame:(i+1)*h_per_frame, :, :]` zurück in 7 Frames ✓
- Modell empfängt `[B, 7, 3, H, W]` ✓

---

### ✅ KOMPATIBEL – Bidirektionale Fusion & n_frames=7

- `config_io.py` Default: `"n_frames": 7` ✓
- Modell `VSRBidirectional_7frames_3x` erwartet `T=7` fest ✓
- Bei Änderung auf 5 würde `LR_5frames/` geschrieben → Modell würde crashen

---

## 3. Modell & Konfiguration

### 3.1 Modell-Hyperparameter

| Parameter | Wert | Quelle |
|---|---|---|
| `N_FEATS` | 72 | `config.active.py` Z. 25 |
| `N_BLOCKS` | 28 | `config.active.py` Z. 29 (Code-Default 26!) |
| Scale | 3 | hardcoded in PixelShuffle |
| n_frames | 7 | hardcoded im Modellnamen + config.active.py |

> **Achtung:** `model_7frame.py` hat `def __init__(self, n_feats=72, n_blocks=26)` als Default, aber `config.active.py` setzt `N_BLOCKS=28`. Das Modell wird mit 28 Blöcken trainiert.

### 3.2 Architektur-Übersicht

```
Input: [B, 7, 3, H, W]  (7 LR-Frames)
  │
  ├── feat_extract (Conv2d 3→72, 3×3)     auf alle 7 Frames
  │
  ├── Backward Pass (F6 → F0):
  │     TemporalAlignBlock(72) → GatedFusionBlock(144→72) → 14× ResidualBlock(72)+AttentionGate
  │
  ├── Forward Pass (F0 → F6):
  │     TemporalAlignBlock(72) → GatedFusionBlock(144→72) → 14× ResidualBlock(72)+AttentionGate
  │
  ├── Finale Fusion: GatedFusionBlock(144→72)
  │
  └── Upsample: Conv(72→72*9) → PixelShuffle(3) → Conv(72→3) + Bilinear-Residual
  
Output: [B, 3, H*3, W*3]  (1 HR-Frame, Center-Frame F3 upscaled)
```

**Attention-Gate:** in jedem `ResidualBlock`, sigmoid auf concat(gate_feat, skip_feat) → skaliert Skip-Connection.  
**Gradient Checkpointing:** aktiviert (`USE_CHECKPOINTING=True`), reduziert VRAM um ~40%.

---

## 4. Training Loop & Losses

### 4.1 Loss-Funktionen

| Loss | Gewicht (config.active.py) | Beschreibung |
|---|---|---|
| **L1** | 0.60 | Pixel-Level L1(pred, gt) |
| **Multi-Scale** | 0.20 | L1 nach AvgPool 2× und 4× |
| **Gradient** | 0.20 | Spatial-Gradient L1 (H+V) |
| **Perceptual (VGG16)** | 0.00 (Standard) | relu1_2, relu2_2, relu3_3, relu4_3 |

Alle Weights werden normalisiert (Summe=1.0 enforced).  
Perceptual-Weight wird durch `DataStrategyScheduler` graduiert von 0.0 → 0.08 eingeführt.

### 4.2 Optimizer & Scheduler

**Optimizer:** `AdamW` mit **layer-wise Learning Rates** (`train.py` Z. 310–351):
- Standard-Parameter: `lr × 1.0`
- Temporal-Align + GatedFusion: `lr × 5.0`
- Finale GatedFusion: `lr × 20.0`

**LR-Scheduler** (`training/lr_scheduler.py`, `AdaptiveLRScheduler`):

| Phase | Verhalten |
|---|---|
| Warmup (0–2000 Steps) | Linear 0 → MAX_LR (1.5e-4) |
| Stable | Konstant bis Plateau erkannt |
| Plateau | LR-Boost oder ×0.5/×0.7 Reduktion |
| Minimum | MIN_LR = 1e-5 |

**Gradient Clipping:** Adaptiv, Initialwert 3.0 (`INITIAL_GRAD_CLIP`).  
**EMA:** Nur für GUI-Smoothing (Faktor 0.95), kein Modell-EMA.

### 4.3 AMP / Mixed Precision

```python
USE_AMP = True   # Tesla P100 FP16 (18.7 TFLOPS FP16 vs 9.3 TFLOPS FP32)
scaler = GradScaler('cuda')
with autocast('cuda', enabled=use_amp):
    output = model(lr_stack)
```

`TemporalAlignBlock.forward()` castet `grid_sample` explizit auf `float32` (AMP-safe, `model_7frame.py` Z. 159–174).

### 4.4 Graduated Data Strategy (`DataStrategyScheduler`)

| Phase | Steps | Daten | Perceptual-Weight |
|---|---|---|---|
| 1 – Warmup | 0–3000 | 100% 720_169 (Full-Frames) | 0.0 |
| 2 – Crop Introduction | 3000–8000 | Linearer Übergang zu Crops | 0.0 → 0.08 |
| 3 – Stable | 8000+ | File-Count-proportional | AdaptiveSystem kontrolliert |

**Problem:** Phase 1/2/3-Distribution (`data_strategy.py` Z. 77–88) ist **hardcodiert auf `720_169`, `540`, `720`** – muss bei neuen Format-Namen angepasst werden.

---

## 5. Konfiguration

### 5.1 Trainer-Konfiguration (`config.active.py`)

```python
N_FEATS = 72
N_BLOCKS = 28
ADAPTIVE_BATCH_CONFIG = {
    '720_169': {'batch': 8, 'accum': 1},   # eff=8 | ~5.14 GB
    '540':     {'batch': 8, 'accum': 2},   # eff=16 | ~5.15 GB
    '720':     {'batch': 6, 'accum': 1},   # eff=6 | ~6.14 GB
}
DATASET_ROOT = "/mnt/data/training/datasetNeu4kNeu"
DEFAULT_DATASET_NAME = "master"   # = Kategorie-Name
USE_AMP = True
USE_CHECKPOINTING = True
MAX_STEPS = 150000
```

### 5.2 Generator-Konfiguration (machine-lokal, nicht committed)

Zwei-Datei-System:
```
dataset_generator_v2/templates.json         → Format- & Degradation-Templates
dataset_generator_v2/generator_config.json  → Laufzeitkonfiguration (in .gitignore)
```

**Validierung beim Start** (`config_io.py` Z. 262–427): Alle Template-Referenzen und Größen-Konsistenz werden beim Generator-Start geprüft. Fehler → sofortiger Exit.

---

## 6. Konkrete ToDo-Liste

### 🔴 TODO 1 – Trainer: `dataset_architecture.json` lesen (FUNDAMENTAL)

**Datei:** `vsr_plusplus_NEU/train.py` + `vsr_plusplus_NEU/core/dataset.py`

Der Trainer muss beim Start `{DATASET_ROOT}/{DEFAULT_DATASET_NAME}/../dataset_architecture.json` (oder `{DATASET_ROOT}/dataset_architecture.json`) lesen und daraus beziehen:

| Feld | Wird aktuell aus | Soll aus |
|---|---|---|
| `size_keys` (format names) | `config.py` hardcoded | `arch["categories"][cat]["formats"][*]["template"]` |
| `gt_size` pro Format | `dataset.py` hardcoded | `arch["categories"][cat]["formats"][*]["gt_size"]` |
| `lr_size` pro Format | implizit aus GT/3 | `arch["categories"][cat]["formats"][*]["lr_size"]` |
| `scale` | PixelShuffle(3) hardcoded | `arch["categories"][cat]["formats"][*]["scale"]` |
| `n_frames` | hardcoded 7 | `arch["n_frames"]` |
| `output_format` | hardcoded `.png` | `arch["output_format"]` |

**Empfohlener Code** (in `train.py`):

```python
import json, os

def load_dataset_architecture(dataset_root):
    arch_path = os.path.join(dataset_root, "dataset_architecture.json")
    if not os.path.exists(arch_path):
        raise FileNotFoundError(f"dataset_architecture.json not found at {arch_path}")
    with open(arch_path) as f:
        return json.load(f)
```

---

### 🔴 TODO 2 – `core/dataset.py`: Bucket-Subdirectory-Support (PFLICHT)

**Vollständige Anleitung:** `LOADER_UPDATE_REQUIRED.md` (bereits im Repo)

1. Hilfsfunktion `_collect_image_files(base_dir)` hinzufügen (s. GAP 2 oben), erweitert auf `.bmp` + `.png`
2. Alle 5 `os.listdir(self.gt_dir)` durch die neue Funktion ersetzen
3. `lr_paths`-Dict: Key wird zu `"0000/foo.bmp"`, Value zu `.../LR_7frames/0000`
4. `__getitem__` Z. 724: `lr_path = os.path.join(lr_dir, os.path.basename(gt_file))`
5. Cache-Invalidierung: `gt_file_count` zusätzlich zu `mtime` speichern

---

### 🔴 TODO 3 – `ADAPTIVE_BATCH_CONFIG` dynamisch aus `dataset_architecture.json`

**Datei:** `config.active.py` + `train.py`

Option A (einfach): Batch-Config Mapping nach GT-Größe statt nach Format-Name:

```python
def batch_config_for_gt_size(gt_w, gt_h):
    pixels = gt_w * gt_h
    if pixels <= 540 * 540:       # klein
        return {'batch': 8, 'accum': 2}
    elif pixels <= 720 * 720:     # mittel
        return {'batch': 6, 'accum': 1}
    else:                          # groß (1152×648 etc.)
        return {'batch': 4, 'accum': 2}
```

Option B (robust): VRAM-gesteuert wie bereits in `_vram_per_size`-Tracking vorhanden.

---

### 🟡 TODO 4 – `DataStrategyScheduler` dynamisch machen

**Datei:** `core/data_strategy.py`

Hardcodierte Werte anpassen:

```python
# Statt:
CROP_INTRO_END_DISTRIBUTION = {'720_169': 0.40, '540': 0.20, '720': 0.40}
WARMUP_DISTRIBUTION = {'720_169': 1.0, '540': 0.0, '720': 0.0}

# Dynamisch (aus dataset_architecture.json):
# - Größtes Format (größtes gt_size) → Phase-1-Format
# - Rest → Crop-Introduction-Zieldistribution proportional zu weights
```

---

### 🟡 TODO 5 – Modell-Scale aus Konfiguration beziehen

**Datei:** `core/model_7frame.py`

```python
# Statt:
nn.PixelShuffle(3)   # hardcoded

# Dynamisch:
class VSRBidirectional_7frames_3x(nn.Module):
    def __init__(self, n_feats=72, n_blocks=26, scale=3, use_checkpointing=False):
        self.scale = scale
        # ...
        nn.PixelShuffle(scale)
```

Scale aus `dataset_architecture.json` lesen und beim Modell-Init übergeben.

---

### 🟡 TODO 6 – `train.py` Z. 413: Dateiendung dynamisch

```python
# Statt:
files = [f for f in os.listdir(train_dir) if f.lower().endswith('.png')]

# Dynamisch:
OUTPUT_EXTS = ('.bmp', '.png')  # aus dataset_architecture.json["output_format"]
files = [f for f in os.listdir(train_dir) if f.lower().endswith(OUTPUT_EXTS)]
```

---

### 🟢 TODO 7 – `DATASET_STRUCTURE.md` aktualisieren

Nach der Loader-Migration die Beispielpfade auf Bucket-Layout und dynamische Format-Namen aktualisieren (wie in `LOADER_UPDATE_REQUIRED.md` Z. 166 vermerkt).

---

## 7. Fallstricke

| Fallstrick | Details | Erkennung |
|---|---|---|
| **Format-Namen-Mismatch** | `ADAPTIVE_BATCH_CONFIG['720']` existiert, aber Daten liegen in `patches/960_43/` → Dataset-Init gibt 0 Dateien zurück | `No image files found` + leere Trainings-Queue |
| **GT/LR-Pairing nach Bucket-Migration** | `GT/0001/foo.bmp` muss mit `LR_7frames/0001/foo.bmp` gepaart werden – nie mit `LR_7frames/0000/foo.bmp` | Falsche Shape-Fehler, schwarze Outputs |
| **Sortier-Reihenfolge im Bucket-Layout** | Dateien müssen bucket-weise sortiert werden (`"0000/foo.bmp" < "0001/bar.bmp"`) – `sorted()` auf dem vollen Relativpfad ist korrekt | Zufällige Validierungsfehler |
| **Cache-Stale nach Dateiformat-Wechsel** | Index-Cache enthält `.png`-Pfade → nach BMP-Wechsel alle Loads fehlgeschlagen | Alle Samples failed → Training stürzt ab |
| **mtime-basierte Cache-Invalidierung** | `getmtime(gt_dir)` ändert sich NICHT wenn Dateien in einen Bucket-Subdir geschrieben werden | Neue Generator-Patches werden nicht trainiert; scheinbar stabiles Dataset trotz Wachstum |
| **Falscher Frame-Window-Count** | `n_frames=5` schreibt `LR_5frames/` statt `LR_7frames/`. Modell erwartet T=7 fest | `ValueError: LR height N not divisible by 7` |
| **Scale-Faktor-Mismatch** | Neues Template mit Scale=4, aber PixelShuffle(3) im Modell → Output-Shape falsch | SSIM/PSNR nahe 0, Loss divergiert |
| **`DataStrategyScheduler` Warmup-Format falsch** | Phase 1 zwingt 100% `720_169` → aber Format-Name auf Disk ist `1152_169` → DataLoader bekommt 0 Samples in Phase 1 | Training hängt sofort |
| **`dataset_architecture.json` fehlt** | Wenn Generator-Run abgebrochen wurde, existiert die Datei nicht → Trainer-Start schlägt fehl | `FileNotFoundError` beim Lesen |

---

## 8. Empfohlene Implementierungsreihenfolge

```
1. core/dataset.py:   _collect_image_files() + 5× os.listdir ersetzen  (GAP 2)
   → Training kann überhaupt starten

2. core/dataset.py:   SUPPORTED_EXTS = ('.png', '.bmp') dynamisch      (GAP 3)
   → BMP-Datasets werden geladen

3. train.py:          dataset_architecture.json lesen                    (TODO 1)
   → size_keys, gt_sizes, n_frames, output_format dynamisch

4. config.active.py + train.py:  ADAPTIVE_BATCH_CONFIG via GT-Größe     (TODO 3)
   → Batch-Config für beliebige Templates

5. core/data_strategy.py:  Warmup-/Intro-Distribution dynamisch          (TODO 4)
   → DataStrategyScheduler unabhängig von Format-Namen

6. core/model_7frame.py:  scale als Parameter                            (TODO 5)
   → Modell generisch für andere Scale-Faktoren
```

---

## Referenz-Dateien

| Datei | Rolle |
|---|---|
| `dataset_generator_v2/templates.json` | Format- und Degradation-Templates (committed) |
| `dataset_generator_v2/generator_config.json` | Machine-lokale Laufzeitkonfiguration (in `.gitignore`) |
| `dataset_generator_v2/utils/config_io.py` | Config-Loader, Validator, Format-Größen-Berechnung |
| `dataset_generator_v2/utils/format_definitions.py` | `BUCKET_SIZE`, `get_synced_bucket_dirs()`, `get_output_dirs_for_format()` |
| `dataset_generator_v2/streaming_extractor.py` | `create_patch_pair()`, `save_patch_pair()`, LR-Stack-Logik |
| `dataset_generator_v2/make_dataset_v2_uhd.py` | `_build_format_config()` Z. 1080, `_write_architecture_file()` Z. 1134 |
| `vsr_plusplus_NEU/core/dataset.py` | **Loader – alle GAPs hier** |
| `vsr_plusplus_NEU/core/dataloader.py` | MultiSizeDataLoader, SizeGroupedSampler |
| `vsr_plusplus_NEU/core/data_strategy.py` | `DataStrategyScheduler` – hardcodierte Format-Namen |
| `vsr_plusplus_NEU/core/model_7frame.py` | `VSRBidirectional_7frames_3x` – Scale hardcoded |
| `vsr_plusplus_NEU/config.active.py` | Trainings-Config mit hardcodierten size_keys |
| `vsr_plusplus_NEU/training/trainer.py` | Training-Loop – hardcodierte size_key-Strings |
| `vsr_plusplus_NEU/LOADER_UPDATE_REQUIRED.md` | Vollständige Bucket-Migration-Anleitung |
| `{DATASET_ROOT}/dataset_architecture.json` | Runtime – vom Generator geschrieben, soll Trainer steuern |
