# Analyse: `dataset_generator_v2` – Konfigurationsdynamik und Altlasten

> Erstellt: 2026-04-25  
> Basis: Branch `main`, Commit `bc3e244`  
> Untersuchte Hauptdateien: `video_manager.py`, `make_dataset_v2_uhd.py`, `streaming_extractor.py`, `utils/format_definitions.py`, `utils/config_normalizer.py`, `utils/dataset_display.py`, `utils/terminal_ui.py`, `utils/ui_terminal.py`, `utils/ui_display.py`, `category_utils.py`, `create_default_config.py`, `create_full_config.py`, `scan_videos.py`

---

## 1. End-to-End Konfigurationsfluss

### Konfigurationspfad

```
video_manager.py
    ↓  schreibt
generator_config_v2.json
    (root_path, source_dirs, videos[], category_patches, output_patches, processing, quality)
    ↓  liest
make_dataset_v2_uhd.py  (DatasetGeneratorV2UHD.__init__)
    ↓  normalisiert via
utils/config_normalizer.py
    →  erzeugt: base_settings, category_targets, format_config
    ↓
streaming_extractor.py
    (extract_and_save_streaming_distributed, create_patch_pair)
    ↓  nutzt für Ausgabepfade
utils/format_definitions.py
    (get_output_dirs_for_format)
```

### Was korrekt konfigurationsgesteuert ist

| Feature | Config-Schlüssel | Normalisierung | Genutzt in |
|---|---|---|---|
| Kategorien + Patch-Targets | `category_patches` | `normalize_config()` → `category_targets` | `calculate_proportional_distribution()` |
| Format-Gewichtungen | `output_patches[n].weight` | `config_normalizer.py` Z.60–70 → `format_config[cat][fmt].probability` | `_extract_format_probabilities()` |
| Framezahl | `processing.n_frames` | `lr_versions` | `process_video()` Z.1063 |
| Blur-Threshold | `quality.blur_threshold` | `base_settings.min_detail_threshold` | `is_interesting_patch()` Z.1822 |
| LR-Degradation | `quality.lr_degrade_prob` etc. | unverändert | `degrade_cfg` in `extract_and_save_streaming_distributed()` Z.1273 |
| Ausgabeverzeichnis | `root_path` | `base_settings.output_base_dir` | alle Schreibpfade |

### Was NICHT vollständig durch die Config gesteuert wird

Der Fluss enthält **drei kritische Bruchstellen**, die neue Formate (z.B. `1152_169`) entweder mit einem `KeyError` oder mit semantisch falschem Verhalten verarbeiten würden. Details in Abschnitt 2.

---

## 2. Harte Verdrahtungen – Details

### 🔴 KRITISCH 1: `create_patch_pair()` – hartkodierte Format-Whitelist

**Datei:** `dataset_generator_v2/streaming_extractor.py`, Zeile **980**

```python
if format_name in ("medium_169", "720_169"):
    # Full-frame resize  → korrekt für Landscape-Formate
else:
    # Square crop        → FALSCH für 1152_169 !
```

**Problem:** Ein neues Format `1152_169` ist kein Mitglied dieser Whitelist und fällt in den `else`-Zweig (Square Crop). Das Ergebnis wäre ein falsch geschnittenes Patch-Paar: statt eines Full-Frame-Resize ein zufälliger Quadrat-Crop.

Die Information, ob ein Format Landscape oder Square ist, steckt bereits in `format_cfg["gt_size"]`. Die korrekte Prüfung wäre rein dimensionsbasiert:

```python
gt_w, gt_h = format_cfg["gt_size"]
is_landscape_format = gt_w > gt_h   # dynamisch, kein Namensvergleich nötig
```

---

### 🔴 KRITISCH 2: `_consume_raw_frame()` – gleiche hartkodierte Whitelist

**Datei:** `dataset_generator_v2/streaming_extractor.py`, Zeile **1709**

```python
is_resize_fmt = fmt_name in ("medium_169", "720_169")
max_attempts = 1 if is_resize_fmt else 6
```

Gleiche Ursache, gleiche Konsequenz: `1152_169` würde 6 Crop-Versuche erhalten statt 1 Resize-Versuch.

**Fix:** `fmt_name in (...)` → Aspect-Ratio-Prüfung via `format_config.get(category, {}).get(fmt_name, {}).get("gt_size")`.

---

### 🔴 KRITISCH 3: `get_output_dirs_for_format()` – KeyError für unbekannte Formate

**Datei:** `dataset_generator_v2/utils/format_definitions.py`, Zeile **119**

```python
format_spec = FORMATS[format_name]          # ← KeyError wenn 1152_169 nicht in FORMATS!
base_format_dir = format_spec['output_dir']
```

`FORMATS` ist eine statische Dict-Konstante mit fest eingetragenen Einträgen (`540`, `720`, `720_169` + drei Legacy-Aliasse). Jeder in der Config definierte Formatschlüssel, der dort nicht auftaucht, führt beim ersten Schreibversuch zu einem `KeyError`.

**Beobachtung:** In allen bekannten Einträgen gilt `output_dir == key` (z.B. `'720_169': {'output_dir': '720_169', ...}`). Der Fix ist trivial:

```python
format_spec = FORMATS.get(format_name, {'output_dir': format_name, 'aspect_ratio': '1:1'})
base_format_dir = format_spec['output_dir']
```

---

### 🟡 MITTEL: `CATEGORY_PATHS`-Fallback mit altem V1-Pfadschema

**Datei:** `dataset_generator_v2/utils/format_definitions.py`, Zeile **118**

```python
category_path = CATEGORY_PATHS.get(
    category,
    f'{category.capitalize()}/{category.capitalize()}Model/Learn'  # ← alter V1-Pfad!
)
```

Für eine neue Kategorie (z.B. `wildlife`) erzeugt der Fallback `Wildlife/WildlifeModel/Learn` – ein V1-Pfadmuster, das nicht zur flachen V2-Verzeichnisstruktur passt. Das V2-Äquivalent wäre schlicht `wildlife`.

---

### 🟡 MITTEL: `n_frames`-Validierung hartkodiert auf 5 und 7

**Datei:** `dataset_generator_v2/streaming_extractor.py`, Zeile **970**

```python
if n not in (5, 7):
    return None, None
```

Kein direktes Problem für neue Formate, aber wer z.B. 9-Frame-Sequenzen möchte, würde hier stillschweigend `(None, None)` erhalten.

---

### 🟡 MITTEL: LR-Verzeichnisname hartkodiert auf 5/7 Frames

**Datei:** `dataset_generator_v2/utils/format_definitions.py`, Zeile **123**

```python
lr_dir_name = 'LR' if lr_frames == 5 else 'LR_7frames'
```

Bei einer anderen Framezahl (z.B. 9) wäre der Verzeichnisname `LR_7frames` schlicht falsch benannt. Sollte auf `f'LR_{lr_frames}frames'` vereinheitlicht werden (mit `5` als Sonderfall `'LR'` für VSR++ Kompatibilität).

---

### ⚪ INFO: Stream-Dimensionskonstanten

**Datei:** `dataset_generator_v2/streaming_extractor.py`, Zeilen **79–86**

```python
STREAM_WIDTH:  int = 1920
STREAM_HEIGHT: int = 1080
STREAM_4K_WIDTH:  int = 3840
STREAM_4K_HEIGHT: int = 2160
```

Diese Konstanten werden in den Filterstring-Templates (`_TONEMAP_FILTER` etc.) auf Zeilen 109–174 direkt eingebettet, aber `build_vf_filter()` empfängt `width`/`height` als Parameter, und der Generator übergibt `STREAM_4K_WIDTH/HEIGHT` explizit (Z.1275–1276). Kein echtes Problem, solange keine 6K-Quellen genutzt werden.

---

## 3. Altlasten und toter Code

### ☠️ Dead Code: `_extract_patches_multi_format_legacy()` mit unreachable Code

**Datei:** `make_dataset_v2_uhd.py`, Zeilen **1305–1631**

Methode ist explizit als `LEGACY` markiert und wird nirgendwo aufgerufen. Besonders auffällig: **Nach dem `return patches_created` auf Zeile 1478 folgen 153 Zeilen unreachable Code** (Zeilen 1479–1631), die eine komplette zweite Implementierung derselben Logik darstellen. Eindeutiges Indiz für hastige Umbauten ohne Aufräumen.

---

### ☠️ Dead Code: `_extract_patches_multi_category()`

**Datei:** `make_dataset_v2_uhd.py`, Zeilen **1084–1142**

Kommentar im Code: *"NOTE: This method is legacy and replaced by `_extract_patches_multi_format`."* Wird nicht aufgerufen. Enthält hardkodierten `stride_seconds = 3.0`.

---

### ☠️ Dead Code: Auskommentierter veralteter Method-Stub

**Datei:** `make_dataset_v2_uhd.py`, Zeilen **1749–1756**

```python
# OLD METHOD - DEPRECATED
# Replaced by _extract_patches_multi_category ...
# def _extract_patches_from_video(self, ...):
#     pass
```

Komplett auskommentierter alter Stub ohne jeden Nutzen.

---

### ☠️ Ungenutzte Imports

**Datei:** `make_dataset_v2_uhd.py`, Zeilen **33–36**

```python
from utils.format_definitions import (
    FORMATS,                       # nie direkt genutzt
    CATEGORY_FORMAT_DISTRIBUTION,  # nie direkt genutzt
    CATEGORY_PATHS,                # nie direkt genutzt
    select_random_format,          # nie direkt genutzt
    get_output_dirs_for_format     # ← einziger echter Nutzer
)
```

Nur `get_output_dirs_for_format` wird tatsächlich verwendet (in `_save_patch_pair()` Z.1844).

---

### ☠️ Legacy-Format-Aliasse in `format_definitions.py`

**Datei:** `dataset_generator_v2/utils/format_definitions.py`, Zeilen **27–48**

```python
# Legacy format names for backward compatibility
'small_540': {'gt_size': (540, 540),  'output_dir': '540',     ...}
'medium_169': {'gt_size': (720, 405), 'output_dir': '720_169', ...}
'large_720':  {'gt_size': (720, 720), 'output_dir': '720',     ...}
```

Parallel zu den neuen Schlüsseln `540`, `720`, `720_169`. Die Legacy-Namen erscheinen noch in den Whitelists in `create_patch_pair()` (Z.980) und `_consume_raw_frame()` (Z.1709). `create_default_config.py` und `video_manager.py` schreiben ausschließlich die neuen Schlüssel.

---

### ☠️ `CATEGORY_FORMAT_DISTRIBUTION` – hartkodiert und obsolet

**Datei:** `dataset_generator_v2/utils/format_definitions.py`, Zeilen **52–72**

```python
CATEGORY_FORMAT_DISTRIBUTION = {
    'master':    {'small_540': 0.50, 'medium_169': 0.35, 'large_720': 0.15},
    'universal': {'small_540': 0.50, 'medium_169': 0.35, 'large_720': 0.15},
    'space':     {'small_540': 0.40, 'medium_169': 0.35, 'large_720': 0.25},
    'toon':      {'small_540': 0.65, 'medium_169': 0.35}
}
```

Diese Verteilung wird weder von `make_dataset_v2_uhd.py` noch von `streaming_extractor.py` direkt genutzt. Der `config_normalizer.py` baut `format_config` aus `output_patches` der JSON-Config. `CATEGORY_FORMAT_DISTRIBUTION` ist ein vergessenes Überbleibsel aus einer früheren Architektur mit 4 hartkodierter Kategorien.

---

### ☠️ `get_format_for_category()` und `select_random_format()` – nie aufgerufen

**Datei:** `dataset_generator_v2/utils/format_definitions.py`, Zeilen **82–95**

```python
def get_format_for_category(category):
    """Get list of available formats for a category."""
    return list(CATEGORY_FORMAT_DISTRIBUTION.get(category, {}).keys())

def select_random_format(category):
    """Select a random format based on category distribution."""
    ...
    return random.choices(formats, weights=weights, k=1)[0]
```

Beide Funktionen basieren auf der obsoleten `CATEGORY_FORMAT_DISTRIBUTION` und werden im gesamten V2-Codebase nicht aufgerufen.

---

### ☠️ Doppelte UI-Module

| Aktiv (V2) | Alt (Altlast) |
|---|---|
| `utils/terminal_ui.py` | `utils/ui_terminal.py` |
| `utils/dataset_display.py` | `utils/ui_display.py` |

`dataset_display.py` importiert `from .terminal_ui import *` und hat die V2-API: `draw_dataset_ui(state_dict)`.  
`ui_display.py` importiert `from .ui_terminal import *` und hat eine alte API: `draw_dataset_generator_ui(generator)` – nimmt das Generator-Objekt direkt.

`make_dataset_v2_uhd.py` nutzt ausschließlich `utils/dataset_display.py`. `ui_display.py` wird zwar im `__init__.py` nicht explizit exportiert, aber `ui_terminal.py` hat dasselbe Naming-Konfliktpotenzial.

---

### ☠️ `create_full_config.py` – Migrationsscript für altes Format

**Datei:** `dataset_generator_v2/create_full_config.py`

```python
config_file = sys.argv[1] if len(sys.argv) > 1 else 'generator_config.json'
```

Referenziert `generator_config.json` (altes Format ohne `_v2`-Suffix). Fügt nur `priority`-Felder hinzu. Gehört zu einer vergangenen Migration und ist funktional obsolet.

---

### 🟡 `scan_videos.py` – möglicherweise durch `rescan_file_list()` ersetzt

**Datei:** `dataset_generator_v2/scan_videos.py`

Eigenständiger Scanner, der heute vermutlich vollständig durch Option 16 (`rescan_file_list()`) in `video_manager.py` ersetzt wurde. Sollte geprüft und ggf. entfernt werden.

---

## 4. Soll-Zustand für vollständig dynamischen Generator

Die Config-Steuerung ist im Kern schon vorhanden – `config_normalizer.py` tut seinen Job korrekt. Die verbleibenden Bruchstellen sind ausnahmslos **in der Schnittstelle zwischen dem Format-Key im JSON und den hardkodierten Fallunterscheidungen im Extractor/Util**.

### Ziel-Architektur (am Beispiel eines neuen Formats `1152_169`)

```
JSON-Config (output_patches)
  └─ "1152_169": {
       "enabled": true,
       "gt_size": [1152, 648],
       "scale": 3,
       "weight": 20
     }
         ↓
config_normalizer.py  (keine Änderung nötig)
  └─ format_config["master"]["1152_169"] = {
       "gt_size": [1152, 648],
       "lr_size": [384, 216],       ← dynamisch berechnet: gt_size // scale
       "probability": 0.2           ← aus weight / total_weight
     }
         ↓
streaming_extractor.create_patch_pair()   (Fix 1)
  └─ gt_w, gt_h = format_cfg["gt_size"]  → [1152, 648]
     is_landscape = gt_w > gt_h          → True → Full-Frame-Resize-Pfad
     (kein Namensvergleich nötig)
         ↓
streaming_extractor._consume_raw_frame()  (Fix 2)
  └─ is_resize_fmt = gt_w > gt_h         → True → max_attempts = 1
         ↓
format_definitions.get_output_dirs_for_format()  (Fix 3)
  └─ FORMATS.get("1152_169", {"output_dir": "1152_169"})
     → base_format_dir = "1152_169"
     → gt_dir  = "{root}/master/patches/1152_169/GT"
     → lr_dir  = "{root}/master/patches/1152_169/LR_7frames"
     (kein Eintrag in FORMATS nötig)
```

Nach diesen drei Fixes ist `video_manager.py` der **einzige Konfigurationspfad** und `make_dataset_v2_uhd.py` verarbeitet alle Formate ohne Sonderfälle.

---

## 5. Priorisierte Änderungsliste

### MUSS geändert werden (blockiert neue Formate vollständig)

| # | Datei | Zeile | Aktueller Code | Fix |
|---|---|---|---|---|
| 1 | `utils/format_definitions.py` | 119 | `FORMATS[format_name]` | `FORMATS.get(format_name, {'output_dir': format_name, 'aspect_ratio': '1:1'})` |
| 2 | `utils/format_definitions.py` | 118 | Fallback `f'{cat.capitalize()}/{...}Model/Learn'` | Fallback → `category` (flaches V2-Schema) |
| 3 | `streaming_extractor.py` | 980 | `format_name in ("medium_169", "720_169")` | `format_cfg["gt_size"][0] > format_cfg["gt_size"][1]` |
| 4 | `streaming_extractor.py` | 1709 | `fmt_name in ("medium_169", "720_169")` | Aspect-Ratio-Prüfung via `format_config.get(cat,{}).get(fmt,{}).get("gt_size")` |

---

### KANN WEG (Altlasten – keine Funktionalität verloren)

| # | Datei | Zeilen | Was |
|---|---|---|---|
| 5 | `make_dataset_v2_uhd.py` | 1084–1142 | `_extract_patches_multi_category()` – als Legacy markiert, nicht aufgerufen |
| 6 | `make_dataset_v2_uhd.py` | 1305–1631 | `_extract_patches_multi_format_legacy()` – als LEGACY markiert; 153 Zeilen unreachable Code nach `return` Z.1478 |
| 7 | `make_dataset_v2_uhd.py` | 1749–1756 | Auskommentierter veralteter Method-Stub |
| 8 | `make_dataset_v2_uhd.py` | 33–36 | Imports `FORMATS`, `CATEGORY_FORMAT_DISTRIBUTION`, `CATEGORY_PATHS`, `select_random_format` – alle ungenutzt |
| 9 | `utils/format_definitions.py` | 27–48 | Legacy-Format-Aliasse `small_540`, `medium_169`, `large_720` (nach Fix 3 & 4 nicht mehr referenziert) |
| 10 | `utils/format_definitions.py` | 52–72 | `CATEGORY_FORMAT_DISTRIBUTION` – hartkodiert, nie mehr genutzt |
| 11 | `utils/format_definitions.py` | 82–95 | `get_format_for_category()`, `select_random_format()` – nie mehr aufgerufen |
| 12 | `utils/ui_display.py` | komplett | Altes UI-Modul mit alter `draw_dataset_generator_ui(generator)`-API – nicht von V2-Code genutzt |
| 13 | `create_full_config.py` | komplett | Migrationsscript für altes `generator_config.json`-Format |

---

### OPTIONAL aufräumen (Codequalität / zukünftige Erweiterbarkeit)

| # | Datei | Änderung |
|---|---|---|
| 14 | `utils/ui_terminal.py` | Nach Entfernung von `ui_display.py` prüfen ob noch referenziert; ggf. mit `terminal_ui.py` zusammenführen |
| 15 | `utils/__init__.py` | Exporte auf tatsächlich Genutztes reduzieren: `get_output_dirs_for_format`, `ProgressTracker`, `draw_dataset_ui` |
| 16 | `scan_videos.py` | Prüfen ob vollständig durch `video_manager.py` Option 16 ersetzt; ggf. entfernen |
| 17 | `make_dataset_v2_uhd.py` Z.825–886 | `extract_frames_batch_uhd()` prüfen: delegiert nur an `extract_frames_uhd()` im Single-Mode, wird durch streaming_extractor ersetzt – vereinfachen oder entfernen |
| 18 | `utils/format_definitions.py` | `CATEGORY_PATHS` durch dynamischen Fallback ersetzen, sobald alle hartkodiert-4-Kategorien-Stellen behoben sind |
| 19 | `streaming_extractor.py` Z.970 | `if n not in (5, 7)` → flexibilisieren oder zumindest dokumentieren |
| 20 | `utils/format_definitions.py` Z.123 | `lr_dir_name = 'LR' if lr_frames == 5 else 'LR_7frames'` → `f'LR_{lr_frames}frames'` mit Sonderfall 5→`'LR'` |

---

## 6. Zusammenfassung

Der `config_normalizer.py` leistet gute Arbeit beim Überführen der V2-Config in das interne Format. Die Hauptprobleme sind **drei punktuelle Hardcodierungen in der Ausgabeschicht**:

1. `create_patch_pair()` – Namensvergleich statt Aspect-Ratio-Prüfung
2. `_consume_raw_frame()` – gleicher Namensvergleich für `max_attempts`
3. `get_output_dirs_for_format()` – statischer `FORMATS`-Dict statt dynamischem Fallback

Ein neues Format wie `1152_169` würde an allen drei Stellen stumm falsch verarbeitet: entweder mit einem `KeyError` (Punkt 3) oder mit einem semantisch falschen Crop-Typ statt Full-Frame-Resize (Punkte 1+2).

Daneben enthält `make_dataset_v2_uhd.py` über **300 Zeilen toten Legacy-Code** und die `format_definitions.py` pflegt eine hartkodierte `CATEGORY_FORMAT_DISTRIBUTION` mit alten Format-Aliasen, die im V2-Betrieb längst nicht mehr genutzt wird.

**Nach Umsetzung der vier Pflicht-Fixes (Abschnitt 5, Nr. 1–4) ist `video_manager.py` der einzige Konfigurationspfad**, und der Generator verarbeitet beliebige neue Formate und Kategorien ohne Code-Änderung.
