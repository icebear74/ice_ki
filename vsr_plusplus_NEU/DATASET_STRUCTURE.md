# Dataset Struktur — Generator V2 und Training System

> **Status**: Vollständig auf Generator V2 mit dynamischen Templates migriert.
> Templates werden dynamisch aus `dataset_architecture.json` geladen — keine festen
> Size-Keys wie `540`, `720`, `720_169` mehr im Code.

---

## Verzeichnisstruktur (Generator V2)

Der `dataset_generator_v2` verwendet `dataset_architecture.json` um Templates pro Kategorie zu
definieren.  Die Templates können sich je nach Konfiguration unterscheiden.  Typische Ausgabe:

```
/mnt/data/training/Dataset/                  ← DATASET_ROOT (config.py)
├── dataset_architecture.json                ← Architekturbeschreibung (templates, n_frames, ...)
│
└── master/                                  ← Kategorie (z.B. master, space, toon)
    ├── patches/
    │   ├── 720_169/                         ← Template-Key (dynamisch, aus dataset_architecture.json)
    │   │   ├── GT/                          ← Ground Truth (z.B. 405×720 BMP/PNG)
    │   │   │   ├── 0000/                    ← Bucket-Verzeichnis (V2 bucket layout)
    │   │   │   └── 0001/
    │   │   └── LR_7frames/                  ← LR gestackt (7 frames vertikal)
    │   │       ├── 0000/
    │   │       └── 0001/
    │   ├── 540/
    │   │   ├── GT/
    │   │   └── LR_7frames/
    │   └── 720/
    │       ├── GT/
    │       └── LR_7frames/
    │
    ├── val/
    │   ├── 720_169/
    │   │   └── GT/                          ← NUR GT hier — kein LR!
    │   ├── 540/
    │   │   └── GT/
    │   └── 720/
    │       └── GT/
    │
    └── training_run_locked.json             ← Gesperrt bei erstem Start (Checkpoint-Kompatibilität)
```

**V2 Bucket Layout**: GT und LR Dateien können in nummerierten Unterverzeichnissen (z.B. `0000/`, `0001/`)
liegen.  Der Loader erkennt beides automatisch (bucket und flat Layout).

---

## dataset_architecture.json

Dieses File beschreibt die verfügbaren Templates, n_frames, Bildformat und Format-Gewichte.
Beispiel:

```json
{
  "n_frames": 7,
  "img_format": "bmp",
  "categories": {
    "master": {
      "templates": {
        "720_169": { "width": 720, "height": 405, "weight": 0.40 },
        "540":     { "width": 540, "height": 540, "weight": 0.20 },
        "720":     { "width": 720, "height": 720, "weight": 0.40 }
      }
    }
  }
}
```

Das Training-System liest dieses File beim Start und:
- entdeckt alle Templates für die gewählte Kategorie dynamisch
- benutzt `n_frames` zur Validierung (muss 7 sein, sonst Abbruch)
- wählt das LR-Verzeichnis automatisch (`LR_7frames` für n_frames=7)
- leitet die Phase-2 Endverteilung aus den Template-Gewichten ab

---

## Modell-Constraints (fixiert)

Das Training-Modell ist auf **7 Frames** und **3× Scale** fixiert.  Beim Start wird
`dataset_architecture.json` dagegen geprüft:

- `n_frames != 7` → Abbruch mit Fehlermeldung
- Scale ist implizit durch LR-Dimensionen (GT / 3 = LR)

---

## Konfiguration (config.py)

```python
# Neue Standard-Defaults (benchmark-validiert)
N_FEATS  = 72   # Feature-Kanäle
N_BLOCKS = 24   # Residual Blocks (26/28 sind teurer ohne ausreichend Gewinn)

# Dataset Root — neuer Standard-Pfad
DATASET_ROOT = "/mnt/data/training/Dataset"

# Kategorie
DEFAULT_DATASET_NAME = "master"  # muss mit dataset_architecture.json übereinstimmen
```

Leichtere Alternative: `N_BLOCKS = 20` (weniger VRAM, etwas geringere Qualität).

**Workflow**:
```bash
cp config.py.example config.py   # Vorlage kopieren
# config.py editieren (Pfad, Kategorie, VRAM-Einstellungen)
python3 train.py
```

---

## Validation Workflow (GT-only)

Validation-Daten werden **nur als GT-Bilder** bereitgestellt.  LR wird nie ins `val/`-Verzeichnis
kopiert — das Training findet LR automatisch über den Basename-Index in `patches/{template}/LR_{n}frames/`.

```bash
# Nur GT ins Validation-Verzeichnis kopieren:
cp repräsentatives_bild.bmp /mnt/data/training/Dataset/master/val/720_169/GT/

# Das Training findet automatisch:
# /mnt/data/training/Dataset/master/patches/720_169/LR_7frames/repräsentatives_bild.bmp
```

**Wichtig**:
- Dateipfade (Basenames) müssen in GT und LR identisch sein
- Kein rekursiver Scan pro Sample — das System baut einen Basename-Index einmalig beim Start
- LR kommt **immer** aus `patches/{template}/LR_{n}frames/`, nie aus `val/`

---

## Locked Run Config

Beim ersten Trainingsstart wird `training_run_locked.json` im Kategorie-Verzeichnis erstellt:

```json
{
  "n_feats": 72,
  "n_blocks": 24,
  "n_frames": 7,
  "scale": 3,
  "dataset_root": "/mnt/data/training/Dataset",
  "category": "master",
  "templates": ["540", "720", "720_169"]
}
```

Bei **Resume** wird diese Datei geladen und gegen die aktuelle Konfiguration geprüft.
Bei Abweichung bricht das Training mit einer klaren Fehlermeldung ab — Checkpoints werden
nie mit einer inkompatiblen Konfiguration fortgesetzt.

Um neu zu starten: `L` beim Startprompt wählen (löscht/sichert Checkpoints und entfernt die Lock-Datei).

---

## DataStrategy Phasen

Die Datenstrategie ist vollständig dynamisch und leitet Warmup-Template und Phase-2 Verteilung
aus `dataset_architecture.json` ab:

| Phase | Schritte | Daten | Perceptual Loss |
|-------|----------|-------|-----------------|
| **Warmup** | 0–3000 | 100 % Warmup-Template (größte GT-Fläche) | 0.0 → 0.03 |
| **Crop Introduction** | 3000–8000 | Linearer Übergang → Arch-Gewichte | 0.03 → 0.08 |
| **Stable** | 8000+ | Natürliche Dateizählung (kein Override) | AdaptiveSystem |

Phase 2 startet erst wenn genug nicht-Warmup-Dateien auf der Disk vorhanden sind
(`MIN_CROP_FILES_TRAINING = 10000`).

---

## Adaptive Batch Config

Für unbekannte V2 Templates wird eine Pixel-Count-Regel angewandt:
- GT-Pixel ≤ 291.600 (405×720): batch=2, accum=4 → eff=8
- GT-Pixel > 291.600:           batch=1, accum=4 → eff=4

Bekannte Werte können in `config.py` unter `ADAPTIVE_BATCH_CONFIG` eingetragen werden
(Einträge überschreiben die automatische Regel).

---

## Fehlersuche

**Problem**: `CHECKPOINT COMPATIBILITY ERROR`
→ Config (`N_FEATS`, `N_BLOCKS`, `N_FRAMES`, Templates) stimmt nicht mit `training_run_locked.json` überein.
→ Entweder Config zurücksetzen oder neu mit `L` starten.

**Problem**: `MODEL ARCHITECTURE MISMATCH`
→ `dataset_architecture.json` sagt `n_frames != 7`.
→ Korrektes Architecture-File verwenden oder Modell verallgemeinern.

**Problem**: Keine GT/LR Matches
→ Sicherstellen dass Basenames in `patches/{template}/GT/` und `patches/{template}/LR_{n}frames/` identisch sind.
→ `DATASET_ROOT` und `DEFAULT_DATASET_NAME` in `config.py` prüfen.

