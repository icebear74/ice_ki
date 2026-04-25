# VSR++ Training Configuration Benchmark — Nutzungsanleitung

## Zweck

Das Skript `benchmark_training_configs.py` beantwortet eine zentrale Frage, bevor du teure Trainingsläufe startest:

> **Welche Kombination aus Auflösung, Framezahl, Modellgröße und Präzision passt auf meine GPU — und wie lange würde das Training dauern?**

Es erzeugt synthetische Batch-Daten, die der echten Trainingspipeline ähneln, und misst VRAM-Nutzung, Iterationszeit und OOM-Status für jede Konfiguration.

---

## Schnellstart

```bash
# Im vsr_plusplus_NEU-Verzeichnis ausführen:
cd vsr_plusplus_NEU

# Fokussierter Scan (empfohlen als erster Test, ~10-20 Minuten auf P100)
python benchmark_training_configs.py --quick

# Vollständiger Sweep (alle Kombinationen, ~60-120 Minuten)
python benchmark_training_configs.py --full

# Nur bestimmte Auflösungen testen
python benchmark_training_configs.py --gt-sizes 960x540 1920x1080

# Trockenlauf: Zeigt geplante Konfigurationen ohne GPU-Ausführung
python benchmark_training_configs.py --dry-run
```

---

## Konfigurationsparameter

### GT-Größen (Auflösungen)

| Key | GT (H×W) | LR (H×W) | Typ | Status |
|-----|----------|----------|-----|--------|
| `720x405` | 720×405 | 240×135 | 16:9 | Produktion |
| `540x540` | 540×540 | 180×180 | Crop | Produktion |
| `720x720` | 720×720 | 240×240 | Crop | Produktion |
| `960x540` | 960×540 | 320×180 | 16:9 | **Neu** |
| `1920x1080` | 1920×1080 | 640×360 | 16:9 FullHD | **Neu** |

Alle 16:9-Größen sind sauber durch 3 teilbar — kein Padding nötig.

### Framezahlen

| Frames | Modellklasse | Status |
|--------|-------------|--------|
| 7 | `VSRBidirectional_7frames_3x` (Produktionsmodell) | Produktion |
| 9 | `VSRBidirectional_Nframes_3x(n_frames=9)` | Neu (Benchmark-only) |
| 11 | `VSRBidirectional_Nframes_3x(n_frames=11)` | Neu (Benchmark-only) |

**Hinweis zu 9/11 Frames:**  
Das Produktionsmodell ist fest auf 7 Frames verdrahtet. Das Benchmark-Skript enthält `VSRBidirectional_Nframes_3x`, eine verallgemeinerte Version mit identischer Architektur (gleiche Komponenten: TemporalAlignBlock, GatedFusionBlock, ResidualBlock mit AttentionGate). Nur die Propagationsschleife ist parametrisiert:
- Center-Frame: `N // 2` (bei N=9: Index 4, bei N=11: Index 5)
- Backward: Frames nach dem Center
- Forward: Frames vor dem Center

### Modellkapazität

| Parameter | Quick-Mode | Full-Mode |
|-----------|-----------|-----------|
| `n_feats` | 60, 72 | 48, 60, 72, 80 |
| `n_blocks` | 24, 26 | 20, 24, 26 |

---

## Synthetische Daten — Was wird simuliert?

Das Skript erzeugt keine zufälligen Tensoren (`torch.randn`), sondern realistische Trainingsdaten:

1. **GT**: Glatt interpoliertes zufälliges Bild (niedrige Frequenzen dominieren → ähnelt echten Videoframes mehr als weißes Rauschen)
2. **LR je Frame**:
   - Kleiner Pixelversatz (±2px) zwischen Frames → simuliert Kamerabewegung
   - 3× Bicubic-Downscale des verschobenen GT-Frames
   - Leichtes Gaußsches Rauschen (σ ≈ 1.5/255)

Dies stellt sicher:
- LR ist mit GT korreliert (das Modell muss wirklich "upscalen")
- Benachbarte Frames zeigen echte temporale Variation (nicht identische Kopien)
- Die VRAM-Messung entspricht dem realen Training-Footprint

---

## Ausgabe interpretieren

### Terminal-Ausgabe (Live)

```
  [ 1/ 24]  7f | B1×A8 | 26b | 72feat | 405×720 | FLOAT16
            ✅  4.82 GB (30.1%)  |  2.341 s/iter
  [ 2/ 24]  9f | B1×A8 | 26b | 72feat | 405×720 | FLOAT16
            ✅  5.64 GB (35.3%)  |  2.897 s/iter
  [ 3/ 24]  9f | B2×A4 | 26b | 72feat | 540×960 | FLOAT16
            ❌  OOM
```

### Zusammenfassungstabelle

```
  #    Config                                              VRAM GB   % P100   s/iter   Fit?
  ─────────────────────────────────────────────────────────────────────────────────────────
   1   7f | B1×A8 | 24b | 60feat | 405×720 | FLOAT16       4.31     26.9%    1.987   ✅
   2   7f | B1×A8 | 26b | 72feat | 405×720 | FLOAT16       4.82     30.1%    2.341   ✅
  ...
```

| Symbol | Bedeutung |
|--------|----------|
| ✅ | VRAM ≤ 90% → stabil für echtes Training |
| ⚠️ | VRAM 90–100% → Risiko bei echtem Training |
| ❌ | VRAM > 100% (OOM) |

### CSV / JSON

```
benchmark_results.csv  — Tabellenformat für Excel/Pandas-Analyse
benchmark_results.json — Maschinenlesbares Format mit allen Metadaten
```

---

## Beispiel-Entscheidungsmatrix für Tesla P100 (16 GB)

Nach einem `--quick`-Lauf könnte die Analyse ergeben:

| Konfiguration | VRAM | s/iter | 150k Steps | Empfehlung |
|--------------|------|--------|-----------|------------|
| 7f, 720×405, n_feats=72, FP16, B2×A4 | ~5.2 GB | ~3.1 s | ~129 h | ✅ Optimal |
| 9f, 720×405, n_feats=72, FP16, B1×A8 | ~6.8 GB | ~4.2 s | ~175 h | ✅ Gut |
| 7f, 960×540, n_feats=72, FP16, B1×A8 | ~8.1 GB | ~5.5 s | ~229 h | ⚠️ Teuer |
| 11f, 960×540, n_feats=72, FP16, B1×A8 | ~11.4 GB | ~7.8 s | ~325 h | ⚠️ Grenzwertig |
| 7f, 1920×1080, n_feats=72, FP16, B1×A4 | ~14.2 GB | ~12.1 s | ~504 h | ❌ Impraktisch |

*Hinweis: Tatsächliche Werte hängen von der jeweiligen GPU und PyTorch-Version ab.*

---

## Beantwortete Fragen

### Frage 1: Wie viel kosten 9 vs. 11 Frames?

```bash
python benchmark_training_configs.py --frames 7 9 11 --gt-sizes 720x405 --precisions float16
```

Erwarteter Overhead pro Frame-Stufe:
- **7 → 9 Frames**: ca. +15-25% VRAM, +20-30% Zeitaufwand pro Iteration
- **9 → 11 Frames**: ca. +10-20% VRAM, +15-20% Zeitaufwand pro Iteration

(Das Verhältnis verbessert sich mit steigender Framezahl, da der Upsampling-Block gleich bleibt.)

### Frage 2: Lohnt sich 960×540 statt 720×405?

```bash
python benchmark_training_configs.py --gt-sizes 720x405 960x540 --frames 7 --precisions float16
```

960×540 hat 1,78× mehr Pixel als 720×405 → erwarte ca. 1.4-1.6× mehr VRAM.

### Frage 3: n_feats=60 vs. 72 — spürbarer Unterschied?

```bash
python benchmark_training_configs.py --n-feats 60 72 --gt-sizes 720x405 --frames 7
```

n_feats=72 vs. 60: ~44% mehr Parameter → ca. +20-35% VRAM, aber bessere Modellkapazität.

### Frage 4: Lohnt sich FP32 für die Qualität?

```bash
python benchmark_training_configs.py --precisions float16 float32 --gt-sizes 720x405
```

FP32 benötigt 2× VRAM für Aktivierungen, ~1.3-1.5× für Gewichte. Für VSR-Training ist FP16 mit AMP i.d.R. ausreichend.

---

## Wichtige Einschränkungen

1. **Synthetische Daten**: Der reale Datenloader (mit Prefetching, mehreren Workers, TensorBoard) verbraucht zusätzlich ~100-300 MB VRAM. Ein 90%-VRAM-Ergebnis im Benchmark ist im echten Training grenzwertig.

2. **9/11-Frame-Produktion**: Das Benchmark-Skript kann 9/11-Frame-Konfigurationen testen, aber das Trainings-Skript (`train.py`) und der Dataset-Loader (`core/dataset.py`) unterstützen aktuell nur 7 Frames. Um 9/11 Frames in echter Produktion zu nutzen, wären Anpassungen an:
   - `core/dataset.py`: Frame-Stack-Größe parametrisieren
   - `train.py`: Modellinstanz mit `n_frames` Parameter
   - `config.active.py`: `N_FRAMES`-Konfigurationsschlüssel
   erforderlich.

3. **720×720 bei BS=2**: Bekannte OOM-Konfiguration für größere Modelle auf P100, wird automatisch übersprungen.

4. **1920×1080**: Sehr hoher VRAM-Bedarf (~12-15 GB). Nur mit kleinsten Modellgrößen (n_feats=48, n_blocks=20) und FP16 möglicherweise machbar.

---

## Alle CLI-Optionen

```
usage: benchmark_training_configs.py [-h] [--quick | --full]
                                      [--gt-sizes HxW [HxW ...]]
                                      [--frames N [N ...]]
                                      [--n-feats F [F ...]]
                                      [--n-blocks B [B ...]]
                                      [--batch-sizes BS [BS ...]]
                                      [--precisions {float16,float32} ...]
                                      [--output-dir DIR]
                                      [--timing-iters N]
                                      [--dry-run]
                                      [--no-csv] [--no-json]

  --quick            Fokussierter Scan: 720x405 + 960x540, 7+9 Frames, FP16
  --full             Exhaustiver Sweep: alle Parameter
  --gt-sizes         Beliebige GT-Größen (z.B. --gt-sizes 960x540 1920x1080)
  --frames           Framezahlen (z.B. --frames 7 9 11)
  --n-feats          Feature-Kanäle (z.B. --n-feats 60 72)
  --n-blocks         Residualblöcke (z.B. --n-blocks 24 26)
  --batch-sizes      Batch-Größen (z.B. --batch-sizes 1 2)
  --precisions       Präzision (float16 und/oder float32)
  --output-dir       Ausgabeverzeichnis für CSV/JSON
  --timing-iters     Anzahl gemessener Iterationen (Standard: 5)
  --dry-run          Konfigurationen auflisten, kein GPU-Test
  --no-csv           CSV-Ausgabe unterdrücken
  --no-json          JSON-Ausgabe unterdrücken
```

---

## Empfohlene Vorgehensweise für P100-Hardware-Entscheidungen

```bash
# Schritt 1: Schneller Überblick
python benchmark_training_configs.py --quick

# Schritt 2: Vielversprechende Konfigurationen vertiefen
python benchmark_training_configs.py \
  --gt-sizes 720x405 960x540 \
  --frames 7 9 \
  --n-feats 60 72 \
  --n-blocks 24 26 \
  --batch-sizes 1 2 \
  --precisions float16

# Schritt 3: Ergebnisse analysieren
python -c "
import csv
rows = list(csv.DictReader(open('benchmark_results.csv')))
ok = [r for r in rows if r['success'] == 'True']
ok.sort(key=lambda r: float(r['vram_gb']))
for r in ok[:10]:
    print(f\"{r['n_frames']}f | {r['gt_key']} | feats={r['n_feats']} | \
{float(r['vram_gb']):.2f}GB | {float(r['time_per_iter']):.2f}s\")
"
```
