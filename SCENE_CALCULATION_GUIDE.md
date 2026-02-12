# Scene Calculation Guide - Dataset Generator

## Übersicht / Overview

Dieses Dokument erklärt, wie der Dataset Generator berechnet, wie viele Szenen (Scenes) für einen Film extrahiert werden müssen.

This document explains how the dataset generator calculates how many scenes need to be extracted for a film.

---

## Grundkonzepte / Basic Concepts

### 1. Categories (Kategorien)
- **master**: Meister-Qualität (höchste Qualität)
- **space**: Weltraum/Science-Fiction
- **toon**: Animation/Cartoon
- **universal**: Universal/Allgemein

### 2. Resolutions (Auflösungen)
- **540p** (small): 960x540 Pixel
- **1080p** (large): 1920x1080 Pixel  
- **2160p** (uhd): 3840x2160 Pixel

### 3. Format Distribution (Format-Verteilung)
Für jede Kategorie gibt es eine Wahrscheinlichkeitsverteilung für die Auflösungen:

**Standard-Gewichtung:**
- **large (1080p)**: 50% der Patches
- **small (540p)**: 25% der Patches
- **medium (720p)**: 25% der Patches

---

## Berechnungsformel / Calculation Formula

### Schritt 1: Kategorie-Targets definieren / Define Category Targets

**Beispiel-Konfiguration:**
```python
category_targets = {
    'master': 150000,    # 150,000 Patches für Master
    'space': 60000,      # 60,000 Patches für Space
    'toon': 50000,       # 50,000 Patches für Toon
    'universal': 50000   # 50,000 Patches für Universal
}
```

**Gesamt-Target:** 310,000 Patches insgesamt

### Schritt 2: Video-Proportionen berechnen / Calculate Video Proportions

Jeder Film bekommt einen Anteil basierend auf der Dauer:

```
Proportion = Video_Dauer / Gesamt_Dauer_Aller_Videos
```

**Beispiel:**
- Gesamt-Dauer aller Videos: 1000 Stunden
- Dauer dieses Films: 2.5 Stunden (150 Minuten)
- **Proportion**: 2.5 / 1000 = 0.0025 (0.25%)

### Schritt 3: Target Patches für diesen Film / Target Patches for this Film

```
Target_Patches_Film = Gesamt_Target × Proportion
```

**Beispiel:**
- Gesamt-Target: 310,000 Patches
- Proportion: 0.0025
- **Target Patches**: 310,000 × 0.0025 = 775 Patches

### Schritt 4: Verteilung auf Kategorien / Distribution to Categories

**Regel:** Patches werden GLEICHMÄSSIG auf alle Kategorien des Films verteilt.

Wenn ein Film in mehreren Kategorien ist:
```
Patches_Pro_Kategorie = Target_Patches_Film / Anzahl_Kategorien
```

**Beispiel 1: Film nur in "master"**
- Target Patches: 775
- Kategorien: ['master']
- **master**: 775 Patches

**Beispiel 2: Film in "master" und "space"**
- Target Patches: 775
- Kategorien: ['master', 'space']
- **master**: 775 / 2 = 387 Patches
- **space**: 775 / 2 = 388 Patches (Rundung)

**Beispiel 3: Film in "master", "space" und "toon"**
- Target Patches: 775
- Kategorien: ['master', 'space', 'toon']
- **master**: 775 / 3 = 258 Patches
- **space**: 775 / 3 = 258 Patches
- **toon**: 775 / 3 = 259 Patches (Rest)

### Schritt 5: Format-Verteilung pro Kategorie / Format Distribution per Category

Für jede Kategorie werden die Patches auf Formate verteilt:

**Standard-Gewichtung:**
- large (1080p): 50%
- small (540p): 25%
- medium (720p): 25%

**Beispiel für master mit 387 Patches:**
```
master/large_1080:  387 × 0.50 = 193 Patches
master/small_540:   387 × 0.25 =  96 Patches
master/medium_720:  387 × 0.25 =  98 Patches (Rest)
```

### Schritt 6: Scenes berechnen / Calculate Scenes Needed

**WICHTIG:** Die Anzahl der Scenes wird durch das MAXIMUM aller Formate bestimmt!

```
Scenes_Needed = MAX(alle Format-Targets)
```

**Warum?** Jede Scene erstellt EIN Patch für JEDES Format. Also brauchen wir so viele Scenes wie das größte Format-Target.

**Beispiel:**
```
Format-Distribution:
  master/large_1080:  193 Patches
  master/small_540:    96 Patches
  master/medium_720:   98 Patches
  space/large_1080:   193 Patches
  space/small_540:     97 Patches
  space/medium_720:    98 Patches

Maximum = 193 Patches

Scenes_Needed = 193
```

### Schritt 7: Zeitstempel berechnen / Calculate Timestamps

Die Scenes werden GLEICHMÄSSIG über die gesamte Filmdauer verteilt:

```
Stride_Seconds = (Film_Dauer - 1.0) / Scenes_Needed
```

**Beispiel:**
- Film-Dauer: 7200 Sekunden (2 Stunden)
- Scenes Needed: 193
- **Stride**: (7200 - 1) / 193 = 37.3 Sekunden

Zeitstempel-Berechnung:
```
Timestamp_i = i × Stride_Seconds
```

**Erste Timestamps:**
- Timestamp 0: 0.0s
- Timestamp 1: 37.3s
- Timestamp 2: 74.6s
- ...
- Timestamp 192: 7,161.6s

---

## Vollständige Beispiele / Complete Examples

### Beispiel 1: Kurzer Film (30 Minuten) nur in "master"

**Ausgangsdaten:**
- Film-Dauer: 1800 Sekunden (30 Minuten)
- Kategorien: ['master']
- Gesamt-Target: 310,000 Patches
- Proportion: 0.001 (0.1%)
- Target Patches: 310 Patches

**Berechnung:**

1. **Kategorie-Verteilung:**
   - master: 310 Patches

2. **Format-Verteilung (master):**
   - large_1080: 310 × 0.50 = 155 Patches
   - small_540: 310 × 0.25 = 77 Patches
   - medium_720: 310 × 0.25 = 78 Patches

3. **Scenes benötigt:**
   - Maximum: 155 (large_1080)
   - **Scenes Needed: 155**

4. **Stride:**
   - (1800 - 1) / 155 = 11.6 Sekunden

5. **Patches pro Scene:**
   - 3 Formate × 1 Kategorie = 3 Patches pro Scene

6. **Gesamt erstellt:**
   - 155 Scenes × 3 Patches/Scene = 465 Patches
   - Aber: Nur 310 werden tatsächlich benötigt
   - Die Formate mit weniger Target stoppen früher

**Ergebnis:**
```
Scenes: 155
Stride: 11.6 Sekunden
Extraction Points: 0s, 11.6s, 23.2s, ..., 1788.4s
Total Patches: 310 (über alle Formate)
```

---

### Beispiel 2: Langer Film (2.5 Stunden) in "master" und "space"

**Ausgangsdaten:**
- Film-Dauer: 9000 Sekunden (2.5 Stunden)
- Kategorien: ['master', 'space']
- Gesamt-Target: 310,000 Patches
- Proportion: 0.0025 (0.25%)
- Target Patches: 775 Patches

**Berechnung:**

1. **Kategorie-Verteilung:**
   - master: 775 / 2 = 387 Patches
   - space: 775 / 2 = 388 Patches

2. **Format-Verteilung (master):**
   - large_1080: 387 × 0.50 = 193 Patches
   - small_540: 387 × 0.25 = 96 Patches
   - medium_720: 387 × 0.25 = 98 Patches

3. **Format-Verteilung (space):**
   - large_1080: 388 × 0.50 = 194 Patches
   - small_540: 388 × 0.25 = 97 Patches
   - medium_720: 388 × 0.25 = 97 Patches

4. **Scenes benötigt:**
   - Maximum: 194 (space/large_1080)
   - **Scenes Needed: 194**

5. **Stride:**
   - (9000 - 1) / 194 = 46.4 Sekunden

6. **Patches pro Scene:**
   - 3 Formate × 2 Kategorien = 6 Patches pro Scene

7. **Gesamt erstellt:**
   - 194 Scenes × 6 Patches/Scene = 1,164 Patches potentiell
   - Aber: Nur 775 werden tatsächlich benötigt
   - Die Formate mit weniger Target stoppen früher

**Ergebnis:**
```
Scenes: 194
Stride: 46.4 Sekunden
Extraction Points: 0s, 46.4s, 92.8s, ..., 8,953.6s
Total Patches: 775 (über alle Formate und Kategorien)

Verteilung:
  master/large_1080:  193 Patches
  master/small_540:    96 Patches
  master/medium_720:   98 Patches
  space/large_1080:   194 Patches
  space/small_540:     97 Patches
  space/medium_720:    97 Patches
```

---

### Beispiel 3: Animations-Film in "master", "toon", "universal"

**Ausgangsdaten:**
- Film-Dauer: 5400 Sekunden (1.5 Stunden)
- Kategorien: ['master', 'toon', 'universal']
- Gesamt-Target: 310,000 Patches
- Proportion: 0.002 (0.2%)
- Target Patches: 620 Patches

**Berechnung:**

1. **Kategorie-Verteilung:**
   - master: 620 / 3 = 206 Patches
   - toon: 620 / 3 = 207 Patches
   - universal: 620 / 3 = 207 Patches

2. **Format-Verteilung (master):**
   - large_1080: 206 × 0.50 = 103 Patches
   - small_540: 206 × 0.25 = 51 Patches
   - medium_720: 206 × 0.25 = 52 Patches

3. **Format-Verteilung (toon):**
   - large_1080: 207 × 0.50 = 103 Patches
   - small_540: 207 × 0.25 = 51 Patches
   - medium_720: 207 × 0.25 = 53 Patches

4. **Format-Verteilung (universal):**
   - large_1080: 207 × 0.50 = 103 Patches
   - small_540: 207 × 0.25 = 51 Patches
   - medium_720: 207 × 0.25 = 53 Patches

5. **Scenes benötigt:**
   - Maximum: 103 (mehrere Formate haben 103)
   - **Scenes Needed: 103**

6. **Stride:**
   - (5400 - 1) / 103 = 52.4 Sekunden

7. **Patches pro Scene:**
   - 3 Formate × 3 Kategorien = 9 Patches pro Scene

8. **Gesamt erstellt:**
   - 103 Scenes × 9 Patches/Scene = 927 Patches potentiell
   - Aber: Nur 620 werden tatsächlich benötigt

**Ergebnis:**
```
Scenes: 103
Stride: 52.4 Sekunden
Extraction Points: 0s, 52.4s, 104.8s, ..., 5,345.2s
Total Patches: 620 (über alle Formate und Kategorien)

Verteilung:
  master/large_1080:   103 Patches
  master/small_540:     51 Patches
  master/medium_720:    52 Patches
  toon/large_1080:     103 Patches
  toon/small_540:       51 Patches
  toon/medium_720:      53 Patches
  universal/large_1080: 103 Patches
  universal/small_540:   51 Patches
  universal/medium_720:  53 Patches
```

---

### Beispiel 4: Sehr kurzer Film (10 Minuten) in allen Kategorien

**Ausgangsdaten:**
- Film-Dauer: 600 Sekunden (10 Minuten)
- Kategorien: ['master', 'space', 'toon', 'universal']
- Gesamt-Target: 310,000 Patches
- Proportion: 0.0003 (0.03%)
- Target Patches: 93 Patches

**Berechnung:**

1. **Kategorie-Verteilung:**
   - master: 93 / 4 = 23 Patches
   - space: 93 / 4 = 23 Patches
   - toon: 93 / 4 = 23 Patches
   - universal: 93 / 4 = 24 Patches (Rest)

2. **Format-Verteilung (master, space, toon - je 23 Patches):**
   - large_1080: 23 × 0.50 = 11 Patches
   - small_540: 23 × 0.25 = 5 Patches
   - medium_720: 23 × 0.25 = 7 Patches

3. **Format-Verteilung (universal - 24 Patches):**
   - large_1080: 24 × 0.50 = 12 Patches
   - small_540: 24 × 0.25 = 6 Patches
   - medium_720: 24 × 0.25 = 6 Patches

4. **Scenes benötigt:**
   - Maximum: 12 (universal/large_1080)
   - **Scenes Needed: 12**

5. **Stride:**
   - (600 - 1) / 12 = 49.9 Sekunden

6. **Patches pro Scene:**
   - 3 Formate × 4 Kategorien = 12 Patches pro Scene

7. **Gesamt erstellt:**
   - 12 Scenes × 12 Patches/Scene = 144 Patches potentiell
   - Aber: Nur 93 werden tatsächlich benötigt

**Ergebnis:**
```
Scenes: 12
Stride: 49.9 Sekunden  
Extraction Points: 0s, 49.9s, 99.8s, ..., 548.9s
Total Patches: 93 (über alle Formate und Kategorien)
```

---

## Spezialfälle / Special Cases

### Fall 1: Film mit unterschiedlichen Format-Gewichtungen

Manchmal haben Kategorien unterschiedliche Format-Verteilungen:

**Beispiel:**
```python
format_probabilities = {
    'master': {
        'large_1080': 0.60,  # 60% statt 50%
        'small_540': 0.20,   # 20% statt 25%
        'medium_720': 0.20   # 20% statt 25%
    },
    'space': {
        'large_1080': 0.50,
        'small_540': 0.30,
        'medium_720': 0.20
    }
}
```

**Film-Daten:**
- Target Patches: 500
- Kategorien: ['master', 'space']
- Patches pro Kategorie: 250 each

**Format-Verteilung:**

master (250 Patches):
- large_1080: 250 × 0.60 = 150 Patches
- small_540: 250 × 0.20 = 50 Patches
- medium_720: 250 × 0.20 = 50 Patches

space (250 Patches):
- large_1080: 250 × 0.50 = 125 Patches
- small_540: 250 × 0.30 = 75 Patches
- medium_720: 250 × 0.20 = 50 Patches

**Scenes Needed:** MAX(150, 50, 50, 125, 75, 50) = **150 Scenes**

### Fall 2: Sehr langer Film (5+ Stunden)

**Ausgangsdaten:**
- Film-Dauer: 18000 Sekunden (5 Stunden)
- Target Patches: 2000
- Kategorien: ['master']

**Berechnung:**
- master: 2000 Patches
- large_1080: 1000 Patches
- **Scenes Needed: 1000**
- **Stride:** (18000 - 1) / 1000 = 17.99 Sekunden

**Hinweis:** Bei sehr langen Filmen kann der Stride sehr klein werden!

### Fall 3: Sehr kurzer Film (< 5 Minuten)

**Ausgangsdaten:**
- Film-Dauer: 240 Sekunden (4 Minuten)
- Target Patches: 50
- Kategorien: ['master']

**Berechnung:**
- master: 50 Patches
- large_1080: 25 Patches
- **Scenes Needed: 25**
- **Stride:** (240 - 1) / 25 = 9.56 Sekunden

**Hinweis:** Minimum Stride ist 0.5 Sekunden, um nicht zu häufig zu extrahieren.

---

## Zusammenfassung / Summary

### Formel-Übersicht / Formula Overview

```
1. Proportion = Video_Dauer / Gesamt_Dauer
2. Target_Patches = Gesamt_Target × Proportion
3. Patches_Pro_Kategorie = Target_Patches / Anzahl_Kategorien
4. Format_Patches = Kategorie_Patches × Format_Probability
5. Scenes_Needed = MAX(alle Format_Patches)
6. Stride = (Dauer - 1) / Scenes_Needed
7. Timestamps = [i × Stride for i in range(Scenes_Needed)]
```

### Wichtige Regeln / Important Rules

1. **Gleichmäßige Kategorie-Verteilung:** Jede Kategorie bekommt gleich viele Patches
2. **Format-Gewichtung:** Standard ist 50% large, 25% small, 25% medium
3. **Maximum-Prinzip:** Scenes = MAX aller Format-Targets
4. **Gleichmäßige Verteilung:** Timestamps über gesamte Filmdauer
5. **Mindest-Stride:** 0.5 Sekunden (nicht zu häufig extrahieren)

### Patch-Zählung / Patch Counting

- **Pro Scene:** Patches = Anzahl_Formate × Anzahl_Kategorien
- **Gesamt:** Nur Target-Anzahl wird erstellt (andere Formate stoppen früher)

---

## Technische Details / Technical Details

### Frame-Extraktion

Pro Scene werden 7 aufeinanderfolgende Frames extrahiert:
- Frame 1-7 bei Timestamp t

Diese 7 Frames werden verwendet um:
- Ground Truth (GT) zu erstellen
- Low Resolution (LR) Patches zu generieren

### Patch-Erstellung

Aus 7 Frames entsteht 1 Patch-Paar:
- GT: Hochauflösend (z.B. 1920×1080)
- LR: Herunterskaliert (z.B. 960×540)

### Speicherort

Patches werden gespeichert in:
```
{output_dir}/{category}/{resolution}/
```

Beispiel:
```
output/master/540/Avatar_00001234.png
output/space/1080/Avatar_00005678.png
```

---

## Praktische Tipps / Practical Tips

### Optimale Scene-Anzahl

- **Zu wenig Scenes:** Nicht genug Patches
- **Zu viel Scenes:** Längere Extraktion, evtl. redundante Frames

**Empfehlung:** 
- Mindestens 10 Sekunden Stride
- Maximal 1 Scene pro Sekunde

### Memory-Management

Bei der incrementellen Verarbeitung:
- Nur 7 Frames gleichzeitig im RAM
- Nach Verarbeitung sofort löschen
- Konstante Memory-Nutzung (~45 MB)

### Fortschritt verfolgen

Der Generator loggt:
```
📍 Scene 42/194: timestamp 1,948.8s
  ✓ Created 2 patches from this scene
  📊 Progress: 42/194 scenes processed, 84 total patches created
```

---

## Anhang: Beispiel-Output / Appendix: Example Output

### Console-Log für Beispiel 2

```
╔══════════════════════════════════════════════════════════╗
║  INCREMENTAL EXTRACTION AND PROCESSING                   ║
╚══════════════════════════════════════════════════════════╝
📹 Video: Avatar (2009)
🎯 Target: 775 patches across 2 categories

📋 Phase 1: Calculating extraction plan...
✓ Format distribution analysis:
  Total target patches: 775
  Patches per scene: 6 (2 categories × 3 formats)
  Maximum patches in any single format: 194
  Scenes needed: 194
  Expected total patches created: 1164

  Video duration: 9000.0s
  Video FPS: 24.00
  Total frames in video: 216000
  Calculated stride: 46.4s = 1113 frames

✓ Planned 194 extraction points (scenes)
  Extraction pattern: One scene every 1113 frames
  Each scene: 7 consecutive frames
  First timestamp: 0.00s (0.0% of video)
  Last timestamp: 8953.52s (99.5% of video)
  Coverage: Entire video from start to end
  Total frames to extract: 1358

🔄 Starting incremental extraction and processing...
  Processing 194 timestamps one at a time

📍 Scene 1/194: timestamp 0.000000s
  🎬 Extracting 7 frames...
  ✓ Extracted 7 frames
    ✓ Saved patch: master/1080 → Avatar_00000000.png
    ✓ Saved patch: space/1080 → Avatar_00000000.png
  ✓ Created 2 patches from this scene
  📊 Progress: 1/194 scenes, 2 total patches

📍 Scene 2/194: timestamp 46.392s
  🎬 Extracting 7 frames...
  ✓ Extracted 7 frames
    ✓ Saved patch: master/1080 → Avatar_00046392.png
    ✓ Saved patch: space/1080 → Avatar_00046392.png
  ✓ Created 2 patches from this scene
  �� Progress: 2/194 scenes, 4 total patches

[...]

📍 Scene 194/194: timestamp 8953.520s
  🎬 Extracting 7 frames...
  ✓ Extracted 7 frames
    ✓ Saved patch: master/1080 → Avatar_08953520.png
    ✓ Saved patch: space/1080 → Avatar_08953520.png
  ✓ Created 2 patches from this scene
  📊 Progress: 194/194 scenes, 775 total patches

✅ EXTRACTION COMPLETE
  Total scenes processed: 194
  Total patches created: 775
  Processing time: 24.5 minutes
```

---

**Ende des Dokuments / End of Document**

**Version:** 1.0
**Datum / Date:** 2026-02-12
**Autor / Author:** Dataset Generator Team
