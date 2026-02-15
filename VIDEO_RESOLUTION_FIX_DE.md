# Video-Auflösung Fix - Zusammenfassung

## Problem
Das Original-Video war PAL (720×576), aber das KI-kodierte Ergebnis war 540×540 (quadratisch, falsches Seitenverhältnis).

**Erwartet:** 3-fach skaliert mit korrektem Seitenverhältnis = 2160×1728

## Ursache
Die Funktion `extract_frames_from_video()` hat alle Frames auf 180×180 quadratisch gezwungen:
```python
'-vf', f'scale={target_size}:{target_size}:flags=lanczos'
```

Dies führte zu:
1. Zwang auf 1:1 Seitenverhältnis (quadratisch)
2. Dann 3x Upscaling durch das Modell: 180×180 → 540×540 (immer noch quadratisch)

## Lösung
Die LR (Low Resolution) Eingabe verwendet **immer die Original-Video-Auflösung**, dann skaliert das Modell 3x hoch:

- **LR Input:** Original-Video-Auflösung (z.B. 720×576 für PAL)
- **HR Output:** 3x hochskaliert (z.B. 2160×1728)

## Änderungen

### 1. Neue Funktion: `get_video_resolution()`
Ermittelt die Video-Breite und -Höhe mit ffprobe:
```python
def get_video_resolution(video_path):
    # Verwendet ffprobe um Breite und Höhe zu ermitteln
    return width, height
```

### 2. Aktualisiert: `extract_frames_from_video()`
- Erkennt Original-Auflösung
- **Erhält Seitenverhältnis** (kein erzwungenes Quadrat mehr)
- Verwendet Original-Auflösung als Standard (scale_factor=None)
- Gibt Auflösungs-Tupel `(width, height)` zurück

**Vorher:**
```python
extract_frames_from_video(video_path, output_dir, target_size=180)
# Zwang auf 180×180 quadratisch
```

**Nachher:**
```python
extract_frames_from_video(video_path, output_dir, scale_factor=None)
# Behält Original-Auflösung bei
```

### 3. Hauptfunktion zeigt Auflösungen an
```
Original-Auflösung: 720×576
Behalte Original-Auflösung bei
LR Input: 720×576
HR Output wird: 2160×1728 (3x Upscaling)
```

## Beispiele

### PAL Video (720×576)
- **Input (LR):** 720×576
- **Output (HR):** 2160×1728 (3x)
- **Seitenverhältnis:** 15:12 (1.25) - **erhalten!**

### HD 720p (1280×720)
- **Input (LR):** 1280×720
- **Output (HR):** 3840×2160 (3x)
- **Seitenverhältnis:** 16:9 (1.778) - **erhalten!**

### Full HD 1080p (1920×1080)
- **Input (LR):** 1920×1080
- **Output (HR):** 5760×3240 (3x)
- **Seitenverhältnis:** 16:9 (1.778) - **erhalten!**

## Tests
Alle Tests bestanden ✅

- Funktionssignaturen validiert
- Auflösungsberechnung getestet für PAL, HD, Full HD, VGA
- Seitenverhältnis-Erhaltung verifiziert
- Kein erzwungenes quadratisches Seitenverhältnis

## Verwendung
Keine Änderungen an der Verwendung erforderlich. Das Skript funktioniert genauso wie vorher, aber jetzt mit korrekten Auflösungen:

```bash
python run_video_inference.py --input video.mkv --output result.mkv
```

Das Skript wird nun:
1. Die Original-Auflösung erkennen (z.B. 720×576)
2. Frames in Original-Auflösung extrahieren
3. Durch das Modell verarbeiten (3x Upscaling)
4. Video mit 3x Auflösung erstellen (z.B. 2160×1728)

**Das Seitenverhältnis wird perfekt erhalten!** 🎉
