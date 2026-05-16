# ice_audio_nexus (Step-1 visual person discovery)

`ice_audio_nexus` was rewritten from a voice-first prototype to a **face-first Step-1 pipeline**.

## Step-1 goal

For each video the scanner now:

1. samples frames (default `4 FPS`)
2. detects faces
3. builds short local tracks
4. promotes only **clear tracks** (high precision, low noise)
5. stores detections, tracks, representative crops, embeddings/descriptors and review metadata

This phase is intentionally person-centric and review-driven. Full role/speaker resolution is left for later phases.

## Core model (visual-first)

- `actors`: real persons
- `productions`, `videos`: source catalog
- `roles`, `actor_roles`: optional role/context mapping for later expansion
- `face_detections`: per-frame detections + bbox + crop + descriptor
- `face_tracks`: aggregated local tracks + quality/relevance status + assignment/match fields
- `face_samples`: reusable confirmed face samples per actor (incremental learning)
- `overlay_events`: precomputed label/bbox timeline for browser video overlay

## Incremental learning behavior

- Assigning a track to an actor can auto-create a new `face_sample`.
- New samples let the system learn appearance changes (beard, haircuts, lighting, pose).
- `POST /api/rematch` re-evaluates tracks against confirmed actor samples.

## Scanner usage

```bash
cd ice_audio_nexus
source venv/bin/activate

python -m processor.scanner \
  --video /absolute/path/to/video.mkv \
  --production "The Big Bang Theory" \
  --title "S05E01 The Skank Reflex Analysis"
```

Useful options:

- `--fps` (default `4.0`)
- `--min-clear-seconds` (default `2.0`)
- `--min-face-area-ratio` (default `0.06`)
- `--min-sharpness` (default `70.0`)
- `--min-stability` (default `0.45`)
- `--dnn-confidence` (default `0.65`)
- `--min-face-size-px` (default `80`)
- `--disable-verifier` (deaktiviert die zweite KI-Verifikation)
- `--verifier-score-threshold` (default `0.92`)
- `--cpu-only` / `--gpu-device-id` / `--gpu-diagnostics`

## False-Positive-Reduktion (2-stufig)

Der Scanner verwendet jetzt zwei Modelle:

1. OpenCV ResNet-SSD (erste Detection-Stufe)
2. YuNet-Verifier (zweite Stufe, bestätigt nur plausible Gesichter)

Treffer aus Stufe 1 werden vor dem Persistieren verworfen, wenn die Verifikation
fehlschlägt (Score/Fläche/Zentrierung konfigurierbar via `.env`).

## GPU/CUDA-Diagnostik

OpenCV-DNN nutzt GPU nur, wenn OpenCV mit CUDA gebaut wurde. Prüfen mit:

```bash
python -m processor.scanner --diagnose-opencv
```

Im Setup (`setup_env.sh`) wird zusätzlich nach Installation automatisch ausgegeben:

- OpenCV-Version
- CUDA/cuDNN-Build-Status
- von OpenCV sichtbare CUDA-Geräte

## Rescan-Cleanup

Beim (Re-)Scan werden alte Bilddaten (`data/faces/crops/<video>` und
`data/faces/tracks/<video>`) vor dem neuen Lauf gelöscht, damit Festplatte und DB
synchron bleiben.

## Verifikation nach Merge (Self-Checks)

1. **GPU vs CPU aktiv?**
   - Scan starten und Log prüfen auf `Face detector: CUDA backend active` oder CPU-Fallback.
2. **OpenCV mit CUDA gebaut?**
   - `python -m processor.scanner --diagnose-opencv`
3. **Verifier aktiv?**
   - Log prüfen auf `Face verifier: enabled ...`
   - Scan-Ergebnis enthält `verifier_rejections`.
4. **False Positives reduziert?**
   - Vorher/Nachher mit gleichem Video vergleichen (`detections`, `clear_tracks`, `verifier_rejections`).
5. **Rescan räumt alte Bilder auf?**
   - Vor Rescan Dateianzahl in `data/faces/crops/<video>` prüfen, dann Rescan starten und Log `Removed stale scan images` verifizieren.

## Web UI

Start server:

```bash
uvicorn web_ui.api:app --host 0.0.0.0 --port 8765
```

UI provides:

- browse productions/videos
- inspect discovered tracks and representative crops
- assign track to existing actor or create new actor (+ optional role)
- mark track as unknown/background/ignored
- precomputed video overlay (bbox + label)
- re-match button for post-assignment re-evaluation

## Environment

Minimal required env keys:

- `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`
- `VIDEO_DIR` (root for allowed video streaming)
- optional: `FACE_DATA_DIR` (defaults to `data/faces`)

See `.env.example` for defaults and thresholds.
