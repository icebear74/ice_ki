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
- `--min-face-area-ratio` (default `0.04`)
- `--min-sharpness` (default `40.0`)
- `--min-stability` (default `0.18`)

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
