# ice_audio_nexus (Step-1 visual person discovery)

`ice_audio_nexus` was rewritten from a voice-first prototype to a **face-first Step-1 pipeline**.
The current follow-up direction is a **seed-first** review flow: find strong visual seeds first, review them conservatively, and only use tracking as later expansion/support context.

## Step-1 goal

For each video in seed mode (`--video`) the scanner now:

1. samples frames (default `4 FPS`)
2. detects faces
3. applies strict quality gates (size/sharpness/brightness/quality/verifier)
4. accepts high-quality face seeds
5. assigns seeds conservatively to visual groups (high-similarity only)
6. stores detections, seed containers and seed-review metadata

This phase is intentionally person-centric and review-driven. Full role/speaker resolution is left for later phases.
`README_FIRST.md` is the binding project goal: Step 1 must prefer identity purity over raw detection count because these visual anchors later seed audio / speaker attribution.

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

## Scanner usage (2 modes)

```bash
cd ice_audio_nexus
source venv/bin/activate

# Mode A: seed discovery for one episode/file
python -m processor.scanner \
  --video /absolute/path/to/video.mkv \
  --production "The Big Bang Theory" \
  --title "S05E01 The Skank Reflex Analysis"

# Mode B: expansion orchestrator (without --video/--dir)
python -m processor.scanner
```

Useful options:

- `--fps` (default `4.0`)
- `--start-offset-seconds` / `--max-sampled-frames`
- `--min-clear-seconds` (default `2.0`)
- `--min-face-area-ratio` (default `0.06`)
- `--max-aspect-ratio-deviation` (default `0.65`)
- `--min-sharpness` (default `70.0`)
- `--min-brightness` / `--min-quality-score` / `--seed-acceptance-threshold`
- `--min-stability` (default `0.45`)
- `--dnn-confidence` (Torch detector score threshold, default `0.65`)
- `--min-face-size-px` (default `80`)
- `--disable-verifier` (deaktiviert die zweite KI-Verifikation)
- `--verifier-score-threshold` (default `0.92`)
- `--cpu-only` / `--gpu-device-id` / `--detector-device` / `--verifier-device` / `--embedding-device`
- `--write-debug-stats` / `--debug-stats-dir`
- `--diagnose-torch` (Torch/CUDA Diagnostik ohne Scan)

Seed-Logs zeigen nun granular:

- `rejected_small`
- `rejected_blurry`
- `rejected_pose`
- `rejected_occluded`
- `rejected_dark`
- `rejected_quality_score`
- `verifier_rejects`
- `duplicate_matches`
- `high_quality_seeds_accepted`
- `new_visual_groups_created`
- `matched_existing_groups`

Optional schreibt der Scanner pro Lauf eine JSON-Datei (`FACE_SEED_DEBUG_STATS_ENABLED=1`) nach
`data/faces/debug/seed_runs` (oder `FACE_SEED_DEBUG_STATS_DIR`).

## False-Positive-Reduktion (2-stufig)

Der Scanner verwendet jetzt zwei Modelle:

1. Torch MTCNN Detector (erste Detection-Stufe)
2. Torch MTCNN Verifier (zweite Stufe, bestätigt nur plausible Gesichter)

Treffer aus Stufe 1 werden vor dem Persistieren verworfen, wenn die Verifikation
fehlschlägt (Score/Fläche/Zentrierung konfigurierbar via `.env`).

## Modell-Download & lokaler Cache

- Die Torch-Modelle werden beim ersten Start automatisch geladen.
- Standard-Cache: `FACE_DATA_DIR/models` (optional überschreibbar via `FACE_MODELS_DIR`).
- Scanner verwendet:
  - `TORCH_HOME=<models>/torch_home`
  - `HF_HOME=<models>/huggingface`
- Bei Downloadfehlern bricht der Scanner mit klarer Fehlermeldung ab
  (Internet/Proxy prüfen oder Cache vorab befüllen).

## GPU/CUDA-Diagnostik

Die Inferenz läuft über PyTorch (CUDA wenn verfügbar), OpenCV bleibt nur für I/O
(`VideoCapture`, Farbraum/Resize/Crops, Bildspeichern).

Komponenten können getrennt zugewiesen werden:

- `FACE_DETECTOR_DEVICE`
- `FACE_VERIFIER_DEVICE`
- `FACE_EMBEDDING_DEVICE`

Wenn ein Device fehlt, ungültig ist oder nicht kompatibel ist, fällt die jeweilige Komponente auf CPU zurück.

Torch-Status prüfen mit:

```bash
python -m processor.scanner --diagnose-torch
```

Im Setup (`setup_env.sh`) wird zusätzlich nach Installation automatisch ausgegeben:

- Torch-Version
- CUDA-Verfügbarkeit / sichtbare Geräte
- ausgewähltes Scanner-Device (`cpu` oder `cuda:<id>`)
- OpenCV-Version (nur I/O)

## Rescan-Cleanup

Beim (Re-)Scan werden alte Bilddaten (`data/faces/crops/<video>` und
`data/faces/tracks/<video>`) vor dem neuen Lauf gelöscht, damit Festplatte und DB
synchron bleiben.

Zusätzlich räumt der Video-Rescan jetzt die scanbezogenen DB-Artefakte dieses Videos
konsequent auf:

- `overlay_events` des Videos
- `face_detections` des Videos
- `face_tracks` des Videos
- `visual_seeds`, die auf diesen Detections/Tracks oder deren Crop-Pfaden basieren
- betroffene `visual_groups` werden bei leerem Seed-Bestand kontrolliert auf
  `review_state=ignored` + `expansion_state=blocked` gesetzt (statt blind zu löschen)

Globale Stammdaten (`actors`, `voice_actors`, `roles`, Persona-/Cast-Stammdaten)
bleiben beim Rescan unberührt.

## Verifikation nach Merge (Self-Checks)

1. **GPU vs CPU aktiv?**
   - Scan starten und Log prüfen auf `Face detector: Torch MTCNN active (gpu / cuda:...)` oder CPU-Fallback.
2. **Torch CUDA Status prüfen**
   - `python -m processor.scanner --diagnose-torch`
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
- **Group-first Step-1 Review**: review state, expansion gate, actor/role assignment direkt auf `visual_groups` (inkl. Neuanlage Actor/Rolle)
- vollständige Group-Zuordnung auch ohne bereits brauchbare Tracks
- tracks als Support-Ebene für spätere Kontrolle/Bereinigung/Expansion
- inspect discovered tracks and representative crops
- mark track as unknown/background/ignored
- precomputed video overlay (bbox + label)
- re-match button for post-assignment re-evaluation

## Environment

Minimal required env keys:

- `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`
- `VIDEO_DIR` (root for allowed video streaming)
- optional: `FACE_DATA_DIR` (defaults to `data/faces`)

See `.env.example` for defaults and thresholds.
