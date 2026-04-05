"""
ice_audio_nexus – processor/remigrate.py
-----------------------------------------
Auto-Migration routine: re-embeds existing voice_samples with WeSpeaker.

Ablauf pro Sample:
  1. Zugehörige WAV-Datei über episode_segments.tts_wav_path ermitteln.
  2. Original-WAV nach <name>.orig.wav sichern (TTS-Daten bleiben erhalten).
  3. DeepFilterNet auf dem Original anwenden → <name>.clean.wav.
  4. Drift-Check: WAV in 3 Sub-Segmente aufteilen, WeSpeaker-Embedding pro
     Sub-Segment berechnen, maximale paarweise Kosinus-Distanz ermitteln.
     Wenn > DRIFT_THRESHOLD → is_low_quality=True (Sprecherwechsel erkannt).
  5. WeSpeaker-Embedding aus der vollständigen bereinigten WAV extrahieren.
  6. voice_samples.embedding, embedding_model und drift_score aktualisieren.
  7. Wenn keine WAV vorhanden: embedding_model='pyannote-legacy', is_active=False.
  8. Nach allen Samples: refresh_supervectors() aufrufen.

Aufruf:
    python -m processor.remigrate [--identity-id ID] [--all] [--dry-run]

Umgebungsvariablen (aus .env gelesen):
    WESPEAKER_MODEL   – WeSpeaker-Modell (default: english)
    DRIFT_THRESHOLD   – Schwellenwert für Drift-Ablehnung (default: 0.15)
    AUDIO_TMP_DIR     – Verzeichnis für temporäre Audiodateien
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile

import numpy as np
from dotenv import load_dotenv

# Load .env from the project root
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)

# Re-use settings from scanner module to stay consistent
from processor.scanner import (  # noqa: E402
    WESPEAKER_MODEL,
    WESPEAKER_DEVICE,
    DRIFT_THRESHOLD,
    AUDIO_TMP_DIR,
    _WESPEAKER_EMBED_TAG,
    _get_wespeaker_model,
    _extract_wespeaker_embedding,
    compute_drift_score,
    apply_deepfilter,
    _fallback_to_16k,
)

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
import sys
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from db.database import (  # noqa: E402
    get_connection,
    ensure_schema,
    refresh_supervectors,
    update_voice_sample_embedding,
    normalize_vector,
    vector_to_bytes,
    bytes_to_vector,
)


# ---------------------------------------------------------------------------
# WAV lookup
# ---------------------------------------------------------------------------

def _find_tts_wav_for_sample(conn, sample_id: int) -> str | None:
    """
    Return the tts_wav_path stored in episode_segments for *sample_id*, or
    None if no WAV has been extracted for this sample yet.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT tts_wav_path
        FROM episode_segments
        WHERE matched_sample_id = ?
          AND tts_wav_path IS NOT NULL
        LIMIT 1
        """,
        (sample_id,),
    )
    row = cur.fetchone()
    cur.close()
    if row and row[0]:
        p = str(row[0])
        return p if os.path.isfile(p) else None
    return None


# ---------------------------------------------------------------------------
# Core re-embedding logic
# ---------------------------------------------------------------------------

def _process_wav_for_embedding(
    wav_path: str,
    dry_run: bool = False,
) -> tuple[list[float], float | None]:
    """
    Given the path to a raw TTS WAV file:
      1. Preserve the original as <path>.orig.wav (idempotent – skipped if
         .orig.wav already exists so repeated runs don't stack-copy).
      2. Apply DeepFilterNet to produce a cleaned 16-kHz WAV.
      3. Compute the drift score from the cleaned WAV.
      4. Extract a WeSpeaker embedding from the cleaned WAV.

    Returns ``(embedding_list, drift_score)`` where *embedding_list* may be
    empty on failure.

    The ``.clean.wav`` file is removed after extraction; the ``.orig.wav``
    is kept permanently for TTS training.
    """
    orig_path  = wav_path + ".orig.wav"
    clean_path = wav_path + ".clean.wav"

    if dry_run:
        logger.info("[DRY-RUN] würde verarbeiten: %s", wav_path)
        return [], None

    # --- Step 1: Preserve original ---
    if not os.path.exists(orig_path):
        shutil.copy2(wav_path, orig_path)
        logger.debug("Original gesichert: %s", orig_path)

    # --- Step 2: DeepFilter cleaning ---
    # The TTS WAV is already 16 kHz mono (extracted by _extract_tts_snippet).
    # DeepFilterNet expects 48 kHz input; resample up first.
    tmp_48k_fd, tmp_48k = tempfile.mkstemp(suffix=".48k.wav", dir=AUDIO_TMP_DIR)
    os.close(tmp_48k_fd)
    try:
        import subprocess
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", orig_path,
                "-ar", "48000", "-ac", "1", "-acodec", "pcm_s16le",
                tmp_48k,
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Resampling zu 48 kHz fehlgeschlagen:\n"
                f"{result.stderr.decode(errors='replace')}"
            )
        apply_deepfilter(tmp_48k, clean_path)
    except Exception as exc:
        logger.warning(
            "DeepFilterNet konnte nicht angewendet werden (%s) – "
            "verwende unkomprimiertes Original für Embedding.", exc
        )
        try:
            _fallback_to_16k(orig_path, clean_path)
        except Exception as fb_exc:
            logger.error("Fallback-Downsampling fehlgeschlagen: %s", fb_exc)
            return [], None
    finally:
        try:
            os.unlink(tmp_48k)
        except OSError:
            pass

    if not os.path.exists(clean_path):
        logger.error("Bereinigte WAV wurde nicht erstellt: %s", clean_path)
        return [], None

    # --- Steps 3+4: Drift check and embedding ---
    try:
        import soundfile as sf
        audio_data, sample_rate = sf.read(clean_path, dtype="float32", always_2d=False)

        drift_score = compute_drift_score(audio_data, sample_rate)
        embedding   = _extract_wespeaker_embedding(audio_data, sample_rate)
    except Exception as exc:
        logger.error("Embedding-Extraktion fehlgeschlagen: %s", exc)
        embedding, drift_score = [], None
    finally:
        try:
            os.unlink(clean_path)
        except OSError:
            pass

    return embedding, drift_score


# ---------------------------------------------------------------------------
# Per-identity migration
# ---------------------------------------------------------------------------

def remigrate_identity(
    conn,
    identity_id: int,
    dry_run: bool = False,
) -> dict:
    """
    Re-embed all eligible voice_samples for *identity_id* with WeSpeaker.

    Returns a summary dict::

        {
          "processed":      int,   # samples with a WAV found and re-embedded
          "rejected_drift": int,   # samples flagged is_low_quality via drift
          "legacy":         int,   # samples without a WAV → marked legacy
          "skipped":        int,   # samples that failed for other reasons
        }
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id
        FROM voice_samples
        WHERE identity_id = ?
          AND (context IS NULL OR context != 'SUPERVECTOR')
          AND is_active = TRUE
        ORDER BY created_at
        """,
        (identity_id,),
    )
    sample_ids = [row[0] for row in cur.fetchall()]
    cur.close()

    stats = {"processed": 0, "rejected_drift": 0, "legacy": 0, "skipped": 0}

    for sample_id in sample_ids:
        wav_path = _find_tts_wav_for_sample(conn, sample_id)

        if wav_path is None:
            # No WAV available – mark as pyannote-legacy and deactivate so
            # it is excluded from supervector construction.
            logger.info(
                "Sample %d: keine WAV gefunden → pyannote-legacy, deaktiviert", sample_id
            )
            if not dry_run:
                cur2 = conn.cursor()
                cur2.execute(
                    """UPDATE voice_samples
                       SET embedding_model = 'pyannote-legacy', is_active = FALSE
                       WHERE id = ?""",
                    (sample_id,),
                )
                conn.commit()
                cur2.close()
            stats["legacy"] += 1
            continue

        embedding, drift_score = _process_wav_for_embedding(wav_path, dry_run=dry_run)

        if not embedding:
            logger.warning("Sample %d: Embedding-Extraktion fehlgeschlagen – übersprungen", sample_id)
            stats["skipped"] += 1
            continue

        # Reject if drift_score exceeds threshold (speaker turn detected)
        is_low_quality: bool | None = None
        if drift_score is not None and drift_score > DRIFT_THRESHOLD:
            is_low_quality = True
            stats["rejected_drift"] += 1
            logger.info(
                "Sample %d: Drift-Score %.3f > %.3f → is_low_quality=True",
                sample_id, drift_score, DRIFT_THRESHOLD,
            )

        if not dry_run:
            update_voice_sample_embedding(
                conn,
                sample_id=sample_id,
                embedding=embedding,
                embedding_model=_WESPEAKER_EMBED_TAG,
                drift_score=drift_score,
                is_low_quality=is_low_quality,
            )

        logger.info(
            "Sample %d re-embedded: drift=%s, model=%s",
            sample_id,
            f"{drift_score:.4f}" if drift_score is not None else "N/A",
            _WESPEAKER_EMBED_TAG,
        )
        stats["processed"] += 1

    return stats


# ---------------------------------------------------------------------------
# Full migration (all identities)
# ---------------------------------------------------------------------------

def remigrate_all(conn, dry_run: bool = False) -> dict:
    """
    Run :func:`remigrate_identity` for every identity in the database, then
    call :func:`refresh_supervectors` to regenerate all supervector centroids
    from the new WeSpeaker embeddings.

    Returns a summary dict keyed by identity name.
    """
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM identities ORDER BY name")
    identities = cur.fetchall()
    cur.close()

    all_stats: dict = {}
    for identity_id, identity_name in identities:
        logger.info("─── Migriere Identität: %s (id=%d) ───", identity_name, identity_id)
        stats = remigrate_identity(conn, identity_id, dry_run=dry_run)
        all_stats[identity_name] = stats
        logger.info(
            "Identität '%s' abgeschlossen: %s",
            identity_name, stats,
        )

    if not dry_run:
        logger.info("Erstelle Supervektoren neu (refresh_supervectors) …")
        sv_summary = refresh_supervectors(conn)
        logger.info("Supervektoren aktualisiert: %s", sv_summary)
        all_stats["_supervectors"] = sv_summary

    return all_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ice_audio_nexus remigrate – WeSpeaker-Migrations-Routine"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all", action="store_true",
        help="Alle Identitäten migrieren und danach Supervektoren neu erstellen",
    )
    group.add_argument(
        "--identity-id", type=int, metavar="ID",
        help="Nur eine einzelne Identität (nach ID) migrieren",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Vorschau: keine Datenbankänderungen, keine Datei-Kopien",
    )
    args = parser.parse_args()

    ensure_schema()
    conn = get_connection()
    try:
        if args.all:
            summary = remigrate_all(conn, dry_run=args.dry_run)
            logger.info("Migration abgeschlossen. Zusammenfassung: %s", summary)
        else:
            stats = remigrate_identity(conn, args.identity_id, dry_run=args.dry_run)
            if not args.dry_run:
                logger.info("Erstelle Supervektoren neu für Identität %d …", args.identity_id)
                sv = refresh_supervectors(conn)
                logger.info("Supervektoren aktualisiert: %s", sv)
            logger.info("Migration abgeschlossen: %s", stats)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
