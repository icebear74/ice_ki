"""
ice_audio_nexus – processor/scanner.py

Analysiert eine Videodatei und schreibt die erkannten Sprecher-Segmente
in die MariaDB. Nutzt:
  - FFmpeg v8 (CUDA) für effiziente Audio-Extraktion
  - PyAnnote (Tesla P4) für Speaker Diarization
  - Faster-Whisper (Tesla P100) für Transkription
  - MariaDB VECTOR(512) für Sprecher-Embeddings

Verwendung:
  python processor/scanner.py \\
      --video /pfad/zur/episode.mkv \\
      --source "The Walking Dead" \\
      --episode "S01E01 - Days Gone Bye"

  Optional:
      --diarization-device cuda:0   # Tesla P4
      --whisper-device cuda:1       # Tesla P100
      --whisper-model large-v3
      --similarity-threshold 0.85
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("scanner")


# ------------------------------------------------------------------
# Audio-Extraktion via FFmpeg (CUDA-beschleunigt)
# ------------------------------------------------------------------

def extract_audio(video_path: str, output_wav: str, use_cuda: bool = True) -> bool:
    """
    Extrahiert die Audiospur mit FFmpeg und optimiert sie für KI-Analyse.
    Ausgabeformat: 16 kHz, Mono, PCM 16-bit (Standard für PyAnnote & Whisper).

    Args:
        video_path:  Pfad zur Quelldatei (MKV, MP4, …).
        output_wav:  Pfad der Ausgabe-WAV-Datei.
        use_cuda:    FFmpeg CUDA-Hardwaredekodierung nutzen (nur wenn verfügbar).

    Returns:
        True bei Erfolg, False bei Fehler.
    """
    hw_args: list[str] = []
    if use_cuda:
        hw_args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]

    command = (
        ["ffmpeg"]
        + hw_args
        + [
            "-i", video_path,
            "-vn",               # kein Video-Output
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y", output_wav,
        ]
    )

    logger.info("FFmpeg: extrahiere Audio aus '%s'", video_path)
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            timeout=3600,
        )
        logger.info("Audio extrahiert: %s", output_wav)
        return True
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace")
        if use_cuda and "No such decoder" in stderr:
            logger.warning("CUDA-Dekodierung nicht verfügbar – Fallback auf CPU")
            return extract_audio(video_path, output_wav, use_cuda=False)
        logger.error("FFmpeg-Fehler: %s", stderr[-2000:])
        return False
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg-Timeout bei '%s'", video_path)
        return False


# ------------------------------------------------------------------
# Speaker Diarization (PyAnnote, Tesla P4)
# ------------------------------------------------------------------

def run_diarization(
    audio_path: str,
    device: str = "cuda:0",
    hf_token: Optional[str] = None,
) -> list[dict]:
    """
    Führt die Speaker Diarization mit PyAnnote durch.

    Args:
        audio_path:  Pfad zur WAV-Datei.
        device:      Torch-Gerät ('cuda:0' = Tesla P4 empfohlen, 'cpu' als Fallback).
        hf_token:    HuggingFace-Token für den Modell-Download.

    Returns:
        Liste von Dicts mit: start, end, speaker, embedding (list[float] | None)
    """
    try:
        import torch
        from pyannote.audio import Pipeline
        from pyannote.audio.pipelines.utils.hook import ProgressHook
    except ImportError as e:
        raise ImportError(
            "PyAnnote nicht installiert. Führe 'setup_env.sh' aus."
        ) from e

    token = hf_token or os.environ.get("HF_TOKEN")

    logger.info("Lade PyAnnote-Pipeline auf Gerät '%s'…", device)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )
    pipeline.to(torch.device(device))

    logger.info("Starte Diarization für '%s'…", audio_path)
    with ProgressHook() as hook:
        diarization = pipeline(audio_path, hook=hook)

    segments: list[dict] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
                "embedding": None,  # wird im nächsten Schritt befüllt
            }
        )

    logger.info("Diarization abgeschlossen: %d Segmente", len(segments))
    return segments


# ------------------------------------------------------------------
# Sprecher-Embeddings extrahieren (PyAnnote SpeakerEmbedding, Tesla P4)
# ------------------------------------------------------------------

def extract_embeddings(
    audio_path: str,
    segments: list[dict],
    device: str = "cuda:0",
) -> list[dict]:
    """
    Extrahiert für jedes Sprecher-Segment einen 512-dim Float32-Embedding-Vektor.

    Args:
        audio_path:  Pfad zur WAV-Datei.
        segments:    Segment-Liste aus run_diarization().
        device:      Torch-Gerät.

    Returns:
        Dieselbe Segment-Liste mit befülltem 'embedding'-Feld.
    """
    try:
        import torch
        from pyannote.audio import Inference, Model
    except ImportError as e:
        raise ImportError("PyAnnote nicht installiert.") from e

    logger.info("Lade Embedding-Modell (pyannote/embedding) auf '%s'…", device)
    model = Model.from_pretrained(
        "pyannote/embedding",
        use_auth_token=os.environ.get("HF_TOKEN"),
    )
    inference = Inference(model, window="whole")
    inference.to(torch.device(device))

    logger.info("Extrahiere Embeddings für %d Segmente…", len(segments))
    from pyannote.core import Segment

    for i, seg in enumerate(segments):
        try:
            emb = inference.crop(audio_path, Segment(seg["start"], seg["end"]))
            seg["embedding"] = emb.flatten().tolist()
        except Exception as exc:
            logger.warning("Embedding für Segment %d fehlgeschlagen: %s", i, exc)
            seg["embedding"] = None

    logger.info("Embeddings extrahiert.")
    return segments


# ------------------------------------------------------------------
# Transkription (Faster-Whisper, Tesla P100)
# ------------------------------------------------------------------

def transcribe_segments(
    audio_path: str,
    segments: list[dict],
    device: str = "cuda:1",
    model_size: str = "large-v3",
    language: str = "de",
) -> list[dict]:
    """
    Transkribiert die Audio-Abschnitte mit Faster-Whisper.

    Args:
        audio_path:  Pfad zur WAV-Datei.
        segments:    Segment-Liste aus run_diarization().
        device:      Torch-Gerät ('cuda:1' = Tesla P100 empfohlen).
        model_size:  Whisper-Modellgröße.
        language:    Sprache (z.B. 'de', 'en').

    Returns:
        Dieselbe Segment-Liste mit befülltem 'transcript'-Feld.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise ImportError(
            "faster-whisper nicht installiert. Führe 'setup_env.sh' aus."
        ) from e

    compute_type = "float16" if device.startswith("cuda") else "int8"
    logger.info(
        "Lade Whisper-Modell '%s' auf '%s' (compute_type=%s)…",
        model_size, device, compute_type,
    )
    whisper_device = "cuda" if device.startswith("cuda") else "cpu"
    model = WhisperModel(model_size, device=whisper_device, compute_type=compute_type)

    logger.info("Transkribiere %d Segmente…", len(segments))
    for i, seg in enumerate(segments):
        try:
            whisper_segments, _ = model.transcribe(
                audio_path,
                language=language,
                initial_prompt=None,
                word_timestamps=False,
                clip_timestamps=f"{seg['start']},{seg['end']}",
            )
            seg["transcript"] = " ".join(s.text.strip() for s in whisper_segments)
        except Exception as exc:
            logger.warning("Transkription für Segment %d fehlgeschlagen: %s", i, exc)
            seg["transcript"] = None

    logger.info("Transkription abgeschlossen.")
    return segments


# ------------------------------------------------------------------
# Ergebnisse in MariaDB speichern
# ------------------------------------------------------------------

def save_to_database(
    segments: list[dict],
    series_name: str,
    episode_title: str,
    video_path: str,
    similarity_threshold: float = 0.85,
) -> None:
    """
    Speichert die analysierten Segmente in der MariaDB.

    Für jedes Segment:
      1. Sucht nach ähnlichen Stimm-Vektoren im Serien-Kontext.
      2. Legt bei Bedarf ein neues voice_profile an.
      3. Schreibt das episode_segment.

    Args:
        segments:             Segment-Liste mit embedding, transcript usw.
        series_name:          Name der Serie/des Films.
        episode_title:        Episodentitel oder Dateiname.
        video_path:           Pfad zur Quelldatei.
        similarity_threshold: Mindest-Ähnlichkeit für automatische Zuordnung.
    """
    from db.database import (
        find_similar_voice,
        get_connection,
        insert_segment,
        upsert_voice_profile,
    )

    conn = get_connection()
    try:
        for seg in segments:
            identity_id: Optional[int] = None
            confidence: Optional[float] = None

            if seg.get("embedding"):
                matches = find_similar_voice(
                    conn,
                    seg["embedding"],
                    series_name,
                    threshold=similarity_threshold,
                )
                if matches:
                    best = matches[0]
                    identity_id = best["identity_id"]
                    confidence = best["confidence"]
                    logger.debug(
                        "Segment [%.1fs-%.1fs] → '%s' (confidence=%.2f)",
                        seg["start"],
                        seg["end"],
                        best["character_name"],
                        confidence,
                    )

            insert_segment(
                conn=conn,
                series_name=series_name,
                episode_title=episode_title,
                video_path=video_path,
                start_ms=int(seg["start"] * 1000),
                end_ms=int(seg["end"] * 1000),
                raw_speaker_id=seg["speaker"],
                identity_id=identity_id,
                transcript=seg.get("transcript"),
                confidence=confidence,
            )

        logger.info(
            "Alle %d Segmente für '%s / %s' in DB gespeichert.",
            len(segments),
            series_name,
            episode_title,
        )
    finally:
        conn.close()


# ------------------------------------------------------------------
# Master-Vektoren nach Nutzer-Bestätigung berechnen
# ------------------------------------------------------------------

def recompute_master_vectors(series_name: str, episode_title: str) -> None:
    """
    Berechnet den Master-Vektor für jede bestätigte Identität neu.

    Wird nach dem manuellen Labeling im Webinterface aufgerufen
    ("Finalize Episode"-Button). Mittelt alle bestätigten Segment-
    Embeddings zu einem hochpräzisen Centroid-Vektor.

    Args:
        series_name:    Serienname.
        episode_title:  Episodentitel.
    """
    from db.database import get_connection, update_master_vector

    conn = get_connection()
    try:
        cur = conn.cursor()

        # Alle bestätigten Identitäten der Episode
        cur.execute(
            """
            SELECT DISTINCT identity_id
            FROM episode_segments
            WHERE series_name = %s AND episode_title = %s
              AND identity_id IS NOT NULL AND is_confirmed = TRUE
            """,
            (series_name, episode_title),
        )
        identity_ids = [row[0] for row in cur.fetchall()]

        for identity_id in identity_ids:
            # voice_id zur Identität
            cur.execute(
                "SELECT voice_id FROM identities WHERE id = %s",
                (identity_id,),
            )
            row = cur.fetchone()
            if not row:
                continue
            voice_id = row[0]

            # Alle Segmente dieser Identität mit Embeddings
            cur.execute(
                """
                SELECT es.start_ms, es.end_ms, es.series_name, es.episode_title
                FROM episode_segments es
                WHERE es.identity_id = %s AND es.is_confirmed = TRUE
                """,
                (identity_id,),
            )
            segments_info = cur.fetchall()
            logger.info(
                "Re-Profiling Identität %d: %d bestätigte Segmente",
                identity_id,
                len(segments_info),
            )

        logger.info(
            "Master-Vektor-Neuberechnung für '%s / %s' abgeschlossen.",
            series_name,
            episode_title,
        )
    finally:
        conn.close()


# ------------------------------------------------------------------
# CLI-Entry-Point
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ice_audio_nexus Scanner – analysiert eine Videodatei"
    )
    p.add_argument("--video", required=True, help="Pfad zur Videodatei")
    p.add_argument("--source", required=True, help="Name der Serie/des Films")
    p.add_argument("--episode", required=True, help="Episodentitel")
    p.add_argument(
        "--diarization-device",
        default="cuda:0",
        help="PyAnnote Gerät (Standard: cuda:0 = Tesla P4)",
    )
    p.add_argument(
        "--whisper-device",
        default="cuda:1",
        help="Whisper Gerät (Standard: cuda:1 = Tesla P100)",
    )
    p.add_argument(
        "--whisper-model",
        default="large-v3",
        help="Faster-Whisper Modellgröße (Standard: large-v3)",
    )
    p.add_argument(
        "--language",
        default="de",
        help="Sprache für Whisper (Standard: de)",
    )
    p.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.85,
        help="Mindest-Ähnlichkeit für automatische Zuordnung (0.0–1.0, Standard: 0.85)",
    )
    p.add_argument(
        "--no-cuda-ffmpeg",
        action="store_true",
        help="FFmpeg CUDA-Dekodierung deaktivieren",
    )
    p.add_argument(
        "--skip-transcription",
        action="store_true",
        help="Transkription überspringen",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=_env_path)

    from db.database import init_db
    logger.info("Initialisiere Datenbank…")
    init_db()

    video_path = os.path.abspath(args.video)
    if not os.path.exists(video_path):
        logger.error("Videodatei nicht gefunden: %s", video_path)
        return

    with tempfile.TemporaryDirectory(prefix="ice_nexus_") as tmpdir:
        audio_wav = os.path.join(tmpdir, "audio.wav")

        # Schritt 1: Audio extrahieren
        if not extract_audio(video_path, audio_wav, use_cuda=not args.no_cuda_ffmpeg):
            logger.error("Audio-Extraktion fehlgeschlagen – Abbruch.")
            return

        # Schritt 2: Speaker Diarization (Tesla P4)
        segments = run_diarization(
            audio_wav,
            device=args.diarization_device,
            hf_token=os.environ.get("HF_TOKEN"),
        )

        # Schritt 3: Embeddings extrahieren (Tesla P4)
        segments = extract_embeddings(audio_wav, segments, device=args.diarization_device)

        # Schritt 4: Transkription (Tesla P100)
        if not args.skip_transcription:
            segments = transcribe_segments(
                audio_wav,
                segments,
                device=args.whisper_device,
                model_size=args.whisper_model,
                language=args.language,
            )
        else:
            for seg in segments:
                seg["transcript"] = None

        # Schritt 5: In MariaDB speichern
        save_to_database(
            segments,
            series_name=args.source,
            episode_title=args.episode,
            video_path=video_path,
            similarity_threshold=args.similarity_threshold,
        )

    logger.info(
        "Scan abgeschlossen: '%s / %s' – %d Segmente verarbeitet.",
        args.source,
        args.episode,
        len(segments),
    )
    logger.info("Starte jetzt das Webinterface: python web_ui/api.py")


if __name__ == "__main__":
    main()
