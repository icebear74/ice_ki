"""
ice_audio_nexus – processor/scanner.py
---------------------------------------
Analyses a video file using:
  • FFmpeg v8 (CUDA) for audio extraction
  • pyannote.audio for speaker diarization  (Tesla P4 / cuda:0)
  • faster-whisper for transcription         (Tesla P100 / cuda:1)

For each detected segment it searches MariaDB using VECTOR_DISTANCE against
ALL stored voice_samples and:
  - distance < MATCH_THRESHOLD   → confirmed match (identity assigned)
  - distance < SUGGEST_THRESHOLD → stores segment with is_suggestion=True
                                   (web UI will prompt the user)
  - otherwise                    → stored as unknown (speaker_label only)

Usage:
    python -m processor.scanner --video /path/to/episode.mkv \
                                 --series "Star Trek TNG" \
                                 --episode "The Inner Light"
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import tempfile

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)

# Thresholds (can be overridden via .env)
MATCH_THRESHOLD   = float(os.getenv("MATCH_THRESHOLD",   "0.25"))
SUGGEST_THRESHOLD = float(os.getenv("SUGGEST_THRESHOLD", "0.45"))

# GPU assignments
DIARIZATION_DEVICE   = os.getenv("DIARIZATION_DEVICE",   "cuda:0")
TRANSCRIPTION_DEVICE = os.getenv("TRANSCRIPTION_DEVICE", "cuda:1")


# ---------------------------------------------------------------------------
# Audio extraction via FFmpeg (CUDA)
# ---------------------------------------------------------------------------

def extract_audio(video_path: str, output_wav: str) -> None:
    """
    Extract a mono 16-kHz PCM WAV from the video using FFMPEG CUDA acceleration.
    16 kHz / mono is the standard format expected by pyannote.audio and Whisper.
    """
    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",           # CUDA hardware-accelerated decoding
        "-i", video_path,
        "-vn",                         # drop video stream
        "-acodec", "pcm_s16le",        # PCM 16-bit little-endian
        "-ar", "16000",                # 16 kHz
        "-ac", "1",                    # mono
        output_wav,
    ]
    logger.info("Extracting audio: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr}")
    logger.info("Audio extracted → %s", output_wav)


# ---------------------------------------------------------------------------
# Diarization
# ---------------------------------------------------------------------------

def run_diarization(audio_path: str) -> list[dict]:
    """
    Run pyannote.audio speaker diarization on *audio_path*.
    Returns a list of dicts: {start_ms, end_ms, speaker_label, embedding}
    """
    try:
        import torch
        from pyannote.audio import Pipeline
        from pyannote.audio.pipelines.utils.hook import ProgressHook
    except ImportError as exc:
        raise ImportError(
            "pyannote.audio is not installed. Run setup_env.sh first."
        ) from exc

    hf_token = os.getenv("HF_TOKEN")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    device = torch.device(DIARIZATION_DEVICE if torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device)
    logger.info("Diarization device: %s", device)

    with ProgressHook() as hook:
        diarization = pipeline(audio_path, hook=hook)

    # Extract per-segment embeddings using pyannote's SpeakerEmbedding model
    try:
        from pyannote.audio import Model, Inference
        emb_model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
        emb_model = emb_model.to(device)
        inference = Inference(emb_model, window="whole")
    except Exception:
        logger.warning("Could not load embedding model; embeddings will be empty.")
        inference = None

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_ms = int(turn.start * 1000)
        end_ms   = int(turn.end   * 1000)

        embedding: list[float] = []
        if inference is not None:
            try:
                import numpy as np
                from pyannote.core import Segment
                emb = inference.crop(audio_path, Segment(turn.start, turn.end))
                embedding = emb.flatten().tolist()
                # Pad or truncate to exactly 512 dimensions
                if len(embedding) < 512:
                    embedding = embedding + [0.0] * (512 - len(embedding))
                else:
                    embedding = embedding[:512]
            except Exception as exc:
                logger.debug("Embedding extraction failed for %s: %s", speaker, exc)

        segments.append({
            "start_ms":     start_ms,
            "end_ms":       end_ms,
            "speaker_label": speaker,
            "embedding":    embedding,
        })

    logger.info("Diarization complete: %d segments", len(segments))
    return segments


# ---------------------------------------------------------------------------
# Transcription (model cached per process to avoid repeated loading)
# ---------------------------------------------------------------------------

_whisper_model = None  # module-level cache


def _get_whisper_model(model_size: str = "large-v3"):
    """Return a cached WhisperModel instance (loaded once per process)."""
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is not installed. Run setup_env.sh first."
            ) from exc
        device = "cuda" if TRANSCRIPTION_DEVICE.startswith("cuda") else "cpu"
        _whisper_model = WhisperModel(model_size, device=device, compute_type="float16")
    return _whisper_model


def transcribe_segment(
    audio_path: str,
    start_s: float,
    end_s: float,
    model_size: str = "large-v3",
) -> str:
    """
    Transcribe a single audio segment using faster-whisper.
    Returns the detected text.
    """
    model = _get_whisper_model(model_size)

    # Use a context manager so the temp file is always cleaned up
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp_path = tmp.name
        # Write the sliced audio while the file handle is still open
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_s), "-to", str(end_s),
            "-i", audio_path,
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            tmp_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        whisper_segments, _ = model.transcribe(tmp_path, beam_size=5)
        return " ".join(seg.text.strip() for seg in whisper_segments)


# ---------------------------------------------------------------------------
# Main scan pipeline
# ---------------------------------------------------------------------------

def scan_video(
    video_path: str,
    series_name: str,
    episode_title: str,
    transcribe: bool = True,
) -> None:
    """
    Full pipeline:
      1. Extract audio
      2. Diarize speakers
      3. For each segment: look up in MariaDB (multi-vector VECTOR_DISTANCE)
      4. Store results in episode_segments
    """
    from db.database import (
        ensure_schema,
        get_connection,
        find_nearest_identity,
        upsert_segment,
    )

    ensure_schema()
    conn = get_connection()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    try:
        extract_audio(video_path, audio_path)
        segments = run_diarization(audio_path)

        for seg in segments:
            transcript = ""
            if transcribe and seg["embedding"]:
                try:
                    transcript = transcribe_segment(
                        audio_path,
                        seg["start_ms"] / 1000.0,
                        seg["end_ms"]   / 1000.0,
                    )
                except Exception as exc:
                    logger.warning("Transcription failed: %s", exc)

            # Multi-vector identity search
            match_result = {"status": "unknown", "identity_id": None,
                            "sample_id": None, "distance": None}
            if seg["embedding"]:
                match_result = find_nearest_identity(
                    conn,
                    seg["embedding"],
                    match_threshold=MATCH_THRESHOLD,
                    suggest_threshold=SUGGEST_THRESHOLD,
                )
                if match_result["status"] != "unknown":
                    logger.info(
                        "[%s–%s] %s → %s (dist=%.3f, status=%s, via sample %s '%s')",
                        seg["start_ms"], seg["end_ms"],
                        seg["speaker_label"],
                        match_result.get("identity_name"),
                        match_result.get("distance", 0),
                        match_result["status"],
                        match_result.get("sample_id"),
                        match_result.get("sample_context", ""),
                    )

            upsert_segment(
                conn,
                series_name=series_name,
                episode_title=episode_title,
                video_path=video_path,
                start_ms=seg["start_ms"],
                end_ms=seg["end_ms"],
                speaker_label=seg["speaker_label"],
                identity_id=match_result.get("identity_id"),
                matched_sample_id=match_result.get("sample_id"),
                match_distance=match_result.get("distance"),
                transcript=transcript,
                is_suggestion=(match_result["status"] == "suggest"),
            )

        logger.info("Scan complete – %d segments stored.", len(segments))
    finally:
        conn.close()
        if os.path.exists(audio_path):
            os.unlink(audio_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ice_audio_nexus scanner – diarize & identify speakers in a video"
    )
    parser.add_argument("--video",   required=True, help="Path to the video file")
    parser.add_argument("--series",  required=True, help="Series name (e.g. 'Star Trek TNG')")
    parser.add_argument("--episode", required=True, help="Episode title")
    parser.add_argument("--no-transcribe", action="store_true",
                        help="Skip Whisper transcription (faster)")
    args = parser.parse_args()

    scan_video(
        video_path=args.video,
        series_name=args.series,
        episode_title=args.episode,
        transcribe=not args.no_transcribe,
    )


if __name__ == "__main__":
    main()
