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

Usage (Series – episode auto-detected from filename):
    python -m processor.scanner --video /path/to/S01E01-Episode.mkv \
                                 --series "Star Trek TNG"

Usage (Series – episode provided explicitly):
    python -m processor.scanner --video /path/to/episode.mkv \
                                 --series "Star Trek TNG" \
                                 --episode "The Inner Light"

Usage (Movie – no SxxExx pattern in filename):
    python -m processor.scanner --video /path/to/X-Men.mkv \
                                 --series "X-Men"
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import tempfile

from dotenv import load_dotenv

# Load .env from the project root (parent of the processor package), so the
# script works regardless of the current working directory.
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)

# Thresholds (can be overridden via .env)
MATCH_THRESHOLD   = float(os.getenv("MATCH_THRESHOLD",   "0.25"))
SUGGEST_THRESHOLD = float(os.getenv("SUGGEST_THRESHOLD", "0.45"))
# Minimum cosine-distance gap required between the best match and the closest
# sample from a *different* identity.  Raises this bar prevents two similar-
# sounding characters (e.g. Sheldon vs Leonard) from collapsing onto the same
# identity.  Increase if you still see false merges; decrease if valid matches
# get downgraded to "suggest".
MIN_MARGIN        = float(os.getenv("MIN_MARGIN",         "0.07"))

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


def transcode_web_preview(video_path: str, output_mp4: str) -> None:
    """
    Transcode *video_path* to a browser-ready H.264/AAC MP4 file stored at
    *output_mp4*.  The file is written with -movflags +faststart so the moov
    atom sits at the front – this lets the browser seek freely without
    downloading the whole file first.

    Output is capped at 480p (width scaled to keep aspect ratio) so the file
    stays small.  Audio is kept as stereo AAC 128k so voice sync works
    correctly in the Web UI.
    """
    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",                   # GPU-accelerated decoding
        "-i", video_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "28",                          # slightly lower quality = smaller file
        "-profile:v", "baseline", "-level", "3.1",
        "-vf", "scale=-2:480",                 # scale to 480p, keep aspect ratio
        "-c:a", "aac", "-b:a", "128k", "-ac", "2",
        "-movflags", "+faststart",             # moov atom at front → seekable
        output_mp4,
    ]
    logger.info("Transcoding web preview: %s → %s", video_path, output_mp4)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg web-preview transcode failed:\n{result.stderr}")
    logger.info("Web preview ready → %s", output_mp4)


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
        emb_model = Model.from_pretrained("pyannote/embedding", token=hf_token)
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
        # CTranslate2 requires tensor-core FP16 support (Volta+, compute ≥7.0).
        # Older CUDA GPUs like Tesla P4/P100 (Pascal, compute 6.x) only support
        # float32 efficiently.  CPU falls back to int8.
        if device == "cuda":
            compute_type_candidates = ["float16", "float32"]
        else:
            compute_type_candidates = ["int8"]
        last_exc: Exception | None = None
        for compute_type in compute_type_candidates:
            try:
                _whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
                logger.info("Whisper loaded with compute_type=%s on %s", compute_type, device)
                break
            except (ValueError, RuntimeError) as exc:
                logger.warning(
                    "Whisper compute_type=%s not supported on %s, trying next: %s",
                    compute_type, device, exc,
                )
                last_exc = exc
        else:
            raise RuntimeError(
                f"Could not load Whisper model on {device}: {last_exc}"
            ) from last_exc
    return _whisper_model


def transcribe_segment(
    audio_path: str,
    start_s: float,
    end_s: float,
    model_size: str = "large-v3",
) -> tuple[str, float]:
    """
    Transcribe a single audio segment using faster-whisper.
    Returns a tuple of (detected_text, max_no_speech_prob).
    Language is fixed to German to avoid misidentification during non-speech
    segments (laughter, noise, music) and to prevent hallucinations.
    """
    model = _get_whisper_model(model_size)

    # Create a temp file, close it, let FFmpeg write to it, clean up in finally
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    try:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_s), "-to", str(end_s),
            "-i", audio_path,
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            tmp_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        whisper_segments, _ = model.transcribe(
            tmp_path,
            beam_size=5,
            language="de",
            task="transcribe",
        )
        # Materialise the generator so we can iterate twice (text + no_speech_prob)
        seg_list = list(whisper_segments)
        text = " ".join(seg.text.strip() for seg in seg_list)
        # Use the highest no_speech_prob across all sub-segments as the quality
        # indicator; default to 0.0 when Whisper provides no segments at all.
        no_speech_prob = max((seg.no_speech_prob for seg in seg_list), default=0.0)
        return text, no_speech_prob
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


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
        vector_to_bytes,
    )

    ensure_schema()
    conn = get_connection()

    # Derive the web-preview path: same directory, same stem, suffix .web.mp4
    web_preview_path = os.path.splitext(video_path)[0] + ".web.mp4"

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    try:
        extract_audio(video_path, audio_path)

        # Generate the browser-ready preview alongside the WAV extraction so
        # the Web UI can serve it directly (seekable, audio + video).
        if not os.path.exists(web_preview_path):
            try:
                transcode_web_preview(video_path, web_preview_path)
            except Exception as exc:
                logger.warning("Web preview transcode failed (non-fatal): %s", exc)

        segments = run_diarization(audio_path)

        for seg in segments:
            transcript = ""
            no_speech_prob = 0.0
            if transcribe and seg["embedding"]:
                try:
                    transcript, no_speech_prob = transcribe_segment(
                        audio_path,
                        seg["start_ms"] / 1000.0,
                        seg["end_ms"]   / 1000.0,
                    )
                except Exception as exc:
                    logger.warning("Transcription failed: %s", exc)

            # Quality heuristic: flag segments likely to contain laughter,
            # noise, or too little speech for a reliable voice embedding.
            duration_s = (seg["end_ms"] - seg["start_ms"]) / 1000.0
            is_low_quality = (
                no_speech_prob > 0.45
                or duration_s < 1.2
                or len(transcript.strip()) < 5
            )
            if is_low_quality:
                logger.info(
                    "[%s–%s] %s → marked is_low_quality "
                    "(no_speech_prob=%.2f, duration=%.2fs, text_len=%d)",
                    seg["start_ms"], seg["end_ms"], seg["speaker_label"],
                    no_speech_prob, duration_s, len(transcript.strip()),
                )

            # Multi-vector identity search
            match_result = {"status": "unknown", "identity_id": None,
                            "sample_id": None, "distance": None}
            if seg["embedding"]:
                match_result = find_nearest_identity(
                    conn,
                    seg["embedding"],
                    match_threshold=MATCH_THRESHOLD,
                    suggest_threshold=SUGGEST_THRESHOLD,
                    min_margin=MIN_MARGIN,
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
                embedding=vector_to_bytes(seg["embedding"]) if seg["embedding"] else None,
                identity_id=match_result.get("identity_id"),
                matched_sample_id=match_result.get("sample_id"),
                match_distance=match_result.get("distance"),
                transcript=transcript,
                is_suggestion=(match_result["status"] == "suggest"),
                is_low_quality=is_low_quality,
            )

        logger.info("Scan complete – %d segments stored.", len(segments))
    finally:
        conn.close()
        if os.path.exists(audio_path):
            os.unlink(audio_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_filename(filename: str) -> str:
    """
    Extract episode code from *filename*.

    Recognises patterns like S01E01, s01e01, S1E1, etc.
    Returns a normalised string such as "S01E01", or "Movie" if no pattern
    is found (indicating a film rather than a series episode).
    """
    match = re.search(r"[Ss](\d+)[Ee](\d+)", filename)
    if match:
        return f"S{int(match.group(1)):02d}E{int(match.group(2)):02d}"
    return "Movie"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ice_audio_nexus scanner – diarize & identify speakers in a video"
    )
    parser.add_argument("--video",   required=True, help="Path to the video file")
    parser.add_argument("--series",  required=True, help="Series or movie title (e.g. 'Star Trek TNG')")
    parser.add_argument("--episode", default=None,
                        help="Episode title or code (optional – auto-detected from filename; "
                             "defaults to 'Movie' if no SxxExx pattern is found)")
    parser.add_argument("--no-transcribe", action="store_true",
                        help="Skip Whisper transcription (faster)")
    args = parser.parse_args()

    # Auto-detect episode from filename when not provided explicitly
    episode_title = args.episode or parse_filename(os.path.basename(args.video))
    logger.info("Series: %s | Episode: %s", args.series, episode_title)

    scan_video(
        video_path=args.video,
        series_name=args.series,
        episode_title=episode_title,
        transcribe=not args.no_transcribe,
    )


if __name__ == "__main__":
    main()
