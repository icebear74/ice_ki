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
import queue
import re
import subprocess
import tempfile
import threading

import numpy as np

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
    # Try GPU-accelerated NVENC first; fall back to CPU libx264 if unavailable.
    _codec_variants = [
        ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "28"],
        ["-c:v", "libx264",    "-preset", "fast", "-crf", "28"],
    ]
    last_stderr = ""
    for codec_args in _codec_variants:
        cmd = [
            "ffmpeg", "-y",
            "-hwaccel", "cuda",                   # GPU-accelerated decoding
            "-i", video_path,
            *codec_args,
            "-profile:v", "baseline", "-level", "3.1",
            "-vf", "scale=-2:480",                 # scale to 480p, keep aspect ratio
            "-c:a", "aac", "-b:a", "128k", "-ac", "2",
            "-movflags", "+faststart",             # moov atom at front → seekable
            output_mp4,
        ]
        logger.info(
            "Transcoding web preview (%s): %s → %s", codec_args[1], video_path, output_mp4
        )
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("Web preview ready → %s", output_mp4)
            return
        last_stderr = result.stderr
        logger.warning(
            "Codec %s failed (will try fallback): %s",
            codec_args[1],
            last_stderr.splitlines()[-5:],  # last 5 lines avoids mid-char truncation
        )
    raise RuntimeError(
        f"FFmpeg web-preview transcode failed for all codec variants:\n{last_stderr}"
    )


# ---------------------------------------------------------------------------
# Diarization
# ---------------------------------------------------------------------------

def _iter_diarization_segments(audio_path: str):
    """
    Internal generator.  Runs the full diarization pipeline on *audio_path*,
    then yields one segment dict per speaker turn as embeddings are extracted.

    Per-segment speaker embeddings are computed via ``Inference.crop()`` with
    ``window="whole"`` (same quality as the original implementation).  Because
    segments are yielded one-by-one, a consumer thread can start transcribing
    while embedding extraction for later segments is still in progress on the
    diarization GPU (cuda:0 / P4).

    Yields dicts: {start_ms, end_ms, speaker_label, embedding}
    """
    try:
        import torch
        from pyannote.audio import Pipeline, Model, Inference
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        from pyannote.core import Segment as _PyannoteSegment
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

    # Load embedding model once; all per-segment crops share the same instance.
    inference = None
    try:
        emb_model = Model.from_pretrained("pyannote/embedding", token=hf_token)
        emb_model = emb_model.to(device)
        inference = Inference(emb_model, window="whole")
    except Exception:
        logger.warning("Could not load embedding model; embeddings will be empty.")

    count = 0
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_ms = int(turn.start * 1000)
        end_ms   = int(turn.end   * 1000)

        embedding: list[float] = []
        if inference is not None:
            try:
                emb = inference.crop(audio_path, _PyannoteSegment(turn.start, turn.end))
                embedding = emb.flatten().tolist()
                # Pad or truncate to exactly 512 dimensions
                if len(embedding) < 512:
                    embedding = embedding + [0.0] * (512 - len(embedding))
                else:
                    embedding = embedding[:512]
            except Exception as exc:
                logger.debug("Embedding extraction failed for %s: %s", speaker, exc)

        count += 1
        yield {
            "start_ms":      start_ms,
            "end_ms":        end_ms,
            "speaker_label": speaker,
            "embedding":     embedding,
        }

    logger.info("Diarization complete: %d segments", count)


def run_diarization(audio_path: str) -> list[dict]:
    """
    Run pyannote.audio speaker diarization on *audio_path*.
    Returns a list of dicts: {start_ms, end_ms, speaker_label, embedding}
    """
    return list(_iter_diarization_segments(audio_path))


# ---------------------------------------------------------------------------
# Transcription (model cached per process to avoid repeated loading)
# ---------------------------------------------------------------------------

_whisper_model = None  # module-level cache


def _get_whisper_model(model_size: str = "large-v3-turbo"):
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
    audio_source: "str | np.ndarray",
    start_s: float,
    end_s: float,
    sample_rate: int = 16000,
    model_size: str = "large-v3-turbo",
) -> tuple[str, float]:
    """
    Transcribe a single audio segment using faster-whisper.
    Returns a tuple of (detected_text, max_no_speech_prob).

    *audio_source* may be:
      - a ``np.ndarray`` (float32, 16 kHz) covering the *whole episode* – the
        segment [start_s, end_s] is sliced in-memory (no subprocess), or
      - a ``str`` file path – a temporary WAV is extracted via FFmpeg (legacy
        fallback, used when the in-memory array is unavailable).

    Language is fixed to German to avoid misidentification during non-speech
    segments (laughter, noise, music) and to prevent hallucinations.
    """
    model = _get_whisper_model(model_size)

    if isinstance(audio_source, np.ndarray):
        # Fast path: slice episode array in-memory – no FFmpeg subprocess needed.
        start_sample = int(start_s * sample_rate)
        end_sample   = int(end_s   * sample_rate)
        audio_chunk  = audio_source[start_sample:end_sample]
        whisper_segments, _ = model.transcribe(
            audio_chunk,
            beam_size=5,
            language="de",
            task="transcribe",
        )
        seg_list = list(whisper_segments)
        text = " ".join(seg.text.strip() for seg in seg_list)
        no_speech_prob = max((seg.no_speech_prob for seg in seg_list), default=0.0)
        return text, no_speech_prob

    # Legacy path: file path – use FFmpeg to carve out the segment.
    # Create a temp file, close it, let FFmpeg write to it, clean up in finally
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    try:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_s), "-to", str(end_s),
            "-i", audio_source,
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
    model_size: str = "large-v3-turbo",
) -> None:
    """
    Full pipeline:
      1. Extract audio
      2. Diarize speakers  (Tesla P4  / cuda:0 – background thread)
      3. For each segment: transcribe (Tesla P100 / cuda:1 – main thread)
                           look up in MariaDB (multi-vector VECTOR_DISTANCE)
      4. Store results in episode_segments

    Diarization and transcription run concurrently via a producer/consumer
    queue: the P4 extracts speaker embeddings while the P100 simultaneously
    transcribes previously diarized segments, maximising GPU utilisation.
    """
    from db.database import (
        ensure_schema,
        get_connection,
        find_nearest_identity,
        upsert_segment,
        vector_to_bytes,
    )

    ensure_schema()

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

        # Pre-load Whisper on cuda:1 *before* diarization starts so the P100
        # is immediately ready when the first segment arrives in the queue.
        if transcribe:
            try:
                _get_whisper_model(model_size)
                logger.info("Whisper model pre-loaded on %s", TRANSCRIPTION_DEVICE)
            except Exception as exc:
                logger.warning(
                    "Whisper pre-load failed (will retry on first segment): %s", exc
                )

        # Load the full episode audio into a float32 numpy array once so each
        # segment can be sliced in-memory without spawning an FFmpeg process.
        audio_data: np.ndarray | None = None
        sample_rate: int = 16000  # default; overwritten by soundfile below
        try:
            import soundfile as sf
            audio_data, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
            logger.info(
                "Episode audio loaded into memory: %d samples @ %d Hz (%.1f s)",
                len(audio_data), sample_rate, len(audio_data) / sample_rate,
            )
        except Exception as exc:
            logger.warning(
                "soundfile load failed – will fall back to per-segment FFmpeg: %s", exc
            )

        # -----------------------------------------------------------------------
        # Producer / consumer: diarization on P4 feeds a queue; transcription
        # on P100 drains it concurrently.
        # 64 slots: large enough that the producer is never blocked waiting for
        # the consumer yet small enough to bound memory for very long episodes.
        # -----------------------------------------------------------------------
        _SEGMENT_QUEUE_SIZE = 64
        seg_queue: queue.Queue = queue.Queue(maxsize=_SEGMENT_QUEUE_SIZE)
        producer_exc: list[Exception] = []  # captures any exception from the thread

        def _diarize_producer() -> None:
            try:
                for seg in _iter_diarization_segments(audio_path):
                    seg_queue.put(seg)
            except Exception as exc:  # noqa: BLE001 – daemon thread, no caller to propagate to
                logger.error("Diarization producer failed: %s", exc)
                producer_exc.append(exc)
            finally:
                seg_queue.put(None)  # sentinel – always sent, even on error

        producer = threading.Thread(
            target=_diarize_producer, daemon=True, name="diarize-P4"
        )
        producer.start()

        # Consumer loop (main thread – transcription on P100 + DB writes)
        conn = get_connection()
        segments_stored = 0
        try:
            while True:
                seg = seg_queue.get()
                if seg is None:
                    break

                transcript = ""
                no_speech_prob = 0.0
                if transcribe and seg["embedding"]:
                    try:
                        transcript, no_speech_prob = transcribe_segment(
                            audio_data if audio_data is not None else audio_path,
                            seg["start_ms"] / 1000.0,
                            seg["end_ms"]   / 1000.0,
                            sample_rate=sample_rate,
                            model_size=model_size,
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
                segments_stored += 1
        finally:
            conn.close()

        producer.join()
        if producer_exc:
            logger.error("Diarization failed: %s", producer_exc[0])
            raise producer_exc[0]

        logger.info("Scan complete – %d segments stored.", segments_stored)
    finally:
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
    parser.add_argument("--model", default="large-v3-turbo",
                        help="Whisper model size (default: large-v3-turbo)")
    args = parser.parse_args()

    # Auto-detect episode from filename when not provided explicitly
    episode_title = args.episode or parse_filename(os.path.basename(args.video))
    logger.info("Series: %s | Episode: %s", args.series, episode_title)

    scan_video(
        video_path=args.video,
        series_name=args.series,
        episode_title=episode_title,
        transcribe=not args.no_transcribe,
        model_size=args.model,
    )


if __name__ == "__main__":
    main()
