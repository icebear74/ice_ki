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

# GPU assignments (override in .env – see .env.example for all options)
# Default layout: P100 (cuda:0, 16 GB) handles the heavy work (diarization +
# transcription); P4 (cuda:1, 8 GB) runs DeepFilterNet only.
DIARIZATION_DEVICE   = os.getenv("DIARIZATION_DEVICE",   "cuda:0")
TRANSCRIPTION_DEVICE = os.getenv("TRANSCRIPTION_DEVICE", "cuda:0")

# Diarization tuning parameters (configurable via .env)
DIARIZATION_MIN_DURATION_ON  = float(os.getenv("DIARIZATION_MIN_DURATION_ON",  "0.3"))
DIARIZATION_MIN_DURATION_OFF = float(os.getenv("DIARIZATION_MIN_DURATION_OFF", "0.1"))
CLUSTERING_THRESHOLD         = float(os.getenv("CLUSTERING_THRESHOLD",         "0.7"))

# DeepFilterNet noise suppression
DEEPFILTER_ENABLED = os.getenv("DEEPFILTER_ENABLED", "true").lower() in ("1", "true", "yes")
DEEPFILTER_DEVICE  = os.getenv("DEEPFILTER_DEVICE", "cuda:1")  # P4 – freed up for noise suppression

# ---------------------------------------------------------------------------
# Configurable temporary file directories
# ---------------------------------------------------------------------------
# AUDIO_TMP_DIR – where extracted / DeepFilter-cleaned WAV files are written.
#                 Defaults to the OS temp directory (tempfile.gettempdir()).
# VIDEO_TMP_DIR – where web-preview MP4 transcodes are written.
#                 Defaults to the same directory as the source video file.
#                 Set to a fast scratch disk (e.g. /mnt/nvme/tmp) to avoid
#                 writing large MP4 files next to the originals.
_audio_tmp_env = os.getenv("AUDIO_TMP_DIR", "").strip()
_video_tmp_env = os.getenv("VIDEO_TMP_DIR", "").strip()

AUDIO_TMP_DIR: str | None = _audio_tmp_env if _audio_tmp_env else None   # None → system tmp
VIDEO_TMP_DIR: str | None = _video_tmp_env if _video_tmp_env else None   # None → beside source

if AUDIO_TMP_DIR:
    os.makedirs(AUDIO_TMP_DIR, exist_ok=True)
    logger.info("Audio temp dir: %s", AUDIO_TMP_DIR)
if VIDEO_TMP_DIR:
    os.makedirs(VIDEO_TMP_DIR, exist_ok=True)
    logger.info("Video temp dir: %s", VIDEO_TMP_DIR)


# ---------------------------------------------------------------------------
# Audio extraction via FFmpeg (CUDA)
# ---------------------------------------------------------------------------

def extract_audio(video_path: str, output_wav: str) -> None:
    """
    Extract a mono 48 kHz PCM WAV from the video using FFmpeg CUDA acceleration.

    48 kHz is DeepFilterNet's native rate, so ``apply_deepfilter`` can feed
    the file directly to ``load_audio`` without any intermediate resampling
    step.  When DeepFilterNet is disabled ``_fallback_to_16k`` downsamples
    from 48 kHz to 16 kHz before handing audio to Whisper / pyannote.
    """
    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",           # CUDA hardware-accelerated decoding
        "-i", video_path,
        "-vn",                         # drop video stream
        "-acodec", "pcm_s16le",        # PCM 16-bit little-endian
        "-ac", "1",                    # mono
        "-ar", "48000",                # 48 kHz – DeepFilterNet's native rate
        output_wav,
    ]
    logger.info("Extracting audio (48 kHz): %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr}")
    logger.info("Audio extracted → %s", output_wav)



# ---------------------------------------------------------------------------
# DeepFilterNet noise suppression (optional pre-processing step)
# ---------------------------------------------------------------------------

def apply_deepfilter(input_wav: str, output_wav: str) -> None:
    """
    Run DeepFilterNet 3 noise suppression on *input_wav* and write the
    cleaned audio to *output_wav* at 16 kHz (the standard rate for Whisper
    and pyannote.audio).

    *input_wav* is expected at **48 kHz** (DeepFilterNet's native rate), which
    is what ``extract_audio`` produces.

    Pipeline:

    1. ``load_audio`` reads the 48 kHz file as a CPU float32 tensor.
    2. ``enhance()`` receives the CPU tensor; DF moves data to the GPU
       internally (``df_state.analysis`` calls ``.numpy()`` and requires CPU
       input – passing a CUDA tensor here raises a RuntimeError).
    3. FFmpeg downsamples the enhanced audio (returned as a CPU tensor by DF)
       to 16 kHz for downstream consumers (Whisper / pyannote).

    Falls back silently to a raw 16-kHz copy if the library is missing or
    processing fails, so the rest of the pipeline is never blocked.
    """
    try:
        import torch
        from df.enhance import enhance, init_df, load_audio, save_audio
    except ImportError:
        logger.warning(
            "deepfilternet not installed – skipping noise suppression. "
            "Run: pip install deepfilternet"
        )
        _fallback_to_16k(input_wav, output_wav)
        return

    tmp_enhanced: str | None = None
    try:
        # ── Step 1: run DeepFilterNet on the 48 kHz WAV (no resampling) ───────
        device = torch.device(DEEPFILTER_DEVICE if torch.cuda.is_available() else "cpu")

        # init_df() calls model.to("cuda") internally and resolves "cuda" to
        # whichever device is currently set as default – which is cuda:0 unless
        # we tell PyTorch otherwise.  If we later do model.to(cuda:1) only the
        # model moves; the DF state object keeps its internal STFT buffers on
        # cuda:0, causing a cross-device error inside enhance().
        # Fix: set the default device *before* init_df() so the entire state
        # (model + df_state buffers) is initialised on the correct GPU.
        if device.type == "cuda":
            torch.cuda.set_device(device)

        model, df_state, _ = init_df()
        model = model.to(device)

        audio, sr = load_audio(input_wav, sr=df_state.sr())
        # Do NOT move audio to the GPU here.  DeepFilterNet's enhance()
        # calls df_state.analysis(audio.numpy()) internally, which requires
        # a CPU tensor.  DF manages GPU memory internally and returns a CPU
        # tensor from enhance().

        enhanced = enhance(model, df_state, audio)

        # Save enhanced audio at 48 kHz to a temp file.
        # enhance() returns a CPU tensor, so no .cpu() conversion needed.
        tmp_fd, tmp_enhanced = tempfile.mkstemp(suffix=".enhanced.wav", dir=AUDIO_TMP_DIR)
        os.close(tmp_fd)
        save_audio(tmp_enhanced, enhanced, sr)

        # Release GPU memory so the diarization pipeline (which runs on the
        # same device right after) finds the VRAM free.
        del enhanced, audio, model, df_state
        torch.cuda.empty_cache()

        # ── Step 2: downsample to 16 kHz for Whisper / pyannote ──────────────
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_enhanced,
                "-ar", "16000", "-ac", "1",
                "-acodec", "pcm_s16le",
                output_wav,
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg downsample to 16 kHz failed:\n{result.stderr}")

        logger.info(
            "DeepFilterNet applied on GPU (48kHz→16kHz): %s → %s",
            input_wav, output_wav,
        )
    except Exception as exc:
        logger.warning("DeepFilterNet processing failed (non-fatal): %s", exc)
        _fallback_to_16k(input_wav, output_wav)
    finally:
        if tmp_enhanced and os.path.exists(tmp_enhanced):
            try:
                os.unlink(tmp_enhanced)
            except OSError:
                pass


def _fallback_to_16k(input_wav: str, output_wav: str) -> None:
    """Downsample *input_wav* to 16 kHz and write to *output_wav* via FFmpeg."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", input_wav,
            "-ar", "16000", "-ac", "1",
            "-acodec", "pcm_s16le",
            output_wav,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        # Last resort: plain copy (may be wrong sample rate but unblocks pipeline)
        import shutil
        shutil.copy2(input_wav, output_wav)


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
            f"pyannote.audio dependency missing ({exc}). Run setup_env.sh first."
        ) from exc

    hf_token = os.getenv("HF_TOKEN")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    device = torch.device(DIARIZATION_DEVICE if torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device)
    logger.info("Diarization device: %s", device)

    # Apply configurable tuning parameters when the pipeline exposes them
    try:
        params = pipeline.parameters(instantiated=True)
        if hasattr(params, "clustering") and hasattr(params.clustering, "threshold"):
            params.clustering.threshold = CLUSTERING_THRESHOLD
        if hasattr(params, "segmentation") and hasattr(params.segmentation, "min_duration_on"):
            params.segmentation.min_duration_on  = DIARIZATION_MIN_DURATION_ON
            params.segmentation.min_duration_off = DIARIZATION_MIN_DURATION_OFF
    except Exception as _p:
        logger.debug("Could not set diarization tuning params: %s", _p)

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
    # Create a temp file in the configured audio-tmp directory, close it,
    # let FFmpeg write to it, then clean it up in finally.
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", dir=AUDIO_TMP_DIR)
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
    update_mode: bool = True,
) -> None:
    """
    Full pipeline:
      1. Extract native-rate mono WAV from video (FFmpeg CUDA, no forced downsampling)
      2. Parallel pre-processing phase (both GPUs active simultaneously):
           • P4  (cuda:0): DeepFilterNet noise suppression (if DEEPFILTER_ENABLED)
                           native rate → upsample to 48 kHz → DeepFilter GPU →
                           downsample to 16 kHz
           • P100 (cuda:1): Whisper model pre-load
      3. Load cleaned 16-kHz audio into memory
      4. Diarize speakers (P4 – background thread / producer)
      5. For each segment: transcribe (P100 – main thread / consumer)
                           look up in MariaDB (multi-vector VECTOR_DISTANCE)
      6. Store / update results in episode_segments

    Update mode (update_mode=True):
      - If a segment already exists at the same timecode it is UPDATED, not
        duplicated.
      - Segments that were manually assigned by the user are preserved;
        auto-detected identity assignments are refreshed using the latest
        supervectors.

    GPU utilisation:
      P4  (cuda:0): DeepFilter → Diarization (sequential on same device)
      P100 (cuda:1): Whisper pre-load runs concurrently with DeepFilter,
                     then transcribes segments while P4 diarizes.
    """
    from db.database import (
        ensure_schema,
        get_connection,
        find_nearest_identity,
        upsert_segment,
        get_existing_segment,
        update_segment_match,
        vector_to_bytes,
    )

    ensure_schema()

    # Derive the web-preview path: configured VIDEO_TMP_DIR or same directory
    # as the source video (original behaviour).
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    if VIDEO_TMP_DIR:
        web_preview_path = os.path.join(VIDEO_TMP_DIR, video_stem + ".web.mp4")
    else:
        web_preview_path = os.path.splitext(video_path)[0] + ".web.mp4"

    with tempfile.NamedTemporaryFile(suffix=".wav", dir=AUDIO_TMP_DIR, delete=False) as tmp:
        audio_path = tmp.name

    # DeepFilterNet output goes to a separate temp file so the original raw WAV
    # is preserved and the cleaned copy can be cleaned up independently.
    clean_audio_path = audio_path + ".clean.wav"

    try:
        # ── Step 1: Extract audio ──────────────────────────────────────────────
        extract_audio(video_path, audio_path)

        # Generate the browser-ready preview in the background so it is ready
        # by the time the analysis is finished.  Non-fatal if it fails.
        if not os.path.exists(web_preview_path):
            try:
                transcode_web_preview(video_path, web_preview_path)
            except Exception as exc:
                logger.warning("Web preview transcode failed (non-fatal): %s", exc)

        # ── Step 2: Parallel pre-processing (P4 + P100 simultaneously) ────────
        #
        #   Thread A  – P4  (cuda:1): DeepFilterNet noise suppression
        #   Thread B  – P100 (cuda:0): Whisper model pre-load
        #
        # Because DeepFilter runs on the P4 and Whisper loads on the P100 they
        # use *different* GPUs: no VRAM conflict.  DeepFilterNet explicitly
        # releases all GPU tensors (del + empty_cache) before returning, so the
        # P4 is fully free for diarization which runs next on the same device.

        deepfilter_exc:   list[Exception] = []
        whisper_preload_exc: list[Exception] = []

        def _deepfilter_worker() -> None:
            if DEEPFILTER_ENABLED:
                logger.info("DeepFilterNet: starting noise suppression on %s", DEEPFILTER_DEVICE)
                try:
                    apply_deepfilter(audio_path, clean_audio_path)
                except Exception as exc:
                    logger.error("DeepFilterNet failed: %s", exc)
                    deepfilter_exc.append(exc)
            else:
                logger.warning(
                    "DeepFilterNet disabled (DEEPFILTER_ENABLED=false) – "
                    "downsampling raw audio to 16 kHz for pipeline"
                )
                # audio_path is at native sample rate; Whisper and pyannote
                # need 16 kHz, so downsample here the same way apply_deepfilter
                # would in its final step.
                _fallback_to_16k(audio_path, clean_audio_path)

        def _whisper_preload_worker() -> None:
            if transcribe:
                try:
                    _get_whisper_model(model_size)
                    logger.info("Whisper model pre-loaded on %s", TRANSCRIPTION_DEVICE)
                except Exception as exc:
                    logger.warning("Whisper pre-load failed (will retry per segment): %s", exc)
                    whisper_preload_exc.append(exc)

        t_deepfilter = threading.Thread(
            target=_deepfilter_worker, daemon=True, name="deepfilter-P4"
        )
        t_whisper = threading.Thread(
            target=_whisper_preload_worker, daemon=True, name="whisper-preload-P100"
        )
        logger.info(
            "Starting parallel pre-processing: DeepFilter=%s (%s) + Whisper pre-load (P100)",
            "ON" if DEEPFILTER_ENABLED else "OFF",
            DEEPFILTER_DEVICE,
        )
        t_deepfilter.start()
        t_whisper.start()
        t_deepfilter.join()
        t_whisper.join()
        logger.info("Parallel pre-processing complete.")

        # ── Step 3: Load cleaned audio into memory ─────────────────────────────
        audio_for_pipeline = clean_audio_path  # always exists (copy if DeepFilter off)
        audio_data: np.ndarray | None = None
        sample_rate: int = 16000
        try:
            import soundfile as sf
            audio_data, sample_rate = sf.read(
                audio_for_pipeline, dtype="float32", always_2d=False
            )
            logger.info(
                "Episode audio loaded into memory: %d samples @ %d Hz (%.1f s)",
                len(audio_data), sample_rate, len(audio_data) / sample_rate,
            )
        except Exception as exc:
            logger.warning(
                "soundfile load failed – will fall back to per-segment FFmpeg: %s", exc
            )

        # ── Steps 4-6: Diarize (P4 producer) + Transcribe/Store (P100 consumer) ──
        #
        # 64-slot queue: large enough that the producer is never blocked yet
        # small enough to bound memory for very long episodes.

        _SEGMENT_QUEUE_SIZE = 64
        seg_queue: queue.Queue = queue.Queue(maxsize=_SEGMENT_QUEUE_SIZE)
        producer_exc: list[Exception] = []

        def _diarize_producer() -> None:
            try:
                for seg in _iter_diarization_segments(audio_for_pipeline):
                    seg_queue.put(seg)
            except Exception as exc:  # noqa: BLE001
                logger.error("Diarization producer failed: %s", exc)
                producer_exc.append(exc)
            finally:
                seg_queue.put(None)  # sentinel – always sent, even on error

        producer = threading.Thread(
            target=_diarize_producer, daemon=True, name="diarize-P4"
        )
        producer.start()

        # Consumer: transcription on P100 + DB writes (main thread)
        conn = get_connection()
        segments_stored = 0
        segments_updated = 0
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
                            audio_data if audio_data is not None else audio_for_pipeline,
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

                # Multi-vector identity search against latest supervectors
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

                emb_bytes = vector_to_bytes(seg["embedding"]) if seg["embedding"] else None

                # ── Update mode: check for an existing segment at this timecode ──
                existing = None
                if update_mode:
                    existing = get_existing_segment(
                        conn, series_name, episode_title,
                        seg["start_ms"], seg["end_ms"],
                    )

                if existing is not None:
                    # Preserve any manual identity assignment the user made; only
                    # refresh auto-detected (non-manual) assignments.
                    # A segment is considered manually assigned when is_suggestion
                    # was FALSE and identity_id was set – meaning the user confirmed
                    # or assigned it via the Web UI.
                    has_manual = (
                        existing["identity_id"] is not None
                        and not existing["is_suggestion"]
                    )
                    if not has_manual:
                        # No manual assignment – safe to overwrite with fresh match
                        update_segment_match(
                            conn,
                            segment_id=existing["id"],
                            identity_id=match_result.get("identity_id"),
                            matched_sample_id=match_result.get("sample_id"),
                            match_distance=match_result.get("distance"),
                            is_suggestion=(match_result["status"] == "suggest"),
                            embedding=emb_bytes,
                            transcript=transcript,
                            is_low_quality=is_low_quality,
                        )
                        segments_updated += 1
                        logger.debug(
                            "[%s–%s] updated existing segment %s",
                            seg["start_ms"], seg["end_ms"], existing["id"],
                        )
                    else:
                        logger.debug(
                            "[%s–%s] segment %s has manual assignment – skipped",
                            seg["start_ms"], seg["end_ms"], existing["id"],
                        )
                else:
                    upsert_segment(
                        conn,
                        series_name=series_name,
                        episode_title=episode_title,
                        video_path=video_path,
                        start_ms=seg["start_ms"],
                        end_ms=seg["end_ms"],
                        speaker_label=seg["speaker_label"],
                        embedding=emb_bytes,
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

        logger.info(
            "Scan complete – %d new segments stored, %d existing segments updated.",
            segments_stored, segments_updated,
        )
    finally:
        for path in (audio_path, clean_audio_path):
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass


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
    parser.add_argument("--no-update-mode", action="store_true",
                        help="Disable update mode (always insert new rows, never deduplicate)")
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
        update_mode=not args.no_update_mode,
    )


if __name__ == "__main__":
    main()
