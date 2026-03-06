#!/usr/bin/env python3
"""
Streaming video extractor for dataset_generator_v2.

Replaces the previous per-timestamp ``-ss`` seek approach with a single
FFmpeg pass that streams the video linearly.  A rolling frame buffer
(default 7 frames) is maintained in memory; patches are written to disk
as their centre frame enters the buffer.

Public API
----------
build_frame_assignments_distributed()
    Convert scene timestamps + format distribution into a sorted list of
    (center_frame_idx, category, format_name) assignments with interleaved
    distribution (all format directories receive patches from the very first
    scenes processed).

build_frame_ranges_from_assignments()
    Merge the per-assignment frame windows into the minimal set of
    contiguous ranges needed.  Useful for logging / optimisation analysis.

extract_and_save_streaming_distributed()
    Main entry point.  Launches one FFmpeg process, streams BGR24 frames,
    saves patches on-the-fly.

create_patch_pair()
    Create a (GT, LR) patch pair from a sequence of frames.

save_patch_pair()
    Persist a (GT, LR) pair to the correct output directories.
"""

import os
import random
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Import path setup – streaming_extractor lives inside dataset_generator_v2/
# ---------------------------------------------------------------------------
import sys as _sys
_sys.path.insert(0, os.path.dirname(__file__))
from utils.format_definitions import get_output_dirs_for_format

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Output resolution after HDR→SDR tonemap (must match what OpenCV expects)
STREAM_WIDTH: int = 1920
STREAM_HEIGHT: int = 1080

# ---------------------------------------------------------------------------
# Filter chains — two families: HDR→SDR tonemap and plain SDR pass-through.
#
# HDR chains handle HDR10 (SMPTE 2084 / BT.2020) and Dolby Vision P5/P8.
# SDR chains are used when the video is already standard-dynamic-range
# (BT.709 transfer) and no tone-mapping is needed.  Applying the HDR
# tonemap chain to an SDR source causes overexposure because zscale would
# linearise the already-gamma-encoded values a second time.
#
# The correct chain is selected per-video by build_vf_filter() based on the
# is_hdr flag returned by _get_video_metadata() / is_hdr_transfer().
# ---------------------------------------------------------------------------

# HDR→SDR: Software (CPU-only) fallback.
# zscale reads tin from stream metadata → works for smpte2084, hlg, bt709.
# range=full: unambiguous 0-255 output for OpenCV.
_TONEMAP_FILTER: str = (
    "zscale=t=linear:npl=100,"
    "format=gbrpf32le,"
    "zscale=p=bt709,"
    "tonemap=tonemap=mobius:desat=0,"
    "zscale=t=bt709:m=bt709:range=full,"
    f"scale={STREAM_WIDTH}:{STREAM_HEIGHT}:flags=lanczos,"
    "format=bgr24"
)

# SDR pass-through: Software (CPU-only).
# No linearisation or tonemap needed — just scale + convert to BGR24.
_SDR_FILTER: str = (
    f"scale={STREAM_WIDTH}:{STREAM_HEIGHT}:flags=lanczos,"
    "format=bgr24"
)

# HDR→SDR: Hybrid GPU/CPU — scale_cuda downscales on GPU, tonemap on CPU.
# hwdownload + format=p010 preserves 10-bit precision; the multi-step
# zscale+tonemap chain is identical to _TONEMAP_FILTER.
# Use together with -init_hw_device cuda=hw -hwaccel cuda
#                  -hwaccel_output_format cuda.
_TONEMAP_FILTER_SCALE_CUDA: str = (
    f"scale_cuda={STREAM_WIDTH}:{STREAM_HEIGHT},"
    "hwdownload,"
    "format=p010,"
    "zscale=t=linear:npl=100,"
    "format=gbrpf32le,"
    "zscale=p=bt709,"
    "tonemap=tonemap=mobius:desat=0,"
    "zscale=t=bt709:m=bt709:range=full,"
    "format=bgr24"
)

# SDR pass-through: Hybrid GPU/CPU — scale on GPU, convert on CPU.
_SDR_FILTER_SCALE_CUDA: str = (
    f"scale_cuda={STREAM_WIDTH}:{STREAM_HEIGHT},"
    "hwdownload,"
    "format=bgr24"
)

# Full-GPU HDR→SDR tonemap filter chain.
# Requires FFmpeg built with --enable-cuda-nvcc / libnpp so that both
# tonemap_cuda and scale_cuda are available.
# Frames stay in GPU memory from decode through tonemap + scale;
# hwdownload copies only the final 1920×1080 result to CPU.
# Use together with -hwaccel cuda -hwaccel_output_format cuda.
# Notes:
#   - interp_algo=bicubic — see _TONEMAP_FILTER_SCALE_CUDA comment above.
#   - tonemap_cuda outputs 8-bit NV12 CUDA frames; scale_cuda receives NV12
#     and outputs NV12.
#   - hwdownload (bare) + scale=iw:ih: same reasoning as above — scale breaks
#     the backward format negotiation, converting NV12→YUV420P in software.
#   - format=yuv420p: ensures planar yuv420p so the final format=bgr24
#     libswscale conversion is unambiguous.
_TONEMAP_FILTER_CUDA: str = (
    f"tonemap_cuda=tonemap=mobius:desat=0:peak=100,"
    f"scale_cuda={STREAM_WIDTH}:{STREAM_HEIGHT}:interp_algo=bicubic,"
    "hwdownload,"
    "scale=iw:ih,"
    "format=yuv420p,"
    "format=bgr24"
)

# ---------------------------------------------------------------------------
# CUDA detection (cached after the first call)
# ---------------------------------------------------------------------------

_cuda_available: Optional[bool] = None
_scale_cuda_available: Optional[bool] = None
_tonemap_cuda_available: Optional[bool] = None

# Cached output of `ffmpeg -filters` (shared by both filter probes).
_ffmpeg_filters_output: Optional[str] = None


def _get_ffmpeg_filters() -> str:
    """Return (cached) output of ``ffmpeg -hide_banner -filters``."""
    global _ffmpeg_filters_output
    if _ffmpeg_filters_output is None:
        try:
            _ffmpeg_filters_output = subprocess.check_output(
                ["ffmpeg", "-hide_banner", "-filters"],
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).decode(errors="replace")
        except Exception:
            _ffmpeg_filters_output = ""
    return _ffmpeg_filters_output


def cuda_available() -> bool:
    """Return True when the local FFmpeg build supports CUDA hw-accel.

    The result is cached after the first call so repeated checks are free.
    """
    global _cuda_available
    if _cuda_available is None:
        try:
            out = subprocess.check_output(
                ["ffmpeg", "-hide_banner", "-hwaccels"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode(errors="replace")
            _cuda_available = "cuda" in out.lower()
        except Exception:
            _cuda_available = False
    return _cuda_available


def scale_cuda_available() -> bool:
    """Return True when FFmpeg exposes the ``scale_cuda`` filter.

    ``scale_cuda`` is present in most FFmpeg CUDA builds and lets us
    downscale frames on the GPU before downloading them to CPU memory,
    reducing both PCIe bandwidth and CPU tonemap workload.

    The result is cached after the first call so repeated checks are free.
    """
    global _scale_cuda_available
    if _scale_cuda_available is None:
        _scale_cuda_available = "scale_cuda" in _get_ffmpeg_filters()
    return _scale_cuda_available


def tonemap_cuda_available() -> bool:
    """Return True when FFmpeg exposes both ``tonemap_cuda`` and ``scale_cuda``.

    These filters are only present in FFmpeg builds compiled with
    ``--enable-cuda-nvcc`` and ``libnpp`` support.  Without them the full-GPU
    tonemap pipeline cannot run.

    The result is cached after the first call so repeated checks are free.
    """
    global _tonemap_cuda_available
    if _tonemap_cuda_available is None:
        out = _get_ffmpeg_filters()
        _tonemap_cuda_available = "tonemap_cuda" in out and "scale_cuda" in out
    return _tonemap_cuda_available

# ---------------------------------------------------------------------------
# HDR detection and per-video filter-chain selection
# ---------------------------------------------------------------------------

# Transfer function values that indicate HDR content.
_HDR_TRANSFERS = frozenset({
    "smpte2084",       # HDR10 / PQ
    "arib-std-b67",    # HLG (broadcast HDR)
    "hlg",             # alternate string used by some encoders
})


def is_hdr_transfer(color_transfer: Optional[str]) -> bool:
    """Return True when *color_transfer* indicates HDR (PQ or HLG) content.

    BT.2020 primaries alone do NOT imply HDR — SDR content encoded in the
    BT.2020 colour space is still SDR.  Only the transfer function reliably
    identifies HDR.

    Args:
        color_transfer: The ``color_transfer`` tag from ffprobe (may be ``None``
                        or an empty string when the stream carries no metadata).

    Returns:
        ``True`` for PQ (smpte2084) and HLG (arib-std-b67 / hlg) streams,
        ``False`` for everything else (bt709, bt601, unknown, …).
    """
    if not color_transfer:
        return False
    return color_transfer.strip().lower() in _HDR_TRANSFERS


def build_vf_filter(is_hdr: bool, use_cuda: bool = True) -> str:
    """Return the FFmpeg ``-vf`` filter string for the given video type.

    Selects the best available pipeline tier at call time:

    * HDR + full-GPU  → ``_TONEMAP_FILTER_CUDA``
    * HDR + scale-GPU → ``_TONEMAP_FILTER_SCALE_CUDA``
    * HDR + CPU-only  → ``_TONEMAP_FILTER``
    * SDR + scale-GPU → ``_SDR_FILTER_SCALE_CUDA``
    * SDR + CPU-only  → ``_SDR_FILTER``

    Args:
        is_hdr:    Whether the source video is HDR (PQ or HLG transfer).
        use_cuda:  Whether CUDA acceleration is requested.  Still falls back
                   to CPU-only when the local FFmpeg has no CUDA support.

    Returns:
        FFmpeg filter string ready for ``-vf``.
    """
    _use_cuda = use_cuda and cuda_available()
    _full_gpu  = _use_cuda and tonemap_cuda_available()
    _scale_gpu = _use_cuda and (not _full_gpu) and scale_cuda_available()

    if is_hdr:
        if _full_gpu:
            return _TONEMAP_FILTER_CUDA
        if _scale_gpu:
            return _TONEMAP_FILTER_SCALE_CUDA
        return _TONEMAP_FILTER
    else:
        # SDR: no tone-mapping needed; applying it would re-linearise the
        # already-correct gamma and make images too bright.
        if _scale_gpu:
            return _SDR_FILTER_SCALE_CUDA
        return _SDR_FILTER

# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def build_frame_assignments_distributed(
    timestamps: List[float],
    format_distribution: Dict[str, Dict[str, int]],
    fps: float,
    n_frames: int = 7,
) -> List[Tuple[int, str, str]]:
    """
    Create format assignments with interleaved distribution.

    Each timestamp is mapped to exactly ONE ``(category, format_name)`` slot.
    Formats are interleaved so every format directory receives patches from
    the very first scenes processed (no "all format1 then all format2" pattern).

    The centre frame index is computed as ``int(ts * fps) + n_frames // 2``
    so that the full n_frames window ``[center-half … center+half]`` starts
    at the original timestamp and all frame indices are non-negative.

    Args:
        timestamps:          Start timestamps of scene windows (seconds).
        format_distribution: ``{category: {format_name: target_count}}``.
        fps:                 Video frame rate.
        n_frames:            Frames per patch window (default 7).

    Returns:
        Sorted list of ``(center_frame_idx, category, format_name)``.
    """
    if not timestamps or not format_distribution:
        return []

    half = n_frames // 2
    # Derive centre-frame indices from start timestamps
    frame_indices: List[int] = [int(ts * fps) + half for ts in sorted(timestamps)]
    total_scenes = len(frame_indices)

    # Build flat slot list: (category, format_name, count)
    all_slots: List[Tuple[str, str, int]] = [
        (category, format_name, count)
        for category, formats in format_distribution.items()
        for format_name, count in formats.items()
    ]

    # Scale down slot counts when the video supplies fewer scenes than needed
    slots_total = sum(c for _, _, c in all_slots)
    if slots_total > total_scenes:
        scale = total_scenes / slots_total
        all_slots = [
            (cat, fmt, max(1, int(cnt * scale))) for cat, fmt, cnt in all_slots
        ]
        excess = sum(c for _, _, c in all_slots) - total_scenes
        if excess > 0:
            all_slots.sort(key=lambda x: -x[2])
            all_slots[0] = (
                all_slots[0][0],
                all_slots[0][1],
                max(0, all_slots[0][2] - excess),
            )

    # Interleaved assignment: each slot's j-th entry gets fractional position
    # j/count in [0, 1).  Sorting by this position distributes all formats
    # evenly across the full scene range while preserving exact per-format counts.
    annotated: List[Tuple[float, str, str]] = []
    for category, format_name, count in all_slots:
        for j in range(count):
            annotated.append((j / count, category, format_name))
    annotated.sort(key=lambda x: x[0])  # stable sort

    assignments: List[Tuple[int, str, str]] = [
        (frame_indices[i], a[1], a[2])
        for i, a in enumerate(annotated)
        if i < total_scenes
    ]
    return sorted(assignments, key=lambda x: x[0])


def build_frame_ranges_from_assignments(
    assignments: List[Tuple[int, str, str]],
    n_frames: int = 7,
) -> List[Tuple[int, int]]:
    """
    Build merged frame ranges needed to satisfy all assignments.

    Each assignment requires a contiguous window of ``n_frames`` frames
    centred on its ``center_frame_idx``.  Overlapping windows are merged
    into a single range to minimise redundant buffer fills.

    Args:
        assignments: Output of :func:`build_frame_assignments_distributed`.
        n_frames:    Frames per patch window (default 7).

    Returns:
        Sorted list of ``(start_frame, end_frame)`` merged ranges (inclusive).
    """
    if not assignments:
        return []

    half = n_frames // 2
    raw: List[Tuple[int, int]] = [
        (max(0, frame_idx - half), frame_idx + half)
        for frame_idx, _, _ in assignments
    ]
    raw.sort()

    merged: List[List[int]] = [list(raw[0])]
    for start, end in raw[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [(s, e) for s, e in merged]


def build_assignments_per_category(
    format_distribution: Dict[str, Dict[str, int]],
    duration: float,
    fps: float,
    n_frames: int = 7,
) -> List[Tuple[int, str, str]]:
    """
    Build streaming assignments treating each category independently.

    For each category a set of *target_count* evenly-spaced timestamps are
    generated and each timestamp is assigned to exactly ONE format within
    that category (interleaved distribution).

    The same video position **can** appear in multiple categories (so the
    same scene is saved to both ``master`` and ``universal``), but will
    **never** appear twice within the same category.

    Because patch windows may overlap (the center frame is what is
    reconstructed; adjacent windows with different centres are distinct
    training samples), the minimum stride between two centres is just one
    frame (``1 / fps``).  This allows up to ``usable * fps`` unique scene
    positions, which is sufficient to reach any practical per-video target.

    Example – video in two categories::

        master:    5 000 patches → 5 000 unique timestamps, each → one format
        universal: 2 000 patches → 2 000 unique timestamps, each → one format
        ─────────────────────────────────────────────────────────────────────
        Total:     7 000 assignments fed into ONE streaming pass.

    Args:
        format_distribution: ``{category: {format_name: target_count}}``.
        duration:            Video duration in seconds.
        fps:                 Video frame rate.
        n_frames:            Frames per patch window (default 7, unused here
                             but kept for API consistency).

    Returns:
        Sorted list of ``(center_frame_idx, category, format_name)``.
        Multiple entries with the same *center_frame_idx* are permitted when
        the same video position is needed by more than one category.
    """
    half = n_frames // 2
    usable = max(0.0, duration - 1.0)
    all_assignments: List[Tuple[int, str, str]] = []

    for category, formats in format_distribution.items():
        cat_total = sum(formats.values())
        if cat_total == 0 or usable <= 0:
            continue

        # Minimum stride: one frame apart.  Overlapping patch windows are
        # fine because each centre frame is a distinct training sample.
        min_stride = (1.0 / fps) if fps > 0 else 0.0
        stride = max(usable / cat_total, min_stride)

        cat_ts: List[float] = []
        for i in range(cat_total):
            ts = i * stride
            if ts < usable:
                cat_ts.append(ts)
            else:
                break

        cat_scenes = len(cat_ts)
        if cat_scenes == 0:
            continue

        # Per-category slot list: [(format_name, count), ...]
        cat_slots: List[Tuple[str, int]] = list(formats.items())
        slots_total = sum(cnt for _, cnt in cat_slots)

        # Scale down when the video is shorter than the per-category target
        if slots_total > cat_scenes:
            scale = cat_scenes / slots_total
            cat_slots = [(fmt, max(1, int(cnt * scale))) for fmt, cnt in cat_slots]
            excess = sum(cnt for _, cnt in cat_slots) - cat_scenes
            if excess > 0:
                cat_slots.sort(key=lambda x: -x[1])
                cat_slots[0] = (cat_slots[0][0], max(0, cat_slots[0][1] - excess))

        # Interleaved format assignment within this category:
        # entry j of a slot with count c gets fractional position j/c → even spread.
        annotated: List[Tuple[float, str]] = []
        for fmt, cnt in cat_slots:
            for j in range(cnt):
                annotated.append((j / cnt, fmt))
        annotated.sort(key=lambda x: x[0])  # stable sort

        for i, (_, fmt) in enumerate(annotated):
            if i < cat_scenes:
                center = int(cat_ts[i] * fps) + half
                all_assignments.append((center, category, fmt))

    return sorted(all_assignments, key=lambda x: x[0])


def degrade_lr_frame(
    frame: np.ndarray,
    degrade_cfg: dict,
    center_frame: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply DVD-typical degradation artefacts to a single LR frame.

    Degradation pipeline (all steps optional / probability-gated):
      1. Random Gaussian noise to simulate sensor/compression noise.
      2. Slight Gaussian blur to simulate the soft lens + encode blur of DVD.
      3. JPEG round-trip at a low quality setting to introduce blocking / DCT
         artefacts characteristic of MPEG-2 / DVD video.

    The function is a **no-op** when ``degrade_cfg`` is ``None`` or when the
    random draw exceeds ``lr_degrade_prob``.

    Args:
        frame:        Single LR BGR frame (numpy uint8 array).
        degrade_cfg:  Dict with degradation parameters (see below).  When
                      ``None`` the frame is returned unchanged.
        center_frame: Optional original-resolution center frame used only to
                      compute mean brightness for the dark-scene boost.  When
                      ``None`` dark-boost is skipped.

    Supported ``degrade_cfg`` keys
    --------------------------------
    lr_degrade_prob         float  Base probability to degrade (default 0.6).
    lr_dark_boost           bool   Increase probability for dark scenes (default True).
    lr_dark_threshold       float  Mean brightness 0-255 below which dark boost
                                   applies (default 60).
    lr_dark_boost_prob      float  Probability used instead of lr_degrade_prob
                                   when the scene is dark (default 0.8).
    lr_jpeg_quality_range   [int, int]  Min/max JPEG quality for round-trip
                                   (default [55, 75]).
    lr_noise_sigma          [float, float]  Min/max Gaussian noise std-dev added
                                   per-channel (default [0.5, 2.5]).
    lr_blur_sigma           [float, float]  Min/max Gaussian blur σ (default
                                   [0.2, 0.7]).  σ < 0.3 is effectively no-op.

    Returns:
        Degraded (or original) frame as uint8 numpy array.
    """
    if degrade_cfg is None:
        return frame

    # Determine effective probability, optionally boosted for dark scenes.
    base_prob: float = float(degrade_cfg.get("lr_degrade_prob", 0.6))
    prob = base_prob
    if degrade_cfg.get("lr_dark_boost", True) and center_frame is not None:
        dark_threshold: float = float(degrade_cfg.get("lr_dark_threshold", 60.0))
        if float(np.mean(center_frame)) < dark_threshold:
            prob = float(degrade_cfg.get("lr_dark_boost_prob", 0.8))

    if random.random() >= prob:
        return frame

    result = frame.astype(np.float32)

    # 1. Gaussian noise (per-channel, additive)
    noise_range = degrade_cfg.get("lr_noise_sigma", [1.0, 4.0])
    sigma = random.uniform(float(noise_range[0]), float(noise_range[1]))
    if sigma > 0.0:
        noise = np.random.normal(0.0, sigma, result.shape).astype(np.float32)
        result = result + noise

    # 2. Gaussian blur (simulates soft lens / encode low-pass)
    blur_range = degrade_cfg.get("lr_blur_sigma", [0.3, 1.0])
    blur_sigma = random.uniform(float(blur_range[0]), float(blur_range[1]))
    result = np.clip(result, 0, 255).astype(np.uint8)
    if blur_sigma >= 0.3:
        # ksize must be odd; derive from sigma: 2*ceil(2σ)+1 capped at 7
        ksize = min(7, 2 * int(np.ceil(2.0 * blur_sigma)) + 1)
        if ksize % 2 == 0:
            ksize += 1
        result = cv2.GaussianBlur(result, (ksize, ksize), blur_sigma)

    # 3. JPEG round-trip (introduces DCT blocking, colour quantisation)
    jpeg_range = degrade_cfg.get("lr_jpeg_quality_range", [35, 60])
    quality = random.randint(int(jpeg_range[0]), int(jpeg_range[1]))
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ok, buf = cv2.imencode(".jpg", result, encode_param)
    if ok:
        result = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return result


def create_patch_pair(
    frames: List[np.ndarray],
    format_name: str,
    format_cfg: dict,
    force_center: bool = False,
    logger=None,
    degrade_cfg: Optional[dict] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Create a ``(GT, LR)`` patch pair from a sequence of frames.

    **16:9 formats** (``medium_169`` / ``720_169``):
      * GT – full-frame resize to ``gt_size`` with ``INTER_LANCZOS4``
        (best quality, no crop).
      * LR – full-frame resize to ``lr_size`` with ``INTER_AREA``
        (DVD-realistic quality).

    **Square formats** (``small_540``, ``large_720``, …):
      * GT – centre frame, cropped to ``gt_size``.
      * LR – all frames cropped and downscaled to ``lr_size`` with
        ``INTER_AREA``, stacked vertically (axis 0).

    In both cases a near-uniform GT (plain black, white, or flat colour) is
    silently discarded (``(None, None)``).  If the source frame is too small
    for the requested resize target a warning is logged.

    When *degrade_cfg* is provided each LR frame is optionally degraded with
    DVD-typical artefacts (noise + blur + JPEG round-trip) by
    :func:`degrade_lr_frame` before stacking.  GT is always kept lossless.

    Args:
        frames:       BGR numpy arrays, length 5 or 7.
        format_name:  Format key (e.g. ``"medium_169"``, ``"small_540"``).
        format_cfg:   Dict with ``'gt_size': [W, H]`` and ``'lr_size': [W, H]``.
        force_center: Square formats only – use the geometric centre of the
                      frame instead of a random crop location.
        logger:       Optional logger instance for warning messages.
        degrade_cfg:  Optional degradation config dict (see :func:`degrade_lr_frame`).
                      When ``None`` no degradation is applied.

    Returns:
        ``(gt, lr_stacked)`` or ``(None, None)`` on failure.
    """
    n = len(frames)
    if n not in (5, 7):
        return None, None

    gt_w, gt_h = format_cfg["gt_size"]
    lr_w, lr_h = format_cfg["lr_size"]

    frame_h, frame_w = frames[0].shape[:2]

    center_idx = n // 2

    if format_name in ("medium_169", "720_169"):
        # Full-frame resize – the whole source frame is scaled to the target
        # size.  No crop is applied so the full 16:9 content is preserved.
        if frame_h < gt_h or frame_w < gt_w:
            if logger:
                logger.warning(
                    f"[{format_name}] Frame too small for resize: "
                    f"{frame_w}×{frame_h} < {gt_w}×{gt_h} – skipped"
                )
            return None, None

        # GT: INTER_LANCZOS4 = highest quality (Lanczos 8×8 neighbourhood)
        gt = cv2.resize(frames[center_idx], (gt_w, gt_h), interpolation=cv2.INTER_LANCZOS4)

        # Variety check: silently discard near-uniform GT (black/white/flat)
        gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        if float(gray.std()) < 15.0:
            return None, None

        # LR: INTER_AREA = DVD-realistic quality, then optional degradation
        center_raw = frames[center_idx]
        lr_frames = []
        for frame in frames:
            lr = cv2.resize(frame, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
            lr = degrade_lr_frame(lr, degrade_cfg, center_frame=center_raw)
            lr_frames.append(lr)
    else:
        if frame_h < gt_h or frame_w < gt_w:
            return None, None

        max_x = frame_w - gt_w
        max_y = frame_h - gt_h

        if force_center:
            crop_x, crop_y = max_x // 2, max_y // 2
        else:
            crop_x = random.randint(0, max_x)
            crop_y = random.randint(0, max_y)

        gt = frames[center_idx][crop_y : crop_y + gt_h, crop_x : crop_x + gt_w]

        # Variety check: silently discard near-uniform GT (black/white/flat)
        gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        if float(gray.std()) < 15.0:
            return None, None

        center_raw = frames[center_idx]
        lr_frames = []
        for frame in frames:
            crop = frame[crop_y : crop_y + gt_h, crop_x : crop_x + gt_w]
            lr = cv2.resize(crop, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
            lr = degrade_lr_frame(lr, degrade_cfg, center_frame=center_raw)
            lr_frames.append(lr)

    lr_stacked = np.concatenate(lr_frames, axis=0)
    return gt, lr_stacked


def save_patch_pair(
    gt: np.ndarray,
    lr: np.ndarray,
    video_path: str,
    timestamp: float,
    category: str,
    format_name: str,
    n_frames: int,
    base_dir: str,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Persist a ``(GT, LR)`` patch pair to the correct output directories.

    Directories are created on demand.  Both images are written with PNG
    compression level 1 for a good speed/size trade-off.

    Args:
        gt:          Ground-truth patch (BGR numpy array).
        lr:          LR stack patch (BGR numpy array).
        video_path:  Source video path (stem used in the patch filename).
        timestamp:   Center-frame timestamp in seconds (used in filename).
        category:    Dataset category (e.g. ``"master"``).
        format_name: Format key (e.g. ``"small_540"``).
        n_frames:    Number of frames (5 or 7) – selects LR subdirectory.
        base_dir:    Root dataset output directory.

    Returns:
        ``(success, gt_path, lr_path)``
    """
    try:
        output_dirs = get_output_dirs_for_format(base_dir, category, format_name, n_frames)
        gt_dir = output_dirs["gt"]
        lr_dir = output_dirs["lr"]

        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(lr_dir, exist_ok=True)

        video_stem = Path(video_path).stem
        patch_name = f"{video_stem}_{int(timestamp * 1000):08d}.png"

        gt_path = os.path.join(gt_dir, patch_name)
        lr_path = os.path.join(lr_dir, patch_name)

        cv2.imwrite(gt_path, gt, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        cv2.imwrite(lr_path, lr, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        return True, gt_path, lr_path
    except Exception:
        return False, None, None


def is_black_frame(gt: np.ndarray, brightness_threshold: float = 20.0) -> bool:
    """Return True when *gt* is predominantly black/dark.

    A mean pixel brightness below *brightness_threshold* (0–255) is used as
    the criterion.  The default of 20 catches solid-black frames and the
    typical fade-in/out darkness at video start/end without affecting normal
    content.

    Args:
        gt:                   Center-frame array (BGR numpy array).
        brightness_threshold: Maximum mean brightness to consider black (default 20).

    Returns:
        ``True`` when the patch is too dark to be useful.
    """
    return float(np.mean(gt)) < brightness_threshold


def extract_and_save_streaming_distributed(
    video_path: str,
    assignments: List[Tuple[int, str, str]],
    n_frames: int,
    format_config: Dict[str, Dict],
    base_dir: str,
    fps: float,
    logger=None,
    is_interesting_fn: Optional[Callable[[np.ndarray], bool]] = None,
    is_black_frame_fn: Optional[Callable[[np.ndarray], bool]] = None,
    progress_fn: Optional[Callable[[int, Dict[str, int], int], None]] = None,
    use_cuda: bool = True,
    nice_level: int = 10,
    is_hdr: bool = True,
    degrade_cfg: Optional[dict] = None,
) -> Dict[str, int]:
    """
    Stream the video once and save patches as frames pass through the buffer.

    A single FFmpeg process reads the video linearly (no ``-ss`` seeking).
    Frames are piped as raw BGR24 data at 1920×1080.  A rolling dictionary
    buffer keeps the last ``n_frames`` decoded frames in memory.  When a
    target centre frame has been decoded and all ``n_frames`` of its window
    are in the buffer, the patch is created and saved immediately.

    The stream is terminated early once the last needed frame has been read.

    Args:
        video_path:        Path to input video.
        assignments:       Output of :func:`build_assignments_per_category`.
        n_frames:          Frames per patch window (default 7).
        format_config:     ``{category: {format_name: {'gt_size': …, 'lr_size': …}}}``.
        base_dir:          Root dataset output directory.
        fps:               Video frame rate.
        logger:            Optional logger instance.
        is_interesting_fn: Optional callable ``(patch: np.ndarray) -> bool`` for
                           quality gating.  When provided, random crops are re-tried
                           up to 5 times before falling back to a centre crop.
        is_black_frame_fn: Optional callable ``(frame: np.ndarray) -> bool``
                           receiving the raw center frame.  When provided and
                           returns ``True`` the entire video position (all its
                           category/format pairs) is skipped without saving.
                           Defaults to :func:`is_black_frame` when ``None`` is
                           passed (i.e. black frames are always filtered unless you
                           explicitly pass ``lambda _: False``).
                           Note: the ``is_black_frame`` default filter (mean < 20)
                           partially overlaps with the variety-std check inside
                           ``create_patch_pair`` (std < 15).  Set
                           ``is_black_frame_fn=lambda _: False`` to disable the
                           pre-filter entirely if you rely solely on the variety
                           check via ``min_variety_std`` in the quality config.
        progress_fn:       Optional callable
                           ``(frames_examined: int,
                              patches_so_far: Dict[str, int],
                              raw_frames_read: int)``
                           invoked after *every* processed assignment (saved **or**
                           skipped).  ``raw_frames_read`` is the total number of
                           raw video frames decoded from the stream so far.
        use_cuda:          When ``True`` (default), enable CUDA hardware-accelerated
                           decoding if the local FFmpeg build supports it.  Falls
                           back to software decoding automatically when CUDA is not
                           available.
        nice_level:        CPU-priority adjustment passed to ``os.nice()`` for the
                           FFmpeg subprocess (default 10 = lower priority).  Has no
                           effect on non-Unix platforms.
        is_hdr:            Whether the source video uses an HDR transfer function
                           (PQ / HLG).  When ``True`` (default) the full HDR→SDR
                           tonemap chain is applied.  When ``False`` a lightweight
                           scale-only chain is used, avoiding incorrect
                           re-linearisation of SDR gamma that would cause
                           overexposure.
        degrade_cfg:       Optional degradation config dict forwarded to
                           :func:`create_patch_pair` / :func:`degrade_lr_frame`.
                           When ``None`` no LR degradation is applied.  Populate
                           from the ``quality`` section of the generator config
                           (keys: ``lr_degrade_prob``, ``lr_dark_boost``, etc.).

    Returns:
        ``{category: patches_saved_count}``
    """

    def _log(msg: str) -> None:
        if logger:
            logger.info(msg)

    # Default: always filter black frames unless caller opts out explicitly.
    _black_fn: Callable[[np.ndarray], bool] = (
        is_black_frame_fn if is_black_frame_fn is not None else is_black_frame
    )

    if not assignments:
        return {}

    half = n_frames // 2
    sorted_asgn = sorted(assignments, key=lambda x: x[0])

    # Build mapping: center_frame_idx → [(category, format_name), …]
    center_map: Dict[int, List[Tuple[str, str]]] = {}
    for frame_idx, category, fmt_name in sorted_asgn:
        center_map.setdefault(frame_idx, []).append((category, fmt_name))

    pending_centers: List[int] = sorted(center_map.keys())
    last_needed: int = pending_centers[-1] + half if pending_centers else 0

    # Build FFmpeg command.
    #
    # Pipeline tier is chosen by build_vf_filter() based on is_hdr and
    # available CUDA capabilities:
    #
    #  HDR source  + full-GPU   → tonemap_cuda + scale_cuda + hwdownload
    #  HDR source  + scale-GPU  → scale_cuda + hwdownload (p010) + zscale+tonemap
    #  HDR source  + CPU-only   → zscale + tonemap + scale (software)
    #  SDR source  + scale-GPU  → scale_cuda + hwdownload (plain scale)
    #  SDR source  + CPU-only   → scale (software, no linearisation)
    _use_cuda = use_cuda and cuda_available()
    _full_gpu  = _use_cuda and is_hdr and tonemap_cuda_available()
    _scale_gpu = _use_cuda and (not _full_gpu) and scale_cuda_available()

    vf_filter = build_vf_filter(is_hdr=is_hdr, use_cuda=use_cuda)

    # -init_hw_device cuda=hw explicitly initialises the CUDA device context
    # before demuxing begins.  Without this flag some FFmpeg builds silently
    # fall back to software decoding when the GPU context fails to auto-init,
    # causing the GPU filter chain to receive CPU frames and crash.
    _CUDA_HW_INIT = ["-init_hw_device", "cuda=hw"]

    hdr_label = "HDR" if is_hdr else "SDR"
    if _full_gpu:
        hw_args        = [*_CUDA_HW_INIT, "-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        pipeline_label = f"full-GPU tonemap_cuda+scale_cuda [{hdr_label}]"
    elif _scale_gpu:
        hw_args        = [*_CUDA_HW_INIT, "-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        pipeline_label = f"scale-GPU + CPU {'zscale/tonemap' if is_hdr else 'passthrough'} [{hdr_label}]"
    elif _use_cuda:
        hw_args        = [*_CUDA_HW_INIT, "-hwaccel", "cuda"]
        pipeline_label = f"decode-GPU + CPU {'tonemap' if is_hdr else 'scale'} [{hdr_label}]"
    else:
        hw_args        = []
        pipeline_label = f"CPU-only {'tonemap' if is_hdr else 'scale'} [{hdr_label}]"

    _log(
        f"🎬 Streaming extractor: {len(sorted_asgn)} assignments, "
        f"last frame needed: {last_needed}, "
        f"pipeline={pipeline_label}, nice={nice_level}"
    )

    cmd = [
        "ffmpeg",
        *hw_args,
        "-i", video_path,
        "-vf", vf_filter,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]

    frame_bytes: int = STREAM_WIDTH * STREAM_HEIGHT * 3
    patches_created: Dict[str, int] = {}

    # Rolling buffer: frame_idx → BGR frame (numpy array)
    buffer: Dict[int, np.ndarray] = {}
    pending_idx: int = 0  # index into pending_centers
    frames_examined: int = 0  # assignments processed (saved + skipped)

    # Reduce FFmpeg CPU priority so interactive processes stay responsive.
    # psutil is used instead of preexec_fn because preexec_fn is unsafe in
    # multi-threaded programs (Python docs recommend avoiding it).
    # nice() is a no-op on Windows where the concept does not apply.
    def _set_nice(pid: int) -> None:
        if nice_level == 0 or _sys.platform == "win32":
            return
        try:
            import psutil as _psutil
            _psutil.Process(pid).nice(nice_level)
        except Exception:
            pass

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _set_nice(process.pid)

    # Drain stderr in a background thread so the pipe never blocks the writer.
    # The collected lines are logged if FFmpeg produces no frames (crash/error).
    stderr_lines: List[str] = []

    def drain_stderr(pipe: "subprocess.IO[bytes]") -> None:
        for raw in pipe:
            stderr_lines.append(raw.decode(errors="replace").rstrip())
        pipe.close()

    stderr_thread = threading.Thread(
        target=drain_stderr, args=(process.stderr,), daemon=True
    )
    stderr_thread.start()

    try:
        current_frame: int = 0
        _t_start: Optional[float] = None   # set on first frame (excludes startup)
        _log_interval: int = 100           # log throughput every N raw frames

        while pending_idx < len(pending_centers):
            raw = process.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                _log("⚠️  Video stream ended before all assignments were processed")
                break

            # Start the clock on the very first frame so FFmpeg startup time
            # (device init, demux, codec open) is excluded from the FPS figure.
            if _t_start is None:
                _t_start = time.monotonic()

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (STREAM_HEIGHT, STREAM_WIDTH, 3)
            ).copy()
            buffer[current_frame] = frame

            # Evict frames that are no longer needed by any pending assignment.
            # The earliest window we still need starts at pending_center - half.
            min_keep = max(0, pending_centers[pending_idx] - half)
            for old_idx in [k for k in buffer if k < min_keep]:
                del buffer[old_idx]

            # Satisfy pending assignments whose full window is now in the buffer
            while pending_idx < len(pending_centers):
                center = pending_centers[pending_idx]
                if current_frame < center + half:
                    break  # Need more frames

                # Build the n_frames window; clamp negative indices to 0
                window: List[np.ndarray] = []
                for fi in range(center - half, center + half + 1):
                    frm = buffer.get(max(0, fi))
                    if frm is None:
                        break
                    window.append(frm)

                if len(window) == n_frames:
                    ts = center / fps

                    # Black-frame check on the raw center frame once per video
                    # position – before iterating over (category, format) pairs.
                    center_raw = window[n_frames // 2]
                    frames_examined += 1

                    if _black_fn(center_raw):
                        _log(f"  ⏭ frame {center} skipped (black frame)")
                    else:
                        for category, fmt_name in center_map[center]:
                            cfg = format_config.get(category, {}).get(fmt_name, {})
                            if not cfg:
                                continue

                            # Resize formats (medium_169/720_169) always use the full
                            # frame – retrying never changes the result.
                            is_resize_fmt = fmt_name in ("medium_169", "720_169")
                            max_attempts = 1 if is_resize_fmt else 6
                            gt, lr = None, None
                            for attempt in range(max_attempts):
                                force = attempt >= 5
                                gt, lr = create_patch_pair(
                                    window, fmt_name, cfg,
                                    force_center=force, logger=logger,
                                    degrade_cfg=degrade_cfg,
                                )
                                if gt is None:
                                    continue
                                if (
                                    is_interesting_fn is None
                                    or is_interesting_fn(gt)
                                    or force
                                ):
                                    break

                            if gt is not None and lr is not None:
                                ok, _, _ = save_patch_pair(
                                    gt, lr, video_path, ts,
                                    category, fmt_name, n_frames, base_dir,
                                )
                                if ok:
                                    patches_created[category] = (
                                        patches_created.get(category, 0) + 1
                                    )
                                    _log(
                                        f"  ✓ frame {center} → {category}/{fmt_name}"
                                    )

                    if progress_fn is not None:
                        progress_fn(frames_examined, dict(patches_created), current_frame)

                pending_idx += 1

            current_frame += 1

            # Periodic throughput log.
            # FPS = raw decoded frames per second (pipeline throughput).
            # SPS = scene-sets completed per second (= assignments processed / s).
            if _t_start is not None and current_frame % _log_interval == 0:
                _elapsed = time.monotonic() - _t_start
                if _elapsed > 0:
                    _fps_actual = current_frame / _elapsed
                    _sps_actual = frames_examined / _elapsed
                    _log(
                        f"  📊 frame {current_frame:>6}  "
                        f"FPS {_fps_actual:>6.1f}  SPS {_sps_actual:>6.2f}  "
                        f"(scenes completed: {frames_examined})"
                    )

            # Early exit once the last required frame has been read
            if current_frame > last_needed:
                break

    finally:
        try:
            process.stdout.close()
        except Exception:
            pass
        process.kill()
        process.wait()
        stderr_thread.join(timeout=2)
        # Log FFmpeg stderr whenever no frames were produced — this is the
        # most useful diagnostic for filter chain errors (e.g. unsupported
        # interp_algo, pixel format mismatch, missing filter).
        if current_frame == 0 and stderr_lines:
            _log("FFmpeg stderr (last 20 lines):")
            for _line in stderr_lines[-20:]:
                _log(f"  [ffmpeg] {_line}")

    # GPU pipeline produced zero frames — most likely a runtime hw-accel failure
    # (e.g. CUDA driver mismatch, scale_cuda format-negotiation bug, or FFmpeg
    # silently falling back to software decode while the filtergraph still
    # contains scale_cuda / hwdownload GPU filters).
    # Retry transparently with the CPU-only pipeline so extraction still
    # completes, rather than silently returning 0 patches.
    if current_frame == 0 and (_full_gpu or _scale_gpu):
        _log(
            "⚠️  GPU pipeline produced no frames — retrying with CPU-only pipeline"
        )
        return extract_and_save_streaming_distributed(
            video_path=video_path,
            assignments=assignments,
            n_frames=n_frames,
            format_config=format_config,
            base_dir=base_dir,
            fps=fps,
            logger=logger,
            is_interesting_fn=is_interesting_fn,
            is_black_frame_fn=is_black_frame_fn,
            progress_fn=progress_fn,
            use_cuda=False,
            nice_level=nice_level,
            is_hdr=is_hdr,
        )

    total = sum(patches_created.values())
    _elapsed_total = (
        (time.monotonic() - _t_start) if _t_start is not None else 0.0
    )
    if _elapsed_total > 0:
        _fps_final = current_frame / _elapsed_total
        _sps_final = frames_examined / _elapsed_total
        _log(
            f"✓ Streaming extraction done: {total} patches saved, "
            f"{frames_examined} assignments examined, "
            f"{current_frame} frames decoded — "
            f"FPS {_fps_final:.1f}  SPS {_sps_final:.2f}"
        )
    else:
        _log(
            f"✓ Streaming extraction done: {total} patches saved, "
            f"{frames_examined} assignments examined"
        )
    return patches_created
