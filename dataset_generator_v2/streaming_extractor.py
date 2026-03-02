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
# HDR→SDR tonemap filter chains
#
# All three chains handle HDR10 (SMPTE 2084 / BT.2020) and Dolby Vision
# Profile 5 / 8 correctly.  Both DV P5 and DV P8 encode the base layer with
# BT.2020 primaries and SMPTE 2084 (PQ) transfer — zscale reads those from
# the stream metadata and converts to linear light automatically.
# DV Profile 4 (rare; only on a handful of Apple TV+ / older Disney+ titles)
# requires Dolby's proprietary decoder for the enhancement layer; FFmpeg
# decodes only the SDR base layer for that profile.
# ---------------------------------------------------------------------------

# Software HDR→SDR tonemap filter chain (CPU-only fallback).
# Used when the local FFmpeg has no CUDA filter support at all.
_TONEMAP_FILTER: str = (
    "zscale=t=linear:npl=100,"
    "format=gbrpf32le,"
    "zscale=p=bt709,"
    "tonemap=tonemap=mobius:desat=0,"
    "zscale=t=bt709:m=bt709:range=limited,"
    f"scale={STREAM_WIDTH}:{STREAM_HEIGHT}:flags=lanczos,"
    "format=bgr24"
)

# Hybrid GPU/CPU filter chain.
# Requires only scale_cuda (no tonemap_cuda needed).
# scale_cuda downscales 4K→1080p on the GPU, so hwdownload transfers only
# the small 1080p frame to CPU (4× less PCIe traffic than downloading 4K).
# The CPU tonemap chain then runs on the already-scaled frame — no final
# CPU scale step required.
# Use together with -hwaccel cuda -hwaccel_output_format cuda.
# Notes:
#   - interp_algo=bicubic: lanczos is not compiled into most pre-built FFmpeg
#     packages (requires --enable-cuda-nvcc Lanczos kernel).
#   - format=nv12 on scale_cuda: forces 8-bit NV12 CUDA surface before
#     hwdownload.  Without this, 10-bit HEVC decodes to p010le on the CUDA
#     surface, which may cause pipeline failure.
#   - hwdownload (bare): copies the NV12 CUDA surface to CPU memory as NV12.
#     We intentionally do NOT use hwdownload=format=nv12 because the 'format'
#     option is not present in older FFmpeg builds and causes:
#       Error applying option 'format' to filter 'hwdownload': Option not found
#     A bare hwdownload would normally crash with:
#       [hwdownload] Invalid output format yuv420p for hwframe download.
#     because FFmpeg's backward format negotiation sees the downstream
#     format=yuv420p filter and tries to make hwdownload produce yuv420p
#     directly from the CUDA NV12 surface, which is impossible.
#   - scale=iw:ih (no resize, same dimensions): breaks the backward format
#     negotiation.  scale (libswscale) accepts NV12 as input and satisfies the
#     downstream yuv420p request by doing the NV12→YUV420P conversion in
#     software.  hwdownload only sees scale's input constraints (many formats
#     including nv12), so it correctly outputs NV12.  Works on all FFmpeg
#     versions.
#   - format=yuv420p after scale: ensures planar yuv420p for zscale/tonemap.
_TONEMAP_FILTER_SCALE_CUDA: str = (
    f"scale_cuda={STREAM_WIDTH}:{STREAM_HEIGHT}:interp_algo=bicubic:format=nv12,"
    "hwdownload,"
    "scale=iw:ih,"
    "format=yuv420p,"
    "zscale=t=linear:npl=100,"
    "format=gbrpf32le,"
    "zscale=p=bt709,"
    "tonemap=tonemap=mobius:desat=0,"
    "zscale=t=bt709:m=bt709:range=limited,"
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

    Example – video in two categories::

        master:    5 000 patches → 5 000 unique timestamps, each → one format
        universal: 2 000 patches → 2 000 unique timestamps, each → one format
        ─────────────────────────────────────────────────────────────────────
        Total:     7 000 assignments fed into ONE streaming pass.

    Args:
        format_distribution: ``{category: {format_name: target_count}}``.
        duration:            Video duration in seconds.
        fps:                 Video frame rate.
        n_frames:            Frames per patch window (default 7).

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

        # Per-category stride – minimum 0.5 s to avoid sub-frame collisions
        stride = max(usable / cat_total, 0.5)

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


def create_patch_pair(
    frames: List[np.ndarray],
    format_name: str,
    format_cfg: dict,
    force_center: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Create a ``(GT, LR)`` patch pair from a sequence of frames.

    * **GT** – centre frame, cropped to ``gt_size``.
    * **LR** – all frames cropped and downscaled to ``lr_size``, stacked
      vertically (axis 0).

    Args:
        frames:       BGR numpy arrays, length 5 or 7.
        format_name:  Format key (e.g. ``"small_540"``).  Currently unused
                      inside the function but kept for future dispatch.
        format_cfg:   Dict with ``'gt_size': [W, H]`` and
                      ``'lr_size': [W, H]``.
        force_center: Use the geometric centre of the frame instead of a
                      random crop location.

    Returns:
        ``(gt, lr_stacked)`` or ``(None, None)`` on failure.
    """
    n = len(frames)
    if n not in (5, 7):
        return None, None

    gt_w, gt_h = format_cfg["gt_size"]
    lr_w, lr_h = format_cfg["lr_size"]

    frame_h, frame_w = frames[0].shape[:2]
    if frame_h < gt_h or frame_w < gt_w:
        return None, None

    max_x = frame_w - gt_w
    max_y = frame_h - gt_h

    if force_center:
        crop_x, crop_y = max_x // 2, max_y // 2
    else:
        crop_x = random.randint(0, max_x)
        crop_y = random.randint(0, max_y)

    center_idx = n // 2
    gt = frames[center_idx][crop_y : crop_y + gt_h, crop_x : crop_x + gt_w]

    lr_frames = []
    for frame in frames:
        crop = frame[crop_y : crop_y + gt_h, crop_x : crop_x + gt_w]
        lr_frames.append(cv2.resize(crop, (lr_w, lr_h), interpolation=cv2.INTER_AREA))

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
    # Four pipeline tiers, chosen automatically at runtime:
    #
    #  1. Full-GPU (best):  -hwaccel cuda -hwaccel_output_format cuda
    #                       + tonemap_cuda + scale_cuda + hwdownload
    #     Frames stay in GPU VRAM from decode through HDR→SDR tonemapping
    #     and 1920×1080 scaling.  Only the final BGR24 result is copied to
    #     CPU.  Requires FFmpeg with --enable-cuda-nvcc/libnpp.
    #
    #  2. Scale-GPU + CPU tonemap:  -hwaccel cuda -hwaccel_output_format cuda
    #                               + scale_cuda + hwdownload + CPU tonemap
    #     scale_cuda downscales 4K→1080p on the GPU.  hwdownload then
    #     transfers only the 1080p frame to CPU (4× less PCIe bandwidth than
    #     downloading 4K).  The CPU tonemap runs on the already-scaled frame
    #     so it processes 4× less data.  No final CPU scale step needed.
    #     Requires only scale_cuda (no tonemap_cuda).
    #
    #  3. Decode-GPU + CPU tonemap:  -hwaccel cuda + full software chain
    #     GPU decoding only; the full 4K zscale/tonemap/scale chain runs on
    #     CPU.  Falls back here when CUDA is available but scale_cuda is not.
    #
    #  4. Pure CPU:  no hwaccel, full software chain.
    _use_cuda = use_cuda and cuda_available()
    _full_gpu  = _use_cuda and tonemap_cuda_available()
    _scale_gpu = _use_cuda and (not _full_gpu) and scale_cuda_available()

    if _full_gpu:
        hw_args        = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        vf_filter      = _TONEMAP_FILTER_CUDA
        pipeline_label = "full-GPU (tonemap_cuda+scale_cuda)"
    elif _scale_gpu:
        hw_args        = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        vf_filter      = _TONEMAP_FILTER_SCALE_CUDA
        pipeline_label = "scale-GPU + CPU tonemap (scale_cuda)"
    elif _use_cuda:
        hw_args        = ["-hwaccel", "cuda"]
        vf_filter      = _TONEMAP_FILTER
        pipeline_label = "decode-GPU + CPU tonemap"
    else:
        hw_args        = []
        vf_filter      = _TONEMAP_FILTER
        pipeline_label = "CPU-only"

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

        while pending_idx < len(pending_centers):
            raw = process.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                _log("⚠️  Video stream ended before all assignments were processed")
                break

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

                            # Up to 5 random crops; 6th attempt is forced centre crop
                            gt, lr = None, None
                            for attempt in range(6):
                                force = attempt >= 5
                                gt, lr = create_patch_pair(
                                    window, fmt_name, cfg, force_center=force
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

    total = sum(patches_created.values())
    _log(
        f"✓ Streaming extraction done: {total} patches saved, "
        f"{frames_examined} assignments examined"
    )
    return patches_created
