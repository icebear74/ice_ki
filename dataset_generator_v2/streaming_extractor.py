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

snap_assignments_to_centers()
    Snap near-duplicate center frame indices (within ±tol_frames) to a
    shared representative center, reducing redundant work when multiple
    categories are active.

build_dual_vf_filter()
    Build a ``-filter_complex`` string that produces a 4K (3840×2160) and
    an HD (1920×1080) output stream in a single FFmpeg pass.
    (Retained for reference; no longer called by the dual-buffer pipeline.)

extract_and_save_streaming_dual()
    Preferred entry point.  Single 4K FFmpeg pass with dual rolling buffers:

    * ``buffer_4k`` holds the raw 4K frames (3840×2160) for FORMATS_4K_STREAM
      (720, large_720, 720_169, medium_169).
    * ``buffer_hd`` holds the same frames downscaled **in Python** to 1920×1080
      via ``cv2.INTER_LANCZOS4`` for FORMATS_HD_STREAM (540, small_540).

    No second FFmpeg pass is needed — the HD frame is derived in-memory from
    the already tone-mapped 4K frame.  LR degradation (DVD artefacts) is still
    applied to every LR patch via ``degrade_cfg``.  Falls back to a single
    1080p pass when the source is smaller than 4K.

extract_and_save_streaming_distributed()
    Core single-stream entry point.  Launches one FFmpeg process, streams
    BGR24 frames at the requested resolution (default 1920×1080), saves
    patches on-the-fly.  Accepts optional ``stream_width``/``stream_height``
    for caller-specified output dimensions.  Passes the filter chain via a
    temp file (``-/filter_complex`` on FFmpeg ≥ 5, ``-filter_complex_script``
    on FFmpeg 4), avoiding OS ARG_MAX limits.

create_patch_pair()
    Create a (GT, LR) patch pair from a sequence of frames.

save_patch_pair()
    Persist a (GT, LR) pair to the correct output directories.

FFmpeg error logging
--------------------
Both streaming functions write all FFmpeg stderr output to
``<base_dir>/ffmpeg_errors.log`` after every video so that codec
warnings, hw-accel failures and filter-chain errors are retained for
post-run analysis.
"""

import os
import queue
import random
import subprocess
import tempfile
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

# 4K-Stream (für 720 Crops und 720_169 Vollbild)
STREAM_4K_WIDTH:  int = 3840
STREAM_4K_HEIGHT: int = 2160
# HD-Stream (für 540 Crops)
STREAM_HD_WIDTH:  int = 1920
STREAM_HD_HEIGHT: int = 1080

# Bytes pro Frame je Stream
FRAME_BYTES_4K: int = STREAM_4K_WIDTH  * STREAM_4K_HEIGHT  * 3  # ~24 MB
FRAME_BYTES_HD: int = STREAM_HD_WIDTH  * STREAM_HD_HEIGHT  * 3  #  ~6 MB

# Formate die den 4K-Stream benötigen (direkt aus 4K)
FORMATS_4K_STREAM = frozenset({"720", "large_720", "720_169", "medium_169"})
# Formate die den HD-Stream benötigen (aus 1080p)
FORMATS_HD_STREAM = frozenset({"540", "small_540"})

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
# Performance notes:
#   - tonemap=reinhard is the fastest tone-mapper (simple x/(1+x) curve).
#   - flags=lanczos gives the highest-quality resize to 1080 (important for GT
#     quality in full-frame formats like 720_169).
#   - filter=bilinear in each zscale step speeds up any incidental resampling.
_TONEMAP_FILTER: str = (
    "zscale=t=linear:npl=100:filter=bilinear,"
    "format=gbrpf32le,"
    "zscale=p=bt709:filter=bilinear,"
    "tonemap=tonemap=reinhard:desat=0,"
    "zscale=t=bt709:m=bt709:range=full:filter=bilinear,"
    f"scale={STREAM_WIDTH}:{STREAM_HEIGHT}:flags=lanczos,"
    "format=bgr24"
)

# SDR pass-through: Software (CPU-only).
# No linearisation or tonemap needed — just scale + convert to BGR24.
# flags=lanczos: highest-quality downscale, best GT fidelity.
_SDR_FILTER: str = (
    f"scale={STREAM_WIDTH}:{STREAM_HEIGHT}:flags=lanczos,"
    "format=bgr24"
)

# HDR→SDR: Hybrid GPU/CPU — scale_cuda downscales on GPU, tonemap on CPU.
# hwdownload + format=p010 preserves 10-bit precision; the multi-step
# zscale+tonemap chain is identical to _TONEMAP_FILTER.
# Use together with -init_hw_device cuda=hw -hwaccel cuda
#                  -hwaccel_output_format cuda.
# interp_algo=bicubic: best quality available in scale_cuda (no lanczos).
_TONEMAP_FILTER_SCALE_CUDA: str = (
    f"scale_cuda={STREAM_WIDTH}:{STREAM_HEIGHT}:interp_algo=bicubic,"
    "hwdownload,"
    "format=p010,"
    "zscale=t=linear:npl=100:filter=bilinear,"
    "format=gbrpf32le,"
    "zscale=p=bt709:filter=bilinear,"
    "tonemap=tonemap=reinhard:desat=0,"
    "zscale=t=bt709:m=bt709:range=full:filter=bilinear,"
    "format=bgr24"
)

# SDR pass-through: Hybrid GPU/CPU — scale on GPU, convert on CPU.
# interp_algo=bicubic: best quality available in scale_cuda (no lanczos).
_SDR_FILTER_SCALE_CUDA: str = (
    f"scale_cuda={STREAM_WIDTH}:{STREAM_HEIGHT}:interp_algo=bicubic,"
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

# Cached FFmpeg major version (4 = conservative fallback).
_ffmpeg_major_ver: Optional[int] = None


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


def _get_ffmpeg_major_version() -> int:
    """Return the major version of the installed FFmpeg (cached).

    Used to select the correct output options:
      * FFmpeg ≥ 5: ``-fps_mode passthrough`` (replaces deprecated ``-vsync``)
                    and ``-/filter_complex file`` (replaces ``-filter_complex_script``)
      * FFmpeg 4:   ``-vsync 0``  and  ``-filter_complex_script file``

    Standard release builds report the version as a numeric string, e.g.
    ``"ffmpeg version 6.1.1 …"``.  Git snapshot builds use a non-numeric
    token such as ``"ffmpeg version N-123114-gfb3012269e …"``.  In that case
    we fall back to parsing the ``libavutil`` major version, which is
    incremented with every FFmpeg major release:

      libavutil 56 → FFmpeg 4.x
      libavutil 57 → FFmpeg 5.x
      libavutil 58 → FFmpeg 6.x
      libavutil 59 → FFmpeg 7.x
      libavutil 60 → FFmpeg 8.x (dev / nightly builds as of early 2026)

    Returns 4 as a conservative fallback if all detection attempts fail.
    """
    global _ffmpeg_major_ver
    if _ffmpeg_major_ver is None:
        detected: Optional[int] = None
        try:
            out = subprocess.check_output(
                ["ffmpeg", "-version"], stderr=subprocess.DEVNULL, timeout=5,
            ).decode(errors="replace")

            # First attempt: parse the standard numeric version token.
            # e.g. "ffmpeg version 6.1.1 Copyright…" → parts[2] = "6.1.1"
            parts = out.split("\n", 1)[0].split()
            if len(parts) >= 3:
                ver_token = parts[2]
                if ver_token[0].isdigit():
                    try:
                        detected = int(ver_token.split(".")[0])
                    except ValueError:
                        pass

            # Second attempt (git/nightly builds like "N-123114-gfb3012269e"):
            # libavutil major is always available in the -version output.
            if detected is None:
                for line in out.split("\n"):
                    if "libavutil" in line:
                        for tok in line.split():
                            if tok[0].isdigit() and "." in tok:
                                try:
                                    # libavutil major 56 = FFmpeg 4, 57 = 5, …
                                    detected = max(4, int(tok.split(".")[0]) - 52)
                                    break
                                except ValueError:
                                    pass
                        if detected is not None:
                            break
        except Exception:
            pass

        _ffmpeg_major_ver = detected if detected is not None else 4
    return _ffmpeg_major_ver

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


def build_vf_filter(is_hdr: bool, use_cuda: bool = True,
                    width: int = STREAM_WIDTH, height: int = STREAM_HEIGHT) -> str:
    """Return the FFmpeg ``-vf`` filter string for the given video type.

    Selects the best available pipeline tier at call time:

    * HDR + full-GPU  → tonemap_cuda + scale_cuda pipeline
    * HDR + scale-GPU → scale_cuda + CPU zscale/tonemap pipeline
    * HDR + CPU-only  → CPU zscale + tonemap + scale pipeline
    * SDR + scale-GPU → scale_cuda pipeline
    * SDR + CPU-only  → scale pipeline

    Args:
        is_hdr:    Whether the source video is HDR (PQ or HLG transfer).
        use_cuda:  Whether CUDA acceleration is requested.  Still falls back
                   to CPU-only when the local FFmpeg has no CUDA support.
        width:     Output width in pixels (default ``STREAM_WIDTH`` = 1920).
        height:    Output height in pixels (default ``STREAM_HEIGHT`` = 1080).

    Returns:
        FFmpeg filter string ready for ``-vf`` (or for wrapping in
        ``-filter_complex`` as ``[0:v]<filter>[label]``).
    """
    _use_cuda = use_cuda and cuda_available()
    _full_gpu  = _use_cuda and tonemap_cuda_available()
    _scale_gpu = _use_cuda and (not _full_gpu) and scale_cuda_available()

    if is_hdr:
        if _full_gpu:
            return (
                f"tonemap_cuda=tonemap=mobius:desat=0:peak=100,"
                f"scale_cuda={width}:{height}:interp_algo=bicubic,"
                "hwdownload,"
                "scale=iw:ih,"
                "format=yuv420p,"
                "format=bgr24"
            )
        if _scale_gpu:
            return (
                f"scale_cuda={width}:{height}:interp_algo=bicubic,"
                "hwdownload,"
                "format=p010,"
                "zscale=t=linear:npl=100:filter=bilinear,"
                "format=gbrpf32le,"
                "zscale=p=bt709:filter=bilinear,"
                "tonemap=tonemap=reinhard:desat=0,"
                "zscale=t=bt709:m=bt709:range=full:filter=bilinear,"
                "format=bgr24"
            )
        return (
            "zscale=t=linear:npl=100:filter=bilinear,"
            "format=gbrpf32le,"
            "zscale=p=bt709:filter=bilinear,"
            "tonemap=tonemap=reinhard:desat=0,"
            "zscale=t=bt709:m=bt709:range=full:filter=bilinear,"
            f"scale={width}:{height}:flags=lanczos,"
            "format=bgr24"
        )
    else:
        # SDR: no tone-mapping needed; applying it would re-linearise the
        # already-correct gamma and make images too bright.
        if _scale_gpu:
            return (
                f"scale_cuda={width}:{height}:interp_algo=bicubic,"
                "hwdownload,"
                "format=bgr24"
            )
        return (
            f"scale={width}:{height}:flags=lanczos,"
            "format=bgr24"
        )


def build_dual_vf_filter(is_hdr: bool, use_cuda: bool = True) -> str:
    """
    Build a filter_complex string that produces TWO output streams in one FFmpeg pass:
      [out4k]   → 3840×2160 BGR24  (used for 720/720_169 formats)
      [out1080] → 1920×1080 BGR24  (used for 540 formats)

    For HDR sources the tonemap runs once at full 4K resolution, then the
    result is split and the 1080p branch is a cheap software downscale.
    For SDR sources the split happens before any scaling.

    Returns a -filter_complex string (NOT a -vf string).
    """
    _use_cuda = use_cuda and cuda_available()

    if is_hdr:
        if _use_cuda and tonemap_cuda_available():
            # Full-GPU tonemap at 4K, then CPU split+scale to HD
            tonemap_part = (
                f"tonemap_cuda=tonemap=mobius:desat=0:peak=100,"
                f"scale_cuda={STREAM_4K_WIDTH}:{STREAM_4K_HEIGHT}:interp_algo=bicubic,"
                "hwdownload,"
                "scale=iw:ih,"
                "format=yuv420p,"
                "format=bgr24"
            )
        elif _use_cuda and scale_cuda_available():
            # scale_cuda to 4K, CPU tonemap
            tonemap_part = (
                f"scale_cuda={STREAM_4K_WIDTH}:{STREAM_4K_HEIGHT}:interp_algo=bicubic,"
                "hwdownload,"
                "format=p010,"
                "zscale=t=linear:npl=100:filter=bilinear,"
                "format=gbrpf32le,"
                "zscale=p=bt709:filter=bilinear,"
                "tonemap=tonemap=reinhard:desat=0,"
                "zscale=t=bt709:m=bt709:range=full:filter=bilinear,"
                "format=bgr24"
            )
        else:
            # CPU-only: tonemap at 4K
            tonemap_part = (
                "zscale=t=linear:npl=100:filter=bilinear,"
                "format=gbrpf32le,"
                "zscale=p=bt709:filter=bilinear,"
                "tonemap=tonemap=reinhard:desat=0,"
                "zscale=t=bt709:m=bt709:range=full:filter=bilinear,"
                f"scale={STREAM_4K_WIDTH}:{STREAM_4K_HEIGHT}:flags=lanczos,"
                "format=bgr24"
            )
        return (
            f"[0:v]{tonemap_part}[sdr4k];"
            f"[sdr4k]split=2[s4k][s4k_b];"
            f"[s4k]null[out4k];"
            f"[s4k_b]scale={STREAM_HD_WIDTH}:{STREAM_HD_HEIGHT}:flags=lanczos[out1080]"
        )
    else:
        # SDR: split first, then scale each branch independently
        if _use_cuda and scale_cuda_available():
            return (
                f"[0:v]split=2[s4k][s4k_b];"
                f"[s4k]scale_cuda={STREAM_4K_WIDTH}:{STREAM_4K_HEIGHT}:interp_algo=bicubic,"
                f"hwdownload,format=bgr24[out4k];"
                f"[s4k_b]scale_cuda={STREAM_HD_WIDTH}:{STREAM_HD_HEIGHT}:interp_algo=bicubic,"
                f"hwdownload,format=bgr24[out1080]"
            )
        else:
            return (
                f"[0:v]split=2[s4k][s4k_b];"
                f"[s4k]scale={STREAM_4K_WIDTH}:{STREAM_4K_HEIGHT}:flags=lanczos,"
                f"format=bgr24[out4k];"
                f"[s4k_b]scale={STREAM_HD_WIDTH}:{STREAM_HD_HEIGHT}:flags=lanczos,"
                f"format=bgr24[out1080]"
            )

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


def snap_assignments_to_centers(
    assignments: List[Tuple[int, str, str]],
    fps: float,
    tol_seconds: float = 1.0,
) -> List[Tuple[int, str, str]]:
    """
    Snap near-duplicate center frame indices across categories to shared centers.

    Assignments whose ``center_frame_idx`` falls within ``±tol_frames`` of an
    already-chosen representative center are remapped to that representative.
    This prevents independent per-category assignments that differ by only a
    few frames from being treated as distinct streaming positions, reducing the
    number of unique centers and improving throughput when multiple categories
    are active.

    The algorithm is O(N log N):

    1. Sort assignments by ``center_frame_idx``.
    2. Walk in order, greedily assigning each center to the current cluster
       representative (first element encountered in that cluster).
    3. A new cluster starts when the current center is more than
       ``tol_frames`` away from the current representative.

    Args:
        assignments:  List of ``(center_frame_idx, category, format_name)``.
        fps:          Video frame rate used to convert seconds → frames.
        tol_seconds:  Tolerance window in seconds (default 1.0).  Set to 0 to
                      disable snapping entirely.

    Returns:
        Sorted list of ``(center_frame_idx, category, format_name)`` with
        snapped center indices.  The ``(category, format_name)`` pairs are
        preserved unchanged; only ``center_frame_idx`` values may be adjusted.
    """
    if not assignments or tol_seconds <= 0.0:
        return sorted(assignments, key=lambda x: x[0])

    tol_frames: int = max(1, int(round(fps * tol_seconds)))

    sorted_asgn = sorted(assignments, key=lambda x: x[0])

    result: List[Tuple[int, str, str]] = []
    rep_center: Optional[int] = None  # current cluster representative

    for frame_idx, category, fmt_name in sorted_asgn:
        if rep_center is None or abs(frame_idx - rep_center) > tol_frames:
            # Start a new cluster: this frame becomes the representative.
            rep_center = frame_idx
        result.append((rep_center, category, fmt_name))

    return result


def _degrade_range(value, default: list) -> list:
    """Return *value* as a two-element ``[lo, hi]`` list.

    Accepts an existing list/tuple (returned as-is), a scalar (broadcast to
    ``[v, v]``), or ``None`` (falls back to *default*).  This makes
    ``degrade_cfg`` robust against both ``"lr_noise_sigma": 2.0`` and
    ``"lr_noise_sigma": [1.0, 3.0]`` entries.
    """
    if value is None:
        return default
    if isinstance(value, (int, float)):
        v = float(value)
        return [v, v]
    return list(value)


def _sample_degrade_params(
    degrade_cfg: dict,
    center_frame: Optional[np.ndarray] = None,
) -> Optional[dict]:
    """
    Draw degradation parameters **once** for an entire scene window.

    Returns a frozen parameter dict that :func:`_apply_degrade_params` can
    apply to every frame in the window, or ``None`` when this scene should
    not be degraded (probability gate not passed).

    Sampling once per scene is the DVD-realistic behaviour: a real MPEG-2
    encoder uses the same quantiser settings for the whole GOP, so all frames
    in the window share the same noise level, blur, and JPEG quality.  Each
    frame still gets **independent** noise samples (sensor noise is
    uncorrelated between frames) but at the same sigma.

    Args:
        degrade_cfg:  Degradation config dict (see :func:`degrade_lr_frame`).
        center_frame: Optional original-resolution center frame used to compute
                      mean brightness for the dark-scene probability boost.

    Returns:
        ``None`` – skip degradation for this scene.
        ``dict`` with keys:

        * ``active_stages`` – list of stage IDs (subset of {1, 2, 3}).
        * ``noise_sigma``   – Gaussian noise std-dev (stage 1).
        * ``blur_sigma``    – Gaussian blur σ (stage 2).
        * ``jpeg_quality``  – JPEG quality integer 1-100 (stage 3).
    """
    # Determine effective probability, optionally boosted for dark scenes.
    base_prob: float = float(degrade_cfg.get("lr_degrade_prob", 0.50))
    prob = base_prob
    if degrade_cfg.get("lr_dark_boost", True) and center_frame is not None:
        dark_threshold: float = float(degrade_cfg.get("lr_dark_threshold", 60.0))
        if float(np.mean(center_frame)) < dark_threshold:
            prob = float(degrade_cfg.get("lr_dark_boost_prob", 0.65))

    if random.random() >= prob:
        return None  # this scene will not be degraded

    # ── Stage selection — DVD/MPEG-2 realistic ────────────────────────────
    # Stage 3 (JPEG) is ALWAYS the primary degradation: it directly simulates
    # MPEG-2 blocking and ringing artefacts present on every compressed DVD.
    #
    # Stage 1 (noise) is an *optional* secondary that simulates film grain
    # which survives MPEG-2 at lower bitrates (25 % chance by default).
    #
    # Stage 2 (blur) is intentionally excluded: the 3× INTER_AREA downscale
    # already reproduces the softness of DVD resolution; real MPEG-2 encoding
    # does not add Gaussian blur on top of the resolution loss.
    #
    # Maximum 2 active stages (JPEG + optional noise).
    stage_prob: float = float(degrade_cfg.get("lr_stage_prob", 0.25))
    active_stages = [3]                        # JPEG always
    if random.random() < stage_prob:
        active_stages.append(1)                # subtle noise — optional

    # Sample scalar parameters once — all frames will use these exact values.
    # Ranges reflect the real DVD bitrate spectrum:
    #   • JPEG 78–92: cheap/heavily compressed discs (78) to premium releases (92)
    #   • noise σ 0.5–2.0: film grain that survived MPEG-2 at lower bitrates
    #   • blur σ kept in config for backward-compat but never activated above
    noise_range  = _degrade_range(degrade_cfg.get("lr_noise_sigma"),        [0.5, 2.0])
    noise_sigma: float = random.uniform(float(noise_range[0]), float(noise_range[1]))

    blur_range   = _degrade_range(degrade_cfg.get("lr_blur_sigma"),         [0.1, 0.4])
    blur_sigma: float = random.uniform(float(blur_range[0]), float(blur_range[1]))

    jpeg_range   = _degrade_range(degrade_cfg.get("lr_jpeg_quality_range"), [78, 92])
    jpeg_quality: int = random.randint(int(jpeg_range[0]), int(jpeg_range[1]))

    return {
        "active_stages": active_stages,
        "noise_sigma":   noise_sigma,
        "blur_sigma":    blur_sigma,
        "jpeg_quality":  jpeg_quality,
    }


def _apply_degrade_params(
    frame: np.ndarray,
    params: dict,
) -> np.ndarray:
    """
    Apply pre-sampled degradation parameters to a single LR frame.

    Unlike :func:`degrade_lr_frame` this function never draws new random
    scalars — it uses the values in *params* verbatim.  Additive noise is
    still drawn freshly for each frame (sensor noise is per-frame
    independent), but the noise sigma is fixed so all frames in the window
    share the same intensity level.

    Args:
        frame:  Single LR BGR frame (uint8 numpy array).
        params: Dict returned by :func:`_sample_degrade_params`.

    Returns:
        Degraded frame as uint8 numpy array.
    """
    active_stages = params["active_stages"]
    result = frame.astype(np.float32)

    # Stage 1: Gaussian noise — new samples per frame, same sigma for all.
    if 1 in active_stages:
        sigma = params["noise_sigma"]
        if sigma > 0.0:
            noise = np.random.normal(0.0, sigma, result.shape).astype(np.float32)
            result = result + noise

    # Stage 2: Gaussian blur — same kernel for every frame in the window.
    if 2 in active_stages:
        blur_sigma = params["blur_sigma"]
        result = np.clip(result, 0, 255).astype(np.uint8)
        if blur_sigma >= 0.3:
            ksize = min(7, 2 * int(np.ceil(2.0 * blur_sigma)) + 1)
            if ksize % 2 == 0:
                ksize += 1
            result = cv2.GaussianBlur(result, (ksize, ksize), blur_sigma)
    else:
        result = np.clip(result, 0, 255).astype(np.uint8)

    # Stage 3: JPEG round-trip — same quality for every frame in the window.
    if 3 in active_stages:
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, params["jpeg_quality"]]
        ok, buf = cv2.imencode(".jpg", result, encode_param)
        if ok:
            result = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return result


def degrade_lr_frame(
    frame: np.ndarray,
    degrade_cfg: dict,
    center_frame: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply DVD-typical degradation artefacts to a single LR frame.

    This is a convenience wrapper for single-frame callers.  When degrading
    a multi-frame scene window, use :func:`_sample_degrade_params` once and
    then :func:`_apply_degrade_params` per frame so that all frames in the
    window share the same degradation parameters (blur sigma, JPEG quality,
    noise sigma) — matching the behaviour of a real MPEG-2 encoder whose
    quantiser settings are constant within a GOP.

    Degradation pipeline (all steps optional / probability-gated):
      1. Gaussian noise to simulate sensor/compression noise.
      2. Gaussian blur to simulate the soft lens + encode blur of DVD.
      3. JPEG round-trip at a low quality setting to introduce blocking / DCT
         artefacts characteristic of MPEG-2 / DVD video.

    Per activation, only a random subset of stages (at most ``lr_max_stages``)
    is applied. The stage order is shuffled so the combination is unpredictable.

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
                                   when the scene is dark (default 0.65).
    lr_stage_prob           float  Probability that film-grain noise (stage 1)
                                   is added on top of the mandatory JPEG stage
                                   (default 0.25).  Stage 2 (blur) is never
                                   activated — the 3× downscale handles
                                   softening; MPEG-2 does not add blur.
    lr_jpeg_quality_range   [int, int]  Min/max JPEG quality for the mandatory
                                   JPEG round-trip (default [78, 92]).
                                   Covers cheap discs (78) to premium releases
                                   (92); maps to real DVD MPEG-2 bitrates.
    lr_noise_sigma          [float, float]  Min/max Gaussian noise std-dev for
                                   the optional film-grain stage (default
                                   [0.5, 2.0]).
    lr_blur_sigma           [float, float]  Kept for config backward-compat;
                                   not used (blur stage excluded, default
                                   [0.1, 0.4]).

    Returns:
        Degraded (or original) frame as uint8 numpy array.
    """
    if degrade_cfg is None:
        return frame
    params = _sample_degrade_params(degrade_cfg, center_frame=center_frame)
    if params is None:
        return frame
    return _apply_degrade_params(frame, params)


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
      * GT – centre frame.  When the source frame is large enough to support
        2× oversampling (``frame_h ≥ 2*gt_h`` **and** ``frame_w ≥ 2*gt_w``,
        e.g. native 4K for the ``720``/``large_720`` formats), a
        ``2*gt_size`` crop is taken from the source and Lanczos4-downsampled
        to ``gt_size``.  The 2× Lanczos4 step averages out per-pixel H.265
        in-loop deblocking softness and produces a clean GT visually
        comparable to the ``INTER_LANCZOS4`` full-frame resize used by the
        ``720_169`` family.  For smaller sources (e.g. 1080p for ``540``) the
        frame is too small to oversample and a direct 1:1 crop is used
        instead (existing behaviour).
      * LR – all frames, same crop region, downscaled to ``lr_size`` with
        ``INTER_AREA`` (stacked vertically on axis 0).  The LR crop covers
        the same spatial area as the GT crop at 3× lower resolution
        (LR/GT = ``scale``), keeping the super-resolution task well-defined.

    In both cases a near-uniform GT (plain black, white, or flat colour) is
    silently discarded (``(None, None)``).  If the source frame is too small
    for the requested resize target a warning is logged.

    When *degrade_cfg* is provided the degradation parameters are **sampled
    once per scene** via :func:`_sample_degrade_params` (using the center
    frame for the dark-scene probability boost), then applied to every LR
    frame with :func:`_apply_degrade_params`.  This means all frames in the
    window share the same noise sigma, blur sigma, and JPEG quality — matching
    the behaviour of a real MPEG-2 encoder where the same quantiser settings
    apply to the whole GOP.  Additive noise samples are still drawn
    independently per frame (sensor noise is uncorrelated), but at the
    consistent sigma.  GT is always kept lossless.

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
        if float(gray.std()) < 7.0:
            return None, None

        # LR: INTER_AREA = DVD-realistic quality, then optional degradation.
        # Parameters are sampled once for the whole scene so that every frame
        # in the window receives identical blur/quality/noise-level settings.
        center_raw = frames[center_idx]
        _scene_params = _sample_degrade_params(degrade_cfg, center_frame=center_raw) if degrade_cfg else None
        lr_frames = []
        for frame in frames:
            lr = cv2.resize(frame, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
            if _scene_params is not None:
                lr = _apply_degrade_params(lr, _scene_params)
            lr_frames.append(lr)
    else:
        # -----------------------------------------------------------------------
        # Square crop formats
        # -----------------------------------------------------------------------
        # When the source is large enough for 2× oversampling (e.g. native 4K
        # for the 720/large_720 formats, where 3840≥1440 and 2160≥1440), take
        # a 2×gt_size crop and Lanczos4-downsample it to gt_size for the GT.
        # The 2× Lanczos4 step averages H.265 in-loop deblocking softness and
        # produces a clean GT comparable to the full-frame INTER_LANCZOS4 resize
        # used by 720_169/medium_169.
        #
        # For smaller sources (1080p → 540/small_540, and any format in the
        # 1080p fallback path) a 2×gt_size crop would exceed the frame height
        # so we fall back to the existing 1:1 native-resolution crop.
        oversample: int = 2 if (frame_h >= 2 * gt_h and frame_w >= 2 * gt_w) else 1
        sample_h: int = gt_h * oversample
        sample_w: int = gt_w * oversample

        if frame_h < sample_h or frame_w < sample_w:
            return None, None

        max_x = frame_w - sample_w
        max_y = frame_h - sample_h

        if force_center:
            crop_x, crop_y = max_x // 2, max_y // 2
        else:
            crop_x = random.randint(0, max_x)
            crop_y = random.randint(0, max_y)

        center_crop = frames[center_idx][
            crop_y : crop_y + sample_h, crop_x : crop_x + sample_w
        ]

        # GT: Lanczos4 downsample from oversampled crop; direct slice otherwise.
        if oversample > 1:
            gt = cv2.resize(center_crop, (gt_w, gt_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            gt = center_crop

        # Variety check: silently discard near-uniform GT (black/white/flat)
        gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        if float(gray.std()) < 7.0:
            return None, None

        center_raw = frames[center_idx]
        # Sample degradation parameters once for the whole scene window so
        # that all 7 LR frames share the same noise sigma, blur sigma, and
        # JPEG quality — consistent with how a real MPEG-2 encoder works.
        _scene_params = _sample_degrade_params(degrade_cfg, center_frame=center_raw) if degrade_cfg else None
        lr_frames = []
        for frame in frames:
            # LR is derived from the same oversampled area so the LR/GT ratio
            # is always exactly `scale` (e.g. 240/720 = 1/3 for scale=3).
            raw_crop = frame[crop_y : crop_y + sample_h, crop_x : crop_x + sample_w]
            lr = cv2.resize(raw_crop, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
            if _scene_params is not None:
                lr = _apply_degrade_params(lr, _scene_params)
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


def _get_video_dimensions(video_path: str) -> Tuple[int, int]:
    """Return ``(width, height)`` of the first video stream, or ``(0, 0)`` on failure.

    Used by :func:`extract_and_save_streaming_dual` to decide whether the
    source resolution is large enough for the 4K dual-stream pipeline.
    """
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                video_path,
            ],
            stderr=subprocess.DEVNULL,
            timeout=15,
        ).decode(errors="replace").strip()
        parts = out.split(",")
        if len(parts) >= 2:
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return 0, 0


def _append_ffmpeg_log(
    base_dir: str,
    video_path: str,
    stderr_lines: List[str],
    pipeline_label: str = "",
) -> None:
    """Append FFmpeg stderr output to ``<base_dir>/ffmpeg_errors.log``.

    The file is created on first use and appended to on subsequent calls so
    that errors from all processed videos accumulate in a single place.  Each
    entry is preceded by a header with the current timestamp and the source
    video path so that entries are easy to correlate with the generator log.

    This function is a no-op when *stderr_lines* is empty.

    Args:
        base_dir:       Root dataset output directory.  The log file is
                        written as ``<base_dir>/ffmpeg_errors.log``.
        video_path:     Path to the source video (written to the header line).
        stderr_lines:   Lines collected from FFmpeg's stderr pipe.
        pipeline_label: Optional human-readable pipeline description included
                        in the header (e.g. ``"CPU-only [HDR]"``).
    """
    if not stderr_lines:
        return
    try:
        from datetime import datetime as _dt
        log_path = os.path.join(base_dir, "ffmpeg_errors.log")
        os.makedirs(base_dir, exist_ok=True)
        sep = "=" * 80
        ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
        header = (
            f"{sep}\n"
            f"[{ts}] {video_path}\n"
        )
        if pipeline_label:
            header += f"Pipeline: {pipeline_label}\n"
        header += f"FFmpeg stderr ({len(stderr_lines)} lines):\n"
        body = "\n".join(f"  {ln}" for ln in stderr_lines)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(header + body + "\n")
    except Exception:
        pass  # never let log I/O crash the extraction


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
    center_snap_seconds: float = 0.0,
    stream_width: int = STREAM_WIDTH,
    stream_height: int = STREAM_HEIGHT,
) -> Dict[str, int]:
    """
    Stream the video once and save patches as frames pass through the buffer.

    A single FFmpeg process reads the video linearly (no ``-ss`` seeking).
    Frames are piped as raw BGR24 data at *stream_width* × *stream_height*
    (default 1920×1080).  A rolling dictionary buffer keeps the last
    ``n_frames`` decoded frames in memory.  When a target centre frame has
    been decoded and all ``n_frames`` of its window are in the buffer, the
    patch is created and saved immediately.

    The stream is terminated early once the last needed frame has been read.

    Args:
        video_path:          Path to input video.
        assignments:         Output of :func:`build_assignments_per_category`.
        n_frames:            Frames per patch window (default 7).
        format_config:       ``{category: {format_name: {'gt_size': …, 'lr_size': …}}}``.
        base_dir:            Root dataset output directory.
        fps:                 Video frame rate.
        logger:              Optional logger instance.
        is_interesting_fn:   Optional callable ``(patch: np.ndarray) -> bool`` for
                             quality gating.  When provided, random crops are re-tried
                             up to 5 times before falling back to a centre crop.
        is_black_frame_fn:   Optional callable ``(frame: np.ndarray) -> bool``
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
        progress_fn:         Optional callable
                             ``(frames_examined: int,
                                patches_so_far: Dict[str, int],
                                raw_frames_read: int)``
                             invoked after *every* processed assignment (saved **or**
                             skipped).  ``raw_frames_read`` is the total number of
                             raw video frames decoded from the stream so far.
        use_cuda:            When ``True`` (default), enable CUDA hardware-accelerated
                             decoding if the local FFmpeg build supports it.  Falls
                             back to software decoding automatically when CUDA is not
                             available.
        nice_level:          CPU-priority adjustment passed to ``os.nice()`` for the
                             FFmpeg subprocess (default 10 = lower priority).  Has no
                             effect on non-Unix platforms.
        is_hdr:              Whether the source video uses an HDR transfer function
                             (PQ / HLG).  When ``True`` (default) the full HDR→SDR
                             tonemap chain is applied.  When ``False`` a lightweight
                             scale-only chain is used, avoiding incorrect
                             re-linearisation of SDR gamma that would cause
                             overexposure.
        degrade_cfg:         Optional degradation config dict forwarded to
                             :func:`create_patch_pair` / :func:`degrade_lr_frame`.
                             When ``None`` no LR degradation is applied.  Populate
                             from the ``quality`` section of the generator config
                             (keys: ``lr_degrade_prob``, ``lr_dark_boost``, etc.).
        center_snap_seconds: Tolerance in seconds for cross-category center snapping
                             (default 0.0 = disabled).  When > 0, assignments whose
                             center frame indices lie within
                             ``±round(fps * center_snap_seconds)`` frames of each
                             other are unified to a shared representative center.
                             Useful when external code generates near-duplicate centers;
                             has no benefit (and halves the SPS metric) when
                             ``build_assignments_per_category`` is used, because that
                             function already places each category on its own
                             evenly-spaced grid.
        stream_width:        Width of the decoded frame piped from FFmpeg (default
                             ``STREAM_WIDTH`` = 1920).  Pass ``STREAM_4K_WIDTH``
                             (3840) to stream at 4K for the 720/720_169 formats.
        stream_height:       Height of the decoded frame (default ``STREAM_HEIGHT``
                             = 1080).  Pass ``STREAM_4K_HEIGHT`` (2160) for 4K.

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

    # --- Optional cross-category center snapping --------------------------
    if center_snap_seconds > 0.0:
        n_before = len(assignments)
        unique_before = len({idx for idx, _, _ in assignments})
        snapped_asgn = snap_assignments_to_centers(
            assignments, fps=fps, tol_seconds=center_snap_seconds
        )
        unique_after = len({idx for idx, _, _ in snapped_asgn})
        _snap_tol_frames = max(1, int(round(fps * center_snap_seconds)))
        _log(
            f"🔗 Center snapping: {n_before} assignments, "
            f"{unique_before} unique centers → {unique_after} unique centers "
            f"(tol={center_snap_seconds:.2f}s / {_snap_tol_frames} frames)"
        )
    else:
        snapped_asgn = assignments  # snapping disabled — use as-is

    sorted_asgn = sorted(snapped_asgn, key=lambda x: x[0])

    # Build mapping: center_frame_idx → [(category, format_name), …]
    center_map: Dict[int, List[Tuple[str, str]]] = {}
    for frame_idx, category, fmt_name in sorted_asgn:
        center_map.setdefault(frame_idx, []).append((category, fmt_name))

    pending_centers: List[int] = sorted(center_map.keys())
    last_needed: int = pending_centers[-1] + half if pending_centers else 0

    # --- Build the exact set of frames that will ever be needed -----------
    # Each assignment requires a contiguous window [center-half … center+half].
    # We collect every frame index in any such window, sort them, and merge
    # adjacent indices into contiguous ranges.  These ranges are then passed
    # to FFmpeg as a `select` filter expression so that the expensive tonemap
    # and scale stages run *only* on the frames Python will actually use.
    #
    # For a typical 10-minute 24fps video with 100 scene assignments and
    # n_frames=7, this reduces filter-chain CPU work from ~14 400 frames
    # to ~700 frames (≈5%).
    _all_needed: List[int] = sorted({
        fi
        for c in pending_centers
        for fi in range(max(0, c - half), c + half + 1)
    })
    # Merge into contiguous ranges for a compact select expression.
    _select_ranges: List[Tuple[int, int]] = []
    if _all_needed:
        _rs, _re = _all_needed[0], _all_needed[0]
        for _f in _all_needed[1:]:
            if _f == _re + 1:
                _re = _f
            else:
                _select_ranges.append((_rs, _re))
                _rs = _re = _f
        _select_ranges.append((_rs, _re))

    # FFmpeg's filtergraph parser treats commas as filter separators, so the
    # commas inside between(n,a,b) must be escaped with a backslash so they
    # are not misread as filter boundaries.
    _select_expr: str = "+".join(
        f"between(n\\,{s}\\,{e})" for s, e in _select_ranges
    )

    _select_pct = (
        100.0 * len(_all_needed) / (last_needed + 1) if last_needed >= 0 else 100.0
    )

    # --- Pre-compute per-video constants ----------------------------------
    # video_stem and output dir paths are the same for every patch in this
    # video — compute them once to avoid Path() and os.makedirs overhead in
    # the hot decode loop.
    _video_stem: str = Path(video_path).stem
    _output_dirs_cache: Dict[Tuple[str, str], Dict[str, str]] = {}
    for _, _cat, _fmt in sorted_asgn:
        _key = (_cat, _fmt)
        if _key not in _output_dirs_cache:
            _dirs = get_output_dirs_for_format(base_dir, _cat, _fmt, n_frames)
            for _d in _dirs.values():
                os.makedirs(_d, exist_ok=True)
            _output_dirs_cache[_key] = _dirs

    # --- Async PNG write queue --------------------------------------------
    # Patch writing is off-loaded to background threads so that disk I/O
    # overlaps with FFmpeg decode.  Use 2 writer threads to fill both GT and
    # LR paths in parallel.  A bounded queue provides back-pressure when the
    # disk is slower than the CPU.
    _png_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    _write_queue: queue.Queue = queue.Queue(maxsize=256)

    def _write_worker() -> None:
        while True:
            item = _write_queue.get()
            if item is None:
                _write_queue.task_done()
                break
            gt_img, lr_img, gt_p, lr_p = item
            try:
                cv2.imwrite(gt_p, gt_img, _png_params)
                cv2.imwrite(lr_p, lr_img, _png_params)
            except Exception as _exc:
                if logger:
                    logger.warning(f"[write_worker] Failed to write patch: {_exc!r}")
            _write_queue.task_done()

    _n_write_threads = 4
    _write_threads = [
        threading.Thread(target=_write_worker, daemon=True)
        for _ in range(_n_write_threads)
    ]
    for _t in _write_threads:
        _t.start()

    # Build FFmpeg command.
    #
    # Pipeline tier is chosen by build_vf_filter() based on is_hdr and
    # available CUDA capabilities:
    #
    #  HDR source  + full-GPU   → tonemap_cuda + scale_cuda + hwdownload
    #  HDR source  + scale-GPU  → scale_cuda + hwdownload (p010) + zscale+tonemap
    #  HDR source  + CPU-only   → zscale + tonemap(reinhard) + scale (bilinear)
    #  SDR source  + scale-GPU  → scale_cuda + hwdownload (plain scale)
    #  SDR source  + CPU-only   → scale bilinear (software, no linearisation)
    _use_cuda = use_cuda and cuda_available()
    _full_gpu  = _use_cuda and is_hdr and tonemap_cuda_available()
    _scale_gpu = _use_cuda and (not _full_gpu) and scale_cuda_available()

    vf_filter = build_vf_filter(
        is_hdr=is_hdr, use_cuda=use_cuda,
        width=stream_width, height=stream_height,
    )

    # --- Inject select filter to skip unused frames in the filter chain ---
    # The `select` filter passes only the frames in `_select_expr` to
    # downstream filter stages.  Non-selected frames are still decoded
    # (unavoidable for H.264/H.265 inter-frame prediction) but bypass the
    # expensive scale/tonemap/zscale stages entirely.
    #
    # Placement depends on the pipeline tier:
    #   CPU path        → select goes at the very start of the filter chain.
    #   Hybrid GPU/CPU  → GPU scale runs first (cheap, already on GPU);
    #                     select is inserted right after hwdownload so the
    #                     expensive CPU tonemap only runs on needed frames.
    #   Full-GPU        → all stages are on GPU; select is placed after the
    #                     final hwdownload to cut pipe bandwidth.
    if _select_expr:
        if _full_gpu:
            # Full-GPU: insert select right before the final format=bgr24
            # (after hwdownload+scale=iw:ih+format=yuv420p — all GPU work
            # is already done, select avoids the final CPU format conversion
            # and the pipe write for unwanted frames).
            _marker = ",format=bgr24"
            if _marker in vf_filter:
                vf_filter = vf_filter.replace(
                    _marker, f",select={_select_expr},format=bgr24", 1
                )
            else:
                _log("⚠️  Could not inject select into full-GPU filter chain — falling back to prepend")
                vf_filter = f"select={_select_expr},{vf_filter}"
        elif _scale_gpu:
            # Hybrid scale-GPU + CPU tonemap: insert select after hwdownload
            # so the CPU-side zscale/tonemap only processes needed frames.
            _marker = "hwdownload,"
            if _marker in vf_filter:
                vf_filter = vf_filter.replace(
                    _marker, f"hwdownload,select={_select_expr},", 1
                )
            else:
                _log("⚠️  Could not inject select into hybrid GPU filter chain — falling back to prepend")
                vf_filter = f"select={_select_expr},{vf_filter}"
        else:
            # CPU-only (or decode-only CUDA): prepend select so tonemap/scale
            # runs on needed frames only.
            vf_filter = f"select={_select_expr},{vf_filter}"

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
        pipeline_label = f"CPU-only {'tonemap/reinhard' if is_hdr else 'scale/bilinear'} [{hdr_label}]"

    _log(
        f"🎬 Streaming extractor: {len(sorted_asgn)} assignments, "
        f"{len(pending_centers)} unique centers, "
        f"last frame needed: {last_needed}, "
        f"stream={stream_width}×{stream_height}, "
        f"pipeline={pipeline_label}, nice={nice_level}"
    )
    _log(
        f"🎯 Frame selection: {len(_all_needed)} frames needed "
        f"({_select_pct:.1f}% of {last_needed + 1} decoded) "
        f"in {len(_select_ranges)} ranges — "
        f"filter-chain CPU reduced proportionally"
    )

    # Write the filter chain to a temp file so that a long _select_expr
    # (thousands of between() terms for a dense assignment list) never
    # exceeds the OS ARG_MAX limit (~2 MB on Linux) and causes execve() to
    # fail with E2BIG.  -filter_complex_script reads from a file and has no
    # length restriction.  We wrap the vf-style filter in a minimal
    # filter_complex graph: [0:v]<filter>[vout], then map [vout] to output.
    #
    # Initialise all variables that the finally block references to safe
    # defaults so that a failure in mkstemp or Popen cannot produce a
    # NameError in the cleanup path.
    _fc_script_path: Optional[str] = None
    process = None
    stderr_thread: Optional[threading.Thread] = None
    stderr_lines: List[str] = []
    selected_idx: int = 0
    _t_start: Optional[float] = None
    _log_interval: int = 50

    frame_bytes: int = stream_width * stream_height * 3
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

    try:
        _fc_fd, _fc_script_path = tempfile.mkstemp(suffix=".txt", prefix="dsg_fc_")
        with os.fdopen(_fc_fd, "w", encoding="utf-8") as _fc_fh:
            _fc_fh.write(f"[0:v]{vf_filter}[vout]")

        # Select the right filter-file and vsync options depending on the
        # installed FFmpeg version.  FFmpeg 5+ deprecated -filter_complex_script
        # (replaced by -/filter_complex) and -vsync (replaced by -fps_mode).
        # -vsync 0 / -fps_mode passthrough is CRITICAL: without it, FFmpeg fills
        # PTS gaps left by the select filter with duplicated frames, so Python
        # would read only frames from the very start of the video.
        _ffmpeg_ver = _get_ffmpeg_major_version()
        _fc_args = (
            ["-/filter_complex", _fc_script_path]
            if _ffmpeg_ver >= 5
            else ["-filter_complex_script", _fc_script_path]
        )
        _vsync_args = (
            ["-fps_mode", "passthrough"]
            if _ffmpeg_ver >= 5
            else ["-vsync", "0"]
        )

        cmd = [
            "ffmpeg",
            "-threads", "0",
            "-filter_threads", "0",
            "-loglevel", "warning",
            *hw_args,
            "-probesize", "100M",
            "-analyzeduration", "100M",
            "-i", video_path,
            *_fc_args,
            "-map", "[vout]",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            *_vsync_args,
            "pipe:1",
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _set_nice(process.pid)

        # Drain stderr in a background thread so the pipe never blocks the writer.
        # The collected lines are logged if FFmpeg produces no frames (crash/error).
        def drain_stderr(pipe: "subprocess.IO[bytes]") -> None:
            for raw in pipe:
                stderr_lines.append(raw.decode(errors="replace").rstrip())
            pipe.close()

        stderr_thread = threading.Thread(
            target=drain_stderr, args=(process.stderr,), daemon=True
        )
        stderr_thread.start()
        # `selected_idx` tracks our position in `_all_needed`.  FFmpeg (via the
        # `select` filter) only outputs the frames in that list, in sorted order,
        # so each pipe read maps directly to `_all_needed[selected_idx]`.

        while pending_idx < len(pending_centers):
            raw = process.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                _log("⚠️  Video stream ended before all selected frames were received")
                break

            # Start the clock on the very first frame so FFmpeg startup time
            # (device init, demux, codec open) is excluded from the FPS figure.
            if _t_start is None:
                _t_start = time.monotonic()

            # Guard against FFmpeg producing more frames than the select
            # expression requested (shouldn't happen, but avoids IndexError).
            if selected_idx >= len(_all_needed):
                _log("⚠️  FFmpeg produced more frames than selected — stopping")
                break

            # Map this pipe read to its actual video frame index.
            actual_frame: int = _all_needed[selected_idx]
            selected_idx += 1

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (stream_height, stream_width, 3)
            ).copy()
            buffer[actual_frame] = frame

            # Evict frames that are no longer needed by any pending assignment.
            # The earliest window we still need starts at pending_center - half.
            min_keep = max(0, pending_centers[pending_idx] - half)
            for old_idx in [k for k in buffer if k < min_keep]:
                del buffer[old_idx]

            # Satisfy pending assignments whose full window is now in the buffer
            while pending_idx < len(pending_centers):
                center = pending_centers[pending_idx]
                if actual_frame < center + half:
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
                                dirs = _output_dirs_cache[(category, fmt_name)]
                                patch_name = f"{_video_stem}_{int(ts * 1000):08d}.png"
                                _write_queue.put((
                                    gt, lr,
                                    os.path.join(dirs["gt"], patch_name),
                                    os.path.join(dirs["lr"], patch_name),
                                ))
                                patches_created[category] = (
                                    patches_created.get(category, 0) + 1
                                )

                    if progress_fn is not None:
                        # 3rd arg = actual video frame index (same semantics as
                        # the previous raw_frames_read / current_frame counter —
                        # monotonically increasing video frame number).
                        progress_fn(frames_examined, dict(patches_created), actual_frame)

                pending_idx += 1

            # Periodic throughput log.
            # SPS = scene-sets completed per second (assignments processed / s).
            # sel/s = selected frames per second (frames FFmpeg actually piped).
            if _t_start is not None and selected_idx % _log_interval == 0:
                _elapsed = time.monotonic() - _t_start
                if _elapsed > 0:
                    _sel_fps = selected_idx / _elapsed
                    _sps_actual = frames_examined / _elapsed
                    _log(
                        f"  📊 sel {selected_idx:>5}/{len(_all_needed)}  "
                        f"sel/s {_sel_fps:>6.1f}  SPS {_sps_actual:>6.2f}  "
                        f"(scenes: {frames_examined})"
                    )

    finally:
        if process is not None:
            try:
                process.stdout.close()
            except Exception:
                pass
            process.kill()
            process.wait()
        if stderr_thread is not None:
            stderr_thread.join(timeout=2)
        # Always persist FFmpeg stderr to the log file so that filter-chain
        # errors, codec warnings and hw-accel failures are visible after the
        # run even when they didn't prevent frame output.
        _append_ffmpeg_log(base_dir, video_path, stderr_lines, pipeline_label)
        # Also echo to the logger when no frames were produced (most useful
        # for diagnosing filter-chain errors interactively).
        if selected_idx == 0 and stderr_lines:
            _log("FFmpeg stderr (last 20 lines):")
            for _line in stderr_lines[-20:]:
                _log(f"  [ffmpeg] {_line}")

        # Drain the async write queue — wait for all pending PNG writes to
        # finish before returning so that patches_created is accurate.
        for _ in _write_threads:
            _write_queue.put(None)  # poison pill per worker
        for _t in _write_threads:
            _t.join()

        # Remove the temporary filter script.
        if _fc_script_path is not None:
            try:
                os.unlink(_fc_script_path)
            except Exception:
                pass

    # GPU pipeline produced zero frames — most likely a runtime hw-accel failure
    # (e.g. CUDA driver mismatch, scale_cuda format-negotiation bug, or FFmpeg
    # silently falling back to software decode while the filtergraph still
    # contains scale_cuda / hwdownload GPU filters).
    # Retry transparently with the CPU-only pipeline so extraction still
    # completes, rather than silently returning 0 patches.
    if selected_idx == 0 and (_full_gpu or _scale_gpu):
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
            center_snap_seconds=center_snap_seconds,
            stream_width=stream_width,
            stream_height=stream_height,
        )

    total = sum(patches_created.values())
    _elapsed_total = (
        (time.monotonic() - _t_start) if _t_start is not None else 0.0
    )
    if _elapsed_total > 0:
        _sps_final = frames_examined / _elapsed_total
        _sel_fps_final = selected_idx / _elapsed_total
        _log(
            f"✓ Streaming extraction done: {total} patches saved, "
            f"{frames_examined} assignments examined, "
            f"{selected_idx}/{len(_all_needed)} selected frames received — "
            f"sel/s {_sel_fps_final:.1f}  SPS {_sps_final:.2f}"
        )
    else:
        _log(
            f"✓ Streaming extraction done: {total} patches saved, "
            f"{frames_examined} assignments examined"
        )
    return patches_created




def extract_and_save_streaming_dual(
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
    center_snap_seconds: float = 0.0,
) -> Dict[str, int]:
    """
    Single 4K FFmpeg pass producing both 720/720_169 and 540 format patches.

    FFmpeg streams BGR24 frames at 3840 x 2160.  For every decoded frame two
    rolling buffers are maintained:

    * ``buffer_4k`` - the raw 4K frame, used by ``FORMATS_4K_STREAM``
      (720, large_720, 720_169, medium_169).
    * ``buffer_hd`` - the same frame downscaled **in Python** to 1920 x 1080
      with ``cv2.INTER_LANCZOS4``, used by ``FORMATS_HD_STREAM``
      (540, small_540).

    This avoids a second FFmpeg decode pass entirely: the HD frame is derived
    in-memory from the already tone-mapped 4K frame via a single OpenCV resize.
    The patch-creation window for each format is drawn from the correct buffer,
    so crop geometry, quality gating, and LR degradation (``degrade_cfg``) work
    exactly as in :func:`extract_and_save_streaming_distributed`.

    LR degradation (DVD artefacts: noise, blur, JPEG round-trip) is applied by
    :func:`create_patch_pair` / :func:`degrade_lr_frame` as usual --
    ``degrade_cfg`` is forwarded unchanged regardless of which buffer (4K or
    HD) was used as the source window.

    Falls back to :func:`extract_and_save_streaming_distributed` (single 1080p
    pass, all formats) when the source video is smaller than 4K.

    Args:
        video_path:          Path to input video.
        assignments:         Output of :func:`build_assignments_per_category`.
        n_frames:            Frames per patch window (default 7).
        format_config:       ``{category: {format_name: {'gt_size': ..., 'lr_size': ...}}}``.
        base_dir:            Root dataset output directory.
        fps:                 Video frame rate.
        logger:              Optional logger instance.
        is_interesting_fn:   Optional quality-gate callable ``(patch) -> bool``.
        is_black_frame_fn:   Optional black-frame filter callable.
        progress_fn:         Optional progress callback
                             ``(frames_examined, patches_so_far, raw_frames_read)``.
        use_cuda:            Enable CUDA hardware acceleration when available.
        nice_level:          CPU priority for the FFmpeg subprocess.
        is_hdr:              Whether the source uses an HDR transfer function.
        degrade_cfg:         Optional LR degradation config (DVD artefacts).
                             Forwarded to :func:`degrade_lr_frame` unchanged.
        center_snap_seconds: Cross-category center snapping tolerance (seconds).

    Returns:
        ``{category: patches_saved_count}``
    """

    def _log(msg: str) -> None:
        if logger:
            logger.info(msg)

    # ------------------------------------------------------------------
    # Fallback: source video smaller than 4K -> single 1080p pass for all
    # ------------------------------------------------------------------
    vid_w, vid_h = _get_video_dimensions(video_path)
    if vid_w > 0 and (vid_w < STREAM_4K_WIDTH or vid_h < STREAM_4K_HEIGHT):
        _log(
            f"Warning: Video kleiner als 4K ({vid_w}x{vid_h}), "
            f"fallback auf Single-Stream 1080p"
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
            use_cuda=use_cuda,
            nice_level=nice_level,
            is_hdr=is_hdr,
            degrade_cfg=degrade_cfg,
            center_snap_seconds=center_snap_seconds,
        )

    # ------------------------------------------------------------------
    # Common setup (mirrors extract_and_save_streaming_distributed)
    # ------------------------------------------------------------------
    _black_fn: Callable[[np.ndarray], bool] = (
        is_black_frame_fn if is_black_frame_fn is not None else is_black_frame
    )

    if not assignments:
        return {}

    half = n_frames // 2

    if center_snap_seconds > 0.0:
        n_before = len(assignments)
        unique_before = len({idx for idx, _, _ in assignments})
        snapped_asgn = snap_assignments_to_centers(
            assignments, fps=fps, tol_seconds=center_snap_seconds
        )
        unique_after = len({idx for idx, _, _ in snapped_asgn})
        _snap_tol_frames = max(1, int(round(fps * center_snap_seconds)))
        _log(
            f"Center snapping: {n_before} assignments, "
            f"{unique_before} unique centers -> {unique_after} unique centers "
            f"(tol={center_snap_seconds:.2f}s / {_snap_tol_frames} frames)"
        )
    else:
        snapped_asgn = assignments

    sorted_asgn = sorted(snapped_asgn, key=lambda x: x[0])

    center_map: Dict[int, List[Tuple[str, str]]] = {}
    for frame_idx, category, fmt_name in sorted_asgn:
        center_map.setdefault(frame_idx, []).append((category, fmt_name))

    pending_centers: List[int] = sorted(center_map.keys())
    last_needed: int = pending_centers[-1] + half if pending_centers else 0

    _all_needed: List[int] = sorted({
        fi
        for c in pending_centers
        for fi in range(max(0, c - half), c + half + 1)
    })
    _select_ranges: List[Tuple[int, int]] = []
    if _all_needed:
        _rs, _re = _all_needed[0], _all_needed[0]
        for _f in _all_needed[1:]:
            if _f == _re + 1:
                _re = _f
            else:
                _select_ranges.append((_rs, _re))
                _rs = _re = _f
        _select_ranges.append((_rs, _re))

    _select_expr: str = "+".join(
        f"between(n\\,{s}\\,{e})" for s, e in _select_ranges
    )
    _select_pct = (
        100.0 * len(_all_needed) / (last_needed + 1) if last_needed >= 0 else 100.0
    )

    _video_stem: str = Path(video_path).stem
    _output_dirs_cache: Dict[Tuple[str, str], Dict[str, str]] = {}
    for _, _cat, _fmt in sorted_asgn:
        _key = (_cat, _fmt)
        if _key not in _output_dirs_cache:
            _dirs = get_output_dirs_for_format(base_dir, _cat, _fmt, n_frames)
            for _d in _dirs.values():
                os.makedirs(_d, exist_ok=True)
            _output_dirs_cache[_key] = _dirs

    _png_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    _write_queue: queue.Queue = queue.Queue(maxsize=256)

    def _write_worker() -> None:
        while True:
            item = _write_queue.get()
            if item is None:
                _write_queue.task_done()
                break
            gt_img, lr_img, gt_p, lr_p = item
            try:
                cv2.imwrite(gt_p, gt_img, _png_params)
                cv2.imwrite(lr_p, lr_img, _png_params)
            except Exception as _exc:
                if logger:
                    logger.warning(f"[write_worker] Failed to write patch: {_exc!r}")
            _write_queue.task_done()

    _n_write_threads = 4
    _write_threads = [
        threading.Thread(target=_write_worker, daemon=True)
        for _ in range(_n_write_threads)
    ]
    for _t in _write_threads:
        _t.start()

    # ------------------------------------------------------------------
    # FFmpeg filter chain -- 4K output, select-filter injected as usual
    # ------------------------------------------------------------------
    _use_cuda = use_cuda and cuda_available()
    _full_gpu  = _use_cuda and is_hdr and tonemap_cuda_available()
    _scale_gpu = _use_cuda and (not _full_gpu) and scale_cuda_available()

    vf_filter = build_vf_filter(
        is_hdr=is_hdr, use_cuda=use_cuda,
        width=STREAM_4K_WIDTH, height=STREAM_4K_HEIGHT,
    )

    if _select_expr:
        if _full_gpu:
            _marker = ",format=bgr24"
            if _marker in vf_filter:
                vf_filter = vf_filter.replace(
                    _marker, f",select={_select_expr},format=bgr24", 1
                )
            else:
                _log("Warning: Could not inject select into full-GPU filter chain -- prepending")
                vf_filter = f"select={_select_expr},{vf_filter}"
        elif _scale_gpu:
            _marker = "hwdownload,"
            if _marker in vf_filter:
                vf_filter = vf_filter.replace(
                    _marker, f"hwdownload,select={_select_expr},", 1
                )
            else:
                _log("Warning: Could not inject select into hybrid GPU filter chain -- prepending")
                vf_filter = f"select={_select_expr},{vf_filter}"
        else:
            vf_filter = f"select={_select_expr},{vf_filter}"

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
        pipeline_label = f"CPU-only {'tonemap/reinhard' if is_hdr else 'scale/bilinear'} [{hdr_label}]"

    _log(
        f"Dual-buffer extractor (single 4K pass): {len(sorted_asgn)} assignments, "
        f"{len(pending_centers)} unique centers, last frame: {last_needed}, "
        f"pipeline={pipeline_label}, nice={nice_level}"
    )
    _log(
        f"Frame selection: {len(_all_needed)} frames needed "
        f"({_select_pct:.1f}% of {last_needed + 1} decoded) "
        f"in {len(_select_ranges)} ranges"
    )
    _log(
        f"  4K buffer ({STREAM_4K_WIDTH}x{STREAM_4K_HEIGHT}) "
        f"-> formats {sorted(FORMATS_4K_STREAM)}"
    )
    _log(
        f"  HD buffer (Python LANCZOS4 {STREAM_4K_WIDTH}x{STREAM_4K_HEIGHT}"
        f" -> {STREAM_HD_WIDTH}x{STREAM_HD_HEIGHT}) -> formats {sorted(FORMATS_HD_STREAM)}"
    )

    # ------------------------------------------------------------------
    # Pre-initialise all finally-referenced variables
    # ------------------------------------------------------------------
    _fc_script_path: Optional[str] = None
    _process = None
    _stderr_thread: Optional[threading.Thread] = None
    stderr_lines: List[str] = []
    selected_idx: int = 0
    _t_start: Optional[float] = None
    _log_interval: int = 50

    frame_bytes_4k: int = STREAM_4K_WIDTH * STREAM_4K_HEIGHT * 3
    patches_created: Dict[str, int] = {}

    # Dual rolling buffers.
    # buffer_4k and buffer_hd are always populated together (same set of keys).
    buffer_4k: Dict[int, np.ndarray] = {}
    buffer_hd: Dict[int, np.ndarray] = {}
    pending_idx: int = 0
    frames_examined: int = 0

    def _set_nice(pid: int) -> None:
        if nice_level == 0 or _sys.platform == "win32":
            return
        try:
            import psutil as _psutil
            _psutil.Process(pid).nice(nice_level)
        except Exception:
            pass

    try:
        _fc_fd, _fc_script_path = tempfile.mkstemp(suffix=".txt", prefix="dsg_fc_")
        with os.fdopen(_fc_fd, "w", encoding="utf-8") as _fc_fh:
            _fc_fh.write(f"[0:v]{vf_filter}[vout]")

        # Select the right filter-file and vsync options depending on the
        # installed FFmpeg version.  FFmpeg 5+ deprecated -filter_complex_script
        # (replaced by -/filter_complex) and -vsync (replaced by -fps_mode).
        # -vsync 0 / -fps_mode passthrough is CRITICAL: without it, FFmpeg fills
        # PTS gaps left by the select filter with duplicated frames, so Python
        # would read only frames from the very start of the video.
        _ffmpeg_ver = _get_ffmpeg_major_version()
        _fc_args = (
            ["-/filter_complex", _fc_script_path]
            if _ffmpeg_ver >= 5
            else ["-filter_complex_script", _fc_script_path]
        )
        _vsync_args = (
            ["-fps_mode", "passthrough"]
            if _ffmpeg_ver >= 5
            else ["-vsync", "0"]
        )

        cmd = [
            "ffmpeg",
            "-threads", "0",
            "-filter_threads", "0",
            "-loglevel", "warning",
            *hw_args,
            "-probesize", "100M",
            "-analyzeduration", "100M",
            "-i", video_path,
            *_fc_args,
            "-map", "[vout]",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            *_vsync_args,
            "pipe:1",
        ]

        _process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _set_nice(_process.pid)

        def drain_stderr(pipe: "subprocess.IO[bytes]") -> None:
            for raw in pipe:
                stderr_lines.append(raw.decode(errors="replace").rstrip())
            pipe.close()

        _stderr_thread = threading.Thread(
            target=drain_stderr, args=(_process.stderr,), daemon=True
        )
        _stderr_thread.start()

        while pending_idx < len(pending_centers):
            raw = _process.stdout.read(frame_bytes_4k)
            if len(raw) < frame_bytes_4k:
                _log("Warning: Video stream ended before all selected frames were received")
                break

            if _t_start is None:
                _t_start = time.monotonic()

            if selected_idx >= len(_all_needed):
                _log("Warning: FFmpeg produced more frames than selected -- stopping")
                break

            actual_frame: int = _all_needed[selected_idx]
            selected_idx += 1

            # Decode 4K frame and store in the 4K rolling buffer.
            frame_4k = np.frombuffer(raw, dtype=np.uint8).reshape(
                (STREAM_4K_HEIGHT, STREAM_4K_WIDTH, 3)
            ).copy()
            buffer_4k[actual_frame] = frame_4k

            # Downscale to 1080p in Python and store in the HD rolling buffer.
            # INTER_LANCZOS4 matches the quality of FFmpeg 'flags=lanczos' and
            # gives the best downscale quality for 540-format crops.
            buffer_hd[actual_frame] = cv2.resize(
                frame_4k, (STREAM_HD_WIDTH, STREAM_HD_HEIGHT),
                interpolation=cv2.INTER_LANCZOS4,
            )

            # Evict frames no longer needed from both buffers together.
            min_keep = max(0, pending_centers[pending_idx] - half)
            for old_idx in [k for k in buffer_4k if k < min_keep]:
                del buffer_4k[old_idx]
                del buffer_hd[old_idx]

            # Satisfy pending assignments whose full window is now in the buffers.
            while pending_idx < len(pending_centers):
                center = pending_centers[pending_idx]
                if actual_frame < center + half:
                    break

                # Build the 4K window.  When it is complete the HD window is
                # also guaranteed complete (same keys in both buffers).
                window_4k_frames: List[np.ndarray] = []
                for fi in range(center - half, center + half + 1):
                    frm = buffer_4k.get(max(0, fi))
                    if frm is None:
                        break
                    window_4k_frames.append(frm)

                if len(window_4k_frames) == n_frames:
                    ts = center / fps
                    center_raw_4k = window_4k_frames[n_frames // 2]
                    frames_examined += 1

                    if _black_fn(center_raw_4k):
                        _log(f"  frame {center} skipped (black frame)")
                    else:
                        # Build the HD window from the pre-computed buffer.
                        window_hd_frames: List[np.ndarray] = [
                            buffer_hd[max(0, fi)]
                            for fi in range(center - half, center + half + 1)
                        ]

                        for category, fmt_name in center_map[center]:
                            cfg = format_config.get(category, {}).get(fmt_name, {})
                            if not cfg:
                                continue

                            # 540/small_540 use the 1080p-downscaled window so
                            # the source resolution matches the crop target.
                            # All other formats use the full 4K window.
                            window = (
                                window_hd_frames
                                if fmt_name in FORMATS_HD_STREAM
                                else window_4k_frames
                            )

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
                                dirs = _output_dirs_cache[(category, fmt_name)]
                                patch_name = f"{_video_stem}_{int(ts * 1000):08d}.png"
                                _write_queue.put((
                                    gt, lr,
                                    os.path.join(dirs["gt"], patch_name),
                                    os.path.join(dirs["lr"], patch_name),
                                ))
                                patches_created[category] = (
                                    patches_created.get(category, 0) + 1
                                )

                    if progress_fn is not None:
                        progress_fn(frames_examined, dict(patches_created), actual_frame)

                pending_idx += 1

            if _t_start is not None and selected_idx % _log_interval == 0:
                _elapsed = time.monotonic() - _t_start
                if _elapsed > 0:
                    _sel_fps = selected_idx / _elapsed
                    _sps_actual = frames_examined / _elapsed
                    _log(
                        f"  sel {selected_idx:>5}/{len(_all_needed)}  "
                        f"sel/s {_sel_fps:>6.1f}  SPS {_sps_actual:>6.2f}  "
                        f"(scenes: {frames_examined})"
                    )

    finally:
        if _process is not None:
            try:
                _process.stdout.close()
            except Exception:
                pass
            _process.terminate()
            _process.wait()
        if _stderr_thread is not None:
            _stderr_thread.join(timeout=2)
        _append_ffmpeg_log(base_dir, video_path, stderr_lines, pipeline_label)
        if selected_idx == 0 and stderr_lines:
            _log("FFmpeg stderr (last 20 lines):")
            for _line in stderr_lines[-20:]:
                _log(f"  [ffmpeg] {_line}")
        for _ in _write_threads:
            _write_queue.put(None)
        for _t in _write_threads:
            _t.join()
        if _fc_script_path is not None:
            try:
                os.unlink(_fc_script_path)
            except Exception:
                pass

    # GPU pipeline produced zero frames -> retry CPU-only
    if selected_idx == 0 and (_full_gpu or _scale_gpu):
        _log("Warning: GPU pipeline produced no frames -- retrying with CPU-only pipeline")
        return extract_and_save_streaming_dual(
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
            degrade_cfg=degrade_cfg,
            center_snap_seconds=center_snap_seconds,
        )

    total = sum(patches_created.values())
    _elapsed_total = (
        (time.monotonic() - _t_start) if _t_start is not None else 0.0
    )
    if _elapsed_total > 0:
        _sps_final = frames_examined / _elapsed_total
        _sel_fps_final = selected_idx / _elapsed_total
        _log(
            f"Dual-buffer extraction done: {total} patches saved, "
            f"{frames_examined} assignments examined, "
            f"{selected_idx}/{len(_all_needed)} selected frames received -- "
            f"sel/s {_sel_fps_final:.1f}  SPS {_sps_final:.2f}"
        )
    else:
        _log(
            f"Dual-buffer extraction done: {total} patches saved, "
            f"{frames_examined} assignments examined"
        )
    return patches_created
