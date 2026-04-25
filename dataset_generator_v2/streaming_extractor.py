#!/usr/bin/env python3
"""
Streaming video extractor for dataset_generator_v2.

Replaces the previous per-timestamp ``-ss`` seek approach with a single
FFmpeg pass that streams the video linearly.  A rolling frame buffer
(default 7 frames) is maintained in memory; patches are written to disk
as their centre frame enters the buffer.

Performance optimisations (active by default)
---------------------------------------------
* **Opt 1 – Reduced stream resolution**: ``STREAM_OPT_WIDTH × STREAM_OPT_HEIGHT``
  (2304 × 1440) instead of native 4K.  Still large enough for 2× oversampled
  Lanczos4 crops for all GT families (1152×648, 960×540, 960×720).
* **Opt 2 – Bilinear FFmpeg scale**: ``flags=bilinear`` in the intermediate
  scale step; Python applies Lanczos4 on the actual patch crops anyway.
* **Opt 3 – yuv420p pipe**: 1.5 bytes/pixel instead of 3 bytes/pixel (bgr24),
  cutting pipe bandwidth by ~33 %.  Python converts with
  ``cv2.cvtColor(…, COLOR_YUV2BGR_I420)``.
* **Opt 4 – libplacebo HDR tonemap**: when ``--enable-libplacebo`` is present
  in the FFmpeg build a single shader-based pass replaces the 4-step
  zscale+tonemap chain (~2-4× faster for equivalent quality).

Combined Opt 1+3 alone reduce per-frame pipe data from 24.9 MB (3840×2160
BGR24) to ~5 MB (2304×1440 yuv420p) — roughly **5× less**.

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

extract_and_save_streaming_distributed()
    Single-stream entry point for all formats.  Launches one FFmpeg process,
    streams yuv420p frames at ``STREAM_OPT_WIDTH × STREAM_OPT_HEIGHT`` by
    default — large enough for ``create_patch_pair`` to apply 2× oversampled
    Lanczos4 crops for both the 1152×648 and 960×720 GT families.  Uses CUDA
    (``tonemap_cuda`` + ``scale_cuda`` or ``scale_cuda`` alone) when available;
    falls back to libplacebo (if present) or CPU zscale automatically.
    Passes the filter chain via a temp file, avoiding OS ARG_MAX limits.

extract_and_save_streaming_dual()
    Deprecated compatibility shim.  Forwards all arguments to
    :func:`extract_and_save_streaming_distributed` at optimised resolution.

create_patch_pair()
    Create a (GT, LR) patch pair from a sequence of frames.

save_patch_pair()
    Persist a (GT, LR) pair to the correct output directories.

FFmpeg error logging
--------------------
The streaming function writes all FFmpeg stderr output to
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
from typing import Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Import path setup – streaming_extractor lives inside dataset_generator_v2/
# ---------------------------------------------------------------------------
import sys as _sys
_sys.path.insert(0, os.path.dirname(__file__))
from utils.format_definitions import (
    get_output_dirs_for_format,
    get_synced_bucket_dirs,
    BUCKET_SIZE,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Output resolution after HDR→SDR tonemap (must match what OpenCV expects)
STREAM_WIDTH: int = 1920
STREAM_HEIGHT: int = 1080

# 4K stream — kept for backward compatibility / explicit callers.
STREAM_4K_WIDTH:  int = 3840
STREAM_4K_HEIGHT: int = 2160

# Optimised stream resolution (Opt 1).
# Minimum resolution that supports 2× oversampled Lanczos4 crops for every
# format in the current template set:
#   1152×648 crop  → needs 2×1152 = 2304 w,  2×648 = 1296 h
#   960×540  crop  → needs 2×960  = 1920 w,  2×540 = 1080 h
#   960×720  crop  → needs 2×960  = 1920 w,  2×720 = 1440 h
#   Resize formats → only need ≥ gt_size (trivially covered)
# → max(2304, 1920, 1920) × max(1296, 1080, 1440) = 2304 × 1440
#
# Pipe-bandwidth compared to 3840×2160 BGR24:
#   Old: 3840 × 2160 × 3      = 24.9 MB/frame
#   New: 2304 × 1440 × 1.5    =  4.98 MB/frame  (~5× less, combined Opt1+Opt3)
STREAM_OPT_WIDTH:  int = 2304
STREAM_OPT_HEIGHT: int = 1440

# ---------------------------------------------------------------------------
# Filter chains — two families: HDR→SDR tonemap and plain SDR pass-through.
#
# HDR chains handle HDR10 (SMPTE 2084 / BT.2020) and Dolby Vision P5/P8.
# SDR chains are used when the video is already standard-dynamic-range
# (BT.709 transfer) and no tone-mapping is needed.  Applying the HDR
# tonemap chain to an SDR source causes overexposure because zscale would
# linearise the already-gamma-encoded values a second time.
#
# All chains output yuv420p (Opt 3) — ~33 % less pipe bandwidth vs bgr24.
# Python converts yuv420p→BGR with cv2.cvtColor(…, COLOR_YUV2BGR_I420).
#
# The correct chain is selected per-video by build_vf_filter() based on the
# is_hdr flag returned by _get_video_metadata() / is_hdr_transfer().
# ---------------------------------------------------------------------------

# HDR→SDR: Software (CPU-only) fallback via zscale+tonemap.
# zscale reads tin from stream metadata → works for smpte2084, hlg, bt709.
# range=full: unambiguous 0-255 output for OpenCV.
# Performance notes:
#   - tonemap=reinhard is the fastest tone-mapper (simple x/(1+x) curve).
#   - flags=bilinear (Opt 2): faster than lanczos; Python does Lanczos4 on
#     the actual patch crops, so the FFmpeg resize only needs to be adequate.
#   - filter=bilinear in each zscale step speeds up any incidental resampling.
_TONEMAP_FILTER: str = (
    "zscale=t=linear:npl=100:filter=bilinear,"
    "format=gbrpf32le,"
    "zscale=p=bt709:filter=bilinear,"
    "tonemap=tonemap=reinhard:desat=0,"
    "zscale=t=bt709:m=bt709:range=full:filter=bilinear,"
    f"scale={STREAM_WIDTH}:{STREAM_HEIGHT}:flags=bilinear,"
    "format=yuv420p"
)

# HDR→SDR: libplacebo CPU path (Opt 4).
# Replaces the 4-step zscale+tonemap chain with a single GPU-shader pass on
# the CPU (via Vulkan/software fallback).  Requires FFmpeg built with
# --enable-libplacebo (Ubuntu 6.1.1-3ubuntu5 includes this).
# range=pc → full range (0-255), downscaler=bilinear → fast resize.
# libplacebo auto-detects source HDR metadata from stream properties.
_TONEMAP_FILTER_PLACEBO: str = (
    f"libplacebo=w={STREAM_WIDTH}:h={STREAM_HEIGHT}"
    ":colorspace=bt709:color_trc=bt709:color_primaries=bt709"
    ":tonemapping=mobius:range=pc:downscaler=bilinear,"
    "format=yuv420p"
)

# SDR pass-through: Software (CPU-only).
# flags=bilinear (Opt 2): adequate for the intermediate scale step.
_SDR_FILTER: str = (
    f"scale={STREAM_WIDTH}:{STREAM_HEIGHT}:flags=bilinear,"
    "format=yuv420p"
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
    "format=yuv420p"
)

# SDR pass-through: Hybrid GPU/CPU — scale on GPU, convert on CPU.
# interp_algo=bicubic: best quality available in scale_cuda (no lanczos).
_SDR_FILTER_SCALE_CUDA: str = (
    f"scale_cuda={STREAM_WIDTH}:{STREAM_HEIGHT}:interp_algo=bicubic,"
    "hwdownload,"
    "format=yuv420p"
)

# Full-GPU HDR→SDR tonemap filter chain.
# Requires FFmpeg built with --enable-cuda-nvcc / libnpp so that both
# tonemap_cuda and scale_cuda are available.
# Frames stay in GPU memory from decode through tonemap + scale;
# hwdownload copies only the final result to CPU.
# Use together with -hwaccel cuda -hwaccel_output_format cuda.
# Notes:
#   - interp_algo=bicubic — see _TONEMAP_FILTER_SCALE_CUDA comment above.
#   - tonemap_cuda outputs 8-bit NV12 CUDA frames; scale_cuda receives NV12
#     and outputs NV12.
#   - hwdownload (bare) + scale=iw:ih: same reasoning as above — scale breaks
#     the backward format negotiation, converting NV12→YUV420P in software.
#   - format=yuv420p: ensures planar yuv420p for the pipe (Opt 3).
_TONEMAP_FILTER_CUDA: str = (
    f"tonemap_cuda=tonemap=mobius:desat=0:peak=100,"
    f"scale_cuda={STREAM_WIDTH}:{STREAM_HEIGHT}:interp_algo=bicubic,"
    "hwdownload,"
    "scale=iw:ih,"
    "format=yuv420p"
)

# ---------------------------------------------------------------------------
# CUDA / QSV / libplacebo detection (cached after the first call)
# ---------------------------------------------------------------------------

_cuda_available: Optional[bool] = None
_scale_cuda_available: Optional[bool] = None
_tonemap_cuda_available: Optional[bool] = None
_libplacebo_avail: Optional[bool] = None
_qsv_avail: Optional[bool] = None
_qsv_decoders: Optional[Set[str]] = None

# Cached output of `ffmpeg -filters` (shared by all filter probes).
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


def libplacebo_available() -> bool:
    """Return True when libplacebo is usable at runtime (Vulkan device opens).

    Two-stage check:
    1. Verify that ``libplacebo`` is listed by ``ffmpeg -filters`` (compiled-in
       with ``--enable-libplacebo``).
    2. Perform a functional Vulkan probe by running a single libplacebo frame
       through a 64×64 dummy source rendered as 10-bit (``yuv420p10le``).

    The 10-bit pixel format is critical: libplacebo only initialises a Vulkan
    device when the input is ≥10-bit (i.e. HDR-like).  An 8-bit ``lavfi color``
    source takes a lightweight software path that never touches Vulkan, so the
    probe would always succeed even when Vulkan is broken.  Using
    ``format=yuv420p10le`` before the filter mirrors exactly what a real UHD/HDR
    video file delivers, triggering the same Vulkan init code path.

    This step fails with a non-zero exit code when the Vulkan device cannot be
    initialised (``VK_ERROR_INITIALIZATION_FAILED`` or similar) — even if the
    filter is compiled in.  Without this probe, machines where Vulkan fails at
    runtime would still select the libplacebo path, causing the entire FFmpeg
    filter chain to crash and decode 0 frames.

    The result is cached after the first call so repeated checks are free.
    """
    global _libplacebo_avail
    if _libplacebo_avail is None:
        if "libplacebo" not in _get_ffmpeg_filters():
            _libplacebo_avail = False
        else:
            # Stage 2: functional Vulkan probe — attempt a real libplacebo pass
            # using a 10-bit source so the same Vulkan code path as real HDR
            # video is exercised.  FFmpeg exits non-zero when Vulkan init fails.
            try:
                probe = subprocess.run(
                    [
                        "ffmpeg", "-hide_banner", "-loglevel", "error",
                        "-f", "lavfi", "-i", "color=c=black:size=64x64:duration=0.04",
                        "-vf", (
                            # Convert to 10-bit first so libplacebo takes the
                            # same Vulkan-accelerated HDR code path as a real
                            # UHD source (8-bit input uses a SW-only path that
                            # never tests Vulkan device creation).
                            "format=yuv420p10le,"
                            "libplacebo=w=64:h=64"
                            ":colorspace=bt709:color_trc=bt709:color_primaries=bt709"
                            ":tonemapping=mobius:range=pc:downscaler=bilinear,"
                            "format=yuv420p"
                        ),
                        "-frames:v", "1", "-f", "null", "-",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                )
                _libplacebo_avail = probe.returncode == 0
            except Exception:
                _libplacebo_avail = False
    return _libplacebo_avail


def qsv_available() -> bool:
    """Return True when Intel QSV hw-accel is usable at runtime.

    Two-stage check:
    1. Verify that ``qsv`` is listed by ``ffmpeg -hwaccels`` (i.e. the FFmpeg
       binary was compiled with ``--enable-libvpl`` or ``--enable-libmfx``).
    2. Perform a functional hardware probe by asking FFmpeg to actually
       initialise a QSV device (``-init_hw_device qsv=qsv:hw``).  This step
       fails with a non-zero exit code when no Intel iGPU / Quick Sync engine
       is present at runtime — even if the binary was compiled with QSV
       support.  Without this probe, machines with only NVIDIA GPUs would
       still pick the QSV path because the binary check passes, but then
       decode 0 frames silently.

    The result is cached after the first call so repeated checks are free.
    """
    global _qsv_avail
    if _qsv_avail is None:
        try:
            # Stage 1: compiled-in check.
            out = subprocess.check_output(
                ["ffmpeg", "-hide_banner", "-hwaccels"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode(errors="replace")
            if "qsv" not in out.lower():
                _qsv_avail = False
                return _qsv_avail

            # Stage 2: functional runtime probe — try to open the QSV device.
            # FFmpeg exits non-zero when the Intel hardware driver is missing.
            probe = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-init_hw_device", "qsv=qsv:hw",
                    "-f", "lavfi", "-i", "nullsrc=duration=0",
                    "-frames:v", "0", "-f", "null", "-",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            _qsv_avail = probe.returncode == 0
        except Exception:
            _qsv_avail = False
    return _qsv_avail


def qsv_decoders_available() -> Set[str]:
    """Return the set of QSV decoder names compiled into this FFmpeg build.

    Queries ``ffmpeg -decoders`` once and caches the result.  The returned
    set contains names such as ``"hevc_qsv"``, ``"h264_qsv"``, etc.  An
    empty set is returned when QSV decoders are unavailable or the probe
    fails.
    """
    global _qsv_decoders
    if _qsv_decoders is None:
        try:
            out = subprocess.check_output(
                ["ffmpeg", "-hide_banner", "-decoders"],
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).decode(errors="replace")
            _qsv_decoders = {
                name
                for name in (
                    "hevc_qsv", "h264_qsv", "mpeg2_qsv", "vp9_qsv", "av1_qsv",
                )
                if name in out
            }
        except Exception:
            _qsv_decoders = set()
    return _qsv_decoders


# Mapping from ffprobe codec_name → QSV decoder name.
_QSV_CODEC_MAP: Dict[str, str] = {
    "hevc":        "hevc_qsv",
    "h264":        "h264_qsv",
    "mpeg2video":  "mpeg2_qsv",
    "vp9":         "vp9_qsv",
    "av1":         "av1_qsv",
}


def _qsv_decoder_for_video(video_path: str) -> Optional[str]:
    """Return the QSV decoder name for the first video stream, or *None*.

    Probes the video codec with ``ffprobe`` and looks up the matching
    ``*_qsv`` decoder.  Returns ``None`` when the codec has no QSV decoder,
    QSV is unavailable, or the probe fails.
    """
    if not qsv_available():
        return None
    avail = qsv_decoders_available()
    if not avail:
        return None
    try:
        codec_name = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode(errors="replace").strip().lower()
    except Exception:
        return None
    candidate = _QSV_CODEC_MAP.get(codec_name)
    return candidate if candidate in avail else None



def _get_ffmpeg_major_version() -> int:
    """Return the major version of the installed FFmpeg (cached).

    Used to select the correct output options:
      * FFmpeg ≥ 7: ``-/filter_complex file`` (file-based option syntax, added in 7.0;
                     ``-filter_complex_script`` was removed in 7.0)
                    ``-fps_mode passthrough``
      * FFmpeg 5–6: ``-filter_complex_script file`` (deprecated but present)
                    ``-fps_mode passthrough``
      * FFmpeg 4:   ``-filter_complex_script file``  and  ``-vsync 0``

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

    * HDR + full-GPU   → tonemap_cuda + scale_cuda pipeline
    * HDR + scale-GPU  → scale_cuda + CPU zscale/tonemap pipeline
    * HDR + libplacebo → single-pass libplacebo HDR→SDR (Opt 4, CPU)
    * HDR + CPU-only   → CPU zscale + tonemap + scale pipeline (fallback)
    * SDR + scale-GPU  → scale_cuda pipeline
    * SDR + CPU-only   → bilinear scale pipeline (Opt 2)

    All paths output ``yuv420p`` (Opt 3) — ~33 % less pipe bandwidth
    compared to ``bgr24``.  Python converts with
    ``cv2.cvtColor(yuv, COLOR_YUV2BGR_I420)``.

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
                "format=yuv420p"
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
                "format=yuv420p"
            )
        # CPU-only HDR: prefer libplacebo (one shader pass) over the 4-step
        # zscale chain when available (Opt 4).
        if libplacebo_available():
            return (
                f"libplacebo=w={width}:h={height}"
                ":colorspace=bt709:color_trc=bt709:color_primaries=bt709"
                ":tonemapping=mobius:range=pc:downscaler=bilinear,"
                "format=yuv420p"
            )
        return (
            "zscale=t=linear:npl=100:filter=bilinear,"
            "format=gbrpf32le,"
            "zscale=p=bt709:filter=bilinear,"
            "tonemap=tonemap=reinhard:desat=0,"
            "zscale=t=bt709:m=bt709:range=full:filter=bilinear,"
            f"scale={width}:{height}:flags=bilinear,"
            "format=yuv420p"
        )
    else:
        # SDR: no tone-mapping needed; applying it would re-linearise the
        # already-correct gamma and make images too bright.
        if _scale_gpu:
            return (
                f"scale_cuda={width}:{height}:interp_algo=bicubic,"
                "hwdownload,"
                "format=yuv420p"
            )
        # Opt 2: bilinear is adequate for the intermediate scale step —
        # Python applies Lanczos4 on the actual patch crops.
        return (
            f"scale={width}:{height}:flags=bilinear,"
            "format=yuv420p"
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


# ---------------------------------------------------------------------------
# Template-based degradation (new in Task 2)
# ---------------------------------------------------------------------------

def sample_degradation_template_params(
    deg_spec: dict,
    center_frame: Optional[np.ndarray] = None,
) -> Optional[dict]:
    """
    Sample scene-level degradation parameters from a **degradation template** dict.

    Unlike the legacy :func:`_sample_degrade_params` (which uses a flat
    ``lr_*`` key schema), this function interprets the new template structure
    with top-level keys ``blur``, ``compression``, ``noise``, ``chroma``, and
    ``color``.  Each sub-dict specifies its own probability gate, so stages are
    independently activated rather than being gated by a single global probability.

    Parameters are sampled **once per scene window** and stored in a frozen
    dict that :func:`apply_degradation_template_params` applies to every LR
    frame in the window.  This mirrors real MPEG-2 encoder behaviour where the
    quantizer settings (and therefore noise level, blur, and JPEG quality) are
    constant within a GOP.  Additive noise is still drawn per-frame inside
    :func:`apply_degradation_template_params` at the sampled sigma.

    Args:
        deg_spec:     Degradation template dict from ``templates["degradation_templates"]``.
                      Keys: ``blur``, ``compression``, ``noise``, ``chroma``, ``color``.
        center_frame: Not used (reserved for future dark-scene boost; accepted
                      for API symmetry with :func:`_sample_degrade_params`).

    Returns:
        A dict of sampled scalar parameters, or ``None`` when no stage was activated.
        Keys depend on which stages were sampled; possible keys:

        * ``blur_sigma``         – Gaussian blur σ (float > 0).
        * ``jpeg_quality``       – JPEG encode quality 1-100 (int).
        * ``noise_sigma``        – Gaussian noise std-dev (float > 0).
        * ``color_noise``        – True = colour noise, False = luma noise (bool).
        * ``saturation``         – HSV saturation multiplier (float).
        * ``chroma_bleed``       – chroma bleed strength (float, 0 = disabled).
        * ``contrast``           – linear contrast multiplier (float).
        * ``brightness``         – additive brightness offset normalised 0-1 (float).
        * ``gamma``              – gamma exponent; output = input^(1/gamma) (float).
        * ``black_lift``         – additive black-level lift normalised 0-1 (float).
    """
    if not deg_spec:
        return None

    params: dict = {}

    # ── Blur ─────────────────────────────────────────────────────────────────
    blur = deg_spec.get("blur", {})
    if blur and random.random() < float(blur.get("prob", 0.0)):
        sr = blur.get("sigma_range", [0.5, 1.5])
        params["blur_sigma"] = random.uniform(float(sr[0]), float(sr[1]))

    # ── Compression (JPEG round-trip) ────────────────────────────────────────
    compression = deg_spec.get("compression", {})
    if compression and random.random() < float(compression.get("prob", 0.0)):
        qr = compression.get("jpeg_quality_range", [70, 90])
        params["jpeg_quality"] = random.randint(int(qr[0]), int(qr[1]))

    # ── Noise ────────────────────────────────────────────────────────────────
    noise_cfg = deg_spec.get("noise", {})
    if noise_cfg and random.random() < float(noise_cfg.get("prob", 0.0)):
        sr = noise_cfg.get("sigma_range", [1.0, 5.0])
        params["noise_sigma"] = random.uniform(float(sr[0]), float(sr[1]))
        params["color_noise"] = random.random() < float(noise_cfg.get("color_noise_prob", 0.2))

    # ── Chroma ───────────────────────────────────────────────────────────────
    chroma = deg_spec.get("chroma", {})
    if chroma:
        sat_range = chroma.get("saturation_range", [1.0, 1.0])
        params["saturation"] = random.uniform(float(sat_range[0]), float(sat_range[1]))
        bleed_prob = float(chroma.get("chroma_bleed_prob", 0.0))
        if bleed_prob > 0.0 and random.random() < bleed_prob:
            params["chroma_bleed"] = float(chroma.get("chroma_bleed_strength", 0.0))

    # ── Color ────────────────────────────────────────────────────────────────
    color = deg_spec.get("color", {})
    if color:
        cr = color.get("contrast_range", [1.0, 1.0])
        br = color.get("brightness_range", [0.0, 0.0])
        gr = color.get("gamma_range", [1.0, 1.0])
        params["contrast"] = random.uniform(float(cr[0]), float(cr[1]))
        params["brightness"] = random.uniform(float(br[0]), float(br[1]))
        params["gamma"] = random.uniform(float(gr[0]), float(gr[1]))
        params["black_lift"] = float(color.get("black_lift", 0.0))

    return params if params else None


def apply_degradation_template_params(
    frame: np.ndarray,
    params: dict,
) -> np.ndarray:
    """
    Apply pre-sampled template-based degradation parameters to a single LR frame.

    This function is the template-aware counterpart to :func:`_apply_degrade_params`.
    It consumes a *params* dict produced by :func:`sample_degradation_template_params`
    and applies each stage that has been activated.

    The application order is:
      1. Blur        – applied first so JPEG artifacts are not blurred away.
      2. Noise       – independent per-frame noise at the sampled sigma.
      3. JPEG        – blocking / ringing artefacts.
      4. Saturation  – chroma scaling in HSV space.
      5. Color       – contrast, brightness, gamma, black-lift in floating point.

    GT frames are never passed through this function; only LR frames are degraded.

    Args:
        frame:  Single LR BGR frame (uint8 numpy array).
        params: Dict produced by :func:`sample_degradation_template_params`.

    Returns:
        Degraded frame as uint8 numpy array.
    """
    result: np.ndarray = frame.copy()

    # ── 1. Blur ──────────────────────────────────────────────────────────────
    if "blur_sigma" in params:
        sigma = float(params["blur_sigma"])
        if sigma >= 0.1:
            ksize = max(3, 2 * int(np.ceil(2.0 * sigma)) + 1)
            if ksize % 2 == 0:
                ksize += 1
            result = cv2.GaussianBlur(result, (ksize, ksize), sigma)

    # ── 2. Noise (per-frame independent, scene-consistent sigma) ─────────────
    if "noise_sigma" in params:
        sigma = float(params["noise_sigma"])
        if sigma > 0.0:
            if params.get("color_noise", False):
                noise = np.random.normal(0.0, sigma, result.shape).astype(np.float32)
            else:
                # Luma noise: identical value across channels
                gray_noise = np.random.normal(0.0, sigma, result.shape[:2]).astype(np.float32)
                noise = np.stack([gray_noise, gray_noise, gray_noise], axis=2)
            result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # ── 3. JPEG round-trip ───────────────────────────────────────────────────
    if "jpeg_quality" in params:
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, int(params["jpeg_quality"])]
        ok, buf = cv2.imencode(".jpg", result, encode_param)
        if ok:
            result = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # ── 4. Saturation ────────────────────────────────────────────────────────
    sat = params.get("saturation", 1.0)
    if sat != 1.0:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ── 5. Color: contrast / brightness / gamma / black-lift ─────────────────
    has_color = any(k in params for k in ("contrast", "brightness", "gamma", "black_lift"))
    if has_color:
        result_f = result.astype(np.float32) / 255.0
        result_f = result_f + float(params.get("black_lift", 0.0))
        result_f = result_f * float(params.get("contrast", 1.0)) + float(params.get("brightness", 0.0))
        gamma = float(params.get("gamma", 1.0))
        if gamma != 1.0 and gamma > 0.0:
            result_f = np.power(np.clip(result_f, 0.0, 1.0), 1.0 / gamma)
        result = np.clip(result_f * 255.0, 0, 255).astype(np.uint8)

    return result


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
    deg_spec: Optional[dict] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Create a ``(GT, LR)`` patch pair from a sequence of frames.

    **Resize mode** (``source_mode == "resize"`` in format_cfg, or legacy
    format names ``medium_169`` / ``720_169``):
      * GT – full-frame resize to ``gt_size`` with ``INTER_LANCZOS4``
        (best quality, preserves full frame content).
      * LR – full-frame resize to ``lr_size`` with ``INTER_AREA``
        (DVD-realistic quality).
      * Suitable for any aspect-ratio target (16:9, 4:3, etc.) when the
        source and target aspect ratios are compatible and no spatial crop is
        desired.

    **Crop mode** (``source_mode == "crop"`` in format_cfg, all other
    legacy format names):
      * GT – centre frame.  When the source frame is large enough for 2×
        oversampling (``frame_h ≥ 2*gt_h`` **and** ``frame_w ≥ 2*gt_w``,
        e.g. native 4K for the 720/720_169 formats), a ``2*gt_size`` crop
        is taken from the source and Lanczos4-downsampled to ``gt_size``.
        The 2× Lanczos4 step averages H.265 in-loop deblocking softness
        and produces a clean GT comparable to the full-frame resize path.
        For smaller sources a direct 1:1 crop is used instead.
      * LR – all frames, same crop region, downscaled to ``lr_size`` with
        ``INTER_AREA`` (stacked vertically on axis 0).
      * Works correctly for both 16:9 and 4:3 crop targets – no aspect-
        ratio name check is performed.

    **Source-mode resolution**:
    The ``source_mode`` field in *format_cfg* takes precedence.  When it is
    absent (legacy callers that do not populate it), the function falls back
    to the old format-name test (``"medium_169"`` / ``"720_169"``).  New code
    should always populate ``source_mode`` in the format config.

    **Degradation**:
    Two degradation paths are supported, applied in this priority order:

    1. *New template-based* (``deg_spec`` arg): parameters are sampled via
       :func:`sample_degradation_template_params` and applied via
       :func:`apply_degradation_template_params`.  Supports blur,
       compression, noise, chroma, and colour stages.

    2. *Legacy flat* (``degrade_cfg`` arg): used when ``deg_spec`` is
       ``None``; forwards to :func:`_sample_degrade_params` /
       :func:`_apply_degrade_params` for backward compatibility.

    In both cases parameters are **sampled once per scene** so that all LR
    frames in the window share the same settings — consistent with real
    MPEG-2 encoder behaviour.  GT is always kept lossless.

    In both cases a near-uniform GT (plain black, white, or flat colour) is
    silently discarded (``(None, None)``).  If the source frame is too small
    for the requested resize target a warning is logged.

    Args:
        frames:       BGR numpy arrays, length 5 or 7.
        format_name:  Format key string (used for logging and legacy
                      source-mode fallback only — all functional decisions are
                      now driven by *format_cfg*).
        format_cfg:   Dict with at minimum ``'gt_size': [W, H]`` and
                      ``'lr_size': [W, H]``.  Optionally:
                      * ``'source_mode'``: ``"resize"`` or ``"crop"``
                        (overrides the legacy format-name test).
        force_center: Crop mode only – use the geometric centre of the frame
                      instead of a random crop location.
        logger:       Optional logger instance for warning messages.
        degrade_cfg:  Legacy degradation config dict.  Ignored when *deg_spec*
                      is provided.  When ``None`` no legacy degradation is
                      applied.
        deg_spec:     New-style degradation template dict (from
                      ``templates["degradation_templates"]``).  When
                      provided, *degrade_cfg* is ignored.

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

    # ── Determine source_mode ─────────────────────────────────────────────────
    # Prefer explicit source_mode from format_cfg; fall back to legacy name check.
    source_mode = format_cfg.get("source_mode")
    if source_mode not in ("resize", "crop"):
        # Legacy fallback: the old hardcoded whitelist
        source_mode = "resize" if format_name in ("medium_169", "720_169") else "crop"

    # ── Degradation: resolve which sampler to use ─────────────────────────────
    # deg_spec (new template) takes priority over degrade_cfg (legacy flat cfg).
    center_raw = frames[center_idx]
    if deg_spec is not None:
        # New template-based degradation – sample once per scene.
        _scene_params = sample_degradation_template_params(deg_spec, center_frame=center_raw)
        _apply_fn = apply_degradation_template_params
    elif degrade_cfg is not None:
        # Legacy degradation – keep old behaviour.
        _scene_params = _sample_degrade_params(degrade_cfg, center_frame=center_raw)
        _apply_fn = _apply_degrade_params
    else:
        _scene_params = None
        _apply_fn = None

    if source_mode == "resize":
        # ── Resize path: full-frame rescale ──────────────────────────────────
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
        lr_frames = []
        for frame in frames:
            lr = cv2.resize(frame, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
            if _scene_params is not None:
                lr = _apply_fn(lr, _scene_params)
            lr_frames.append(lr)

    else:
        # ── Crop path: spatial crop + optional oversampled Lanczos4 ──────────
        #
        # When the source is large enough for 2× oversampling (e.g. native 4K
        # for 720/1152 GT sizes), take a 2×gt_size crop and Lanczos4-
        # downsample it.  Produces a clean GT comparable to full-frame resize.
        # For smaller sources fall back to a 1:1 native-resolution crop.
        #
        # This path is dimension-driven, not name-driven, so it works correctly
        # for any aspect ratio (16:9 crop, 4:3 crop, etc.).
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

        # GT: INTER_AREA for exact-integer downsampling (box filter; identical
        # quality to Lanczos4 at 2× but significantly faster).  INTER_LANCZOS4
        # is only superior for non-integer scale factors.
        if oversample > 1:
            gt = cv2.resize(center_crop, (gt_w, gt_h), interpolation=cv2.INTER_AREA)
        else:
            gt = center_crop

        # Variety check: silently discard near-uniform GT (black/white/flat)
        gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        if float(gray.std()) < 7.0:
            return None, None

        # LR: same oversampled area → exact scale ratio (e.g. 240/720 = 1/3).
        lr_frames = []
        for frame in frames:
            raw_crop = frame[crop_y : crop_y + sample_h, crop_x : crop_x + sample_w]
            lr = cv2.resize(raw_crop, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
            if _scene_params is not None:
                lr = _apply_fn(lr, _scene_params)
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
        # Resolve the current bucket once before writing so this patch pair
        # goes into a consistent location (GT and LR share the same bucket).
        gt_dir, lr_dir = get_synced_bucket_dirs(output_dirs["gt"], output_dirs["lr"])

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


def is_black_frame(gt: np.ndarray, unique_ratio_threshold: float = 0.07) -> bool:
    """Return True when *gt* is a near-uniform frame (black, white, solid colour).

    Converts the frame to grayscale and counts the number of distinct intensity
    levels that appear.  When fewer than *unique_ratio_threshold* × 256 distinct
    levels are present (default → fewer than ~18 out of 256 possible values) the
    frame is considered non-informative and is skipped.

    This criterion catches solid-black, solid-white, colour-bars, and any
    near-uniform frame regardless of its average brightness — unlike the
    previous mean-brightness check which only caught dark frames.

    Performance note: the function stride-samples the frame by a factor of 8 in
    each spatial dimension (producing a ~288×180 view from a 2304×1440 source)
    before the grayscale conversion.  Near-uniform frames are uniformly-valued at
    any sampling density, so the detection is lossless while processing ~64×
    fewer pixels.  ``np.bincount`` on the uint8 ravel is O(n); ``np.mean`` with
    a simple channel average is used (not the 0.299/0.587/0.114 luminosity
    formula) because the uniqueness count is insensitive to the exact weighting.

    Args:
        gt:                     Center-frame BGR numpy array (H × W × 3, uint8).
        unique_ratio_threshold: Fraction of 256 grey levels that must be present
                                for the frame to be kept (default 0.07 → at least
                                ~18 distinct grey levels required).

    Returns:
        ``True`` when the frame has too few unique intensity levels to be useful.
    """
    # Stride-sample to 1/8 in each dimension — creates a view (no copy).
    # 2304×1440 → 288×180 = 51 840 pixels instead of 3 317 760: ~64× cheaper.
    sample = gt[::8, ::8, :]
    gray = np.mean(sample, axis=2).astype(np.uint8)
    unique_count = np.count_nonzero(np.bincount(gray.ravel(), minlength=256))
    return unique_count < unique_ratio_threshold * 256


def _get_video_dimensions(video_path: str) -> Tuple[int, int]:
    """Return ``(width, height)`` of the first video stream, or ``(0, 0)`` on failure.

    Used to check whether the source resolution is sufficient for the
    requested stream dimensions (e.g. 4K extraction on a sub-4K source).
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
    ffmpeg_cmd: Optional[List[str]] = None,
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
        ffmpeg_cmd:     The complete FFmpeg argument list that was executed.
                        When provided it is logged before the stderr output so
                        the exact invocation can be reproduced from the log.
    """
    if not stderr_lines:
        return
    try:
        import shlex
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
        if ffmpeg_cmd:
            header += f"FFmpeg command: {shlex.join(ffmpeg_cmd)}\n"
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
    cuda_device: int = 0,
    use_qsv: bool = True,
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
                             passed (i.e. near-uniform frames are always filtered unless you
                             explicitly pass ``lambda _: False``).
                             Note: the ``is_black_frame`` default filter (unique grey
                             levels < 7% of 256) partially overlaps with the variety-std
                             check inside ``create_patch_pair`` (std < 15).  Set
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
        cuda_device:         CUDA device ordinal used for hardware-accelerated
                             decoding (default 0 = first GPU).  Passed to FFmpeg
                             as ``-init_hw_device cuda=hw:<cuda_device>``.  Has no
                             effect when ``use_cuda`` is ``False`` or CUDA is not
                             available in the local FFmpeg build.  Use the index
                             reported by ``nvidia-smi`` to target a specific GPU
                             when multiple are present.

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

    # --- Seek-mode decision -----------------------------------------------
    # Streaming decodes every frame up to the last needed one even though the
    # select filter skips the expensive scale/tonemap stages for most frames.
    # For sparse assignment sets the raw decode cost dominates.  Seeking to
    # each cluster of nearby ranges is cheaper when:
    #
    #   N_ranges × assumed_GOP_size  <  last_needed_frame
    #
    # i.e. the total frames decoded by seeking to each cluster is less than
    # the total frames decoded by a single streaming pass.
    # GOP estimate of 150 frames (≈6 s at 24 fps) is conservative for modern
    # H.265 HDR encodes; the actual value only affects the crossover threshold.
    _SEEK_GOP_ESTIMATE: int = 150
    _use_seek_mode: bool = (
        bool(_select_ranges)
        and len(_select_ranges) * _SEEK_GOP_ESTIMATE < last_needed + 1
    )
    if _use_seek_mode:
        # Group consecutive ranges into seek clusters so one -ss call covers
        # several nearby assignments without extra keyframe-seek overhead.
        _seek_clusters: List[List[Tuple[int, int]]] = []
        _cur_cl: List[Tuple[int, int]] = [_select_ranges[0]]
        for _sr in _select_ranges[1:]:
            if _sr[0] - _cur_cl[-1][1] <= _SEEK_GOP_ESTIMATE:
                _cur_cl.append(_sr)
            else:
                _seek_clusters.append(_cur_cl)
                _cur_cl = [_sr]
        _seek_clusters.append(_cur_cl)
    else:
        _seek_clusters = []

    # --- Pre-compute per-video constants ----------------------------------
    # video_stem and output dir paths are the same for every patch in this
    # video — compute them once to avoid Path() and os.makedirs overhead in
    # the hot decode loop.
    _video_stem: str = Path(video_path).stem
    _output_dirs_cache: Dict[Tuple[str, str], Dict[str, str]] = {}
    for _, _cat, _fmt in sorted_asgn:
        _key = (_cat, _fmt)
        if _key not in _output_dirs_cache:
            _base_dirs = get_output_dirs_for_format(base_dir, _cat, _fmt, n_frames)
            # Determine the write bucket ONCE per (category, format) before this
            # video's first patch is written.  All patches from this video land in
            # the same bucket — never split across a bucket boundary mid-video.
            _gt_bucket, _lr_bucket = get_synced_bucket_dirs(
                _base_dirs["gt"], _base_dirs["lr"]
            )
            _dirs = dict(_base_dirs)  # keep val_gt / val_lr unchanged
            _dirs["gt"] = _gt_bucket
            _dirs["lr"] = _lr_bucket
            os.makedirs(_gt_bucket, exist_ok=True)
            os.makedirs(_lr_bucket, exist_ok=True)
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

    # 4 write threads: at 4K source resolution each PNG write is ~4× heavier
    # than 1080p, so more writers keep disk I/O overlapping with FFmpeg decode.
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
    # available CUDA/QSV capabilities:
    #
    #  HDR source  + full-GPU   → tonemap_cuda + scale_cuda + hwdownload
    #  HDR source  + scale-GPU  → scale_cuda + hwdownload (p010) + zscale+tonemap
    #  HDR source  + QSV decode → Intel QSV H.265/H.264 decode + CPU libplacebo/zscale
    #  HDR source  + CPU-only   → zscale + tonemap(reinhard) + scale (bilinear)
    #  SDR source  + scale-GPU  → scale_cuda + hwdownload (plain scale)
    #  SDR source  + CPU-only   → scale bilinear (software, no linearisation)
    _use_cuda = use_cuda and cuda_available()
    _full_gpu  = _use_cuda and is_hdr and tonemap_cuda_available()
    _scale_gpu = _use_cuda and (not _full_gpu) and scale_cuda_available()

    # QSV decode: try Intel hardware H.265/H.264 decode when no CUDA is active.
    # QSV only accelerates the decode step; the filter chain (libplacebo,
    # zscale, scale) still runs on the CPU — so build_vf_filter is unchanged.
    _qsv_codec: Optional[str] = (
        _qsv_decoder_for_video(video_path)
        if use_qsv and not _use_cuda
        else None
    )

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
    # In seek mode this injection is skipped: each cluster subprocess already
    # starts at the right position, so every decoded frame is a needed one.
    #
    # Placement depends on the pipeline tier:
    #   CPU path        → select goes at the very start of the filter chain.
    #   Hybrid GPU/CPU  → GPU scale runs first (cheap, already on GPU);
    #                     select is inserted right after hwdownload so the
    #                     expensive CPU tonemap only runs on needed frames.
    #   Full-GPU        → all stages are on GPU; select is placed after the
    #                     final hwdownload to cut pipe bandwidth.
    if _select_expr and not _use_seek_mode:
        if _full_gpu:
            # Full-GPU: insert select right before the terminal format=yuv420p
            # (after hwdownload+scale=iw:ih — all GPU work is done; select
            # avoids the final CPU format conversion and the pipe write for
            # unwanted frames).
            _marker = ",format=yuv420p"
            if _marker in vf_filter:
                vf_filter = vf_filter.replace(
                    _marker, f",select={_select_expr},format=yuv420p", 1
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
    # Device index is parameterised so callers can target a specific GPU
    # (e.g. GPU 1 on a dual-GPU system) without setting CUDA_VISIBLE_DEVICES.
    _CUDA_HW_INIT = ["-init_hw_device", f"cuda=hw:{cuda_device}"]

    hdr_label = "HDR" if is_hdr else "SDR"
    _placebo = is_hdr and (not _full_gpu) and (not _scale_gpu) and libplacebo_available()
    if _full_gpu:
        hw_args        = [*_CUDA_HW_INIT, "-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        pipeline_label = f"full-GPU tonemap_cuda+scale_cuda [{hdr_label}] yuv420p {stream_width}×{stream_height}"
    elif _scale_gpu:
        hw_args        = [*_CUDA_HW_INIT, "-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        pipeline_label = f"scale-GPU + CPU {'zscale/tonemap' if is_hdr else 'passthrough'} [{hdr_label}] yuv420p {stream_width}×{stream_height}"
    elif _use_cuda:
        hw_args        = [*_CUDA_HW_INIT, "-hwaccel", "cuda"]
        _cpu_algo = ("libplacebo" if _placebo else "tonemap/reinhard") if is_hdr else "scale/bilinear"
        pipeline_label = f"decode-GPU + CPU {_cpu_algo} [{hdr_label}] yuv420p {stream_width}×{stream_height}"
    elif _qsv_codec:
        # Intel QSV hardware decode: frame data lands in CPU memory after decode,
        # so the existing CPU filter chain (libplacebo / zscale) is unchanged.
        hw_args        = ["-hwaccel", "qsv", "-c:v", _qsv_codec]
        _cpu_algo = ("libplacebo" if _placebo else "zscale/tonemap") if is_hdr else "scale/bilinear"
        pipeline_label = f"QSV decode ({_qsv_codec}) + CPU {_cpu_algo} [{hdr_label}] yuv420p {stream_width}×{stream_height}"
    else:
        hw_args        = []
        _cpu_algo = ("libplacebo" if _placebo else "zscale/tonemap") if is_hdr else "scale/bilinear"
        pipeline_label = f"CPU-only {_cpu_algo} [{hdr_label}] yuv420p {stream_width}×{stream_height}"

    # Pipe bandwidth for the log (yuv420p = 1.5 bytes/pixel).
    _pipe_mb_per_frame = stream_width * stream_height * 1.5 / (1024 * 1024)

    _log(
        f"🎬 Streaming extractor: {len(sorted_asgn)} assignments, "
        f"{len(pending_centers)} unique centers, "
        f"last frame needed: {last_needed}, "
        f"stream={stream_width}×{stream_height}, "
        f"pipeline={pipeline_label}, nice={nice_level}"
    )
    _mode_label = (
        f"seek ({len(_seek_clusters)} clusters)"
        if _use_seek_mode
        else "stream (select filter)"
    )
    _log(
        f"🎯 Frame selection: {len(_all_needed)} frames needed "
        f"({_select_pct:.1f}% of {last_needed + 1} decoded) "
        f"in {len(_select_ranges)} ranges — mode={_mode_label}"
    )
    _log(
        f"📦 Pipe: yuv420p {stream_width}×{stream_height} "
        f"= {_pipe_mb_per_frame:.2f} MB/frame"
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
    _ffmpeg_cmd: List[str] = []  # last FFmpeg command; logged on error

    # yuv420p: 1.5 bytes per pixel (Y plane + half-res U+V planes).
    # Compared to bgr24 (3 bytes/pixel) this cuts pipe bandwidth by ~33 %.
    frame_bytes: int = stream_width * stream_height * 3 // 2
    patches_created: Dict[str, int] = {}

    # Rolling buffer: frame_idx → BGR frame (numpy array)
    buffer: Dict[int, np.ndarray] = {}
    pending_idx: int = 0  # index into pending_centers
    frames_examined: int = 0  # assignments processed (saved + skipped)

    # --- Per-video timing accumulators (mutated inside _consume_raw_frame) ---
    _t_phases: dict = {
        "n_frames_buf": 0,   # total raw frames processed through buffer
        "n_centers":    0,   # centers fully evaluated (= frames_examined, incl. black)
        "t_buf_s":      0.0, # total time: yuv→bgr convert + copy + buffer insert/evict
        "t_black_s":    0.0, # total time: black-frame check (per center)
        "t_patch_s":    0.0, # total time: create_patch_pair calls (per center×format)
        "t_write_s":    0.0, # total time: write_queue.put (per patch)
        "n_patches":    0,   # patches enqueued for writing
        "q_size_last":  0,   # last observed write-queue depth
    }
    _next_timing_log: List[int] = [50]  # mutable box: write debug log at this n_centers

    def _write_timing_log_entry() -> None:
        """Append a one-line timing summary to <base_dir>/timing_debug.log."""
        nc = _t_phases["n_centers"]
        nf = _t_phases["n_frames_buf"]
        np_ = _t_phases["n_patches"]
        if nc == 0:
            return
        from datetime import datetime as _dt
        ms_buf   = _t_phases["t_buf_s"]   / max(nf, 1) * 1000
        ms_black = _t_phases["t_black_s"] / nc * 1000
        ms_patch = _t_phases["t_patch_s"] / nc * 1000
        ms_write = _t_phases["t_write_s"] / max(np_, 1) * 1000
        # Pipe throughput in MB/s (yuv420p = 1.5 bytes/pixel).
        elapsed = _t_phases["t_buf_s"] + _t_phases["t_black_s"] + _t_phases["t_patch_s"]
        pipe_mbs = (nf * _pipe_mb_per_frame / elapsed) if elapsed > 0 else 0.0
        line = (
            f"[{_dt.now().strftime('%H:%M:%S')}] "
            f"video={os.path.basename(video_path)} "
            f"pipe={stream_width}x{stream_height} yuv420p {_pipe_mb_per_frame:.2f}MB/frame "
            f"pipe_mbs={pipe_mbs:.1f}MB/s "
            f"centers={nc} frames={nf} patches={np_} "
            f"buf={ms_buf:.1f}ms/frame "
            f"black={ms_black:.1f}ms/ctr "
            f"patch={ms_patch:.1f}ms/ctr "
            f"write={ms_write:.1f}ms/patch "
            f"qsize={_t_phases['q_size_last']}"
        )
        try:
            log_path = os.path.join(base_dir, "timing_debug.log")
            os.makedirs(base_dir, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as _fh:
                _fh.write(line + "\n")
        except Exception:
            pass

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
        # installed FFmpeg version.
        #   FFmpeg 7+: -/filter_complex <file>  (file-based option syntax, added in 7.0)
        #              -filter_complex_script was removed in 7.0
        #   FFmpeg 5-6: -filter_complex_script <file>  (deprecated but still present)
        #   FFmpeg 4:   -filter_complex_script <file>
        #   FFmpeg 5+:  -fps_mode passthrough  (replaces deprecated -vsync)
        # -vsync 0 / -fps_mode passthrough is CRITICAL: without it, FFmpeg fills
        # PTS gaps left by the select filter with duplicated frames, so Python
        # would read only frames from the very start of the video.
        _ffmpeg_ver = _get_ffmpeg_major_version()
        _fc_args = (
            ["-/filter_complex", _fc_script_path]
            if _ffmpeg_ver >= 7
            else ["-filter_complex_script", _fc_script_path]
        )
        _vsync_args = (
            ["-fps_mode", "passthrough"]
            if _ffmpeg_ver >= 5
            else ["-vsync", "0"]
        )

        # Drain stderr helper: shared by stream mode and each seek-mode cluster.
        def drain_stderr(pipe: "subprocess.IO[bytes]") -> None:
            for line in pipe:
                stderr_lines.append(line.decode(errors="replace").rstrip())
            pipe.close()

        # ------------------------------------------------------------------
        # Inner helper: receive one decoded yuv420p frame, convert to BGR,
        # fill the rolling buffer, and satisfy any pending assignments whose
        # window is now complete.
        # Shared by both stream mode and seek mode to avoid code duplication.
        # ------------------------------------------------------------------
        def _consume_raw_frame(raw: bytes, actual_frame: int) -> None:
            nonlocal pending_idx, frames_examined, selected_idx

            selected_idx += 1
            _t_phases["n_frames_buf"] += 1

            _ta = time.monotonic()
            # yuv420p (I420) layout: Y plane (H rows × W cols) followed by
            # U plane (H/2 × W/2) and V plane (H/2 × W/2).
            # Total bytes = W × H × 3/2.  cv2.COLOR_YUV2BGR_I420 expects the
            # array shaped (H*3//2, W) and produces an (H, W, 3) BGR array.
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape(
                (stream_height * 3 // 2, stream_width)
            )
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            buffer[actual_frame] = frame

            # Evict frames no longer needed by any pending assignment.
            min_keep = max(0, pending_centers[pending_idx] - half)
            for old_idx in [k for k in buffer if k < min_keep]:
                del buffer[old_idx]
            _t_phases["t_buf_s"] += time.monotonic() - _ta

            # Satisfy pending assignments whose full window is now in the buffer.
            while pending_idx < len(pending_centers):
                center = pending_centers[pending_idx]
                if actual_frame < center + half:
                    break  # Need more frames

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
                    _t_phases["n_centers"] += 1

                    _tb = time.monotonic()
                    _is_black = _black_fn(center_raw)
                    _t_phases["t_black_s"] += time.monotonic() - _tb

                    if _is_black:
                        _log(f"  ⏭ frame {center} skipped (black frame)")
                    else:
                        for category, fmt_name in center_map[center]:
                            cfg = format_config.get(category, {}).get(fmt_name, {})
                            if not cfg:
                                continue

                            # Determine source_mode from cfg; fall back to legacy name
                            # check so existing callers that don't populate source_mode
                            # still work correctly.
                            _source_mode = cfg.get("source_mode")
                            if _source_mode not in ("resize", "crop"):
                                _source_mode = (
                                    "resize"
                                    if fmt_name in ("medium_169", "720_169")
                                    else "crop"
                                )
                            # Resize mode produces the same output on every attempt —
                            # no benefit in retrying a random crop.
                            max_attempts = 1 if _source_mode == "resize" else 6

                            # Per-format degradation: sample a template from the
                            # format's degradation_mix if available; fall back to the
                            # global degrade_cfg for legacy callers.
                            _deg_spec: Optional[dict] = None
                            _deg_mix = cfg.get("degradation_mix")
                            _deg_tmpls = cfg.get("degradation_templates")
                            if _deg_mix and _deg_tmpls:
                                _names = list(_deg_mix.keys())
                                _weights = [float(_deg_mix[k]) for k in _names]
                                _chosen = random.choices(_names, weights=_weights, k=1)[0]
                                _deg_spec = _deg_tmpls.get(_chosen)

                            gt, lr = None, None
                            _tp = time.monotonic()
                            for attempt in range(max_attempts):
                                force = attempt >= 5
                                gt, lr = create_patch_pair(
                                    window, fmt_name, cfg,
                                    force_center=force, logger=logger,
                                    degrade_cfg=degrade_cfg if _deg_spec is None else None,
                                    deg_spec=_deg_spec,
                                )
                                if gt is None:
                                    continue
                                if (
                                    is_interesting_fn is None
                                    or is_interesting_fn(gt)
                                    or force
                                ):
                                    break
                            _t_phases["t_patch_s"] += time.monotonic() - _tp

                            if gt is not None and lr is not None:
                                dirs = _output_dirs_cache[(category, fmt_name)]
                                patch_name = f"{_video_stem}_{int(ts * 1000):08d}.png"
                                _tw = time.monotonic()
                                _write_queue.put((
                                    gt, lr,
                                    os.path.join(dirs["gt"], patch_name),
                                    os.path.join(dirs["lr"], patch_name),
                                ))
                                _t_phases["t_write_s"] += time.monotonic() - _tw
                                _t_phases["n_patches"] += 1
                                _t_phases["q_size_last"] = _write_queue.qsize()
                                patches_created[category] = (
                                    patches_created.get(category, 0) + 1
                                )

                    # Periodic timing debug log (every 50 centres evaluated)
                    if _t_phases["n_centers"] >= _next_timing_log[0]:
                        _write_timing_log_entry()
                        _next_timing_log[0] += 50

                    if progress_fn is not None:
                        # 3rd arg = selected_idx: count of frames actually piped
                        # from FFmpeg to Python (used for real "piped fps" metric).
                        # 4th arg = snapshot of phase timings for GUI display.
                        progress_fn(frames_examined, dict(patches_created),
                                    selected_idx, dict(_t_phases))

                pending_idx += 1

        # ------------------------------------------------------------------
        if _use_seek_mode:
            # --- SEEK MODE ------------------------------------------------
            # One short FFmpeg subprocess per cluster of nearby ranges.
            # Each subprocess seeks (accurately, -ss after -i) to the cluster
            # start and reads exactly the frames in that cluster window.
            # Avoids decoding the entire video for sparse assignment sets.
            for _cl_idx, _cluster in enumerate(_seek_clusters):
                if pending_idx >= len(pending_centers):
                    break

                _cl_seek_frame = _cluster[0][0]
                _cl_end_frame  = _cluster[-1][1]
                _cl_n_read     = _cl_end_frame - _cl_seek_frame + 1
                _cl_seek_sec   = _cl_seek_frame / fps

                _cl_cmd = [
                    "ffmpeg",
                    "-threads", "0",
                    "-filter_threads", "0",
                    "-loglevel", "warning",
                    *hw_args,
                    "-probesize", "100M",
                    "-analyzeduration", "100M",
                    "-i", video_path,
                    "-ss", f"{_cl_seek_sec:.6f}",   # accurate seek (after -i)
                    "-frames:v", str(_cl_n_read),
                    *_fc_args,
                    "-map", "[vout]",
                    "-f", "rawvideo",
                    "-pix_fmt", "yuv420p",
                    *_vsync_args,
                    "pipe:1",
                ]
                _ffmpeg_cmd = _cl_cmd  # capture for error logging

                _cl_proc = subprocess.Popen(
                    _cl_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                process = _cl_proc   # expose for finally-block kill on exception
                _set_nice(_cl_proc.pid)

                _cl_stderr_t = threading.Thread(
                    target=drain_stderr, args=(_cl_proc.stderr,), daemon=True
                )
                _cl_stderr_t.start()

                for _cl_i in range(_cl_n_read):
                    if pending_idx >= len(pending_centers):
                        break

                    raw = _cl_proc.stdout.read(frame_bytes)
                    if len(raw) < frame_bytes:
                        break  # cluster extends past video end — stop gracefully

                    if _t_start is None:
                        _t_start = time.monotonic()

                    _consume_raw_frame(raw, _cl_seek_frame + _cl_i)

                    # Periodic throughput log (every _log_interval frames).
                    if _t_start is not None and selected_idx % _log_interval == 0:
                        _elapsed = time.monotonic() - _t_start
                        if _elapsed > 0:
                            _sel_fps = selected_idx / _elapsed
                            _sps_actual = frames_examined / _elapsed
                            _log(
                                f"  📊 cluster {_cl_idx + 1}/{len(_seek_clusters)}  "
                                f"sel {selected_idx:>5}/{len(_all_needed)}  "
                                f"sel/s {_sel_fps:>6.1f}  SPS {_sps_actual:>6.2f}  "
                                f"(scenes: {frames_examined})"
                            )

                # Clean up this cluster's process before starting the next.
                # Keep process pointing to _cl_proc until after successful reap
                # so the finally block can kill it if an exception interrupts us.
                try:
                    _cl_proc.stdout.close()
                except Exception:
                    pass
                try:
                    _cl_proc.kill()
                    _cl_proc.wait()
                except Exception:
                    pass
                _cl_stderr_t.join(timeout=2)
                process = None  # disarm finally block only after successful reap

        else:
            # --- STREAM MODE (single FFmpeg pass with select filter) ------
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
                "-pix_fmt", "yuv420p",
                *_vsync_args,
                "pipe:1",
            ]
            _ffmpeg_cmd = cmd  # capture for error logging

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _set_nice(process.pid)

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

                _consume_raw_frame(raw, actual_frame)

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
        _append_ffmpeg_log(base_dir, video_path, stderr_lines, pipeline_label, _ffmpeg_cmd or None)
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
            cuda_device=cuda_device,
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
    cuda_device: int = 0,
) -> Dict[str, int]:
    """Deprecated compatibility shim — forwards to extract_and_save_streaming_distributed.

    The former dual-buffer (4K stream + Python LANCZOS4 downscale to 1080p)
    approach has been removed.  All formats are now extracted in a single
    optimised FFmpeg pass via :func:`extract_and_save_streaming_distributed`
    (resolution ``STREAM_OPT_WIDTH × STREAM_OPT_HEIGHT``, yuv420p pipe, bilinear
    scale), which lets :func:`create_patch_pair` apply 2× oversampled Lanczos4
    crops for every GT family (1152×648, 960×540, 960×720).
    """
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
        stream_width=STREAM_OPT_WIDTH,
        stream_height=STREAM_OPT_HEIGHT,
        cuda_device=cuda_device,
    )
