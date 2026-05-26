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

build_remaining_assignments()
    Resume-aware counterpart to ``build_assignments_per_category``.  Given
    already-completed patch counts from the plan, subtracts them from the
    configured targets, estimates a resume timestamp, and returns only the
    assignments for the unfinished remainder — starting from that timestamp.
    Combine with the ``start_ts`` parameter of
    ``extract_and_save_streaming_distributed`` to skip the already-processed
    portion of the video entirely (FFmpeg fast seek), preserving all
    existing patches and avoiding redundant decoding.

extract_and_save_streaming_distributed()
    Single-stream entry point for all formats.  Launches one FFmpeg process,
    streams yuv420p frames at ``STREAM_OPT_WIDTH × STREAM_OPT_HEIGHT`` by
    default — large enough for ``create_patch_pair`` to apply 2× oversampled
    Lanczos4 crops for both the 1152×648 and 960×720 GT families.  Uses CUDA
    (``tonemap_cuda`` + ``scale_cuda`` or ``scale_cuda`` alone) when available;
    falls back to libplacebo (if present) or CPU zscale automatically.
    Passes the filter chain via a temp file, avoiding OS ARG_MAX limits.
    Accepts an optional ``start_ts`` parameter: when > 0 an FFmpeg ``-ss``
    fast-seek is inserted before the input so only frames from that timestamp
    onwards are decoded, skipping the already-processed prefix entirely.

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
from collections import OrderedDict, deque
from enum import Enum
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

# HDR→SDR: libplacebo **primary** production filter (validated).
# Single GPU-shader pass via Vulkan (or software Vulkan fallback).
# Requires FFmpeg built with --enable-libplacebo.
# Uses STREAM_OPT_WIDTH × STREAM_OPT_HEIGHT (2304×1440) so that
# all GT crop families (1152×648, 960×540, 960×720) fit inside the stream.
#
# range=tv (studio-swing / limited range: Y 16-235, Cb/Cr 16-240) rather than
# range=pc (full range 0-255).  This matches the BT.709 broadcast standard and
# the expectation of downstream cv2 / PNG/BMP writers — full-range input to
# cv2.imwrite would appear washed-out on a studio-calibrated monitor.  Changed
# from the earlier 'range=pc' after visual validation confirmed range=tv
# produces the expected gamma on the reference test frames.
_LIBPLACEBO_RANGE: str = "tv"   # studio-swing / BT.709 limited range
_TONEMAP_FILTER_PLACEBO: str = (
    f"libplacebo=w={STREAM_OPT_WIDTH}:h={STREAM_OPT_HEIGHT}"
    ":colorspace=bt709:color_primaries=bt709:color_trc=bt709"
    f":range={_LIBPLACEBO_RANGE},"
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

# Default ring-buffer size cap (8 GiB).  Used as the default in both
# StreamRingBuffer.__init__ and extract_and_save_streaming_distributed.
RING_BUFFER_DEFAULT_BYTES_LIMIT: int = 8 * 1024 ** 3

# ---------------------------------------------------------------------------
# Output format enum
# ---------------------------------------------------------------------------

class OutputFormat(Enum):
    """Disk format used when writing extracted patch pairs."""
    PNG = "png"
    BMP = "bmp"


# ---------------------------------------------------------------------------
# Stream ring buffer (hard 8 GB cap, evicts oldest frames)
# ---------------------------------------------------------------------------

class StreamRingBuffer:
    """In-memory ring buffer for decoded video frames with a hard byte limit.

    When ``put`` would exceed ``bytes_limit``, the oldest stored frame (lowest
    index) is evicted first.  Uses an ``OrderedDict`` for O(1) eviction of the
    oldest entry without scanning all keys.

    Frame size defaults to ``width * height * 3 // 2`` bytes (YUV 4:2:0 packed),
    matching the raw pipe format used by FFmpeg.
    """

    def __init__(
        self,
        bytes_limit: int = RING_BUFFER_DEFAULT_BYTES_LIMIT,
        frame_size: Optional[int] = None,
        width: int = STREAM_OPT_WIDTH,
        height: int = STREAM_OPT_HEIGHT,
    ) -> None:
        self._frame_size: int = (
            frame_size if frame_size is not None else width * height * 3 // 2
        )
        self._bytes_limit: int = bytes_limit
        # OrderedDict preserves insertion order so popitem(last=False) evicts
        # the oldest (first inserted) entry in O(1) without scanning all keys.
        self._frames: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._bytes_used: int = 0

    # -- read-only properties ------------------------------------------------

    @property
    def bytes_used(self) -> int:
        return self._bytes_used

    @property
    def frames_stored(self) -> int:
        return len(self._frames)

    @property
    def mb_used(self) -> float:
        return self._bytes_used / (1024 * 1024)

    # -- mutation ------------------------------------------------------------

    def put(self, idx: int, frame: "np.ndarray") -> None:
        """Store *frame* at *idx*, evicting oldest entries as needed.

        Frames are always inserted with monotonically increasing indices so the
        OrderedDict insertion order == ascending index order.
        """
        if idx in self._frames:
            return
        # Evict oldest (first) entry until there is room.  Each iteration is
        # O(1) because popitem(last=False) removes the first key in O(1).
        while self._bytes_used + self._frame_size > self._bytes_limit and self._frames:
            self._frames.popitem(last=False)
            self._bytes_used -= self._frame_size
        self._frames[idx] = frame
        self._bytes_used += self._frame_size

    def get(self, idx: int) -> "Optional[np.ndarray]":
        """Return the frame stored at *idx*, or ``None`` if not present."""
        return self._frames.get(idx)

    def evict_before(self, min_idx: int) -> None:
        """Drop all frames with index < *min_idx* to reclaim memory."""
        to_remove = [k for k in self._frames if k < min_idx]
        for k in to_remove:
            del self._frames[k]
            self._bytes_used -= self._frame_size


# ---------------------------------------------------------------------------
# CUDA / QSV / libplacebo detection (cached after the first call)
# ---------------------------------------------------------------------------

_cuda_available: Optional[bool] = None
_scale_cuda_available: Optional[bool] = None
_tonemap_cuda_available: Optional[bool] = None
_libplacebo_avail: Optional[bool] = None
# Per-Vulkan-device probe cache: {vulkan_device_index: True/False}
# Populated by libplacebo_available() when called with a specific device index.
_libplacebo_avail_per_device: Dict[Optional[int], Optional[bool]] = {}

# Cached Vulkan device list from FFmpeg.
# Format: [(vulkan_index, description_string), …]  – populated on first call to
# _discover_vulkan_devices() and reused for all subsequent queries.
_vulkan_device_list: Optional[List[Tuple[int, str]]] = None
_qsv_avail: Optional[bool] = None
_qsv_decoders: Optional[Set[str]] = None

# Strings that indicate a Vulkan / libplacebo initialisation failure.
# Used both in the libplacebo_available() probe and in the runtime stderr
# scanner that invalidates the cache when the real FFmpeg run fails.
_VULKAN_FAIL_STRINGS: tuple = (
    "VK_ERROR_",
    "Failed creating Vulkan device",
    "Failed initializing vulkan device",
    "Failed creating logical device",
    "Query format failed",
    "Error reinitializing filters",
    "Generic error in an external library",
)

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


# Strings found in Vulkan device descriptions that identify software renderers.
# Used by is_software_vulkan_device() to warn when no real GPU is available.
_SOFTWARE_VULKAN_KEYWORDS: tuple = (
    "llvmpipe",
    "lavapipe",
    "swiftshader",
    "swrast",
    "softpipe",
)


def is_software_vulkan_device(description: str) -> bool:
    """Return True when *description* looks like a software Vulkan renderer.

    Checks for well-known software-renderer names (llvmpipe, lavapipe,
    SwiftShader, swrast, softpipe) in the device description string that
    FFmpeg reports during Vulkan device enumeration.  A ``True`` result means
    the device is not a real GPU and HDR→SDR tone-mapping will run on the CPU.

    Note: deliberately does not match "mesa" or "software" on their own because
    Mesa is also used for real hardware drivers (e.g. RADV for AMD GPUs) and
    "software" appears legitimately in many GPU feature descriptions.
    """
    desc_lower = description.lower()
    return any(kw in desc_lower for kw in _SOFTWARE_VULKAN_KEYWORDS)


def _discover_vulkan_devices() -> List[Tuple[int, str]]:
    """Return ``[(vulkan_index, description), …]`` by asking FFmpeg directly.

    Uses ``ffmpeg -init_hw_device vulkan=probe_list:list`` to enumerate all
    Vulkan-capable devices in the order FFmpeg numbers them.  This is the only
    reliable source of Vulkan device indices — CUDA/nvidia-smi ordinals are
    **not** guaranteed to match.

    The result is cached after the first call.  Returns an empty list when
    libplacebo is not compiled in, no Vulkan devices exist, or FFmpeg fails.
    """
    global _vulkan_device_list
    if _vulkan_device_list is not None:
        return _vulkan_device_list

    # Quick compiled-in check before running FFmpeg.
    if "libplacebo" not in _get_ffmpeg_filters():
        _vulkan_device_list = []
        return _vulkan_device_list

    devices: List[Tuple[int, str]] = []
    try:
        # FFmpeg prints a list of Vulkan devices to stderr when initialising
        # the dummy device "list".  Lines look like:
        #   [AVHWDeviceContext @ 0x…]  0: NVIDIA GeForce RTX 4090 (…)
        #   [AVHWDeviceContext @ 0x…]  1: NVIDIA GeForce RTX 3080 (…)
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "verbose",
             "-init_hw_device", "vulkan=probe_list:list",
             "-f", "null", "-"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=15,
        )
        text = out.stderr.decode(errors="replace")
        # Parse lines matching the "  <N>: <description>" pattern from the
        # Vulkan device enumeration block that FFmpeg writes to stderr.
        import re as _re
        for m in _re.finditer(r"^\s*(\d+)\s*:\s*(.+)$", text, _re.MULTILINE):
            idx = int(m.group(1))
            desc = m.group(2).strip()
            devices.append((idx, desc))
    except Exception:
        devices = []

    _vulkan_device_list = devices
    return _vulkan_device_list


def map_cuda_to_vulkan_device(cuda_index: int) -> Optional[int]:
    """Map a CUDA device index to the matching Vulkan device index.

    Searches the FFmpeg Vulkan device list for a description that contains the
    GPU name reported by nvidia-smi for *cuda_index*.  Returns the Vulkan index
    when a unique match is found.

    When name-based matching fails (no Vulkan devices found, GPU name not in
    any Vulkan description, or multiple ambiguous matches), the function falls
    back to a **positional best-effort**: CUDA device *N* is assumed to
    correspond to Vulkan device *N*.  The caller should validate the result via
    :func:`libplacebo_available` before use — if the device does not actually
    support Vulkan/libplacebo that probe will return ``False`` and the stream
    will fall back to CPU mode automatically.

    ``None`` is returned only when there is genuinely a single Vulkan device
    already covered by the ``len == 1`` shortcut and an unexpected code path is
    reached, or when all other strategies have been exhausted.
    """
    vulkan_devices = _discover_vulkan_devices()
    if not vulkan_devices:
        # Vulkan enumeration was empty or failed (e.g. FFmpeg output format did
        # not match the parser).  Use the CUDA index as a positional best-effort
        # so that different streams target different GPU ordinals.  The
        # libplacebo_available() per-device probe will validate before first use.
        return cuda_index

    # If there is only one Vulkan device, that is the only possible mapping
    # regardless of the CUDA index.
    if len(vulkan_devices) == 1:
        return vulkan_devices[0][0]

    # Try to get the CUDA GPU name via nvidia-smi.
    cuda_name: Optional[str] = None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu=name",
                f"--format=csv,noheader",
                f"--id={cuda_index}",
            ],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode(errors="replace").strip()
        if out:
            cuda_name = out.splitlines()[0].strip()
    except Exception:
        pass

    if not cuda_name:
        # Cannot resolve name — fall back to positional mapping.
        return cuda_index

    # Match by checking whether the CUDA GPU name is a substring of the Vulkan
    # device description (both report e.g. "NVIDIA GeForce RTX 4090").
    # Strip common prefix noise like "NVIDIA " so partial matches still work.
    _cuda_key = cuda_name.lower().replace("nvidia ", "").strip()
    matches = [
        vk_idx
        for vk_idx, vk_desc in vulkan_devices
        if _cuda_key in vk_desc.lower()
    ]
    if len(matches) == 1:
        return matches[0]

    # Multiple or no matches — fall back to positional mapping as last resort.
    # This is only reached when the Vulkan description does not contain the
    # nvidia-smi GPU name (unusual driver / vendor combination).
    # Unconditionally use cuda_index so that each CUDA device targets a
    # distinct Vulkan ordinal even when the Vulkan list is shorter than the
    # CUDA device count.
    return cuda_index


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


def libplacebo_available(
    video_path: Optional[str] = None,
    vulkan_device: Optional[int] = None,
) -> bool:
    """Return True when libplacebo is usable for the given Vulkan device.

    Two-stage check:

    1. Verify that ``libplacebo`` is listed by ``ffmpeg -filters`` (compiled-in
       with ``--enable-libplacebo``).
    2. When *video_path* is provided, decode one frame from that real HDR video
       through the libplacebo filter, optionally targeting a specific Vulkan
       device via ``-init_hw_device vulkan=vk:<vulkan_device>``.  This exercises
       exactly the same device-init code path that the extractor will use, so
       any per-device hardware or driver incompatibility surfaces here rather
       than mid-extraction.

    Results are cached per *vulkan_device* so that each stream worker's GPU is
    individually validated on first use.  Subsequent calls for the same device
    index are free.

    Args:
        video_path:    Path to a real HDR video for the Stage-2 probe.  When
                       ``None`` and no cached result exists yet, ``False`` is
                       returned conservatively.
        vulkan_device: The Vulkan device index (from :func:`_discover_vulkan_devices`
                       or :func:`map_cuda_to_vulkan_device`) that this stream
                       will use.  ``None`` means "let FFmpeg choose" and is
                       cached as a separate entry from any specific device index.
    """
    global _libplacebo_avail, _libplacebo_avail_per_device

    # Per-device cache lookup.
    cached = _libplacebo_avail_per_device.get(vulkan_device, _SENTINEL)
    if cached is not _SENTINEL:
        return cached  # type: ignore[return-value]

    # Stage 1: compiled-in check (same for all devices).
    if "libplacebo" not in _get_ffmpeg_filters():
        # Store False for all devices so Stage 1 is not repeated.
        _libplacebo_avail = False
        _libplacebo_avail_per_device[vulkan_device] = False
        return False

    # Stage 2: real-file probe.  Without a video we cannot confirm Vulkan works.
    if video_path is None:
        return False

    # Build the optional device-selection prefix for the probe command.
    _vk_init: List[str] = []
    if vulkan_device is not None:
        _vk_init = [
            "-init_hw_device", f"vulkan=vk:{vulkan_device}",
            "-filter_hw_device", "vk",
        ]

    try:
        probe = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "verbose",
                *_vk_init,
                "-probesize", "100M", "-analyzeduration", "100M",
                "-i", video_path,
                "-frames:v", "1",
                # Use the same filter string as production so that the probe
                # exercises the exact same Vulkan shader path.
                "-vf", _TONEMAP_FILTER_PLACEBO,
                "-f", "null", "-",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        stderr_txt = probe.stderr.decode(errors="replace")
        vulkan_ok = probe.returncode == 0 and not any(
            kw in stderr_txt for kw in _VULKAN_FAIL_STRINGS
        )
    except Exception:
        vulkan_ok = False

    _libplacebo_avail_per_device[vulkan_device] = vulkan_ok
    # Also update the legacy global so callers that do not pass vulkan_device
    # get a usable answer once at least one device has been probed.
    if vulkan_ok and _libplacebo_avail is None:
        _libplacebo_avail = True
    elif not vulkan_ok and _libplacebo_avail is None:
        _libplacebo_avail = False

    return vulkan_ok


# Sentinel used by the per-device cache to distinguish "not probed yet" from False.
_SENTINEL = object()


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
                    width: int = STREAM_WIDTH, height: int = STREAM_HEIGHT,
                    color_trc: str = "smpte2084",
                    vulkan_device: Optional[int] = None) -> str:
    """Return the FFmpeg ``-vf`` filter string for the given video type.

    Selects the best available pipeline tier at call time.  **Priority order
    for HDR sources:**

    1. libplacebo (primary production path) — single-pass Vulkan shader,
       validated on the specific *vulkan_device* index, most reliable HDR→SDR.
    2. full-GPU CUDA (optional fallback) — tonemap_cuda + scale_cuda.
    3. scale-GPU CUDA (optional fallback) — scale_cuda + CPU zscale/tonemap.
    4. CPU-only zscale + tonemap (final fallback).

    SDR sources:

    * scale-GPU → scale_cuda pipeline
    * CPU-only  → bilinear scale pipeline

    All paths output ``yuv420p`` — ~33 % less pipe bandwidth compared to
    ``bgr24``.  Python converts with ``cv2.cvtColor(yuv, COLOR_YUV2BGR_I420)``.

    Args:
        is_hdr:        Whether the source video is HDR (PQ or HLG transfer).
        use_cuda:      Whether CUDA acceleration is requested.  Still falls back
                       to CPU-only when the local FFmpeg has no CUDA support.
        width:         Output width in pixels (default ``STREAM_WIDTH`` = 1920).
        height:        Output height in pixels (default ``STREAM_HEIGHT`` = 1080).
        color_trc:     Transfer function string from ffprobe (e.g. ``"smpte2084"``
                       for HDR10/PQ, ``"arib-std-b67"`` for HLG).  Used to
                       annotate the explicit ``tin=`` parameter of ``zscale`` in
                       the scale-GPU path so that the correct HDR→linear
                       conversion is applied even when CUDA pipeline stages do not
                       reliably propagate frame colour metadata through to CPU.
        vulkan_device: Vulkan device index (from :func:`_discover_vulkan_devices`)
                       to validate when probing libplacebo availability.  ``None``
                       lets FFmpeg choose any available Vulkan device.

    Returns:
        FFmpeg filter string ready for ``-vf`` (or for wrapping in
        ``-filter_complex`` as ``[0:v]<filter>[label]``).
    """
    _use_cuda = use_cuda and cuda_available()
    _full_gpu  = _use_cuda and tonemap_cuda_available()
    _scale_gpu = _use_cuda and (not _full_gpu) and scale_cuda_available()

    # Normalise color_trc to the vocabulary understood by zscale/zimg.
    # "hlg" is a common shorthand used by some encoders; zscale's canonical
    # name for the HLG transfer is "arib-std-b67" (ARIB STD-B67).
    _zscale_trc = (color_trc or "smpte2084").strip().lower()
    if _zscale_trc == "hlg":
        _zscale_trc = "arib-std-b67"  # map shorthand to zscale's expected identifier

    if is_hdr:
        # Primary: libplacebo — validated production filter, single shader pass.
        # The per-device check ensures the specific Vulkan device is usable;
        # passing vulkan_device=None falls back to the global (any-device) probe.
        if libplacebo_available(vulkan_device=vulkan_device):
            return (
                f"libplacebo=w={width}:h={height}"
                ":colorspace=bt709:color_primaries=bt709:color_trc=bt709"
                f":range={_LIBPLACEBO_RANGE},"
                "format=yuv420p"
            )
        # Optional fallback: full-GPU CUDA tonemap + scale.
        if _full_gpu:
            return (
                f"tonemap_cuda=tonemap=mobius:desat=0:peak=100,"
                f"scale_cuda={width}:{height}:interp_algo=bicubic,"
                "hwdownload,"
                "scale=iw:ih,"
                "format=yuv420p"
            )
        # Optional fallback: scale on GPU, tonemap on CPU.
        if _scale_gpu:
            # After hwdownload the CUDA pipeline may not reliably propagate
            # HDR frame metadata (color_trc, colorspace) to the CPU frame.
            # Specifying tin= and primariesin= explicitly in zscale ensures the
            # correct PQ/HLG→linear conversion regardless of frame metadata.
            return (
                f"scale_cuda={width}:{height}:interp_algo=bicubic,"
                "hwdownload,"
                "format=p010,"
                f"zscale=tin={_zscale_trc}:primariesin=bt2020:t=linear:npl=100:filter=bilinear,"
                "format=gbrpf32le,"
                "zscale=p=bt709:filter=bilinear,"
                "tonemap=tonemap=reinhard:desat=0,"
                "zscale=t=bt709:m=bt709:range=full:filter=bilinear,"
                "format=yuv420p"
            )
        # Final fallback: pure CPU zscale + tonemap chain.
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
        # Bilinear is adequate for the intermediate scale step —
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


def build_remaining_assignments(
    format_distribution: Dict[str, Dict[str, int]],
    completed_per_category: Dict[str, int],
    completed_per_format_template: Dict[str, int],
    duration: float,
    fps: float,
    n_frames: int = 7,
) -> Tuple[float, List[Tuple[int, str, str]]]:
    """
    Build streaming assignments for the REMAINING unfinished work only.

    This is the resume-aware counterpart to
    :func:`build_assignments_per_category`.  It reads already-completed patch
    counts from the persisted plan, subtracts them from the configured targets
    to compute a *remaining distribution*, then estimates a resume timestamp
    based on the fraction of work already done.  Only frames **after** that
    timestamp are assigned, so the caller can combine the returned
    ``resume_ts`` with the ``start_ts`` parameter of
    :func:`extract_and_save_streaming_distributed` to let FFmpeg fast-seek
    past the already-processed portion — avoiding redundant decoding and
    leaving all previously generated patches intact.

    Behaviour
    ---------
    * If no patches are completed yet (``total_completed == 0``), falls back
      to :func:`build_assignments_per_category` (full plan, no seek).
    * The resume timestamp is estimated as
      ``(total_completed / total_planned) × duration``.  This assumes patches
      were extracted at approximately uniform density across the video, which
      is true for the default even-spacing strategy.
    * Per-format remaining counts use stored ``completed_per_format_template``
      entries when available; missing entries are estimated proportionally from
      the category-level completion ratio.
    * This function does **not** touch any files or the plan on disk — it is a
      pure planning utility.  The caller is responsible for accumulating the
      new completed counts with prior counts when updating the plan.

    Args:
        format_distribution:           ``{category: {format_name: target_count}}``.
        completed_per_category:        Already-done patch counts per category,
                                       from
                                       ``plan_item["completed"]["per_category"]``.
        completed_per_format_template: Already-done patch counts per format
                                       template (summed across categories), from
                                       ``plan_item["completed"]["per_format_template"]``.
        duration:                      Video duration in seconds.
        fps:                           Video frame rate.
        n_frames:                      Frames per patch window (default 7).

    Returns:
        ``(resume_ts, assignments)`` where *resume_ts* is the estimated resume
        timestamp in seconds (0.0 when no prior completion) and *assignments*
        is a sorted list of ``(center_frame_idx, category, format_name)``
        tuples with frame indices **absolute** (relative to the video start,
        not to *resume_ts*).  Pass *resume_ts* as ``start_ts`` to
        :func:`extract_and_save_streaming_distributed` and the extractor will
        seek to that position automatically.
    """
    total_planned = sum(
        cnt for formats in format_distribution.values() for cnt in formats.values()
    )
    total_completed = sum(completed_per_category.values())

    if total_planned <= 0 or total_completed <= 0:
        # Nothing completed yet — full plan from the start, no seek.
        return 0.0, build_assignments_per_category(
            format_distribution, duration, fps, n_frames
        )

    if total_completed >= total_planned:
        # All work already done — nothing to assign.
        return duration, []

    # --- Estimate resume timestamp ----------------------------------------
    # Clamped to [0, 0.99] to prevent rounding artefacts from placing the
    # resume point at or past the video end.
    resume_fraction = min(0.99, total_completed / total_planned)
    resume_ts = resume_fraction * duration
    remaining_duration = duration - resume_ts

    # --- Build remaining distribution per category/format ----------------
    remaining_distribution: Dict[str, Dict[str, int]] = {}
    for category, formats in format_distribution.items():
        cat_remaining: Dict[str, int] = {}
        for fmt, target in formats.items():
            # Prefer stored per-format completed counts; fall back to a
            # proportional estimate when per-format data is unavailable.
            completed_fmt = completed_per_format_template.get(fmt, 0)
            if completed_fmt == 0 and total_planned > 0:
                fmt_fraction = target / total_planned
                completed_fmt = int(total_completed * fmt_fraction)
            remaining = max(0, target - completed_fmt)
            if remaining > 0:
                cat_remaining[fmt] = remaining
        if cat_remaining:
            remaining_distribution[category] = cat_remaining

    if not remaining_distribution:
        return resume_ts, []

    # --- Generate assignments for the remaining [resume_ts, duration] ----
    # build_assignments_per_category produces indices relative to a video
    # that starts at 0 and has duration `remaining_duration`.  Offset all
    # returned frame indices by `resume_frame_offset` to get absolute
    # positions within the original video so the caller gets a consistent
    # frame-index space regardless of whether a seek is used.
    sub_asgn = build_assignments_per_category(
        remaining_distribution, remaining_duration, fps, n_frames
    )
    resume_frame_offset = int(resume_ts * fps)
    abs_asgn: List[Tuple[int, str, str]] = [
        (fi + resume_frame_offset, cat, fmt)
        for fi, cat, fmt in sub_asgn
    ]
    return resume_ts, sorted(abs_asgn, key=lambda x: x[0])


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


# Keys that belong to the post-stack color/intensity adjustment stage.
# Centralised here so both _apply_degrade_template_poststack and any future
# caller that needs to check for active color stages use the same definition.
_COLOR_ADJUSTMENT_KEYS: Tuple[str, ...] = (
    "contrast", "brightness", "gamma", "black_lift"
)


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
      3. JPEG        – blocking / ringing artefacts (JPEG quality round-trip).
      4. Chroma bleed – horizontal Cr/Cb smearing simulating analog bandwidth.
      5. Saturation  – chroma scaling in HSV space.
      6. Color       – contrast, brightness, gamma, black-lift in floating point.

    GT frames are never passed through this function; only LR frames are degraded.

    .. note::
        Inside :func:`create_patch_pair` the pipeline is split into a
        **pre-stack** stage (:func:`_apply_degrade_template_prestack`, stages
        1–4) and a **post-stack** stage
        (:func:`_apply_degrade_template_poststack`, stages 5–6) so that the
        global color/intensity adjustments are applied once on the stacked LR
        image instead of once per frame.  This function remains available for
        single-frame use cases.

    Args:
        frame:  Single LR BGR frame (uint8 numpy array).
        params: Dict produced by :func:`sample_degradation_template_params`.

    Returns:
        Degraded frame as uint8 numpy array.
    """
    result = _apply_degrade_template_prestack(frame, params)
    result = _apply_degrade_template_poststack(result, params)
    return result


def _apply_degrade_template_prestack(
    frame: np.ndarray,
    params: dict,
) -> np.ndarray:
    """
    Apply the spatially sensitive (pre-stack) degradation stages to a single LR frame.

    These stages must remain per-frame because applying them to the vertically
    stacked LR image would produce incorrect cross-frame artifacts:

    * **Blur** – Gaussian kernel would smear pixel rows across frame boundaries.
    * **Noise** – each frame must receive independent noise samples (same sigma,
      different draws); merging into one post-stack draw would alter the
      statistical independence between frames.
    * **JPEG** – the DCT codec operates on 8×8 blocks; encoding the full stack
      would generate blocking artifacts that span frame boundaries.
    * **Chroma bleed** – horizontal Gaussian blur in YCrCb space; frame-boundary
      rows would contaminate adjacent frames.

    The global color/intensity stages (saturation, contrast, brightness, gamma,
    black_lift) are intentionally absent here; they are applied once to the
    stacked LR image by :func:`_apply_degrade_template_poststack`.

    Args:
        frame:  Single LR BGR frame (uint8 numpy array).
        params: Dict produced by :func:`sample_degradation_template_params`.

    Returns:
        Degraded frame as uint8 numpy array.

    .. note::
        When no pre-stack stage is active the function returns *frame* directly
        (no copy).  This is safe inside :func:`create_patch_pair` because
        *frame* is always the result of a ``cv2.resize`` call (which allocates
        a fresh array), so the caller never holds an alias to the original
        source data.  If you call this function outside of that context and need
        a guaranteed independent copy, copy the input beforehand.
    """
    result = frame  # avoid upfront copy – each active stage returns a new array

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
                # Luma noise: identical value replicated across channels
                gray_noise = np.random.normal(0.0, sigma, result.shape[:2]).astype(np.float32)
                noise = np.stack([gray_noise, gray_noise, gray_noise], axis=2)
            result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # ── 3. JPEG round-trip ───────────────────────────────────────────────────
    if "jpeg_quality" in params:
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, int(params["jpeg_quality"])]
        ok, buf = cv2.imencode(".jpg", result, encode_param)
        if ok:
            result = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # ── 4. Chroma bleed (analog horizontal chroma smearing) ──────────────────
    # Simulates the lower chroma bandwidth of analog / early digital TV:
    # Cb and Cr channels are blurred horizontally proportional to the strength.
    bleed = float(params.get("chroma_bleed", 0.0))
    if bleed > 0.0:
        ycrcb = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        # kernel width scales with strength: bleed=0.08 → kw=3, bleed=0.3 → kw=7
        kw = max(3, int(bleed * 24 + 1) | 1)   # must be odd
        for ch in (1, 2):  # Cr (ch=1), Cb (ch=2)
            ycrcb[:, :, ch] = cv2.GaussianBlur(
                ycrcb[:, :, ch], (kw, 1), sigmaX=0
            )
        result = cv2.cvtColor(
            np.clip(ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR
        )

    return result


def _apply_degrade_template_poststack(
    lr_stacked: np.ndarray,
    params: dict,
) -> np.ndarray:
    """
    Apply global color/intensity degradation stages to the stacked LR image.

    These stages are safe to run **once** on the stacked image (H×N × W × 3)
    instead of once per individual frame (H × W × 3) because they are purely
    per-pixel operations with no spatial neighbourhood that could produce
    cross-frame contamination:

    * **Saturation** – per-pixel HSV S-channel scaling; identical result whether
      applied per-frame or to the full stack.
    * **Contrast / brightness / gamma / black_lift** – per-pixel linear and
      power operations; same argument as saturation.

    Running these stages once on the stacked image instead of N times on
    individual frames (N = 5 or 7) eliminates:

    * N−1 BGR↔HSV color-space round-trips (saved per scene: up to 6 × 2 = 12).
    * N−1 float32 passes for the color adjustment (saved: up to 6).
    * N−1 astype() calls (saved: up to 12).

    Args:
        lr_stacked: Vertically stacked LR image (H*N × W × 3, uint8).
        params:     Dict produced by :func:`sample_degradation_template_params`.

    Returns:
        Adjusted stacked LR image (uint8).  Returns *lr_stacked* unchanged
        (no copy) when neither the saturation nor the color stage is active.
    """
    sat = params.get("saturation", 1.0)
    has_sat = float(sat) != 1.0
    has_color = any(k in params for k in _COLOR_ADJUSTMENT_KEYS)

    if not has_sat and not has_color:
        return lr_stacked

    result = lr_stacked

    # ── Saturation ────────────────────────────────────────────────────────────
    if has_sat:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(sat), 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ── Color: contrast / brightness / gamma / black-lift ─────────────────────
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

    **Resize mode** (``source_mode == "resize"`` in format_cfg):
      * GT – full-frame resize to ``gt_size`` with ``INTER_LANCZOS4``
        (best quality, preserves full frame content).
      * LR – full-frame resize to ``lr_size`` with ``INTER_AREA``
        (DVD-realistic quality).
      * Suitable for any aspect-ratio target (16:9, 4:3, etc.) when the
        source and target aspect ratios are compatible and no spatial crop is
        desired.

    **Crop mode** (``source_mode == "crop"`` in format_cfg):
      * GT – centre frame.  When the source frame is large enough for 2×
        oversampling (``frame_h ≥ 2*gt_h`` **and** ``frame_w ≥ 2*gt_w``,
        e.g. native 4K for 1152×648 or 960×720 GT targets), a ``2*gt_size``
        crop is taken from the source and Lanczos4-downsampled to ``gt_size``.
        The 2× Lanczos4 step averages H.265 in-loop deblocking softness
        and produces a clean GT comparable to the full-frame resize path.
        For smaller sources a direct 1:1 crop is used instead.
      * LR – all frames, same crop region, downscaled to ``lr_size`` with
        ``INTER_AREA`` (stacked vertically on axis 0).
      * Works correctly for any crop target aspect ratio — decisions are
        purely dimension-driven, not format-name-driven.

    **Source-mode resolution**:
    The ``source_mode`` field in *format_cfg* is **required** and is the sole
    source of truth.  It is validated by ``config_io.validate_active_config``
    at startup.  If it is absent or invalid a ``ValueError`` is raised
    immediately so the misconfiguration is never silently hidden.

    **Degradation**:
    Two degradation paths are supported, applied in this priority order:

    1. *New template-based* (``deg_spec`` arg): parameters are sampled via
       :func:`sample_degradation_template_params` once per scene and applied
       in a **split pre/post-stack pipeline**:

       * **Pre-stack** (:func:`_apply_degrade_template_prestack`) – applied to
         each individual LR frame: blur, noise, JPEG, chroma bleed.  These
         stages are spatially sensitive and must not span frame boundaries.
       * **Post-stack** (:func:`_apply_degrade_template_poststack`) – applied
         once to the vertically stacked LR image: saturation and global color
         adjustments (contrast, brightness, gamma, black_lift).  These are
         pure per-pixel operations that are safe on the stack, and running
         them once instead of N times eliminates N−1 color-space round-trips
         per scene.

    2. *Legacy flat* (``degrade_cfg`` arg): used when ``deg_spec`` is
       ``None``; forwards to :func:`_sample_degrade_params` /
       :func:`_apply_degrade_params` for backward compatibility.  All stages
       (noise, blur, JPEG) are applied per-frame.

    In both cases parameters are **sampled once per scene** so that all LR
    frames in the window share the same settings — consistent with real
    MPEG-2 encoder behaviour.  GT is always kept lossless.

    A near-uniform GT (plain black, white, or flat colour) is silently
    discarded (``(None, None)``).  If the source frame is too small for the
    requested resize target a warning is logged.

    Args:
        frames:       BGR numpy arrays, odd length >= 3.
        format_name:  Format key string (used for logging only — all functional
                      decisions are driven by *format_cfg*).
        format_cfg:   Dict with at minimum ``'gt_size': [W, H]``,
                      ``'lr_size': [W, H]``, and ``'source_mode': "resize"|"crop"``.
        force_center: Crop mode only – use the geometric centre of the frame
                      instead of a random crop location.
        logger:       Optional logger instance for warning messages.
        degrade_cfg:  Legacy degradation config dict.  Ignored when *deg_spec*
                      is provided.  When ``None`` no legacy degradation is
                      applied.
        deg_spec:     Template-based degradation dict (from
                      ``templates["degradation_templates"]``).  When
                      provided, *degrade_cfg* is ignored.

    Returns:
        ``(gt, lr_stacked)`` or ``(None, None)`` on failure.

    Raises:
        ValueError: When ``format_cfg["source_mode"]`` is missing or invalid.
    """
    n = len(frames)
    if n < 3 or n % 2 == 0:
        return None, None

    gt_w, gt_h = format_cfg["gt_size"]
    lr_w, lr_h = format_cfg["lr_size"]

    frame_h, frame_w = frames[0].shape[:2]

    center_idx = n // 2

    # ── Determine source_mode ─────────────────────────────────────────────────
    # source_mode is always present and already validated by config_io.py at
    # startup.  If it is missing or invalid here, the config–runtime contract
    # has been broken: raise immediately rather than hiding the error with a
    # silent default.
    source_mode = format_cfg.get("source_mode")
    if source_mode not in ("resize", "crop"):
        raise ValueError(
            f"create_patch_pair: format_cfg is missing a valid 'source_mode' "
            f"(got {source_mode!r}).  Expected 'resize' or 'crop'.  "
            f"Ensure the format config is built via _build_format_config() "
            f"and that config_io.validate_active_config() has passed."
        )

    # ── Degradation: resolve which sampler / apply functions to use ──────────
    # deg_spec (new template) takes priority over degrade_cfg (legacy flat cfg).
    #
    # Template path uses a split pre/post-stack design:
    #   _apply_fn    → per-frame (blur, noise, JPEG, chroma bleed)
    #   _poststack_fn → once on stacked LR (saturation, color adjustments)
    #
    # Legacy path keeps all stages per-frame (blur, noise, JPEG only – none of
    # those are safe to move post-stack given their spatial nature).
    center_raw = frames[center_idx]
    if deg_spec is not None:
        # New template-based degradation – sample once per scene.
        _scene_params = sample_degradation_template_params(deg_spec, center_frame=center_raw)
        _apply_fn = _apply_degrade_template_prestack
        _poststack_fn = _apply_degrade_template_poststack
    elif degrade_cfg is not None:
        # Legacy degradation – all stages are per-frame (spatial).
        _scene_params = _sample_degrade_params(degrade_cfg, center_frame=center_raw)
        _apply_fn = _apply_degrade_params
        _poststack_fn = None
    else:
        _scene_params = None
        _apply_fn = None
        _poststack_fn = None

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

    # ── Post-stack degradation (template path only) ───────────────────────────
    # Saturation and global color adjustments are applied once here on the full
    # stacked LR image instead of once per frame.  For 7 frames this eliminates
    # 6 × (BGR↔HSV round-trip + float32 color pass) per scene.
    if _poststack_fn is not None and _scene_params is not None:
        lr_stacked = _poststack_fn(lr_stacked, _scene_params)

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
    output_format: OutputFormat = OutputFormat.PNG,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Persist a ``(GT, LR)`` patch pair to the correct output directories.

    Directories are created on demand.  PNG patches are written at compression
    level 1 (fast).  BMP patches are written uncompressed for maximum write
    throughput at the cost of ~3× larger files.

    Args:
        gt:           Ground-truth patch (BGR numpy array).
        lr:           LR stack patch (BGR numpy array).
        video_path:   Source video path (stem used in the patch filename).
        timestamp:    Center-frame timestamp in seconds (used in filename).
        category:     Dataset category (e.g. ``"master"``).
        format_name:  Format key (e.g. ``"small_540"``).
        n_frames:     Number of frames (5 or 7) – selects LR subdirectory.
        base_dir:     Root dataset output directory.
        output_format: Disk format for the patch images (PNG or BMP).

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
        ext = output_format.value
        patch_name = f"{video_stem}_{int(timestamp * 1000):08d}.{ext}"

        gt_path = os.path.join(gt_dir, patch_name)
        lr_path = os.path.join(lr_dir, patch_name)

        if output_format is OutputFormat.BMP:
            cv2.imwrite(gt_path, gt)
            cv2.imwrite(lr_path, lr)
        else:
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
    color_trc: str = "smpte2084",
    vulkan_device: Optional[int] = None,
    output_format: OutputFormat = OutputFormat.PNG,
    ring_buffer_bytes_limit: int = RING_BUFFER_DEFAULT_BYTES_LIMIT,
    start_ts: float = 0.0,
) -> Dict[str, int]:
    """
    Stream the video once and save patches as frames pass through the buffer.

    A single FFmpeg process reads the video linearly.  When *start_ts* > 0 an
    FFmpeg ``-ss`` fast-seek is inserted before the input so that only frames
    from *start_ts* onwards are decoded — the already-processed prefix of the
    video is skipped entirely and all previously generated patches are
    preserved untouched.  A rolling dictionary buffer keeps the last
    ``n_frames`` decoded frames in memory.  When a target centre frame has
    been decoded and all ``n_frames`` of its window are in the buffer, the
    patch is created and saved immediately.

    The stream is terminated early once the last needed frame has been read.

    Args:
        video_path:          Path to input video.
        assignments:         Output of :func:`build_assignments_per_category`
                             or :func:`build_remaining_assignments`.  Frame
                             indices must be **absolute** (relative to the
                             start of the video, not to *start_ts*).
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
                             available in the local FFmpeg build.
        color_trc:           Transfer-function string from ffprobe for the source
                             video (e.g. ``"smpte2084"`` for HDR10/PQ,
                             ``"arib-std-b67"`` for HLG).  Forwarded to
                             :func:`build_vf_filter` so the scale-GPU HDR path
                             can use explicit ``tin=`` / ``primariesin=`` zscale
                             parameters, which prevents misidentification of the
                             source transfer function when HDR frame metadata is
                             not reliably propagated through the CUDA pipeline.
                             Defaults to ``"smpte2084"`` (HDR10/PQ).
        vulkan_device:       Vulkan device index obtained from
                             :func:`_discover_vulkan_devices` / :func:`map_cuda_to_vulkan_device`
                             to pass to FFmpeg as ``-init_hw_device vulkan=vk:<n>``
                             when the libplacebo pipeline is active.
                             **Important**: this is the FFmpeg Vulkan device index,
                             which may differ from the CUDA / nvidia-smi index.
                             Use :func:`map_cuda_to_vulkan_device` to translate
                             between the two numbering schemes.
                             Pass ``None`` (default) to let FFmpeg choose any
                             available Vulkan device automatically.
        output_format:       Disk format for saved patch images.
                             :attr:`OutputFormat.PNG` (default) writes PNG at
                             compression level 1.  :attr:`OutputFormat.BMP`
                             writes uncompressed BMP for maximum write
                             throughput (≈3× larger files).
        ring_buffer_bytes_limit: Hard memory cap for the internal frame ring
                             buffer in bytes (default 8 GiB).  Oldest frames
                             are evicted when the limit is reached.
        start_ts:            Resume seek offset in seconds (default 0.0 =
                             start from the beginning).  When > 0 an FFmpeg
                             ``-ss`` fast-seek is added before the input so
                             only frames from this position onwards are
                             decoded.  Assignments must use absolute frame
                             indices (as returned by
                             :func:`build_remaining_assignments`); the
                             extractor subtracts the offset internally so
                             the 0-based frame counter in the read loop
                             matches correctly.  Patch filenames always use
                             the absolute video timestamp so they are unique
                             even across multiple partial runs.

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

    # --- Resume seek: adjust frame indices to be relative to start_ts ----
    # When start_ts > 0 the caller supplies absolute frame indices (relative
    # to the video start) but FFmpeg will output frames beginning at 0 after
    # the fast seek.  Subtract the seek offset from every assignment so the
    # 0-based _actual_frame counter in the read loop matches the adjusted
    # pending_centers.  The original absolute offset is kept in
    # start_frame_offset for use in patch filenames (so every patch has a
    # unique name based on its true video timestamp, not its position within
    # the remaining portion).
    start_frame_offset: int = max(0, int(start_ts * fps)) if start_ts > 0.0 else 0

    # Build mapping: center_frame_idx → [(category, format_name), …]
    # Frame indices are adjusted to be relative to the seek point.
    center_map: Dict[int, List[Tuple[str, str]]] = {}
    for frame_idx, category, fmt_name in sorted_asgn:
        rel_idx = frame_idx - start_frame_offset
        center_map.setdefault(rel_idx, []).append((category, fmt_name))

    pending_centers: List[int] = sorted(center_map.keys())
    last_needed: int = pending_centers[-1] + half if pending_centers else 0

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

    # --- Async write queue ------------------------------------------------
    # Patch writing is off-loaded to background threads so that disk I/O
    # overlaps with FFmpeg decode.  Use 2 writer threads to fill both GT and
    # LR paths in parallel.  A bounded queue provides back-pressure when the
    # disk is slower than the CPU.
    _png_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    _use_bmp = output_format is OutputFormat.BMP
    _write_queue: queue.Queue = queue.Queue(maxsize=256)

    def _write_worker() -> None:
        while True:
            item = _write_queue.get()
            if item is None:
                _write_queue.task_done()
                break
            gt_img, lr_img, gt_p, lr_p = item
            try:
                if _use_bmp:
                    cv2.imwrite(gt_p, gt_img)
                    cv2.imwrite(lr_p, lr_img)
                else:
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

    # --- Async processing queue -------------------------------------------
    # 7-frame windows captured by the main reader are handed to processing
    # workers which run: black-frame check → crop/resize → degradation →
    # GT/LR assembly → enqueue result to the write queue.
    # This keeps the FFmpeg pipe reader as light as possible (read → convert
    # → buffer → snapshot → enqueue → continue) without blocking it on heavy
    # Python/OpenCV image work.
    # maxsize=32: at 2304×1440 BGR (~10 MB/frame) a 7-frame window is ~70 MB;
    # 32 slots ≈ 2 GB — provides back-pressure without unbounded memory use.
    _process_queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=32)

    def _processing_worker() -> None:
        """Pop captured 7-frame windows and produce GT/LR pairs asynchronously."""
        while True:
            item = _process_queue.get()
            if item is None:
                _process_queue.task_done()
                break
            # Track how many workers are currently busy.
            with _patches_lock:
                _active_workers_ctr[0] += 1
                _t_phases["n_workers_active"] = _active_workers_ctr[0]
            try:
                center, window_frames, cat_fmt_list = item
                # center is a relative frame index (0-based from seek point).
                # Add start_frame_offset to get the absolute video timestamp
                # so that patch filenames are unique across partial runs and
                # remain consistent with patches written in earlier runs.
                ts = (center + start_frame_offset) / fps
                center_raw = window_frames[n_frames // 2]

                # --- Black-frame check (once per extraction point) --------
                if _black_fn(center_raw):
                    with _patches_lock:
                        _n_black_ctr[0] += 1
                    if logger:
                        logger.info(
                            f"  ⏭ frame {center + start_frame_offset} "
                            f"(ts {ts:.2f}s) skipped (black frame)"
                        )
                    # Do NOT call task_done() here — the finally block handles it.
                    continue

                _any_patch_saved = False
                for category, fmt_name in cat_fmt_list:
                    cfg = format_config.get(category, {}).get(fmt_name, {})
                    if not cfg:
                        continue

                    _source_mode = cfg.get("source_mode", "crop")
                    if _source_mode not in ("resize", "crop"):
                        _source_mode = "crop"
                    # Resize mode always produces the same result — no benefit
                    # in retrying a random crop.
                    max_attempts = 1 if _source_mode == "resize" else 6

                    # --- Per-format degradation template selection ---------
                    _deg_spec: Optional[dict] = None
                    _chosen: Optional[str] = None
                    _deg_mix = cfg.get("degradation_mix")
                    _deg_tmpls = cfg.get("degradation_templates")
                    if _deg_mix and _deg_tmpls:
                        _names = list(_deg_mix.keys())
                        _weights = [float(_deg_mix[k]) for k in _names]
                        _chosen = random.choices(_names, weights=_weights, k=1)[0]
                        _deg_spec = _deg_tmpls.get(_chosen)

                    # --- Crop / resize → degradation → GT/LR assembly -----
                    gt, lr = None, None
                    for attempt in range(max_attempts):
                        force = attempt >= 5
                        gt, lr = create_patch_pair(
                            window_frames, fmt_name, cfg,
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

                    # --- Enqueue result for async disk writing -------------
                    if gt is not None and lr is not None:
                        _any_patch_saved = True
                        dirs = _output_dirs_cache[(category, fmt_name)]
                        _ext = output_format.value
                        patch_name = f"{_video_stem}_{int(ts * 1000):08d}.{_ext}"
                        _write_queue.put((
                            gt, lr,
                            os.path.join(dirs["gt"], patch_name),
                            os.path.join(dirs["lr"], patch_name),
                        ))
                        with _patches_lock:
                            patches_created[category] = (
                                patches_created.get(category, 0) + 1
                            )
                            _t_phases["n_patches"] += 1
                            if _chosen is not None:
                                _dc = _t_phases["degrade_counts"]
                                _dc.setdefault(category, {})
                                _dc[category][_chosen] = (
                                    _dc[category].get(_chosen, 0) + 1
                                )

                if not _any_patch_saved:
                    with _patches_lock:
                        _n_quality_fail_ctr[0] += 1

            except Exception as _exc:
                if logger:
                    logger.warning(f"[processing_worker] Error: {_exc!r}")
            finally:
                with _patches_lock:
                    _active_workers_ctr[0] = max(0, _active_workers_ctr[0] - 1)
                    _t_phases["n_workers_active"] = _active_workers_ctr[0]
                _process_queue.task_done()

    # Processing workers: image work (crop, degradation, PNG encode) is
    # CPU-bound.  Scale with available CPU cores, capped at 8 to avoid
    # excessive memory pressure from concurrent 7-frame window copies.
    # Computed here so it can be stored in _t_phases (defined below) without
    # a forward-reference.
    _n_processing_workers = min(8, os.cpu_count() or 4)
    _processing_threads = [
        threading.Thread(target=_processing_worker, daemon=True)
        for _ in range(_n_processing_workers)
    ]
    for _pt in _processing_threads:
        _pt.start()

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

    # Pre-warm the libplacebo runtime probe for the specific Vulkan device
    # this stream will use.  Without this call the per-device cache is empty
    # and libplacebo_available(vulkan_device=…) inside build_vf_filter returns
    # False conservatively, causing the filter to fall back to the zscale chain
    # even when libplacebo is actually available.  The probe result is cached
    # per device after this first call, so all subsequent calls are free.
    if is_hdr:
        libplacebo_available(video_path, vulkan_device=vulkan_device)

    vf_filter = build_vf_filter(
        is_hdr=is_hdr, use_cuda=use_cuda,
        width=stream_width, height=stream_height,
        color_trc=color_trc,
        vulkan_device=vulkan_device,
    )

    # -init_hw_device cuda=hw explicitly initialises the CUDA device context
    # before demuxing begins.  Without this flag some FFmpeg builds silently
    # fall back to software decoding when the GPU context fails to auto-init,
    # causing the GPU filter chain to receive CPU frames and crash.
    # Device index is parameterised so callers can target a specific GPU
    # (e.g. GPU 1 on a dual-GPU system) without setting CUDA_VISIBLE_DEVICES.
    _CUDA_HW_INIT = ["-init_hw_device", f"cuda=hw:{cuda_device}"]

    hdr_label = "HDR" if is_hdr else "SDR"
    # Use the per-device cached result (already populated by the pre-warm call above).
    _placebo = is_hdr and (not _full_gpu) and (not _scale_gpu) and libplacebo_available(
        video_path, vulkan_device=vulkan_device
    )
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
        # Inject Vulkan device selection when libplacebo is active and the
        # caller specified a device index for round-robin GPU assignment.
        if _placebo and vulkan_device is not None:
            hw_args = [
                "-init_hw_device", f"vulkan=vk:{vulkan_device}",
                "-filter_hw_device", "vk",
            ]

    # Pipe bandwidth for the log (yuv420p = 1.5 bytes/pixel).
    _pipe_mb_per_frame = stream_width * stream_height * 1.5 / (1024 * 1024)

    _log(
        f"🎬 Streaming extractor: {len(sorted_asgn)} assignments, "
        f"{len(pending_centers)} unique centers, "
        f"last rel. frame needed: {last_needed}"
        + (f" (abs {last_needed + start_frame_offset})" if start_frame_offset else "")
        + f", stream={stream_width}×{stream_height}, "
        f"pipeline={pipeline_label}, nice={nice_level}"
    )
    if start_frame_offset > 0:
        _log(
            f"🎯 Mode: resume stream — FFmpeg fast-seek to {start_ts:.2f}s "
            f"(≈ abs frame {start_frame_offset}), "
            f"reads rel frames 0..{last_needed} "
            f"(abs {start_frame_offset}..{start_frame_offset + last_needed}), "
            f"snapshots {len(pending_centers)} {n_frames}-frame windows"
        )
    else:
        _log(
            f"🎯 Mode: full stream from start — "
            f"reads frames 0..{last_needed} ({last_needed + 1} total), "
            f"snapshots {len(pending_centers)} {n_frames}-frame windows asynchronously"
        )
    _log(
        f"📦 Pipe: yuv420p {stream_width}×{stream_height} "
        f"= {_pipe_mb_per_frame:.2f} MB/frame"
    )

    # Write the filter chain to a temp file to avoid exceeding the OS
    # ARG_MAX limit (~2 MB on Linux) for very long filter expressions.
    # -filter_complex_script reads from a file and has no length restriction.
    # We wrap the vf-style filter in a minimal filter_complex graph:
    # [0:v]<filter>[vout], then map [vout] to output.
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

    # Rejection counters: mutable containers so processing workers can update
    # them without needing to hold the GIL or change the nonlocal declaration.
    _n_black_ctr: List[int] = [0]         # centers skipped by black-frame check
    _n_quality_fail_ctr: List[int] = [0]  # centers where create_patch_pair returned None

    # Lock protecting patches_created, _n_black_ctr, _n_quality_fail_ctr,
    # and _t_phases["degrade_counts"] / _t_phases["n_patches"] which are
    # written by processing worker threads and read by the main thread.
    _patches_lock = threading.Lock()

    # Active-worker counter: how many processing workers are currently busy.
    # Updated inside _processing_worker under _patches_lock.
    _active_workers_ctr: List[int] = [0]

    # --- Per-video timing accumulators (mutated inside _consume_raw_frame) ---
    _t_phases: dict = {
        "n_frames_buf":      0,   # total raw frames processed through buffer
        "n_centers":         0,   # centers fully evaluated (= frames_examined, incl. black)
        "t_buf_s":           0.0, # total time: yuv→bgr convert + copy + buffer insert/evict
        "t_black_s":         0.0, # total time: black-frame check (per center)
        "t_patch_s":         0.0, # total time: create_patch_pair calls (per center×format)
        "t_write_s":         0.0, # total time: write_queue.put (per patch)
        "n_patches":         0,   # patches enqueued for writing
        "q_size_last":       0,   # last observed write-queue depth
        "proc_queue_size":   0,   # last observed processing-queue depth
        "n_workers_active":  0,   # processing workers currently busy
        "n_workers_total":   _n_processing_workers,  # total processing-worker count
        # Degradation-template counters: {category: {template_name: count}}
        # Written every time a patch is enqueued so the GUI can show live
        # per-degradation-template statistics without post-processing.
        "degrade_counts": {},
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

        # Drain stderr helper.
        def drain_stderr(pipe: "subprocess.IO[bytes]") -> None:
            for line in pipe:
                stderr_lines.append(line.decode(errors="replace").rstrip())
            pipe.close()

        # ------------------------------------------------------------------
        # Inner helper: receive one decoded yuv420p frame, convert to BGR,
        # fill the rolling buffer, and enqueue any completed 7-frame windows
        # for background processing.  This is the lightweight FFmpeg pipe
        # reader — it does NOT call create_patch_pair directly.
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
            if pending_idx < len(pending_centers):
                min_keep = max(0, pending_centers[pending_idx] - half)
                for old_idx in [k for k in buffer if k < min_keep]:
                    del buffer[old_idx]
            _t_phases["t_buf_s"] += time.monotonic() - _ta

            # Detect completed 7-frame windows and enqueue for async processing.
            # The main reader is NOT blocked by heavy image work — processing
            # happens entirely in the background worker threads.
            while pending_idx < len(pending_centers):
                center = pending_centers[pending_idx]
                if actual_frame < center + half:
                    break  # need more frames

                window: List[np.ndarray] = []
                for fi in range(center - half, center + half + 1):
                    frm = buffer.get(max(0, fi))
                    if frm is None:
                        break
                    window.append(frm)

                if len(window) == n_frames:
                    # Hand off the window (list of frame references) to a
                    # background worker.  Frames are not copied — the worker
                    # holds references that keep the arrays alive until it
                    # finishes.  The buffer eviction above only removes dict
                    # entries, not the underlying numpy arrays.
                    _process_queue.put((center, window, center_map[center]))
                    frames_examined += 1
                    _t_phases["n_centers"] += 1
                    _t_phases["proc_queue_size"] = _process_queue.qsize()

                    # Periodic timing debug log (every 50 centres enqueued).
                    if _t_phases["n_centers"] >= _next_timing_log[0]:
                        _write_timing_log_entry()
                        _next_timing_log[0] += 50

                    if progress_fn is not None:
                        # 3rd arg = selected_idx: raw frames read so far
                        # (used for "piped fps" metric in the GUI).
                        # 4th arg = snapshot of phase timings.
                        progress_fn(frames_examined, dict(patches_created),
                                    selected_idx, dict(_t_phases))

                pending_idx += 1

        # ------------------------------------------------------------------
        # --- CONTINUOUS STREAM MODE (single FFmpeg pass) ------------------
        # FFmpeg behaves as a continuous producer: it decodes, tone-maps
        # (libplacebo/Vulkan), and resizes frames linearly, writing yuv420p
        # frames into the pipe.  Python reads the stream, tracks the frame
        # counter, and enqueues n_frames windows when extraction points are
        # reached.  All heavy image work runs in the background worker pool.
        #
        # When start_ts > 0 a fast-seek (-ss before -i) is injected so that
        # FFmpeg begins decoding from that timestamp.  This avoids re-decoding
        # the portion of the video that was already processed in a previous
        # run, preserving all existing patches on disk.
        # ------------------------------------------------------------------
        _seek_args = ["-ss", f"{start_ts:.3f}"] if start_ts > 0.0 else []
        cmd = [
            "ffmpeg",
            "-threads", "0",
            "-filter_threads", "0",
            "-loglevel", "warning",
            *hw_args,
            *_seek_args,          # fast seek before input (no-op when start_ts==0)
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

        # Read frames continuously until the last needed frame has been
        # consumed.  `actual_frame` is the 0-based video frame index;
        # `selected_idx` (incremented inside _consume_raw_frame) is the
        # count of frames read so far — both track the same position.
        _actual_frame: int = 0
        while _actual_frame <= last_needed:
            if pending_idx >= len(pending_centers):
                break  # all extraction-point windows enqueued; stop reading

            raw = process.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                _log("⚠️  Video stream ended before all frames were received")
                break

            # Start the clock on the very first frame so FFmpeg startup time
            # (device init, demux, codec open) is excluded from the FPS figure.
            if _t_start is None:
                _t_start = time.monotonic()

            _consume_raw_frame(raw, _actual_frame)
            _actual_frame += 1

            # Periodic throughput log.
            # fps  = raw frames piped per second.
            # SPS  = scene windows enqueued per second.
            if _t_start is not None and _actual_frame % _log_interval == 0:
                _elapsed = time.monotonic() - _t_start
                if _elapsed > 0:
                    _fps = _actual_frame / _elapsed
                    _sps_actual = frames_examined / _elapsed
                    _pq_size = _process_queue.qsize()
                    _t_phases["proc_queue_size"] = _pq_size
                    _log(
                        f"  📊 raw {_actual_frame:>5}/{last_needed + 1}  "
                        f"fps {_fps:>6.1f}  SPS {_sps_actual:>6.2f}  "
                        f"(scenes: {frames_examined}  pq:{_pq_size}"
                        f"  workers:{_active_workers_ctr[0]}/{_n_processing_workers})"
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

        # Drain the async processing queue — wait for all enqueued windows to
        # be fully processed before draining the write queue.
        for _ in _processing_threads:
            _process_queue.put(None)  # poison pill per worker
        for _pt in _processing_threads:
            _pt.join()

        # Drain the async write queue — wait for all pending PNG/BMP writes to
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

    # Safety net: if libplacebo still failed at runtime despite passing the
    # startup probe (should not happen with a real-file probe), disable it for
    # all subsequent videos in this process so the failure is not repeated.
    global _libplacebo_avail
    _stderr_text = "\n".join(stderr_lines)
    if _placebo and any(kw in _stderr_text for kw in _VULKAN_FAIL_STRINGS):
        _libplacebo_avail = False
        _log(
            "⚠️  libplacebo Vulkan failure detected — "
            "disabling libplacebo for the remainder of this run"
        )

    # GPU pipeline produced zero frames — retry with CPU-only (zscale) pipeline.
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
            color_trc=color_trc,
            vulkan_device=vulkan_device,
            output_format=output_format,
            ring_buffer_bytes_limit=ring_buffer_bytes_limit,
        )

    total = sum(patches_created.values())
    _elapsed_total = (
        (time.monotonic() - _t_start) if _t_start is not None else 0.0
    )
    _n_black = _n_black_ctr[0]
    _n_quality_fail = _n_quality_fail_ctr[0]
    _rejection_info = (
        f", {_n_black} black"
        + (f", {_n_quality_fail} quality-rejected" if _n_quality_fail else "")
        if (_n_black or _n_quality_fail)
        else ""
    )
    if _elapsed_total > 0:
        _sps_final = frames_examined / _elapsed_total
        _fps_final = selected_idx / _elapsed_total
        _log(
            f"✓ Continuous-stream extraction done: {total} patches saved, "
            f"{frames_examined} windows enqueued{_rejection_info}, "
            f"{selected_idx}/{last_needed + 1} raw frames read — "
            f"fps {_fps_final:.1f}  SPS {_sps_final:.2f}"
        )
    else:
        _log(
            f"✓ Continuous-stream extraction done: {total} patches saved, "
            f"{frames_examined} windows enqueued{_rejection_info}"
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
    color_trc: str = "smpte2084",
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
        color_trc=color_trc,
    )
