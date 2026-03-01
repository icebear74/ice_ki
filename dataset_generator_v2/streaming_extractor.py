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

# HDR→SDR tonemap filter chain: zscale → linear light → bt709 → tonemap → output
_TONEMAP_FILTER: str = (
    "zscale=t=linear:npl=100,"
    "format=gbrpf32le,"
    "zscale=p=bt709,"
    "tonemap=tonemap=mobius:desat=0,"
    "zscale=t=bt709:m=bt709:range=limited,"
    f"scale={STREAM_WIDTH}:{STREAM_HEIGHT}:flags=lanczos,"
    "format=bgr24"
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
        timestamp:   Centre-frame timestamp in seconds (used in filename).
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


def extract_and_save_streaming_distributed(
    video_path: str,
    assignments: List[Tuple[int, str, str]],
    n_frames: int,
    format_config: Dict[str, Dict],
    base_dir: str,
    fps: float,
    logger=None,
    is_interesting_fn: Optional[Callable[[np.ndarray], bool]] = None,
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
        video_path:       Path to input video.
        assignments:      Output of :func:`build_frame_assignments_distributed`.
        n_frames:         Frames per patch window (default 7).
        format_config:    ``{category: {format_name: {'gt_size': …, 'lr_size': …}}}``.
        base_dir:         Root dataset output directory.
        fps:              Video frame rate.
        logger:           Optional logger instance.
        is_interesting_fn: Optional callable ``(patch: np.ndarray) -> bool``
                          for quality gating.  When provided, random crops are
                          re-tried up to 5 times before falling back to a
                          centre crop.

    Returns:
        ``{category: patches_saved_count}``
    """

    def _log(msg: str) -> None:
        if logger:
            logger.info(msg)

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

    _log(
        f"🎬 Streaming extractor: {len(sorted_asgn)} assignments, "
        f"last frame needed: {last_needed}"
    )

    # FFmpeg: read linearly, apply HDR→SDR tonemap, pipe rawvideo BGR24
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", _TONEMAP_FILTER,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]

    frame_bytes: int = STREAM_WIDTH * STREAM_HEIGHT * 3
    patches_created: Dict[str, int] = {}

    # Rolling buffer: frame_idx → BGR frame (numpy array)
    buffer: Dict[int, np.ndarray] = {}
    pending_idx: int = 0  # index into pending_centers

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

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

    total = sum(patches_created.values())
    _log(
        f"✓ Streaming extraction done: {total} patches "
        f"from {len(sorted_asgn)} assignments"
    )
    return patches_created
