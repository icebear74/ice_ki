#!/usr/bin/env python3
"""
CUDA pipeline smoke-test
========================
Tests every meaningful FFmpeg CUDA pipeline combination against the first
video listed in generator_config.json.  Useful for verifying that the local
FFmpeg build and NVIDIA driver actually support each tier before running a
full extraction job.

Run from the repository root::

    python3 dataset_generator_v2/cuda_pipeline_smoketest.py

Or pass an explicit video path::

    python3 dataset_generator_v2/cuda_pipeline_smoketest.py /path/to/video.mkv

Exit code
---------
0  – all pipelines that are available on this system passed.
1  – at least one available pipeline failed (produced zero frames).

Pipelines tested
----------------
0. CPU-only                  – pure software decode + tonemap + scale (baseline)
1. CUDA decode               – GPU decode, CPU zscale/tonemap/scale
2. CUDA decode + scale p010  – GPU decode + scale_cuda, hwdownload p010, single-step
                               zscale with explicit HDR params (proven ~12 fps)
3. CUDA full-GPU             – GPU decode + tonemap_cuda + scale_cuda + hwdownload
4. CUDA decode + tonemap     – GPU decode + tonemap_cuda (no GPU scale)
5. CUDA bare download        – GPU decode + hwdownload (raw NV12, no tonemap/scale)
6. CUDA decode + scale_cuda + hwdownload (NV12 1080p, no tonemap)

Note on HDR vs SDR input
------------------------
Pipelines 0-4 use the full HDR->SDR zscale/tonemap chain which requires HDR
input with proper color-space metadata (PQ/BT.2020 or equivalent).  When an
SDR source is detected, those pipelines automatically switch to a plain
``scale=1920:1080`` filter so the CUDA transfer/decode mechanisms can still
be validated without the HDR filter chain.  A warning is printed in that case.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Locate config + streaming_extractor
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _REPO_ROOT / "generator_config.json"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from streaming_extractor import (
    STREAM_WIDTH,
    STREAM_HEIGHT,
    _TONEMAP_FILTER,
    _TONEMAP_FILTER_SCALE_CUDA,
    _TONEMAP_FILTER_CUDA,
    cuda_available,
    scale_cuda_available,
    tonemap_cuda_available,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Probe only the first N seconds of the video so the test is fast.
_PROBE_SECONDS: int = 3

# Raw frame size for 1920×1080 BGR24 (used to count frames from BGR24 pipelines).
_FRAME_BYTES_BGR24: int = STREAM_WIDTH * STREAM_HEIGHT * 3

# Explicit CUDA device initialisation flag.
# Passed before -hwaccel so FFmpeg always initialises the CUDA context up
# front.  Without this, some builds silently fall back to software decoding
# when the GPU context fails to auto-init, causing GPU filter chains to crash.
_CUDA_HW_INIT_ARGS: List[str] = ["-init_hw_device", "cuda=hw"]

# SDR-safe filter chains (no zscale/tonemap).  Used when the source is not HDR
# so the CUDA decode/transfer/scale mechanics can still be validated.
_SDR_CPU: str = (
    f"scale={STREAM_WIDTH}:{STREAM_HEIGHT}:flags=lanczos,format=bgr24"
)
# SDR fallback for the p010 pipeline: scale on GPU, bare download,
# then CPU format conversion to bgr24.
_SDR_SCALE_CUDA: str = (
    f"scale_cuda={STREAM_WIDTH}:{STREAM_HEIGHT},"
    "hwdownload,"
    "format=bgr24"
)
# tonemap_cuda requires HDR input; SDR fallback uses GPU scale + download only.
_SDR_TONEMAP_CUDA: str = _SDR_SCALE_CUDA
# tonemap_cuda only on GPU, CPU scale — SDR fallback: bare download + CPU scale.
_SDR_TONEMAP_ONLY_CUDA: str = (
    "hwdownload,"
    "scale=iw:ih,"
    f"scale={STREAM_WIDTH}:{STREAM_HEIGHT}:flags=lanczos,"
    "format=bgr24"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_video_from_config() -> Optional[str]:
    """Return the path of the first video in generator_config.json, or None."""
    if not _CONFIG_PATH.exists():
        return None
    try:
        with open(_CONFIG_PATH) as fh:
            cfg = json.load(fh)
        videos = cfg.get("videos", [])
        if videos:
            return videos[0].get("path")
    except Exception:
        pass
    return None


def _probe_video(video_path: str) -> Tuple[int, int, float]:
    """Return (width, height, fps) via ffprobe, or (3840, 2160, 24.0) on error."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate",
                "-of", "csv=p=0",
                video_path,
            ],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode().strip().splitlines()[0]
        parts = out.split(",")
        w, h = int(parts[0]), int(parts[1])
        num, den = parts[2].split("/")
        fps = float(num) / float(den)
        return w, h, fps
    except Exception:
        return 3840, 2160, 24.0


def _is_hdr(video_path: str) -> bool:
    """Return True when the first video stream uses PQ (smpte2084) or HLG transfer.

    BT.2020 primaries alone do NOT indicate HDR — SDR content can use BT.2020
    color primaries.  Only the transfer function (smpte2084 / arib-std-b67 / hlg)
    is a reliable HDR indicator.
    """
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=color_transfer",
                "-of", "default=noprint_wrappers=1",
                video_path,
            ],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode().lower()
        return any(kw in out for kw in ("smpte2084", "arib-std-b67", "hlg"))
    except Exception:
        return False


def _run_pipeline(
    label: str,
    video_path: str,
    hw_args: List[str],
    vf_filter: Optional[str],
    pix_fmt: str,
    frame_bytes: int,
    extra_in_args: Optional[List[str]] = None,
) -> Tuple[bool, int, float, str]:
    """
    Run FFmpeg for _PROBE_SECONDS and count how many complete frames arrive.

    Returns (success, frame_count, fps_achieved, error_hint).
    success = True when at least 1 frame was decoded.
    """
    cmd = ["ffmpeg", "-hide_banner"]
    cmd += hw_args
    # Limit input duration so the test is fast
    if extra_in_args:
        cmd += extra_in_args
    cmd += ["-t", str(_PROBE_SECONDS), "-i", video_path]
    if vf_filter:
        cmd += ["-vf", vf_filter]
    cmd += ["-f", "rawvideo", "-pix_fmt", pix_fmt, "pipe:1"]

    stderr_lines: List[str] = []
    frames = 0
    t0 = time.monotonic()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Drain stderr in a thread so we never block the writer.
        import threading

        def _drain(pipe):
            for raw in pipe:
                stderr_lines.append(raw.decode(errors="replace").rstrip())
            pipe.close()

        t = threading.Thread(target=_drain, args=(proc.stderr,), daemon=True)
        t.start()

        while True:
            chunk = proc.stdout.read(frame_bytes)
            if len(chunk) < frame_bytes:
                break
            frames += 1

        proc.stdout.close()
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        t.join(timeout=2)
    except FileNotFoundError:
        return False, 0, 0.0, "ffmpeg not found"
    except Exception as exc:
        return False, 0, 0.0, str(exc)

    elapsed = max(time.monotonic() - t0, 0.001)
    fps_achieved = frames / elapsed

    # Collect a short error hint from stderr when no frames arrived.
    error_hint = ""
    if frames == 0:
        # Grab the most informative stderr lines.
        relevant = [
            ln for ln in stderr_lines
            if any(kw in ln for kw in (
                "Error", "error", "Invalid", "failed", "Failed",
                "Cannot", "cannot", "No such", "hwdownload",
            ))
        ]
        error_hint = " | ".join(relevant[-3:]) if relevant else "(no stderr)"

    return frames > 0, frames, fps_achieved, error_hint


# ---------------------------------------------------------------------------
# Pipeline definitions
# ---------------------------------------------------------------------------

# HDR-only tonemap_cuda pipeline: GPU tonemap, CPU scale.
# tonemap_cuda outputs NV12 CUDA frames; hwdownload + scale=iw:ih brings them
# to CPU as NV12->YUV420P, then a CPU scale step resizes to 1920x1080.
_TONEMAP_ONLY_CUDA_FILTER: str = (
    "tonemap_cuda=tonemap=mobius:desat=0:peak=100,"
    "hwdownload,"
    "scale=iw:ih,"
    "format=yuv420p,"
    f"scale={STREAM_WIDTH}:{STREAM_HEIGHT}:flags=lanczos,"
    "format=bgr24"
)

# NV12 1080p GPU scale + download (no tonemap, works for SDR and HDR alike).
_SCALE_CUDA_NV12_FILTER: str = (
    f"scale_cuda={STREAM_WIDTH}:{STREAM_HEIGHT}:interp_algo=bicubic:format=nv12,"
    "hwdownload,"
    "scale=iw:ih,"
    "format=nv12"
)


def _build_pipelines(src_w: int, src_h: int, is_hdr: bool) -> List[dict]:
    """
    Return a list of pipeline descriptors.  Each dict has:
        label       – human-readable name
        requires    – callable() → bool; skipped when False
        hw_args     – FFmpeg input-side hw-accel arguments
        vf_filter   – -vf value, or None
        pix_fmt     – expected output pixel format
        frame_bytes – bytes per output frame

    When *is_hdr* is False the full HDR->SDR tonemap chains (zscale/tonemap)
    are replaced with module-level SDR-safe alternatives so that the CUDA
    decode/transfer/scale mechanics can still be validated.
    """
    cuda = cuda_available()
    scale_cuda = scale_cuda_available()
    tonemap_cuda = tonemap_cuda_available()

    def _always():
        return True

    def _need_cuda():
        return cuda

    def _need_scale_cuda():
        return cuda and scale_cuda

    def _need_tonemap_cuda():
        return cuda and tonemap_cuda

    # Bare-download frame size depends on the actual source resolution.
    bare_frame_bytes = src_w * src_h * 3 // 2  # NV12

    cpu_filter        = _TONEMAP_FILTER            if is_hdr else _SDR_CPU
    scale_cuda_filt   = _TONEMAP_FILTER_SCALE_CUDA if is_hdr else _SDR_SCALE_CUDA
    tonemap_cuda_filt = _TONEMAP_FILTER_CUDA       if is_hdr else _SDR_TONEMAP_CUDA
    tonemap_only_filt = _TONEMAP_ONLY_CUDA_FILTER  if is_hdr else _SDR_TONEMAP_ONLY_CUDA

    hdr_note = "" if is_hdr else "  ⚠ SDR source: HDR tonemap replaced by plain scale"

    # Common hw_args variants with explicit CUDA device init.
    _hw_decode_only  = [*_CUDA_HW_INIT_ARGS, "-hwaccel", "cuda"]
    _hw_decode_scale = [*_CUDA_HW_INIT_ARGS, "-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]

    return [
        # ── 0. CPU-only baseline ─────────────────────────────────────────
        {
            "label": f"0  CPU-only  (software decode + {'zscale/tonemap/scale' if is_hdr else 'scale'}){hdr_note}",
            "requires": _always,
            "hw_args": [],
            "vf_filter": cpu_filter,
            "pix_fmt": "bgr24",
            "frame_bytes": _FRAME_BYTES_BGR24,
        },
        # ── 1. CUDA decode + CPU tonemap ─────────────────────────────────
        {
            "label": f"1  CUDA decode  (GPU decode, CPU {'zscale/tonemap/scale' if is_hdr else 'scale'}){hdr_note}",
            "requires": _need_cuda,
            "hw_args": _hw_decode_only,
            "vf_filter": cpu_filter,
            "pix_fmt": "bgr24",
            "frame_bytes": _FRAME_BYTES_BGR24,
        },
        # ── 2. CUDA decode + scale_cuda + p010 + single-step zscale ──────
        # The PROVEN production pipeline (~12 fps on a mid-range GPU).
        # scale_cuda scales 4K→1080p on the GPU keeping the 10-bit surface;
        # hwdownload transfers as p010; one zscale call with explicit HDR
        # colour-space params converts to BT.709 SDR in a single step.
        # This is the chain used by extract_and_save_streaming_distributed.
        {
            "label": f"2  CUDA decode + scale_cuda + p010 + zscale  (⭐ production pipeline){hdr_note}",
            "requires": _need_scale_cuda,
            "hw_args": _hw_decode_scale,
            "vf_filter": scale_cuda_filt,
            "pix_fmt": "bgr24",
            "frame_bytes": _FRAME_BYTES_BGR24,
        },
        # ── 3. CUDA full-GPU (tonemap_cuda + scale_cuda) ─────────────────
        {
            "label": f"3  CUDA full-GPU  (tonemap_cuda + scale_cuda + hwdownload){hdr_note}",
            "requires": _need_tonemap_cuda,
            "hw_args": _hw_decode_scale,
            "vf_filter": tonemap_cuda_filt,
            "pix_fmt": "bgr24",
            "frame_bytes": _FRAME_BYTES_BGR24,
        },
        # ── 4. CUDA decode + tonemap_cuda (no GPU scale) ─────────────────
        # tonemap_cuda outputs NV12 CUDA frames; hwdownload + scale=iw:ih
        # brings them to CPU as NV12->YUV420P, then CPU scale to 1080p.
        {
            "label": f"4  CUDA decode + tonemap_cuda  (GPU tonemap, CPU scale){hdr_note}",
            "requires": _need_tonemap_cuda,
            "hw_args": _hw_decode_scale,
            "vf_filter": tonemap_only_filt,
            "pix_fmt": "bgr24",
            "frame_bytes": _FRAME_BYTES_BGR24,
        },
        # ── 5. CUDA bare download (GPU decode -> hwdownload, no tonemap) ──
        # Validates that hwdownload itself works on this driver/FFmpeg combo.
        # Output is raw NV12 at the source resolution (HDR or SDR).
        {
            "label": "5  CUDA bare download  (GPU decode + hwdownload, NV12 raw)",
            "requires": _need_cuda,
            "hw_args": _hw_decode_scale,
            "vf_filter": "hwdownload,scale=iw:ih,format=nv12",
            "pix_fmt": "nv12",
            "frame_bytes": bare_frame_bytes,
        },
        # ── 6. CUDA decode + scale_cuda + hwdownload (no tonemap, 1080p) ─
        # Checks the full GPU->CPU transfer path without any colour-science.
        # Works for both SDR and HDR sources.
        {
            "label": "6  CUDA decode + scale_cuda + hwdownload  (NV12 1080p, no tonemap)",
            "requires": _need_scale_cuda,
            "hw_args": _hw_decode_scale,
            "vf_filter": _SCALE_CUDA_NV12_FILTER,
            "pix_fmt": "nv12",
            "frame_bytes": STREAM_WIDTH * STREAM_HEIGHT * 3 // 2,
        },
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: List[str]) -> int:
    # ── Determine video path ────────────────────────────────────────────────
    if len(argv) >= 2:
        video_path = argv[1]
    else:
        video_path = _first_video_from_config()

    if not video_path:
        print("ERROR: No video path provided and generator_config.json has no videos.", file=sys.stderr)
        return 1

    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}", file=sys.stderr)
        print("  Provide an existing path as the first argument or update generator_config.json.",
              file=sys.stderr)
        return 1

    # ── Probe source resolution ────────────────────────────────────────────
    src_w, src_h, src_fps = _probe_video(video_path)

    # ── Print environment summary ──────────────────────────────────────────
    sep = "─" * 72
    print(sep)
    print("🔬  CUDA Pipeline Smoke-Test")
    print(sep)
    print(f"  Video            : {video_path}")
    print(f"  Source resolution: {src_w}×{src_h}  @  {src_fps:.3f} fps")
    print(f"  Probe duration   : {_PROBE_SECONDS} s  →  ~{int(src_fps * _PROBE_SECONDS)} frames expected")
    print(f"  Output resolution: {STREAM_WIDTH}×{STREAM_HEIGHT} (BGR24 pipelines)")
    print()

    # ── HDR detection ─────────────────────────────────────────────────────
    is_hdr = _is_hdr(video_path)
    print(f"  HDR source       : {is_hdr}")
    if not is_hdr:
        print("  ⚠  SDR source detected — HDR tonemap chains replaced by plain scale.")
        print("     Run with a real HDR (PQ/BT.2020) file for a full production test.")
    print()
    print("  FFmpeg CUDA support detected at startup:")
    print(f"    cuda_available()         = {cuda_available()}")
    print(f"    scale_cuda_available()   = {scale_cuda_available()}")
    print(f"    tonemap_cuda_available() = {tonemap_cuda_available()}")
    print()

    # ── Run pipelines ──────────────────────────────────────────────────────
    pipelines = _build_pipelines(src_w, src_h, is_hdr)
    results = []

    for p in pipelines:
        if not p["requires"]():
            status = "SKIP"
            print(f"  [{status}]  {p['label']}")
            results.append((p["label"], status, 0, 0.0, ""))
            continue

        print(f"  [....] {p['label']}  ", end="", flush=True)
        ok, n_frames, fps_out, hint = _run_pipeline(
            label=p["label"],
            video_path=video_path,
            hw_args=p["hw_args"],
            vf_filter=p["vf_filter"],
            pix_fmt=p["pix_fmt"],
            frame_bytes=p["frame_bytes"],
        )

        if ok:
            status = "PASS"
            print(f"\r  [{status}]  {p['label']}")
            print(f"           → {n_frames} frames  ({fps_out:.1f} fps)")
        else:
            status = "FAIL"
            print(f"\r  [{status}]  {p['label']}")
            print(f"           → 0 frames  |  {hint}")

        results.append((p["label"], status, n_frames, fps_out, hint))

    # ── Summary ─────────────────────────────────────────────────────────────
    print()
    print(sep)
    print("📊  Summary")
    print(sep)
    n_pass = sum(1 for r in results if r[1] == "PASS")
    n_fail = sum(1 for r in results if r[1] == "FAIL")
    n_skip = sum(1 for r in results if r[1] == "SKIP")
    print(f"  PASS: {n_pass}   FAIL: {n_fail}   SKIP: {n_skip}")
    print()

    if n_fail:
        print("  Failed pipelines:")
        for label, status, _, _, hint in results:
            if status == "FAIL":
                print(f"    ✗  {label}")
                if hint:
                    print(f"       {hint}")
        print()

    # ── Recommendation ───────────────────────────────────────────────────────
    print("  Recommended pipeline for extract_and_save_streaming_distributed:")
    if any(r[1] == "PASS" and r[0].startswith("3") for r in results):
        print("    ✅  full-GPU (tonemap_cuda + scale_cuda)  → use_cuda=True")
    elif any(r[1] == "PASS" and r[0].startswith("2") for r in results):
        print("    ✅  ⭐ scale_cuda + p010 + single-step zscale (~12 fps)  → use_cuda=True")
    elif any(r[1] == "PASS" and r[0].startswith("1") for r in results):
        print("    ✅  CUDA decode + CPU tonemap             → use_cuda=True")
    else:
        print("    ✅  CPU-only                              → use_cuda=False")
    if not is_hdr:
        print()
        print("  ℹ  Re-run with an HDR source for a definitive result.")
    print(sep)

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
