#!/usr/bin/env python3
"""
Dataset Generator V2 – UHD Quality

Loads configuration exclusively from:
  - dataset_generator_v2/templates.json
  - dataset_generator_v2/generator_config.json

via the shared config utility (utils/config_io.py) introduced in Task 1.

No hard-coded format names, category names, output paths, or distribution
assumptions.  All functional decisions are driven entirely by the active config
and the templates file.

NOTE – config file naming convention
=====================================
generator_config.json       → the ONLY file used at runtime by all tools.
                              It is listed in .gitignore (machine-local, not committed).
generator_config_active.json → a read-only snapshot given to AI agents for review.
                              It is NEVER loaded by any code; only humans/agents read it.
"""

import os
import sys
import json
import cv2
import numpy as np
import subprocess
import random
import tempfile
import shutil
import logging
import signal
import threading
import queue
import time
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add utils to path
sys.path.insert(0, os.path.dirname(__file__))

from utils.config_io import (
    load_templates,
    load_active_config,
    save_active_config,
    ensure_templates_file,
    validate_templates,
    validate_active_config,
)
from utils.format_definitions import get_output_dirs_for_format
from streaming_extractor import (
    build_assignments_per_category,
    extract_and_save_streaming_distributed,
    STREAM_4K_WIDTH,
    STREAM_4K_HEIGHT,
    STREAM_OPT_WIDTH,
    STREAM_OPT_HEIGHT,
    create_patch_pair,
    is_black_frame as _streaming_is_black_frame,
    is_hdr_transfer,
    build_vf_filter,
    cuda_available,
    scale_cuda_available,
    tonemap_cuda_available,
    libplacebo_available,
    _discover_vulkan_devices,
    map_cuda_to_vulkan_device,
    _get_ffmpeg_major_version,
    OutputFormat,
)
from utils.progress_tracker import ProgressTracker
from generation_plan import GenerationPlan
from utils.dataset_display import draw_dataset_ui
from utils.terminal_ui import hide_cursor, show_cursor, clear_screen
from category_utils import get_video_categories, normalize_categories

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: 'rich' library not found. Install with: pip install rich")

console = Console() if RICH_AVAILABLE else None
logger = logging.getLogger(__name__)

# Default config file names (relative to the script directory)
_TEMPLATES_FILENAME = "templates.json"
_ACTIVE_CONFIG_FILENAME = "generator_config.json"


def _detect_nvidia_gpus() -> List[Tuple[int, str]]:
    """Return ``[(device_index, gpu_name), …]`` for all NVIDIA GPUs.

    Queries ``nvidia-smi``.  Returns an empty list when ``nvidia-smi`` is not
    installed, the driver is unavailable, or no NVIDIA GPU is present.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode(errors="replace").strip()
        gpus: List[Tuple[int, str]] = []
        for line in out.splitlines():
            parts = line.split(",", 1)
            if len(parts) == 2:
                try:
                    gpus.append((int(parts[0].strip()), parts[1].strip()))
                except ValueError:
                    pass
        return gpus
    except Exception:
        return []


class DatasetGeneratorV2UHD:
    """
    Dataset Generator V2 – dynamic, template-driven, no hard-coded formats.

    Configuration is loaded exclusively from:
      * ``templates.json``         – format and degradation templates
      * ``generator_config.json``  – categories, videos, settings

    Both files are validated at startup via ``utils/config_io.py``.  The
    generator fails early with a clear error message when a required field is
    missing or a template reference cannot be resolved.
    """

    MAX_DISPLAYED_PRIORITIES = 10

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialise the generator.

        Args:
            config_dir: Directory that contains ``templates.json`` and
                        ``generator_config.json``.  Defaults to the directory
                        that contains this script.
        """
        if config_dir is None:
            config_dir = os.path.dirname(os.path.abspath(__file__))

        templates_path = os.path.join(config_dir, _TEMPLATES_FILENAME)
        active_config_path = os.path.join(config_dir, _ACTIVE_CONFIG_FILENAME)

        # ── Load and validate configs ─────────────────────────────────────────
        self.templates = ensure_templates_file(templates_path)

        if not os.path.exists(active_config_path):
            print(
                f"❌ Active config not found: {active_config_path}\n"
                "   Please create it via video_manager.py or copy the default."
            )
            sys.exit(1)

        self.config = load_active_config(active_config_path)

        tmpl_errors = validate_templates(self.templates)
        cfg_errors = validate_active_config(self.config, self.templates)
        if tmpl_errors or cfg_errors:
            print("❌ Config validation failed:")
            for e in tmpl_errors:
                print(f"  [templates] {e}")
            for e in cfg_errors:
                print(f"  [active config] {e}")
            sys.exit(1)

        # ── Extract the fields that the rest of the code relies on ────────────
        self.categories: Dict[str, dict] = self.config["categories"]
        self.videos: List[dict] = self.config.get("videos", [])
        self.category_targets: Dict[str, int] = {
            name: cat["target_total"] for name, cat in self.categories.items()
        }

        # Build the format_config dict expected by the streaming extractor.
        # Resolves template references and attaches source_mode + degradation.
        self.format_config: Dict[str, Dict[str, dict]] = self._build_format_config()

        # ── Output paths ──────────────────────────────────────────────────────
        self.base_dir: str = self.config["root_path"]
        self.temp_dir: str = os.path.join(self.base_dir, "tmp")
        self.status_file: str = os.path.join(self.base_dir, "generation_status.json")

        # Terminal UI setting (must be set before logger setup)
        self.use_terminal_ui = True

        # ── Logger ────────────────────────────────────────────────────────────
        self.logger = self._setup_logger()
        sys.logger = self.logger

        # ── GPU / Vulkan device discovery ────────────────────────────────────
        self.use_cuda: bool = cuda_available()

        # Discover CUDA GPUs via nvidia-smi (for display names).
        _detected_gpus = _detect_nvidia_gpus()
        _cuda_indices: List[int] = (
            [idx for idx, _ in _detected_gpus]
            if _detected_gpus
            else ([0] if self.use_cuda else [])
        )
        self._available_gpu_names: Dict[int, str] = {
            idx: name for idx, name in _detected_gpus
        }

        # Discover Vulkan devices from FFmpeg — these are the indices used by
        # the libplacebo pipeline.  CUDA indices from nvidia-smi are NOT
        # guaranteed to match Vulkan indices; we map them explicitly.
        _vulkan_devs = _discover_vulkan_devices()  # [(vk_idx, desc), …]
        if _vulkan_devs:
            self.logger.info(
                f"🖥️  Vulkan devices reported by FFmpeg: "
                + ", ".join(f"{i}: {d}" for i, d in _vulkan_devs)
            )
        else:
            self.logger.info("🖥️  No Vulkan devices found via FFmpeg (CPU/software fallback)")

        # Build the list of Vulkan device indices to use for round-robin
        # assignment.  Each CUDA GPU is mapped to its Vulkan counterpart.
        # When map_cuda_to_vulkan_device() returns None the entry is still
        # included as None so that FFmpeg picks any available Vulkan device
        # for that worker slot (better than skipping the GPU entirely).
        if _cuda_indices:
            _mapped: List[Optional[int]] = [
                map_cuda_to_vulkan_device(c) for c in _cuda_indices
            ]
            for c_idx, v_idx in zip(_cuda_indices, _mapped):
                if v_idx is not None:
                    self.logger.info(
                        f"  CUDA {c_idx} ({self._available_gpu_names.get(c_idx, '?')}) "
                        f"→ Vulkan {v_idx}"
                    )
                else:
                    self.logger.warning(
                        f"  CUDA {c_idx} ({self._available_gpu_names.get(c_idx, '?')}) "
                        f"→ Vulkan mapping not found; FFmpeg will choose device automatically"
                    )
            self._available_gpu_indices: List[int] = [
                v for v in _mapped if v is not None
            ]
            # Keep None-mapped slots in the full worker list so we still attempt
            # round-robin across all detected GPUs even if the Vulkan index is
            # unknown.  _run_multi_stream / _extract_film_parallel both handle
            # vulkan_device=None safely (FFmpeg picks any available device).
            self._vulkan_device_pool: List[Optional[int]] = _mapped
        elif _vulkan_devs:
            # No CUDA but Vulkan devices exist — use them directly.
            self._available_gpu_indices = [i for i, _ in _vulkan_devs]
            self._vulkan_device_pool = self._available_gpu_indices[:]
        else:
            self._available_gpu_indices = []
            self._vulkan_device_pool = [None]  # CPU/software Vulkan

        if self.use_cuda:
            self.logger.info("🚀 CUDA/GPU mode enabled (hardware-accelerated decoding & scaling)")
        else:
            self.logger.info("🖥️  CPU-only mode enabled (CUDA not available in this FFmpeg build)")

        # Placeholder — overwritten in the "Parallel worker configs" block
        # below, once self.workers is known.  Do not use before that point.
        self._parallel_worker_configs: Optional[List[dict]] = None

        self.logger.info(f"Loaded {len(self.videos)} videos from active config")
        self.logger.info(f"Categories: {list(self.category_targets.keys())}")
        for cat, total in self.category_targets.items():
            self.logger.info(f"  {cat}: target_total={total:,}")

        # ── Metadata cache ────────────────────────────────────────────────────
        self.metadata_cache_file = os.path.join(self.base_dir, ".video_metadata_cache.json")
        self.metadata_cache = self._load_metadata_cache()

        # ── Progress tracking ─────────────────────────────────────────────────
        self.tracker = ProgressTracker(self.status_file)
        self.tracker.update_progress(total_videos=len(self.videos))
        self.tracker.initialize_categories(self.category_targets)

        plan_file = os.path.join(self.base_dir, "extraction_plan.json")
        self.plan = GenerationPlan(plan_file)

        # ── Runtime state ─────────────────────────────────────────────────────
        proc = self.config.get("processing", {})
        self.workers: int = self.config.get("workers", 6)
        self.running = True
        self.paused = False
        self.last_update_time = time.time()
        self.update_interval = 0.5
        self.logger.info(f"⚡ Using {self.workers} threads for FFmpeg extraction")

        # ── Parallel worker configs ───────────────────────────────────────────
        # Worker count: max(config workers, number of GPU slots) — at least 1.
        # Each entry uses use_cuda=False; the libplacebo/Vulkan pipeline does
        # not need CUDA hardware decode.  Vulkan device assignment is handled
        # round-robin in _extract_film_parallel using _vulkan_device_pool.
        _n_workers = max(
            self.workers,
            len(self._vulkan_device_pool) if self._vulkan_device_pool else 1,
        )
        self._parallel_worker_configs = [
            {"use_cuda": False, "cuda_device": 0}
            for _ in range(_n_workers)
        ]
        self.logger.info(
            f"🔀 Parallel extraction: {_n_workers} workers, "
            f"{len(self._vulkan_device_pool)} Vulkan device slot(s) for round-robin assignment"
        )

        # ── UI heartbeat ──────────────────────────────────────────────────────
        # A background thread refreshes the terminal UI every second so the
        # display stays live even when the main thread is blocked (e.g. waiting
        # for parallel FFmpeg workers to finish).
        self._ui_lock = threading.Lock()          # prevents concurrent redraws
        self._ui_heartbeat_stop = threading.Event()
        self._ui_heartbeat_thread: Optional[threading.Thread] = None

        # Statistics
        self.start_time = time.time()
        self.extractions_count = 0
        self.success_count = 0
        self.current_video_name = ""

        # ── Terminal UI state ─────────────────────────────────────────────────
        # Build a display-label dict: template_name → "WxH" from gt_size.
        # Used in the patch-distribution table so every column header shows
        # the real pixel size (e.g. "1152×648") instead of the truncated
        # template name suffix (which causes duplicates like "169" / "169").
        _format_labels: Dict[str, str] = {}
        for _cat_fc in self.format_config.values():
            for _tmpl, _cfg in _cat_fc.items():
                if _tmpl not in _format_labels:
                    _gt = _cfg.get("gt_size", [0, 0])
                    _format_labels[_tmpl] = f"{_gt[0]}×{_gt[1]}"

        self.ui_state = {
            "current_video_name": "",
            "current_video_index": 0,
            "total_videos": len(self.videos),
            "current_video_progress": {},
            "overall_progress": {},
            "patch_distribution": {},
            "scenes_processed": 0,
            "patches_created_total": 0,
            "frames_processed_total": 0,
            "frames_read_total": 0,
            "avg_time_per_scene": 0.0,
            "eta": {},
            "live_fps": 0.0,
            "live_sps": 0.0,
            # Decode-backend label: reflects Vulkan device availability.
            # libplacebo is the primary HDR path; device assignment uses
            # round-robin across the mapped Vulkan device pool.
            "decode_backend": (
                f"libplacebo [Vulkan HDR→SDR] — "
                f"{len(self._vulkan_device_pool)} Vulkan device slot(s) round-robin"
                if self._vulkan_device_pool and self._vulkan_device_pool[0] is not None
                else "libplacebo [Vulkan HDR→SDR — CPU/software fallback]"
            ),
            "categories": list(self.category_targets.keys()),
            "format_sizes": list(next(iter(self.format_config.values()), {}).keys()),
            "format_labels": _format_labels,  # template_name → "WxH" string
            "timing_phases": {},              # phase timings from streaming extractor
            "parallel_status": "",            # e.g. "⚡ 16 workers active"
            "active_streams": [],             # per-stream state dicts for GUI panels
            "n_active_streams": 0,            # active stream count for quick reads
            "n_gpus_available": len(self._vulkan_device_pool),
            # Output format (BMP by default, configurable).
            # Updated at runtime in _run_multi_stream().
            "output_format": self.config.get("output_format", "bmp").upper(),
            # Degradation-template breakdown: {category: {tmpl_name: count}}
            # Aggregated from active_streams[*].degrade_counts by the UI layer.
            "degrade_counts_global": {},
        }
        self.ui_update_counter = 0

        if RICH_AVAILABLE:
            self._show_priority_distribution()

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ── Decode pipeline benchmark ─────────────────────────────────────────────

    def run_benchmark_tool(self, force: bool = False) -> None:
        """Run a short FFmpeg decode-throughput benchmark and select the fastest pipeline.

        This method is a standalone diagnostic tool.  It is **not** called
        automatically at startup; invoke it explicitly when you want to
        measure or re-measure decode throughput (e.g. after installing a new
        GPU driver or replacing hardware).

        Tests every meaningful decode pipeline variant (CPU-only, CUDA NVDEC per
        GPU, CUDA NVDEC + scale_cuda per GPU, full-GPU tonemap_cuda per GPU when
        HDR source and libnpp are available), then runs a comprehensive parallel
        matrix with N = ``self.workers`` concurrent jobs to find the optimal
        GPU/CPU split for production use.

        All FFmpeg child processes are started at idle nice priority (same value
        as the main extraction loop uses) to avoid disturbing other system
        activity and to keep measurements consistent.

        Results are printed to stdout in a detailed table so the operator can
        see exactly which pipeline won and by how much.

        The winning configuration is written to::

            <base_dir>/decode_benchmark.json

        and the instance attributes ``self.use_cuda`` and ``self.cuda_device``
        are updated so every subsequent call to
        ``extract_and_save_streaming_distributed`` uses the optimal backend.

        The file is re-used on the next run when it is younger than
        ``CACHE_MAX_AGE_DAYS`` days, unless *force* is ``True`` (triggered by
        the ``--benchmark`` CLI flag).

        Notes
        -----
        * Multi-category videos are fully handled in a single FFmpeg pass by the
          extractor, so benchmarking one-pass 4 K decode is representative even
          for videos that belong to several categories simultaneously.
        * Parallel variants measure Family A (K × GPU + (N-K) × CPU sweep) and
          Family B/C (dual-GPU split) so we can find the NVDEC saturation point
          and whether CPU fill-up after saturating the GPU still pays off.
        """
        CACHE_MAX_AGE_DAYS = 7
        WARMUP_FRAMES      = 20
        BENCH_FRAMES       = 80
        SEEK_SEC           = 60.0   # skip opening credits / black frames
        W, H               = STREAM_OPT_WIDTH, STREAM_OPT_HEIGHT
        FRAME_BYTES        = W * H * 3 // 2   # yuv420p = 1.5 bytes/pixel
        # Increment this version whenever the benchmark probe logic changes so
        # that stale cached results are automatically invalidated without
        # requiring the user to pass --benchmark manually.
        _CACHE_VERSION     = 4      # v4: libplacebo probe injects HDR metadata to force Vulkan init

        cache_path = os.path.join(self.base_dir, "decode_benchmark.json")

        # ── Load cache ────────────────────────────────────────────────────────
        if not force:
            try:
                if os.path.exists(cache_path):
                    cache = json.loads(Path(cache_path).read_text())
                    age_days = (time.time() - cache.get("_ts", 0)) / 86400.0
                    cache_ver = cache.get("_probe_version", 1)
                    if age_days < CACHE_MAX_AGE_DAYS and cache_ver >= _CACHE_VERSION:
                        best = cache.get("best", {})
                        self.cuda_device = best.get("cuda_device", 0)
                        self.use_cuda    = best.get("use_cuda", self.use_cuda)
                        _backend_lbl = best.get("label", "?")
                        _n_w         = cache.get("best_parallel", {})
                        # Load optimal per-worker decode configs for parallel
                        # within-film extraction (stored since this PR).
                        if _n_w and _n_w.get("worker_configs"):
                            self._parallel_worker_configs = _n_w["worker_configs"]
                        _n_w_str     = (
                            f"  |  {_n_w['n_workers']}× parallel: {_n_w['fps']:.0f} fps"
                            if _n_w and _n_w.get("fps") else ""
                        )
                        self.ui_state["decode_backend"] = (
                            f"{_backend_lbl}  [{best.get('fps', 0):.0f} fps single"
                            f"{_n_w_str}]"
                        )
                        print(
                            f"\n  ♻️  Decode benchmark cache ({age_days:.1f}d old) — "
                            f"skipping re-run.\n"
                            f"  Best pipeline : {_backend_lbl}\n"
                            f"  Throughput    : {best.get('fps', 0):.1f} fps  "
                            f"(use_cuda={self.use_cuda}, cuda_device={self.cuda_device})\n"
                            f"  Cache file    : {cache_path}\n"
                            f"  Tip           : run with --benchmark to force a fresh measurement.\n"
                        )
                        self.logger.info(
                            f"Decode backend from cache: {_backend_lbl} "
                            f"[cuda_device={self.cuda_device}, use_cuda={self.use_cuda}, "
                            f"fps={best.get('fps',0):.1f}]"
                        )
                        return
            except Exception:
                pass  # corrupt / unreadable cache → run fresh

        # ── Find a test video ─────────────────────────────────────────────────
        test_video: Optional[str] = None
        test_is_hdr: bool         = True
        for v in self.videos:
            p = v.get("path", "")
            if os.path.exists(p):
                meta = self._get_video_metadata(p)
                if meta:
                    test_video  = p
                    test_is_hdr = meta.get("is_hdr", True)
                    if test_is_hdr:
                        break   # prefer HDR for most demanding / realistic test

        if not test_video:
            print(
                "\n  ⚠️  Decode benchmark skipped: no accessible video found in the config.\n"
                "       Add at least one reachable video path to generator_config.json.\n"
            )
            self.logger.warning("Decode benchmark skipped — no accessible video found")
            return

        # ── Detect GPUs ───────────────────────────────────────────────────────
        available_gpus: List[Tuple[int, str]] = _detect_nvidia_gpus()

        # ── FFmpeg version ────────────────────────────────────────────────────
        ffmpeg_ver = _get_ffmpeg_major_version()

        # ── Print header ──────────────────────────────────────────────────────
        _W = 72
        _SEP  = "═" * _W
        _SEP2 = "─" * _W
        print(f"\n{_SEP}")
        print(f"  FFmpeg Decode Benchmark  –  {W}×{H} yuv420p (optimised)")
        print(_SEP2)
        print(f"  Test video    : {os.path.basename(test_video)}")
        print(f"  Content type  : {'HDR (PQ/HLG)' if test_is_hdr else 'SDR (BT.709)'}")
        if available_gpus:
            for idx, name in available_gpus:
                print(f"  GPU {idx}          : {name}")
        else:
            print(f"  GPUs          : none detected via nvidia-smi")
        print(f"  Benchmark     : {BENCH_FRAMES} frames timed  +  {WARMUP_FRAMES} warmup frames")
        print(f"  Seek offset   : {SEEK_SEC:.0f} s  (skip credits / black frames)")
        print(f"  Output dir    : {self.base_dir}")
        print(_SEP)

        # ── Build filter chains ───────────────────────────────────────────────
        # Filter chains are constructed explicitly for each tier so the benchmark
        # can test each independently, regardless of which tier build_vf_filter()
        # would auto-select.

        def _cpu_filter(hdr: bool) -> str:
            if hdr:
                if libplacebo_available():
                    return (
                        f"libplacebo=w={W}:h={H}"
                        ":colorspace=bt709:color_primaries=bt709:color_trc=bt709"
                        ":range=tv,"
                        "format=yuv420p"
                    )
                return (
                    "zscale=t=linear:npl=100:filter=bilinear,"
                    "format=gbrpf32le,"
                    "zscale=p=bt709:filter=bilinear,"
                    "tonemap=tonemap=reinhard:desat=0,"
                    "zscale=t=bt709:m=bt709:range=full:filter=bilinear,"
                    f"scale={W}:{H}:flags=bilinear,"
                    "format=yuv420p"
                )
            return f"scale={W}:{H}:flags=bilinear,format=yuv420p"

        def _scale_gpu_filter(hdr: bool) -> str:
            if hdr:
                # Use explicit tin= and primariesin= to guarantee correct HDR→SDR
                # conversion regardless of CUDA frame metadata propagation.
                return (
                    f"scale_cuda={W}:{H}:interp_algo=bicubic,"
                    "hwdownload,"
                    "format=p010,"
                    "zscale=tin=smpte2084:primariesin=bt2020:t=linear:npl=100:filter=bilinear,"
                    "format=gbrpf32le,"
                    "zscale=p=bt709:filter=bilinear,"
                    "tonemap=tonemap=reinhard:desat=0,"
                    "zscale=t=bt709:m=bt709:range=full:filter=bilinear,"
                    "format=yuv420p"
                )
            return (
                f"scale_cuda={W}:{H}:interp_algo=bicubic,"
                "hwdownload,"
                "format=yuv420p"
            )

        def _full_gpu_filter() -> str:
            return (
                "tonemap_cuda=tonemap=mobius:desat=0:peak=100,"
                f"scale_cuda={W}:{H}:interp_algo=bicubic,"
                "hwdownload,"
                "scale=iw:ih,"
                "format=yuv420p"
            )

        # ── Build variant list ────────────────────────────────────────────────
        # Each entry: (variant_id, label, hw_args, filter_chain)
        _placebo_suffix = "+libplacebo" if (test_is_hdr and libplacebo_available()) else ""
        _cpu_fchain = _cpu_filter(test_is_hdr)
        variants: List[Tuple[str, str, List[str], str]] = [
            ("cpu", f"CPU-only{_placebo_suffix}", [], _cpu_fchain),
        ]
        if self.use_cuda:
            for gpu_idx, gpu_name in available_gpus:
                hw_init   = ["-init_hw_device", f"cuda=hw:{gpu_idx}"]
                hw_decode = [*hw_init, "-hwaccel", "cuda"]
                hw_cuda   = [*hw_init, "-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]

                # Tier A: NVDEC decode only, CPU filter chain
                variants.append((
                    f"gpu{gpu_idx}_decode",
                    f"GPU {gpu_idx} ({gpu_name})  NVDEC decode only",
                    hw_decode,
                    _cpu_fchain,
                ))

                # Tier B: NVDEC + scale_cuda (GPU scale, CPU tonemap when HDR)
                if scale_cuda_available():
                    variants.append((
                        f"gpu{gpu_idx}_scale",
                        f"GPU {gpu_idx} ({gpu_name})  NVDEC + scale_cuda",
                        hw_cuda,
                        _scale_gpu_filter(test_is_hdr),
                    ))

                # Tier C: full-GPU tonemap (HDR only, requires libnpp)
                if test_is_hdr and tonemap_cuda_available():
                    variants.append((
                        f"gpu{gpu_idx}_full",
                        f"GPU {gpu_idx} ({gpu_name})  full-GPU tonemap_cuda+scale_cuda",
                        hw_cuda,
                        _full_gpu_filter(),
                    ))

        # ── Core single-run helper ─────────────────────────────────────────────
        nice_val = self.config.get("processing", {}).get("ffmpeg_nice", 10)

        def _bench_one(hw_args: List[str], fchain: str) -> Optional[float]:
            """Decode WARMUP+BENCH frames through the pipeline; return fps or None."""
            total   = WARMUP_FRAMES + BENCH_FRAMES
            fc_fd, fc_path = tempfile.mkstemp(suffix=".txt", prefix="bench_fc_")
            try:
                with os.fdopen(fc_fd, "w") as fh:
                    fh.write(f"[0:v]{fchain}[vout]")

                fc_args = (
                    ["-/filter_complex", fc_path]
                    if ffmpeg_ver >= 7
                    else ["-filter_complex_script", fc_path]
                )
                vsync = (
                    ["-fps_mode", "passthrough"]
                    if ffmpeg_ver >= 5
                    else ["-vsync", "0"]
                )
                cmd = [
                    "ffmpeg",
                    "-threads", "0", "-filter_threads", "0",
                    "-loglevel", "error",
                    *hw_args,
                    "-probesize", "100M", "-analyzeduration", "100M",
                    "-ss", str(SEEK_SEC),
                    "-i", test_video,
                    "-frames:v", str(total),
                    *fc_args,
                    "-map", "[vout]",
                    "-f", "rawvideo", "-pix_fmt", "yuv420p",
                    *vsync,
                    "pipe:1",   # direct raw-frame output to stdout for Python to read
                ]
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
                try:
                    psutil.Process(proc.pid).nice(nice_val)
                except Exception:
                    pass

                # Warmup: run the pipeline hot before timing begins
                for _ in range(WARMUP_FRAMES):
                    if len(proc.stdout.read(FRAME_BYTES)) < FRAME_BYTES:
                        proc.kill(); proc.wait()
                        return None

                # Timed benchmark
                t0 = time.monotonic()
                received = 0
                for _ in range(BENCH_FRAMES):
                    if len(proc.stdout.read(FRAME_BYTES)) < FRAME_BYTES:
                        break
                    received += 1
                elapsed = time.monotonic() - t0

                try:
                    proc.stdout.close()
                except Exception:
                    pass
                proc.kill(); proc.wait()

                if received < 10 or elapsed <= 0:
                    return None
                return received / elapsed

            except Exception:
                return None
            finally:
                try:
                    os.unlink(fc_path)
                except Exception:
                    pass

        # ── Run single-worker variants ────────────────────────────────────────
        COL = 55
        print(f"\n  {'Variant':<{COL}}  {'fps':>8}  Status")
        print(f"  {'─'*COL}  {'─'*8}  ──────")

        single_results: List[dict] = []
        n_total = len(variants)
        for i, (vid, label, hw_args, fchain) in enumerate(variants, 1):
            # Print the variant label with a trailing ellipsis while running
            prefix = f"  [{i}/{n_total}] {label}"
            print(f"{prefix:<{COL + 10}} …", end="", flush=True)

            fps = _bench_one(hw_args, fchain)

            if fps is not None:
                line = f"\r  [{i}/{n_total}] {label:<{COL - 7}}   {fps:7.1f}   ✓ OK"
            else:
                line = f"\r  [{i}/{n_total}] {label:<{COL - 7}}      n/a   ✗ pipeline unavailable"
            print(line)

            single_results.append({
                "variant_id": vid, "label": label, "fps": fps,
                "hw_args": hw_args, "filter": fchain,
            })

        # ── Parallel N-worker variants ────────────────────────────────────────
        # Build every meaningful (GPU-streams, CPU-streams) combination for
        # N = self.workers concurrent jobs, so we can answer:
        #   • Does adding more GPU decode streams help or saturate NVDEC?
        #   • Is it better to split N workers across two GPUs?
        #   • Is CPU fill-up after GPU streams worth it?
        #
        # Variant families tested (N = self.workers):
        #   A) K × best-GPU  +  (N-K) × CPU    K = 0 … N  (saturation sweep)
        #   B) K × GPU0  +  K × GPU1  + (N-2K) × CPU     K = 1 … N//2  (if ≥2 GPUs)
        #   C) balanced split:  ceil(N/2) × GPU0  +  floor(N/2) × GPU1 (if ≥2 GPUs)
        #
        # Each test runs all N threads simultaneously, measures combined fps.

        valid_singles = [r for r in single_results if r["fps"] is not None]
        best_single   = max(valid_singles, key=lambda r: r["fps"]) if valid_singles else None
        parallel_results: List[dict] = []
        N = max(self.workers, 2)   # at least 2 to make parallel testing meaningful

        # Best single-GPU config (cpu args + filter chain) and CPU config
        gpu_best_by_idx: Dict[int, dict] = {}   # gpu_device_int → best single result
        for r in single_results:
            vid_id = r["variant_id"]
            if r["fps"] and not vid_id.startswith("cpu"):
                try:
                    gidx = int(vid_id.split("_")[0].replace("gpu", ""))
                except ValueError:
                    continue
                if gidx not in gpu_best_by_idx or r["fps"] > gpu_best_by_idx[gidx]["fps"]:
                    gpu_best_by_idx[gidx] = r
        sorted_gpu_results = sorted(gpu_best_by_idx.values(),
                                    key=lambda r: r["fps"], reverse=True)
        cpu_result = next((r for r in single_results if r["variant_id"] == "cpu"), None)

        def _bench_n_workers(configs: List[Tuple[List[str], str]]) -> Optional[float]:
            """Run len(configs) FFmpeg workers simultaneously; return combined fps."""
            n = len(configs)
            fps_slots: List[Optional[float]] = [None] * n

            def _w(slot: int, hw: List[str], fc: str) -> None:
                fps_slots[slot] = _bench_one(hw, fc)

            threads = [
                threading.Thread(target=_w, args=(i, hw, fc), daemon=True)
                for i, (hw, fc) in enumerate(configs)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            valid = [f for f in fps_slots if f is not None]
            return sum(valid) if valid else None

        p_count = 0

        def _hw_to_worker_cfg(hw: List[str]) -> dict:
            """Convert hw_args list to a compact (use_cuda, cuda_device) dict."""
            if not hw:
                return {"use_cuda": False, "cuda_device": 0}
            # "-init_hw_device cuda=hw:N" encodes the device ordinal.
            for i, tok in enumerate(hw):
                if tok == "-init_hw_device" and i + 1 < len(hw):
                    val = hw[i + 1]
                    if val.startswith("cuda=hw:"):
                        try:
                            device = int(val.split(":")[1])
                            return {"use_cuda": True, "cuda_device": device}
                        except (ValueError, IndexError):
                            pass
            return {"use_cuda": True, "cuda_device": 0}

        def _run_par(vid_id: str, label: str, configs: List[Tuple[List[str], str]],
                     reference_fps: Optional[float] = None) -> None:
            nonlocal p_count
            p_count += 1
            n_w = len(configs)
            line_pfx = f"  [P{p_count:02d}/{n_w}w] {label}"
            print(f"{line_pfx:<{COL + 8}} …", end="", flush=True)
            fps = _bench_n_workers(configs)
            ratio = (
                f"×{fps / reference_fps:.2f}"
                if fps and reference_fps else "  n/a"
            )
            if fps:
                print(f"\r{line_pfx:<{COL + 8}}   {fps:7.1f} fps  {ratio:>6}")
            else:
                print(f"\r{line_pfx:<{COL + 8}}      n/a fps     n/a")
            parallel_results.append({
                "variant_id": vid_id,
                "label": label,
                "fps": fps,
                "n_workers": n_w,
                # Store per-worker decode config so _extract_film_parallel can
                # reproduce the optimal worker mix without re-running the benchmark.
                "worker_configs": [_hw_to_worker_cfg(hw) for hw, _ in configs],
            })

        if best_single:
            print(f"\n  Parallel variants  (N = {N} workers  —  simulates production load)")
            print(f"  {'─' * (_W - 2)}")
            print(f"  {'Label':<{COL + 8}}  {'total fps':>10}  {'vs 1×':>6}")
            print(f"  {'─' * (_W - 2)}")

            # ── Family A: K × best-GPU  +  (N-K) × CPU  (K = 0 … N) ──────────
            if sorted_gpu_results:
                bg = sorted_gpu_results[0]   # best GPU result overall
                bg_hw, bg_fc = bg["hw_args"], bg["filter"]
                bg_name = bg["label"]

                if cpu_result:
                    for k in range(0, N + 1):
                        if k == 0:
                            lbl = f"{N} × CPU-only  (GPU=0 baseline)"
                            vid_id = f"par_A_k0"
                            configs = [([], _cpu_fchain)] * N
                            ref = cpu_result["fps"]
                        elif k == N:
                            lbl = f"{N} × {bg_name}  (GPU only)"
                            vid_id = f"par_A_kN"
                            configs = [(bg_hw, bg_fc)] * N
                            ref = bg["fps"]
                        else:
                            lbl = (
                                f"{k} × {bg_name}  +  {N - k} × CPU"
                            )
                            vid_id = f"par_A_k{k}"
                            configs = (
                                [(bg_hw, bg_fc)] * k
                                + [([], _cpu_fchain)] * (N - k)
                            )
                            ref = bg["fps"]
                        _run_par(vid_id, lbl, configs, ref)
                else:
                    # No CPU baseline — just GPU saturation sweep
                    for k in range(1, N + 1):
                        lbl = f"{k} × {bg_name}"
                        vid_id = f"par_A_k{k}"
                        _run_par(vid_id, lbl, [(bg_hw, bg_fc)] * k, bg["fps"])

            elif cpu_result:
                # CUDA unavailable — test CPU scaling only
                for k in range(2, N + 1):
                    lbl = f"{k} × CPU-only"
                    _run_par(f"par_cpu_k{k}", lbl,
                             [([], _cpu_fchain)] * k, cpu_result["fps"])

            # ── Family B: K×GPU0 + K×GPU1 + (N-2K)×CPU  (if ≥2 GPUs) ─────────
            if len(sorted_gpu_results) >= 2:
                g0 = sorted_gpu_results[0]
                g1 = sorted_gpu_results[1]
                g0_hw, g0_fc = g0["hw_args"], g0["filter"]
                g1_hw, g1_fc = g1["hw_args"], g1["filter"]
                g0_name = g0["label"]
                g1_name = g1["label"]

                def _gpu_idx_str(r: dict) -> str:
                    """Extract plain GPU ordinal string from a single-result dict."""
                    return r["variant_id"].split("_")[0].replace("gpu", "")

                g0_idx = _gpu_idx_str(sorted_gpu_results[0])
                g1_idx = _gpu_idx_str(sorted_gpu_results[1])

                print(f"\n  {'─' * (_W - 2)}")
                print(f"  Dual-GPU split variants  (GPU {g0_idx} + GPU {g1_idx})")
                print(f"  {'─' * (_W - 2)}")

                for k in range(1, N // 2 + 1):
                    rest = N - 2 * k
                    if rest > 0 and cpu_result:
                        lbl = (
                            f"{k} × GPU{g0_idx}  +  {k} × GPU{g1_idx}  +  {rest} × CPU"
                        )
                        vid_id = f"par_B_k{k}_cpu{rest}"
                        configs = (
                            [(g0_hw, g0_fc)] * k
                            + [(g1_hw, g1_fc)] * k
                            + [([], _cpu_fchain)] * rest
                        )
                    else:
                        lbl = (
                            f"{k} × GPU{g0_idx}  +  {k} × GPU{g1_idx}  (no CPU fill)"
                        )
                        vid_id = f"par_B_k{k}_nogpu"
                        configs = [(g0_hw, g0_fc)] * k + [(g1_hw, g1_fc)] * k
                    ref = max(g0["fps"], g1["fps"])
                    _run_par(vid_id, lbl, configs, ref)

                # C) balanced split across both GPUs, no CPU
                # Use integer arithmetic to avoid floating-point edge cases.
                c0 = (N + 1) // 2
                c1 = N - c0
                lbl_c = (
                    f"{c0} × GPU{g0_idx}  +  {c1} × GPU{g1_idx}  (balanced, no CPU)"
                )
                _run_par("par_C_balanced", lbl_c,
                         [(g0_hw, g0_fc)] * c0 + [(g1_hw, g1_fc)] * c1,
                         max(g0["fps"], g1["fps"]))

        # ── Summary ───────────────────────────────────────────────────────────
        all_valid     = [r for r in single_results if r["fps"] is not None]
        winner        = max(all_valid, key=lambda r: r["fps"]) if all_valid else None
        all_par_valid = [r for r in parallel_results if r["fps"] is not None]
        par_winner    = (
            max(all_par_valid, key=lambda r: r["fps"]) if all_par_valid else None
        )

        print(f"\n  {_SEP2}")
        if winner:
            print(
                f"  🏆 Best single   : {winner['label']:<{COL - 5}}  {winner['fps']:7.1f} fps"
            )
        if par_winner:
            print(
                f"  🏆 Best parallel : {par_winner['label']:<{COL - 5}}  "
                f"{par_winner['fps']:7.1f} fps  ({par_winner['n_workers']} workers combined)"
            )

        if not all_valid:
            print("  ⚠️  All variants failed — keeping default decode settings.")
            print(f"\n{_SEP}\n")
            return

        # ── Apply winner to instance ──────────────────────────────────────────
        vid      = winner["variant_id"]
        use_cuda = not vid.startswith("cpu")
        cuda_device = 0
        if use_cuda:
            try:
                cuda_device = int(vid.split("_")[0].replace("gpu", ""))
            except Exception:
                cuda_device = 0

        self.use_cuda    = use_cuda
        self.cuda_device = cuda_device
        # Store per-worker decode configs for within-film parallel extraction.
        if par_winner and par_winner.get("worker_configs"):
            self._parallel_worker_configs = par_winner["worker_configs"]

        # Build the UI backend label: shows active pipeline + parallel fps
        _par_str = (
            f"  |  {par_winner['n_workers']}w parallel: {par_winner['fps']:.0f} fps"
            if par_winner and par_winner["fps"] else ""
        )
        self.ui_state["decode_backend"] = (
            f"{winner['label']}  [{winner['fps']:.0f} fps single{_par_str}]"
        )

        print(
            f"\n  ✅ Generator configured: use_cuda={use_cuda}, "
            f"cuda_device={cuda_device}"
        )
        print(f"  Active backend : {self.ui_state['decode_backend']}")

        # ── Save results to output dir ────────────────────────────────────────
        os.makedirs(self.base_dir, exist_ok=True)
        cache: dict = {
            "_ts":            time.time(),
            "_probe_version": _CACHE_VERSION,
            "_test_video":    test_video,
            "_is_hdr":        test_is_hdr,
            "_n_workers":     N,
            "_benchmark":     (
                f"{BENCH_FRAMES} frames timed + {WARMUP_FRAMES} warmup at "
                f"{W}×{H}, seek {SEEK_SEC:.0f}s, {N} workers"
            ),
            "best": {
                "variant_id":  vid,
                "label":       winner["label"],
                "use_cuda":    use_cuda,
                "cuda_device": cuda_device,
                "fps":         winner["fps"],
            },
            "best_parallel": (
                {
                    "variant_id":    par_winner["variant_id"],
                    "label":         par_winner["label"],
                    "fps":           par_winner["fps"],
                    "n_workers":     par_winner["n_workers"],
                    "worker_configs": par_winner.get("worker_configs"),
                }
                if par_winner else None
            ),
            "single_results": [
                {
                    "variant_id": r["variant_id"],
                    "label":      r["label"],
                    "fps":        round(r["fps"], 2) if r["fps"] else None,
                }
                for r in single_results
            ],
            "parallel_results": [
                {
                    "variant_id": r["variant_id"],
                    "label":      r["label"],
                    "fps":        round(r["fps"], 2) if r["fps"] else None,
                    "n_workers":  r["n_workers"],
                }
                for r in parallel_results
            ],
        }
        try:
            Path(cache_path).write_text(
                json.dumps(cache, indent=2, ensure_ascii=False)
            )
            print(f"  📝 Results saved → {cache_path}")
        except Exception as exc:
            print(f"  ⚠️  Could not save benchmark results: {exc}")
            self.logger.warning(f"Could not save benchmark results: {exc}")

        print(f"\n{_SEP}\n")
        self.logger.info(
            f"Decode benchmark complete — best: {winner['label']}  "
            f"[use_cuda={use_cuda}, cuda_device={cuda_device}, fps={winner['fps']:.1f}]"
        )

    # ── Config helpers ────────────────────────────────────────────────────────

    def _build_format_config(self) -> Dict[str, Dict[str, dict]]:
        """
        Build the ``format_config`` dict that the streaming extractor expects.

        Shape::

            {
              category_name: {
                template_name: {
                  "gt_size":               [W, H],
                  "lr_size":               [W, H],
                  "source_mode":           "resize" | "crop",
                  "degradation_mix":       {template_name: weight, …},
                  "degradation_templates": {template_name: {…spec…}, …},
                }
              }
            }

        All referenced format and degradation templates are resolved from
        ``self.templates``.  Any missing reference raises ``SystemExit`` because
        the config was already validated at startup; this is a safety net only.
        """
        fmt_tmpls = self.templates["format_templates"]
        deg_tmpls = self.templates["degradation_templates"]
        result: Dict[str, Dict[str, dict]] = {}

        for cat_name, cat_cfg in self.categories.items():
            result[cat_name] = {}
            for fmt_entry in cat_cfg["formats"]:
                tmpl_name = fmt_entry["template"]
                if tmpl_name not in fmt_tmpls:
                    print(f"❌ format_template '{tmpl_name}' not found (category '{cat_name}')")
                    sys.exit(1)
                fmt_spec = fmt_tmpls[tmpl_name]

                # Resolve degradation templates referenced in this format's mix.
                deg_mix = fmt_entry.get("degradation_mix", {})
                resolved_deg_tmpls: Dict[str, dict] = {}
                for dname in deg_mix:
                    if dname not in deg_tmpls:
                        print(f"❌ degradation_template '{dname}' not found (category '{cat_name}', format '{tmpl_name}')")
                        sys.exit(1)
                    resolved_deg_tmpls[dname] = deg_tmpls[dname]

                result[cat_name][tmpl_name] = {
                    "gt_size": fmt_spec["gt_size"],
                    "lr_size": fmt_spec["lr_size"],
                    "source_mode": fmt_entry["source_mode"],
                    "degradation_mix": deg_mix,
                    "degradation_templates": resolved_deg_tmpls,
                }

        return result

    def _write_architecture_file(self) -> None:
        """Write ``<base_dir>/dataset_architecture.json``.

        The file contains the complete configuration block that describes how
        the dataset was built: categories, format templates (with gt_size /
        lr_size / scale), source_mode, degradation mixes, and per-category
        targets.  The trainer can load this file to determine patch dimensions,
        scale factors, category distribution, and whether patches are crops or
        resizes without needing access to the original generator config.
        """
        from datetime import datetime as _dt

        # Collect all format templates and degradation templates actually used.
        used_fmt_tmpls: Dict[str, dict] = {}
        used_deg_tmpls: Dict[str, dict] = {}
        for cat_name, fmt_map in self.format_config.items():
            for tmpl_name, cfg in fmt_map.items():
                if tmpl_name not in used_fmt_tmpls:
                    raw_tmpl = self.templates.get("format_templates", {}).get(tmpl_name, {})
                    used_fmt_tmpls[tmpl_name] = raw_tmpl
                for dname, dspec in cfg.get("degradation_templates", {}).items():
                    if dname not in used_deg_tmpls:
                        used_deg_tmpls[dname] = dspec

        # Build per-category section with human-readable format entries.
        categories_out: Dict[str, dict] = {}
        for cat_name, cat_cfg in self.categories.items():
            formats_out = []
            for fmt_entry in cat_cfg.get("formats", []):
                tmpl_name = fmt_entry["template"]
                fc = self.format_config.get(cat_name, {}).get(tmpl_name, {})
                formats_out.append({
                    "template":        tmpl_name,
                    "weight":          fmt_entry.get("weight", 1),
                    "source_mode":     fmt_entry.get("source_mode", "resize"),
                    "gt_size":         fc.get("gt_size", []),
                    "lr_size":         fc.get("lr_size", []),
                    "scale":           (
                        used_fmt_tmpls.get(tmpl_name, {}).get("scale", 3)
                    ),
                    "aspect_ratio":    (
                        used_fmt_tmpls.get(tmpl_name, {}).get("aspect_ratio", "")
                    ),
                    "description":     (
                        used_fmt_tmpls.get(tmpl_name, {}).get("description", "")
                    ),
                    "degradation_mix": fmt_entry.get("degradation_mix", {}),
                })
            categories_out[cat_name] = {
                "target_total": self.category_targets.get(cat_name, 0),
                "formats":      formats_out,
            }

        proc = self.config.get("processing", {})
        _fmt_str = self.config.get("output_format", "bmp").lower()
        arch = {
            "generated_at":      _dt.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generator_version": "dataset_generator_v2",
            "root_path":         self.base_dir,
            "n_frames":          int(proc.get("n_frames", 7)),
            "output_format":     _fmt_str,
            "category_targets":  dict(self.category_targets),
            "categories":        categories_out,
            "format_templates":  used_fmt_tmpls,
            "degradation_templates": used_deg_tmpls,
        }

        out_path = os.path.join(self.base_dir, "dataset_architecture.json")
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(arch, fh, indent=2, ensure_ascii=False)
            self.logger.info(f"📄 Architecture file written: {out_path}")
        except Exception as exc:
            self.logger.warning(f"Could not write architecture file: {exc}")

    def _build_format_distribution_for_video(
        self,
        video: dict,
        category_patch_targets: Dict[str, int],
    ) -> Dict[str, Dict[str, int]]:
        """
        Build ``format_distribution = {category: {template_name: count}}``
        for a single video.

        Within each category the patch budget is split proportionally across
        the category's format entries using their ``weight`` values.  The last
        format entry absorbs any rounding remainder.

        Args:
            video:                  Video config dict.
            category_patch_targets: ``{category: patch_count}`` for this video.

        Returns:
            ``{category: {template_name: count}}``
        """
        video_cats = get_video_categories(video)
        distribution: Dict[str, Dict[str, int]] = {}

        for cat_name in video_cats:
            if cat_name not in category_patch_targets or cat_name not in self.categories:
                continue

            cat_total = category_patch_targets[cat_name]
            if cat_total <= 0:
                continue

            formats = self.categories[cat_name]["formats"]
            total_weight = sum(f["weight"] for f in formats)

            distribution[cat_name] = {}
            remaining = cat_total

            for i, fmt_entry in enumerate(formats):
                tmpl_name = fmt_entry["template"]
                if i == len(formats) - 1:
                    count = remaining
                else:
                    count = int(cat_total * fmt_entry["weight"] / total_weight)
                    remaining -= count
                distribution[cat_name][tmpl_name] = max(0, count)

        return distribution

    def _setup_logger(self):
        """Setup file and console logger (console disabled when terminal UI active)"""
        log_dir = os.path.join(self.base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        logger = logging.getLogger("DatasetGenerator")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []

        log_file = os.path.join(log_dir, f"generator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)

        if not self.use_terminal_ui:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(ch)
            logger.info("Console logging enabled (terminal UI disabled)")
        else:
            logger.info("Console logging disabled (terminal UI active - see GUI)")

        return logger

    def _show_priority_distribution(self):
        """Display priority distribution in console"""
        priority_counts: Dict[int, int] = {}
        for v in self.videos:
            p = v.get("priority", 255)
            priority_counts[p] = priority_counts.get(p, 0) + 1

        console.print("\n[bold]📋 Video Processing Order:[/bold]")
        sorted_priorities = sorted(priority_counts.keys())

        priorities_to_show = []
        if 255 in priority_counts:
            priorities_to_show = [
                p for p in sorted_priorities if p != 255
            ][: self.MAX_DISPLAYED_PRIORITIES - 1]
            priorities_to_show.append(255)
            priorities_to_show.sort()
        else:
            priorities_to_show = sorted_priorities[: self.MAX_DISPLAYED_PRIORITIES]

        for priority in priorities_to_show:
            count = priority_counts[priority]
            label = "(default)" if priority == 255 else ""
            console.print(f"   Priority {priority} {label}: {count} videos")

        remaining = [p for p in sorted_priorities if p not in priorities_to_show]
        if remaining:
            count = sum(priority_counts[p] for p in remaining)
            console.print(f"   ... and {count} more videos in other priority levels")

    def _load_metadata_cache(self):
        """Load video metadata cache from disk"""
        if os.path.exists(self.metadata_cache_file):
            try:
                with open(self.metadata_cache_file, 'r') as f:
                    cache = json.load(f)
                self.logger.info(f"Loaded metadata cache with {len(cache)} videos")
                return cache
            except Exception as e:
                self.logger.warning(f"Could not load metadata cache: {e}")
        return {}
    
    def _save_metadata_cache(self):
        """Save video metadata cache to disk"""
        try:
            os.makedirs(os.path.dirname(self.metadata_cache_file), exist_ok=True)
            with open(self.metadata_cache_file, 'w') as f:
                json.dump(self.metadata_cache, f, indent=2)
            self.logger.debug(f"Saved metadata cache with {len(self.metadata_cache)} videos")
        except Exception as e:
            self.logger.warning(f"Could not save metadata cache: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully - fast exit on Ctrl+C"""
        print("\n\n⚠️  Ctrl+C detected! Aborting immediately...")
        self.running = False
        # Always restore cursor/terminal regardless of use_terminal_ui flag
        show_cursor()
        # Save progress before exit
        if hasattr(self, 'tracker'):
            try:
                self.tracker.save()
                print("✓ Progress saved")
            except:
                pass
        # Immediate exit
        sys.exit(0)
    
    def _start_ui_heartbeat(self) -> None:
        """Start a background thread that refreshes the terminal UI every second.

        This guarantees at least one UI redraw per second even when the main
        thread is blocked (e.g. waiting on parallel FFmpeg workers).  The
        thread is a daemon so it is automatically killed if the process exits.
        """
        if not self.use_terminal_ui:
            return
        self._ui_heartbeat_stop.clear()

        def _beat() -> None:
            while not self._ui_heartbeat_stop.wait(timeout=1.0):
                # Force a redraw on the next call regardless of throttle timer
                self.last_update_time = 0.0
                self._update_terminal_ui()

        self._ui_heartbeat_thread = threading.Thread(
            target=_beat, daemon=True, name="ui-heartbeat"
        )
        self._ui_heartbeat_thread.start()

    def _stop_ui_heartbeat(self) -> None:
        """Stop the background UI heartbeat thread (idempotent)."""
        self._ui_heartbeat_stop.set()
        if self._ui_heartbeat_thread is not None:
            self._ui_heartbeat_thread.join(timeout=2.0)
            self._ui_heartbeat_thread = None

    def _update_terminal_ui(self):
        """Update and redraw the terminal UI (throttled to update_interval).

        Thread-safe: uses a non-blocking lock so concurrent calls from the
        heartbeat thread and worker callbacks never produce garbled output.
        """
        if not self.use_terminal_ui:
            return

        now = time.time()
        if now - self.last_update_time < self.update_interval:
            return

        # Skip if another thread is already redrawing
        if not self._ui_lock.acquire(blocking=False):
            return

        self.last_update_time = time.time()
        self.ui_update_counter += 1

        try:
            # Update overall progress from tracker
            category_stats = self.tracker.status.get('category_stats', {})
            for category in self.category_targets.keys():
                if category in category_stats:
                    stats = category_stats[category]
                    # Use the user-configured target (category_targets), not the
                    # rounded distribution sum (distribution_totals), so the progress
                    # bar reflects exactly what the user asked for (30 000 GT images
                    # means 30 000 GT images, not 29 850 due to per-video rounding).
                    target = self.category_targets.get(category, 0)
                    current = stats.get('images_created', 0)
                    percent = (current / target * 100) if target > 0 else 0.0
                    self.ui_state['overall_progress'][category] = {
                        'created': current,
                        'target': target,
                        'percent': percent,
                    }

            # Patch distribution by category and format — derive weights from
            # the category config instead of old format_probabilities dict.
            patch_dist = {}
            for category, fmt_map in self.format_config.items():
                patch_dist[category] = {}
                total_weight = sum(
                    fe["weight"]
                    for fe in self.categories.get(category, {}).get("formats", [])
                )
                for format_name in fmt_map:
                    # Look up the weight for this template name in the category.
                    weight = 0
                    for fe in self.categories.get(category, {}).get("formats", []):
                        if fe["template"] == format_name:
                            weight = fe["weight"]
                            break
                    prob = (weight / total_weight) if total_weight > 0 else 0.0
                    if category in category_stats:
                        cat_done = category_stats[category].get("images_created", 0)
                        cat_target = self.category_targets.get(category, 0)
                        patch_dist[category][format_name] = {
                            "count": int(cat_done * prob),
                            "target": int(cat_target * prob),
                        }
                    else:
                        patch_dist[category][format_name] = {"count": 0, "target": 0}
            self.ui_state["patch_distribution"] = patch_dist

            # ETA calculation: use global rate (total saved / elapsed)
            elapsed = time.time() - self.start_time
            patches_done = self.ui_state.get('patches_created_total', 0)
            if patches_done > 0 and elapsed > 0:
                rate = patches_done / elapsed
                eta_by_category = {}
                max_eta = 0
                for category in self.ui_state['overall_progress']:
                    cat_data = self.ui_state['overall_progress'][category]
                    remaining = cat_data['target'] - cat_data['created']
                    if remaining > 0 and rate > 0:
                        eta_s = remaining / rate
                        eta_by_category[category] = eta_s
                        max_eta = max(max_eta, eta_s)
                self.ui_state['eta'] = eta_by_category
                self.ui_state['eta']['total'] = max_eta

            clear_screen()
            draw_dataset_ui(self.ui_state)

        except Exception as e:
            self.logger.error(f"UI update error: {e}", exc_info=True)
        finally:
            self._ui_lock.release()
    
    def _log_system_resources(self, operation: str = ""):
        """Log current system resource usage"""
        try:
            # Get system memory info
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            self.logger.info(f"System Resources{' - ' + operation if operation else ''}:")
            self.logger.info(f"  RAM: {mem.used / (1024**3):.1f}GB / {mem.total / (1024**3):.1f}GB ({mem.percent}% used)")
            self.logger.info(f"  Available RAM: {mem.available / (1024**3):.1f}GB")
            self.logger.info(f"  Swap: {swap.used / (1024**3):.1f}GB / {swap.total / (1024**3):.1f}GB ({swap.percent}% used)")
            
            # Warn if memory is getting low
            if mem.percent > 90:
                self.logger.warning("⚠️  WARNING: RAM usage >90%! Risk of OOM kill!")
            elif mem.percent > 80:
                self.logger.warning("⚠️  WARNING: RAM usage >80%! Monitor carefully!")
        except Exception as e:
            self.logger.debug(f"Could not log system resources: {e}")

    def scan_video_durations(self) -> Dict[str, float]:
        """
        Scan all videos to get their durations.
        This is Phase 1 - required for proportional distribution.
        
        Returns:
            Dictionary mapping video_path -> duration in seconds
        """
        self.logger.info("=" * 80)
        self.logger.info("PHASE 1: Scanning Video Durations")
        self.logger.info("=" * 80)
        
        # Log system resources before heavy operation
        self._log_system_resources("Before video scanning")
        
        if RICH_AVAILABLE:
            console.print("\n[bold cyan]📹 Phase 1: Scanning Video Durations[/bold cyan]")
            console.print("Analyzing all videos to calculate proportional distribution...")
        
        durations = {}
        total_duration = 0.0
        errors = 0
        
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console if RICH_AVAILABLE else None
            ) as progress:
                
                task = progress.add_task("Scanning videos...", total=len(self.videos))
                
                for idx, video in enumerate(self.videos):
                    video_path = video['path']
                    video_name = video.get('name', os.path.basename(video_path))
                    
                    try:
                        if not os.path.exists(video_path):
                            self.logger.warning(f"Video not found: {video_path}")
                            errors += 1
                            progress.update(task, advance=1)
                            continue
                        
                        # Get video metadata with timeout protection
                        try:
                            metadata = self._get_video_metadata(video_path)
                        except Exception as e:
                            self.logger.error(f"Error getting metadata for {video_name}: {e}")
                            errors += 1
                            progress.update(task, advance=1)
                            continue
                        
                        if metadata and 'duration' in metadata:
                            duration = metadata['duration']
                            durations[video_path] = duration
                            total_duration += duration

                            hdr_label = "HDR" if metadata.get('is_hdr', True) else "SDR"
                            ct = metadata.get('color_transfer') or 'unknown'
                            progress.update(task, description=f"Scanned: {video_name[:40]}...", advance=1)
                            # Log with newline for clean output
                            print(f"Scanned: {video_name}: {duration:.1f}s [{hdr_label}, {ct}]")
                            self.logger.debug(f"Scanned {video_name}: {duration:.1f}s [{hdr_label}, {ct}]")
                        else:
                            self.logger.warning(f"Could not get duration for: {video_name}")
                            errors += 1
                            progress.update(task, advance=1)
                            
                    except Exception as e:
                        self.logger.error(f"Unexpected error scanning {video_name}: {e}")
                        errors += 1
                        progress.update(task, advance=1)
                        continue
        
        except Exception as e:
            self.logger.error(f"FATAL: Error during video scanning progress display: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        if RICH_AVAILABLE:
            console.print(f"\n✓ Scanned {len(durations)} videos")
            console.print(f"✓ Total duration: {total_duration/3600:.1f} hours ({total_duration:.0f} seconds)")
            if errors > 0:
                console.print(f"⚠️  Errors: {errors} videos could not be scanned")
        
        # Save metadata cache after scanning
        try:
            self._save_metadata_cache()
        except Exception as e:
            self.logger.warning(f"Could not save metadata cache: {e}")
        
        self.logger.info(f"Scan complete: {len(durations)} videos, total duration: {total_duration:.1f}s, errors: {errors}")
        self._log_system_resources("After video scanning")
        
        return durations
    
    def calculate_proportional_distribution(self, durations: Dict[str, float]) -> Dict[str, Dict[str, int]]:
        """
        Calculate how many patches each video should get PER CATEGORY based on its duration.
        This is Phase 2 - distribute proportionally PER CATEGORY (NOT globally!).
        
        CRITICAL: Each category target is divided among videos IN THAT CATEGORY only.
        Example: Master 200k target divided among 63 master videos, not all 500 videos!
        
        Args:
            durations: Dictionary of video_path -> duration in seconds
        
        Returns:
            Dictionary of video_path -> {category: patches_for_category}
        """
        self.logger.info("=" * 80)
        self.logger.info("PHASE 2: Calculating PER-CATEGORY Proportional Distribution")
        self.logger.info("=" * 80)
        
        try:
            # Store results as: video_path -> {category: patch_count}
            video_targets = {}
            
            # Initialize all videos
            for video_path in durations.keys():
                video_targets[video_path] = {}
            
            if RICH_AVAILABLE:
                console.print(f"\n[bold cyan]📊 Phase 2: Calculating PER-CATEGORY Distribution[/bold cyan]")
            
            # For EACH category separately
            for category, category_target in self.category_targets.items():
                self.logger.info(f"\n  Processing category: {category} (target: {category_target:,})")

                # Separate videos into forced-frames and proportional buckets.
                # Videos whose 'forced_frames' dict contains a positive value for
                # this category get that exact count; the remainder of the category
                # budget is distributed proportionally among the other videos.
                forced_videos: Dict[str, int] = {}   # path → forced_count
                normal_videos = []                   # (path, name, duration)
                normal_total_duration = 0.0
                forced_total = 0

                for v in self.videos:
                    video_cats = get_video_categories(v)
                    if category not in video_cats:
                        continue
                    video_path = v['path']
                    if video_path not in durations:
                        continue
                    forced = v.get('forced_frames', {}).get(category, 0)
                    if forced > 0:
                        forced_videos[video_path] = forced
                        forced_total += forced
                        self.logger.info(
                            f"    ⚡ {v.get('name','?')}: forced {forced:,} frames "
                            f"for category '{category}'"
                        )
                    else:
                        dur = durations[video_path]
                        normal_videos.append((video_path, v['name'], dur))
                        normal_total_duration += dur

                # Budget remaining after honouring forced frames
                remaining_budget = max(0, category_target - forced_total)
                if forced_total > 0:
                    self.logger.info(
                        f"    Category '{category}': target {category_target:,}, "
                        f"forced {forced_total:,}, remaining for proportional "
                        f"distribution: {remaining_budget:,}"
                    )

                self.logger.info(
                    f"    {category}: {len(forced_videos)} forced + "
                    f"{len(normal_videos)} proportional videos, "
                    f"{normal_total_duration/3600:.1f} hours proportional"
                )

                if len(normal_videos) == 0 and not forced_videos:
                    self.logger.warning(f"    No videos or zero duration for {category}, skipping")
                    continue

                # Apply forced-frame targets
                for video_path, forced_count in forced_videos.items():
                    video_targets[video_path][category] = forced_count

                # Distribute remaining_budget proportionally among normal videos
                for video_path, video_name, duration in normal_videos:
                    if normal_total_duration > 0:
                        patches = int(remaining_budget * duration / normal_total_duration)
                    else:
                        patches = 0
                    video_targets[video_path][category] = patches
                    self.logger.debug(
                        f"      {video_name}: {duration:.0f}s "
                        f"({duration / normal_total_duration * 100:.1f}% of proportional pool) "
                        f"→ {patches} patches"
                        if normal_total_duration > 0 else
                        f"      {video_name}: {duration:.0f}s → {patches} patches (zero-duration pool)"
                    )
            
            # Show summary
            self.logger.info("\n  Per-video summary (top 10 by total patches):")
            
            # Calculate total patches per video
            video_totals = {}
            for video_path, cat_patches in video_targets.items():
                video_totals[video_path] = sum(cat_patches.values())
            
            # Sort by total patches
            sorted_videos = sorted(video_totals.items(), key=lambda x: x[1], reverse=True)[:10]
            
            for video_path, total_patches in sorted_videos:
                video_name = "Unknown"
                for v in self.videos:
                    if v['path'] == video_path:
                        video_name = v['name']
                        break
                
                cat_breakdown = video_targets[video_path]
                cat_str = ", ".join([f"{cat}: {cnt}" for cat, cnt in cat_breakdown.items()])
                self.logger.info(f"    {video_name}: {total_patches} total ({cat_str})")
            
            # Calculate actual totals from distribution (not raw targets)
            # This is what will actually be created based on video assignments
            self.distribution_totals = {}
            for category in self.category_targets.keys():
                total = 0
                for video_path, cat_targets in video_targets.items():
                    if category in cat_targets:
                        total += cat_targets[category]
                self.distribution_totals[category] = total
            
            self.logger.info(f"\n📊 Actual Distribution Totals (sum of all video assignments):")
            for cat, total in self.distribution_totals.items():
                self.logger.info(f"  {cat}: {total:,} patches")
            
            return video_targets
            
        except Exception as e:
            self.logger.error(f"FATAL: Error in calculate_proportional_distribution: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_temp_dir(self, prefix: str = "extract") -> str:
        """
        Create a temporary directory in the configured temp location.
        
        Args:
            prefix: Prefix for temp directory name
            
        Returns:
            Path to created temp directory
        """
        # Ensure base temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Create unique subdirectory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        temp_subdir = os.path.join(self.temp_dir, f"{prefix}_{timestamp}")
        os.makedirs(temp_subdir, exist_ok=True)
        
        return temp_subdir
    
    def extract_frames_uhd(self, video_path: str, start_time: float, n_frames: int = 7,
                           is_hdr: Optional[bool] = None) -> Optional[Dict]:
        """
        LEGACY helper — not part of the production extraction path.

        The production path is:
            run() → _run_multi_stream() → _extract_film_parallel()
                  → extract_and_save_streaming_distributed()

        This method is retained only because ``extract_frames_single_mode``
        calls it for diagnostic / fallback purposes.  It should **not** be
        called from any hot path.

        Output format is BMP (lossless, no compression) written to a temp
        directory.  Callers must clean up the returned ``temp_dir``.

        Args:
            video_path: Path to video
            start_time: Start timestamp
            n_frames: Number of frames (7 or 5)
            is_hdr: Override HDR detection; ``None`` → auto-detect.

        Returns:
            Dict with ``'frame_paths'`` (list) and ``'temp_dir'`` (must be
            cleaned up by caller), or ``None`` on failure.
        """
        temp_dir = None
        # Use BMP as temp format — lossless, no compression overhead, no
        # external dependency.  PNG would impose unnecessary CPU cost here.
        _ext = "bmp"
        try:
            if is_hdr is None:
                meta = self._get_video_metadata(video_path)
                is_hdr = meta.get('is_hdr', True) if meta else True

            temp_dir = self._create_temp_dir("extract_single")
            output_pattern = os.path.join(temp_dir, f"frame_%04d.{_ext}")

            # CPU-only; this path is used only for diagnostics / small jobs.
            vf_filter = build_vf_filter(is_hdr=is_hdr, use_cuda=False)

            cmd = [
                'nice', '-n', '19',
                'ffmpeg',
                '-threads', str(self.workers),
                '-ss', str(start_time),
                '-i', video_path,
                '-vf', vf_filter,
                '-frames:v', str(n_frames),
                '-y',
                output_pattern
            ]

            self.logger.debug(f"[LEGACY extract_frames_uhd] command: {' '.join(cmd)}")

            timeout = self.config.get("processing", {}).get("ffmpeg_timeout", 120)
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout
            )

            if result.returncode != 0:
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                return None

            frame_paths = []
            for i in range(1, n_frames + 1):
                frame_path = os.path.join(temp_dir, f"frame_{i:04d}.{_ext}")
                if not os.path.exists(frame_path):
                    if temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    return None
                frame_paths.append(frame_path)

            return {
                'frame_paths': frame_paths,
                'temp_dir': temp_dir,
            }

        except Exception as e:
            self.logger.error(f"[LEGACY extract_frames_uhd] Error: {e}")
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            return None
    
    def _run_ffmpeg_with_progress(self, cmd: List[str], description: str = "FFmpeg", timeout: int = 300) -> int:
        """
        Run FFmpeg command and display progress in real-time.
        Shows only the progress line (frame, fps, time, speed).
        
        Args:
            cmd: FFmpeg command as list
            description: Description to show before progress
            timeout: Timeout in seconds
            
        Returns:
            Return code (0 = success)
        """
        import re
        
        try:
            # Start FFmpeg process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            print(f"\n{description}:")
            
            # Read stderr line by line for progress
            last_line = ""
            for line in iter(process.stderr.readline, ''):
                # FFmpeg progress format: frame= 123 fps=25 q=... time=00:00:05.00 bitrate=... speed=1.0x
                if 'frame=' in line and 'fps=' in line:
                    # Extract the progress line
                    # Clean up the line
                    progress_line = line.strip()
                    # Display with carriage return to update same line
                    print(f"\r  {progress_line}", end='', flush=True)
                    last_line = progress_line
            
            # Wait for process to complete
            returncode = process.wait(timeout=timeout)
            
            # Print newline after progress
            if last_line:
                print()  # Move to next line
            
            return returncode
            
        except subprocess.TimeoutExpired:
            process.kill()
            print(f"\n  ERROR: FFmpeg timeout after {timeout}s")
            return -1
        except Exception as e:
            self.logger.error(f"Error running FFmpeg: {e}")
            return -1
    
    def extract_frames_batch_uhd(self, video_path: str, timestamps: List[float],
                                 n_frames: int = 7, fps: float = 25.0) -> Dict:
        """
        Extract frames using SINGLE extraction mode (one FFmpeg call per timestamp).
        
        Returns PATHS to frames (memory-efficient) instead of loading into RAM.
        
        MEMORY OPTIMIZATION: Does NOT keep frames in memory!
        Returns dict of file paths. Caller must load frames when needed and clean up temp_dirs.
        
        This approach:
        - One FFmpeg call per timestamp (simple, proven)
        - Uses -ss for fast seeking before input
        - Extracts exactly n_frames to disk
        - Returns PATHS (not frames) - memory efficient!
        - Proven to work reliably (no mysterious failures)
        
        Args:
            video_path: Path to video file
            timestamps: List of start timestamps to extract from
            n_frames: Number of consecutive frames per timestamp (default 7)
            fps: Video frame rate (NOT used in single mode)
        
        Returns:
            Dict with 'frame_paths' (mapping timestamp -> list of file paths)
            and 'temp_dirs' (list of temp directories to clean up)
        """
        if not timestamps:
            return {'frame_paths': {}, 'temp_dirs': []}
        
        # Sort timestamps for predictable extraction order
        sorted_ts = sorted(timestamps)
        
        # USE SINGLE EXTRACTION MODE (reliable!)
        # User explicitly requested: "we'll go back to single extraction mode with ss"
        self.logger.info(f"Extracting {len(sorted_ts)} scenes using SINGLE extraction mode (reliable, memory-efficient):")
        
        frame_paths_dict = {}
        temp_dirs = []
        total_frames = 0
        
        for idx, ts in enumerate(sorted_ts, 1):
            # Call the proven extract_frames_uhd() method for each timestamp
            # Returns PATHS (not frames!) - memory efficient
            result = self.extract_frames_uhd(video_path, ts, n_frames)
            if result and result['frame_paths']:
                frame_paths_dict[ts] = result['frame_paths']
                temp_dirs.append(result['temp_dir'])
                total_frames += len(result['frame_paths'])
                self.logger.info(f"  Timestamp {ts:.1f}s: ✓ {len(result['frame_paths'])} frames ({total_frames}/{len(sorted_ts)*n_frames} total)")
            else:
                self.logger.warning(f"  Timestamp {ts:.1f}s: ✗ Failed to extract frames")
        
        # Summary
        success_count = len(frame_paths_dict)
        self.logger.info(f"✓ Extraction complete: {success_count}/{len(sorted_ts)} timestamps successful, {total_frames}/{len(sorted_ts)*n_frames} frames extracted")
        self.logger.info(f"💾 Memory-efficient: Frames on disk (NOT in RAM)")

        return {
            "frame_paths": frame_paths_dict,
            "temp_dirs": temp_dirs  # Caller MUST clean up!
        }

    def process_video(self, video_idx: int, category_targets: Dict[str, int] = None) -> Dict[str, int]:
        """
        Process a single video and extract patches for all assigned categories.

        Format distribution within each category is derived from the weighted
        ``formats`` list in ``categories[cat_name]`` via
        :meth:`_build_format_distribution_for_video`.

        Args:
            video_idx:        Index in ``self.videos``.
            category_targets: ``{category: patch_count}`` for this video.
                              Must be provided; comes from
                              :meth:`calculate_proportional_distribution`.

        Returns:
            ``{category: patches_created_count}`` or a sentinel dict with
            ``"skipped": True`` when the video is skipped.
        """
        if video_idx >= len(self.videos):
            return {}
        
        video = self.videos[video_idx]
        video_path = video['path']
        video_name = video['name']
        self.current_video_name = video_name
        
        # Skip videos without any category assignments
        video_categories = video.get('categories', {})
        if not video_categories:
            self.logger.info(f"⏭️  Skipping video {video_idx + 1}/{len(self.videos)}: {video_name} (no categories assigned)")
            return {'skipped': True, 'reason': 'no_categories'}
        
        if not os.path.exists(video_path):
            self.logger.warning(f"Video not found: {video_path}")
            return {}

        self.logger.info(f"Processing video {video_idx + 1}/{len(self.videos)}: {video_name}")

        metadata = self._get_video_metadata(video_path)
        if not metadata:
            return {}

        duration = metadata["duration"]

        # Build format distribution using the new weight-based method.
        # category_targets is always the per-category patch budget dict.
        if not category_targets:
            self.logger.warning(f"No category targets for video: {video_name}")
            return {}

        format_distribution = self._build_format_distribution_for_video(
            video, category_targets
        )

        if not format_distribution:
            self.logger.warning(f"No valid format distribution for video: {video_name}")
            return {}

        target_total = sum(category_targets.values())
        self.logger.info(
            f"Format distribution for {video_name} (target: {target_total} total):"
        )
        for category, formats in format_distribution.items():
            total = sum(formats.values())
            self.logger.info(f"  {category} ({total} patches): {formats}")

        # n_frames from processing config (default 7)
        proc = self.config.get("processing", {})
        n_frames = int(proc.get("n_frames", 7))

        fps = metadata.get("fps", 25.0) or 25.0
        is_hdr = metadata.get("is_hdr", True)
        color_trc: str = metadata.get("color_transfer") or "smpte2084"

        self.logger.info(
            f"  Color format: {metadata.get('color_transfer', 'unknown')!r} "
            f"→ {'HDR tonemap' if is_hdr else 'SDR pass-through'}"
        )

        patches_created = self._extract_patches_multi_format_batch(
            video_path, duration, format_distribution, n_frames, video_name, fps, video_idx,
            is_hdr=is_hdr,
            color_trc=color_trc,
        )
        
        return patches_created

    def _extract_film_parallel(
        self,
        video_path: str,
        assignments: List[Tuple[int, str, str]],
        n_frames: int,
        fps: float,
        is_hdr: bool,
        prior_total: int,
        color_trc: str = "smpte2084",
    ) -> Dict[str, int]:
        """Run N parallel streaming extractors on temporal segments of the same film.

        Splits *assignments* into N temporally-ordered groups and launches one
        ``extract_and_save_streaming_distributed`` worker per group.  Each worker
        opens its own FFmpeg process and decodes only the frames in its slice,
        reducing per-worker decode cost from ``|film_frames|`` to
        ``|film_frames| / N``.

        The optimal N and per-worker decode configs (GPU vs CPU) come from
        ``self._parallel_worker_configs``, which is populated from the stored
        ``decode_benchmark.json`` when the benchmark is run.

        Args:
            video_path:   Path to the source video file.
            assignments:  All (center_frame, category, format_name) assignments
                          for this film — will be split across workers.
            n_frames:     Rolling-buffer window size (same for all workers).
            fps:          Video frame rate.
            is_hdr:       Whether the source uses an HDR transfer function.
            prior_total:  Cumulative patch count before this film (for UI).
            color_trc:    Transfer-function string from ffprobe (e.g. "smpte2084").

        Returns:
            Dict mapping category name → number of patches created.
        """
        worker_configs = self._parallel_worker_configs  # guaranteed not None
        n = len(worker_configs)

        # Resolve output format from config (default: BMP for write throughput).
        _fmt_str = self.config.get("output_format", "bmp").lower()
        output_format = OutputFormat.BMP if _fmt_str == "bmp" else OutputFormat.PNG

        # Vulkan device pool for round-robin assignment (libplacebo path).
        # Uses _vulkan_device_pool (FFmpeg Vulkan indices, not CUDA indices).
        # Entries may be None when a CUDA GPU could not be mapped to a Vulkan
        # device; FFmpeg will then pick any available Vulkan device for that slot.
        _vk_pool = self._vulkan_device_pool  # [(vk_idx | None), …]
        _n_vk = len(_vk_pool)  # 0 is handled below as CPU-only

        # Split assignments into N temporally-ordered chunks.
        # Temporal ordering ensures each worker's FFmpeg process seeks to a
        # compact range and decodes a contiguous slice of the film, avoiding
        # large decode ranges caused by interleaved assignments.
        sorted_asgn = sorted(assignments, key=lambda a: a[0])
        chunk = max(1, (len(sorted_asgn) + n - 1) // n)
        groups: List[List[Tuple[int, str, str]]] = [
            sorted_asgn[i * chunk : (i + 1) * chunk]
            for i in range(n)
        ]
        groups = [g for g in groups if g]  # drop empty trailing groups

        nice_level = self.config.get("processing", {}).get("ffmpeg_nice", 10)
        center_snap = self.config.get("processing", {}).get("center_snap_seconds", 1.0)

        patches_lock = threading.Lock()
        patches_total: Dict[str, int] = {}
        # Track per-category totals for UI (not accurate until workers finish,
        # but used for the final progress bar update).
        _tracker_updated: Dict[str, int] = {}

        # Per-worker live patch counts (worker_idx → {cat: count}).
        # Written under patches_lock by each worker's progress callback;
        # read by _update_terminal_ui (called from the heartbeat thread).
        _worker_live: Dict[int, Dict[str, int]] = {}

        # Initialise per-worker stream-state entries for the terminal UI.
        # Truncate to the last 40 chars so the name fits in one terminal line.
        _video_stem = Path(video_path).stem[-40:]
        _stream_states: List[dict] = []
        for _wi in range(len(groups)):
            _vk_idx = _vk_pool[_wi % _n_vk] if _n_vk > 0 else None
            # For display: find a human-readable GPU name.  We look up by Vulkan
            # index in _available_gpu_names (which is keyed by Vulkan index after
            # the mapping in __init__).  Fall back to generic label when unknown.
            _gpu_display_name = (
                self._available_gpu_names.get(_vk_idx, f"Vulkan {_vk_idx}")
                if _vk_idx is not None else "CPU (software Vulkan)"
            )
            _stream_states.append({
                "stream_id": _wi,
                "video_name": _video_stem,
                "gpu_index": _vk_idx if _vk_idx is not None else -1,
                "gpu_name": _gpu_display_name,
                "state": "queued",
                "frames_processed": 0,
                "patches_created": 0,
                "write_queue_depth": 0,
            })
        with patches_lock:
            self.ui_state["active_streams"] = _stream_states
            self.ui_state["n_active_streams"] = len(_stream_states)

        def _make_progress_fn(worker_idx: int):
            """Return a thread-safe progress callback for one parallel worker."""
            def _on_progress(frames_examined: int, patches_so_far: Dict[str, int],
                             raw_frames_piped: int, timing: dict = None) -> None:
                with patches_lock:
                    _worker_live[worker_idx] = dict(patches_so_far)
                    # Aggregate partial counts from all workers
                    agg: Dict[str, int] = {}
                    for wp in _worker_live.values():
                        for cat, cnt in wp.items():
                            agg[cat] = agg.get(cat, 0) + cnt
                    self.ui_state['patches_created_total'] = prior_total + sum(agg.values())
                    # Update per-video progress bars with live aggregated counts
                    current_progress = self.ui_state.get('current_video_progress', {})
                    for cat, new_total in agg.items():
                        if cat in current_progress:
                            target = current_progress[cat].get('target', 0)
                            pct = (new_total / target * 100) if target > 0 else 0.0
                            current_progress[cat]['created'] = new_total
                            current_progress[cat]['percent'] = pct
                    # Update active stream entry for this worker.
                    if worker_idx < len(self.ui_state["active_streams"]):
                        _s = self.ui_state["active_streams"][worker_idx]
                        _s["state"] = "running"
                        _s["frames_processed"] = raw_frames_piped
                        _s["patches_created"] = sum(patches_so_far.values())
                        if timing:
                            _s["write_queue_depth"] = timing.get("q_size_last", 0)
                        self.ui_state["n_active_streams"] = sum(
                            1 for s in self.ui_state["active_streams"]
                            if s["state"] == "running"
                        )
            return _on_progress

        def _run_worker(worker_idx: int, group: List[Tuple[int, str, str]], wcfg: dict) -> None:
            # Round-robin Vulkan device assignment (libplacebo path).
            # Uses _vulkan_device_pool (FFmpeg Vulkan indices, already mapped
            # from CUDA indices in __init__).  None means "FFmpeg picks any".
            _vk_idx = _vk_pool[worker_idx % _n_vk] if _n_vk > 0 else None
            # Mark this stream as running.
            with patches_lock:
                if worker_idx < len(self.ui_state["active_streams"]):
                    self.ui_state["active_streams"][worker_idx]["state"] = "running"
            result = extract_and_save_streaming_distributed(
                video_path=video_path,
                assignments=group,
                n_frames=n_frames,
                format_config=self.format_config,
                base_dir=self.base_dir,
                fps=fps,
                logger=self.logger,
                is_interesting_fn=self.is_interesting_patch,
                is_black_frame_fn=_streaming_is_black_frame,
                # Live per-frame progress from every worker feeds into ui_state;
                # the heartbeat thread picks it up every second for display.
                progress_fn=_make_progress_fn(worker_idx),
                use_cuda=wcfg.get("use_cuda", False),
                cuda_device=wcfg.get("cuda_device", 0),
                nice_level=nice_level,
                is_hdr=is_hdr,
                center_snap_seconds=center_snap,
                stream_width=STREAM_OPT_WIDTH,
                stream_height=STREAM_OPT_HEIGHT,
                color_trc=color_trc,
                vulkan_device=_vk_idx,
                output_format=output_format,
            )
            with patches_lock:
                # Mark stream done.
                if worker_idx < len(self.ui_state["active_streams"]):
                    self.ui_state["active_streams"][worker_idx]["state"] = "done"
                for cat, count in result.items():
                    patches_total[cat] = patches_total.get(cat, 0) + count
                    # Update tracker once per worker completion (not per-frame).
                    delta = count - _tracker_updated.get(cat, 0)
                    if delta > 0:
                        self.tracker.increment_category_images(cat, delta)
                        _tracker_updated[cat] = count
                # Refresh UI so the progress bars advance as workers finish.
                self.ui_state['patches_created_total'] = (
                    prior_total + sum(patches_total.values())
                )
                self.ui_state["n_active_streams"] = sum(
                    1 for s in self.ui_state["active_streams"]
                    if s["state"] == "running"
                )
                self.last_update_time = 0.0   # force immediate redraw
                self._update_terminal_ui()

        n_workers_configured = len(worker_configs)
        threads = [
            threading.Thread(
                target=_run_worker,
                args=(i, g, w),
                daemon=True,
            )
            for i, (g, w) in enumerate(zip(groups, worker_configs))
        ]
        n_actual = len(threads)
        self.logger.info(
            f"⚡ Parallel film extraction: {n_actual} workers × "
            f"~{len(sorted_asgn) // max(n_actual, 1)} assignments each "
            f"(total {len(sorted_asgn)} assignments)"
        )
        # Signal the UI that parallel workers are running so the status line
        # shows something informative while workers are busy.
        self.ui_state['parallel_status'] = (
            f"⚡ {n_actual} parallele Worker aktiv …"
        )
        self.last_update_time = 0.0
        self._update_terminal_ui()

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Clear the parallel-status indicator and active stream states
        # now that all workers are done.
        self.ui_state['parallel_status'] = ""
        self.ui_state["active_streams"] = []
        self.ui_state["n_active_streams"] = 0

        return patches_total

    def _extract_patches_multi_format_batch(self, video_path: str, duration: float,
                                           format_distribution: Dict[str, Dict[str, int]],
                                           n_frames: int, video_name: str, fps: float = 25.0,
                                           video_idx: int = 0, is_hdr: bool = True,
                                           color_trc: str = "smpte2084") -> Dict[str, int]:
        """
        OPTIMIZED: Extract patches using BATCH frame extraction (10-50x faster).
        
        Uses extract_frames_batch_uhd() to extract all needed frames in ONE FFmpeg call,
        then processes them into patches. This is much faster than calling FFmpeg
        thousands of times.
        
        Args:
            video_path: Path to video file
            duration: Video duration in seconds
            format_distribution: Dict of {category: {format_name: target_count}}
            n_frames: Number of frames to extract (5 or 7)
            video_name: Video name for logging
            fps: Video frame rate (default 25.0)
            is_hdr: Whether the source video uses HDR transfer (PQ/HLG).
                    When False a lightweight scale-only chain is used instead
                    of the full HDR→SDR tonemap pipeline.
        
        Returns:
            Dict of {category: patches_created_count}
        """
        import time
        start_time = time.time()
        
        # Initialize counters
        patches_created = {}
        patches_targets = {}
        
        for category, formats in format_distribution.items():
            patches_created[category] = 0
            patches_targets[category] = {}
            for format_name, target_count in formats.items():
                patches_targets[category][format_name] = {
                    'target': target_count,
                    'created': 0
                }
        
        total_target = sum(sum(formats.values()) for formats in format_distribution.values())
        
        self.logger.info(f"╔══════════════════════════════════════════════════════════╗")
        self.logger.info(f"║  BATCH EXTRACTION MODE (OPTIMIZED)                       ║")
        self.logger.info(f"╚══════════════════════════════════════════════════════════╝")
        self.logger.info(f"📹 Video: {video_name}")
        self.logger.info(f"🎯 Target: {total_target} patches across {len(format_distribution)} categories")
        
        # Phase 1: Build per-category assignments independently.
        #
        # Rule: within each category every scene appears at most ONCE (assigned
        # to exactly one format).  Across categories the same video position
        # CAN appear in multiple categories – e.g. 5 000 scenes for master and
        # 2 000 for universal gives 7 000 assignments fed into one streaming pass.
        usable_duration = duration - 1.0
        assignments = build_assignments_per_category(
            format_distribution=format_distribution,
            duration=duration,
            fps=fps,
            n_frames=n_frames,
        )

        from collections import Counter as _Counter
        cat_counts = _Counter(cat for _, cat, _ in assignments)
        self.logger.info(f"\n📊 Assignments per category (independent sets):")
        for cat, cnt in sorted(cat_counts.items()):
            self.logger.info(f"  {cat}: {cnt} unique scenes")
        self.logger.info(f"  Total: {len(assignments)} assignments → one streaming pass")

        # Phase 2: Stream the video once, saving patches via progress callback.
        self.logger.info(f"\n🚀 SINGLE-PASS streaming extraction "
                         f"({n_frames}-frame rolling buffer, no seeking)…")

        # Snapshot of patches already counted before this video so the callback
        # can report a cumulative total in ui_state['patches_created_total'].
        prior_total: int = self.ui_state.get('patches_created_total', 0)
        # raw frames already accumulated by previous videos (so the cumulative
        # frames_read_total is correct across all videos)
        self._prior_raw_frames: int = self.ui_state.get('frames_read_total', 0)
        # Per-category counts already tracked (for delta-based tracker updates).
        last_tracker: Dict[str, int] = {cat: 0 for cat in patches_created}
        # Wall-clock start for per-video FPS / SPS measurement
        video_t0: float = time.monotonic()

        def _on_progress(frames_examined: int, patches_so_far: Dict[str, int],
                         raw_frames_piped: int, timing: dict = None) -> None:
            # Live UI counters
            self.ui_state['frames_processed_total'] = frames_examined
            # raw_frames_piped = selected_idx from the extractor: the count of
            # BGR frames actually piped from FFmpeg to Python (not a frame index).
            self.ui_state['frames_read_total'] = self._prior_raw_frames + raw_frames_piped
            # Cumulative patch total across all videos
            self.ui_state['patches_created_total'] = (
                prior_total + sum(patches_so_far.values())
            )
            # Live throughput metrics
            elapsed_time = time.monotonic() - video_t0
            if elapsed_time > 0:
                self.ui_state['live_fps'] = raw_frames_piped / elapsed_time
                self.ui_state['live_sps'] = frames_examined / elapsed_time
            # Phase timing snapshot from extractor
            if timing:
                self.ui_state['timing_phases'] = timing
            # Update per-video progress bars with live per-category patch counts
            current_progress = self.ui_state.get('current_video_progress', {})
            for cat, new_total in patches_so_far.items():
                if cat in current_progress:
                    target = current_progress[cat].get('target', 0)
                    pct = (new_total / target * 100) if target > 0 else 0.0
                    current_progress[cat]['created'] = new_total
                    current_progress[cat]['percent'] = pct
            # Increment tracker by delta to avoid double-counting on final merge
            for cat, new_total in patches_so_far.items():
                delta = new_total - last_tracker.get(cat, 0)
                if delta > 0:
                    self.tracker.increment_category_images(cat, delta)
                    last_tracker[cat] = new_total
            # Throttled redraw (respects self.update_interval)
            self._update_terminal_ui()

        streaming_result: Dict[str, int]
        if (
            self._parallel_worker_configs is not None
            and len(assignments) >= len(self._parallel_worker_configs)
        ):
            # ── Parallel within-film extraction ─────────────────────────────
            # Split the assignment list into N temporal segments and run N
            # FFmpeg workers simultaneously.  Each worker decodes only its
            # slice of the film (not the whole thing), which reduces the total
            # frames decoded from |film| to |film|/N per worker.
            streaming_result = self._extract_film_parallel(
                video_path=video_path,
                assignments=assignments,
                n_frames=n_frames,
                fps=fps,
                is_hdr=is_hdr,
                prior_total=prior_total,
                color_trc=color_trc,
            )
        else:
            # ── Single-worker extraction (original path) ─────────────────────
            _fmt_str = self.config.get("output_format", "bmp").lower()
            output_format = OutputFormat.BMP if _fmt_str == "bmp" else OutputFormat.PNG
            streaming_result = extract_and_save_streaming_distributed(
                video_path=video_path,
                assignments=assignments,
                n_frames=n_frames,
                format_config=self.format_config,
                base_dir=self.base_dir,
                fps=fps,
                logger=self.logger,
                is_interesting_fn=self.is_interesting_patch,
                is_black_frame_fn=_streaming_is_black_frame,
                progress_fn=_on_progress,
                use_cuda=self.use_cuda,
                nice_level=self.config.get("processing", {}).get("ffmpeg_nice", 10),
                is_hdr=is_hdr,
                # degrade_cfg intentionally omitted: per-format degradation templates
                # are embedded in format_config and sampled per-patch in the extractor.
                center_snap_seconds=self.config.get("processing", {}).get("center_snap_seconds", 1.0),
                stream_width=STREAM_OPT_WIDTH,
                stream_height=STREAM_OPT_HEIGHT,
                # Only pass cuda_device when CUDA is actually in use.
                cuda_device=(
                    self._available_gpu_indices[0] if self.use_cuda and self._available_gpu_indices else 0
                ),
                color_trc=color_trc,
                # vulkan_device: use the first mapped Vulkan index from the pool.
                # None means FFmpeg picks any available Vulkan device.
                vulkan_device=(
                    self._vulkan_device_pool[0] if self._vulkan_device_pool else None
                ),
                output_format=output_format,
            )

        # Merge final result into patches_created.
        # Tracker already updated: incrementally via _on_progress (single-worker)
        # or once per worker completion (parallel) — do NOT call
        # tracker.increment_category_images again here to avoid double-counting.
        for category, count in streaming_result.items():
            patches_created[category] = patches_created.get(category, 0) + count

        total_created = sum(patches_created.values())
        self.ui_state['patches_created_total'] = prior_total + total_created
        self.ui_state['current_video_name'] = video_name
        self.ui_state['current_video_index'] = video_idx
        # Force a final UI redraw regardless of throttle
        self.last_update_time = 0.0
        self._update_terminal_ui()

        total_time = time.time() - start_time

        self.logger.info(f"\n╔══════════════════════════════════════════════════════════╗")
        self.logger.info(f"║  EXTRACTION COMPLETE (streaming)                         ║")
        self.logger.info(f"╚══════════════════════════════════════════════════════════╝")
        self.logger.info(f"✓ Created {total_created} patches in {total_time:.1f}s")
        self.logger.info(f"\n📊 Per-category breakdown:")
        for category, count in sorted(patches_created.items()):
            self.logger.info(f"  {category}: {count} patches")

        return patches_created

    def _get_video_metadata(self, video_path: str) -> Optional[dict]:
        """
        Get video metadata using ffprobe with caching.
        Cache is based on file size and modification time.

        In addition to duration / fps / resolution the method also extracts
        the ``color_transfer`` tag from the first video stream and derives an
        ``is_hdr`` boolean so callers can choose the appropriate FFmpeg filter
        chain without running a separate ffprobe pass.
        """
        try:
            # Get file stats for cache validation
            file_stat = os.stat(video_path)
            file_size = file_stat.st_size
            file_mtime = file_stat.st_mtime
            
            # Create cache key
            cache_key = video_path
            
            # Check if we have valid cached data
            if cache_key in self.metadata_cache:
                cached = self.metadata_cache[cache_key]
                # Validate cache: same file size and modification time
                if (cached.get('file_size') == file_size and 
                    cached.get('file_mtime') == file_mtime):
                    self.logger.debug(f"Using cached metadata for: {os.path.basename(video_path)}")
                    return {
                        'duration': cached['duration'],
                        'fps': cached.get('fps'),
                        'resolution': cached.get('resolution'),
                        'color_transfer': cached.get('color_transfer'),
                        'is_hdr': cached.get('is_hdr', True),
                    }
            
            # Cache miss or invalid - query ffprobe
            self.logger.debug(f"Scanning video metadata: {os.path.basename(video_path)}")
            
            # FIXED: Added nice priority for lower system impact
            cmd = [
                'nice', '-n', '19',  # Lowest priority
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            timeout = self.config.get("processing", {}).get("ffprobe_timeout", 60)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            if result.returncode != 0:
                return None
            
            data = json.loads(result.stdout)
            duration = float(data.get('format', {}).get('duration', 0))
            
            # Extract additional metadata
            video_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            fps = None
            resolution = None
            color_transfer = None
            if video_stream:
                # Parse FPS
                fps_str = video_stream.get('avg_frame_rate', '0/1')
                if '/' in fps_str:
                    num, den = fps_str.split('/')
                    if int(den) > 0:
                        fps = float(num) / float(den)
                
                # Parse resolution
                width = video_stream.get('width')
                height = video_stream.get('height')
                if width and height:
                    resolution = [width, height]

                # Color transfer (determines HDR vs SDR)
                color_transfer = video_stream.get('color_transfer') or video_stream.get('color_trc')

            is_hdr = is_hdr_transfer(color_transfer)
            self.logger.debug(
                f"{os.path.basename(video_path)}: color_transfer={color_transfer!r} → is_hdr={is_hdr}"
            )

            # Cache the metadata
            self.metadata_cache[cache_key] = {
                'duration': duration,
                'fps': fps,
                'resolution': resolution,
                'color_transfer': color_transfer,
                'is_hdr': is_hdr,
                'file_size': file_size,
                'file_mtime': file_mtime
            }
            
            # Save cache periodically (every 10 videos)
            if len(self.metadata_cache) % 10 == 0:
                self._save_metadata_cache()
            
            return {
                'duration': duration,
                'fps': fps,
                'resolution': resolution,
                'color_transfer': color_transfer,
                'is_hdr': is_hdr,
            }
        
        except Exception as e:
            self.logger.error(f"ffprobe error: {e}")
            return None
    
    # OLD METHOD - DEPRECATED
    # Replaced by _extract_patches_multi_category which extracts once for all categories
    # def _extract_patches_from_video(self, video_path: str, duration: float,
    #                                category: str, format_name: str,
    #                                format_config: dict, n_frames: int) -> int:
    #     """Extract patches from video for a specific category/format"""
    #     # This method has been replaced to avoid multiple video scans
    #     pass
    
    
    def _is_black_frame(self, gt_path: str, threshold_kb: int = 15) -> bool:
        """
        Check if GT file is likely a black/dark frame based on file size.
        
        Args:
            gt_path: Path to GT file
            threshold_kb: File size threshold in KB (default: 15 KB)
        
        Returns:
            True if file is likely a black frame (< threshold), False otherwise
        """
        try:
            if not os.path.exists(gt_path):
                return False
            
            file_size = os.path.getsize(gt_path)
            threshold_bytes = threshold_kb * 1024
            
            if file_size < threshold_bytes:
                self.logger.debug(f"Black frame detected: {gt_path} ({file_size} bytes < {threshold_bytes} bytes)")
                return True
            
            return False
        
        except Exception as e:
            self.logger.error(f"Error checking file size: {e}")
            return False
    
    def is_interesting_patch(self, patch: np.ndarray) -> bool:
        """
        Check if a patch has enough detail/sharpness to be interesting.
        
        Uses Laplacian variance to detect blur/lack of detail.
        Black or very dark frames are always considered interesting to preserve user's requested cuts.
        
        Typical threshold values:
        - < 50: Very permissive, accepts most patches
        - 80 (default): Good balance, filters out very blurry/uniform patches
        - > 150: Strict, only accepts very sharp patches
        
        Args:
            patch: Image patch to check (numpy array)
        
        Returns:
            True if patch is interesting (has detail or is very dark), False otherwise
        """
        try:
            # Check if patch is very dark/black (average brightness < 5)
            # These are always considered "interesting" to preserve black frames/cuts
            avg_brightness = np.mean(patch)
            if avg_brightness < 5:
                return True
            
            # Convert to grayscale if needed
            if len(patch.shape) == 3:
                gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            else:
                gray = patch
            
            # Calculate Laplacian variance (measure of sharpness/detail)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Get threshold from processing settings (default 80.0)
            threshold = self.config.get("processing", {}).get("min_detail_threshold", 80.0)
            
            # Patch is interesting if it has enough detail
            return laplacian_var >= threshold
        
        except Exception as e:
            self.logger.error(f"Error checking patch interestingness: {e}")
            return True  # Default to interesting on error
    
    def _save_patch_pair(self, gt: np.ndarray, lr: np.ndarray,
                        video_path: str, timestamp: float,
                        category: str, format_name: str, n_frames: int) -> tuple:
        """
        Save GT and LR patches to appropriate directories.

        Uses the output format from config (default: BMP).

        Returns:
            Tuple of (success: bool, gt_path: str or None, lr_path: str or None)
        """
        try:
            # Get output directories (returns a dictionary)
            output_dirs = get_output_dirs_for_format(
                self.base_dir, category, format_name, n_frames
            )
            gt_dir = output_dirs['gt']
            lr_dir = output_dirs['lr']

            # Create directories
            os.makedirs(gt_dir, exist_ok=True)
            os.makedirs(lr_dir, exist_ok=True)

            # Resolve output format from config (default: BMP for write throughput)
            _fmt_str = self.config.get("output_format", "bmp").lower()
            _use_bmp = _fmt_str == "bmp"
            _ext = "bmp" if _use_bmp else "png"

            # Generate filename using the configured extension
            video_name = Path(video_path).stem
            patch_name = f"{video_name}_{int(timestamp*1000):08d}.{_ext}"

            gt_path = os.path.join(gt_dir, patch_name)
            lr_path = os.path.join(lr_dir, patch_name)

            if _use_bmp:
                cv2.imwrite(gt_path, gt)
                cv2.imwrite(lr_path, lr)
            else:
                cv2.imwrite(gt_path, gt, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                cv2.imwrite(lr_path, lr, [cv2.IMWRITE_PNG_COMPRESSION, 1])

            return (True, gt_path, lr_path)

        except Exception as e:
            self.logger.error(f"Error saving patches: {e}")
            return (False, None, None)

    def _run_multi_stream(self, start_idx: int, distribution: dict) -> None:
        """
        Multi-stream video extraction: N concurrent stream workers.

        Each worker is bound to a specific GPU (via Vulkan device index for
        libplacebo) and processes videos from a shared queue.  Multiple FFmpeg
        processes run concurrently across different films — this is the
        production extraction loop, not the old sequential per-video loop.

        Concurrency is controlled by ``processing.streams_per_gpu`` in the
        active config (default: 1).  With 2 GPUs and streams_per_gpu=1 this
        gives 2 concurrent FFmpeg processes; streams_per_gpu=2 gives 4.

        Each stream maintains its own runtime state entry in
        ``ui_state["active_streams"]`` so the GUI can display per-GPU status,
        current film, live FPS, write-queue depth, and degradation counts.
        """
        proc = self.config.get("processing", {})
        # Use _vulkan_device_pool length to determine GPU slot count.
        n_vk_slots = len(self._vulkan_device_pool) if self._vulkan_device_pool else 1
        streams_per_gpu = int(proc.get("streams_per_gpu", 1))
        n_streams = max(1, n_vk_slots * streams_per_gpu)
        n_frames = int(proc.get("n_frames", 7))
        nice_level = int(proc.get("ffmpeg_nice", 10))
        center_snap = float(proc.get("center_snap_seconds", 1.0))

        _fmt_str = self.config.get("output_format", "bmp").lower()
        output_format = OutputFormat.BMP if _fmt_str == "bmp" else OutputFormat.PNG

        # Propagate output format to UI state so the GUI can display it.
        self.ui_state["output_format"] = _fmt_str.upper()

        # Build the pending-video queue (skip done / zero-target videos).
        pending: list = []
        for idx in range(start_idx, len(self.videos)):
            video = self.videos[idx]
            video_path = video['path']
            if self.plan.is_video_done(video_path):
                self.tracker.update_progress(current_video_index=idx + 1)
                continue
            video_cat_targets = distribution.get(video_path, {})
            total = sum(video_cat_targets.values()) if video_cat_targets else 0
            if total == 0 or not video.get('categories', {}):
                continue
            pending.append((idx, video, video_cat_targets))

        if not pending:
            self.logger.info("No pending videos to process.")
            return

        video_queue: queue.Queue = queue.Queue()
        for item in pending:
            video_queue.put(item)

        # ── Initialise per-stream state for the GUI ──────────────────────────
        stream_states: list = []
        for sid in range(n_streams):
            # Assign each stream a Vulkan device index from the mapped pool.
            _vk_idx = (
                self._vulkan_device_pool[sid % n_vk_slots]
                if self._vulkan_device_pool else None
            )
            gpu_name = (
                self._available_gpu_names.get(_vk_idx, f"Vulkan {_vk_idx}")
                if _vk_idx is not None else "CPU / software Vulkan"
            )
            stream_states.append({
                "stream_id":        sid,
                "video_name":       "—",
                "gpu_index":        _vk_idx if _vk_idx is not None else -1,
                "gpu_name":         gpu_name,
                "state":            "idle",
                "frames_processed": 0,
                "patches_created":  0,
                "write_queue_depth": 0,
                "live_fps":         0.0,
                "pipeline":         (
                    "libplacebo" if self._vulkan_device_pool and self._vulkan_device_pool[0] is not None
                    else "CPU/Vulkan-SW"
                ),
                "degrade_counts":   {},   # {cat: {template_name: count}}
                "current_video_idx": -1,
                "n_videos_done":    0,
            })

        streams_lock = threading.Lock()
        self.ui_state["active_streams"] = stream_states
        self.ui_state["n_active_streams"] = 0
        self.ui_state["n_gpus_available"] = n_vk_slots

        # ── Stream worker ────────────────────────────────────────────────────
        def stream_worker(stream_id: int) -> None:
            # vulkan_device comes from the pre-mapped Vulkan pool (not CUDA indices).
            _vk = stream_states[stream_id]["gpu_index"]
            vulkan_device = _vk if _vk >= 0 else None

            while self.running:
                try:
                    idx, video, video_cat_targets = video_queue.get(block=True, timeout=0.5)
                except queue.Empty:
                    break

                video_path = video['path']
                video_name = video.get('name', os.path.basename(video_path))
                short_name = video_name[-40:]

                # ── Mark stream running ──────────────────────────────────
                with streams_lock:
                    stream_states[stream_id].update({
                        "video_name":       short_name,
                        "state":            "running",
                        "frames_processed": 0,
                        "patches_created":  0,
                        "live_fps":         0.0,
                        "degrade_counts":   {},
                        "current_video_idx": idx,
                    })
                    self.ui_state["n_active_streams"] = sum(
                        1 for s in stream_states if s["state"] == "running"
                    )
                    # Mark in-progress BEFORE work so a crash causes retry.
                    self.tracker.update_progress(current_video_index=idx)

                try:
                    # ── Get video metadata ───────────────────────────────
                    metadata = self._get_video_metadata(video_path)
                    if not metadata:
                        self.logger.warning(
                            f"Stream {stream_id}: Cannot get metadata for "
                            f"{video_name} — skipping"
                        )
                        video_queue.task_done()
                        with streams_lock:
                            stream_states[stream_id]["state"] = "idle"
                        continue

                    duration = float(metadata.get("duration") or 0.0)
                    fps = float(metadata.get("fps") or 25.0) or 25.0
                    is_hdr = bool(metadata.get("is_hdr", True))
                    color_trc = metadata.get("color_transfer") or "smpte2084"

                    # ── Build format distribution and assignments ────────
                    format_distribution = self._build_format_distribution_for_video(
                        video, video_cat_targets
                    )
                    if not format_distribution:
                        video_queue.task_done()
                        with streams_lock:
                            stream_states[stream_id]["state"] = "idle"
                        continue

                    assignments = build_assignments_per_category(
                        format_distribution=format_distribution,
                        duration=duration,
                        fps=fps,
                        n_frames=n_frames,
                    )
                    if not assignments:
                        video_queue.task_done()
                        with streams_lock:
                            stream_states[stream_id]["state"] = "idle"
                        continue

                    # ── Progress callback ────────────────────────────────
                    prior_total = self.ui_state.get("patches_created_total", 0)
                    _t0 = time.monotonic()

                    def _make_progress_fn(sid=stream_id):
                        def _on_progress(frames_examined, patches_so_far,
                                         raw_frames_piped, timing=None):
                            elapsed = time.monotonic() - _t0
                            live_fps = raw_frames_piped / elapsed if elapsed > 0 else 0.0
                            with streams_lock:
                                ss = stream_states[sid]
                                ss["frames_processed"] = raw_frames_piped
                                ss["patches_created"] = sum(patches_so_far.values())
                                ss["live_fps"] = live_fps
                                if timing:
                                    ss["write_queue_depth"] = timing.get("q_size_last", 0)
                                    dc = timing.get("degrade_counts")
                                    if dc:
                                        # Deep-copy so the dict is not mutated externally.
                                        import copy
                                        ss["degrade_counts"] = copy.deepcopy(dc)
                                # Aggregate cross-stream patch total for global display.
                                agg = prior_total + sum(
                                    s.get("patches_created", 0) for s in stream_states
                                )
                                self.ui_state["patches_created_total"] = agg
                            self._update_terminal_ui()
                        return _on_progress

                    _on_progress = _make_progress_fn()

                    # ── Run extraction via libplacebo ────────────────────
                    # use_cuda=False forces the libplacebo path (Vulkan-based).
                    # vulkan_device selects the specific GPU for this stream.
                    result = extract_and_save_streaming_distributed(
                        video_path=video_path,
                        assignments=assignments,
                        n_frames=n_frames,
                        format_config=self.format_config,
                        base_dir=self.base_dir,
                        fps=fps,
                        logger=self.logger,
                        is_interesting_fn=self.is_interesting_patch,
                        is_black_frame_fn=_streaming_is_black_frame,
                        progress_fn=_on_progress,
                        use_cuda=False,     # libplacebo path
                        cuda_device=0,
                        nice_level=nice_level,
                        is_hdr=is_hdr,
                        center_snap_seconds=center_snap,
                        stream_width=STREAM_OPT_WIDTH,
                        stream_height=STREAM_OPT_HEIGHT,
                        color_trc=color_trc,
                        vulkan_device=vulkan_device,
                        output_format=output_format,
                    )

                    # ── Update progress tracking ─────────────────────────
                    patches_created = sum(
                        v for v in result.values() if isinstance(v, int)
                    )
                    with streams_lock:
                        for cat, count in result.items():
                            if count > 0:
                                self.tracker.increment_category_images(cat, count)
                        if patches_created > 0:
                            self.tracker.update_progress(
                                current_video_index=idx + 1,
                                patches_created=patches_created,
                            )
                            self.plan.mark_video_done(video_path, result)
                        else:
                            self.tracker.update_progress(patches_created=0)
                            self.plan.mark_video_pending(video_path)
                            self.logger.warning(
                                f"⚠️  Stream {stream_id}: {video_name} → 0 patches "
                                f"— will retry on next run"
                            )
                        self.tracker.save()
                        stream_states[stream_id]["n_videos_done"] += 1

                    self.logger.info(
                        f"✅ Stream {stream_id} [GPU {gpu_idx}]: "
                        f"{video_name} → {patches_created} patches"
                    )

                except Exception as exc:
                    self.logger.error(
                        f"❌ Stream {stream_id} [GPU {gpu_idx}]: "
                        f"Error processing {video_name}: {exc}"
                    )
                    import traceback
                    traceback.print_exc()
                    with streams_lock:
                        self.tracker.save()

                finally:
                    with streams_lock:
                        stream_states[stream_id]["state"] = "idle"
                        self.ui_state["n_active_streams"] = sum(
                            1 for s in stream_states if s["state"] == "running"
                        )
                    video_queue.task_done()

        # ── Launch stream workers ────────────────────────────────────────────
        self.logger.info(
            f"🚀 Launching {n_streams} stream worker(s) — "
            f"{len(pending)} videos in queue"
        )
        threads = [
            threading.Thread(target=stream_worker, args=(i,), daemon=True)
            for i in range(n_streams)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Clear stream display on completion.
        self.ui_state["active_streams"] = []
        self.ui_state["n_active_streams"] = 0

    def run(self):
        """Main generation loop with proportional distribution"""
        try:
            # Hide cursor for clean terminal UI — inside try so finally always restores it
            if self.use_terminal_ui:
                hide_cursor()

            # Start background heartbeat so the UI refreshes every second
            # regardless of where the main thread is blocked.
            self._start_ui_heartbeat()

            if RICH_AVAILABLE:
                console.print(Panel.fit(
                    "[bold cyan]Dataset Generator V2 - UHD Quality[/bold cyan]\n"
                    "UHD Preservation • Multi-Category • Priorities • Proportional Distribution",
                    border_style="cyan"
                ))

            # Write architecture file once at the start of each run so the
            # trainer always has an up-to-date description of the dataset layout.
            self._write_architecture_file()
            
            # Phase 1: Scan all videos to get durations
            self.logger.info("Starting Phase 1: Scanning video durations...")
            try:
                durations = self.scan_video_durations()
            except Exception as e:
                self.logger.error(f"FATAL: Error during video duration scanning: {e}")
                self.logger.error(f"This often indicates: out of memory, file access issues, or corrupted videos")
                import traceback
                traceback.print_exc()
                return
            
            if not durations:
                self.logger.error("No video durations found, cannot proceed")
                return
            
            # Phase 2: Calculate proportional distribution
            self.logger.info("Starting Phase 2: Calculating proportional distribution...")
            try:
                distribution = self.calculate_proportional_distribution(durations)
                
                # Count only videos that have at least one category assigned
                videos_with_categories = sum(1 for v in self.videos 
                                            if distribution.get(v['path'], {}))
                self.logger.info(f"Videos with categories: {videos_with_categories} / {len(self.videos)}")
                
                # Store for UI display
                self.total_videos_with_categories = videos_with_categories
                
                # Initialize UI with starting state
                if self.use_terminal_ui:
                    clear_screen()
                    draw_dataset_ui(self.ui_state)
                    time.sleep(1)  # Give user a moment to see initial state
                    
            except Exception as e:
                self.logger.error(f"FATAL: Error during distribution calculation: {e}")
                import traceback
                traceback.print_exc()
                return
            
            # Console output removed - all info shown in terminal GUI
            # No need to print here, user sees progress in the GUI

            # Sort videos so that any video with forced_frames is processed first.
            # Stable sort preserves the relative order within each group.
            forced_count = sum(1 for v in self.videos if v.get('forced_frames'))
            self.videos.sort(key=lambda v: 0 if v.get('forced_frames') else 1)
            if forced_count:
                self.logger.info(
                    f"⚡ Forced-frame videos promoted to front of queue: {forced_count} / {len(self.videos)}"
                )

            # Populate the plan with all videos in the (possibly re-sorted)
            # order.  Videos already tracked — including those marked "done" —
            # are left untouched so that previous progress is preserved.
            self.plan.initialize(self.videos)
            done_count = self.plan.count_done()
            if done_count > 0:
                self.logger.info(
                    f"▶️  Resuming: {done_count}/{self.plan.count_total()} video(s) "
                    f"already done (skipped via plan)"
                )

            # Get resume point (index-based, for a fast forward through the list).
            # When the plan already has done videos, find the index of the first
            # video that has NOT been done yet — this skips the leading done-prefix
            # in O(N) rather than re-checking every video from 0 each restart.
            raw_start_idx = self.tracker.status['progress']['current_video_index']
            if done_count > 0:
                # Locate the first video not yet done in the plan.
                start_idx = 0
                for _i, _v in enumerate(self.videos):
                    if not self.plan.is_video_done(_v['path']):
                        start_idx = _i
                        break
                else:
                    # All videos are done — start past the end to exit immediately.
                    start_idx = len(self.videos)
            else:
                # Index-based resume (no plan progress yet): fast-forward.
                start_idx = raw_start_idx

            if 0 < start_idx < len(self.videos):
                self.logger.info(f"Resuming from video {start_idx + 1}/{len(self.videos)}")

            # --- Multi-stream parallel extraction ---
            # N concurrent stream workers, each assigned to a specific Vulkan device.
            # Multiple FFmpeg processes run simultaneously across different videos.
            self.logger.info("=" * 80)
            n_vk_slots = len(self._vulkan_device_pool) if self._vulkan_device_pool else 1
            streams_per_gpu = self.config.get("processing", {}).get("streams_per_gpu", 1)
            n_streams = max(1, n_vk_slots * streams_per_gpu)
            self.logger.info(
                f"🚀 MULTI-STREAM MODE: {n_streams} concurrent stream(s) "
                f"across {n_vk_slots} Vulkan device slot(s)"
            )
            self.logger.info("=" * 80)

            self._run_multi_stream(start_idx, distribution)

            if RICH_AVAILABLE:
                console.print("\n[bold green]✅ Generation Complete![/bold green]")

            self.logger.info("Generation completed")
            
        except Exception as e:
            self.logger.error(f"FATAL: Unexpected error in run(): {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Stop the background UI heartbeat before touching the terminal
            self._stop_ui_heartbeat()
            # Restore cursor and clean terminal on exit
            if self.use_terminal_ui:
                show_cursor()
                print("\n")  # Clean exit


def main():
    """
    Main entry point.

    Usage::

        python make_dataset_v2_uhd.py [config_dir]

    Arguments
    ---------
    config_dir   (optional) Directory that contains both ``templates.json``
                 and ``generator_config.json``.  Defaults to the directory
                 where this script resides.

    The active config and templates are loaded, validated, and then the
    generator is started.  Run ``video_manager.py`` first to create or edit
    the config files.

    To run the decode pipeline benchmark manually::

        from make_dataset_v2_uhd import DatasetGeneratorV2UHD
        g = DatasetGeneratorV2UHD(config_dir='.')
        g.run_benchmark_tool(force=True)
    """
    import argparse

    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    parser = argparse.ArgumentParser(
        prog="make_dataset_v2_uhd.py",
        description="Dataset Generator V2 – UHD Quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config_dir",
        nargs="?",
        default=str(script_dir),
        help=(
            "Directory containing templates.json and generator_config.json "
            "(default: same directory as this script)"
        ),
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help=(
            "[DEPRECATED] The decode pipeline benchmark no longer runs automatically "
            "at startup.  Use generator.run_benchmark_tool() from Python instead."
        ),
    )
    args = parser.parse_args()

    if args.benchmark:
        print(
            "\n⚠️  --benchmark is deprecated.\n"
            "   The benchmark no longer runs at startup.  To re-measure decode\n"
            "   throughput, call generator.run_benchmark_tool() from Python:\n\n"
            "       from make_dataset_v2_uhd import DatasetGeneratorV2UHD\n"
            "       g = DatasetGeneratorV2UHD(config_dir='.')\n"
            "       g.run_benchmark_tool(force=True)\n"
        )
        sys.exit(0)

    config_dir = args.config_dir

    active_cfg = Path(config_dir) / _ACTIVE_CONFIG_FILENAME
    if not active_cfg.exists():
        print(
            f"❌ Active config not found: {active_cfg}\n"
            "   Please create it first with video_manager.py:\n"
            "       python video_manager.py\n"
            "   Then edit the generated generator_config.json."
        )
        sys.exit(1)

    print(f"📂 Config directory : {config_dir}")
    print(f"   templates        : {Path(config_dir) / _TEMPLATES_FILENAME}")
    print(f"   active cfg       : {active_cfg.name}")

    try:
        generator = DatasetGeneratorV2UHD(config_dir=config_dir)
        generator.run()
    except KeyboardInterrupt:
        show_cursor()
        print("\n⚠️  Interrupted by user")
        print("Progress saved. Run again to resume.")
        sys.exit(0)
    except Exception as e:
        show_cursor()
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
