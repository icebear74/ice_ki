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
    create_patch_pair,
    is_black_frame as _streaming_is_black_frame,
    is_hdr_transfer,
    build_vf_filter,
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
                        ``generator_config.json``.  Defaults to the
                        directory that contains this script.
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

        # ── CUDA ──────────────────────────────────────────────────────────────
        from streaming_extractor import cuda_available
        self.use_cuda = cuda_available()
        if self.use_cuda:
            self.logger.info("🚀 CUDA/GPU mode enabled (hardware-accelerated decoding & scaling)")
        else:
            self.logger.info("🖥️  CPU-only mode enabled (CUDA not available in this FFmpeg build)")

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

        # Statistics
        self.start_time = time.time()
        self.extractions_count = 0
        self.success_count = 0
        self.current_video_name = ""

        # ── Terminal UI state ─────────────────────────────────────────────────
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
            "categories": list(self.category_targets.keys()),
            "format_sizes": list(next(iter(self.format_config.values()), {}).keys()),
        }
        self.ui_update_counter = 0

        if RICH_AVAILABLE:
            self._show_priority_distribution()

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

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
    
    def _update_terminal_ui(self):
        """Update and redraw the terminal UI (throttled to update_interval)."""
        if not self.use_terminal_ui:
            return

        now = time.time()
        if now - self.last_update_time < self.update_interval:
            return
        self.last_update_time = now
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
        Extract frames to DISK (memory-efficient).

        Applies the HDR→SDR tonemap chain when the source is HDR, or a
        lightweight scale-only chain when it is SDR.  When *is_hdr* is
        ``None`` (default) the color format is determined automatically by
        calling ``_get_video_metadata``.

        MEMORY OPTIMIZATION: Returns PATHS to frames (NOT loaded into memory).
        Caller must load frames when needed and clean up temp_dir when done.
        
        Args:
            video_path: Path to video
            start_time: Start timestamp
            n_frames: Number of frames (7 or 5)
            is_hdr: Override for HDR detection.  Pass ``True``/``False`` to
                    skip the metadata look-up (useful when the caller already
                    has the value from ``_get_video_metadata``).
        
        Returns:
            Dict with 'frame_paths' (list of paths) and 'temp_dir' (must be cleaned up)
            or None on failure
        """
        temp_dir = None
        try:
            # Determine HDR/SDR if not supplied by caller
            if is_hdr is None:
                meta = self._get_video_metadata(video_path)
                is_hdr = meta.get('is_hdr', True) if meta else True

            # Use configured temp directory
            temp_dir = self._create_temp_dir("extract_single")
            output_pattern = os.path.join(temp_dir, "frame_%04d.png")

            # Build the correct filter chain for this video's color format.
            # extract_frames_uhd uses CPU-only (no CUDA) for stability and
            # because it processes only a few frames at a time.
            # Replace bgr24 with yuv420p so ffmpeg can write PNG files.
            # (Both _TONEMAP_FILTER and _SDR_FILTER already contain the scale
            # step, so only the final pixel-format needs to change.)
            vf_filter = build_vf_filter(is_hdr=is_hdr, use_cuda=False)
            vf_filter = vf_filter.replace("format=bgr24", "format=yuv420p")
            
            # CPU-only mode (no CUDA) - more stable and reliable
            # FIXED: Added nice priority for lower system impact
            cmd = [
                'nice', '-n', '19',  # Lowest priority
                'ffmpeg',
                '-threads', str(self.workers),  # 6 threads for faster extraction
                '-ss', str(start_time),
                '-i', video_path,
                '-vf', vf_filter,
                '-frames:v', str(n_frames),
                '-y',
                output_pattern
            ]
            
            # LOG THE FFMPEG COMMAND (for debugging)
            self.logger.debug(f"Single frame extraction command: {' '.join(cmd)}")
            
            timeout = self.config.get("processing", {}).get("ffmpeg_timeout", 120)
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout
            )
            
            if result.returncode != 0:
                # Clean up on failure
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                return None
            
            # Build list of frame paths (do NOT load into memory!)
            frame_paths = []
            for i in range(1, n_frames + 1):
                frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                if not os.path.exists(frame_path):
                    # Clean up on failure
                    if temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    return None
                frame_paths.append(frame_path)
            
            # Return paths (NOT frames!) - memory efficient!
            return {
                'frame_paths': frame_paths,
                'temp_dir': temp_dir  # Caller MUST clean up!
            }
        
        except Exception as e:
            self.logger.error(f"Error extracting frames: {e}")
            # Clean up on exception
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

        self.logger.info(
            f"  Color format: {metadata.get('color_transfer', 'unknown')!r} "
            f"→ {'HDR tonemap' if is_hdr else 'SDR pass-through'}"
        )

        patches_created = self._extract_patches_multi_format_batch(
            video_path, duration, format_distribution, n_frames, video_name, fps, video_idx,
            is_hdr=is_hdr,
        )
        
        return patches_created

    def _extract_patches_multi_format_batch(self, video_path: str, duration: float,
                                           format_distribution: Dict[str, Dict[str, int]],
                                           n_frames: int, video_name: str, fps: float = 25.0,
                                           video_idx: int = 0, is_hdr: bool = True) -> Dict[str, int]:
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

        def _on_progress(frames_examined: int, patches_so_far: Dict[str, int], raw_frames_read: int) -> None:
            # Live UI counters
            self.ui_state['frames_processed_total'] = frames_examined
            self.ui_state['frames_read_total'] = self._prior_raw_frames + raw_frames_read
            # Cumulative patch total across all videos
            self.ui_state['patches_created_total'] = (
                prior_total + sum(patches_so_far.values())
            )
            # Live throughput metrics
            elapsed_time = time.monotonic() - video_t0
            if elapsed_time > 0:
                self.ui_state['live_fps'] = raw_frames_read / elapsed_time
                self.ui_state['live_sps'] = frames_examined / elapsed_time
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
            stream_width=STREAM_4K_WIDTH,
            stream_height=STREAM_4K_HEIGHT,
        )

        # Merge final result into patches_created.
        # Tracker already updated incrementally in _on_progress – do NOT call
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
        
        Returns:
            Tuple of (success: bool, gt_path: str or None, lr_path: str or None)
        """
        try:
            lr_version = f"{n_frames}frames"
            
            # Get output directories (returns a dictionary)
            output_dirs = get_output_dirs_for_format(
                self.base_dir, category, format_name, n_frames
            )
            gt_dir = output_dirs['gt']
            lr_dir = output_dirs['lr']
            
            # Create directories
            os.makedirs(gt_dir, exist_ok=True)
            os.makedirs(lr_dir, exist_ok=True)
            
            # Generate filename
            video_name = Path(video_path).stem
            patch_name = f"{video_name}_{int(timestamp*1000):08d}.png"
            
            # Save
            gt_path = os.path.join(gt_dir, patch_name)
            lr_path = os.path.join(lr_dir, patch_name)
            
            cv2.imwrite(gt_path, gt, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            cv2.imwrite(lr_path, lr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            
            return (True, gt_path, lr_path)
        
        except Exception as e:
            self.logger.error(f"Error saving patches: {e}")
            return (False, None, None)
    
    def run(self):
        """Main generation loop with proportional distribution"""
        try:
            # Hide cursor for clean terminal UI — inside try so finally always restores it
            if self.use_terminal_ui:
                hide_cursor()

            if RICH_AVAILABLE:
                console.print(Panel.fit(
                    "[bold cyan]Dataset Generator V2 - UHD Quality[/bold cyan]\n"
                    "UHD Preservation • Multi-Category • Priorities • Proportional Distribution",
                    border_style="cyan"
                ))
            
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
            
            # Sequential processing: One video at a time, fully complete before moving to next
            self.logger.info("\n" + "=" * 80)
            self.logger.info("SEQUENTIAL MODE: Processing one video completely before moving to next")
            self.logger.info("=" * 80)
            
            # Process videos sequentially
            for idx in range(start_idx, len(self.videos)):
                if not self.running:
                    break
                
                video = self.videos[idx]
                video_path = video['path']
                video_name = video.get('name', os.path.basename(video_path))

                # Skip videos already marked done in the generation plan.
                # This check uses the video's *path* as the identifier which
                # makes it robust against reordering, additions, or removals
                # in the video list between runs.
                if self.plan.is_video_done(video_path):
                    self.logger.info(
                        f"\n⏭️  Skipping video {idx + 1}/{len(self.videos)}: "
                        f"{video_name} (already done)"
                    )
                    # Keep the index-based tracker in sync so a fresh restart
                    # can still fast-forward through the done prefix.
                    self.tracker.update_progress(current_video_index=idx + 1)
                    continue

                video_cat_targets = distribution.get(video_path, {})
                
                # Calculate total patches for this video (sum across all categories)
                total_patches = sum(video_cat_targets.values()) if video_cat_targets else 0
                
                # Skip if no patches allocated
                if total_patches == 0:
                    self.logger.info(f"\n⏭️  Skipping video {idx + 1}/{len(self.videos)}: {video_name} (no patches allocated)")
                    continue
                
                # Also skip if video has no categories
                if not video.get('categories', {}):
                    self.logger.info(f"\n⏭️  Skipping video {idx + 1}/{len(self.videos)}: {video_name} (no categories assigned)")
                    continue
                
                # Log start
                self.logger.info(f"\n{'='*80}")
                self.logger.info(f"📹 Processing video {idx + 1}/{len(self.videos)}: {video_name}")
                self.logger.info(f"   Target: {total_patches} patches across {len(video_cat_targets)} categories")
                if video_cat_targets:
                    cat_summary = ", ".join([f"{cat}: {cnt}" for cat, cnt in video_cat_targets.items()])
                    self.logger.info(f"   Per-category: {cat_summary}")
                self.logger.info(f"{'='*80}")
                
                # Set target for this video
                self._current_video_target = total_patches
                
                # Set current video info in UI state BEFORE processing starts
                self.ui_state['current_video_name'] = video_name
                self.ui_state['current_video_index'] = idx + 1  # 1-based for display
                self.ui_state['total_videos'] = self.total_videos_with_categories  # Only count videos with categories!
                
                # Initialize current video progress with targets (0 created so far)
                self.ui_state['current_video_progress'] = {}
                for category in video_cat_targets.keys():
                    self.ui_state['current_video_progress'][category] = {
                        'created': 0,  # Fixed: was 'current', display expects 'created'
                        'target': video_cat_targets[category],
                        'percent': 0.0  # Added: display expects this
                    }
                
                # Update UI to show video info before processing starts
                if self.use_terminal_ui:
                    print(f"\n{'='*80}")
                    print(f"🎬 STARTING VIDEO: {video_name} ({idx+1}/{len(self.videos)})")
                    print(f"   Category targets: {video_cat_targets}")
                    print(f"{'='*80}\n")
                    self._update_terminal_ui()
                
                # Mark this video as "in progress" BEFORE we start work so
                # that a crash or pipeline failure causes a retry on the next
                # run rather than silently skipping it (the old code wrote
                # idx+1 AFTER completion, meaning a video that produced 0
                # patches was treated as done and never retried).
                self.tracker.update_progress(current_video_index=idx)
                self.tracker.save()

                try:
                    # Process this video completely (extraction + processing)
                    stats = self.process_video(idx, video_cat_targets)
                    
                    # Check if video was skipped
                    if stats.get('skipped'):
                        self.logger.info(f"⏭️  Skipped: {video_name} - {stats.get('reason', 'unknown')}")
                        # Advance past this video only after a deliberate skip
                        self.tracker.update_progress(
                            current_video_index=idx + 1,
                            patches_created=0
                        )
                        # A skipped video won't be retried — treat as done in plan
                        self.plan.mark_video_done(video_path, {})
                    else:
                        # process_video() returns {category: count, …}.
                        # Sum all integer values to get the total patch count.
                        patches_created = sum(
                            v for v in stats.values() if isinstance(v, int)
                        )
                        if patches_created > 0:
                            self.tracker.update_progress(
                                current_video_index=idx + 1,
                                patches_created=patches_created
                            )
                            # Mark as done in the plan so a restart skips it
                            self.plan.mark_video_done(video_path, stats)
                        else:
                            self.tracker.update_progress(patches_created=0)
                            # Leave as pending in the plan — will be retried
                            self.plan.mark_video_pending(video_path)
                            self.logger.warning(
                                f"⚠️  {video_name}: 0 patches created — "
                                f"video will be retried on next run"
                            )
                        self.logger.info(f"✅ Complete: {video_name} - {patches_created} patches created")
                    
                    # Log category progress after each video
                    progress_info = self.tracker.get_all_category_progress()
                    self.logger.info(f"\n{progress_info}\n")
                    
                    # Save progress after each video
                    self.tracker.save()
                    
                except Exception as e:
                    self.logger.error(f"❌ Error processing {video_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Save progress even on error
                    self.tracker.save()
            
            if RICH_AVAILABLE:
                console.print("\n[bold green]✅ Generation Complete![/bold green]")
            
            self.logger.info("Generation completed")
            
        except Exception as e:
            self.logger.error(f"FATAL: Unexpected error in run(): {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Restore cursor and clean terminal on exit
            if self.use_terminal_ui:
                show_cursor()
                print("\n")  # Clean exit


def main():
    """
    Main entry point.

    Usage:
        python make_dataset_v2_uhd.py [config_dir]

    *config_dir* (optional) – directory that contains both
    ``templates.json`` and ``generator_config.json``.
    Defaults to the directory where this script resides.

    The active config and templates are loaded, validated, and then the
    generator is started.  Run ``video_manager.py`` first to create or edit
    the config files.
    """
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Optional: allow passing a config directory as the first argument.
    if len(sys.argv) > 1:
        config_dir = sys.argv[1]
    else:
        config_dir = str(script_dir)

    active_cfg = Path(config_dir) / _ACTIVE_CONFIG_FILENAME
    if not active_cfg.exists():
        print(
            f"❌ Active config not found: {active_cfg}\n"
            "   Please create it first with video_manager.py:\n"
            "       python video_manager.py\n"
            "   Then edit the generated generator_config.json."
        )
        sys.exit(1)

    print(f"📂 Config directory: {config_dir}")
    print(f"   templates  : {Path(config_dir) / _TEMPLATES_FILENAME}")
    print(f"   active cfg : {active_cfg.name}")

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
