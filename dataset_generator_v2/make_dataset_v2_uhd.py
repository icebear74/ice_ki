#!/usr/bin/env python3
"""
Dataset Generator V2 - UHD Quality with Original Features
Combines:
- UHD quality preservation from new implementation
- GUI, priorities, multi-category support from original
- Complete video list and configurations
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
import psutil  # For memory monitoring
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add utils to path
sys.path.insert(0, os.path.dirname(__file__))

from utils.format_definitions import (
    FORMATS, CATEGORY_FORMAT_DISTRIBUTION, CATEGORY_PATHS,
    select_random_format, get_output_dirs_for_format
)
from utils.progress_tracker import ProgressTracker
from utils.dataset_display import draw_dataset_ui
from utils.terminal_ui import hide_cursor, show_cursor, clear_screen
from utils.config_normalizer import normalize_config
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
# Don't configure basic logging here - will be done in _setup_logger based on UI mode
logger = logging.getLogger(__name__)


class DatasetGeneratorV2UHD:
    """
    Enhanced Dataset Generator V2
    - UHD quality preservation (tonemap only, NO resize)
    - Multi-category support (master, universal, space, toon)
    - Priority-based video processing
    - Rich GUI with progress tracking
    - Complete video list from original config
    """
    
    MAX_DISPLAYED_PRIORITIES = 10
    
    def __init__(self, config_path: str = "generator_config_v2.json"):
        """Initialize generator with full config support"""
        # Load and normalize V2 config
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.config = normalize_config(self.config)

        self.settings = self.config['base_settings']
        self.videos = self.config.get('videos', [])
        self.format_config = self.config.get('format_config', {})
        self.category_targets = self.config.get('category_targets', {})
        
        # Initialize paths (MUST be before logger setup!)
        self.base_dir = self.settings['output_base_dir']
        self.temp_dir = self.settings['temp_dir']
        self.status_file = self.settings['status_file']
        
        # Terminal UI setting (MUST be before logger setup!)
        self.use_terminal_ui = True  # Enable terminal GUI by default
        
        # Initialize logger
        self.logger = self._setup_logger()
        sys.logger = self.logger
        
        # CUDA/GPU disabled - using CPU-only mode for better stability
        # CPU extraction is reliable and seeking is the bottleneck anyway
        self.use_cuda = False
        self.logger.info("🖥️  CPU-only mode enabled (CUDA/GPU disabled for stability)")
        
        # Videos are already sorted in JSON by multi-category priority
        # Process them in exact JSON order (no additional sorting)
        # This ensures videos with multiple categories are processed first
        
        self.logger.info(f"Loaded {len(self.videos)} videos from config (processing in JSON order)")
        
        # Extract format probabilities from format_config
        self.format_probabilities = self._extract_format_probabilities()
        
        # Initialize video metadata cache
        self.metadata_cache_file = os.path.join(self.base_dir, '.video_metadata_cache.json')
        self.metadata_cache = self._load_metadata_cache()
        
        # Initialize progress tracker
        self.tracker = ProgressTracker(self.status_file)
        self.tracker.update_progress(total_videos=len(self.videos))
        self.tracker.initialize_categories(self.category_targets)
        
        # Runtime state
        self.workers = self.config.get('workers', 6)
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
        
        # Terminal UI state
        self.ui_state = {
            'current_video_name': "",
            'current_video_index': 0,
            'total_videos': len(self.videos),
            'current_video_progress': {},
            'overall_progress': {},
            'patch_distribution': {},
            'scenes_processed': 0,
            'patches_created_total': 0,
            'avg_time_per_scene': 0.0,
            'eta': {},
            # Only categories that actually exist in the config
            'categories': list(self.category_targets.keys()),
            # Only format-size columns that actually exist in the config
            'format_sizes': list(next(iter(self.format_config.values()), {}).keys()),
        }
        # Terminal UI already set before logger init (line 89)
        self.ui_update_counter = 0
        
        # Display priority distribution
        if RICH_AVAILABLE:
            self._show_priority_distribution()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logger(self):
        """Setup file and console logger (console disabled when terminal UI active)"""
        log_dir = os.path.join(self.base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        logger = logging.getLogger('DatasetGenerator')
        logger.setLevel(logging.DEBUG)
        logger.handlers = []  # Clear any existing handlers
        
        # File handler (always enabled)
        log_file = os.path.join(log_dir, f"generator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
        
        # Console handler (only if terminal UI is disabled)
        if not self.use_terminal_ui:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(ch)
            logger.info("Console logging enabled (terminal UI disabled)")
        else:
            logger.info("Console logging disabled (terminal UI active - see GUI)")
        
        return logger
    
    def _show_priority_distribution(self):
        """Display priority distribution in console"""
        priority_counts = {}
        for v in self.videos:
            p = v.get('priority', 255)
            priority_counts[p] = priority_counts.get(p, 0) + 1
        
        console.print("\n[bold]📋 Video Processing Order:[/bold]")
        sorted_priorities = sorted(priority_counts.keys())
        
        # Show first priorities and default (255)
        priorities_to_show = []
        if 255 in priority_counts:
            priorities_to_show = [p for p in sorted_priorities if p != 255][:self.MAX_DISPLAYED_PRIORITIES - 1]
            priorities_to_show.append(255)
            priorities_to_show.sort()
        else:
            priorities_to_show = sorted_priorities[:self.MAX_DISPLAYED_PRIORITIES]
        
        for priority in priorities_to_show:
            count = priority_counts[priority]
            label = "(default)" if priority == 255 else ""
            console.print(f"   Priority {priority} {label}: {count} videos")
        
        remaining = [p for p in sorted_priorities if p not in priorities_to_show]
        if remaining:
            count = sum(priority_counts[p] for p in remaining)
            console.print(f"   ... and {count} more videos in other priority levels")
    
    def _load_metadata_cache(self) -> dict:
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
        # Show cursor before exit
        if self.use_terminal_ui:
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
        """Update and redraw the terminal UI"""
        if not self.use_terminal_ui:
            return
        
        # Update every call for real-time progress during extraction
        self.ui_update_counter += 1
        # Log update to verify it's being called (visible on screen for debugging)
        print(f"[GUI UPDATE #{self.ui_update_counter}] Patches: {self.ui_state.get('patches_created_total', 0)}", flush=True)
        self.logger.info(f"GUI update #{self.ui_update_counter} - patches: {self.ui_state.get('patches_created_total', 0)}")
        
        try:
            # Update overall progress from tracker
            category_stats = self.tracker.status.get('category_stats', {})
            for category in self.category_targets.keys():
                if category in category_stats:
                    stats = category_stats[category]
                    # Use actual distribution totals instead of raw category targets
                    target = self.distribution_totals.get(category, 0)
                    current = stats.get('images_created', 0)  # ProgressTracker uses 'images_created'
                    percent = (current / target * 100) if target > 0 else 0.0
                    self.ui_state['overall_progress'][category] = {
                        'created': current,  # Fixed: was 'current', display expects 'created'
                        'target': target,
                        'percent': percent  # Added: display expects this
                    }
            
            # Calculate patch distribution by category and size
            # Use ACTUAL format names from configuration (e.g., small_540, medium_720_169, large_720)
            patch_dist = {}
            for category in self.format_config.keys():
                patch_dist[category] = {}
                # Get actual format names from configuration for this category
                if category in self.format_config:
                    for format_name in self.format_config[category].keys():
                        # Get from tracker if available
                        if category in category_stats:
                            # Get distribution based on actual format probabilities
                            total = category_stats[category].get('images_created', 0)  # ProgressTracker uses 'images_created'
                            prob = self.format_probabilities.get(category, {}).get(format_name, 0.0)
                            current = int(total * prob)
                            # Use actual distribution totals
                            target_total = self.distribution_totals.get(category, 0)
                            target = int(target_total * prob)
                            
                            patch_dist[category][format_name] = {
                                'count': current,
                                'target': target
                            }
                        else:
                            patch_dist[category][format_name] = {'count': 0, 'target': 0}
            
            self.ui_state['patch_distribution'] = patch_dist
            
            # Calculate ETAs
            elapsed = time.time() - self.start_time
            if self.ui_state['patches_created_total'] > 0 and elapsed > 0:
                rate = self.ui_state['patches_created_total'] / elapsed
                
                eta_by_category = {}
                max_eta = 0
                for category in self.ui_state['overall_progress'].keys():
                    if category in self.ui_state['overall_progress']:
                        current = self.ui_state['overall_progress'][category]['created']  # Fixed: was 'current'
                        target = self.ui_state['overall_progress'][category]['target']
                        remaining = target - current
                        if rate > 0 and remaining > 0:
                            eta_seconds = remaining / rate
                            eta_by_category[category] = eta_seconds
                            max_eta = max(max_eta, eta_seconds)
                
                self.ui_state['eta'] = eta_by_category
                self.ui_state['eta']['total'] = max_eta
            
            # Draw the UI
            print(f"[DRAWING GUI...]", flush=True)
            clear_screen()
            draw_dataset_ui(self.ui_state)
            # Force flush to ensure display updates immediately
            sys.stdout.flush()
            # Extra newline to ensure terminal processes the output
            print("", end='', flush=True)
            print(f"[GUI DRAWN]", flush=True)
            
        except Exception as e:
            # Don't let UI errors crash the program, but DO show them on screen!
            print(f"\n⚠️  GUI UPDATE ERROR: {e}")
            print(f"   (Check log for stack trace)")
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
    
    def _extract_format_probabilities(self) -> Dict[str, Dict[str, float]]:
        """
        Extract format probabilities from format_config.
        
        Returns:
            Dictionary mapping category -> {format_name: probability}
            
        Example:
            {
                'master': {'small_540': 0.5, 'medium_169': 0.35, 'large_720': 0.15},
                'universal': {'small_540': 0.5, 'medium_169': 0.35, 'large_720': 0.15}
            }
        """
        probabilities = {}
        
        for category, formats in self.format_config.items():
            probabilities[category] = {}
            for format_name, format_info in formats.items():
                probabilities[category][format_name] = format_info.get('probability', 0.0)
        
        self.logger.debug(f"Extracted format probabilities: {probabilities}")
        return probabilities
    
    # CUDA support removed - using CPU-only mode for better stability
    
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
                            
                            progress.update(task, description=f"Scanned: {video_name[:40]}...", advance=1)
                            # Log with newline for clean output
                            print(f"Scanned: {video_name}: {duration:.1f}s")
                            self.logger.debug(f"Scanned {video_name}: {duration:.1f}s")
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
                
                # Find videos that have THIS category
                category_videos = []
                category_total_duration = 0.0
                
                for v in self.videos:
                    video_cats = get_video_categories(v)
                    if category in video_cats:
                        video_path = v['path']
                        if video_path in durations:
                            duration = durations[video_path]
                            category_videos.append((video_path, v['name'], duration))
                            category_total_duration += duration
                
                self.logger.info(f"    {category}: {len(category_videos)} videos, {category_total_duration/3600:.1f} hours total")
                
                if category_total_duration == 0 or len(category_videos) == 0:
                    self.logger.warning(f"    No videos or zero duration for {category}, skipping")
                    continue
                
                # Distribute category target among these videos proportionally
                for video_path, video_name, duration in category_videos:
                    proportion = duration / category_total_duration
                    patches = int(category_target * proportion)
                    video_targets[video_path][category] = patches
                    
                    self.logger.debug(f"      {video_name}: {duration:.0f}s ({proportion*100:.1f}%) → {patches} patches")
            
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
    
    def extract_frames_uhd(self, video_path: str, start_time: float, n_frames: int = 7) -> Optional[Dict]:
        """
        Extract frames with HDR→SDR tonemap to DISK (memory-efficient).
        
        MEMORY OPTIMIZATION: Returns PATHS to frames (NOT loaded into memory).
        Caller must load frames when needed and clean up temp_dir when done.
        
        Args:
            video_path: Path to video
            start_time: Start timestamp
            n_frames: Number of frames (7 or 5)
        
        Returns:
            Dict with 'frame_paths' (list of paths) and 'temp_dir' (must be cleaned up)
            or None on failure
        """
        temp_dir = None
        try:
            # Use configured temp directory
            temp_dir = self._create_temp_dir("extract_single")
            output_pattern = os.path.join(temp_dir, "frame_%04d.png")
            
            # UHD tonemap filter with 1080p scaling
            vf_filter = (
                "zscale=t=linear:npl=100,"
                "format=gbrpf32le,"
                "zscale=p=bt709,"
                "tonemap=tonemap=mobius:desat=0,"
                "zscale=t=bt709:m=bt709:range=limited,"
                "scale=1920:1080:flags=lanczos,"
                "format=yuv420p"
            )
            
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
            
            timeout = self.config.get('ffmpeg_timeout', 120)
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
            'frame_paths': frame_paths_dict,
            'temp_dirs': temp_dirs  # Caller MUST clean up!
        }
    
    
    def create_patch_pair(self, frames: List[np.ndarray], format_name: str, 
                         format_config: dict, force_center: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Create GT + LR pair with random crop from UHD
        
        Args:
            frames: UHD frames
            format_name: Format key (small_540, medium_169, large_720)
            format_config: Format configuration
            force_center: If True, use center crop instead of random crop
        
        Returns:
            (gt, lr_stacked) or (None, None)
        """
        n_frames = len(frames)
        if n_frames not in [5, 7]:
            return None, None
        
        gt_h, gt_w = format_config['gt_size']
        lr_h, lr_w = format_config['lr_size']
        
        # Get frame dimensions (UHD!)
        frame_h, frame_w = frames[0].shape[:2]
        
        if frame_h < gt_h or frame_w < gt_w:
            return None, None
        
        # Calculate crop position
        max_x = frame_w - gt_w
        max_y = frame_h - gt_h
        
        if force_center:
            # Center crop: exact center of frame
            crop_x = max_x // 2
            crop_y = max_y // 2
        else:
            # Random crop
            crop_x = random.randint(0, max_x)
            crop_y = random.randint(0, max_y)
        
        # GT: Center frame
        center_idx = n_frames // 2
        center_frame = frames[center_idx]
        gt = center_frame[crop_y:crop_y+gt_h, crop_x:crop_x+gt_w]
        
        # LR: All frames with DVD-realistic downscale
        lr_frames = []
        for frame in frames:
            crop = frame[crop_y:crop_y+gt_h, crop_x:crop_x+gt_w]
            lr = cv2.resize(crop, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
            lr_frames.append(lr)
        
        # Stack vertically (übereinander) - axis=0 stacks frames underneath each other
        lr_stacked = np.concatenate(lr_frames, axis=0)
        
        return gt, lr_stacked
    
    def calculate_format_distribution_for_video(self, video: dict, target_patches: int) -> Dict[str, Dict[str, int]]:
        """
        Calculate exact format distribution for a video.
        
        Each video extracts ALL formats according to pre-calculated distribution.
        
        Example:
        - Video needs 4000 patches total
        - Categories: master 50%, universal 50%
        - Formats: large 50%, small 25%, medium 25%
        
        Result:
        {
            'master': {'large_720': 1000, 'small_540': 500, 'medium_169': 500},
            'universal': {'large_720': 1000, 'small_540': 500, 'medium_169': 500}
        }
        
        Args:
            video: Video configuration dict
            target_patches: Total patches for this video
        
        Returns:
            Dictionary of {category: {format_name: count}}
        
        Calculate format distribution for this video across ALL its categories.
        
        NEW LOGIC (NO WEIGHTS):
        - Video is 100% in each assigned category
        - Distribute patches evenly across assigned categories
        - Each category gets: target_patches / num_categories
        
        Args:
            video: Video dict with categories
            target_patches: Total patches to extract from this video
        
        Returns:
            Dict[category][format] = patch_count
        """
        distribution = {}
        
        # Get categories for this video (handle both dict and list formats)
        video_cats = get_video_categories(video)
        
        if not video_cats:
            return distribution
        
        # Calculate patches per category (equal distribution)
        num_categories = len(video_cats)
        patches_per_category = target_patches // num_categories
        remainder = target_patches % num_categories
        
        for cat_idx, category in enumerate(video_cats):
            if category not in self.format_config:
                continue
            
            # This category gets equal share (+ 1 if remainder)
            category_patches = patches_per_category
            if cat_idx < remainder:
                category_patches += 1
            
            # Get format probabilities for this category
            format_probs = self.format_probabilities.get(category, {})
            
            # Calculate patches per format
            distribution[category] = {}
            remaining_patches = category_patches
            
            # Sort by probability (descending) for better rounding
            sorted_formats = sorted(format_probs.items(), key=lambda x: x[1], reverse=True)
            
            for idx, (format_name, prob) in enumerate(sorted_formats):
                if idx == len(sorted_formats) - 1:
                    # Last format gets remaining patches
                    distribution[category][format_name] = remaining_patches
                else:
                    count = int(category_patches * prob)
                    distribution[category][format_name] = count
                    remaining_patches -= count
        
        return distribution
    
    def process_video(self, video_idx: int, category_targets: Dict[str, int] = None) -> Dict[str, int]:
        """
        Process a single video and extract patches for ALL formats.
        
        NEW BEHAVIOR (per-category distribution):
        - Each category has its own target patch count
        - Format distribution calculated per category
        - Ensures category targets are met
        
        Args:
            video_idx: Index of video in self.videos
            category_targets: Dict of {category: patch_count} for this video
        
        Returns:
            Statistics dict with patches created per category
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
        
        # Get video metadata
        metadata = self._get_video_metadata(video_path)
        if not metadata:
            return {}
        
        duration = metadata['duration']
        
        # Use category targets if provided, otherwise use old method
        if category_targets:
            # NEW: Per-category targets
            format_distribution = {}
            
            for category, patches in category_targets.items():
                if category not in self.format_config:
                    continue
                
                # Get format probabilities for this category
                format_probs = self.format_probabilities.get(category, {})
                
                # Calculate patches per format
                format_distribution[category] = {}
                remaining_patches = patches
                
                # Sort by probability (descending) for better rounding
                sorted_formats = sorted(format_probs.items(), key=lambda x: x[1], reverse=True)
                
                for idx, (format_name, prob) in enumerate(sorted_formats):
                    if idx == len(sorted_formats) - 1:
                        # Last format gets remaining patches
                        format_distribution[category][format_name] = remaining_patches
                    else:
                        count = int(patches * prob)
                        format_distribution[category][format_name] = count
                        remaining_patches -= count
        else:
            # OLD: Total target (backward compatibility)
            target_patches = getattr(self, '_current_video_target', 1000)
            format_distribution = self.calculate_format_distribution_for_video(video, target_patches)
        
        if not format_distribution:
            self.logger.warning(f"No valid format distribution for video: {video_name}")
            return {}
        
        # Calculate total target for logging
        if category_targets:
            target_total = sum(category_targets.values())
        else:
            target_total = target_patches
        
        # Log the distribution plan
        self.logger.info(f"Format distribution for {video_name} (target: {target_total} total):")
        for category, formats in format_distribution.items():
            total = sum(formats.values())
            self.logger.info(f"  {category} ({total} patches): {formats}")
        
        # Determine frame count
        lr_versions = self.settings.get('lr_versions', ['7frames'])
        n_frames = 7 if '7frames' in lr_versions else 5
        
        # Get video FPS for batch extraction
        metadata = self._get_video_metadata(video_path)
        fps = metadata.get('fps', 25.0) if metadata else 25.0
        
        # Extract patches using OPTIMIZED batch mode
        patches_created = self._extract_patches_multi_format_batch(
            video_path, duration, format_distribution, n_frames, video_name, fps, video_idx
        )
        
        return patches_created
    
    def _extract_patches_multi_category(self, video_path: str, duration: float,
                                       category_configs: dict, n_frames: int) -> Dict[str, int]:
        """
        Extract patches from video for MULTIPLE categories simultaneously.
        This avoids opening the video file multiple times.
        
        NOTE: This method is legacy and replaced by _extract_patches_multi_format.
        Kept for backward compatibility.
        
        Args:
            video_path: Path to video file
            duration: Video duration in seconds
            category_configs: Dict of {category: {'weight', 'format_name', 'format_config'}}
            n_frames: Number of frames to extract (5 or 7)
        
        Returns:
            Dict of {category: patches_created_count}
        """
        patches_created = {cat: 0 for cat in category_configs.keys()}
        stride_seconds = 3.0  # Default stride
        current_time = 0.0
        
        self.logger.info(f"Extracting for {len(category_configs)} categories: {list(category_configs.keys())}")
        
        while current_time < duration - 1.0:
            # Extract frames ONCE
            frames = self.extract_frames_uhd(video_path, current_time, n_frames)
            
            if frames is None:
                current_time += stride_seconds
                continue
            
            # Create and save patches for ALL categories from the same frames
            for category, config in category_configs.items():
                format_name = config['format_name']
                format_config = config['format_config']
                
                # Create patch pair for this category/format
                gt, lr = self.create_patch_pair(frames, format_name, format_config)
                
                if gt is None or lr is None:
                    continue
                
                # Save patches for this category
                saved, gt_path, lr_path = self._save_patch_pair(
                    gt, lr, video_path, current_time,
                    category, format_name, n_frames
                )
                
                if saved:
                    patches_created[category] += 1
            
            current_time += stride_seconds
            
            # Check if should stop
            if not self.running:
                break
        
        return patches_created
    
    def _extract_patches_multi_format_batch(self, video_path: str, duration: float,
                                           format_distribution: Dict[str, Dict[str, int]], 
                                           n_frames: int, video_name: str, fps: float = 25.0,
                                           video_idx: int = 0) -> Dict[str, int]:
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
        
        # Phase 1: Calculate all extraction timestamps
        self.logger.info(f"\n📋 Phase 1: Calculating extraction plan...")
        
        # Each scene is assigned to EXACTLY ONE (category, format) slot → no duplicates.
        # Total scenes needed = total patches across all slots.
        scenes_needed = total_target
        
        self.logger.info(f"✓ Format distribution analysis:")
        self.logger.info(f"  Total target patches: {total_target}")
        self.logger.info(f"  Scenes needed: {scenes_needed} (each scene used exactly once)")
        
        # Calculate stride to EVENLY DISTRIBUTE across ENTIRE video duration
        # This ensures frames from beginning, middle, AND END of video
        usable_duration = duration - 1.0  # Leave 1 second at end
        
        if scenes_needed > 0 and usable_duration > 0:
            # Calculate stride: divide total duration by number of SCENES needed
            stride_seconds = usable_duration / scenes_needed
            # Minimum stride to avoid extracting too frequently
            stride_seconds = max(stride_seconds, 0.5)
        else:
            stride_seconds = 3.0  # Fallback
        
        self.logger.info(f"\n  Video duration: {duration:.1f}s")
        self.logger.info(f"  Video FPS: {fps:.2f}")
        self.logger.info(f"  Total frames in video: {int(duration * fps)}")
        self.logger.info(f"  Calculated stride: {stride_seconds:.2f}s = {int(stride_seconds * fps)} frames")
        
        # Generate timestamps evenly across entire video
        timestamps = []
        for i in range(scenes_needed):
            timestamp = i * stride_seconds
            if timestamp < usable_duration:
                timestamps.append(timestamp)
            else:
                break
        
        self.logger.info(f"\n✓ Planned {len(timestamps)} extraction points (scenes)")
        self.logger.info(f"  Extraction pattern: One scene every {int(stride_seconds * fps)} frames")
        self.logger.info(f"  Each scene: 7 consecutive frames")
        if timestamps:
            self.logger.info(f"  First timestamp: {timestamps[0]:.2f}s (0.0% of video)")
            self.logger.info(f"  Last timestamp: {timestamps[-1]:.2f}s ({100*timestamps[-1]/duration:.1f}% of video)")
            self.logger.info(f"  Coverage: Entire video from start to end")
        self.logger.info(f"  Total frames to extract: {len(timestamps) * n_frames}")
        self.logger.info(f"  All {len(timestamps)} scenes will be used (0% waste)")
        
        # Phase 2: INCREMENTAL extraction and processing
        # User request: "extrahier 7 frames .. verteile die .. extrahiere die nächsten 7 usw usw"
        # Translation: "extract 7 frames .. distribute them .. extract the next 7, etc."
        self.logger.info(f"\n🎬 Phase 2: INCREMENTAL extraction and processing...")
        self.logger.info(f"  Extract 7 frames → Process → Clean up → Repeat")
        self.logger.info(f"  Memory-efficient: Only 7 frames in RAM at a time")
        self.logger.info(f"  Target directories (master/, space/, etc.) will be created as patches are saved")
        
        black_frame_threshold_kb = 15
        black_frame_detection_limit_seconds = 10.0
        black_frames_detected = 0
        black_frames_skipped = 0
        total_created = 0
        
        # Build a scene→slot partition: each scene goes to EXACTLY ONE (category, format).
        # This prevents any scene from being saved to more than one output directory.
        total_scenes = len(timestamps)

        # Flatten all (category, format_name, count) slots
        all_slots: List[Tuple[str, str, int]] = []
        for category, formats in format_distribution.items():
            for format_name, count in formats.items():
                all_slots.append((category, format_name, count))

        # If the video is too short, scale down slot counts proportionally
        slots_total = sum(c for _, _, c in all_slots)
        if slots_total > total_scenes:
            scale = total_scenes / slots_total
            all_slots = [(cat, fmt, max(1, int(cnt * scale))) for cat, fmt, cnt in all_slots]
            # Trim any excess caused by rounding
            excess = sum(c for _, _, c in all_slots) - total_scenes
            if excess > 0:
                all_slots.sort(key=lambda x: -x[2])
                all_slots[0] = (all_slots[0][0], all_slots[0][1],
                                max(0, all_slots[0][2] - excess))

        # Assign consecutive index ranges to each slot
        scene_to_slot: Dict[int, Tuple[str, str]] = {}
        offset = 0
        for category, format_name, count in all_slots:
            for i in range(count):
                scene_to_slot[offset + i] = (category, format_name)
            offset += count

        self.logger.info(f"\n📊 Scene partition per format (no duplicates):")
        for category, format_name, count in all_slots:
            self.logger.info(f"  {category}/{format_name}: {count} unique scenes")
        
        # INCREMENTAL PROCESSING: Extract → Process → Clean → Repeat
        self.logger.info(f"\n🔄 Starting incremental extraction and processing...")
        self.logger.info(f"  Processing {len(timestamps)} timestamps one at a time")
        
        processed_count = 0
        extraction_failures = 0
        
        for scene_idx, ts in enumerate(timestamps):
            self.logger.info(f"\n📍 Scene {scene_idx+1}/{len(timestamps)}: timestamp {ts:.2f}s")
            scene_start_time = time.time()  # Track scene processing time
            
            # Each scene is assigned to exactly one (category, format) slot
            if scene_idx not in scene_to_slot:
                self.logger.debug(f"  ⏭️  Skipping (no slot assigned)")
                continue
            
            # Check if we should abort
            if not self.running:
                self.logger.info("⚠️  Aborting extraction (Ctrl+C detected)")
                break
            
            # EXTRACT 7 frames for this timestamp
            self.logger.info(f"  🎬 Extracting {n_frames} frames...")
            result = self.extract_frames_uhd(video_path, ts, n_frames)
            
            if not result or not result.get('frame_paths'):
                self.logger.error(f"  ❌ Extraction failed for timestamp {ts:.2f}s")
                extraction_failures += 1
                continue
            
            frame_file_paths = result['frame_paths']
            temp_dir = result['temp_dir']
            self.logger.info(f"  ✓ Extracted {len(frame_file_paths)} frames to temp directory")
            
            # Load frames into memory (only 7 frames - minimal RAM usage)
            frames = []
            for frame_path in frame_file_paths:
                frame = cv2.imread(frame_path)
                if frame is None:
                    self.logger.warning(f"  ⚠️  Could not read frame {frame_path}")
                    break
                frames.append(frame)
            
            if len(frames) != n_frames:
                self.logger.warning(f"  ⚠️  Incomplete frames ({len(frames)}/{n_frames}), skipping")
                # Clean up
                for frame_path in frame_file_paths:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                if temp_dir and os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                continue
            
            self.logger.info(f"  ✓ Loaded {len(frames)} frames into memory")
            
            # Update GUI to show extraction progress
            self.ui_state['scenes_processed'] = processed_count + 1
            self._update_terminal_ui()
            
            # PROCESS frames: this scene goes to exactly ONE (category, format) slot
            patches_created_this_scene = 0
            category, format_name = scene_to_slot[scene_idx]
            format_config = self.format_config[category][format_name]

            # Try up to 5 times to find an interesting patch with random crops
            MAX_RANDOM_ATTEMPTS = 5
            gt, lr = None, None

            for attempt in range(MAX_RANDOM_ATTEMPTS + 1):
                if attempt < MAX_RANDOM_ATTEMPTS:
                    gt, lr = self.create_patch_pair(frames, format_name, format_config, force_center=False)
                else:
                    gt, lr = self.create_patch_pair(frames, format_name, format_config, force_center=True)

                if gt is None or lr is None:
                    continue

                if self.is_interesting_patch(gt) or attempt >= MAX_RANDOM_ATTEMPTS:
                    break

            if gt is not None and lr is not None:
                saved, gt_path, lr_path = self._save_patch_pair(
                    gt, lr, video_path, ts,
                    category, format_name, n_frames
                )

                if saved:
                    # Check for black frames only in first 10 seconds
                    if ts <= black_frame_detection_limit_seconds and \
                       self._is_black_frame(gt_path, black_frame_threshold_kb):
                        black_frames_detected += 1
                        self.logger.debug(f"    Black frame detected, deleting")
                        try:
                            if os.path.exists(gt_path):
                                os.remove(gt_path)
                            if os.path.exists(lr_path):
                                os.remove(lr_path)
                        except Exception as e:
                            self.logger.error(f"    Error deleting black frame files: {e}")
                    else:
                        if ts > black_frame_detection_limit_seconds:
                            black_frames_skipped += 1

                        patches_targets[category][format_name]['created'] += 1
                        patches_created[category] += 1
                        total_created += 1
                        patches_created_this_scene += 1

                        # Update UI state with new patch count
                        self.ui_state['patches_created_total'] = total_created

                        self.logger.info(f"    ✓ Saved patch: {category}/{format_name} → {os.path.basename(gt_path)}")
            
            # CLEAN UP: Delete frame files immediately to free disk space
            for frame_path in frame_file_paths:
                try:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                except Exception as e:
                    self.logger.warning(f"  ⚠️  Could not delete frame file: {e}")
            
            # Clean up temp directory
            if temp_dir and os.path.exists(temp_dir):
                import shutil
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    self.logger.warning(f"  ⚠️  Could not delete temp directory: {e}")
            
            # Free memory
            del frames
            
            processed_count += 1
            
            # Update UI state immediately after scene completion
            self.ui_state['current_video_name'] = video_name
            self.ui_state['current_video_index'] = video_idx
            self.ui_state['scenes_processed'] = processed_count
            self.ui_state['patches_created_total'] = total_created
            
            # Update current video progress
            for category in patches_created.keys():
                category_created = patches_created[category]
                category_target = sum(patches_targets[category][fmt]['target'] for fmt in patches_targets[category])
                category_percent = (category_created / category_target * 100) if category_target > 0 else 0.0
                self.ui_state['current_video_progress'][category] = {
                    'created': category_created,  # Fixed: was 'current', display expects 'created'
                    'target': category_target,
                    'percent': category_percent  # Added: display expects this
                }
            
            # Calculate average time per scene
            elapsed = time.time() - scene_start_time if 'scene_start_time' in locals() else 0
            if processed_count > 0:
                self.ui_state['avg_time_per_scene'] = elapsed / processed_count
            
            # Update tracker with patches created in this scene (for overall progress in GUI)
            for category, count in patches_created.items():
                # Get the number of new patches created for this category in this iteration
                # patches_created is cumulative, so we need to calculate the delta
                previous_count = self.ui_state.get('previous_patches_per_category', {}).get(category, 0)
                new_patches = count - previous_count
                if new_patches > 0:
                    self.tracker.increment_category_images(category, new_patches)
            
            # Store current counts for next iteration
            if 'previous_patches_per_category' not in self.ui_state:
                self.ui_state['previous_patches_per_category'] = {}
            for category, count in patches_created.items():
                self.ui_state['previous_patches_per_category'][category] = count
            
            # Update terminal UI
            self._update_terminal_ui()
            
            # Progress update
            self.logger.info(f"  ✓ Created {patches_created_this_scene} patches from this scene")
            self.logger.info(f"  📊 Progress: {processed_count}/{len(timestamps)} scenes processed, {total_created} total patches created")
            
            if processed_count % 10 == 0:
                # Show category progress every 10 scenes
                self.logger.info(f"\n  📊 Category progress after {processed_count} scenes:")
                for category in sorted(patches_created.keys()):
                    category_created = patches_created[category]
                    category_target = sum(patches_targets[category][fmt]['target'] for fmt in patches_targets[category])
                    pct = 100 * category_created / category_target if category_target > 0 else 0
                    self.logger.info(f"    {category:12s}: {category_created:5d}/{category_target:5d} patches ({pct:5.1f}%)")
        
        # Final summary
        self.logger.info(f"\n{'═'*60}")
        self.logger.info(f"✓ Incremental extraction and processing complete!")
        self.logger.info(f"  Scenes processed: {processed_count}/{len(timestamps)}")
        if extraction_failures > 0:
            self.logger.warning(f"  Extraction failures: {extraction_failures}")
        self.logger.info(f"  Total patches created: {total_created}")
        self.logger.info(f"{'═'*60}")
        
        # Final statistics
        total_time = time.time() - start_time
        
        self.logger.info(f"\n╔══════════════════════════════════════════════════════════╗")
        self.logger.info(f"║  EXTRACTION COMPLETE                                     ║")
        self.logger.info(f"╚══════════════════════════════════════════════════════════╝")
        self.logger.info(f"✓ Processed {processed_count}/{len(timestamps)} scenes in {total_time:.1f}s")
        self.logger.info(f"✓ Created {total_created} patches total")
        
        if black_frames_detected > 0:
            self.logger.info(f"  🚫 Black frames detected and removed: {black_frames_detected}")
        if black_frames_skipped > 0:
            self.logger.info(f"  ⏭️  Frames saved without check (after 10s): {black_frames_skipped}")
        
        self.logger.info(f"\n📊 Per-category breakdown:")
        for category, formats in patches_targets.items():
            cat_total = sum(stats['created'] for stats in formats.values())
            cat_target = sum(stats['target'] for stats in formats.values())
            self.logger.info(f"  {category}: {cat_total}/{cat_target} patches")
            for format_name, stats in formats.items():
                self.logger.info(f"    └─ {format_name}: {stats['created']}/{stats['target']}")
        
        return patches_created
    
    def _extract_patches_multi_format_legacy(self, video_path: str, duration: float,
                                      format_distribution: Dict[str, Dict[str, int]], 
                                      n_frames: int, video_name: str) -> Dict[str, int]:
        """
        LEGACY: Extract patches using individual FFmpeg calls (SLOW).
        
        This is the old method that calls FFmpeg once per extraction.
        Kept as fallback in case batch extraction fails.
        
        Args:
            video_path: Path to video file
            duration: Video duration in seconds
            format_distribution: Dict of {category: {format_name: target_count}}
            n_frames: Number of frames to extract (5 or 7)
            video_name: Video name for logging
        
        Returns:
            Dict of {category: patches_created_count}
        """
        self.logger.warning(f"Using LEGACY extraction mode (slower)")
        
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
        
        # Rest of original implementation
        stride_seconds = 3.0
        current_time = 0.0
        max_retries = 5
        retry_jump_seconds = 1.0
        black_frame_threshold_kb = 15
        black_frame_detection_limit_seconds = 10.0
        
        total_created = 0
        black_frames_detected = 0
        black_frames_skipped = 0
        
        self.logger.info(f"Extracting {total_target} patches for {len(format_distribution)} categories")
        self.logger.info(f"Black frame detection active for first {black_frame_detection_limit_seconds:.1f} seconds only")
        
        # Extract frames and create patches until all targets are met
        while current_time < duration - 1.0 and total_created < total_target:
            # For each category-format combination that needs more patches
            for category, formats in format_distribution.items():
                for format_name, target_count in formats.items():
                    # Check if this format still needs patches
                    if patches_targets[category][format_name]['created'] >= target_count:
                        continue
                    
                    # Get format config
                    format_config = self.format_config[category][format_name]
                    
                    # Try extraction with retry logic for black frames
                    retry_count = 0
                    extraction_successful = False
                    retry_time = current_time
                    
                    while retry_count <= max_retries and not extraction_successful:
                        # Extract frames for this retry attempt
                        frames = self.extract_frames_uhd(video_path, retry_time, n_frames)
                        
                        if frames is None:
                            retry_count += 1
                            retry_time += retry_jump_seconds
                            if retry_time >= duration - 1.0:
                                break
                            continue
                        
                        # Create patch pair for this category/format
                        gt, lr = self.create_patch_pair(frames, format_name, format_config)
                        
                        if gt is None or lr is None:
                            retry_count += 1
                            retry_time += retry_jump_seconds
                            if retry_time >= duration - 1.0:
                                break
                            continue
                        
                        # Save patches for this category/format
                        saved, gt_path, lr_path = self._save_patch_pair(
                            gt, lr, video_path, retry_time,
                            category, format_name, n_frames
                        )
                        
                        if saved:
                            # Check if GT is a black frame (< 15 KB)
                            # Only check during first 10 seconds of video
                            if retry_time <= black_frame_detection_limit_seconds and \
                               self._is_black_frame(gt_path, black_frame_threshold_kb):
                                black_frames_detected += 1
                                self.logger.warning(
                                    f"Black frame detected at {retry_time:.2f}s "
                                    f"(retry {retry_count}/{max_retries}). Deleting and retrying..."
                                )
                                
                                # Delete the files
                                try:
                                    if os.path.exists(gt_path):
                                        os.remove(gt_path)
                                    if os.path.exists(lr_path):
                                        os.remove(lr_path)
                                except Exception as e:
                                    self.logger.error(f"Error deleting black frame files: {e}")
                                
                                # Jump forward 1 second and retry
                                retry_count += 1
                                retry_time += retry_jump_seconds
                                
                                if retry_count > max_retries:
                                    # Max retries reached, count it but don't create patch
                                    self.logger.warning(
                                        f"Max retries ({max_retries}) reached for black frame. "
                                        f"Counting as created but no patch saved."
                                    )
                                    patches_targets[category][format_name]['created'] += 1
                                    patches_created[category] += 1
                                    total_created += 1
                                    extraction_successful = True
                                
                                if retry_time >= duration - 1.0:
                                    # Reached end of video, count it but don't create patch
                                    self.logger.warning(
                                        f"Reached end of video during black frame retry. "
                                        f"Counting as created but no patch saved."
                                    )
                                    patches_targets[category][format_name]['created'] += 1
                                    patches_created[category] += 1
                                    total_created += 1
                                    extraction_successful = True
                            else:
                                # Valid frame (not black), successfully saved
                                # Or black frame detection skipped (after 10 seconds)
                                if retry_time > black_frame_detection_limit_seconds:
                                    black_frames_skipped += 1
                                    
                                patches_targets[category][format_name]['created'] += 1
                                patches_created[category] += 1
                                total_created += 1
                                extraction_successful = True
                        else:
                            # Save failed, retry
                            retry_count += 1
                            retry_time += retry_jump_seconds
                            if retry_time >= duration - 1.0:
                                break
            
            current_time += stride_seconds
            
            # Check if should stop
            if not self.running:
                break
        
        # Log final statistics
        self.logger.info(f"Extraction complete for {video_name}: {total_created}/{total_target} patches")
        if black_frames_detected > 0:
            self.logger.info(f"  Black frames detected and handled: {black_frames_detected}")
        if black_frames_skipped > 0:
            self.logger.info(f"  Frames saved without black frame check (after {black_frame_detection_limit_seconds}s): {black_frames_skipped}")
        for category, formats in patches_targets.items():
            for format_name, stats in formats.items():
                self.logger.info(f"  {category}/{format_name}: {stats['created']}/{stats['target']}")
        
        return patches_created
        # Initialize counters for each category-format combination
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
        
        stride_seconds = 3.0
        current_time = 0.0
        max_retries = 5
        retry_jump_seconds = 1.0
        black_frame_threshold_kb = 15
        black_frame_detection_limit_seconds = 10.0  # Only check first 10 seconds
        
        total_target = sum(sum(formats.values()) for formats in format_distribution.values())
        total_created = 0
        black_frames_detected = 0
        black_frames_skipped = 0  # Count of frames where detection was skipped
        
        self.logger.info(f"Extracting {total_target} patches for {len(format_distribution)} categories")
        self.logger.info(f"Black frame detection active for first {black_frame_detection_limit_seconds:.1f} seconds only")
        
        # Extract frames and create patches until all targets are met
        while current_time < duration - 1.0 and total_created < total_target:
            # For each category-format combination that needs more patches
            for category, formats in format_distribution.items():
                for format_name, target_count in formats.items():
                    # Check if this format still needs patches
                    if patches_targets[category][format_name]['created'] >= target_count:
                        continue
                    
                    # Get format config
                    format_config = self.format_config[category][format_name]
                    
                    # Try extraction with retry logic for black frames
                    retry_count = 0
                    extraction_successful = False
                    retry_time = current_time
                    
                    while retry_count <= max_retries and not extraction_successful:
                        # Extract frames for this retry attempt
                        frames = self.extract_frames_uhd(video_path, retry_time, n_frames)
                        
                        if frames is None:
                            retry_count += 1
                            retry_time += retry_jump_seconds
                            if retry_time >= duration - 1.0:
                                break
                            continue
                        
                        # Create patch pair for this category/format
                        gt, lr = self.create_patch_pair(frames, format_name, format_config)
                        
                        if gt is None or lr is None:
                            retry_count += 1
                            retry_time += retry_jump_seconds
                            if retry_time >= duration - 1.0:
                                break
                            continue
                        
                        # Save patches for this category/format
                        saved, gt_path, lr_path = self._save_patch_pair(
                            gt, lr, video_path, retry_time,
                            category, format_name, n_frames
                        )
                        
                        if saved:
                            # Check if GT is a black frame (< 15 KB)
                            # Only check during first 10 seconds of video
                            if retry_time <= black_frame_detection_limit_seconds and \
                               self._is_black_frame(gt_path, black_frame_threshold_kb):
                                black_frames_detected += 1
                                self.logger.warning(
                                    f"Black frame detected at {retry_time:.2f}s "
                                    f"(retry {retry_count}/{max_retries}). Deleting and retrying..."
                                )
                                
                                # Delete the files
                                try:
                                    if os.path.exists(gt_path):
                                        os.remove(gt_path)
                                    if os.path.exists(lr_path):
                                        os.remove(lr_path)
                                except Exception as e:
                                    self.logger.error(f"Error deleting black frame files: {e}")
                                
                                # Jump forward 1 second and retry
                                retry_count += 1
                                retry_time += retry_jump_seconds
                                
                                if retry_count > max_retries:
                                    # Max retries reached, count it but don't create patch
                                    self.logger.warning(
                                        f"Max retries ({max_retries}) reached for black frame. "
                                        f"Counting as created but no patch saved."
                                    )
                                    patches_targets[category][format_name]['created'] += 1
                                    patches_created[category] += 1
                                    total_created += 1
                                    extraction_successful = True
                                
                                if retry_time >= duration - 1.0:
                                    # Reached end of video, count it but don't create patch
                                    self.logger.warning(
                                        f"Reached end of video during black frame retry. "
                                        f"Counting as created but no patch saved."
                                    )
                                    patches_targets[category][format_name]['created'] += 1
                                    patches_created[category] += 1
                                    total_created += 1
                                    extraction_successful = True
                            else:
                                # Valid frame (not black), successfully saved
                                # Or black frame detection skipped (after 10 seconds)
                                if retry_time > black_frame_detection_limit_seconds:
                                    black_frames_skipped += 1
                                    
                                patches_targets[category][format_name]['created'] += 1
                                patches_created[category] += 1
                                total_created += 1
                                extraction_successful = True
                        else:
                            # Save failed, retry
                            retry_count += 1
                            retry_time += retry_jump_seconds
                            if retry_time >= duration - 1.0:
                                break
            
            current_time += stride_seconds
            
            # Check if should stop
            if not self.running:
                break
        
        # Log final statistics
        self.logger.info(f"Extraction complete for {video_name}: {total_created}/{total_target} patches")
        if black_frames_detected > 0:
            self.logger.info(f"  Black frames detected and handled: {black_frames_detected}")
        if black_frames_skipped > 0:
            self.logger.info(f"  Frames saved without black frame check (after {black_frame_detection_limit_seconds}s): {black_frames_skipped}")
        for category, formats in patches_targets.items():
            for format_name, stats in formats.items():
                created = stats['created']
                target = stats['target']
                self.logger.info(f"  {category}/{format_name}: {created}/{target} patches")
        
        return patches_created
    
    def _get_video_metadata(self, video_path: str) -> Optional[dict]:
        """
        Get video metadata using ffprobe with caching.
        Cache is based on file size and modification time.
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
                        'resolution': cached.get('resolution')
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
            
            timeout = self.config.get('ffprobe_timeout', 60)
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
            
            # Cache the metadata
            self.metadata_cache[cache_key] = {
                'duration': duration,
                'fps': fps,
                'resolution': resolution,
                'file_size': file_size,
                'file_mtime': file_mtime
            }
            
            # Save cache periodically (every 10 videos)
            if len(self.metadata_cache) % 10 == 0:
                self._save_metadata_cache()
            
            return {
                'duration': duration,
                'fps': fps,
                'resolution': resolution
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
            
            # Get threshold from settings (default 80.0)
            threshold = self.settings.get('min_detail_threshold', 80.0)
            
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
        # Hide cursor for clean terminal UI
        if self.use_terminal_ui:
            hide_cursor()
        
        try:
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
            
            # Get resume point
            start_idx = self.tracker.status['progress']['current_video_index']
            
            if start_idx > 0:
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
                
                try:
                    # Process this video completely (extraction + processing)
                    stats = self.process_video(idx, video_cat_targets)
                    
                    # Check if video was skipped
                    if stats.get('skipped'):
                        self.logger.info(f"⏭️  Skipped: {video_name} - {stats.get('reason', 'unknown')}")
                        # Update tracker with skip
                        self.tracker.update_progress(
                            current_video_index=idx + 1,
                            patches_created=0
                        )
                    else:
                        # Update tracker with actual patches
                        patches_created = stats.get('patches_created', 0)
                        self.tracker.update_progress(
                            current_video_index=idx + 1,
                            patches_created=patches_created
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
    """Main entry point"""
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        v2_config = script_dir / 'generator_config_v2.json'
        if v2_config.exists():
            config_path = str(v2_config)
        else:
            print(
                "❌ No config file found. "
                "Please run from dataset_generator_v2 directory "
                "(expected generator_config_v2.json)."
            )
            sys.exit(1)

    print(f"📂 Using config: {Path(config_path).name}")

    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    try:
        generator = DatasetGeneratorV2UHD(config_path)
        generator.run()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        print("Progress saved. Run again to resume.")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
