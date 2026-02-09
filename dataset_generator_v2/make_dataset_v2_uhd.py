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
import time
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    
    def __init__(self, config_path: str = "generator_config.json"):
        """Initialize generator with full config support"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.settings = self.config['base_settings']
        self.videos = self.config.get('videos', [])
        self.format_config = self.config.get('format_config', {})
        self.category_targets = self.config.get('category_targets', {})
        
        # Initialize paths (MUST be before logger setup!)
        self.base_dir = self.settings['output_base_dir']
        self.temp_dir = self.settings['temp_dir']
        self.status_file = self.settings['status_file']
        
        # Initialize logger
        self.logger = self._setup_logger()
        sys.logger = self.logger
        
        # Sort videos by priority (0 first, 255 last)
        random.seed(42)  # Reproducible
        for i, video in enumerate(self.videos):
            video['_sort_random'] = random.random()
        self.videos.sort(key=lambda v: (v.get('priority', 255), v['_sort_random']))
        for video in self.videos:
            video.pop('_sort_random', None)
        
        self.logger.info(f"Loaded {len(self.videos)} videos from config")
        
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
        self.workers = self.settings.get('max_workers', 4)
        self.running = True
        self.paused = False
        self.last_update_time = time.time()
        self.update_interval = 0.5
        
        # Statistics
        self.start_time = time.time()
        self.extractions_count = 0
        self.success_count = 0
        self.current_video_name = ""
        
        # Display priority distribution
        if RICH_AVAILABLE:
            self._show_priority_distribution()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logger(self):
        """Setup file and console logger"""
        log_dir = os.path.join(self.base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        logger = logging.getLogger('DatasetGenerator')
        logger.setLevel(logging.DEBUG)
        
        # File handler
        log_file = os.path.join(log_dir, f"generator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
        
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
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
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
    
    def scan_video_durations(self) -> Dict[str, float]:
        """
        Scan all videos to get their durations.
        This is Phase 1 - required for proportional distribution.
        
        Returns:
            Dictionary mapping video_path -> duration in seconds
        """
        if RICH_AVAILABLE:
            console.print("\n[bold cyan]📹 Phase 1: Scanning Video Durations[/bold cyan]")
            console.print("Analyzing all videos to calculate proportional distribution...")
        
        durations = {}
        total_duration = 0.0
        
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
        
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
                
                if not os.path.exists(video_path):
                    self.logger.warning(f"Video not found: {video_path}")
                    progress.update(task, advance=1)
                    continue
                
                # Get video metadata
                metadata = self._get_video_metadata(video_path)
                if metadata and 'duration' in metadata:
                    duration = metadata['duration']
                    durations[video_path] = duration
                    total_duration += duration
                    
                    progress.update(task, description=f"Scanned: {video['name'][:40]}...", advance=1)
                else:
                    self.logger.warning(f"Could not get duration for: {video_path}")
                    progress.update(task, advance=1)
        
        if RICH_AVAILABLE:
            console.print(f"\n✓ Scanned {len(durations)} videos")
            console.print(f"✓ Total duration: {total_duration/3600:.1f} hours ({total_duration:.0f} seconds)")
        
        # Save metadata cache after scanning
        self._save_metadata_cache()
        
        self.logger.info(f"Scanned {len(durations)} videos, total duration: {total_duration:.1f}s")
        
        return durations
    
    def calculate_proportional_distribution(self, durations: Dict[str, float]) -> Dict[str, int]:
        """
        Calculate how many patches each video should get based on its duration.
        This is Phase 2 - distribute proportionally.
        
        Args:
            durations: Dictionary of video_path -> duration in seconds
        
        Returns:
            Dictionary of video_path -> number of patches to create
        """
        total_duration = sum(durations.values())
        total_target_patches = sum(self.category_targets.values())
        
        if total_duration == 0:
            self.logger.warning("Total duration is 0, using equal distribution")
            return {path: total_target_patches // len(durations) for path in durations.keys()}
        
        distribution = {}
        
        if RICH_AVAILABLE:
            console.print(f"\n[bold cyan]📊 Phase 2: Calculating Proportional Distribution[/bold cyan]")
            console.print(f"Target patches: {total_target_patches:,}")
            console.print(f"Total duration: {total_duration/3600:.1f} hours\n")
        
        for video_path, duration in durations.items():
            # Calculate proportional share
            proportion = duration / total_duration
            patches_for_video = int(total_target_patches * proportion)
            distribution[video_path] = patches_for_video
            
            # Find video name for display
            video_name = "Unknown"
            for v in self.videos:
                if v['path'] == video_path:
                    video_name = v['name']
                    break
            
            self.logger.debug(f"  {video_name}: {duration:.0f}s ({proportion*100:.1f}%) → {patches_for_video} patches")
        
        # Show summary
        if RICH_AVAILABLE:
            from rich.table import Table
            
            table = Table(title="Distribution Summary (Top 10 videos)")
            table.add_column("Video", style="cyan")
            table.add_column("Duration", justify="right", style="yellow")
            table.add_column("Patches", justify="right", style="green")
            table.add_column("%", justify="right", style="magenta")
            
            # Sort by patches descending
            sorted_dist = sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:10]
            
            for video_path, patches in sorted_dist:
                video_name = "Unknown"
                for v in self.videos:
                    if v['path'] == video_path:
                        video_name = v['name']
                        break
                
                duration = durations.get(video_path, 0)
                proportion = duration / total_duration * 100
                
                table.add_row(
                    video_name[:40],
                    f"{duration/60:.1f} min",
                    f"{patches:,}",
                    f"{proportion:.1f}%"
                )
            
            console.print(table)
        
        return distribution
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def extract_frames_uhd(self, video_path: str, start_time: float, n_frames: int = 7) -> Optional[List[np.ndarray]]:
        """
        Extract frames with HDR→SDR tonemap, NO resize (UHD quality)
        
        Args:
            video_path: Path to video
            start_time: Start timestamp
            n_frames: Number of frames (7 or 5)
        
        Returns:
            List of UHD frames or None
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_pattern = os.path.join(temp_dir, "frame_%04d.png")
                
                # UHD tonemap filter (NO scale!)
                vf_filter = (
                    "zscale=t=linear:npl=100,"
                    "format=gbrpf32le,"
                    "zscale=p=bt709,"
                    "tonemap=tonemap=mobius:desat=0,"
                    "zscale=t=bt709:m=bt709:range=limited,"
                    "format=yuv420p"
                )
                
                cmd = [
                    'ffmpeg',
                    '-ss', str(start_time),
                    '-i', video_path,
                    '-vf', vf_filter,
                    '-frames:v', str(n_frames),
                    '-y',
                    output_pattern
                ]
                
                timeout = self.config.get('ffmpeg_timeout', 120)
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=timeout
                )
                
                if result.returncode != 0:
                    return None
                
                # Load frames
                frames = []
                for i in range(1, n_frames + 1):
                    frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                    if not os.path.exists(frame_path):
                        return None
                    
                    frame = cv2.imread(frame_path)
                    if frame is None:
                        return None
                    
                    frames.append(frame)
                
                return frames
        
        except Exception as e:
            self.logger.error(f"Error extracting frames: {e}")
            return None
    
    def create_patch_pair(self, frames: List[np.ndarray], format_name: str, 
                         format_config: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Create GT + LR pair with random crop from UHD
        
        Args:
            frames: UHD frames
            format_name: Format key (small_540, medium_169, large_720)
            format_config: Format configuration
        
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
        
        # Random crop
        max_x = frame_w - gt_w
        max_y = frame_h - gt_h
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
        """
        distribution = {}
        
        # Get category weights for this video
        video_categories = video.get('categories', {})
        
        for category, category_weight in video_categories.items():
            if category not in self.format_config:
                continue
            
            # Calculate patches for this category
            category_patches = int(target_patches * category_weight)
            
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
    
    def process_video(self, video_idx: int) -> Dict[str, int]:
        """
        Process a single video and extract patches for ALL formats.
        
        NEW BEHAVIOR (per user requirement):
        - Each video extracts ALL formats (not randomly selected)
        - Format distribution is pre-calculated per video
        - Ensures every video has all formats in all categories
        
        Returns:
            Statistics dict with patches created per category
        """
        if video_idx >= len(self.videos):
            return {}
        
        video = self.videos[video_idx]
        video_path = video['path']
        video_name = video['name']
        self.current_video_name = video_name
        
        if not os.path.exists(video_path):
            self.logger.warning(f"Video not found: {video_path}")
            return {}
        
        self.logger.info(f"Processing video {video_idx + 1}/{len(self.videos)}: {video_name}")
        
        # Get video metadata
        metadata = self._get_video_metadata(video_path)
        if not metadata:
            return {}
        
        duration = metadata['duration']
        
        # Calculate target patches for this video (from proportional distribution)
        # This is passed via an instance variable from the run() method
        target_patches = getattr(self, '_current_video_target', 1000)
        
        # Calculate format distribution for this video
        format_distribution = self.calculate_format_distribution_for_video(video, target_patches)
        
        if not format_distribution:
            self.logger.warning(f"No valid format distribution for video: {video_name}")
            return {}
        
        # Log the distribution plan
        self.logger.info(f"Format distribution for {video_name} (target: {target_patches} total):")
        for category, formats in format_distribution.items():
            total = sum(formats.values())
            self.logger.info(f"  {category} ({total} patches): {formats}")
        
        # Determine frame count
        lr_versions = self.settings.get('lr_versions', ['7frames'])
        n_frames = 7 if '7frames' in lr_versions else 5
        
        # Extract patches for ALL categories and formats
        patches_created = self._extract_patches_multi_format(
            video_path, duration, format_distribution, n_frames, video_name
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
    
    def _extract_patches_multi_format(self, video_path: str, duration: float,
                                      format_distribution: Dict[str, Dict[str, int]], 
                                      n_frames: int, video_name: str) -> Dict[str, int]:
        """
        Extract patches from video for MULTIPLE categories and MULTIPLE formats.
        
        This implements the NEW requirement: each video extracts ALL formats,
        not randomly selected. Distribution is pre-calculated per video.
        
        Also implements black frame detection with retry logic:
        - If GT < 15 KB (likely black frame): delete, retry with +1 second
        - Maximum 5 retries per extraction
        - Failed frames still count in statistics
        
        Args:
            video_path: Path to video file
            duration: Video duration in seconds
            format_distribution: Dict of {category: {format_name: target_count}}
            n_frames: Number of frames to extract (5 or 7)
            video_name: Video name for logging
        
        Returns:
            Dict of {category: patches_created_count}
        """
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
            
            cmd = [
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
        if RICH_AVAILABLE:
            console.print(Panel.fit(
                "[bold cyan]Dataset Generator V2 - UHD Quality[/bold cyan]\n"
                "UHD Preservation • Multi-Category • Priorities • Proportional Distribution",
                border_style="cyan"
            ))
        
        # Phase 1: Scan all videos to get durations
        durations = self.scan_video_durations()
        
        if not durations:
            self.logger.error("No video durations found, cannot proceed")
            return
        
        # Phase 2: Calculate proportional distribution
        distribution = self.calculate_proportional_distribution(durations)
        
        if RICH_AVAILABLE:
            console.print(f"\n[bold green]✓ Distribution calculated[/bold green]")
            console.print(f"[bold cyan]📹 Phase 3: Generating Patches[/bold cyan]\n")
        
        # Get resume point
        start_idx = self.tracker.status['progress']['current_video_index']
        
        if start_idx > 0:
            self.logger.info(f"Resuming from video {start_idx + 1}/{len(self.videos)}")
        
        # Process videos with proportional targets
        for idx in range(start_idx, len(self.videos)):
            if not self.running:
                break
            
            video = self.videos[idx]
            video_path = video['path']
            
            # Get target patches for this video from distribution
            target_patches = distribution.get(video_path, 0)
            
            if target_patches == 0:
                self.logger.warning(f"No patches allocated for {video['name']}, skipping")
                continue
            
            self.logger.info(f"Processing {video['name']}: target={target_patches} patches")
            
            # Set target for this video (used in process_video method)
            self._current_video_target = target_patches
            
            stats = self.process_video(idx)
            
            # Update tracker
            self.tracker.update_progress(
                current_video_index=idx + 1,
                completed_videos=idx + 1
            )
            
            for category, count in stats.items():
                current = self.tracker.status['category_stats'].get(category, {}).get('images_created', 0)
                self.tracker.update_category_stats(category, images_created=current + count)
            
            self.tracker.save()
        
        if RICH_AVAILABLE:
            console.print("\n[bold green]✅ Generation Complete![/bold green]")
        
        self.logger.info("Generation completed")


def main():
    """Main entry point"""
    config_path = "generator_config.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # Change to script directory
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    
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
