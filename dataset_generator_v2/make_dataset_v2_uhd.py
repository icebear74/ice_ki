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
    
    def process_video(self, video_idx: int) -> Dict[str, int]:
        """
        Process a single video
        
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
        
        stats = {}
        
        # Get video metadata
        metadata = self._get_video_metadata(video_path)
        if not metadata:
            return {}
        
        duration = metadata['duration']
        
        # Process for each category distribution
        categories = video.get('categories', {})
        for category, weight in categories.items():
            if category not in self.format_config:
                continue
            
            # Get format for this category
            format_name = select_random_format(category)
            format_config = self.format_config[category][format_name]
            
            # Determine frame count
            lr_versions = self.settings.get('lr_versions', ['7frames'])
            n_frames = 7 if '7frames' in lr_versions else 5
            
            # Extract and create patches
            patches_created = self._extract_patches_from_video(
                video_path, duration, category, format_name, 
                format_config, n_frames
            )
            
            if category not in stats:
                stats[category] = 0
            stats[category] += patches_created
        
        return stats
    
    def _get_video_metadata(self, video_path: str) -> Optional[dict]:
        """Get video metadata using ffprobe"""
        try:
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
            
            return {'duration': duration}
        
        except Exception as e:
            self.logger.error(f"ffprobe error: {e}")
            return None
    
    def _extract_patches_from_video(self, video_path: str, duration: float,
                                   category: str, format_name: str,
                                   format_config: dict, n_frames: int) -> int:
        """Extract patches from video for a specific category/format"""
        patches_created = 0
        stride_seconds = 3.0  # Default stride
        current_time = 0.0
        
        while current_time < duration - 1.0:
            # Extract frames
            frames = self.extract_frames_uhd(video_path, current_time, n_frames)
            
            if frames is None:
                current_time += stride_seconds
                continue
            
            # Create patch pair
            gt, lr = self.create_patch_pair(frames, format_name, format_config)
            
            if gt is None or lr is None:
                current_time += stride_seconds
                continue
            
            # Save patches
            saved = self._save_patch_pair(
                gt, lr, video_path, current_time,
                category, format_name, n_frames
            )
            
            if saved:
                patches_created += 1
            
            current_time += stride_seconds
            
            # Check if should stop
            if not self.running:
                break
        
        return patches_created
    
    def _save_patch_pair(self, gt: np.ndarray, lr: np.ndarray,
                        video_path: str, timestamp: float,
                        category: str, format_name: str, n_frames: int) -> bool:
        """Save GT and LR patches to appropriate directories"""
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
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error saving patches: {e}")
            return False
    
    def run(self):
        """Main generation loop"""
        if RICH_AVAILABLE:
            console.print(Panel.fit(
                "[bold cyan]Dataset Generator V2 - UHD Quality[/bold cyan]\n"
                "UHD Preservation • Multi-Category • Priorities • GUI",
                border_style="cyan"
            ))
        
        # Get resume point
        start_idx = self.tracker.status['progress']['current_video_index']
        
        if start_idx > 0:
            self.logger.info(f"Resuming from video {start_idx + 1}/{len(self.videos)}")
        
        # Process videos
        for idx in range(start_idx, len(self.videos)):
            if not self.running:
                break
            
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
