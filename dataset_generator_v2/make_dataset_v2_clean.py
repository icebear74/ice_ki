#!/usr/bin/env python3
"""
Dataset Generator V2 - Complete Rewrite
✅ UHD Quality Preservation (tonemap only, NO resize)
✅ 7-Frame Only (5-frame code removed)
✅ New Flat Directory Structure
✅ Complete State Management & Resume
✅ Category-Based Weighted Distribution
✅ Bug Fixes
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
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Import state manager
from state_manager import StateManager

console = Console()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetGeneratorV2:
    """
    Dataset Generator V2 - Complete Rewrite
    
    Features:
    - UHD quality preservation (tonemap only, NO resize to HD)
    - Random cropping from full UHD resolution
    - DVD-realistic LR downscaling (INTER_AREA)
    - 7-frame only support
    - New directory structure (patches/720/, etc.)
    - Complete state management
    - Resume capability
    """
    
    def __init__(self, config_path: str = "generator_config_v2.json"):
        """Initialize generator"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.root_path = Path(self.config['root_path']).expanduser()
        self.dataset_name = self.config['dataset_name']
        self.dataset_path = self.root_path / self.dataset_name
        
        # Set random seed if configured (for reproducible patch generation)
        random_seed = self.config.get('random_seed')
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            logger.info(f"🎲 Random seed set to {random_seed} (reproducible mode)")
        
        # Initialize state manager
        state_file = self.dataset_path / "generation_state.json"
        self.state_manager = StateManager(self.config, str(state_file))
        
        # Setup directories
        self._setup_directories()
        
        # Stats
        self.patches_since_save = 0
        self.save_interval = 100  # Auto-save every 100 patches
        
        logger.info(f"✅ Generator initialized: {self.dataset_path}")
    
    def _setup_directories(self):
        """Create new flat directory structure"""
        for size_key, size_config in self.config['output_patches'].items():
            if not size_config.get('enabled', True):
                continue
            
            # Training patches
            (self.dataset_path / "patches" / size_key / "GT").mkdir(parents=True, exist_ok=True)
            (self.dataset_path / "patches" / size_key / "LR").mkdir(parents=True, exist_ok=True)
            
            # Validation directories (user populates GT manually)
            (self.dataset_path / "val" / size_key / "GT").mkdir(parents=True, exist_ok=True)
            (self.dataset_path / "val" / size_key / "LR").mkdir(parents=True, exist_ok=True)
        
        console.print(f"[green]✅ Directory structure created: {self.dataset_path}[/green]")
    
    def extract_frames_uhd(self, video_path: str, start_time: float, n_frames: int = 7) -> Optional[List[np.ndarray]]:
        """
        Extract frames with HDR→SDR tonemap, NO resize
        Keeps full UHD resolution (e.g., 3840×2160)
        
        Args:
            video_path: Path to video file
            start_time: Start timestamp in seconds
            n_frames: Number of frames to extract (default: 7)
        
        Returns:
            List of frames in UHD resolution, or None on error
        """
        try:
            # Create temporary directory for frames
            with tempfile.TemporaryDirectory() as temp_dir:
                output_pattern = os.path.join(temp_dir, "frame_%04d.png")
                
                # FFmpeg command: HDR→SDR tonemap ONLY, NO scale!
                # This preserves full UHD resolution
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
                
                # Run ffmpeg with configurable timeout (default 120s for UHD)
                timeout = self.config.get('ffmpeg_timeout', 120)
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=timeout
                )
                
                if result.returncode != 0:
                    logger.error(f"FFmpeg failed for {video_path} at {start_time}s")
                    return None
                
                # Load frames
                frames = []
                for i in range(1, n_frames + 1):
                    frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                    if not os.path.exists(frame_path):
                        logger.warning(f"Missing frame {i} for {video_path}")
                        return None
                    
                    frame = cv2.imread(frame_path)
                    if frame is None:
                        logger.warning(f"Failed to read frame {i} for {video_path}")
                        return None
                    
                    frames.append(frame)
                
                return frames
        
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            return None
    
    def create_patch_pair(self, frames: List[np.ndarray], size_key: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Create GT + LR pair with random crop from UHD frames
        
        Args:
            frames: 7 UHD frames (e.g., 3840×2160 each)
            size_key: Size configuration key ('720', '540', '720_169')
        
        Returns:
            (gt, lr_stacked) or (None, None) on error
        """
        if len(frames) != 7:
            return None, None
        
        size_config = self.config['output_patches'][size_key]
        gt_h, gt_w = size_config['gt_size']
        lr_h, lr_w = size_config['lr_size']
        
        # Get frame dimensions (full UHD!)
        frame_h, frame_w = frames[0].shape[:2]
        
        # Check if frame is large enough
        if frame_h < gt_h or frame_w < gt_w:
            logger.warning(f"Frame too small: {frame_w}×{frame_h}, need {gt_w}×{gt_h}")
            return None, None
        
        # RANDOM crop position (can be edges, corners, anywhere!)
        max_x = frame_w - gt_w
        max_y = frame_h - gt_h
        crop_x = random.randint(0, max_x)
        crop_y = random.randint(0, max_y)
        
        # GT: Center frame (index 3) crop from FULL UHD quality
        center_frame = frames[3]
        gt = center_frame[crop_y:crop_y+gt_h, crop_x:crop_x+gt_w]
        
        # LR: All 7 frames, same crop, DVD-realistic downscale
        lr_frames = []
        for frame in frames:
            # Crop same region
            crop = frame[crop_y:crop_y+gt_h, crop_x:crop_x+gt_w]
            
            # INTER_AREA = DVD-realistic quality (sweet spot)
            # Not too good (LANCZOS/CUBIC), not too bad (LINEAR/NEAREST)
            lr = cv2.resize(crop, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
            lr_frames.append(lr)
        
        # Stack vertically (übereinander): frames underneath each other
        # For 7 frames of 240×240, this creates 1680×240
        lr_stacked = np.concatenate(lr_frames, axis=0)
        
        return gt, lr_stacked
    
    def is_scene_change(self, frames: List[np.ndarray]) -> bool:
        """
        Detect scene change in frame sequence
        Compares first and last frame histograms
        """
        if len(frames) < 2:
            return False
        
        threshold = self.config['processing']['scene_threshold']
        
        # Compare first and last frame
        gray1 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
        
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        diff_percent = diff * 100
        
        return diff_percent > threshold
    
    def is_blurry(self, frame: np.ndarray) -> bool:
        """Detect blurry frames using Laplacian variance"""
        threshold = self.config['quality']['blur_threshold']
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < threshold
    
    def process_video(self, video_path: str, video_info: dict, target_patches: int):
        """
        Process a single video to generate patches
        
        Args:
            video_path: Path to video file
            video_info: Video metadata and progress info
            target_patches: Number of patches to generate from this video
        """
        logger.info(f"📹 Processing: {Path(video_path).name}")
        logger.info(f"   Target: {target_patches} patches")
        logger.info(f"   Resume from: {video_info['last_timestamp']:.2f}s")
        
        # Get video metadata
        metadata = self.state_manager.state['video_metadata'].get(video_path)
        if not metadata:
            logger.error(f"No metadata for {video_path}")
            return
        
        duration = metadata['duration']
        fps = metadata['fps']
        
        # Calculate resume position
        start_time = video_info['last_timestamp']
        patches_created_total = 0
        
        # Process enabled sizes
        enabled_sizes = [
            key for key, config in self.config['output_patches'].items()
            if config.get('enabled', True)
        ]
        
        # Process video
        current_time = start_time
        stride_seconds = self.config['processing']['stride'] / fps if fps > 0 else 1.0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            task_id = progress.add_task(
                f"[cyan]{Path(video_path).name}",
                total=target_patches
            )
            
            while patches_created_total < target_patches and current_time < duration - 1.0:
                try:
                    # Extract 7 UHD frames
                    frames = self.extract_frames_uhd(video_path, current_time, n_frames=7)
                    
                    if frames is None:
                        current_time += stride_seconds
                        continue
                    
                    # Check for scene change
                    if self.is_scene_change(frames):
                        current_time += stride_seconds
                        continue
                    
                    # Check for blur (center frame)
                    if self.is_blurry(frames[3]):
                        current_time += stride_seconds
                        continue
                    
                    # Create patches for each enabled size
                    for size_key in enabled_sizes:
                        gt, lr_stacked = self.create_patch_pair(frames, size_key)
                        
                        if gt is None or lr_stacked is None:
                            continue
                        
                        # Generate filename
                        video_name = Path(video_path).stem
                        patch_name = f"{video_name}_{int(current_time*1000):08d}_{size_key}.png"
                        
                        # Save patches
                        gt_path = self.dataset_path / "patches" / size_key / "GT" / patch_name
                        lr_path = self.dataset_path / "patches" / size_key / "LR" / patch_name
                        
                        cv2.imwrite(str(gt_path), gt, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                        cv2.imwrite(str(lr_path), lr_stacked, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                    
                    patches_created_total += 1
                    progress.update(task_id, advance=1)
                    
                    # Update progress in state manager
                    self.patches_since_save += 1
                    if self.patches_since_save >= self.save_interval:
                        self.state_manager.update_video_progress(
                            video_path,
                            self.patches_since_save,
                            current_time
                        )
                        self.state_manager.save()
                        self.patches_since_save = 0
                    
                    # Move to next position
                    current_time += stride_seconds
                
                except Exception as e:
                    logger.error(f"Error at {current_time}s: {e}")
                    current_time += stride_seconds
                    continue
        
        # Final update
        if self.patches_since_save > 0:
            self.state_manager.update_video_progress(
                video_path,
                self.patches_since_save,
                current_time
            )
            self.state_manager.save()
            self.patches_since_save = 0
        
        logger.info(f"✅ Completed: {patches_created_total} patches created")
    
    def run(self):
        """Main generation loop with resume support"""
        console.print(Panel.fit(
            "[bold cyan]Dataset Generator V2[/bold cyan]\n"
            "UHD Quality Preservation • 7-Frame • State Management",
            border_style="cyan"
        ))
        
        # Scan videos (cached!)
        self.state_manager.scan_videos()
        
        # Calculate distribution
        if not self.state_manager.state['category_distribution']:
            self.state_manager.calculate_distribution()
        
        # Display progress summary
        console.print("\n[bold]Current Progress:[/bold]")
        console.print(self.state_manager.get_progress_summary())
        console.print()
        
        # Process videos
        while True:
            task = self.state_manager.get_next_video_task()
            if task is None:
                break
            
            video_path, video_info, remaining_patches = task
            self.process_video(video_path, video_info, remaining_patches)
        
        # Mark complete
        self.state_manager.mark_complete()
        
        console.print("\n[bold green]✅ Generation Complete![/bold green]")
        console.print(self.state_manager.get_progress_summary())


def main():
    """Main entry point"""
    config_path = "generator_config_v2.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    if not os.path.exists(config_path):
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        console.print("[yellow]Please create generator_config_v2.json or specify config path[/yellow]")
        sys.exit(1)
    
    try:
        generator = DatasetGeneratorV2(config_path)
        generator.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Interrupted by user[/yellow]")
        console.print("[green]Progress saved. Run again to resume.[/green]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
