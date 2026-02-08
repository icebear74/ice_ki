#!/usr/bin/env python3
"""
Dataset Generator V2 - Clean Rewrite
7-Frame Only, New Directory Structure, Horizontal LR Stacking
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import traceback
from datetime import datetime

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.console import Console
from rich.table import Table

console = Console()


class DatasetConfig:
    """Configuration loader and validator"""
    
    def __init__(self, config_path: str = "generator_config_v2.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> dict:
        """Load JSON configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            console.print(f"[red]Error: Config file not found: {self.config_path}[/red]")
            sys.exit(1)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error: Invalid JSON in config: {e}[/red]")
            sys.exit(1)
    
    def _validate_config(self):
        """Validate required config fields"""
        required = ["root_path", "dataset_name", "videos", "sizes"]
        missing = [field for field in required if field not in self.config]
        if missing:
            console.print(f"[red]Error: Missing required config fields: {missing}[/red]")
            sys.exit(1)
        
        # Validate size keys
        valid_size_keys = {"720", "540", "720_169"}
        invalid_keys = set(self.config["sizes"].keys()) - valid_size_keys
        if invalid_keys:
            console.print(f"[yellow]Warning: Invalid size keys: {invalid_keys}[/yellow]")
            console.print(f"[yellow]Valid keys are: {valid_size_keys}[/yellow]")
    
    def get_root_path(self) -> Path:
        return Path(self.config["root_path"]).expanduser()
    
    def get_dataset_name(self) -> str:
        return self.config["dataset_name"]
    
    def get_videos(self) -> List[str]:
        return self.config["videos"]
    
    def get_sizes(self) -> Dict:
        return self.config["sizes"]
    
    def get_workers(self) -> int:
        return self.config.get("workers", 4)
    
    def get_scene_threshold(self) -> float:
        return self.config.get("scene_threshold", 30.0)
    
    def get_blur_threshold(self) -> float:
        return self.config.get("blur_threshold", 100.0)
    
    def get_stride(self) -> int:
        return self.config.get("stride", 3)
    
    def get_patch_size(self) -> int:
        return self.config.get("patch_size", 256)


class SceneDetector:
    """Scene change and blur detection"""
    
    def __init__(self, scene_threshold: float = 30.0, blur_threshold: float = 100.0):
        self.scene_threshold = scene_threshold
        self.blur_threshold = blur_threshold
        self.prev_hist = None
    
    def reset(self):
        """Reset detector state"""
        self.prev_hist = None
    
    def is_scene_change(self, frame: np.ndarray) -> bool:
        """Detect scene change using histogram comparison"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        if self.prev_hist is None:
            self.prev_hist = hist
            return False
        
        diff = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
        diff_percent = diff * 100
        
        self.prev_hist = hist
        return diff_percent > self.scene_threshold
    
    def is_blurry(self, frame: np.ndarray) -> bool:
        """Detect blurry frames using Laplacian variance"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < self.blur_threshold


class DatasetGenerator:
    """Main dataset generator class"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.root_path = config.get_root_path()
        self.dataset_name = config.get_dataset_name()
        self.dataset_path = self.root_path / self.dataset_name
        
        self.scene_detector = SceneDetector(
            config.get_scene_threshold(),
            config.get_blur_threshold()
        )
        
        self.stats = {
            size_key: {
                "total_patches": 0,
                "skipped_scene": 0,
                "skipped_blur": 0,
                "errors": 0
            }
            for size_key in config.get_sizes().keys()
        }
        
        self.error_log = []
        self._setup_directories()
    
    def _setup_directories(self):
        """Create directory structure"""
        sizes = self.config.get_sizes()
        
        for size_key in sizes.keys():
            # Training patches
            (self.dataset_path / "patches" / size_key / "GT").mkdir(parents=True, exist_ok=True)
            (self.dataset_path / "patches" / size_key / "LR").mkdir(parents=True, exist_ok=True)
            
            # Validation directories (empty for user to populate)
            (self.dataset_path / "val" / size_key / "GT").mkdir(parents=True, exist_ok=True)
            (self.dataset_path / "val" / size_key / "LR").mkdir(parents=True, exist_ok=True)
        
        console.print(f"[green]Created directory structure at: {self.dataset_path}[/green]")
    
    def process_video(self, video_path: str, size_key: str, size_config: dict) -> Tuple[int, int, int, int]:
        """
        Process a single video for a specific size configuration
        Returns: (patches_created, skipped_scene, skipped_blur, errors)
        """
        if not os.path.exists(video_path):
            error_msg = f"Video not found: {video_path}"
            self.error_log.append(error_msg)
            return (0, 0, 0, 1)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_msg = f"Failed to open video: {video_path}"
            self.error_log.append(error_msg)
            return (0, 0, 0, 1)
        
        try:
            patches_created = 0
            skipped_scene = 0
            skipped_blur = 0
            errors = 0
            
            # Get target dimensions
            target_height = size_config["height"]
            target_width = size_config["width"]
            patch_size = self.config.get_patch_size()
            stride = self.config.get_stride()
            
            video_name = Path(video_path).stem
            self.scene_detector.reset()
            
            frame_buffer = []
            frame_idx = 0
            patch_counter = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame
                frame_resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
                
                # Scene detection
                if self.scene_detector.is_scene_change(frame_resized):
                    frame_buffer.clear()
                    skipped_scene += 1
                    frame_idx += 1
                    continue
                
                # Blur detection
                if self.scene_detector.is_blurry(frame_resized):
                    skipped_blur += 1
                    frame_idx += 1
                    continue
                
                frame_buffer.append(frame_resized)
                
                # Extract 7-frame sequences
                if len(frame_buffer) >= 7:
                    try:
                        created = self._extract_patches(
                            frame_buffer[:7],
                            video_name,
                            size_key,
                            patch_counter,
                            patch_size
                        )
                        patches_created += created
                        patch_counter += created
                    except Exception as e:
                        error_msg = f"Error extracting patches from {video_path} at frame {frame_idx}: {str(e)}"
                        self.error_log.append(error_msg)
                        errors += 1
                    
                    # Slide window by stride
                    frame_buffer = frame_buffer[stride:]
                
                frame_idx += 1
            
            return (patches_created, skipped_scene, skipped_blur, errors)
        
        finally:
            cap.release()
    
    def _extract_patches(self, frames: List[np.ndarray], video_name: str, 
                        size_key: str, patch_counter: int, patch_size: int) -> int:
        """
        Extract patches from 7-frame sequence
        Returns: number of patches created
        """
        if len(frames) != 7:
            return 0
        
        height, width = frames[0].shape[:2]
        patches_created = 0
        
        # Extract patches from the frame
        for y in range(0, height - patch_size + 1, patch_size):
            for x in range(0, width - patch_size + 1, patch_size):
                # Extract GT patch (middle frame, index 3)
                gt_patch = frames[3][y:y+patch_size, x:x+patch_size]
                
                # Extract LR patches from all 7 frames
                lr_patches = []
                for frame in frames:
                    lr_patch = frame[y:y+patch_size, x:x+patch_size]
                    lr_patches.append(lr_patch)
                
                # Stack LR patches HORIZONTALLY (H, W×7)
                lr_stacked = np.hstack(lr_patches)  # Horizontal concatenation
                
                # Generate filename
                patch_name = f"{video_name}_p{patch_counter + patches_created:06d}.png"
                
                # Save patches
                gt_path = self.dataset_path / "patches" / size_key / "GT" / patch_name
                lr_path = self.dataset_path / "patches" / size_key / "LR" / patch_name
                
                cv2.imwrite(str(gt_path), gt_patch)
                cv2.imwrite(str(lr_path), lr_stacked)
                
                patches_created += 1
        
        return patches_created
    
    def process_all_videos(self):
        """Process all videos with multi-worker support"""
        videos = self.config.get_videos()
        sizes = self.config.get_sizes()
        workers = self.config.get_workers()
        
        # Create task list
        tasks = []
        for video_path in videos:
            for size_key, size_config in sizes.items():
                tasks.append((video_path, size_key, size_config))
        
        console.print(f"[cyan]Processing {len(videos)} videos with {len(sizes)} size configurations[/cyan]")
        console.print(f"[cyan]Total tasks: {len(tasks)}, Workers: {workers}[/cyan]")
        
        # Process with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            task_id = progress.add_task("[green]Processing videos...", total=len(tasks))
            
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self._process_video_worker, video_path, size_key, size_config): (video_path, size_key)
                    for video_path, size_key, size_config in tasks
                }
                
                for future in as_completed(futures):
                    video_path, size_key = futures[future]
                    try:
                        patches, skipped_scene, skipped_blur, errors = future.result()
                        
                        # Update stats
                        self.stats[size_key]["total_patches"] += patches
                        self.stats[size_key]["skipped_scene"] += skipped_scene
                        self.stats[size_key]["skipped_blur"] += skipped_blur
                        self.stats[size_key]["errors"] += errors
                        
                    except Exception as e:
                        error_msg = f"Worker error processing {video_path} [{size_key}]: {str(e)}\n{traceback.format_exc()}"
                        self.error_log.append(error_msg)
                        self.stats[size_key]["errors"] += 1
                    
                    progress.update(task_id, advance=1)
    
    def _process_video_worker(self, video_path: str, size_key: str, size_config: dict) -> Tuple[int, int, int, int]:
        """Worker function for multiprocessing"""
        # Create new generator instance for this worker
        worker_gen = DatasetGenerator.__new__(DatasetGenerator)
        worker_gen.config = self.config
        worker_gen.root_path = self.root_path
        worker_gen.dataset_name = self.dataset_name
        worker_gen.dataset_path = self.dataset_path
        worker_gen.scene_detector = SceneDetector(
            self.config.get_scene_threshold(),
            self.config.get_blur_threshold()
        )
        worker_gen.error_log = []
        
        return worker_gen.process_video(video_path, size_key, size_config)
    
    def print_statistics(self):
        """Print final statistics"""
        console.print("\n[bold cyan]Dataset Generation Statistics[/bold cyan]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Size", style="cyan")
        table.add_column("Patches", justify="right", style="green")
        table.add_column("Skipped (Scene)", justify="right", style="yellow")
        table.add_column("Skipped (Blur)", justify="right", style="yellow")
        table.add_column("Errors", justify="right", style="red")
        
        total_patches = 0
        total_skipped_scene = 0
        total_skipped_blur = 0
        total_errors = 0
        
        for size_key, stats in self.stats.items():
            table.add_row(
                size_key,
                str(stats["total_patches"]),
                str(stats["skipped_scene"]),
                str(stats["skipped_blur"]),
                str(stats["errors"])
            )
            total_patches += stats["total_patches"]
            total_skipped_scene += stats["skipped_scene"]
            total_skipped_blur += stats["skipped_blur"]
            total_errors += stats["errors"]
        
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{total_patches}[/bold]",
            f"[bold]{total_skipped_scene}[/bold]",
            f"[bold]{total_skipped_blur}[/bold]",
            f"[bold]{total_errors}[/bold]",
            style="bold"
        )
        
        console.print(table)
        
        # Save statistics to file
        stats_path = self.dataset_path / "generation_stats.json"
        with open(stats_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "stats": self.stats,
                "totals": {
                    "patches": total_patches,
                    "skipped_scene": total_skipped_scene,
                    "skipped_blur": total_skipped_blur,
                    "errors": total_errors
                }
            }, f, indent=2)
        
        console.print(f"\n[green]Statistics saved to: {stats_path}[/green]")
    
    def save_error_log(self):
        """Save error log to file"""
        if not self.error_log:
            return
        
        error_log_path = self.dataset_path / "generation_errors.log"
        with open(error_log_path, 'w') as f:
            f.write(f"Dataset Generation Error Log\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, error in enumerate(self.error_log, 1):
                f.write(f"Error #{i}:\n{error}\n\n")
        
        console.print(f"[yellow]Errors logged to: {error_log_path}[/yellow]")


def main():
    """Main entry point"""
    console.print("[bold blue]Dataset Generator V2 - Clean Rewrite[/bold blue]")
    console.print("[blue]7-Frame Only, Horizontal LR Stacking[/blue]\n")
    
    # Load configuration
    config_path = "generator_config_v2.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    console.print(f"[cyan]Loading configuration from: {config_path}[/cyan]")
    config = DatasetConfig(config_path)
    
    # Create generator
    generator = DatasetGenerator(config)
    
    # Process videos
    try:
        generator.process_all_videos()
    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Fatal error: {str(e)}[/red]")
        traceback.print_exc()
        sys.exit(1)
    
    # Print statistics
    generator.print_statistics()
    
    # Save error log if any errors occurred
    generator.save_error_log()
    
    console.print("\n[bold green]Dataset generation complete![/bold green]")


if __name__ == "__main__":
    main()
