#!/usr/bin/env python3
"""
Multi-Category Dataset Generator v2.0
Generates training patches for multiple model categories (dynamically configured).
"""

import os
import sys
import cv2
import subprocess
import random
import json
import shutil
import re
import time
import signal
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

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
    print("Falling back to basic output...")

class DatasetGeneratorV2:
    """Multi-category dataset generator with beautiful GUI."""
    
    # Maximum number of priority levels to display in console output
    MAX_DISPLAYED_PRIORITIES = 10
    
    def __init__(self, config_path: str):
        """Initialize the generator."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.settings = self.config['base_settings']
        self.videos = self.config['videos']
        self.format_config = self.config.get('format_config', {})
        
        # Initialize logger
        self.logger = self._setup_logger()
        
        # Make logger available globally for exception handler
        sys.logger = self.logger
        
        # Sort videos by priority (ascending: 0 first, 255 last)
        # Within same priority, randomize order using pre-generated random values
        random.seed(42)  # Reproducible randomization
        # Attach a random value to each video for stable sorting
        for i, video in enumerate(self.videos):
            video['_sort_random'] = random.random()
        self.videos.sort(key=lambda v: (v.get('priority', 255), v['_sort_random']))
        # Clean up temporary sort keys
        for video in self.videos:
            video.pop('_sort_random', None)
        
        # Log initialization info
        self.logger.info(f"Initializing generator with {len(self.videos)} videos")
        self.logger.debug(f"First 5 videos: {[v['name'] for v in self.videos[:5]]}")
        
        # Initialize paths
        self.base_dir = self.settings['output_base_dir']
        self.temp_dir = self.settings['temp_dir']
        self.status_file = self.settings['status_file']
        
        # Initialize progress tracker
        self.tracker = ProgressTracker(self.status_file)
        self.tracker.update_progress(total_videos=len(self.videos))
        
        # Initialize category stats from config
        if 'category_targets' in self.config:
            self.tracker.initialize_categories(self.config['category_targets'])
        
        # Runtime state
        self.workers = self.settings['max_workers']
        self.running = True
        self.paused = False
        self.last_update_time = time.time()
        self.update_interval = 0.5  # Update GUI every 0.5 seconds
        
        # Rich console
        if RICH_AVAILABLE:
            self.console = Console()
            
            # Show priority distribution
            priority_counts = {}
            for v in self.videos:
                p = v.get('priority', 255)
                priority_counts[p] = priority_counts.get(p, 0) + 1
            
            self.console.print("\n[bold]📋 Video Processing Order:[/bold]")
            sorted_priorities = sorted(priority_counts.keys())
            
            # Always show priority 255 (default) if it exists, plus first MAX_DISPLAYED_PRIORITIES-1 levels
            priorities_to_show = []
            if 255 in priority_counts:
                # Show first levels (excluding 255 if present)
                priorities_to_show = [p for p in sorted_priorities if p != 255][:self.MAX_DISPLAYED_PRIORITIES - 1]
                # Always include 255
                priorities_to_show.append(255)
                priorities_to_show.sort()
            else:
                priorities_to_show = sorted_priorities[:self.MAX_DISPLAYED_PRIORITIES]
            
            for priority in priorities_to_show:
                count = priority_counts[priority]
                if priority == 255:
                    self.console.print(f"   Priority {priority} (default): {count} videos")
                else:
                    self.console.print(f"   Priority {priority}: {count} videos")
            
            # Show remaining count if there are more priorities
            remaining_priorities = [p for p in sorted_priorities if p not in priorities_to_show]
            if remaining_priorities:
                remaining = sum(priority_counts[p] for p in remaining_priorities)
                self.console.print(f"   ... and {remaining} more videos in other priority levels")
        
        # Statistics
        self.start_time = time.time()
        self.extractions_count = 0
        self.success_count = 0
        self.current_video_name = ""
        
        # Keyboard input handling
        self.input_thread = None
        self.stop_input_thread = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup and configure the debug logger."""
        logger = logging.getLogger('DatasetGeneratorV2')
        
        # Check if logging is enabled
        enable_logging = self.settings.get('enable_debug_logging', True)
        
        if not enable_logging:
            # Use NullHandler if logging is disabled
            logger.addHandler(logging.NullHandler())
            logger.setLevel(logging.CRITICAL)
            return logger
        
        # Set logging level
        logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        logger.handlers = []
        
        # Get log file path from config
        log_path = self.settings.get('debug_log_path', '/mnt/data/training/dataset/generator_debug.log')
        
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        return logger
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        if RICH_AVAILABLE:
            self.console.print("\n[yellow]⏸️  Saving current progress before exit...[/yellow]")
        else:
            print("\n⏸️  Saving current progress before exit...")
        
        self.stop_input_thread = True
        self.running = False
        
        # Force save current state
        self.tracker.set_status("interrupted")
        self.tracker.save()
        
        if RICH_AVAILABLE:
            self.console.print("[green]✅ Progress saved. You can resume later.[/green]")
        else:
            print("✅ Progress saved. You can resume later.")
        
        sys.exit(0)
    
    def _keyboard_listener(self):
        """Listen for keyboard input in a separate thread."""
        import sys
        import tty
        import termios
        
        # Save terminal settings
        old_settings = None
        try:
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            
            while not self.stop_input_thread and self.running:
                try:
                    # Non-blocking read with timeout
                    import select
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        ch = sys.stdin.read(1)
                        
                        if ch == ' ':  # Space bar - pause/resume
                            self.paused = not self.paused
                            status = "PAUSED" if self.paused else "RESUMED"
                            if RICH_AVAILABLE:
                                self.console.print(f"\n[yellow]⏸️  {status}[/yellow]")
                        
                        elif ch == '+' or ch == '=':  # Increase workers
                            if self.workers < 32:  # Max 32 workers
                                self.workers += 1
                                if RICH_AVAILABLE:
                                    self.console.print(f"\n[green]⬆️  Workers increased to {self.workers}[/green]")
                        
                        elif ch == '-' or ch == '_':  # Decrease workers
                            if self.workers > 1:  # Min 1 worker
                                self.workers -= 1
                                if RICH_AVAILABLE:
                                    self.console.print(f"\n[yellow]⬇️  Workers decreased to {self.workers}[/yellow]")
                        
                        elif ch == 'q' or ch == 'Q':  # Quit
                            if RICH_AVAILABLE:
                                self.console.print("\n[yellow]Quitting...[/yellow]")
                            self.running = False
                            self.stop_input_thread = True
                            break
                
                except Exception:
                    pass
        
        finally:
            # Restore terminal settings
            if old_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    def _start_keyboard_listener(self):
        """Start keyboard listener thread."""
        if not RICH_AVAILABLE:
            return  # Only enable for rich mode
        
        try:
            self.input_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
            self.input_thread.start()
        except Exception as e:
            # If keyboard listener fails, continue without it
            if RICH_AVAILABLE:
                self.console.print(f"[yellow]⚠️  Keyboard controls unavailable: {e}[/yellow]")
    
    def get_category_path(self, category: str) -> str:
        """
        Get the base path for a category.
        Falls back to hard-coded paths for known categories,
        or generates a default path for custom categories.
        """
        # Check if there's a category_paths config (future enhancement)
        if 'category_paths' in self.config:
            if category in self.config['category_paths']:
                return self.config['category_paths'][category]
        
        # Fall back to hard-coded paths for backward compatibility
        if category in CATEGORY_PATHS:
            return CATEGORY_PATHS[category]
        
        # Generate default path for custom categories
        # Format: CategoryName/CategoryNameModel/Learn
        category_title = category.capitalize()
        return f"{category_title}/{category_title}Model/Learn"
    
    def select_format_for_category(self, category: str) -> str:
        """
        Select a random format for a category based on configured distribution.
        """
        import random
        
        # Get format distribution from config
        distribution = self.format_config.get(category, {})
        
        if not distribution:
            # Fallback to hard-coded distribution if not in config
            distribution = CATEGORY_FORMAT_DISTRIBUTION.get(category, {})
        
        if not distribution:
            # Ultimate fallback
            return 'small_540'
        
        formats = list(distribution.keys())
        # Extract probability from dict or use value directly
        weights = []
        for fmt in formats:
            if isinstance(distribution[fmt], dict):
                weights.append(distribution[fmt].get('probability', 1.0))
            else:
                weights.append(distribution[fmt])
        
        return random.choices(formats, weights=weights, k=1)[0]
    
    def get_output_dirs_for_category_format(self, category: str, format_name: str, lr_frames: int = 5) -> dict:
        """
        Get output directory paths for a specific category and format.
        
        Args:
            category: Category name
            format_name: Format name (small_540, etc.)
            lr_frames: Number of LR frames to use (5 or 7)
        
        Returns:
            Dictionary with 'gt', 'lr', 'val_gt', 'val_lr' paths
        """
        category_path = self.get_category_path(category)
        format_spec = FORMATS[format_name]
        base_format_dir = format_spec['output_dir']
        
        # V2 Generator: Use 'LR' for 7-frame (new standard)
        lr_dir_name = 'LR'
        
        return {
            'gt': f"{self.base_dir}/{category_path}/{base_format_dir}/GT",
            'lr': f"{self.base_dir}/{category_path}/{base_format_dir}/{lr_dir_name}",
            'val_gt': f"{self.base_dir}/{category_path}/Val/GT",
            'val_lr': f"{self.base_dir}/{category_path}/Val/LR"
        }
    
    def get_video_info(self, video_path: str) -> Tuple[float, float]:
        """Get video FPS and duration using ffprobe."""
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'format=duration:stream=avg_frame_rate',
                '-of', 'json', video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            data = json.loads(result.stdout)
            fps = eval(data['streams'][0]['avg_frame_rate'])
            duration = float(data['format']['duration'])
            return fps, duration
        except:
            return 23.976, 3600.0
    
    def create_output_directories(self):
        """
        Create all necessary output directories.
        
        New V2 structure:
            - Patches/GT/ and Patches/LR/ (7-frame horizontal, for training)
            - Val/GT/ and Val/LR/ (validation)
        """
        for category in self.config.get('category_targets', {}).keys():
            # Get format distribution for this category
            category_formats = self.format_config.get(category, {})
            
            if not category_formats:
                # Fallback to hard-coded distribution
                category_formats = CATEGORY_FORMAT_DISTRIBUTION.get(category, {'small_540': 1.0})
            
            for format_name in category_formats.keys():
                # Create directories for 7-frame LR (new V2 standard)
                dirs_7 = self.get_output_dirs_for_category_format(category, format_name, lr_frames=7)
                for dir_path in dirs_7.values():
                    os.makedirs(dir_path, exist_ok=True)
        
        # Create temp directory
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def extract_full_resolution_frames(self, video_path: str, timestamp: float, thread_id: str) -> Optional[List]:
        """Extract 7 frames at FULL 1920×1080 resolution ONCE."""
        thread_temp = os.path.join(self.temp_dir, f"extract_{thread_id}")
        os.makedirs(thread_temp, exist_ok=True)
        
        try:
            # HDR tonemap filter
            tonemap_vf = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=mobius,zscale=t=bt709:m=bt709,format=yuv420p,scale=1920:1080:flags=lanczos"
            
            # Extract 7 frames at full resolution with 4 threads
            cmd = [
                'nice', '-n', '19',
                'ffmpeg', '-y', 
                '-threads', '4',  # USE 4 CORES instead of 1!
                '-ss', str(round(timestamp, 3)),
                '-i', video_path,
                '-vf', tonemap_vf,
                '-vframes', '7',
                os.path.join(thread_temp, 'frame_%d.png')
            ]
            
            # Log FFmpeg command
            self.logger.debug(f"FFmpeg command: {' '.join(cmd)}")
            
            subprocess.run(cmd, capture_output=True, check=False, timeout=30)
            
            # Load all 7 frames
            frames = []
            for i in range(1, 8):
                frame_path = os.path.join(thread_temp, f"frame_{i}.png")
                if os.path.exists(frame_path) and os.path.getsize(frame_path) > self.settings['min_file_size']:
                    img = cv2.imread(frame_path)
                    if img is not None and img.shape[0] == 1080 and img.shape[1] == 1920:
                        frames.append(img)
            
            if len(frames) == 7:
                return frames
            
            return None
            
        except Exception as e:
            return None
        finally:
            # Clean up temp files
            if os.path.exists(thread_temp):
                shutil.rmtree(thread_temp, ignore_errors=True)
    
    def process_all_categories_from_frames(self, frames: List, categories: Dict[str, float], 
                                          video_name: str, frame_idx: int) -> bool:
        """Process all category patches from the same 7 full-resolution frames."""
        
        # Accept all frames (including scenes with cuts - realistic training data)
        all_success = True
        
        # Process each category with different random crops
        for category, weight in categories.items():
            # Select format for this category
            format_name = self.select_format_for_category(category)
            
            # Save patches (uses DIFFERENT random crop per category)
            success = self.save_patches(frames, category, format_name, 
                                      video_name, frame_idx)
            
            if success:
                self.tracker.increment_category_images(category, 1)
            else:
                all_success = False
        
        return all_success
    
    def create_lr_stack(self, frames: List, lr_size: Tuple[int, int], crop_y: int, crop_x: int, crop_h: int, crop_w: int) -> any:
        """Create horizontally stacked LR frames (7-frame horizontal stacking)."""
        lr_frames = []
        for frame in frames:
            # Crop from the frame
            cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            # Resize to LR size
            resized = cv2.resize(cropped, lr_size, interpolation=cv2.INTER_LANCZOS4)
            lr_frames.append(resized)
        
        # Stack horizontally (width × 7)
        return cv2.hconcat(lr_frames)
    
    def save_patches(self, frames: List, category: str, format_name: str, 
                     video_name: str, frame_idx: int) -> bool:
        """
        Save GT and LR patches for a specific category and format.
        
        New V2 Training expects:
            - GT: Single ground truth frame (e.g., 540×540)
            - LR: 7-frame stack horizontally (e.g., 180×1260 for 540 patches)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get format specifications
            format_spec = FORMATS[format_name]
            gt_h, gt_w = format_spec['gt_size']
            lr_h, lr_w = format_spec['lr_size']
            suffix = format_spec['suffix']
            
            # Get output directories (7-frame LR only)
            dirs_7 = self.get_output_dirs_for_category_format(category, format_name, lr_frames=7)
            
            # Generate random crop position
            max_y = 1080 - gt_h
            max_x = 1920 - gt_w
            crop_y = random.randint(0, max_y) if max_y > 0 else 0
            crop_x = random.randint(0, max_x) if max_x > 0 else 0
            
            # Clean video name for filename
            clean_name = re.sub(r'[^a-zA-Z0-9]', '_', video_name)
            filename = f"patch_{clean_name}_idx{frame_idx}{suffix}.png"
            
            # Save GT (middle frame = frames[3])
            gt_frame = frames[3][crop_y:crop_y+gt_h, crop_x:crop_x+gt_w]
            gt_path = os.path.join(dirs_7['gt'], filename)
            cv2.imwrite(gt_path, gt_frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            
            # Save 7-frame LR (all 7 frames, horizontally stacked)
            # Result shape: (H, W×7, 3) - e.g., (180, 1260, 3) for 540 patches
            lr_7 = self.create_lr_stack(frames[0:7], (lr_w, lr_h), crop_y, crop_x, gt_h, gt_w)
            lr7_path = os.path.join(dirs_7['lr'], filename)
            cv2.imwrite(lr7_path, lr_7, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            
            # Verify files were created
            if os.path.exists(gt_path) and os.path.exists(lr7_path):
                return True
            
            return False
            
        except Exception as e:
            return False
    
    def extract_with_retry(self, video_path: str, video_name: str, 
                          categories: Dict[str, float], frame_idx: int, 
                          duration: float) -> Tuple[bool, int]:
        """Extract frames once with retry logic, process all categories."""
        timestamp = (frame_idx * duration / self.settings['base_frame_limit']) % duration
        thread_id = f"{random.randint(1000, 9999)}_{int(time.time()*1000) % 10000}"
        
        for attempt in range(self.settings['max_retry_attempts']):
            # Extract 7 full-resolution frames ONCE
            frames = self.extract_full_resolution_frames(video_path, timestamp, thread_id)
            
            if frames is None:
                self.logger.debug(f"Extracted 0 frames on attempt {attempt + 1}")
                timestamp = (timestamp + self.settings['retry_skip_seconds']) % duration
                continue
            
            # Log successful extraction
            self.logger.debug(f"Extracted {len(frames)} frames on attempt {attempt + 1}")
            
            # Process ALL categories from these frames
            success = self.process_all_categories_from_frames(
                frames, categories, video_name, frame_idx
            )
            
            if success:
                return True, attempt + 1
            
            timestamp = (timestamp + self.settings['retry_skip_seconds']) % duration
        
        return False, self.settings['max_retry_attempts']
    
    def process_video(self, video_idx: int, video_info: Dict) -> Dict:
        """Process a single video and generate all patches."""
        video_path = video_info['path']
        video_name = video_info['name']
        categories = video_info['categories']
        
        try:
            self.logger.debug(f"process_video({video_idx}): {video_name}")
            self.logger.debug(f"Video path: {video_path}")
            self.logger.debug(f"Video exists: {os.path.exists(video_path)}")
            
            # Check if video exists
            if not os.path.exists(video_path):
                self.logger.error(f"Video {video_idx} not found: {video_path}")
                if RICH_AVAILABLE:
                    self.console.print(f"[red]⚠️  Skipping '{video_name}': File not found[/red]")
                    self.console.print(f"[dim]    Path: {video_path}[/dim]")
                else:
                    print(f"⚠️  Skipping '{video_name}': File not found")
                    print(f"    Path: {video_path}")
                
                return {
                    'success': False,
                    'video_name': video_name,
                    'message': 'Video file not found'
                }
            
            # Get video info
            fps, duration = self.get_video_info(video_path)
            
            # Calculate total weighted extractions for this video
            total_weight = sum(categories.values())
            total_extractions = int(self.settings['base_frame_limit'] * total_weight)
            
            self.logger.debug(f"Video {video_idx}: total_extractions={total_extractions}, duration={duration}s")
            
            # Update tracker
            self.tracker.update_progress(
                current_video_index=video_idx,
                current_video_path=video_path
            )
            self.tracker.update_video_checkpoint(
                video_idx, 
                "in_progress",
                extractions_done=0,
                extractions_target=total_extractions
            )
            self.tracker.save()
            
            # Process extractions
            success_count = 0
            self.current_video_name = video_name
            
            for frame_idx in range(total_extractions):
                if not self.running:
                    # Save checkpoint before breaking on stop
                    self.tracker.update_video_checkpoint(
                        video_idx,
                        "interrupted",
                        last_frame_idx=frame_idx,
                        extractions_done=frame_idx,
                        extractions_target=total_extractions
                    )
                    self.tracker.save()
                    break
                
                while self.paused:
                    time.sleep(0.1)
                
                # Log extraction progress every 100 frames
                if frame_idx % 100 == 0:
                    self.logger.debug(f"Video {video_idx}: extraction {frame_idx}/{total_extractions}")
                
                success, attempts = self.extract_with_retry(
                    video_path, video_name, categories, frame_idx, duration
                )
                
                if success:
                    success_count += 1
                    self.success_count += 1
                
                self.extractions_count += 1
                
                # Update checkpoint EVERY extraction for instant resume capability
                # Save to disk every 5 extractions to balance performance and safety
                self.tracker.update_video_checkpoint(
                    video_idx,
                    "in_progress",
                    last_frame_idx=frame_idx,
                    extractions_done=frame_idx + 1,
                    extractions_target=total_extractions
                )
                
                if frame_idx % 5 == 0:  # Save every 5 extractions (was 10)
                    self.tracker.save()
                    
                    # Update live display if enabled
                    if hasattr(self, 'live_display') and self.live_display and self._should_update_display():
                        try:
                            self.live_display.update(self._build_complete_layout())
                        except:
                            pass  # Ignore display errors
                    elif hasattr(self, '_should_update_display') and self._should_update_display():
                        # Use professional box-drawing GUI
                        try:
                            from utils.ui_display import draw_dataset_generator_ui
                            draw_dataset_generator_ui(self)
                        except:
                            pass  # Ignore display errors
            
            # Mark video as completed
            self.tracker.update_video_checkpoint(video_idx, "completed")
            for category in categories.keys():
                self.tracker.increment_category_videos(category)
            
            self.tracker.save()
            
            # Log completion
            self.logger.info(f"Video {video_idx} COMPLETED: {success_count}/{total_extractions} successful")
            
            return {
                'success': True,
                'video_name': video_name,
                'extractions': total_extractions,
                'success_count': success_count
            }
        except Exception as e:
            self.logger.error(f"Exception in process_video({video_idx}): {e}", exc_info=True)
            return {
                'success': False,
                'video_name': video_name,
                'message': f'Exception: {e}'
            }
    
    def _get_terminal_width(self) -> int:
        """Get terminal width, with fallback to default."""
        try:
            import shutil
            return shutil.get_terminal_size().columns
        except:
            return 120  # Default fallback
    
    def _calculate_bar_widths(self) -> dict:
        """Calculate optimal bar widths based on terminal width."""
        terminal_width = self._get_terminal_width()
        
        # Reserve space for labels and other content
        # Overall bar: "Overall Progress  " (20 chars) + percentage (10) + ETA (15) = ~45 chars
        # Category bar: "CATEGORYNAME  " (15) + percentage (10) + "Images: X,XXX,XXX" (20) + "ETA: X:XX:XX" (15) = ~60 chars
        
        # Calculate bar widths as percentage of terminal width
        overall_bar_width = max(30, min(80, int((terminal_width - 45) * 0.6)))
        video_bar_width = max(30, min(80, int((terminal_width - 40) * 0.6)))
        category_bar_width = max(25, min(60, int((terminal_width - 60) * 0.5)))
        
        return {
            'overall': overall_bar_width,
            'video': video_bar_width,
            'category': category_bar_width
        }
    
    def build_gui_layout(self) -> tuple:
        """Build the beautiful GUI layout using rich with progress bars."""
        if not RICH_AVAILABLE:
            return self._build_simple_status(), None, None, None
        
        # Calculate statistics
        elapsed = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        current_idx = self.tracker.status['progress']['current_video_index']
        total_videos = self.tracker.status['progress']['total_videos']
        completed_videos = self.tracker.status['progress']['completed_videos']
        
        # ETA calculation for overall progress
        if completed_videos > 0:
            avg_time_per_video = elapsed / completed_videos
            remaining_videos = total_videos - completed_videos
            eta_seconds = avg_time_per_video * remaining_videos
            overall_eta_str = str(timedelta(seconds=int(eta_seconds)))
        else:
            overall_eta_str = "Calculating..."
        
        # Extraction speed
        if elapsed > 0:
            extractions_per_sec = self.extractions_count / elapsed
            speed_str = f"{extractions_per_sec:.1f} extractions/sec"
        else:
            speed_str = "Calculating..."
        
        # Get dynamic bar widths based on terminal size
        bar_widths = self._calculate_bar_widths()
        
        # ===== OVERALL PROGRESS BAR =====
        from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TaskProgressColumn
        
        overall_progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=bar_widths['overall']),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        )
        overall_task = overall_progress.add_task(
            "Overall Progress", 
            total=total_videos if total_videos > 0 else 100,
            completed=completed_videos
        )
        
        # Build prominent header with current movie
        current_movie_name = self.current_video_name if self.current_video_name else "Initializing..."
        header = Panel(
            f"[bold yellow]🎬 CURRENT: {current_movie_name[:70]}[/bold yellow]\n"
            f"[bold cyan]⚙️  CPU CORES: {self.workers}[/bold cyan] • "
            f"[bold green]⚡ SPEED: {speed_str}[/bold green]",
            title="[bold white on blue] DATASET GENERATOR v2.0 - LIVE ",
            border_style="bold blue"
        )
        
        # Overall progress section with bar
        completion_pct = (completed_videos/total_videos*100) if total_videos > 0 else 0
        overall = f"""[bold]📊 OVERALL PROGRESS[/bold]
├─ Videos: {completed_videos}/{total_videos} ({completion_pct:.1f}%)
├─ Remaining: {total_videos - completed_videos} videos
├─ Elapsed: {elapsed_str} | ETA: {overall_eta_str}
├─ Total Extractions: {self.extractions_count:,}
├─ Successful: {self.success_count:,} ({(self.success_count/self.extractions_count*100) if self.extractions_count > 0 else 0:.1f}%)
└─ Status: {'[green]●RUNNING[/green]' if not self.paused else '[yellow]●PAUSED[/yellow]'}
"""
        
        # Current video section with progress bar
        checkpoint = self.tracker.get_video_checkpoint(current_idx)
        if checkpoint and checkpoint.get('status') == 'in_progress':
            done = checkpoint.get('extractions_done', 0)
            target = checkpoint.get('extractions_target', 1)
            progress_pct = (done / target * 100) if target > 0 else 0
            
            # Create a Rich progress bar for current video
            video_progress = Progress(
                TextColumn("[cyan]{task.description}"),
                BarColumn(bar_width=bar_widths['video']),
                TaskProgressColumn(),
            )
            video_task = video_progress.add_task(
                f"{self.current_video_name[:40]}", 
                total=target,
                completed=done
            )
            
            current_video = video_progress
        else:
            current_video = Text("Waiting for next video...", style="dim")
        
        # ===== CATEGORY PROGRESS BARS =====
        category_progress = Progress(
            TextColumn("[bold]{task.description}", justify="left", style="cyan"),
            BarColumn(bar_width=bar_widths['category']),
            TaskProgressColumn(),
            TextColumn("[bold green]Images:"),
            TextColumn("[green]{task.fields[images]:>8,}"),
            TextColumn("[bold yellow]ETA:"),
            TextColumn("[yellow]{task.fields[eta]}"),
        )
        
        for cat_name in sorted(self.config.get('category_targets', {}).keys()):
            stats = self.tracker.status['category_stats'].get(cat_name, {})
            images = stats.get('images_created', 0)
            target = stats.get('target', 1)
            
            # Calculate ETA for this category
            if images > 0 and elapsed > 0:
                rate = images / elapsed
                remaining = target - images
                eta_secs = remaining / rate if rate > 0 else 0
                eta_str = str(timedelta(seconds=int(eta_secs))) if eta_secs > 0 else "Complete"
            else:
                eta_str = "Calculating..."
            
            category_progress.add_task(
                f"{cat_name.upper():12s}",
                total=target,
                completed=images,
                images=images,
                eta=eta_str
            )
        
        # Disk usage
        total_disk = sum(s.get('disk_usage_gb', 0) for s in self.tracker.status['category_stats'].values())
        disk_lines = ["[bold]💾 DISK USAGE[/bold]"]
        categories = sorted(self.config.get('category_targets', {}).keys())
        for i, cat_name in enumerate(categories):
            usage = self.tracker.status['category_stats'].get(cat_name, {}).get('disk_usage_gb', 0)
            prefix = "├─"
            disk_lines.append(f"{prefix} {cat_name.upper()}: {usage:.2f} GB")
        disk_lines.append(f"└─ [bold]Total: {total_disk:.2f} GB[/bold]")
        disk_usage = "\n".join(disk_lines)
        
        # Controls with live status
        pause_status = "[yellow]●PAUSED[/yellow]" if self.paused else "[green]●RUNNING[/green]"
        controls = f"""[bold]⚙️  LIVE CONTROLS[/bold]
├─ Status: {pause_status} | Workers: [bold cyan]{self.workers}[/bold cyan] cores
├─ [SPACE] Pause/Resume | [+/-] Adjust workers
└─ [Ctrl+C] Save & Exit | [q] Quick quit
"""
        
        return header, overall, overall_progress, current_video, category_progress, disk_usage, controls
    
    def _should_update_display(self) -> bool:
        """Check if enough time has passed to update the display."""
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            return True
        return False
    
    def _build_simple_status(self) -> str:
        """Build simple text status when Rich is not available."""
        elapsed = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        current_idx = self.tracker.status['progress']['current_video_index']
        total_videos = self.tracker.status['progress']['total_videos']
        completed_videos = self.tracker.status['progress']['completed_videos']
        
        # Build simple text status
        status_lines = [
            f"\n{'='*60}",
            f"Dataset Generator Progress",
            f"{'='*60}",
            f"Videos: {completed_videos}/{total_videos} ({current_idx+1} processing)",
            f"Elapsed: {elapsed_str}",
            f"Current: {self.current_video_name}",
            f"{'='*60}\n"
        ]
        
        return '\n'.join(status_lines)
    
    def _build_complete_layout(self):
        """Build complete layout for live display."""
        if not RICH_AVAILABLE:
            return self._build_simple_status()
        
        header, overall, overall_progress, current_video, category_progress, disk_usage, controls = self.build_gui_layout()
        
        # Combine everything into a single renderable with proper spacing
        from rich.console import Group
        from rich.text import Text
        
        # Build the complete display
        components = [
            header,
            Text(""),  # Blank line
            Text(overall, style=""),
            Text(""),
            Text("[bold]📊 OVERALL PROGRESS BAR:[/bold]"),
            overall_progress,
            Text(""),
            Text("[bold]🎬 CURRENT VIDEO PROGRESS:[/bold]"),
            current_video,
            Text(""),
            Text("[bold]📦 CATEGORY PROGRESS BARS:[/bold]"),
            category_progress,
            Text(""),
            Text(disk_usage),
            Text(""),
            Text(controls),
        ]
        
        return Group(*components)
    
        """Build simple text status for when rich is not available."""
        elapsed = time.time() - self.start_time
        current_idx = self.tracker.status['progress']['current_video_index']
        total_videos = self.tracker.status['progress']['total_videos']
        
        return f"""
Dataset Generator v2.0
=====================
Videos: {current_idx}/{total_videos}
Elapsed: {int(elapsed)}s
Current: {self.current_video_name}
Total Images: {self.tracker.get_total_images()}
"""
    
    def run(self):
        """Main execution loop."""
        if RICH_AVAILABLE:
            self.console.print("[bold green]🚀 Initializing Dataset Generator v2.0...[/bold green]")
        else:
            print("🚀 Initializing Dataset Generator v2.0...")
        
        # Validate video files before starting
        if RICH_AVAILABLE:
            self.console.print("[yellow]🔍 Validating video files...[/yellow]")
        else:
            print("🔍 Validating video files...")
        
        missing_videos = []
        existing_videos = []
        
        for idx, video_info in enumerate(self.videos):
            if os.path.exists(video_info['path']):
                existing_videos.append(idx)
            else:
                missing_videos.append((idx, video_info['name'], video_info['path']))
        
        # Show validation results
        if RICH_AVAILABLE:
            self.console.print(f"[green]✓ Found: {len(existing_videos)} videos[/green]")
            self.console.print(f"[red]✗ Missing: {len(missing_videos)} videos[/red]")
        else:
            print(f"✓ Found: {len(existing_videos)} videos")
            print(f"✗ Missing: {len(missing_videos)} videos")
        
        # If too many videos are missing, show error and guide
        if len(existing_videos) == 0:
            error_msg = """
[bold red]❌ ERROR: No video files found![/bold red]

The configuration contains {total} videos, but none exist at the specified paths.

[bold yellow]📝 Solutions:[/bold yellow]

1. [cyan]Use the video scanner to generate config from your actual videos:[/cyan]
   cd dataset_generator_v2
   python scan_videos.py /path/to/your/videos
   mv generator_config_REAL.json generator_config.json

2. [cyan]Or manually edit generator_config.json with correct paths[/cyan]

[bold]Example config entry:[/bold]
{{
  "name": "My Video",
  "path": "/actual/path/to/video.mkv",
  "categories": {{"general": 1.0}}
}}

[dim]First missing video path:[/dim]
{first_path}
""".format(
                total=len(self.videos),
                first_path=missing_videos[0][2] if missing_videos else "N/A"
            )
            
            if RICH_AVAILABLE:
                self.console.print(error_msg)
            else:
                print(error_msg.replace('[bold red]', '').replace('[/bold red]', '')
                           .replace('[bold yellow]', '').replace('[/bold yellow]', '')
                           .replace('[cyan]', '').replace('[/cyan]', '')
                           .replace('[bold]', '').replace('[/bold]', '')
                           .replace('[dim]', '').replace('[/dim]', ''))
            
            return
        
        # If more than 50% are missing, show warning but continue
        if len(missing_videos) > len(self.videos) * 0.5:
            warning = f"""
[bold yellow]⚠️  WARNING: {len(missing_videos)} of {len(self.videos)} videos not found![/bold yellow]

Only {len(existing_videos)} videos will be processed.
Consider using scan_videos.py to generate a config from your actual video files.

Continue? Processing will start in 5 seconds... (Ctrl+C to cancel)
"""
            if RICH_AVAILABLE:
                self.console.print(warning)
            else:
                print(warning)
            
            time.sleep(5)
        else:
            # All or most videos found - show success message
            if len(missing_videos) == 0:
                success_msg = "\n[bold green]✅ All videos found! Starting dataset generation...[/bold green]\n"
            else:
                success_msg = f"\n[bold green]✅ Ready to process {len(existing_videos)} videos. Starting dataset generation...[/bold green]\n"
            
            if RICH_AVAILABLE:
                self.console.print(success_msg)
            else:
                # Strip rich formatting for plain text
                plain_msg = success_msg.replace('[bold green]', '').replace('[/bold green]', '').replace('[bold]', '').replace('[/bold]', '')
                print(plain_msg)

        
        # Create directories
        self.create_output_directories()
        
        # Check for resume
        resume_idx, resume_frame = self.tracker.get_resume_point()
        if resume_idx > 0:
            if RICH_AVAILABLE:
                self.console.print(f"[yellow]📍 Resuming from video {resume_idx}[/yellow]")
            else:
                print(f"📍 Resuming from video {resume_idx}")
        
        # Set status to running
        self.tracker.set_status("running")
        self.tracker.save()
        
        # Start keyboard listener for live controls
        self._start_keyboard_listener()
        
        # Show initial message about controls
        if RICH_AVAILABLE:
            self.console.print("\n[bold cyan]🎮 Live controls enabled:[/bold cyan]")
            self.console.print("  [SPACE] = Pause/Resume  |  [+/-] = Adjust workers  |  [q] = Quit")
            self.console.print()
        
        # Clear screen once at start for clean display (professional way)
        if RICH_AVAILABLE:
            from utils.ui_terminal import clear_and_home, hide_cursor
            clear_and_home()
            hide_cursor()
        
        # Initialize live display
        self.live_display = None
        if RICH_AVAILABLE:
            try:
                self.live_display = Live(
                    self._build_complete_layout(),
                    refresh_per_second=2,  # Update twice per second
                    console=self.console
                )
                self.live_display.start()
                if RICH_AVAILABLE:
                    self.console.print("[dim]✓ Live display mode activated[/dim]")
            except Exception as e:
                # If Live doesn't work, fall back to regular display
                self.live_display = None
                if RICH_AVAILABLE:
                    self.console.print(f"[yellow]⚠ Live display failed ({e}), using professional box GUI[/yellow]")

        
        # Log main loop start
        self.logger.info("=== STARTING GENERATOR ===")
        self.logger.info(f"Resume from video index: {resume_idx}")
        
        try:
            # Process videos
            last_processed_idx = resume_idx - 1  # Track the last successfully processed video
            for idx in range(resume_idx, len(self.videos)):
                self.logger.debug(f"--- Loop iteration {idx} / {len(self.videos)} ---")
                
                if not self.running:
                    self.logger.warning(f"Generator stopped by self.running=False at video {idx}")
                    break
                
                video_info = self.videos[idx]
                self.logger.info(f"Processing video {idx}: {video_info['name']}")
                
                # Skip if already completed
                if self.tracker.is_video_completed(idx):
                    self.logger.info(f"Video {idx} already completed - SKIPPING")
                    continue
                
                try:
                    # Process video
                    self.logger.debug(f"Calling process_video() for video {idx}")
                    result = self.process_video(idx, video_info)
                    self.logger.debug(f"process_video() returned: {result}")
                    
                    # Update progress
                    self.tracker.update_progress(completed_videos=idx + 1)
                    self.tracker.calculate_disk_usage(self.base_dir)
                    self.tracker.save()
                    
                    # Log successful completion
                    self.logger.info(f"Video {idx} completed successfully")
                    self.logger.debug(f"Moving to next video (idx={idx+1})")
                    last_processed_idx = idx  # Update last processed index
                    
                    # Update live display or print status
                    if self.live_display:
                        self.live_display.update(self._build_complete_layout())
                    elif RICH_AVAILABLE:
                        # Use professional box-drawing GUI (vsr_plusplus style)
                        from utils.ui_display import draw_dataset_generator_ui
                        draw_dataset_generator_ui(self)
                    else:
                        print(self._build_simple_status())
                except Exception as e:
                    self.logger.error(f"EXCEPTION in video {idx}: {type(e).__name__}: {e}", exc_info=True)
                    # Continue to next video instead of crashing
                    continue
            
            # Log main loop ended
            videos_processed = last_processed_idx - resume_idx + 1 if last_processed_idx >= resume_idx else 0
            self.logger.info(f"=== MAIN LOOP ENDED === (processed {videos_processed} videos, last index: {last_processed_idx})")
        except Exception as e:
            self.logger.critical(f"FATAL EXCEPTION in main loop: {type(e).__name__}: {e}", exc_info=True)
            raise
        finally:
            self.logger.info("Entering finally block - cleaning up")
            # Stop live display
            if self.live_display:
                self.live_display.stop()
            
            # Show cursor again
            if RICH_AVAILABLE:
                from utils.ui_terminal import show_cursor
                show_cursor()
            
            # Stop keyboard listener
            self.stop_input_thread = True
            if self.input_thread and self.input_thread.is_alive():
                self.input_thread.join(timeout=1)
        
        # Finalize
        self.logger.info("Setting status to 'finished'")
        self.tracker.set_status("finished")
        self.tracker.save()
        
        if RICH_AVAILABLE:
            self.console.print("\n[bold green]✅ Dataset generation complete![/bold green]")
        else:
            print("\n✅ Dataset generation complete!")

def exception_handler(exc_type, exc_value, exc_traceback):
    """Log uncaught exceptions."""
    if hasattr(sys, 'logger'):
        sys.logger.critical("UNCAUGHT EXCEPTION", exc_info=(exc_type, exc_value, exc_traceback))
    else:
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = exception_handler

def main():
    """Main entry point."""
    # Get config path
    config_path = os.path.join(
        os.path.dirname(__file__),
        'generator_config.json'
    )
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Create and run generator
    generator = DatasetGeneratorV2(config_path)
    generator.run()

if __name__ == "__main__":
    main()
