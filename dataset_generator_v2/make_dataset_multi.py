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
    
    def __init__(self, config_path: str):
        """Initialize the generator."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.settings = self.config['base_settings']
        self.videos = self.config['videos']
        self.format_config = self.config.get('format_config', {})
        
        # Initialize paths
        self.base_dir = self.settings['output_base_dir']
        self.temp_dir = self.settings['temp_dir']
        self.status_file = self.settings['status_file']
        
        # Initialize progress tracker
        self.tracker = ProgressTracker(self.status_file)
        self.tracker.update_progress(total_videos=len(self.videos))
        
        # Runtime state
        self.workers = self.settings['max_workers']
        self.running = True
        self.paused = False
        self.last_update_time = time.time()
        self.update_interval = 0.5  # Update GUI every 0.5 seconds
        
        # Rich console
        if RICH_AVAILABLE:
            self.console = Console()
        
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
        
        # VSR++ compatible: Use 'LR' for 5-frame, 'LR_7frames' for extended
        lr_dir_name = 'LR' if lr_frames == 5 else 'LR_7frames'
        
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
        
        For VSR++ compatibility:
            - Patches/GT/ and Patches/LR/ (5-frame, for training)
            - Patches/LR_7frames/ (7-frame, optional extended)
            - Val/GT/ and Val/LR/ (validation)
        """
        for category in self.config.get('category_targets', {}).keys():
            # Get format distribution for this category
            category_formats = self.format_config.get(category, {})
            
            if not category_formats:
                # Fallback to hard-coded distribution
                category_formats = CATEGORY_FORMAT_DISTRIBUTION.get(category, {'small_540': 1.0})
            
            for format_name in category_formats.keys():
                # Create directories for 5-frame LR (VSR++ compatible)
                dirs_5 = self.get_output_dirs_for_category_format(category, format_name, lr_frames=5)
                for dir_path in dirs_5.values():
                    os.makedirs(dir_path, exist_ok=True)
                
                # Create directories for 7-frame LR (optional extended)
                dirs_7 = self.get_output_dirs_for_category_format(category, format_name, lr_frames=7)
                for dir_path in dirs_7.values():
                    os.makedirs(dir_path, exist_ok=True)
        
        # Create temp directory
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def extract_7_frames(self, video_path: str, timestamp: float, thread_id: str) -> Optional[List]:
        """Extract 7 frames centered at timestamp using HDR tonemap."""
        thread_temp = os.path.join(self.temp_dir, f"extract_{thread_id}")
        os.makedirs(thread_temp, exist_ok=True)
        
        try:
            # HDR tonemap filter from original
            tonemap_vf = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=mobius,zscale=t=bt709:m=bt709,format=yuv420p,scale=1920:1080:flags=lanczos"
            
            # Extract 7 frames
            cmd = [
                'nice', '-n', '19',
                'ffmpeg', '-y', '-threads', '1',
                '-ss', str(round(timestamp, 3)),
                '-i', video_path,
                '-vf', tonemap_vf,
                '-vframes', '7',
                os.path.join(thread_temp, 'f_%d.png')
            ]
            
            subprocess.run(cmd, capture_output=True, check=False, timeout=30)
            
            # Load frames
            frames = []
            for i in range(1, 8):
                frame_path = os.path.join(thread_temp, f"f_{i}.png")
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
            if os.path.exists(thread_temp):
                shutil.rmtree(thread_temp, ignore_errors=True)
    
    def validate_scene_stability(self, frames: List) -> bool:
        """Check if scene is stable (not a cut/transition)."""
        # Check difference between first and last frame
        gray_first = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        gray_last = cv2.cvtColor(frames[6], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_first, gray_last).mean()
        
        return diff < self.settings['scene_diff_threshold']
    
    def create_lr_stack(self, frames: List, lr_size: Tuple[int, int], crop_y: int, crop_x: int, crop_h: int, crop_w: int) -> any:
        """Create vertically stacked LR frames."""
        lr_frames = []
        for frame in frames:
            # Crop from the frame
            cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            # Resize to LR size
            resized = cv2.resize(cropped, lr_size, interpolation=cv2.INTER_LANCZOS4)
            lr_frames.append(resized)
        
        # Stack vertically
        return cv2.vconcat(lr_frames)
    
    def save_patches(self, frames: List, category: str, format_name: str, 
                     video_name: str, frame_idx: int) -> bool:
        """
        Save GT and LR patches for a specific category and format.
        
        VSR++ Training expects:
            - GT: Single ground truth frame (e.g., 540×540)
            - LR: 5-frame stack vertically (e.g., 180×900)
            - Optional: 7-frame stack in separate directory
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get format specifications
            format_spec = FORMATS[format_name]
            gt_h, gt_w = format_spec['gt_size']
            lr_h, lr_w = format_spec['lr_size']
            suffix = format_spec['suffix']
            
            # Get output directories (5-frame LR for VSR++ compatibility)
            dirs_5 = self.get_output_dirs_for_category_format(category, format_name, lr_frames=5)
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
            gt_path = os.path.join(dirs_5['gt'], filename)
            cv2.imwrite(gt_path, gt_frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            
            # Save 5-frame LR (VSR++ compatible: frames 1-5, indices 1:6)
            # This goes into 'LR' directory for VSR++ training
            lr_5 = self.create_lr_stack(frames[1:6], (lr_w, lr_h), crop_y, crop_x, gt_h, gt_w)
            lr5_path = os.path.join(dirs_5['lr'], filename)
            cv2.imwrite(lr5_path, lr_5, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            
            # Save 7-frame LR (optional extended version)
            # This goes into 'LR_7frames' directory for future use
            lr_7 = self.create_lr_stack(frames[0:7], (lr_w, lr_h), crop_y, crop_x, gt_h, gt_w)
            lr7_path = os.path.join(dirs_7['lr'], filename)
            cv2.imwrite(lr7_path, lr_7, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            
            # Verify files were created
            if os.path.exists(gt_path) and os.path.exists(lr5_path) and os.path.exists(lr7_path):
                return True
            
            return False
            
        except Exception as e:
            return False
    
    def extract_with_retry(self, video_path: str, video_name: str, 
                          categories: Dict[str, float], frame_idx: int, 
                          duration: float) -> Tuple[bool, int]:
        """Extract patches with retry logic."""
        timestamp = (frame_idx * duration / self.settings['base_frame_limit']) % duration
        thread_id = f"{random.randint(1000, 9999)}_{int(time.time()*1000) % 10000}"
        
        for attempt in range(self.settings['max_retry_attempts']):
            # Extract 7 frames
            frames = self.extract_7_frames(video_path, timestamp, thread_id)
            
            if frames is None:
                timestamp = (timestamp + self.settings['retry_skip_seconds']) % duration
                continue
            
            # Validate scene stability
            if not self.validate_scene_stability(frames):
                timestamp = (timestamp + self.settings['retry_skip_seconds']) % duration
                continue
            
            # Save patches for each category this video belongs to
            all_success = True
            for category, weight in categories.items():
                # Select format for this category
                format_name = self.select_format_for_category(category)
                
                # Save with DIFFERENT random crop per category
                success = self.save_patches(frames, category, format_name, 
                                          video_name, frame_idx)
                
                if success:
                    self.tracker.increment_category_images(category, 1)
                else:
                    all_success = False
            
            if all_success:
                return True, attempt + 1
            
            timestamp = (timestamp + self.settings['retry_skip_seconds']) % duration
        
        return False, self.settings['max_retry_attempts']
    
    def process_video(self, video_idx: int, video_info: Dict) -> Dict:
        """Process a single video and generate all patches."""
        video_path = video_info['path']
        video_name = video_info['name']
        categories = video_info['categories']
        
        # Check if video exists
        if not os.path.exists(video_path):
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
        
        # Mark video as completed
        self.tracker.update_video_checkpoint(video_idx, "completed")
        for category in categories.keys():
            self.tracker.increment_category_videos(category)
        
        self.tracker.save()
        
        return {
            'success': True,
            'video_name': video_name,
            'extractions': total_extractions,
            'success_count': success_count
        }
    
    def build_gui_layout(self) -> tuple:
        """Build the beautiful GUI layout using rich."""
        if not RICH_AVAILABLE:
            return self._build_simple_status(), None, None, None
        
        # Calculate statistics
        elapsed = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        current_idx = self.tracker.status['progress']['current_video_index']
        total_videos = self.tracker.status['progress']['total_videos']
        completed_videos = self.tracker.status['progress']['completed_videos']
        
        # ETA calculation
        if completed_videos > 0:
            avg_time_per_video = elapsed / completed_videos
            remaining_videos = total_videos - completed_videos
            eta_seconds = avg_time_per_video * remaining_videos
            eta_str = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta_str = "Calculating..."
        
        # Extraction speed
        if elapsed > 0:
            extractions_per_sec = self.extractions_count / elapsed
            speed_str = f"{extractions_per_sec:.1f} extractions/sec"
        else:
            speed_str = "Calculating..."
        
        # Build header
        header = Panel(
            "[bold cyan]DATASET GENERATOR v2.0 - MULTI-CATEGORY[/bold cyan]",
            style="bold white on blue"
        )
        
        # Overall progress section
        completion_pct = (completed_videos/total_videos*100) if total_videos > 0 else 0
        overall = f"""[bold]📊 OVERALL PROGRESS[/bold]
├─ Total Videos: {total_videos}
├─ Completed: {completed_videos} ({completion_pct:.1f}%)
├─ Current: {self.current_video_name[:50]}
├─ Remaining: {total_videos - completed_videos} videos
├─ Elapsed: {elapsed_str}
├─ ETA: {eta_str}
├─ Speed: {speed_str}
└─ Workers: {self.workers}
"""
        
        # Current video section
        checkpoint = self.tracker.get_video_checkpoint(current_idx)
        if checkpoint and checkpoint.get('status') == 'in_progress':
            done = checkpoint.get('extractions_done', 0)
            target = checkpoint.get('extractions_target', 1)
            progress_pct = (done / target * 100) if target > 0 else 0
            success_rate = (self.success_count / self.extractions_count * 100) if self.extractions_count > 0 else 0
            
            # Create a visual progress bar
            bar_width = 30
            filled = int(progress_pct / 100 * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            
            current_video = f"""[bold]🎬 CURRENT VIDEO[/bold]
├─ Name: {self.current_video_name[:60]}
├─ Path: ...{self.tracker.status['progress']['current_video_path'][-50:]}
├─ Extractions: {done} / {target} ({progress_pct:.1f}%)
├─ Progress: [{bar}] {progress_pct:.0f}%
├─ Success Rate: {success_rate:.1f}%
├─ Total Extractions: {self.extractions_count}
├─ Successful: {self.success_count}
└─ Status: {'[green]Running[/green]' if self.running else '[red]Stopped[/red]'}
"""
        else:
            current_video = "[bold]🎬 CURRENT VIDEO[/bold]\n└─ Initializing..."
        
        # Category progress table
        table = Table(title="📦 CATEGORY PROGRESS", show_header=True, header_style="bold magenta")
        table.add_column("Category", style="cyan", width=12)
        table.add_column("Videos", justify="right")
        table.add_column("Images", justify="right")
        table.add_column("Target", justify="right")
        table.add_column("Progress", width=12)
        
        for cat_name in self.config.get('category_targets', {}).keys():
            stats = self.tracker.status['category_stats'][cat_name]
            videos = stats['videos_processed']
            images = stats['images_created']
            target = stats['target']
            progress = (images / target * 100) if target > 0 else 0
            
            # Progress bar
            filled = int(progress / 10)
            bar = "█" * filled + "░" * (10 - filled)
            
            table.add_row(
                cat_name.upper(),
                str(videos),
                f"{images:,}",
                f"{target:,}",
                f"{bar} {progress:.1f}%"
            )
        
        # Disk usage
        total_disk = sum(s['disk_usage_gb'] for s in self.tracker.status['category_stats'].values())
        disk_lines = ["[bold]💾 DISK USAGE[/bold]"]
        categories = list(self.config.get('category_targets', {}).keys())
        for i, cat_name in enumerate(categories):
            usage = self.tracker.status['category_stats'][cat_name]['disk_usage_gb']
            prefix = "├─"
            disk_lines.append(f"{prefix} {cat_name.upper()}: {usage:.1f} GB")
        disk_lines.append(f"└─ Total: {total_disk:.1f} GB")
        disk_usage = "\n".join(disk_lines)
        
        # Controls with live status
        pause_status = "[yellow]PAUSED[/yellow]" if self.paused else "[green]RUNNING[/green]"
        controls = f"""[bold]⚙️  LIVE CONTROLS[/bold]
├─ Status: {pause_status}
├─ Workers: {self.workers} cores
├─ [bold cyan][SPACE][/bold cyan] Pause/Resume
├─ [bold cyan][+][/bold cyan] Increase workers (current: {self.workers})
├─ [bold cyan][-][/bold cyan] Decrease workers (current: {self.workers})
├─ [bold cyan][Ctrl+C][/bold cyan] Save & Exit
└─ [bold cyan][q][/bold cyan] Quick quit
"""
        
        # Combine all sections
        output = f"\n{overall}\n{current_video}\n"
        
        return output, table, disk_usage, controls
    
    def _should_update_display(self) -> bool:
        """Check if enough time has passed to update the display."""
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            return True
        return False
    
    def _build_complete_layout(self):
        """Build complete layout for live display."""
        if not RICH_AVAILABLE:
            return self._build_simple_status()
        
        output, table, disk, controls = self.build_gui_layout()
        
        # Combine everything into a single renderable
        from rich.console import Group
        from rich.panel import Panel
        
        header = Panel(
            "[bold cyan]DATASET GENERATOR v2.0 - LIVE MODE[/bold cyan]",
            style="bold white on blue"
        )
        
        return Group(
            header,
            output,
            table,
            disk,
            controls
        )
    
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
            except:
                # If Live doesn't work, fall back to regular display
                self.live_display = None
        
        try:
            # Process videos
            for idx in range(resume_idx, len(self.videos)):
                if not self.running:
                    break
                
                video_info = self.videos[idx]
                
                # Skip if already completed
                if self.tracker.is_video_completed(idx):
                    continue
                
                # Process video
                result = self.process_video(idx, video_info)
                
                # Update progress
                self.tracker.update_progress(completed_videos=idx + 1)
                self.tracker.calculate_disk_usage(self.base_dir)
                self.tracker.save()
                
                # Update live display or print status
                if self.live_display:
                    self.live_display.update(self._build_complete_layout())
                elif RICH_AVAILABLE:
                    output, table, disk, controls = self.build_gui_layout()
                    self.console.clear()
                    self.console.print(output)
                    self.console.print(table)
                    self.console.print(disk)
                    self.console.print(controls)
                else:
                    print(self._build_simple_status())
        
        finally:
            # Stop live display
            if self.live_display:
                self.live_display.stop()
            
            # Stop keyboard listener
            self.stop_input_thread = True
            if self.input_thread and self.input_thread.is_alive():
                self.input_thread.join(timeout=1)
        
        # Finalize
        self.tracker.set_status("finished")
        self.tracker.save()
        
        if RICH_AVAILABLE:
            self.console.print("\n[bold green]✅ Dataset generation complete![/bold green]")
        else:
            print("\n✅ Dataset generation complete!")

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
