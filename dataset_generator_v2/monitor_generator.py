#!/usr/bin/env python3
"""
Live monitoring tool for the dataset generator.
Reads status file and displays real-time progress.
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    from rich.layout import Layout
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: 'rich' library not available. Install with: pip install rich")

class GeneratorMonitor:
    """Monitor for dataset generator progress."""
    
    def __init__(self, status_file: str):
        self.status_file = status_file
        if RICH_AVAILABLE:
            self.console = Console()
    
    def load_status(self):
        """Load status from file."""
        if not os.path.exists(self.status_file):
            return None
        
        try:
            with open(self.status_file, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def build_display(self, status):
        """Build the monitoring display."""
        if not RICH_AVAILABLE:
            return self._build_simple_display(status)
        
        # Parse timestamps
        started = datetime.fromisoformat(status['started_at'])
        last_update = datetime.fromisoformat(status['last_update'])
        elapsed = datetime.now() - started
        
        # Overall progress
        progress_data = status['progress']
        total_videos = progress_data['total_videos']
        completed_videos = progress_data['completed_videos']
        current_idx = progress_data['current_video_index']
        
        # Calculate ETA
        if completed_videos > 0:
            avg_time = elapsed.total_seconds() / completed_videos
            remaining = total_videos - completed_videos
            eta = timedelta(seconds=int(avg_time * remaining))
        else:
            eta = "Calculating..."
        
        # Header
        header = Panel(
            f"[bold cyan]DATASET GENERATOR MONITOR[/bold cyan]\n"
            f"Status: [{'green' if status['status'] == 'running' else 'yellow'}]{status['status'].upper()}[/]",
            style="bold white on blue"
        )
        
        # Overall section
        overall = f"""[bold]📊 OVERALL PROGRESS[/bold]
├─ Total Videos: {total_videos}
├─ Completed: {completed_videos} ({completed_videos/total_videos*100 if total_videos > 0 else 0:.1f}%)
├─ Current Index: {current_idx}
├─ Remaining: {total_videos - completed_videos} videos
├─ Elapsed: {str(elapsed).split('.')[0]}
├─ ETA: {eta if isinstance(eta, str) else str(eta).split('.')[0]}
├─ Workers: {status['workers']}
└─ Last Update: {last_update.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Current video info
        current_video = f"""[bold]🎬 CURRENT VIDEO[/bold]
├─ Index: {current_idx}
├─ Path: {progress_data['current_video_path'][-70:] if progress_data['current_video_path'] else 'N/A'}
"""
        
        # Check for current video checkpoint
        checkpoint = status['video_checkpoints'].get(str(current_idx))
        if checkpoint and checkpoint.get('status') == 'in_progress':
            done = checkpoint.get('extractions_done', 0)
            target = checkpoint.get('extractions_target', 1)
            pct = (done / target * 100) if target > 0 else 0
            current_video += f"├─ Extractions: {done} / {target} ({pct:.1f}%)\n"
            current_video += f"└─ Last Frame: {checkpoint.get('last_frame_idx', 0)}\n"
        else:
            current_video += "└─ Status: Waiting or completed\n"
        
        # Category table
        table = Table(title="📦 CATEGORY STATISTICS", show_header=True, header_style="bold magenta")
        table.add_column("Category", style="cyan", width=12)
        table.add_column("Videos", justify="right", width=10)
        table.add_column("Images", justify="right", width=12)
        table.add_column("Target", justify="right", width=12)
        table.add_column("Progress", width=20)
        table.add_column("Disk (GB)", justify="right", width=12)
        
        category_stats = status['category_stats']
        for cat_name in ['general', 'space', 'toon']:
            stats = category_stats[cat_name]
            videos = stats['videos_processed']
            images = stats['images_created']
            target = stats['target']
            disk = stats['disk_usage_gb']
            progress = (images / target * 100) if target > 0 else 0
            
            # Progress bar
            filled = int(progress / 5)
            bar = "█" * filled + "░" * (20 - filled)
            
            table.add_row(
                cat_name.upper(),
                str(videos),
                f"{images:,}",
                f"{target:,}",
                f"{bar} {progress:.1f}%",
                f"{disk:.2f}"
            )
        
        # Totals
        total_images = sum(s['images_created'] for s in category_stats.values())
        total_target = sum(s['target'] for s in category_stats.values())
        total_disk = sum(s['disk_usage_gb'] for s in category_stats.values())
        
        table.add_row(
            "[bold]TOTAL[/bold]",
            "",
            f"[bold]{total_images:,}[/bold]",
            f"[bold]{total_target:,}[/bold]",
            f"[bold]{(total_images/total_target*100 if total_target > 0 else 0):.1f}%[/bold]",
            f"[bold]{total_disk:.2f}[/bold]"
        )
        
        # Disk usage details
        disk_info = f"""[bold]💾 DISK USAGE BREAKDOWN[/bold]
├─ GENERAL: {category_stats['general']['disk_usage_gb']:.2f} GB
├─ SPACE: {category_stats['space']['disk_usage_gb']:.2f} GB
├─ TOON: {category_stats['toon']['disk_usage_gb']:.2f} GB
└─ Total: {total_disk:.2f} GB
"""
        
        # Video checkpoints summary
        checkpoints = status['video_checkpoints']
        completed_count = sum(1 for c in checkpoints.values() if isinstance(c, str) and c == "completed" or isinstance(c, dict) and c.get('status') == 'completed')
        in_progress_count = sum(1 for c in checkpoints.values() if isinstance(c, dict) and c.get('status') == 'in_progress')
        
        checkpoint_info = f"""[bold]🔖 CHECKPOINT SUMMARY[/bold]
├─ Total Checkpoints: {len(checkpoints)}
├─ Completed: {completed_count}
├─ In Progress: {in_progress_count}
└─ Status File: {self.status_file[-50:]}
"""
        
        return header, overall, current_video, table, disk_info, checkpoint_info
    
    def _build_simple_display(self, status):
        """Build simple text display."""
        progress = status['progress']
        output = f"""
Dataset Generator Monitor
=========================
Status: {status['status']}
Videos: {progress['completed_videos']}/{progress['total_videos']}
Workers: {status['workers']}

Category Stats:
"""
        for cat, stats in status['category_stats'].items():
            output += f"  {cat}: {stats['images_created']}/{stats['target']} images, {stats['disk_usage_gb']:.2f} GB\n"
        
        return output
    
    def run(self):
        """Run the monitor."""
        if not RICH_AVAILABLE:
            # Simple polling loop
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                status = self.load_status()
                if status:
                    print(self._build_simple_display(status))
                else:
                    print("Waiting for generator to start...")
                time.sleep(2)
        else:
            # Rich live display
            try:
                with Live(console=self.console, refresh_per_second=0.5) as live:
                    while True:
                        status = self.load_status()
                        
                        if status:
                            header, overall, current, table, disk, checkpoint = self.build_display(status)
                            
                            # Build layout
                            layout = Layout()
                            layout.split_column(
                                Layout(header, size=3),
                                Layout(overall, size=10),
                                Layout(current, size=6),
                                Layout(table, size=10),
                                Layout(disk, size=6),
                                Layout(checkpoint, size=6)
                            )
                            
                            live.update(layout)
                        else:
                            live.update(Panel(
                                "[yellow]Waiting for generator to start...\n"
                                f"Looking for: {self.status_file}[/yellow]",
                                title="Monitor Status"
                            ))
                        
                        time.sleep(2)
                        
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Monitor stopped.[/yellow]")

def main():
    """Main entry point."""
    # Default status file location
    status_file = "/mnt/data/training/dataset/.generator_status.json"
    
    # Allow override via command line
    if len(sys.argv) > 1:
        status_file = sys.argv[1]
    
    if RICH_AVAILABLE:
        console = Console()
        console.print(f"[bold green]Starting monitor...[/bold green]")
        console.print(f"Status file: [cyan]{status_file}[/cyan]")
        console.print("[dim]Press Ctrl+C to exit[/dim]\n")
    else:
        print(f"Starting monitor...")
        print(f"Status file: {status_file}")
        print("Press Ctrl+C to exit\n")
    
    monitor = GeneratorMonitor(status_file)
    monitor.run()

if __name__ == "__main__":
    main()
