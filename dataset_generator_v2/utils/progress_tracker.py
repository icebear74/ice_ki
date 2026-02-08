"""Progress tracking and checkpoint management for dataset generator."""

import json
import os
import time
from typing import Dict, Any, Optional
from datetime import datetime

class ProgressTracker:
    """Manages generator progress, checkpoints, and status tracking."""
    
    def __init__(self, status_file: str):
        self.status_file = status_file
        self.status = self._load_or_create_status()
    
    def _load_or_create_status(self) -> Dict[str, Any]:
        """Load existing status or create new one."""
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Create new status
        return {
            "version": "2.0",
            "started_at": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "workers": 12,
            "status": "initialized",
            "progress": {
                "total_videos": 0,
                "completed_videos": 0,
                "current_video_index": 0,
                "current_video_path": ""
            },
            "category_stats": {
                "general": {
                    "videos_processed": 0,
                    "images_created": 0,
                    "target": 80000,
                    "disk_usage_gb": 0.0
                },
                "space": {
                    "videos_processed": 0,
                    "images_created": 0,
                    "target": 55000,
                    "disk_usage_gb": 0.0
                },
                "toon": {
                    "videos_processed": 0,
                    "images_created": 0,
                    "target": 30000,
                    "disk_usage_gb": 0.0
                }
            },
            "video_checkpoints": {}
        }
    
    def save(self):
        """Save current status to file."""
        self.status["last_update"] = datetime.now().isoformat()
        os.makedirs(os.path.dirname(self.status_file), exist_ok=True)
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)
            # Flush to ensure write completes
            f.flush()
            try:
                os.fsync(f.fileno())
            except:
                pass  # Some systems don't support fsync
    
    def initialize_categories(self, category_targets: Dict[str, int]):
        """Initialize category stats from config."""
        for category, target in category_targets.items():
            if category not in self.status["category_stats"]:
                self.status["category_stats"][category] = {
                    "videos_processed": 0,
                    "images_created": 0,
                    "target": target,
                    "disk_usage_gb": 0.0
                }
            else:
                # Update target if it changed
                self.status["category_stats"][category]["target"] = target
    
    def update_progress(self, **kwargs):
        """Update progress fields."""
        for key, value in kwargs.items():
            if key in self.status["progress"]:
                self.status["progress"][key] = value
    
    def update_category_stats(self, category: str, **kwargs):
        """Update category-specific statistics."""
        if category in self.status["category_stats"]:
            for key, value in kwargs.items():
                if key in self.status["category_stats"][category]:
                    self.status["category_stats"][category][key] = value
    
    def increment_category_images(self, category: str, count: int = 1):
        """Increment image count for a category."""
        # Auto-create category stats if it doesn't exist
        if category not in self.status["category_stats"]:
            self.status["category_stats"][category] = {
                "videos_processed": 0,
                "images_created": 0,
                "target": 0,
                "disk_usage_gb": 0.0
            }
        self.status["category_stats"][category]["images_created"] += count
    
    def increment_category_videos(self, category: str):
        """Increment video count for a category."""
        # Auto-create category stats if it doesn't exist
        if category not in self.status["category_stats"]:
            self.status["category_stats"][category] = {
                "videos_processed": 0,
                "images_created": 0,
                "target": 0,
                "disk_usage_gb": 0.0
            }
        self.status["category_stats"][category]["videos_processed"] += 1
    
    def update_video_checkpoint(self, video_index: int, status: str, **kwargs):
        """Update checkpoint for a specific video."""
        checkpoint_data = {"status": status}
        checkpoint_data.update(kwargs)
        self.status["video_checkpoints"][str(video_index)] = checkpoint_data
    
    def get_video_checkpoint(self, video_index: int) -> Optional[Dict]:
        """Get checkpoint data for a video."""
        return self.status["video_checkpoints"].get(str(video_index))
    
    def is_video_completed(self, video_index: int) -> bool:
        """Check if a video is already completed."""
        checkpoint = self.get_video_checkpoint(video_index)
        return checkpoint and checkpoint.get("status") == "completed"
    
    def get_resume_point(self) -> tuple:
        """Get the point to resume from (video_index, frame_index)."""
        current_idx = self.status["progress"]["current_video_index"]
        checkpoint = self.get_video_checkpoint(current_idx)
        
        if checkpoint and checkpoint.get("status") == "in_progress":
            return current_idx, checkpoint.get("last_frame_idx", 0)
        
        return current_idx, 0
    
    def set_workers(self, workers: int):
        """Update worker count."""
        self.status["workers"] = workers
        self.save()
    
    def set_status(self, status: str):
        """Update overall status."""
        self.status["status"] = status
    
    def calculate_disk_usage(self, base_path: str):
        """Calculate disk usage for each category."""
        from .format_definitions import CATEGORY_PATHS
        
        for category, rel_path in CATEGORY_PATHS.items():
            category_path = os.path.join(base_path, rel_path)
            if os.path.exists(category_path):
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(category_path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        try:
                            total_size += os.path.getsize(filepath)
                        except:
                            pass
                
                size_gb = total_size / (1024**3)
                self.update_category_stats(category, disk_usage_gb=round(size_gb, 2))
    
    def get_category_progress_percent(self, category: str) -> float:
        """Get progress percentage for a category."""
        stats = self.status["category_stats"].get(category, {})
        images = stats.get("images_created", 0)
        target = stats.get("target", 1)
        return (images / target) * 100 if target > 0 else 0
    
    def get_total_images(self) -> int:
        """Get total images created across all categories."""
        return sum(
            stats.get("images_created", 0) 
            for stats in self.status["category_stats"].values()
        )
    
    def get_total_target(self) -> int:
        """Get total target images across all categories."""
        return sum(
            stats.get("target", 0) 
            for stats in self.status["category_stats"].values()
        )
