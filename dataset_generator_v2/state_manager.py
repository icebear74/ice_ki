#!/usr/bin/env python3
"""
State Manager for Dataset Generator V2
Handles:
- Video metadata caching
- Category and video distribution
- Progress tracking
- Resume capability
"""

import os
import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class StateManager:
    """
    Complete state management for dataset generation:
    - Video metadata caching (scan once, reuse forever)
    - Category-based weighted distribution
    - Video duration-based distribution within categories
    - Progress tracking per video
    - Resume capability
    """
    
    def __init__(self, config: dict, state_file: str = "generation_state.json"):
        """
        Initialize state manager
        
        Args:
            config: Generator configuration dict
            state_file: Path to state JSON file
        """
        self.config = config
        self.state_file = state_file
        self.state = self._load_or_create_state()
        
    def _load_or_create_state(self) -> dict:
        """Load existing state or create new one"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                # Validate config hasn't changed
                current_hash = self._compute_config_hash()
                if state.get('config_hash') == current_hash:
                    logger.info(f"✅ Resuming from existing state: {state.get('generation_id')}")
                    return state
                else:
                    logger.warning("⚠️  Config changed, creating new state")
            except Exception as e:
                logger.error(f"Error loading state: {e}")
        
        # Create new state
        return self._create_new_state()
    
    def _create_new_state(self) -> dict:
        """Create a new state structure"""
        generation_id = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        state = {
            "config_hash": self._compute_config_hash(),
            "generation_id": generation_id,
            "started_at": datetime.now().isoformat(),
            "status": "initializing",
            
            "video_metadata": {},
            "category_distribution": {},
            
            "progress": {
                "total_patches": self.config['processing']['total_patches'],
                "completed_patches": 0,
                "failed_patches": 0,
                "remaining_patches": self.config['processing']['total_patches'],
                "percentage": 0.0
            },
            
            "errors": []
        }
        
        logger.info(f"✨ Created new generation: {generation_id}")
        return state
    
    def _compute_config_hash(self) -> str:
        """
        Compute hash of configuration for change detection
        Note: Using MD5 for simple change detection (not security)
        """
        # Hash only the relevant parts
        config_str = json.dumps({
            'source': self.config.get('source'),
            'processing': self.config.get('processing'),
            'output_patches': self.config.get('output_patches')
        }, sort_keys=True)
        # SHA256 for better collision resistance
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]
    
    def scan_videos(self):
        """
        Scan all videos and cache metadata
        Only rescans if file has changed (mtime/size check)
        """
        logger.info("🔍 Scanning videos for metadata...")
        
        categories = self.config['source']['categories']
        
        for category_name, category_config in categories.items():
            video_dir = category_config['video_dir']
            extensions = category_config['extensions']
            
            if not os.path.exists(video_dir):
                logger.warning(f"Category '{category_name}' directory not found: {video_dir}")
                continue
            
            # Find all videos
            video_files = []
            for ext in extensions:
                video_files.extend(Path(video_dir).rglob(f'*{ext}'))
            
            logger.info(f"  {category_name}: Found {len(video_files)} videos")
            
            for video_path in video_files:
                video_path_str = str(video_path)
                
                # Check if we need to rescan
                try:
                    stat = os.stat(video_path)
                    file_size = stat.st_size
                    last_modified = datetime.fromtimestamp(stat.st_mtime).isoformat()
                    
                    cached = self.state['video_metadata'].get(video_path_str)
                    if cached and cached.get('file_size') == file_size and cached.get('last_modified') == last_modified:
                        # Already scanned, skip
                        continue
                    
                    # Get video metadata using ffprobe
                    metadata = self._get_video_metadata(video_path_str)
                    if metadata:
                        metadata['category'] = category_name
                        metadata['file_size'] = file_size
                        metadata['last_modified'] = last_modified
                        self.state['video_metadata'][video_path_str] = metadata
                        
                except Exception as e:
                    logger.error(f"Error scanning {video_path}: {e}")
                    self.state['errors'].append({
                        'timestamp': datetime.now().isoformat(),
                        'type': 'scan_error',
                        'video': video_path_str,
                        'error': str(e)
                    })
        
        logger.info(f"✅ Total videos in metadata cache: {len(self.state['video_metadata'])}")
        self.save()
    
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
            
            # Configurable timeout (default 60s for large files)
            timeout = self.config.get('ffprobe_timeout', 60)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode != 0:
                return None
            
            data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                return None
            
            # Extract metadata
            duration = float(data.get('format', {}).get('duration', 0))
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            
            # Parse fps
            fps_str = video_stream.get('r_frame_rate', '0/1')
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                fps = num / den if den > 0 else 0
            else:
                fps = float(fps_str)
            
            return {
                'duration': duration,
                'resolution': [width, height],
                'fps': fps
            }
            
        except Exception as e:
            logger.error(f"ffprobe error for {video_path}: {e}")
            return None
    
    def calculate_distribution(self):
        """
        Calculate patch distribution:
        1. By category weight (e.g., master=25%, universal=75%)
        2. Within category by video duration
        """
        logger.info("📊 Calculating patch distribution...")
        
        total_patches = self.config['processing']['total_patches']
        category_weights = self.config['source']['category_weights']
        
        # Initialize category distribution
        for category_name, weight in category_weights.items():
            category_patches = int(total_patches * weight)
            
            # Get all videos in this category
            category_videos = {
                path: meta for path, meta in self.state['video_metadata'].items()
                if meta.get('category') == category_name
            }
            
            if not category_videos:
                logger.warning(f"No videos found for category '{category_name}'")
                continue
            
            # Calculate total duration
            total_duration = sum(meta['duration'] for meta in category_videos.values())
            
            if total_duration == 0:
                logger.warning(f"Total duration is 0 for category '{category_name}'")
                continue
            
            # Initialize category in state
            self.state['category_distribution'][category_name] = {
                'weight': weight,
                'total_patches': category_patches,
                'total_duration': total_duration,
                'video_count': len(category_videos),
                'videos': {}
            }
            
            # Distribute patches to videos by duration
            for video_path, meta in category_videos.items():
                duration = meta['duration']
                weight_in_category = duration / total_duration
                allocated_patches = int(category_patches * weight_in_category)
                
                self.state['category_distribution'][category_name]['videos'][video_path] = {
                    'duration': duration,
                    'weight_in_category': weight_in_category,
                    'allocated_patches': allocated_patches,
                    'completed_patches': 0,
                    'status': 'pending',
                    'last_timestamp': 0.0
                }
            
            logger.info(f"  {category_name}: {category_patches} patches across {len(category_videos)} videos")
        
        self.save()
    
    def get_next_video_task(self) -> Optional[Tuple[str, dict, int]]:
        """
        Get next video that needs patches
        Returns: (video_path, video_info, remaining_patches) or None
        """
        for category_name, category_data in self.state['category_distribution'].items():
            for video_path, video_info in category_data['videos'].items():
                if video_info['status'] in ['pending', 'in_progress']:
                    remaining = video_info['allocated_patches'] - video_info['completed_patches']
                    if remaining > 0:
                        # Mark as in_progress
                        video_info['status'] = 'in_progress'
                        return (video_path, video_info, remaining)
        
        return None
    
    def update_video_progress(self, video_path: str, patches_created: int, timestamp: float):
        """
        Update progress after creating patches
        
        Args:
            video_path: Path to video
            patches_created: Number of patches created in this iteration
            timestamp: Last processed timestamp in video
        """
        # Find video in distribution
        for category_data in self.state['category_distribution'].values():
            if video_path in category_data['videos']:
                video_info = category_data['videos'][video_path]
                video_info['completed_patches'] += patches_created
                video_info['last_timestamp'] = timestamp
                
                # Mark as complete if done
                if video_info['completed_patches'] >= video_info['allocated_patches']:
                    video_info['status'] = 'complete'
                
                # Update overall progress
                self.state['progress']['completed_patches'] += patches_created
                total = self.state['progress']['total_patches']
                completed = self.state['progress']['completed_patches']
                self.state['progress']['remaining_patches'] = total - completed
                self.state['progress']['percentage'] = (completed / total * 100) if total > 0 else 0
                
                break
    
    def save(self):
        """Save state to JSON"""
        try:
            self.state['status'] = 'in_progress'
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def force_rescan(self):
        """
        Force a complete rescan of all configured source directories.
        Clears the video metadata cache and category distribution, then
        rescans from scratch. Use this when source directories have changed
        (videos added, removed, or moved).
        """
        logger.info("🔄 Forcing complete rescan of source directories...")
        self.state['video_metadata'] = {}
        self.state['category_distribution'] = {}
        self.scan_videos()
        logger.info(f"✅ Force rescan complete: {len(self.state['video_metadata'])} videos found")
        self.save()

    def mark_complete(self):
        """Mark generation as complete"""
        self.state['status'] = 'complete'
        self.state['completed_at'] = datetime.now().isoformat()
        self.save()
        logger.info("✅ Generation complete!")
    
    def get_progress_summary(self) -> str:
        """Get human-readable progress summary"""
        progress = self.state['progress']
        completed = progress['completed_patches']
        total = progress['total_patches']
        percentage = progress['percentage']
        
        lines = [
            f"Progress: {completed:,} / {total:,} patches ({percentage:.2f}%)",
            f"Failed: {progress['failed_patches']}"
        ]
        
        # Per-category progress
        for category_name, category_data in self.state['category_distribution'].items():
            total_cat = category_data['total_patches']
            completed_cat = sum(v['completed_patches'] for v in category_data['videos'].values())
            lines.append(f"  {category_name}: {completed_cat:,} / {total_cat:,}")
        
        return '\n'.join(lines)
