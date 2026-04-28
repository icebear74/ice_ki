"""
Size Tracking System - Track training progress per image size category

Tracks:
- Images trained per size category
- Target images per category (from distribution)
- Progress towards target
- Auto-save every 100 steps
- Integration with checkpoints
"""

import os
import json
import time
import threading
from typing import Dict, Any, Optional
from datetime import datetime


class SizeTracker:
    """
    Track training progress per image size category
    
    Args:
        save_path: Path to save size_tracking.json
        size_categories: List of size categories to track
        auto_save_interval: Auto-save every N steps (default: 100)
    """
    
    def __init__(
        self, 
        save_path: str = "/mnt/data/training/size_tracking.json",
        size_categories: Optional[list] = None,
        auto_save_interval: int = 100
    ):
        self.save_path = save_path
        self.auto_save_interval = auto_save_interval
        self.lock = threading.Lock()
        
        # Accept whatever categories the caller provides.
        # No hardcoded default — the caller must pass the actual template keys
        # from dataset_architecture.json so this tracker works for any dataset.
        if size_categories is None:
            size_categories = []
        
        self.size_categories = size_categories
        
        # Initialize tracking data
        self.data = {
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'last_step': 0,
            'total_images_trained': 0,
            'size_stats': {},
            'metadata': {
                'version': '1.0',
                'auto_save_interval': auto_save_interval,
            }
        }
        
        # Initialize size stats
        for category in size_categories:
            self.data['size_stats'][category] = {
                'images_trained': 0,
                'target_images': 0,
                'percentage_complete': 0.0,
                'last_trained_step': 0,
            }
        
        # Load existing data if available
        if os.path.exists(save_path):
            self.load()
        else:
            # Create directory if needed
            save_dir = os.path.dirname(save_path)
            # Only create directory if path has a directory component (not empty and not None)
            if save_dir and save_dir != '.':
                os.makedirs(save_dir, exist_ok=True)
            self.save()
        
        self.steps_since_save = 0
    
    def update_targets(self, size_distribution: Dict[str, float], total_target: int):
        """
        Update target images per size category based on distribution
        
        Args:
            size_distribution: Dict mapping size category to percentage (0.0-1.0)
            total_target: Total target number of images
        """
        with self.lock:
            for category in self.size_categories:
                if category in size_distribution:
                    percentage = size_distribution[category]
                    target = int(total_target * percentage)
                    
                    self.data['size_stats'][category]['target_images'] = target
                    
                    # Update percentage complete
                    trained = self.data['size_stats'][category]['images_trained']
                    if target > 0:
                        pct = (trained / target) * 100.0
                        self.data['size_stats'][category]['percentage_complete'] = pct
                    else:
                        self.data['size_stats'][category]['percentage_complete'] = 0.0
            
            self.data['last_updated'] = datetime.now().isoformat()
    
    def record_batch(self, size_category: str, batch_size: int, step: int):
        """
        Record a batch of images trained for a size category
        
        Args:
            size_category: Size category ('540', '720_169', '720')
            batch_size: Number of images in batch
            step: Current training step
        """
        with self.lock:
            if size_category not in self.data['size_stats']:
                # Initialize if new category
                self.data['size_stats'][size_category] = {
                    'images_trained': 0,
                    'target_images': 0,
                    'percentage_complete': 0.0,
                    'last_trained_step': 0,
                }
            
            # Update stats
            self.data['size_stats'][size_category]['images_trained'] += batch_size
            self.data['size_stats'][size_category]['last_trained_step'] = step
            
            # Update percentage
            trained = self.data['size_stats'][size_category]['images_trained']
            target = self.data['size_stats'][size_category]['target_images']
            
            if target > 0:
                pct = (trained / target) * 100.0
                self.data['size_stats'][size_category]['percentage_complete'] = pct
            
            # Update totals
            self.data['total_images_trained'] += batch_size
            self.data['last_step'] = step
            self.data['last_updated'] = datetime.now().isoformat()
            
            self.steps_since_save += 1
            
            # Auto-save if interval reached
            if self.steps_since_save >= self.auto_save_interval:
                self.save()
                self.steps_since_save = 0
    
    def get_stats(self, size_category: Optional[str] = None) -> Dict[str, Any]:
        """
        Get tracking statistics
        
        Args:
            size_category: If provided, return stats for that category only
            
        Returns:
            Statistics dict
        """
        with self.lock:
            if size_category:
                if size_category in self.data['size_stats']:
                    return {
                        'category': size_category,
                        **self.data['size_stats'][size_category]
                    }
                else:
                    return {}
            else:
                # Return all stats
                return self.data.copy()
    
    def get_summary(self) -> str:
        """
        Get formatted summary string
        
        Returns:
            Formatted summary
        """
        with self.lock:
            lines = []
            lines.append("="*60)
            lines.append("Size Training Progress")
            lines.append("="*60)
            lines.append(f"Total Images Trained: {self.data['total_images_trained']}")
            lines.append(f"Last Step: {self.data['last_step']}")
            lines.append("-"*60)
            
            for category in sorted(self.data['size_stats'].keys()):
                stats = self.data['size_stats'][category]
                trained = stats['images_trained']
                target = stats['target_images']
                pct = stats['percentage_complete']
                
                lines.append(f"{category:>15}: {trained:>8} / {target:>8} ({pct:>6.2f}%)")
            
            lines.append("="*60)
            
            return "\n".join(lines)
    
    def save(self) -> bool:
        """
        Save tracking data to file
        
        Returns:
            True if saved successfully
        """
        try:
            with self.lock:
                # Update timestamp
                self.data['last_updated'] = datetime.now().isoformat()
                
                # Write to file
                with open(self.save_path, 'w') as f:
                    json.dump(self.data, f, indent=2)
                
                return True
        except Exception as e:
            print(f"⚠️  Error saving size tracking: {e}")
            return False
    
    def load(self) -> bool:
        """
        Load tracking data from file
        
        Returns:
            True if loaded successfully
        """
        try:
            with self.lock:
                with open(self.save_path, 'r') as f:
                    loaded_data = json.load(f)
                
                # Merge with existing data (preserve structure)
                if 'size_stats' in loaded_data:
                    self.data['size_stats'].update(loaded_data['size_stats'])
                
                if 'total_images_trained' in loaded_data:
                    self.data['total_images_trained'] = loaded_data['total_images_trained']
                
                if 'last_step' in loaded_data:
                    self.data['last_step'] = loaded_data['last_step']
                
                if 'created_at' in loaded_data:
                    self.data['created_at'] = loaded_data['created_at']
                
                return True
        except Exception as e:
            print(f"⚠️  Error loading size tracking: {e}")
            return False
    
    def to_checkpoint_dict(self) -> Dict[str, Any]:
        """
        Export data for checkpoint saving
        
        Returns:
            Dict suitable for checkpoint
        """
        with self.lock:
            return {
                'size_tracking': self.data.copy(),
                'save_path': self.save_path,
            }
    
    def from_checkpoint_dict(self, checkpoint_data: Dict[str, Any]) -> bool:
        """
        Restore data from checkpoint
        
        Args:
            checkpoint_data: Data from checkpoint
            
        Returns:
            True if restored successfully
        """
        try:
            with self.lock:
                if 'size_tracking' in checkpoint_data:
                    # Restore data
                    self.data = checkpoint_data['size_tracking'].copy()
                    
                    # Ensure all expected size categories exist
                    for category in self.size_categories:
                        if category not in self.data['size_stats']:
                            self.data['size_stats'][category] = {
                                'images_trained': 0,
                                'target_images': 0,
                                'percentage_complete': 0.0,
                                'last_trained_step': 0,
                            }
                    
                    return True
            return False
        except Exception as e:
            print(f"⚠️  Error restoring size tracking from checkpoint: {e}")
            return False
    
    def reset(self):
        """Reset all tracking data (use with caution!)"""
        with self.lock:
            for category in self.data['size_stats']:
                self.data['size_stats'][category]['images_trained'] = 0
                self.data['size_stats'][category]['percentage_complete'] = 0.0
                self.data['size_stats'][category]['last_trained_step'] = 0
            
            self.data['total_images_trained'] = 0
            self.data['last_step'] = 0
            self.data['last_updated'] = datetime.now().isoformat()
            
            self.save()


if __name__ == "__main__":
    # Demo usage (uses example V2 template names)
    print("Size Tracker Demo\n")
    
    # Create tracker (use temp path for demo)
    tracker = SizeTracker(
        save_path="/tmp/size_tracking_demo.json",
        size_categories=['720_169', '540', '720'],
    )
    
    # Set targets based on distribution (example: equal shares)
    distribution = {
        '720_169': 0.40,
        '540':     0.20,
        '720':     0.40,
    }
    tracker.update_targets(distribution, total_target=10000)
    
    # Simulate training
    print("Simulating training...\n")
    for step in range(1, 101):
        # Simulate batches
        if step % 2 == 0:
            tracker.record_batch('540', batch_size=1, step=step)
        else:
            tracker.record_batch('720_169', batch_size=1, step=step)
    
    # Print summary
    print(tracker.get_summary())
    
    # Get specific stats
    print("\n540 stats:")
    print(tracker.get_stats('540'))
