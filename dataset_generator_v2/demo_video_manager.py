#!/usr/bin/env python3
"""
Quick demo/test of video_manager.py
Shows how to use the VideoManager class programmatically.
"""

import sys
from pathlib import Path

# Import the VideoManager class
sys.path.insert(0, str(Path(__file__).parent))
from video_manager import VideoManager


def demo():
    """Demo the VideoManager."""
    
    config_path = Path(__file__).parent / 'generator_config.json'
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return
    
    print("="*60)
    print("VIDEO MANAGER DEMO")
    print("="*60)
    
    # Load
    manager = VideoManager(str(config_path))
    manager.load()
    
    # Show statistics
    print("\n1. CURRENT STATISTICS")
    print("-" * 60)
    manager.show_statistics()
    
    # Search for Star Trek videos
    print("\n2. SEARCH: Star Trek")
    print("-" * 60)
    videos = manager.list_videos(filter_pattern="Star Trek")
    manager.print_video_list(videos, max_display=10)
    
    # Search for unassigned
    print("\n3. UNASSIGNED VIDEOS")
    print("-" * 60)
    videos = manager.list_videos(show_unassigned=True)
    if videos:
        manager.print_video_list(videos, max_display=5)
    else:
        print("No unassigned videos found")
    
    # Show videos in master category
    print("\n4. VIDEOS IN 'master' CATEGORY")
    print("-" * 60)
    videos = manager.list_videos(category='master')
    print(f"Found {len(videos)} videos in 'master' category")
    manager.print_video_list(videos, max_display=5)
    
    # Show Shrek movies
    print("\n5. SEARCH: Shrek")
    print("-" * 60)
    videos = manager.list_videos(filter_pattern="Shrek")
    manager.print_video_list(videos, max_display=None)
    
    print("\n" + "="*60)
    print("DEMO COMPLETE - No changes were made")
    print("="*60)
    print("\nTo actually use the tool, run:")
    print("  python3 video_manager.py")


if __name__ == '__main__':
    demo()
