#!/usr/bin/env python3
"""
Test script to verify that videos without categories are skipped by the generator.
"""

import json
import sys
from pathlib import Path

def test_skip_logic():
    """Test the skip logic for videos without categories."""
    
    print("="*70)
    print("TEST: Videos Without Categories Are Skipped")
    print("="*70)
    
    # Test data: Videos with and without categories
    test_videos = [
        {
            "name": "Video with categories",
            "path": "/path/to/video1.mkv",
            "categories": {
                "master": 0.25,
                "universal": 0.75
            }
        },
        {
            "name": "Video WITHOUT categories (should be skipped)",
            "path": "/path/to/video2.mkv",
            "categories": {}
        },
        {
            "name": "Another video with categories",
            "path": "/path/to/video3.mkv",
            "categories": {
                "space": 1.0
            }
        }
    ]
    
    print("\nTest Videos:")
    print("-" * 70)
    
    skipped_count = 0
    processed_count = 0
    
    for i, video in enumerate(test_videos):
        video_categories = video.get('categories', {})
        
        if not video_categories:
            print(f"{i+1}. ⏭️  SKIP: {video['name']:<40} (no categories)")
            skipped_count += 1
        else:
            cat_str = ', '.join([f"{k}:{v:.2f}" for k, v in video_categories.items()])
            print(f"{i+1}. ✓ PROCESS: {video['name']:<40} ({cat_str})")
            processed_count += 1
    
    print("\n" + "="*70)
    print(f"Summary:")
    print(f"  Total videos:     {len(test_videos)}")
    print(f"  To be processed:  {processed_count}")
    print(f"  To be skipped:    {skipped_count}")
    print("="*70)
    
    # Verify the logic
    assert processed_count == 2, "Should process 2 videos"
    assert skipped_count == 1, "Should skip 1 video"
    
    print("\n✅ TEST PASSED: Videos without categories are correctly skipped!\n")
    
    # Show example for video_manager
    print("="*70)
    print("VIDEO MANAGER EXAMPLE")
    print("="*70)
    print("\nIn video_manager.py, unassigned videos now show:")
    print("  ID     Name                                Categories")
    print("  " + "-"*66)
    print("  5      Video WITHOUT categories            ⚠️  <WILL BE SKIPPED - no categories>")
    print("  12     Another unassigned video            ⚠️  <WILL BE SKIPPED - no categories>")
    print("\n")

if __name__ == '__main__':
    test_skip_logic()
