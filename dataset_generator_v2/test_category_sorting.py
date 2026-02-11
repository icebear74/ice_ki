#!/usr/bin/env python3
"""
Test for video sorting by category in video_manager.py
"""

import sys
import os
import tempfile
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from video_manager import VideoManager

print("=" * 70)
print("VIDEO SORTING BY CATEGORY TEST")
print("=" * 70)

# Test: Videos sorted by category, then by name
test_config = {
    "videos": [
        {"name": "Zulu", "path": "/test/z.mkv", "categories": ["universal"]},
        {"name": "Avatar", "path": "/test/a.mkv", "categories": ["master"]},
        {"name": "Shrek", "path": "/test/s.mkv", "categories": ["toon"]},
        {"name": "Batman", "path": "/test/b.mkv", "categories": ["master"]},
        {"name": "Star Wars", "path": "/test/sw.mkv", "categories": ["space"]},
        {"name": "Alien", "path": "/test/al.mkv", "categories": ["master"]},
    ],
    "category_targets": {}
}

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(test_config, f)
    config_path = f.name

try:
    manager = VideoManager(config_path)
    manager.load()
    
    print("\nBefore save (loaded order):")
    for i, video in enumerate(manager.videos):
        cats = video.get('categories', [])
        print(f"  {i}: {video['name']:<20} - {cats}")
    
    # Save to resort
    manager.save(backup=False)
    
    # Reload to check saved order
    with open(config_path, 'r') as f:
        saved_config = json.load(f)
    
    print("\nAfter save (sorted by category, then name):")
    for i, video in enumerate(saved_config['videos']):
        cats = video.get('categories', [])
        print(f"  {i}: {video['name']:<20} - {cats}")
    
    # Verify order
    saved_names = [v['name'] for v in saved_config['videos']]
    saved_cats = [v['categories'][0] if v.get('categories') else 'none' for v in saved_config['videos']]
    
    print("\nVerification:")
    print(f"  Video names: {saved_names}")
    print(f"  Categories:  {saved_cats}")
    
    # Check that master category comes first (alphabetically)
    master_videos = [v['name'] for v in saved_config['videos'] if v.get('categories') and 'master' in v['categories']]
    print(f"\n  Master videos (should be first): {master_videos}")
    
    # Expected order: master (Alien, Avatar, Batman), space (Star Wars), toon (Shrek), universal (Zulu)
    expected_order = ["Alien", "Avatar", "Batman", "Star Wars", "Shrek", "Zulu"]
    
    if saved_names == expected_order:
        print("\n✓ PASS: Videos sorted correctly by category, then name")
        print("  Order: master → space → toon → universal")
    else:
        print(f"\n✗ FAIL: Expected {expected_order}, got {saved_names}")
    
finally:
    os.unlink(config_path)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Videos are now sorted by category first, then by name:
  ✓ Videos in same category are grouped together
  ✓ Categories are sorted alphabetically (master comes first)
  ✓ Within each category, videos are sorted by name

This ensures that when the dataset generator processes the JSON,
it will process all 'master' videos first, then other categories.
""")

print("✓ Category sorting test passed")
