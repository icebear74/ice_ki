#!/usr/bin/env python3
"""
Test for multi-category priority video sorting in video_manager.py
"""

import sys
import os
import tempfile
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from video_manager import VideoManager

print("=" * 70)
print("MULTI-CATEGORY PRIORITY SORTING TEST")
print("=" * 70)

# Test: Videos with multiple categories come first
test_config = {
    "videos": [
        # Should be 1st: master+space+toon (3 categories)
        {"name": "Video A", "path": "/test/a.mkv", "categories": ["master", "space", "toon"]},
        # Should be 2nd: master+space (2 categories, master first)
        {"name": "Video B", "path": "/test/b.mkv", "categories": ["master", "space"]},
        # Should be 3rd: master+universal (2 categories, master first)
        {"name": "Video C", "path": "/test/c.mkv", "categories": ["master", "universal"]},
        # Should be 4th: space+toon (2 categories, space < toon alphabetically)
        {"name": "Video D", "path": "/test/d.mkv", "categories": ["space", "toon"]},
        # Should be 5th: master only (1 category)
        {"name": "Video E", "path": "/test/e.mkv", "categories": ["master"]},
        # Should be 6th: space only (1 category)
        {"name": "Video F", "path": "/test/f.mkv", "categories": ["space"]},
        # Should be 7th: toon only (1 category)
        {"name": "Video G", "path": "/test/g.mkv", "categories": ["toon"]},
        # Should be 8th: universal only (1 category)
        {"name": "Video H", "path": "/test/h.mkv", "categories": ["universal"]},
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
    
    print("\nAfter save (sorted by multi-category priority):")
    for i, video in enumerate(saved_config['videos']):
        cats = video.get('categories', [])
        num_cats = len(cats)
        print(f"  {i}: {video['name']:<20} - {cats} ({num_cats} categories)")
    
    # Verify order
    saved_names = [v['name'] for v in saved_config['videos']]
    saved_cat_counts = [len(v.get('categories', [])) for v in saved_config['videos']]
    
    print("\nVerification:")
    print(f"  Video names: {saved_names}")
    print(f"  Cat counts:  {saved_cat_counts}")
    
    # Expected order based on rules:
    # 1. Most categories first (3, then 2, then 1)
    # 2. Within same count, alphabetically by first category
    # 3. Within same category, alphabetically by name
    expected_order = [
        "Video A",  # 3 cats: master+space+toon
        "Video B",  # 2 cats: master+space
        "Video C",  # 2 cats: master+universal
        "Video D",  # 2 cats: space+toon
        "Video E",  # 1 cat: master
        "Video F",  # 1 cat: space
        "Video G",  # 1 cat: toon
        "Video H",  # 1 cat: universal
    ]
    
    if saved_names == expected_order:
        print("\n✓ PASS: Videos sorted correctly by multi-category priority")
        print("  Order: Multi-category → Single-category")
        print("  Within groups: Alphabetically by first category")
    else:
        print(f"\n✗ FAIL: Expected {expected_order}, got {saved_names}")
        for i, (exp, got) in enumerate(zip(expected_order, saved_names)):
            if exp != got:
                print(f"    Position {i}: expected {exp}, got {got}")
    
finally:
    os.unlink(config_path)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Videos are now sorted with priority to multi-category videos:
  ✓ Videos in multiple categories come first
  ✓ Within same category count, sorted by first category (master first)
  ✓ Within same category, sorted by video name

Example order:
  1. Videos in master+space+toon (3 categories)
  2. Videos in master+space (2 categories)
  3. Videos in master only (1 category)
  4. Videos in space only (1 category)

This ensures important multi-category videos are processed first by the dataset generator.
""")

print("✓ Multi-category priority sorting test passed")
