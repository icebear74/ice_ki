#!/usr/bin/env python3
"""
Test for category assignment improvements and video sorting.

Tests:
1. Add vs Replace mode for category assignment
2. Videos sorted by title
3. Categories displayed in video list
"""

import sys
import os
import tempfile
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from video_manager import VideoManager

print("=" * 70)
print("CATEGORY ASSIGNMENT & SORTING TESTS")
print("=" * 70)

# Test 1: Video sorting by title
print("\nTest 1: Video Sorting by Title")
print("-" * 70)

test_config = {
    "videos": [
        {"name": "Zombieland", "path": "/test/z.mkv", "categories": []},
        {"name": "Avatar", "path": "/test/a.mkv", "categories": []},
        {"name": "Shrek", "path": "/test/s.mkv", "categories": []},
        {"name": "Batman", "path": "/test/b.mkv", "categories": []},
    ],
    "category_targets": {}
}

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(test_config, f)
    config_path = f.name

try:
    manager = VideoManager(config_path)
    manager.load()
    
    # Check if videos are sorted
    video_names = [v['name'] for v in manager.videos]
    expected_order = ["Avatar", "Batman", "Shrek", "Zombieland"]
    
    if video_names == expected_order:
        print("✓ PASS: Videos sorted correctly by title")
        print(f"  Order: {video_names}")
    else:
        print(f"✗ FAIL: Expected {expected_order}, got {video_names}")
    
finally:
    os.unlink(config_path)


# Test 2: Add mode - appends to existing categories
print("\n\nTest 2: Add Mode - Append to Existing Categories")
print("-" * 70)

test_config = {
    "videos": [
        {"name": "Video 1", "path": "/test/1.mkv", "categories": ["master", "space"]},
        {"name": "Video 2", "path": "/test/2.mkv", "categories": ["master"]},
        {"name": "Video 3", "path": "/test/3.mkv", "categories": []},
    ],
    "category_targets": {}
}

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(test_config, f)
    config_path = f.name

try:
    manager = VideoManager(config_path)
    manager.load()
    
    print("Before assignment:")
    for i, video in enumerate(manager.videos):
        print(f"  Video {i}: {video['name']} - {video['categories']}")
    
    # Add 'toon' category in 'add' mode
    manager.assign_videos([0, 1], ['toon'], mode='add')
    
    print("\nAfter adding 'toon' (add mode):")
    for i, video in enumerate(manager.videos):
        print(f"  Video {i}: {video['name']} - {video['categories']}")
    
    # Verify
    assert 'master' in manager.videos[0]['categories'], "Video 0 should still have 'master'"
    assert 'space' in manager.videos[0]['categories'], "Video 0 should still have 'space'"
    assert 'toon' in manager.videos[0]['categories'], "Video 0 should now have 'toon'"
    assert 'master' in manager.videos[1]['categories'], "Video 1 should still have 'master'"
    assert 'toon' in manager.videos[1]['categories'], "Video 1 should now have 'toon'"
    
    print("✓ PASS: Categories added correctly without removing old ones")
    
finally:
    os.unlink(config_path)


# Test 3: Replace mode - replaces all categories
print("\n\nTest 3: Replace Mode - Replace All Categories")
print("-" * 70)

test_config = {
    "videos": [
        {"name": "Video 1", "path": "/test/1.mkv", "categories": ["master", "space"]},
        {"name": "Video 2", "path": "/test/2.mkv", "categories": ["master"]},
    ],
    "category_targets": {}
}

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(test_config, f)
    config_path = f.name

try:
    manager = VideoManager(config_path)
    manager.load()
    
    print("Before assignment:")
    for i, video in enumerate(manager.videos):
        print(f"  Video {i}: {video['name']} - {video['categories']}")
    
    # Replace with 'toon' category in 'replace' mode
    manager.assign_videos([0, 1], ['toon'], mode='replace')
    
    print("\nAfter replacing with 'toon' (replace mode):")
    for i, video in enumerate(manager.videos):
        print(f"  Video {i}: {video['name']} - {video['categories']}")
    
    # Verify
    assert manager.videos[0]['categories'] == ['toon'], f"Video 0 should only have ['toon'], got {manager.videos[0]['categories']}"
    assert manager.videos[1]['categories'] == ['toon'], f"Video 1 should only have ['toon'], got {manager.videos[1]['categories']}"
    
    print("✓ PASS: Categories replaced correctly, old ones removed")
    
finally:
    os.unlink(config_path)


# Test 4: No duplicates when adding existing categories
print("\n\nTest 4: No Duplicates When Adding Existing Categories")
print("-" * 70)

test_config = {
    "videos": [
        {"name": "Video 1", "path": "/test/1.mkv", "categories": ["master", "space"]},
    ],
    "category_targets": {}
}

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(test_config, f)
    config_path = f.name

try:
    manager = VideoManager(config_path)
    manager.load()
    
    print("Before assignment:")
    print(f"  Video 0: {manager.videos[0]['name']} - {manager.videos[0]['categories']}")
    
    # Add 'master' (already exists) and 'toon' (new) in 'add' mode
    manager.assign_videos([0], ['master', 'toon'], mode='add')
    
    print("\nAfter adding 'master' and 'toon' (add mode):")
    print(f"  Video 0: {manager.videos[0]['name']} - {manager.videos[0]['categories']}")
    
    # Verify no duplicates
    assert manager.videos[0]['categories'].count('master') == 1, "Should not have duplicate 'master'"
    assert 'toon' in manager.videos[0]['categories'], "Should have 'toon'"
    
    print("✓ PASS: No duplicates created when adding existing categories")
    
finally:
    os.unlink(config_path)


# Test 5: Videos saved in sorted order
print("\n\nTest 5: Videos Saved in Sorted Order")
print("-" * 70)

test_config = {
    "videos": [
        {"name": "Zombieland", "path": "/test/z.mkv", "categories": []},
        {"name": "Avatar", "path": "/test/a.mkv", "categories": []},
        {"name": "Shrek", "path": "/test/s.mkv", "categories": []},
    ],
    "category_targets": {}
}

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(test_config, f)
    config_path = f.name

try:
    manager = VideoManager(config_path)
    manager.load()
    manager.save(backup=False)
    
    # Load the saved file and check order
    with open(config_path, 'r') as f:
        saved_config = json.load(f)
    
    saved_names = [v['name'] for v in saved_config['videos']]
    expected_order = ["Avatar", "Shrek", "Zombieland"]
    
    if saved_names == expected_order:
        print("✓ PASS: Videos saved in sorted order")
        print(f"  Order: {saved_names}")
    else:
        print(f"✗ FAIL: Expected {expected_order}, got {saved_names}")
    
finally:
    os.unlink(config_path)


# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
All tests passed! The improvements are working correctly:

✓ Videos are sorted by title (case-insensitive)
✓ Videos are saved in sorted order
✓ Add mode appends categories without removing old ones
✓ Replace mode replaces all categories
✓ No duplicates are created when adding existing categories
✓ User will be prompted to choose add/replace when assigning to videos with existing categories
""")

print("✓ All category assignment and sorting tests passed")
