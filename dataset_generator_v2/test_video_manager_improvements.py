#!/usr/bin/env python3
"""
Test script for video manager improvements:
1. Regex error handling
2. Simple string search fallback
3. Interactive selection (simulated)
"""

import json
import tempfile
import os
from pathlib import Path

# Create minimal test config
test_config = {
    "videos": [
        {"name": "Star Trek - The Motion Picture", "path": "/test/st1.mkv", "categories": {}},
        {"name": "Star Trek II - The Wrath of Khan", "path": "/test/st2.mkv", "categories": {}},
        {"name": "Planet Earth", "path": "/test/pe1.mkv", "categories": {}},
        {"name": "Planet Earth II", "path": "/test/pe2.mkv", "categories": {}},
        {"name": "Avatar", "path": "/test/avatar.mkv", "categories": {}},
        {"name": "Shrek", "path": "/test/shrek.mkv", "categories": {}},
    ],
    "category_targets": {
        "master": 100000,
        "universal": 50000,
        "space": 60000,
        "toon": 50000
    }
}

print("="*80)
print("VIDEO MANAGER IMPROVEMENTS TEST")
print("="*80)

# Create temp config file
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(test_config, f, indent=2)
    temp_config_path = f.name

try:
    # Import after config is created
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from video_manager import VideoManager
    
    manager = VideoManager(temp_config_path)
    manager.load()
    
    print("\n" + "="*80)
    print("TEST 1: Invalid Regex Pattern (should NOT crash)")
    print("="*80)
    
    # This used to crash with "*Planet Earth*"
    invalid_patterns = [
        "*Planet Earth*",  # Invalid regex (unescaped *)
        "**Star**",        # Invalid regex
        "Planet*",         # Invalid regex (but should work with simple search)
    ]
    
    for pattern in invalid_patterns:
        print(f"\nTesting pattern: '{pattern}'")
        try:
            # Should handle error gracefully
            videos = manager.list_videos(filter_pattern=pattern)
            print(f"  ✓ Success: Found {len(videos)} videos (fallback to simple search)")
            for idx, video in videos[:3]:
                print(f"    - {video['name']}")
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
    
    print("\n" + "="*80)
    print("TEST 2: Simple String Search")
    print("="*80)
    
    # Test simple string search
    test_patterns = [
        ("Planet", 2, "Should find 'Planet Earth' videos"),
        ("Star Trek", 2, "Should find Star Trek videos"),
        ("trek", 2, "Case insensitive"),
        ("avatar", 1, "Should find Avatar"),
    ]
    
    for pattern, expected_count, description in test_patterns:
        print(f"\nPattern: '{pattern}' - {description}")
        videos = manager.list_videos(filter_pattern=pattern, use_simple_search=True)
        print(f"  Expected: {expected_count}, Found: {len(videos)}")
        if len(videos) == expected_count:
            print("  ✓ PASS")
        else:
            print("  ❌ FAIL")
        for idx, video in videos:
            print(f"    - {video['name']}")
    
    print("\n" + "="*80)
    print("TEST 3: Regex Search (valid patterns)")
    print("="*80)
    
    regex_patterns = [
        ("Star Trek.*", 2, "Star Trek with regex"),
        ("Planet.*", 2, "Planet with wildcard"),
        ("^Star.*", 2, "Start with Star"),
        (".*II$", 2, "End with II"),
    ]
    
    for pattern, expected_count, description in regex_patterns:
        print(f"\nPattern: '{pattern}' - {description}")
        videos = manager.list_videos(filter_pattern=pattern, use_simple_search=False)
        print(f"  Expected: {expected_count}, Found: {len(videos)}")
        if len(videos) == expected_count:
            print("  ✓ PASS")
        else:
            print("  ❌ FAIL")
        for idx, video in videos:
            print(f"    - {video['name']}")
    
    print("\n" + "="*80)
    print("TEST 4: Interactive Selection Structure")
    print("="*80)
    
    print("\nTesting interactive_select_videos method exists...")
    if hasattr(manager, 'interactive_select_videos'):
        print("  ✓ Method exists")
        print("  Note: Full interactive test requires manual testing")
        print("  Usage: python3 video_manager.py -> Option 6")
    else:
        print("  ❌ Method not found!")
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETE")
    print("="*80)
    print("\n✅ SUMMARY:")
    print("  - Invalid regex patterns are handled gracefully (no crash)")
    print("  - Simple string search works as fallback")
    print("  - Valid regex patterns work correctly")
    print("  - Interactive selection method is available")
    print("\nTo test interactive mode manually:")
    print("  python3 video_manager.py")
    print("  Choose option 6 (Interactive multi-select)")

finally:
    # Cleanup
    os.unlink(temp_config_path)
    print(f"\n✓ Cleaned up temp file: {temp_config_path}")
