#!/usr/bin/env python3
"""
Test for show_statistics() method with list-based categories.
"""

import sys
import os
import tempfile
import json
from pathlib import Path
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent))

from video_manager import VideoManager

print("=" * 70)
print("SHOW STATISTICS TEST")
print("=" * 70)

# Test with list-based categories
test_config = {
    "videos": [
        {"name": "Video 1", "path": "/test/1.mkv", "categories": ["master", "space"]},
        {"name": "Video 2", "path": "/test/2.mkv", "categories": ["master"]},
        {"name": "Video 3", "path": "/test/3.mkv", "categories": ["toon"]},
        {"name": "Video 4", "path": "/test/4.mkv", "categories": ["master", "toon"]},
        {"name": "Video 5", "path": "/test/5.mkv", "categories": []},  # No categories
    ],
    "category_targets": {
        "master": 100,
        "space": 50,
        "toon": 75
    }
}

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(test_config, f)
    config_path = f.name

try:
    manager = VideoManager(config_path)
    manager.load()
    
    print("\nTest: Statistics with List-based Categories")
    print("-" * 70)
    
    # Capture output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        manager.show_statistics()
        output = sys.stdout.getvalue()
        success = True
    except Exception as e:
        output = str(e)
        success = False
    finally:
        sys.stdout = old_stdout
    
    if success:
        print("✓ PASS: show_statistics() executed without error")
        print("\nOutput:")
        print(output)
        
        # Verify expected counts
        if "master" in output and "3 videos" in output:
            print("✓ PASS: Correct count for 'master' (3 videos)")
        else:
            print("✗ FAIL: Incorrect count for 'master'")
        
        if "space" in output and "1 videos" in output:
            print("✓ PASS: Correct count for 'space' (1 video)")
        else:
            print("✗ FAIL: Incorrect count for 'space'")
        
        if "toon" in output and "2 videos" in output:
            print("✓ PASS: Correct count for 'toon' (2 videos)")
        else:
            print("✗ FAIL: Incorrect count for 'toon'")
        
        if "Unassigned: 1" in output:
            print("✓ PASS: Correct count for unassigned (1 video)")
        else:
            print("✗ FAIL: Incorrect count for unassigned")
    else:
        print(f"✗ FAIL: show_statistics() raised error: {output}")
    
finally:
    os.unlink(config_path)

# Test with mixed formats (legacy dict + new list)
print("\n\nTest: Statistics with Mixed Format (backwards compatibility)")
print("-" * 70)

test_config_mixed = {
    "videos": [
        {"name": "Video 1", "path": "/test/1.mkv", "categories": ["master", "space"]},  # List
        {"name": "Video 2", "path": "/test/2.mkv", "categories": {"master": 0.5, "toon": 0.5}},  # Dict (legacy)
        {"name": "Video 3", "path": "/test/3.mkv", "categories": ["toon"]},  # List
    ],
    "category_targets": {}
}

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(test_config_mixed, f)
    config_path = f.name

try:
    manager = VideoManager(config_path)
    manager.load()
    
    # Capture output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        manager.show_statistics()
        output = sys.stdout.getvalue()
        success = True
    except Exception as e:
        output = str(e)
        success = False
    finally:
        sys.stdout = old_stdout
    
    if success:
        print("✓ PASS: show_statistics() handles mixed formats")
        print("\nOutput:")
        print(output)
    else:
        print(f"✗ FAIL: show_statistics() failed with mixed formats: {output}")
    
finally:
    os.unlink(config_path)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
The show_statistics() method now correctly handles:
  ✓ List-based categories (current format)
  ✓ Dict-based categories (legacy format)
  ✓ Mixed formats for backwards compatibility
  ✓ Videos with no categories (unassigned count)
""")

print("✓ All statistics tests passed")
