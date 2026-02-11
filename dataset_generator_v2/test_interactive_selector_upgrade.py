#!/usr/bin/env python3
"""
Test the upgraded interactive selector integration in video_manager.py

This test verifies that:
1. interactive_select_videos() uses the curses-based selector
2. The method handles filter parameters correctly
3. Video IDs are correctly returned
4. Error handling works when curses is not available
"""

import sys
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from video_manager import VideoManager

print("=" * 70)
print("INTERACTIVE SELECTOR UPGRADE TEST")
print("=" * 70)

# Test 1: Verify method exists and has correct signature
print("\nTest 1: Method Signature")
print("-" * 70)

test_config = {
    "videos": [
        {"name": "Video A", "path": "/test/a.mkv", "categories": ["master", "space"]},
        {"name": "Video B", "path": "/test/b.mkv", "categories": ["master"]},
        {"name": "Video C", "path": "/test/c.mkv", "categories": ["toon"]},
    ],
    "category_targets": {}
}

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(test_config, f)
    config_path = f.name

try:
    manager = VideoManager(config_path)
    manager.load()
    
    # Check method exists
    assert hasattr(manager, 'interactive_select_videos'), "Method not found"
    print("✓ Method exists: interactive_select_videos()")
    
    # Check it's callable
    assert callable(manager.interactive_select_videos), "Method not callable"
    print("✓ Method is callable")
    
    # Check method signature accepts optional filter
    import inspect
    sig = inspect.signature(manager.interactive_select_videos)
    params = list(sig.parameters.keys())
    assert 'initial_filter' in params, "Missing initial_filter parameter"
    print(f"✓ Method signature: {sig}")
    
finally:
    os.unlink(config_path)


# Test 2: Verify it attempts to use curses selector
print("\n\nTest 2: Curses Selector Integration")
print("-" * 70)

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(test_config, f)
    config_path = f.name

try:
    manager = VideoManager(config_path)
    manager.load()
    
    # Mock select_items to avoid curses requirement
    with patch('video_manager.select_items') as mock_select:
        # Simulate user selecting indices 0 and 2
        mock_select.return_value = [0, 2]
        
        # Call the method
        result = manager.interactive_select_videos()
        
        # Verify select_items was called
        assert mock_select.called, "select_items was not called"
        print("✓ Method calls select_items()")
        
        # Check the call arguments
        call_args = mock_select.call_args
        assert 'items' in call_args.kwargs, "Missing items argument"
        assert 'title' in call_args.kwargs, "Missing title argument"
        assert 'get_label' in call_args.kwargs, "Missing get_label argument"
        assert 'get_details' in call_args.kwargs, "Missing get_details argument"
        print("✓ Correct arguments passed to select_items()")
        
        # Verify the result
        assert result == [0, 2], f"Expected [0, 2], got {result}"
        print(f"✓ Returns correct video IDs: {result}")
        
finally:
    os.unlink(config_path)


# Test 3: Verify filter parameter works
print("\n\nTest 3: Filter Parameter")
print("-" * 70)

test_config_large = {
    "videos": [
        {"name": "Star Wars Episode IV", "path": "/test/sw4.mkv", "categories": ["master", "space"]},
        {"name": "Star Wars Episode V", "path": "/test/sw5.mkv", "categories": ["master", "space"]},
        {"name": "Star Trek", "path": "/test/st.mkv", "categories": ["master", "space"]},
        {"name": "Avatar", "path": "/test/avatar.mkv", "categories": ["master"]},
        {"name": "Shrek", "path": "/test/shrek.mkv", "categories": ["toon"]},
    ],
    "category_targets": {}
}

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(test_config_large, f)
    config_path = f.name

try:
    manager = VideoManager(config_path)
    manager.load()
    
    with patch('video_manager.select_items') as mock_select:
        mock_select.return_value = [0, 1]
        
        # Call with filter
        result = manager.interactive_select_videos(initial_filter="Star Wars")
        
        # Verify select_items was called
        assert mock_select.called, "select_items was not called"
        
        # Check that the items list was filtered
        call_args = mock_select.call_args
        items = call_args.kwargs['items']
        
        # Should only have Star Wars videos
        print(f"  Filtered to {len(items)} items (from filter 'Star Wars')")
        assert len(items) == 2, f"Expected 2 filtered items, got {len(items)}"
        
        # Verify the items are Star Wars videos
        names = [manager.videos[i]['name'] for i, v in manager.list_videos(filter_pattern="Star Wars", use_simple_search=True)]
        print(f"  Filtered videos: {names}")
        assert all("Star Wars" in name for name in names), "Filter didn't work correctly"
        print("✓ Filter parameter works correctly")
        
finally:
    os.unlink(config_path)


# Test 4: Error handling when select_items returns None (cancelled)
print("\n\nTest 4: Cancellation Handling")
print("-" * 70)

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(test_config, f)
    config_path = f.name

try:
    manager = VideoManager(config_path)
    manager.load()
    
    with patch('video_manager.select_items') as mock_select:
        # Simulate user cancelling (Esc key)
        mock_select.return_value = None
        
        result = manager.interactive_select_videos()
        
        assert result is None, f"Expected None, got {result}"
        print("✓ Returns None when user cancels")
        
finally:
    os.unlink(config_path)


# Test 5: Error handling when curses fails
print("\n\nTest 5: Curses Failure Handling")
print("-" * 70)

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(test_config, f)
    config_path = f.name

try:
    manager = VideoManager(config_path)
    manager.load()
    
    with patch('video_manager.select_items') as mock_select:
        # Simulate curses failure
        mock_select.side_effect = Exception("Curses not available")
        
        # Should catch exception and return None
        result = manager.interactive_select_videos()
        
        assert result is None, f"Expected None on error, got {result}"
        print("✓ Handles curses failure gracefully")
        
finally:
    os.unlink(config_path)


# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
All tests passed! The interactive selector upgrade is working correctly:

✓ Method signature is correct with optional filter parameter
✓ Uses curses-based select_items() instead of text commands
✓ Passes correct arguments to the selector
✓ Returns correct video IDs from selection
✓ Filter parameter works to pre-filter videos
✓ Handles user cancellation (returns None)
✓ Handles curses failures gracefully

The text-based command interface has been successfully replaced with
the modern curses-based interactive selector!
""")

print("✓ All interactive selector upgrade tests passed")
