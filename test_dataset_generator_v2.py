#!/usr/bin/env python3
"""
Basic tests for dataset generator v2.
"""

import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataset_generator_v2.utils.format_definitions import (
    FORMATS, CATEGORY_FORMAT_DISTRIBUTION, CATEGORY_PATHS,
    select_random_format, get_output_dirs_for_format
)
from dataset_generator_v2.utils.progress_tracker import ProgressTracker

def test_format_definitions():
    """Test format definitions."""
    print("Testing format definitions...")
    
    # Check all formats are defined
    assert 'small_540' in FORMATS
    assert 'medium_169' in FORMATS
    assert 'large_720' in FORMATS
    assert 'xlarge_1440' in FORMATS
    assert 'fullhd_1920' in FORMATS
    
    # Check all categories have distributions
    assert 'general' in CATEGORY_FORMAT_DISTRIBUTION
    assert 'space' in CATEGORY_FORMAT_DISTRIBUTION
    assert 'toon' in CATEGORY_FORMAT_DISTRIBUTION
    
    # Check distribution probabilities sum to 1.0 (or close)
    for cat, dist in CATEGORY_FORMAT_DISTRIBUTION.items():
        total = sum(dist.values())
        assert abs(total - 1.0) < 0.01, f"Category {cat} distribution sums to {total}, not 1.0"
    
    # Test random format selection
    for _ in range(10):
        fmt = select_random_format('general')
        assert fmt in CATEGORY_FORMAT_DISTRIBUTION['general']
    
    print("✓ Format definitions tests passed")

def test_output_dirs():
    """Test output directory generation."""
    print("Testing output directory generation...")
    
    base = "/test/path"
    
    # Test 5-frame LR (VSR++ compatible)
    dirs_5 = get_output_dirs_for_format(base, 'general', 'small_540', lr_frames=5)
    
    assert 'gt' in dirs_5
    assert 'lr' in dirs_5
    assert 'val_gt' in dirs_5
    assert 'val_lr' in dirs_5
    
    assert dirs_5['gt'].endswith('/GT')
    assert dirs_5['lr'].endswith('/LR')  # VSR++ compatible
    assert dirs_5['val_gt'].endswith('/Val/GT')
    
    # Test 7-frame LR (extended version)
    dirs_7 = get_output_dirs_for_format(base, 'general', 'small_540', lr_frames=7)
    
    assert 'gt' in dirs_7
    assert 'lr' in dirs_7
    assert dirs_7['lr'].endswith('/LR_7frames')  # Extended version
    
    print("✓ Output directory tests passed")

def test_progress_tracker():
    """Test progress tracker."""
    print("Testing progress tracker...")
    
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        status_file = f.name
    
    try:
        # Create tracker
        tracker = ProgressTracker(status_file)
        
        # Test basic operations
        tracker.update_progress(total_videos=100, completed_videos=10)
        tracker.update_category_stats('general', images_created=500)
        tracker.increment_category_images('space', 10)
        tracker.increment_category_videos('toon')
        
        # Save and reload
        tracker.save()
        
        # Create new tracker from same file
        tracker2 = ProgressTracker(status_file)
        
        assert tracker2.status['progress']['total_videos'] == 100
        assert tracker2.status['progress']['completed_videos'] == 10
        assert tracker2.status['category_stats']['general']['images_created'] == 500
        assert tracker2.status['category_stats']['space']['images_created'] == 10
        assert tracker2.status['category_stats']['toon']['videos_processed'] == 1
        
        print("✓ Progress tracker tests passed")
        
    finally:
        if os.path.exists(status_file):
            os.unlink(status_file)

def test_config_loading():
    """Test that config file loads correctly."""
    print("Testing config file loading...")
    
    config_path = os.path.join(
        os.path.dirname(__file__),
        'dataset_generator_v2',
        'generator_config.json'
    )
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    assert 'base_settings' in config
    assert 'videos' in config
    assert len(config['videos']) > 0
    
    # Check first video has required fields
    video = config['videos'][0]
    assert 'name' in video
    assert 'path' in video
    assert 'categories' in video
    
    # Check category weights
    for video in config['videos']:
        total_weight = sum(video['categories'].values())
        assert total_weight > 0, f"Video {video['name']} has zero total weight"
    
    print(f"✓ Config file loaded successfully ({len(config['videos'])} videos)")

def main():
    """Run all tests."""
    print("Running dataset generator v2 tests...\n")
    
    try:
        test_format_definitions()
        test_output_dirs()
        test_progress_tracker()
        test_config_loading()
        
        print("\n✅ All tests passed!")
        return 0
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
