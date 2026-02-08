#!/usr/bin/env python3
"""
Test the optimized frame extraction logic.
"""

import sys
import os
import tempfile
import json
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from dataset_generator_v2.make_dataset_multi import DatasetGeneratorV2

def test_method_exists():
    """Test that new methods exist and old ones are removed."""
    print("Testing method existence...")
    
    # Create a minimal config
    config = {
        'base_settings': {
            'output_base_dir': '/tmp/test_output',
            'temp_dir': '/tmp/test_temp',
            'status_file': '/tmp/test_status.json',
            'max_workers': 1,
            'base_frame_limit': 100,
            'max_retry_attempts': 3,
            'retry_skip_seconds': 5,
            'min_file_size': 1000,
            'scene_diff_threshold': 30
        },
        'videos': [],
        'format_config': {},
        'category_targets': {
            'master': {'target': 1000},
            'universal': {'target': 1000},
            'space': {'target': 1000},
            'toon': {'target': 1000}
        }
    }
    
    # Create temp config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_file = f.name
    
    try:
        # Initialize generator
        generator = DatasetGeneratorV2(config_file)
        
        # Check new methods exist
        assert hasattr(generator, 'extract_full_resolution_frames'), "Missing extract_full_resolution_frames method"
        assert hasattr(generator, 'process_all_categories_from_frames'), "Missing process_all_categories_from_frames method"
        
        # Check old method doesn't exist
        assert not hasattr(generator, 'extract_7_frames'), "Old extract_7_frames method still exists"
        
        # Check method is callable
        assert callable(generator.extract_full_resolution_frames), "extract_full_resolution_frames is not callable"
        assert callable(generator.process_all_categories_from_frames), "process_all_categories_from_frames is not callable"
        
        print("✓ Method existence tests passed")
        
    finally:
        if os.path.exists(config_file):
            os.unlink(config_file)
        # Clean up status file if created
        if os.path.exists('/tmp/test_status.json'):
            os.unlink('/tmp/test_status.json')

def test_extract_with_retry_signature():
    """Test that extract_with_retry has correct signature."""
    print("Testing extract_with_retry signature...")
    
    config = {
        'base_settings': {
            'output_base_dir': '/tmp/test_output',
            'temp_dir': '/tmp/test_temp',
            'status_file': '/tmp/test_status2.json',
            'max_workers': 1,
            'base_frame_limit': 100,
            'max_retry_attempts': 3,
            'retry_skip_seconds': 5,
            'min_file_size': 1000,
            'scene_diff_threshold': 30
        },
        'videos': [],
        'format_config': {},
        'category_targets': {
            'master': {'target': 1000}
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_file = f.name
    
    try:
        generator = DatasetGeneratorV2(config_file)
        
        # Get method signature
        import inspect
        sig = inspect.signature(generator.extract_with_retry)
        params = list(sig.parameters.keys())
        
        # Check parameters
        assert 'video_path' in params, "Missing video_path parameter"
        assert 'video_name' in params, "Missing video_name parameter"
        assert 'categories' in params, "Missing categories parameter"
        assert 'frame_idx' in params, "Missing frame_idx parameter"
        assert 'duration' in params, "Missing duration parameter"
        
        print("✓ extract_with_retry signature tests passed")
        
    finally:
        if os.path.exists(config_file):
            os.unlink(config_file)
        if os.path.exists('/tmp/test_status2.json'):
            os.unlink('/tmp/test_status2.json')

def main():
    """Run all tests."""
    print("Running optimized extraction tests...\n")
    
    try:
        test_method_exists()
        test_extract_with_retry_signature()
        
        print("\n✅ All tests passed!")
        return 0
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
