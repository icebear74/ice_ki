#!/usr/bin/env python3
"""
Test that custom categories like 'master' work without KeyError.
This test validates the fix for the CATEGORY_FORMAT_DISTRIBUTION KeyError.
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

def test_custom_category_master():
    """Test that 'master' category works without KeyError."""
    print("Testing custom category 'master'...")
    
    from dataset_generator_v2.make_dataset_multi import DatasetGeneratorV2
    
    # Create a test config with 'master' category
    test_config = {
        "base_settings": {
            "base_frame_limit": 100,
            "max_workers": 2,
            "val_percent": 0.0,
            "output_base_dir": "/tmp/test_master",
            "temp_dir": "/tmp/test_master/temp",
            "status_file": "/tmp/test_master/.status.json",
            "min_file_size": 10000,
            "scene_diff_threshold": 45,
            "max_retry_attempts": 3,
            "retry_skip_seconds": 30,
            "lr_versions": ["5frames", "7frames"]
        },
        "category_targets": {
            "master": 10000,
            "universal": 20000
        },
        "format_config": {
            "master": {
                "small_540": {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 0.6},
                "medium_169": {"gt_size": [720, 405], "lr_size": [240, 135], "probability": 0.4}
            },
            "universal": {
                "small_540": {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 1.0}
            }
        },
        "videos": []
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f)
        config_path = f.name
    
    try:
        # This should NOT raise KeyError: 'master'
        generator = DatasetGeneratorV2(config_path)
        
        # Test that we can get category path for 'master'
        master_path = generator.get_category_path('master')
        assert master_path is not None
        print(f"  ✓ master category path: {master_path}")
        
        # Test that we can select format for 'master'
        format_name = generator.select_format_for_category('master')
        assert format_name in ['small_540', 'medium_169']
        print(f"  ✓ master format selection: {format_name}")
        
        # Test that we can get output dirs for 'master'
        dirs = generator.get_output_dirs_for_category_format('master', 'small_540')
        assert 'gt' in dirs
        assert 'lr' in dirs
        print(f"  ✓ master output dirs: {dirs['gt']}")
        
        # Test that create_output_directories works (this is where the KeyError was)
        generator.create_output_directories()
        print("  ✓ create_output_directories succeeded")
        
        print("✓ Custom category 'master' works correctly")
        
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)

def test_mixed_categories():
    """Test mix of standard and custom categories."""
    print("Testing mixed standard and custom categories...")
    
    from dataset_generator_v2.make_dataset_multi import DatasetGeneratorV2
    
    test_config = {
        "base_settings": {
            "base_frame_limit": 100,
            "max_workers": 2,
            "val_percent": 0.0,
            "output_base_dir": "/tmp/test_mixed",
            "temp_dir": "/tmp/test_mixed/temp",
            "status_file": "/tmp/test_mixed/.status.json",
            "min_file_size": 10000,
            "scene_diff_threshold": 45,
            "max_retry_attempts": 3,
            "retry_skip_seconds": 30,
            "lr_versions": ["5frames", "7frames"]
        },
        "category_targets": {
            "general": 10000,
            "master": 15000,
            "space": 8000,
            "custom1": 5000
        },
        "format_config": {
            "general": {
                "small_540": {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 1.0}
            },
            "master": {
                "small_540": {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 1.0}
            },
            "space": {
                "small_540": {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 1.0}
            },
            "custom1": {
                "small_540": {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 1.0}
            }
        },
        "videos": []
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f)
        config_path = f.name
    
    try:
        generator = DatasetGeneratorV2(config_path)
        
        # Test all categories
        for category in ['general', 'master', 'space', 'custom1']:
            path = generator.get_category_path(category)
            format_name = generator.select_format_for_category(category)
            dirs = generator.get_output_dirs_for_category_format(category, format_name)
            
            assert path is not None
            assert format_name is not None
            assert 'gt' in dirs
            
            print(f"  ✓ {category}: path={path}, format={format_name}")
        
        # This should work for all categories
        generator.create_output_directories()
        print("  ✓ create_output_directories succeeded for all categories")
        
        print("✓ Mixed categories work correctly")
        
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)

def test_category_without_format_config():
    """Test category that has no format_config (should use fallback)."""
    print("Testing category without format_config...")
    
    from dataset_generator_v2.make_dataset_multi import DatasetGeneratorV2
    
    test_config = {
        "base_settings": {
            "base_frame_limit": 100,
            "max_workers": 2,
            "val_percent": 0.0,
            "output_base_dir": "/tmp/test_fallback",
            "temp_dir": "/tmp/test_fallback/temp",
            "status_file": "/tmp/test_fallback/.status.json",
            "min_file_size": 10000,
            "scene_diff_threshold": 45,
            "max_retry_attempts": 3,
            "retry_skip_seconds": 30,
            "lr_versions": ["5frames", "7frames"]
        },
        "category_targets": {
            "general": 10000,  # Has format_config in hard-coded CATEGORY_FORMAT_DISTRIBUTION
            "newcat": 5000     # Does NOT have format_config anywhere
        },
        "format_config": {
            # Only general is configured, newcat is not
            "general": {
                "small_540": {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 1.0}
            }
        },
        "videos": []
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f)
        config_path = f.name
    
    try:
        generator = DatasetGeneratorV2(config_path)
        
        # Test 'general' works
        general_format = generator.select_format_for_category('general')
        assert general_format == 'small_540'
        print(f"  ✓ general format: {general_format}")
        
        # Test 'newcat' falls back to default
        newcat_format = generator.select_format_for_category('newcat')
        assert newcat_format == 'small_540'  # Ultimate fallback
        print(f"  ✓ newcat format (fallback): {newcat_format}")
        
        # This should still work with fallback
        generator.create_output_directories()
        print("  ✓ create_output_directories succeeded with fallback")
        
        print("✓ Fallback handling works correctly")
        
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)

def main():
    """Run all tests."""
    print("Running custom category KeyError fix tests...\n")
    
    try:
        test_custom_category_master()
        print()
        test_mixed_categories()
        print()
        test_category_without_format_config()
        
        print("\n✅ All KeyError fix tests passed!")
        return 0
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
