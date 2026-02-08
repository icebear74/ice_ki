#!/usr/bin/env python3
"""
Test debug logging implementation in dataset generator.
"""

import os
import sys
import json
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_logger_initialization():
    """Test that logger is properly initialized."""
    print("Testing logger initialization...")
    
    # Create a temporary config file
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, 'test_config.json')
    log_path = os.path.join(temp_dir, 'test_debug.log')
    
    try:
        # Create minimal config
        config = {
            "base_settings": {
                "base_frame_limit": 10,
                "max_workers": 1,
                "val_percent": 0.0,
                "output_base_dir": os.path.join(temp_dir, "output"),
                "temp_dir": os.path.join(temp_dir, "temp"),
                "status_file": os.path.join(temp_dir, "status.json"),
                "min_file_size": 10000,
                "scene_diff_threshold": 45,
                "max_retry_attempts": 3,
                "retry_skip_seconds": 30,
                "lr_versions": ["5frames", "7frames"],
                "enable_debug_logging": True,
                "debug_log_path": log_path
            },
            "category_targets": {
                "master": 1000
            },
            "format_config": {
                "master": {
                    "small_540": {
                        "gt_size": [540, 540],
                        "lr_size": [180, 180],
                        "probability": 1.0
                    }
                }
            },
            "videos": [
                {
                    "name": "Test Video 1",
                    "path": "/non/existent/path1.mkv",
                    "categories": {"master": 1.0}
                },
                {
                    "name": "Test Video 2",
                    "path": "/non/existent/path2.mkv",
                    "categories": {"master": 1.0}
                },
                {
                    "name": "Test Video 3",
                    "path": "/non/existent/path3.mkv",
                    "categories": {"master": 1.0}
                }
            ]
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Import and initialize generator
        from dataset_generator_v2.make_dataset_multi import DatasetGeneratorV2
        
        generator = DatasetGeneratorV2(config_path)
        
        # Check logger was created
        assert hasattr(generator, 'logger'), "Generator should have logger attribute"
        assert generator.logger is not None, "Logger should not be None"
        
        # Check log file was created
        assert os.path.exists(log_path), f"Log file should be created at {log_path}"
        
        # Read log file and verify initial messages
        with open(log_path, 'r') as f:
            log_content = f.read()
        
        # Check for key log messages
        assert "Initializing generator with 3 videos" in log_content, \
            "Should log video count"
        assert "First 5 videos:" in log_content, \
            "Should log first 5 videos"
        assert "Test Video 1" in log_content or "Test Video 2" in log_content, \
            "Should include video names in logs"
        
        print("✓ Logger initialization test passed")
        print(f"  - Log file created: {log_path}")
        print(f"  - Log entries: {len(log_content.splitlines())} lines")
        print("\nSample log output:")
        for line in log_content.splitlines()[:5]:
            print(f"  {line}")
        
        return True
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_logging_disabled():
    """Test that logging can be disabled."""
    print("\nTesting disabled logging...")
    
    # Create a temporary config file
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, 'test_config.json')
    log_path = os.path.join(temp_dir, 'test_debug.log')
    
    try:
        # Create minimal config with logging disabled
        config = {
            "base_settings": {
                "base_frame_limit": 10,
                "max_workers": 1,
                "val_percent": 0.0,
                "output_base_dir": os.path.join(temp_dir, "output"),
                "temp_dir": os.path.join(temp_dir, "temp"),
                "status_file": os.path.join(temp_dir, "status.json"),
                "min_file_size": 10000,
                "scene_diff_threshold": 45,
                "max_retry_attempts": 3,
                "retry_skip_seconds": 30,
                "lr_versions": ["5frames", "7frames"],
                "enable_debug_logging": False,  # DISABLED
                "debug_log_path": log_path
            },
            "category_targets": {
                "master": 1000
            },
            "format_config": {
                "master": {
                    "small_540": {
                        "gt_size": [540, 540],
                        "lr_size": [180, 180],
                        "probability": 1.0
                    }
                }
            },
            "videos": [
                {
                    "name": "Test Video 1",
                    "path": "/non/existent/path1.mkv",
                    "categories": {"master": 1.0}
                }
            ]
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Import and initialize generator
        from dataset_generator_v2.make_dataset_multi import DatasetGeneratorV2
        
        generator = DatasetGeneratorV2(config_path)
        
        # Logger should exist but not write to file
        assert hasattr(generator, 'logger'), "Generator should have logger attribute"
        
        # Log file should not be created or be empty
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                content = f.read()
            assert len(content) == 0, "Log file should be empty when logging is disabled"
        
        print("✓ Disabled logging test passed")
        print("  - Logger exists but does not write to file")
        
        return True
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_config_settings():
    """Test that logging config is properly set in generator_config.json."""
    print("\nTesting production config settings...")
    
    config_path = os.path.join(
        os.path.dirname(__file__),
        'dataset_generator_v2',
        'generator_config.json'
    )
    
    assert os.path.exists(config_path), f"Config file should exist at {config_path}"
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Check logging settings
    assert 'base_settings' in config, "Config should have base_settings"
    assert 'enable_debug_logging' in config['base_settings'], \
        "Config should have enable_debug_logging"
    assert 'debug_log_path' in config['base_settings'], \
        "Config should have debug_log_path"
    
    enable_logging = config['base_settings']['enable_debug_logging']
    log_path = config['base_settings']['debug_log_path']
    
    print("✓ Production config test passed")
    print(f"  - enable_debug_logging: {enable_logging}")
    print(f"  - debug_log_path: {log_path}")
    
    return True

if __name__ == "__main__":
    try:
        # Run all tests
        test_config_settings()
        test_logger_initialization()
        test_logging_disabled()
        
        print("\n" + "="*60)
        print("✅ All logging tests passed!")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
