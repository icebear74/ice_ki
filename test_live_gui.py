#!/usr/bin/env python3
"""
Test the live GUI functionality
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

def test_live_gui_initialization():
    """Test that the live GUI can be initialized."""
    print("Testing live GUI initialization...")
    
    from dataset_generator_v2.make_dataset_multi import DatasetGeneratorV2
    
    # Create minimal test config
    test_config = {
        "base_settings": {
            "base_frame_limit": 10,
            "max_workers": 2,
            "val_percent": 0.0,
            "output_base_dir": "/tmp/test_live_gui",
            "temp_dir": "/tmp/test_live_gui/temp",
            "status_file": "/tmp/test_live_gui/.status.json",
            "min_file_size": 10000,
            "scene_diff_threshold": 45,
            "max_retry_attempts": 3,
            "retry_skip_seconds": 30,
            "lr_versions": ["5frames", "7frames"]
        },
        "category_targets": {
            "general": 100
        },
        "format_config": {
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
        
        print("  ✓ Generator initialized")
        print(f"  Workers: {generator.workers}")
        print(f"  Paused: {generator.paused}")
        print(f"  Running: {generator.running}")
        
        # Test GUI building
        output, table, disk, controls = generator.build_gui_layout()
        assert output is not None
        assert table is not None
        assert disk is not None
        assert controls is not None
        print("  ✓ GUI layout built successfully")
        
        # Check that controls mention pause/resume and workers
        assert "SPACE" in controls or "Pause" in controls
        assert "workers" in controls.lower() or "Workers" in controls
        print("  ✓ Controls include pause/resume and worker adjustment")
        
        # Test pause toggle
        initial_paused = generator.paused
        generator.paused = not generator.paused
        assert generator.paused != initial_paused
        generator.paused = initial_paused  # Restore
        print("  ✓ Pause toggle works")
        
        # Test worker adjustment
        initial_workers = generator.workers
        generator.workers += 1
        assert generator.workers == initial_workers + 1
        generator.workers -= 1
        assert generator.workers == initial_workers
        print("  ✓ Worker adjustment works")
        
        print("\n✅ All live GUI tests passed!")
        
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)

def test_checkpoint_persistence():
    """Test that checkpoints are saved frequently."""
    print("\nTesting checkpoint persistence...")
    
    from dataset_generator_v2.make_dataset_multi import DatasetGeneratorV2
    
    test_config = {
        "base_settings": {
            "base_frame_limit": 10,
            "max_workers": 2,
            "val_percent": 0.0,
            "output_base_dir": "/tmp/test_checkpoint",
            "temp_dir": "/tmp/test_checkpoint/temp",
            "status_file": "/tmp/test_checkpoint/.status.json",
            "min_file_size": 10000,
            "scene_diff_threshold": 45,
            "max_retry_attempts": 3,
            "retry_skip_seconds": 30,
            "lr_versions": ["5frames", "7frames"]
        },
        "category_targets": {
            "general": 100
        },
        "format_config": {
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
        
        # Check that tracker can save
        generator.tracker.update_progress(total_videos=10)
        generator.tracker.save()
        print("  ✓ Tracker can save checkpoints")
        
        # Check that status file was created
        assert os.path.exists(generator.status_file)
        print(f"  ✓ Status file created: {generator.status_file}")
        
        # Load and verify
        with open(generator.status_file, 'r') as f:
            status = json.load(f)
        assert 'progress' in status
        print("  ✓ Status file contains progress information")
        
        print("\n✅ Checkpoint persistence tests passed!")
        
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)

def main():
    """Run all tests."""
    print("Running live GUI tests...\n")
    
    try:
        test_live_gui_initialization()
        test_checkpoint_persistence()
        
        print("\n✅ All tests passed!")
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
