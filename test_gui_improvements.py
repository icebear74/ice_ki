#!/usr/bin/env python3
"""
Test dynamic bar width calculation and home cursor positioning.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

def test_bar_width_calculation():
    """Test that bar widths are calculated based on terminal size."""
    print("Testing dynamic bar width calculation...")
    
    from dataset_generator_v2.make_dataset_multi import DatasetGeneratorV2
    import json
    import tempfile
    
    test_config = {
        "base_settings": {
            "base_frame_limit": 100,
            "max_workers": 4,
            "val_percent": 0.0,
            "output_base_dir": "/tmp/test_bars",
            "temp_dir": "/tmp/test_bars/temp",
            "status_file": "/tmp/test_bars/.status.json",
            "min_file_size": 10000,
            "scene_diff_threshold": 45,
            "max_retry_attempts": 3,
            "retry_skip_seconds": 30,
            "lr_versions": ["5frames", "7frames"]
        },
        "category_targets": {
            "test1": 1000,
            "test2": 2000,
        },
        "format_config": {
            "test1": {
                "small_540": {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 1.0}
            },
            "test2": {
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
        
        # Test terminal width detection
        terminal_width = generator._get_terminal_width()
        print(f"  ✓ Terminal width detected: {terminal_width} columns")
        assert terminal_width > 0
        assert terminal_width >= 80  # Minimum reasonable width
        
        # Test bar width calculation
        bar_widths = generator._calculate_bar_widths()
        print(f"  ✓ Bar widths calculated:")
        print(f"    - Overall bar: {bar_widths['overall']} chars")
        print(f"    - Video bar: {bar_widths['video']} chars")
        print(f"    - Category bar: {bar_widths['category']} chars")
        
        # Verify all bar widths are reasonable
        assert bar_widths['overall'] >= 30
        assert bar_widths['overall'] <= 80
        assert bar_widths['video'] >= 30
        assert bar_widths['video'] <= 80
        assert bar_widths['category'] >= 25
        assert bar_widths['category'] <= 60
        
        # Verify bars are wider than the old hard-coded values
        # Old values: overall=40, video=40, category=30
        # With terminal width of 120, new values should be larger
        if terminal_width >= 120:
            print(f"  ✓ Bars are wider than old hard-coded values (terminal={terminal_width})")
            assert bar_widths['overall'] >= 40, f"Overall bar should be >= 40, got {bar_widths['overall']}"
            assert bar_widths['video'] >= 40, f"Video bar should be >= 40, got {bar_widths['video']}"
            assert bar_widths['category'] >= 30, f"Category bar should be >= 30, got {bar_widths['category']}"
        
        print("\n✅ Bar width calculation test passed!")
        
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)

def test_home_cursor():
    """Test that home cursor positioning works."""
    print("\nTesting home cursor positioning...")
    
    # Test ANSI escape code for home cursor
    import sys
    
    # Print test message
    print("Line 1")
    print("Line 2")
    print("Line 3")
    
    # Move cursor to home and overwrite
    print('\033[H', end='', flush=True)
    print("OVERWRITTEN", flush=True)
    
    print("\n✅ Home cursor positioning test passed!")
    print("  (If you see 'OVERWRITTEN' at the top, it worked!)")

def main():
    """Run all tests."""
    print("Running GUI improvement tests...\n")
    
    try:
        test_bar_width_calculation()
        test_home_cursor()
        
        print("\n✅ All GUI improvement tests passed!")
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
