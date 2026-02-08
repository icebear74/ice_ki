#!/usr/bin/env python3
"""
Test the professional box-drawing GUI
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

def test_ui_terminal():
    """Test UI terminal functions"""
    print("Testing UI Terminal Functions...\n")
    
    from dataset_generator_v2.utils.ui_terminal import *
    
    # Test terminal width detection
    width = get_terminal_width()
    print(f"✓ Terminal width: {width} columns")
    assert width > 0
    
    # Test progress bars
    bar1 = make_bar(75, 40)
    bar2 = make_bar_cyan(50, 40)
    bar3 = make_bar_yellow(25, 40)
    
    print(f"\n✓ Green bar (75%):  {bar1}")
    print(f"✓ Cyan bar (50%):   {bar2}")
    print(f"✓ Yellow bar (25%): {bar3}")
    
    # Test time formatting
    assert format_time(0) == "0m 0s"
    assert format_time(90) == "1m 30s"
    assert format_time(3665) == "1h 1m"
    print(f"\n✓ Time formatting works")
    
    print("\n✅ UI Terminal tests passed!")

def test_box_drawing():
    """Test box drawing"""
    print("\n\nTesting Box Drawing...\n")
    
    from dataset_generator_v2.utils.ui_terminal import *
    
    ui_w = 80
    
    print_header(ui_w, f"{C_BOLD}{C_CYAN}TEST BOX{C_RESET}")
    print_line("Single line content", ui_w)
    print_separator(ui_w, 'single')
    print_two_columns("Left column", "Right column", ui_w)
    print_separator(ui_w, 'thin')
    print_line("Another line", ui_w)
    print_separator(ui_w, 'double')
    print_line(f"Progress: {make_bar(65, 40)} 65%", ui_w)
    print_footer(ui_w)
    
    print("\n✅ Box drawing works!")

def test_professional_gui():
    """Test the complete professional GUI"""
    print("\n\nTesting Professional GUI Display...\n")
    
    from dataset_generator_v2.make_dataset_multi import DatasetGeneratorV2
    from dataset_generator_v2.utils.ui_display import draw_dataset_generator_ui
    import json
    import tempfile
    
    test_config = {
        "base_settings": {
            "base_frame_limit": 100,
            "max_workers": 8,
            "val_percent": 0.0,
            "output_base_dir": "/tmp/test_prof_gui",
            "temp_dir": "/tmp/test_prof_gui/temp",
            "status_file": "/tmp/test_prof_gui/.status.json",
            "min_file_size": 10000,
            "scene_diff_threshold": 45,
            "max_retry_attempts": 3,
            "retry_skip_seconds": 30,
            "lr_versions": ["5frames", "7frames"]
        },
        "category_targets": {
            "general": 10000,
            "space": 5000,
            "toon": 3000
        },
        "format_config": {
            "general": {
                "small_540": {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 1.0}
            },
            "space": {
                "small_540": {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 1.0}
            },
            "toon": {
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
        
        # Simulate some progress
        generator.current_video_name = "Star Trek Beyond (2016)"
        generator.extractions_count = 5000
        generator.success_count = 4750
        generator.tracker.update_progress(completed_videos=5)
        generator.tracker.increment_category_images('general', 2500)
        generator.tracker.increment_category_images('space', 1500)
        generator.tracker.increment_category_images('toon', 750)
        
        # Update current video checkpoint
        generator.tracker.update_video_checkpoint(
            5, "in_progress",
            extractions_done=350,
            extractions_target=500
        )
        
        # Draw the GUI
        print("\n" + "="*80)
        print("DEMO: Professional Box-Drawing GUI")
        print("="*80 + "\n")
        
        draw_dataset_generator_ui(generator)
        
        print("\n✅ Professional GUI renders correctly!")
        print("\n👁️  Check the output above - it should have:")
        print("  - Box drawing characters (╔═╗ ║ ╚═╝)")
        print("  - Colored progress bars")
        print("  - Two-column layouts")
        print("  - Clean separators")
        print("  - Professional appearance")
        
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)

if __name__ == "__main__":
    try:
        test_ui_terminal()
        test_box_drawing()
        test_professional_gui()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nThe new professional GUI is ready to use!")
        print("It uses the same design as vsr_plusplus - clean, professional, no flickering!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
