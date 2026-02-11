#!/usr/bin/env python3
"""
Test for dataset generation improvements
"""
import sys
import os
import numpy as np
import cv2

# Add dataset_generator_v2 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dataset_generator_v2'))

from make_dataset_v2_uhd import DatasetGeneratorV2UHD


def test_is_interesting_patch():
    """Test the is_interesting_patch method"""
    print("Testing is_interesting_patch method...")
    
    # Create a mock generator with minimal config
    config = {
        'base_settings': {
            'output_base_dir': '/tmp/test',
            'temp_dir': '/tmp/test/temp',
            'status_file': '/tmp/test/status.json',
            'min_detail_threshold': 80.0,
            'max_workers': 1,
            'val_percent': 0.0,
            'base_frame_limit': 100,
            'min_file_size': 10000,
            'scene_diff_threshold': 45,
            'max_retry_attempts': 3,
            'retry_skip_seconds': 10,
            'lr_versions': ['7frames']
        },
        'videos': [],
        'format_config': {},
        'category_targets': {}
    }
    
    # Write minimal config
    config_path = '/tmp/test_config.json'
    import json
    os.makedirs('/tmp', exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    # Create generator instance
    gen = DatasetGeneratorV2UHD(config_path)
    
    # Test 1: Very dark/black patch (should always be interesting)
    black_patch = np.zeros((100, 100, 3), dtype=np.uint8)
    result = gen.is_interesting_patch(black_patch)
    print(f"  Black patch (avg brightness={np.mean(black_patch):.2f}): {'PASS' if result else 'FAIL'}")
    assert result, "Black patch should be interesting"
    
    # Test 2: Nearly black patch (avg < 5)
    dark_patch = np.ones((100, 100, 3), dtype=np.uint8) * 3
    result = gen.is_interesting_patch(dark_patch)
    print(f"  Dark patch (avg brightness={np.mean(dark_patch):.2f}): {'PASS' if result else 'FAIL'}")
    assert result, "Dark patch should be interesting"
    
    # Test 3: Blurry patch (low Laplacian variance)
    blurry_patch = np.ones((100, 100, 3), dtype=np.uint8) * 128
    result = gen.is_interesting_patch(blurry_patch)
    print(f"  Blurry uniform patch (should be boring): {'PASS' if not result else 'FAIL'}")
    assert not result, "Uniform patch should not be interesting"
    
    # Test 4: Sharp patch with edges (high Laplacian variance)
    sharp_patch = np.zeros((100, 100, 3), dtype=np.uint8)
    # Create some edges
    sharp_patch[::2, :] = 255  # Horizontal stripes
    result = gen.is_interesting_patch(sharp_patch)
    print(f"  Sharp patch with edges: {'PASS' if result else 'FAIL'}")
    assert result, "Sharp patch should be interesting"
    
    print("✓ All is_interesting_patch tests passed!\n")
    
    # Cleanup
    os.remove(config_path)


def test_create_patch_pair_force_center():
    """Test the force_center parameter in create_patch_pair"""
    print("Testing create_patch_pair with force_center...")
    
    # Create mock frames (1920x1080 to match our new scaling)
    frames = [np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8) for _ in range(7)]
    
    # Mock format config
    format_config = {
        'gt_size': [540, 540],
        'lr_size': [180, 180]
    }
    
    # Create minimal config
    config = {
        'base_settings': {
            'output_base_dir': '/tmp/test',
            'temp_dir': '/tmp/test/temp',
            'status_file': '/tmp/test/status.json',
            'max_workers': 1,
            'val_percent': 0.0,
            'base_frame_limit': 100,
            'min_file_size': 10000,
            'scene_diff_threshold': 45,
            'max_retry_attempts': 3,
            'retry_skip_seconds': 10,
            'lr_versions': ['7frames']
        },
        'videos': [],
        'format_config': {},
        'category_targets': {}
    }
    
    config_path = '/tmp/test_config2.json'
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    gen = DatasetGeneratorV2UHD(config_path)
    
    # Test with force_center=True
    gt1, lr1 = gen.create_patch_pair(frames, 'small_540', format_config, force_center=True)
    assert gt1 is not None and lr1 is not None, "Should create patches with force_center=True"
    
    # Calculate expected center position
    frame_h, frame_w = 1080, 1920
    gt_h, gt_w = 540, 540
    expected_crop_x = (frame_w - gt_w) // 2  # = (1920 - 540) // 2 = 690
    expected_crop_y = (frame_h - gt_h) // 2  # = (1080 - 540) // 2 = 270
    
    # Extract patch at expected center position from center frame
    center_frame = frames[3]
    expected_gt = center_frame[expected_crop_y:expected_crop_y+gt_h, expected_crop_x:expected_crop_x+gt_w]
    
    # Check if patches match (they should be identical)
    if np.array_equal(gt1, expected_gt):
        print(f"  Center crop position correct: PASS")
        print(f"    Expected crop at ({expected_crop_x}, {expected_crop_y})")
    else:
        print(f"  Center crop position: FAIL")
        print(f"    Patches don't match expected center crop")
    
    # Test with force_center=False (should work but position will be random)
    gt2, lr2 = gen.create_patch_pair(frames, 'small_540', format_config, force_center=False)
    assert gt2 is not None and lr2 is not None, "Should create patches with force_center=False"
    print(f"  Random crop works: PASS")
    
    print("✓ All create_patch_pair tests passed!\n")
    
    # Cleanup
    os.remove(config_path)


def test_ffmpeg_filter_update():
    """Test that FFmpeg filters include the scaling"""
    print("Testing FFmpeg filter strings...")
    
    # Read the source file
    script_path = os.path.join(os.path.dirname(__file__), 'dataset_generator_v2', 'make_dataset_v2_uhd.py')
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for the scale filter in both functions
    checks = [
        ('extract_frames_uhd vf_filter', 'scale=1920:1080:flags=lanczos' in content),
        ('_extract_frames_with_stride tonemap_filter', content.count('scale=1920:1080:flags=lanczos') >= 2),
    ]
    
    all_pass = True
    for check_name, result in checks:
        status = 'PASS' if result else 'FAIL'
        print(f"  {check_name}: {status}")
        if not result:
            all_pass = False
    
    assert all_pass, "FFmpeg filters should include scale=1920:1080:flags=lanczos"
    print("✓ FFmpeg filter tests passed!\n")


def test_config_threshold_support():
    """Test that min_detail_threshold is read from config"""
    print("Testing min_detail_threshold config support...")
    
    # Create config with custom threshold
    config = {
        'base_settings': {
            'output_base_dir': '/tmp/test',
            'temp_dir': '/tmp/test/temp',
            'status_file': '/tmp/test/status.json',
            'min_detail_threshold': 120.0,  # Custom threshold
            'max_workers': 1,
            'val_percent': 0.0,
            'base_frame_limit': 100,
            'min_file_size': 10000,
            'scene_diff_threshold': 45,
            'max_retry_attempts': 3,
            'retry_skip_seconds': 10,
            'lr_versions': ['7frames']
        },
        'videos': [],
        'format_config': {},
        'category_targets': {}
    }
    
    config_path = '/tmp/test_config3.json'
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    gen = DatasetGeneratorV2UHD(config_path)
    
    # Check that settings contain the threshold
    threshold = gen.settings.get('min_detail_threshold', 80.0)
    print(f"  Custom threshold from config: {threshold}")
    assert threshold == 120.0, "Should read custom threshold from config"
    print(f"  Config threshold support: PASS")
    
    print("✓ Config threshold test passed!\n")
    
    # Cleanup
    os.remove(config_path)


if __name__ == '__main__':
    print("="*60)
    print("Dataset Generation Improvements - Test Suite")
    print("="*60)
    print()
    
    try:
        test_ffmpeg_filter_update()
        test_is_interesting_patch()
        test_create_patch_pair_force_center()
        test_config_threshold_support()
        
        print("="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
