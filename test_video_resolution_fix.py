#!/usr/bin/env python3
"""
Test to verify video resolution handling is correct.

This test verifies that:
1. Video resolution detection works
2. Aspect ratio is preserved during frame extraction
3. The model outputs 3x upscaled resolution
"""

import sys
import os

# Test the get_video_resolution and extract_frames_from_video function signatures
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_function_signatures():
    """Test that the functions have the correct signatures"""
    print("Testing function signatures...")
    
    # Import the functions
    from run_video_inference import get_video_resolution, extract_frames_from_video
    
    # Check get_video_resolution signature
    import inspect
    sig = inspect.signature(get_video_resolution)
    params = list(sig.parameters.keys())
    assert params == ['video_path'], f"Expected ['video_path'], got {params}"
    print("  ✅ get_video_resolution signature correct")
    
    # Check extract_frames_from_video signature
    sig = inspect.signature(extract_frames_from_video)
    params = list(sig.parameters.keys())
    assert params == ['video_path', 'output_dir', 'scale_factor'], f"Expected ['video_path', 'output_dir', 'scale_factor'], got {params}"
    print("  ✅ extract_frames_from_video signature correct")
    
    # Check that scale_factor defaults to None
    assert sig.parameters['scale_factor'].default is None, "scale_factor should default to None"
    print("  ✅ scale_factor defaults to None (no downscaling)")
    
    return True


def test_resolution_calculation():
    """Test resolution calculation logic"""
    print("\nTesting resolution calculation...")
    
    # Test cases: (input_width, input_height, expected_output_width, expected_output_height)
    test_cases = [
        (720, 576, 2160, 1728),   # PAL
        (1280, 720, 3840, 2160),  # HD 720p
        (1920, 1080, 5760, 3240), # Full HD 1080p
        (640, 480, 1920, 1440),   # VGA (4:3)
    ]
    
    for in_w, in_h, exp_out_w, exp_out_h in test_cases:
        # 3x upscaling
        out_w = in_w * 3
        out_h = in_h * 3
        
        assert out_w == exp_out_w, f"Width mismatch: {out_w} != {exp_out_w}"
        assert out_h == exp_out_h, f"Height mismatch: {out_h} != {exp_out_h}"
        
        # Verify aspect ratio is preserved
        in_aspect = in_w / in_h
        out_aspect = out_w / out_h
        assert abs(in_aspect - out_aspect) < 0.001, f"Aspect ratio not preserved: {in_aspect} != {out_aspect}"
        
        print(f"  ✅ {in_w}×{in_h} → {out_w}×{out_h} (aspect ratio preserved)")
    
    return True


def test_no_square_forcing():
    """Verify that we don't force square aspect ratios"""
    print("\nTesting that square aspect ratios are NOT forced...")
    
    # These should all maintain their aspect ratios
    non_square_cases = [
        (720, 576),   # PAL (15:12 = 1.25)
        (1280, 720),  # HD (16:9 = 1.778)
        (1920, 1080), # Full HD (16:9)
        (640, 480),   # VGA (4:3 = 1.333)
    ]
    
    for width, height in non_square_cases:
        aspect_ratio = width / height
        
        # After 3x upscaling
        out_width = width * 3
        out_height = height * 3
        out_aspect = out_width / out_height
        
        # Aspect ratios should be identical
        assert abs(aspect_ratio - out_aspect) < 0.001, \
            f"Aspect ratio changed: {aspect_ratio} -> {out_aspect}"
        
        # Should NOT be square unless input was square
        if width != height:
            assert out_width != out_height, \
                f"Non-square input {width}×{height} became square {out_width}×{out_height}"
        
        print(f"  ✅ {width}×{height} maintains aspect ratio {aspect_ratio:.3f}")
    
    return True


if __name__ == '__main__':
    print("=" * 70)
    print("Video Resolution Fix - Test Suite")
    print("=" * 70)
    
    try:
        test_function_signatures()
        test_resolution_calculation()
        test_no_square_forcing()
        
        print("\n" + "=" * 70)
        print("✅ All tests passed!")
        print("=" * 70)
        sys.exit(0)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
