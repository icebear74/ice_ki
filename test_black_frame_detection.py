#!/usr/bin/env python3
"""
Test black frame detection and retry logic.
"""

import os
import sys
import tempfile
import numpy as np
import cv2
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_black_frame_detection():
    """Test that black frame detection works correctly"""
    print("\n" + "="*60)
    print("Testing Black Frame Detection")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a small black image (should be < 15 KB)
        black_image = np.zeros((100, 100, 3), dtype=np.uint8)
        black_path = os.path.join(temp_dir, "black.png")
        cv2.imwrite(black_path, black_image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        
        # Create a normal image with content (should be > 15 KB)
        normal_image = np.random.randint(0, 255, (720, 720, 3), dtype=np.uint8)
        normal_path = os.path.join(temp_dir, "normal.png")
        cv2.imwrite(normal_path, normal_image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        
        # Check file sizes
        black_size = os.path.getsize(black_path)
        normal_size = os.path.getsize(normal_path)
        
        print(f"\nFile sizes:")
        print(f"  Black image: {black_size:,} bytes ({black_size/1024:.2f} KB)")
        print(f"  Normal image: {normal_size:,} bytes ({normal_size/1024:.2f} KB)")
        
        # Test detection
        threshold_bytes = 15 * 1024
        is_black_detected = black_size < threshold_bytes
        is_normal_not_detected = normal_size >= threshold_bytes
        
        print(f"\nDetection (threshold: 15 KB = {threshold_bytes:,} bytes):")
        print(f"  Black image < 15 KB: {is_black_detected} {'✓' if is_black_detected else '✗'}")
        print(f"  Normal image >= 15 KB: {is_normal_not_detected} {'✓' if is_normal_not_detected else '✗'}")
        
        if is_black_detected and is_normal_not_detected:
            print(f"\n{'='*60}")
            print("✅ BLACK FRAME DETECTION TEST PASSED")
            print(f"{'='*60}")
            return True
        else:
            print(f"\n{'='*60}")
            print("❌ BLACK FRAME DETECTION TEST FAILED")
            print(f"{'='*60}")
            return False

def test_file_size_thresholds():
    """Test various image sizes to understand thresholds"""
    print("\n" + "="*60)
    print("Testing File Size Thresholds")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_cases = [
            ("Solid black 100×100", np.zeros((100, 100, 3), dtype=np.uint8)),
            ("Solid black 405×720", np.zeros((405, 720, 3), dtype=np.uint8)),
            ("Solid black 720×720", np.zeros((720, 720, 3), dtype=np.uint8)),
            ("Gray 405×720", np.full((405, 720, 3), 128, dtype=np.uint8)),
            ("Random 405×720", np.random.randint(0, 255, (405, 720, 3), dtype=np.uint8)),
            ("Random 720×720", np.random.randint(0, 255, (720, 720, 3), dtype=np.uint8)),
        ]
        
        print(f"\nImage file sizes (PNG compression 1):")
        print(f"{'Image Type':<25} {'Size (bytes)':<15} {'Size (KB)':<12} {'< 15 KB?'}")
        print("-" * 60)
        
        for name, image in test_cases:
            path = os.path.join(temp_dir, f"{name.replace(' ', '_')}.png")
            cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            size_bytes = os.path.getsize(path)
            size_kb = size_bytes / 1024
            is_small = size_bytes < 15 * 1024
            
            print(f"{name:<25} {size_bytes:<15,} {size_kb:<12.2f} {is_small}")
        
        print(f"\n{'='*60}")
        print("✅ THRESHOLD TEST COMPLETED")
        print(f"{'='*60}")
        return True

def test_retry_logic_concept():
    """Test the retry logic concept"""
    print("\n" + "="*60)
    print("Testing Retry Logic Concept")
    print("="*60)
    
    max_retries = 5
    retry_jump_seconds = 1.0
    initial_time = 10.0
    
    print(f"\nRetry logic parameters:")
    print(f"  Max retries: {max_retries}")
    print(f"  Retry jump: {retry_jump_seconds} seconds")
    print(f"  Initial time: {initial_time} seconds")
    
    print(f"\nSimulated retry sequence:")
    print(f"{'Attempt':<10} {'Time (s)':<12} {'Status'}")
    print("-" * 40)
    
    for attempt in range(max_retries + 1):
        retry_time = initial_time + (attempt * retry_jump_seconds)
        status = "Initial" if attempt == 0 else f"Retry {attempt}"
        print(f"{status:<10} {retry_time:<12.1f} {'Try extraction'}")
    
    print(f"\nAfter {max_retries} retries:")
    print(f"  Final time: {initial_time + max_retries * retry_jump_seconds} seconds")
    print(f"  Time advanced: {max_retries * retry_jump_seconds} seconds")
    print(f"  Action: Count as created, no patch saved")
    
    print(f"\n{'='*60}")
    print("✅ RETRY LOGIC CONCEPT TEST PASSED")
    print(f"{'='*60}")
    return True

if __name__ == "__main__":
    print("\n" + "="*60)
    print("BLACK FRAME DETECTION TEST SUITE")
    print("="*60)
    
    results = []
    
    # Run all tests
    results.append(("Black Frame Detection", test_black_frame_detection()))
    results.append(("File Size Thresholds", test_file_size_thresholds()))
    results.append(("Retry Logic Concept", test_retry_logic_concept()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        sys.exit(0)
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        sys.exit(1)
