#!/usr/bin/env python3
"""
Test to verify aspect ratio and stacking issues
"""

import numpy as np
import cv2

def test_aspect_ratio():
    """Test aspect ratio calculations"""
    print("=" * 60)
    print("ASPECT RATIO TEST")
    print("=" * 60)
    
    # Current configuration
    gt_size_720_169 = (405, 720)  # From format_definitions.py line 14
    lr_size_720_169 = (135, 240)  # From format_definitions.py line 15
    
    print(f"\nCurrent 720_169 format:")
    print(f"  gt_size: {gt_size_720_169}")
    print(f"  lr_size: {lr_size_720_169}")
    
    # In OpenCV/numpy, shape is (height, width, channels)
    # So (405, 720) means height=405, width=720
    height, width = gt_size_720_169
    ratio = width / height
    print(f"\n  Height: {height}")
    print(f"  Width: {width}")
    print(f"  Aspect ratio (width/height): {ratio:.4f}")
    print(f"  16/9 = {16/9:.4f}")
    print(f"  9/16 = {9/16:.4f}")
    
    if abs(ratio - 16/9) < 0.01:
        print(f"  ✓ This is 16:9 (landscape)")
    elif abs(ratio - 9/16) < 0.01:
        print(f"  ✓ This is 9:16 (portrait)")
    
    # What would 9:16 be?
    print(f"\n  For 9:16 (portrait), we would need:")
    print(f"    - Width smaller than height")
    print(f"    - Dimensions like (720, 405) - height=720, width=405")
    print(f"    - Ratio: 405/720 = {405/720:.4f} = 9/16")
    
    # Test with actual image
    print(f"\n  Creating test image with shape {gt_size_720_169}:")
    test_img = np.zeros(gt_size_720_169 + (3,), dtype=np.uint8)
    print(f"    Image shape: {test_img.shape}")
    print(f"    Height x Width x Channels: {test_img.shape[0]} x {test_img.shape[1]} x {test_img.shape[2]}")
    
    if test_img.shape[0] < test_img.shape[1]:
        print(f"    → This is LANDSCAPE (width > height)")
    else:
        print(f"    → This is PORTRAIT (height > width)")

def test_stacking():
    """Test stacking direction"""
    print("\n" + "=" * 60)
    print("STACKING TEST")
    print("=" * 60)
    
    # Create 7 test frames
    lr_h, lr_w = 240, 240
    n_frames = 7
    
    lr_frames = []
    for i in range(n_frames):
        frame = np.ones((lr_h, lr_w, 3), dtype=np.uint8) * (i * 30)
        lr_frames.append(frame)
    
    print(f"\nCreated {n_frames} frames, each {lr_h}x{lr_w}")
    
    # Test horizontal stacking (axis=1)
    print(f"\nHorizontal stacking (axis=1):")
    horizontal = np.concatenate(lr_frames, axis=1)
    print(f"  Result shape: {horizontal.shape}")
    print(f"  Height x Width: {horizontal.shape[0]} x {horizontal.shape[1]}")
    print(f"  → Frames are SIDE-BY-SIDE (nebeneinander)")
    
    # Test vertical stacking (axis=0)
    print(f"\nVertical stacking (axis=0):")
    vertical = np.concatenate(lr_frames, axis=0)
    print(f"  Result shape: {vertical.shape}")
    print(f"  Height x Width: {vertical.shape[0]} x {vertical.shape[1]}")
    print(f"  → Frames are UNDERNEATH (untereinander)")
    
    # What's expected for 7 frames of 240x240?
    print(f"\nFor 7 frames of {lr_w}x{lr_h}:")
    print(f"  Horizontal (axis=1): {lr_h} x {lr_w * n_frames} = {lr_h} x {lr_w * n_frames}")
    print(f"  Vertical (axis=0): {lr_h * n_frames} x {lr_w} = {lr_h * n_frames} x {lr_w}")
    
    print(f"\nCurrent code uses axis=1, which is HORIZONTAL (nebeneinander)")
    print(f"This seems CORRECT based on the problem statement!")

if __name__ == "__main__":
    test_aspect_ratio()
    test_stacking()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\n1. Aspect Ratio:")
    print("   Current (405, 720) is 16:9 LANDSCAPE")
    print("   User wants 9:16 PORTRAIT → Need (720, 405)")
    print("\n2. Stacking:")
    print("   Current axis=1 is HORIZONTAL (side-by-side)")
    print("   This appears CORRECT - user says 'nebeneinander' (side-by-side)")
    print("   BUT user says 'stacking ist falsch'... confusing!")
