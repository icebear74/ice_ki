#!/usr/bin/env python3
"""
Test to verify all formats (small_540, medium_169, large_720) are being extracted.
"""
import sys
sys.path.insert(0, 'dataset_generator_v2/utils')

from format_definitions import select_random_format, CATEGORY_FORMAT_DISTRIBUTION

def test_format_selection():
    """Test that select_random_format returns all formats over multiple calls."""
    print("="*70)
    print("TEST: All Formats Are Selected")
    print("="*70)
    
    for category in ['master', 'universal', 'space', 'toon']:
        print(f"\n{category.upper()}:")
        print(f"  Available formats: {list(CATEGORY_FORMAT_DISTRIBUTION[category].keys())}")
        print(f"  Probabilities: {CATEGORY_FORMAT_DISTRIBUTION[category]}")
        
        # Select 100 times and count
        counts = {}
        for _ in range(100):
            format_name = select_random_format(category)
            counts[format_name] = counts.get(format_name, 0) + 1
        
        print(f"  Results from 100 selections:")
        for fmt, count in sorted(counts.items()):
            expected_prob = CATEGORY_FORMAT_DISTRIBUTION[category].get(fmt, 0)
            actual_prob = count / 100
            print(f"    {fmt}: {count}/100 ({actual_prob:.2f}) - expected ~{expected_prob:.2f}")
        
        # Verify all formats were selected at least once
        expected_formats = set(CATEGORY_FORMAT_DISTRIBUTION[category].keys())
        actual_formats = set(counts.keys())
        
        if expected_formats == actual_formats:
            print(f"  ✓ All formats selected!")
        else:
            missing = expected_formats - actual_formats
            print(f"  ✗ Missing formats: {missing}")
            return False
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED - All formats are being selected!")
    print("="*70)
    return True

def test_aspect_ratios():
    """Test that aspect ratios are correct."""
    from format_definitions import FORMATS
    
    print("\n" + "="*70)
    print("TEST: Aspect Ratios")
    print("="*70)
    
    print("\n720_169 (should be 16:9 landscape):")
    gt_h, gt_w = FORMATS['720_169']['gt_size']
    lr_h, lr_w = FORMATS['720_169']['lr_size']
    
    print(f"  GT: {gt_h} tall × {gt_w} wide")
    print(f"  LR: {lr_h} tall × {lr_w} wide")
    print(f"  GT aspect (w/h): {gt_w}/{gt_h} = {gt_w/gt_h:.4f} (expected: {16/9:.4f} for 16:9)")
    print(f"  LR aspect (w/h): {lr_w}/{lr_h} = {lr_w/lr_h:.4f} (expected: {16/9:.4f} for 16:9)")
    
    # Check if it's landscape (wider than tall)
    if gt_w > gt_h and lr_w > lr_h:
        print("  ✓ Landscape orientation (wider than tall)")
    else:
        print("  ✗ Not landscape!")
        return False
    
    # Check aspect ratio
    gt_ratio = gt_w / gt_h
    expected_ratio = 16 / 9
    if abs(gt_ratio - expected_ratio) < 0.01:
        print(f"  ✓ Correct 16:9 aspect ratio")
    else:
        print(f"  ✗ Wrong aspect ratio: {gt_ratio:.4f} != {expected_ratio:.4f}")
        return False
    
    print("\n540 and 720 (should be 1:1 square):")
    for fmt in ['540', '720']:
        gt_h, gt_w = FORMATS[fmt]['gt_size']
        print(f"  {fmt}: {gt_h} × {gt_w} - {'✓ square' if gt_h == gt_w else '✗ not square'}")
    
    return True

if __name__ == '__main__':
    success = test_format_selection() and test_aspect_ratios()
    sys.exit(0 if success else 1)
