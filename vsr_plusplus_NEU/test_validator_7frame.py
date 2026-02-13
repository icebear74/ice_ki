#!/usr/bin/env python3
"""
Test script to verify validator uses correct center frame for 7-frame model

This test verifies that the validator correctly uses frame index 3 (the 4th frame)
as the center frame for a 7-frame model, not index 2 which is for 5-frame models.

Frame layouts:
- 5-Frame: [0, 1, **2**, 3, 4] → Center = Index 2
- 7-Frame: [0, 1, 2, **3**, 4, 5, 6] → Center = Index 3
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_center_frame_index():
    """Test that validator uses correct center frame index (3) for 7-frame model"""
    print("\n" + "="*80)
    print("Testing Validator Center Frame Index for 7-Frame Model")
    print("="*80 + "\n")
    
    # Read the validator code to check the index
    validator_file = os.path.join(
        os.path.dirname(__file__), 
        'training', 
        'validator.py'
    )
    
    with open(validator_file, 'r') as f:
        content = f.read()
    
    # Check that the code uses index 3 for center frame
    if 'lr_stack[:, 3]' in content and '# Center frame (7-frame model)' in content:
        print("✅ PASS: Validator correctly uses index 3 for center frame (7-frame model)")
        print("   Found: 'lr_stack[:, 3]  # Center frame (7-frame model)'")
        
        # Additional verification: ensure old index 2 is not used for center frame
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'lr_stack[:, 2]' in line and 'center' in line.lower():
                print(f"⚠️  WARNING: Old 5-frame index found at line {i+1}:")
                print(f"   {line.strip()}")
                return False
        
        print("   ✓ No old 5-frame index (2) found for center frame")
        return True
    else:
        print("❌ FAIL: Validator does not use correct index for 7-frame center frame")
        print("   Expected: 'lr_stack[:, 3]' with comment about 7-frame model")
        
        # Show what was found instead
        for line in content.split('\n'):
            if 'lr_stack[:, ' in line and 'center' in line.lower():
                print(f"   Found: {line.strip()}")
        
        return False


def test_frame_count_assumptions():
    """Test that the system is configured for 7 frames"""
    print("\n" + "="*80)
    print("Testing Frame Count Configuration")
    print("="*80 + "\n")
    
    # Check runtime config
    config_file = os.path.join(
        os.path.dirname(__file__),
        'runtime_config.json'
    )
    
    if os.path.exists(config_file):
        import json
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        n_frames = config.get('model', {}).get('n_frames', None)
        
        if n_frames == 7:
            print(f"✅ PASS: Runtime config correctly set to {n_frames} frames")
            return True
        else:
            print(f"⚠️  WARNING: Runtime config shows {n_frames} frames (expected 7)")
            return False
    else:
        print("ℹ️  INFO: No runtime_config.json found (this is optional)")
        return True


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("7-Frame Validator Tests - Center Frame Index")
    print("="*80)
    
    results = []
    
    # Test 1: Check center frame index
    results.append(test_center_frame_index())
    
    # Test 2: Check frame count configuration
    results.append(test_frame_count_assumptions())
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total}")
    
    if all(results):
        print("\n✅ All tests PASSED!")
        return 0
    else:
        print("\n❌ Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
