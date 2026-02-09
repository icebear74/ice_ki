#!/usr/bin/env python3
"""
Test for the unpacking fix in make_dataset_v2_uhd.py
Verifies that get_output_dirs_for_format is handled correctly
"""

import sys
import os

# Add dataset_generator_v2 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dataset_generator_v2'))

def test_get_output_dirs_for_format():
    """Test that get_output_dirs_for_format returns a dictionary"""
    print("=" * 60)
    print("TEST: get_output_dirs_for_format Return Type")
    print("=" * 60)
    
    from utils.format_definitions import get_output_dirs_for_format
    
    # Test the function
    result = get_output_dirs_for_format(
        base_path="/tmp/test",
        category="master",
        format_name="small_540",
        lr_frames=7
    )
    
    print(f"Return type: {type(result)}")
    print(f"Return value: {result}")
    
    # Verify it's a dictionary
    if not isinstance(result, dict):
        print("❌ FAILED: Expected dictionary, got", type(result))
        return False
    
    # Verify it has the expected keys
    expected_keys = {'gt', 'lr', 'val_gt', 'val_lr'}
    actual_keys = set(result.keys())
    
    if actual_keys != expected_keys:
        print(f"❌ FAILED: Expected keys {expected_keys}, got {actual_keys}")
        return False
    
    print("✅ SUCCESS: Returns dictionary with correct keys")
    print(f"   Keys: {list(result.keys())}")
    print(f"   GT dir: {result['gt']}")
    print(f"   LR dir: {result['lr']}")
    
    return True


def test_unpacking_pattern():
    """Test the correct unpacking pattern"""
    print("\n" + "=" * 60)
    print("TEST: Correct Unpacking Pattern")
    print("=" * 60)
    
    from utils.format_definitions import get_output_dirs_for_format
    
    try:
        # This should work - correct pattern
        output_dirs = get_output_dirs_for_format(
            base_path="/tmp/test",
            category="master",
            format_name="small_540",
            lr_frames=7
        )
        gt_dir = output_dirs['gt']
        lr_dir = output_dirs['lr']
        
        print("✅ SUCCESS: Correct unpacking pattern works")
        print(f"   gt_dir: {gt_dir}")
        print(f"   lr_dir: {lr_dir}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wrong_unpacking_pattern():
    """Test that the old wrong pattern fails"""
    print("\n" + "=" * 60)
    print("TEST: Wrong Unpacking Pattern (Should Fail)")
    print("=" * 60)
    
    from utils.format_definitions import get_output_dirs_for_format
    
    try:
        # This should FAIL - wrong pattern (what we're fixing)
        gt_dir, lr_dir = get_output_dirs_for_format(
            base_path="/tmp/test",
            category="master",
            format_name="small_540",
            lr_frames=7
        )
        
        print("❌ UNEXPECTED: Wrong pattern didn't fail!")
        print("   This means the function signature changed?")
        return False
        
    except ValueError as e:
        if "too many values to unpack" in str(e):
            print("✅ SUCCESS: Wrong pattern correctly fails with unpacking error")
            print(f"   Error: {e}")
            return True
        else:
            print(f"❌ FAILED: Got different error: {e}")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: Got unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Unpacking Fix Verification".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    results = []
    
    # Test 1: Return type
    results.append(("Return Type", test_get_output_dirs_for_format()))
    
    # Test 2: Correct unpacking
    results.append(("Correct Unpacking", test_unpacking_pattern()))
    
    # Test 3: Wrong unpacking should fail
    results.append(("Wrong Unpacking Fails", test_wrong_unpacking_pattern()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    sys.exit(0 if passed == total else 1)
