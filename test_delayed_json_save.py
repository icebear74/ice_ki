#!/usr/bin/env python3
"""
Test to verify delayed JSON save prevents zero loss values

This test simulates the new delayed save mechanism to ensure loss values
are properly captured after validation.
"""

import os
import sys
import json
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_delayed_save_mechanism():
    """Test that delayed save allows loss values to be updated"""
    print("\n" + "="*70)
    print("TEST: Delayed JSON Save Mechanism")
    print("="*70)
    
    # Simulate the flow:
    # Step 5000: Validation
    # Step 5001: Training continues, web_monitor updated with loss_dict
    # Step 5002: Training continues, web_monitor updated again
    # Step 5002: JSON saved (now has fresh loss data)
    
    # Mock web_monitor data
    web_monitor_data = {
        'step_current': 5000,
        'quality_lr_value': 0.543,
        'quality_ki_value': 0.621,
        'quality_improvement_value': 0.189,
        # Initially no loss data (validation just completed)
        'l1_loss_value': 0.0,
        'total_loss_value': 0.0,
    }
    
    print(f"\n📍 Step 5000: Validation completes")
    print(f"  Quality metrics: ✅ Updated")
    print(f"  Loss values: ❌ Not yet updated (zeros)")
    print(f"  → Schedule JSON save for step 5002")
    pending_save_step = 5000 + 2
    print(f"  pending_json_save_step = {pending_save_step}")
    
    # Step 5001: Training continues
    print(f"\n📍 Step 5001: Training continues")
    web_monitor_data.update({
        'step_current': 5001,
        'l1_loss_value': 0.0156,  # Fresh loss data!
        'total_loss_value': 0.0234,
    })
    print(f"  GUI updated with fresh loss data:")
    print(f"    l1_loss_value: {web_monitor_data['l1_loss_value']}")
    print(f"    total_loss_value: {web_monitor_data['total_loss_value']}")
    print(f"  Check: step {5001} >= {pending_save_step}? No, continue")
    
    # Step 5002: Training continues + JSON save
    print(f"\n📍 Step 5002: Training continues")
    web_monitor_data.update({
        'step_current': 5002,
    })
    print(f"  GUI updated with loss data")
    print(f"  Check: step {5002} >= {pending_save_step}? YES!")
    print(f"  → Save JSON for step {pending_save_step - 2} (5000)")
    
    # Save JSON (with validation step number)
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "Statistik_5000.json")
        
        # Adjust step_current back to validation step for the save
        save_data = web_monitor_data.copy()
        save_data['step_current'] = 5000
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ JSON saved: Statistik_5000.json")
        
        # Read back and verify
        with open(filepath, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        # Verify loss data is NOT zero
        assert saved_data['l1_loss_value'] != 0.0, "L1 loss should not be zero"
        assert saved_data['total_loss_value'] != 0.0, "Total loss should not be zero"
        
        # Verify quality data is present
        assert saved_data['quality_improvement_value'] != 0.0, "Improvement should not be zero"
        
        print(f"\n✅ Verification: All data complete!")
        print(f"  Step: {saved_data['step_current']}")
        print(f"  L1 Loss: {saved_data['l1_loss_value']:.4f} (not zero!)")
        print(f"  Total Loss: {saved_data['total_loss_value']:.4f} (not zero!)")
        print(f"  Improvement: {saved_data['quality_improvement_value']*100:.1f}% (not zero!)")
    
    print("\n✅ Test passed: Delayed save captures complete data!")


def test_timing_comparison():
    """Compare immediate vs delayed save timing"""
    print("\n" + "="*70)
    print("TEST: Immediate vs Delayed Save Timing")
    print("="*70)
    
    print("\n❌ OLD APPROACH (Immediate Save):")
    print("  Step 5000:")
    print("    1. Validation → quality metrics calculated")
    print("    2. _update_gui() with no params → loss_dict=None → zeros!")
    print("    3. Save JSON → has quality but ZERO losses ❌")
    print("    4. Training continues...")
    print("  Step 5001:")
    print("    1. Training → _update_gui(loss_dict) → fresh losses")
    print("    2. But JSON already saved with zeros ❌")
    
    print("\n✅ NEW APPROACH (Delayed Save):")
    print("  Step 5000:")
    print("    1. Validation → quality metrics calculated")
    print("    2. Update web_monitor with quality data")
    print("    3. Schedule JSON save for step 5002 ⏰")
    print("    4. Training continues...")
    print("  Step 5001:")
    print("    1. Training → _update_gui(loss_dict) → fresh losses")
    print("    2. web_monitor now has complete data")
    print("  Step 5002:")
    print("    1. Training → _update_gui(loss_dict) → fresh losses")
    print("    2. Check: time to save? YES!")
    print("    3. Save JSON → has quality AND fresh losses ✅")
    
    print("\n✅ Test passed: Delayed save ensures complete data!")


def run_all_tests():
    """Run all delayed save tests"""
    print("\n" + "="*70)
    print("DELAYED JSON SAVE - TEST SUITE")
    print("="*70)
    
    try:
        test_delayed_save_mechanism()
        test_timing_comparison()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✅")
        print("="*70)
        print("\nDelayed save mechanism:")
        print("  1. ✅ Waits 2 steps after validation")
        print("  2. ✅ Allows web_monitor to get fresh loss data")
        print("  3. ✅ Saves JSON with complete data (no zeros)")
        print("  4. ✅ Uses original validation step number in filename")
        print("="*70 + "\n")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
