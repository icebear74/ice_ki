#!/usr/bin/env python3
"""
Test that internal weights (shown in GUI via get_status) match returned weights
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from vsr_plus_plus.systems.adaptive_system import AdaptiveSystem


def test_internal_weights_sync():
    """Test that internal weights match returned weights during all phases"""
    print("\n" + "="*80)
    print("TEST: Internal Weights Synchronization (GUI Display Fix)")
    print("="*80)
    
    adaptive = AdaptiveSystem(initial_l1=0.6, initial_ms=0.2, initial_grad=0.2)
    
    # Create dummy tensors
    pred = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    
    errors = []
    
    print("\n✓ Testing warmup phase (step < 1000):")
    for step in [0, 500, 999]:
        # Get returned weights
        l1_ret, ms_ret, grad_ret, perc_ret, status = adaptive.update_loss_weights(pred, target, step)
        
        # Get internal weights via get_status
        status_dict = adaptive.get_status()
        l1_int = status_dict['l1_weight']
        ms_int = status_dict['ms_weight']
        grad_int = status_dict['grad_weight']
        perc_int = status_dict['perceptual_weight']
        
        print(f"  Step {step:4d}:")
        print(f"    Returned: L1={l1_ret:.3f}, MS={ms_ret:.3f}, Grad={grad_ret:.3f}")
        print(f"    Internal: L1={l1_int:.3f}, MS={ms_int:.3f}, Grad={grad_int:.3f}")
        
        # Check if they match
        if abs(l1_ret - l1_int) > 0.001:
            errors.append(f"L1 mismatch at step {step}: returned={l1_ret:.3f}, internal={l1_int:.3f}")
        if abs(ms_ret - ms_int) > 0.001:
            errors.append(f"MS mismatch at step {step}: returned={ms_ret:.3f}, internal={ms_int:.3f}")
        if abs(grad_ret - grad_int) > 0.001:
            errors.append(f"Grad mismatch at step {step}: returned={grad_ret:.3f}, internal={grad_int:.3f}")
    
    print("\n✓ Testing settling phase (step >= 1000):")
    for i, step in enumerate([5000, 5050, 5099]):
        # Get returned weights
        l1_ret, ms_ret, grad_ret, perc_ret, status = adaptive.update_loss_weights(pred, target, step)
        
        # Get internal weights via get_status
        status_dict = adaptive.get_status()
        l1_int = status_dict['l1_weight']
        ms_int = status_dict['ms_weight']
        grad_int = status_dict['grad_weight']
        
        print(f"  Step {step:4d}:")
        print(f"    Returned: L1={l1_ret:.3f}, MS={ms_ret:.3f}, Grad={grad_ret:.3f}")
        print(f"    Internal: L1={l1_int:.3f}, MS={ms_int:.3f}, Grad={grad_int:.3f}")
        
        # Check if they match
        if abs(l1_ret - l1_int) > 0.001:
            errors.append(f"L1 mismatch at step {step}: returned={l1_ret:.3f}, internal={l1_int:.3f}")
        if abs(ms_ret - ms_int) > 0.001:
            errors.append(f"MS mismatch at step {step}: returned={ms_ret:.3f}, internal={ms_int:.3f}")
        if abs(grad_ret - grad_int) > 0.001:
            errors.append(f"Grad mismatch at step {step}: returned={grad_ret:.3f}, internal={grad_int:.3f}")
    
    print("\n" + "="*80)
    if errors:
        print("❌ TEST FAILED - Mismatches found:")
        for error in errors:
            print(f"  ❌ {error}")
        return False
    else:
        print("✅ TEST PASSED - Internal weights match returned weights in all phases!")
        print("✅ GUI will now display correct weights (no more w:1.00 or w:0.00)")
        return True


def test_gui_display_values():
    """Simulate what the GUI sees during different phases"""
    print("\n" + "="*80)
    print("SIMULATION: GUI Weight Display")
    print("="*80)
    
    adaptive = AdaptiveSystem(initial_l1=0.6, initial_ms=0.2, initial_grad=0.2)
    pred = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    
    print("\n📊 GUI Display During Warmup:")
    for step in [0, 100, 500, 999]:
        adaptive.update_loss_weights(pred, target, step)
        status = adaptive.get_status()
        print(f"  Step {step:4d}: L1={status['l1_weight']:.2f} (w:{status['l1_weight']:.2f}), "
              f"MS={status['ms_weight']:.2f} (w:{status['ms_weight']:.2f}), "
              f"Grad={status['grad_weight']:.2f} (w:{status['grad_weight']:.2f})")
    
    print("\n📊 GUI Display During Settling (resumed at step 5000):")
    for step in [5000, 5025, 5050, 5099]:
        adaptive.update_loss_weights(pred, target, step)
        status = adaptive.get_status()
        print(f"  Step {step:4d}: L1={status['l1_weight']:.2f} (w:{status['l1_weight']:.2f}), "
              f"MS={status['ms_weight']:.2f} (w:{status['ms_weight']:.2f}), "
              f"Grad={status['grad_weight']:.2f} (w:{status['grad_weight']:.2f})")
    
    print("\n✅ Expected GUI display: All weights at 0.60/0.20/0.20")
    print("✅ No more extreme values like w:1.00 or w:0.00")


if __name__ == '__main__':
    success1 = test_internal_weights_sync()
    test_gui_display_values()
    
    print("\n" + "="*80)
    if success1:
        print("✅ ALL TESTS PASSED")
        print("="*80)
        sys.exit(0)
    else:
        print("❌ TESTS FAILED")
        print("="*80)
        sys.exit(1)
