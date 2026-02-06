#!/usr/bin/env python3
"""
Quick validation script to ensure hotfix works correctly
Tests edge cases and verifies all requirements are met
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from vsr_plus_plus.systems.adaptive_system import AdaptiveSystem


def validate_requirements():
    """Validate all requirements from the problem statement"""
    print("\n" + "="*80)
    print("VALIDATION: All Requirements from Problem Statement")
    print("="*80)
    
    errors = []
    
    # Requirement 1: Respect config values
    print("\n1. Testing: Config values respected...")
    adaptive = AdaptiveSystem(initial_l1=0.7, initial_ms=0.15, initial_grad=0.15)
    if adaptive.l1_weight != 0.7 or adaptive.ms_weight != 0.15 or adaptive.grad_weight != 0.15:
        errors.append("Config values not respected in initialization")
    else:
        print("   ✅ Config values respected: L1=0.7, MS=0.15, Grad=0.15")
    
    # Requirement 2a: Warmup for step < 1000
    print("\n2a. Testing: Warmup phase (step < 1000)...")
    adaptive = AdaptiveSystem(initial_l1=0.6, initial_ms=0.2, initial_grad=0.2)
    pred = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    
    for step in [0, 500, 999]:
        l1, ms, grad, perc, status = adaptive.update_loss_weights(pred, target, step)
        if l1 != 0.6 or ms != 0.2 or grad != 0.2:
            errors.append(f"Weights changed during warmup at step {step}")
    print("   ✅ Weights unchanged during warmup (step < 1000)")
    
    # Requirement 2b: Settling period when step >= 1000 without history
    print("\n2b. Testing: Settling period (step >= 1000, no history)...")
    adaptive = AdaptiveSystem(initial_l1=0.6, initial_ms=0.2, initial_grad=0.2)
    
    # Simulate resuming at step 5000
    for i in range(100):
        l1, ms, grad, perc, status = adaptive.update_loss_weights(pred, target, 5000 + i)
        if i < 100:
            if l1 != 0.6 or ms != 0.2 or grad != 0.2:
                errors.append(f"Weights changed during settling at iteration {i}")
    print("   ✅ Weights unchanged during 100-step settling period")
    
    # Requirement 3: Safety guards
    print("\n3. Testing: Safety guards (MS >= 0.05, Grad >= 0.05, L1 <= 0.9)...")
    adaptive = AdaptiveSystem(initial_l1=0.6, initial_ms=0.2, initial_grad=0.2)
    adaptive.history_settling_complete = True
    adaptive._warmup_complete = True
    
    # Create very blurry image to trigger extreme adjustments
    pred_blur = torch.ones(1, 3, 64, 64) * 0.5
    target_sharp = torch.rand(1, 3, 64, 64)
    
    min_ms = 1.0
    min_grad = 1.0
    max_l1 = 0.0
    
    for step in range(1000, 1500):
        l1, ms, grad, perc, status = adaptive.update_loss_weights(pred_blur, target_sharp, step, current_l1_loss=0.05)
        min_ms = min(min_ms, ms)
        min_grad = min(min_grad, grad)
        max_l1 = max(max_l1, l1)
        
        if ms < 0.05:
            errors.append(f"MS fell below 0.05: {ms:.4f} at step {step}")
        if grad < 0.05:
            errors.append(f"Grad fell below 0.05: {grad:.4f} at step {step}")
        if l1 > 0.9:
            errors.append(f"L1 exceeded 0.9: {l1:.4f} at step {step}")
    
    print(f"   ✅ Min MS observed: {min_ms:.4f} (>= 0.05)")
    print(f"   ✅ Min Grad observed: {min_grad:.4f} (>= 0.05)")
    print(f"   ✅ Max L1 observed: {max_l1:.4f} (<= 0.9)")
    
    # Requirement 4: Smooth transitions
    print("\n4. Testing: Smooth transitions (max change per step)...")
    adaptive = AdaptiveSystem(initial_l1=0.6, initial_ms=0.2, initial_grad=0.2)
    adaptive.history_settling_complete = True
    adaptive._warmup_complete = True
    
    prev_l1, prev_ms, prev_grad = 0.6, 0.2, 0.2
    max_delta = 0.0
    
    for step in range(1000, 1200):
        l1, ms, grad, perc, status = adaptive.update_loss_weights(pred, target, step, current_l1_loss=0.01)
        
        delta_l1 = abs(l1 - prev_l1)
        delta_ms = abs(ms - prev_ms)
        delta_grad = abs(grad - prev_grad)
        
        max_delta = max(max_delta, delta_l1, delta_ms, delta_grad)
        prev_l1, prev_ms, prev_grad = l1, ms, grad
    
    if max_delta > 0.05:  # Should be well below 5%
        errors.append(f"Weight changes too large: {max_delta:.4f}")
    print(f"   ✅ Max weight change per step: {max_delta:.4f} (< 5%)")
    
    # Summary
    print("\n" + "="*80)
    if errors:
        print("❌ VALIDATION FAILED")
        print("="*80)
        for error in errors:
            print(f"  ❌ {error}")
        return False
    else:
        print("✅ ALL REQUIREMENTS VALIDATED SUCCESSFULLY!")
        print("="*80)
        print("""
Summary:
✅ Config values are respected
✅ Warmup prevents changes for step < 1000
✅ Settling period (100 steps) works when resuming
✅ Safety guards enforce: MS >= 0.05, Grad >= 0.05, L1 <= 0.9
✅ Smooth transitions with controlled change rate
✅ No more "w:0.00" in GUI!
        """)
        return True


if __name__ == '__main__':
    success = validate_requirements()
    sys.exit(0 if success else 1)
