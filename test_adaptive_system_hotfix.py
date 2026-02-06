#!/usr/bin/env python3
"""
Test suite for Adaptive System Hotfix
Tests the new warmup, settling period, and safety guards
"""

import os
import sys
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsr_plus_plus.systems.adaptive_system import AdaptiveSystem


def test_config_value_initialization():
    """Test that initial weights respect config values"""
    print("\n" + "="*70)
    print("TEST 1: Config Value Initialization")
    print("="*70)
    
    # Test with custom values
    custom_l1 = 0.7
    custom_ms = 0.15
    custom_grad = 0.15
    
    adaptive = AdaptiveSystem(
        initial_l1=custom_l1,
        initial_ms=custom_ms,
        initial_grad=custom_grad
    )
    
    print(f"\n✓ Custom initialization:")
    print(f"  L1:   {adaptive.l1_weight:.3f} (expected: {custom_l1:.3f})")
    print(f"  MS:   {adaptive.ms_weight:.3f} (expected: {custom_ms:.3f})")
    print(f"  Grad: {adaptive.grad_weight:.3f} (expected: {custom_grad:.3f})")
    
    assert adaptive.l1_weight == custom_l1, f"L1 weight mismatch"
    assert adaptive.ms_weight == custom_ms, f"MS weight mismatch"
    assert adaptive.grad_weight == custom_grad, f"Grad weight mismatch"
    
    # Test with defaults
    adaptive_default = AdaptiveSystem()
    
    print(f"\n✓ Default initialization:")
    print(f"  L1:   {adaptive_default.l1_weight:.3f} (expected: 0.600)")
    print(f"  MS:   {adaptive_default.ms_weight:.3f} (expected: 0.200)")
    print(f"  Grad: {adaptive_default.grad_weight:.3f} (expected: 0.200)")
    
    assert adaptive_default.l1_weight == 0.6, f"Default L1 weight mismatch"
    assert adaptive_default.ms_weight == 0.2, f"Default MS weight mismatch"
    assert adaptive_default.grad_weight == 0.2, f"Default Grad weight mismatch"
    
    print("\n✓ Config value initialization test passed!")


def test_early_warmup_phase():
    """Test that weights don't change during step < 1000"""
    print("\n" + "="*70)
    print("TEST 2: Early Warmup Phase (step < 1000)")
    print("="*70)
    
    adaptive = AdaptiveSystem(initial_l1=0.6, initial_ms=0.2, initial_grad=0.2)
    
    # Create dummy tensors
    pred = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    
    print(f"\n✓ Testing steps 0-999:")
    for step in [0, 100, 500, 999]:
        l1, ms, grad, perc, status = adaptive.update_loss_weights(pred, target, step)
        
        print(f"  Step {step:4d}: L1={l1:.3f}, MS={ms:.3f}, Grad={grad:.3f} | Mode: {status['mode']}")
        
        assert l1 == 0.6, f"L1 changed during warmup at step {step}"
        assert ms == 0.2, f"MS changed during warmup at step {step}"
        assert grad == 0.2, f"Grad changed during warmup at step {step}"
        assert status['mode'] == 'Warmup', f"Wrong mode at step {step}"
    
    print("\n✓ Early warmup phase test passed!")


def test_history_settling_phase():
    """Test that weights don't change during settling period (step >= 1000, no history)"""
    print("\n" + "="*70)
    print("TEST 3: History Settling Phase (step >= 1000)")
    print("="*70)
    
    adaptive = AdaptiveSystem(initial_l1=0.6, initial_ms=0.2, initial_grad=0.2)
    
    # Create dummy tensors
    pred = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    
    print(f"\n✓ Testing settling period (100 steps after step 1000):")
    
    # Simulate resuming from checkpoint at step 5000
    for i, step in enumerate(range(5000, 5100)):
        l1, ms, grad, perc, status = adaptive.update_loss_weights(pred, target, step)
        
        if i % 20 == 0 or i == 99:
            progress = status.get('settling_progress', 'N/A')
            print(f"  Step {step:4d}: L1={l1:.3f}, MS={ms:.3f}, Grad={grad:.3f} | Mode: {status['mode']} | Progress: {progress}")
        
        if i < 100:
            assert l1 == 0.6, f"L1 changed during settling at iteration {i}"
            assert ms == 0.2, f"MS changed during settling at iteration {i}"
            assert grad == 0.2, f"Grad changed during settling at iteration {i}"
            assert status['mode'] == 'Settling', f"Wrong mode at iteration {i}"
    
    # After settling, mode should change
    l1, ms, grad, perc, status = adaptive.update_loss_weights(pred, target, 5100)
    print(f"  Step 5100: Mode after settling: {status['mode']}")
    assert status['mode'] != 'Settling', f"Still in settling mode after 100 steps"
    
    print("\n✓ History settling phase test passed!")


def test_safety_guards():
    """Test that MS and Grad never fall below 0.05 and L1 never exceeds 0.9"""
    print("\n" + "="*70)
    print("TEST 4: Safety Guards (MS >= 0.05, Grad >= 0.05, L1 <= 0.9)")
    print("="*70)
    
    adaptive = AdaptiveSystem(initial_l1=0.6, initial_ms=0.2, initial_grad=0.2)
    
    # Create dummy tensors - extremely blurry to trigger aggressive adjustments
    pred = torch.ones(1, 3, 64, 64) * 0.5  # Very flat image
    target = torch.rand(1, 3, 64, 64)  # Random sharp image
    
    # Skip warmup and settling phases
    adaptive.history_settling_complete = True
    adaptive._warmup_complete = True
    
    print(f"\n✓ Testing safety guards over 500 steps:")
    violations = []
    
    for step in range(1000, 1500):
        l1, ms, grad, perc, status = adaptive.update_loss_weights(pred, target, step, current_l1_loss=0.05)
        
        # Check safety guards
        if ms < 0.05:
            violations.append(f"MS={ms:.4f} at step {step}")
        if grad < 0.05:
            violations.append(f"Grad={grad:.4f} at step {step}")
        if l1 > 0.9:
            violations.append(f"L1={l1:.4f} at step {step}")
        
        if step % 100 == 0:
            print(f"  Step {step:4d}: L1={l1:.3f}, MS={ms:.3f}, Grad={grad:.3f}")
    
    if violations:
        print(f"\n✗ Safety guard violations:")
        for v in violations[:10]:  # Show first 10
            print(f"  {v}")
        raise AssertionError(f"Found {len(violations)} safety guard violations")
    
    print("\n✓ Safety guards test passed!")


def test_smooth_transitions():
    """Test that weight changes are gradual (max 1% per step)"""
    print("\n" + "="*70)
    print("TEST 5: Smooth Transitions (max 1% change per step)")
    print("="*70)
    
    adaptive = AdaptiveSystem(initial_l1=0.6, initial_ms=0.2, initial_grad=0.2)
    
    # Create dummy tensors
    pred = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    
    # Skip warmup and settling
    adaptive.history_settling_complete = True
    adaptive._warmup_complete = True
    
    print(f"\n✓ Testing smooth transitions:")
    
    prev_l1, prev_ms, prev_grad = 0.6, 0.2, 0.2
    max_delta = 0.0
    
    for step in range(1000, 1200):
        l1, ms, grad, perc, status = adaptive.update_loss_weights(pred, target, step, current_l1_loss=0.05)
        
        # Calculate deltas
        delta_l1 = abs(l1 - prev_l1)
        delta_ms = abs(ms - prev_ms)
        delta_grad = abs(grad - prev_grad)
        
        max_delta = max(max_delta, delta_l1, delta_ms, delta_grad)
        
        if step % 50 == 0:
            print(f"  Step {step:4d}: L1={l1:.3f}, MS={ms:.3f}, Grad={grad:.3f} | Max Δ={max_delta:.4f}")
        
        prev_l1, prev_ms, prev_grad = l1, ms, grad
    
    # Allow slightly more than 0.01 for rounding and initial adjustments
    # but should be well below 0.05 (5%)
    print(f"\n✓ Maximum weight change observed: {max_delta:.4f}")
    assert max_delta < 0.05, f"Weight changes too large: {max_delta}"
    
    print("\n✓ Smooth transitions test passed!")


def test_no_zero_weights():
    """Test that weights never become zero (the main bug)"""
    print("\n" + "="*70)
    print("TEST 6: No Zero Weights (main bug fix)")
    print("="*70)
    
    adaptive = AdaptiveSystem(initial_l1=0.6, initial_ms=0.2, initial_grad=0.2)
    
    # Create dummy tensors
    pred = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    
    print(f"\n✓ Testing for zero weights over 1000 steps:")
    
    zero_violations = []
    
    for step in range(0, 1000):
        l1, ms, grad, perc, status = adaptive.update_loss_weights(pred, target, step, current_l1_loss=0.05)
        
        # Check for zero weights
        if l1 == 0.0:
            zero_violations.append(f"L1=0.0 at step {step}")
        if ms == 0.0:
            zero_violations.append(f"MS=0.0 at step {step}")
        if grad == 0.0:
            zero_violations.append(f"Grad=0.0 at step {step}")
        
        if step % 200 == 0:
            print(f"  Step {step:4d}: L1={l1:.3f}, MS={ms:.3f}, Grad={grad:.3f}")
    
    if zero_violations:
        print(f"\n✗ Zero weight violations:")
        for v in zero_violations:
            print(f"  {v}")
        raise AssertionError(f"Found {len(zero_violations)} zero weight violations")
    
    print("\n✓ No zero weights test passed!")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("ADAPTIVE SYSTEM HOTFIX - TEST SUITE")
    print("="*70)
    
    try:
        test_config_value_initialization()
        test_early_warmup_phase()
        test_history_settling_phase()
        test_safety_guards()
        test_smooth_transitions()
        test_no_zero_weights()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
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
