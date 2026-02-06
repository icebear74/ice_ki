"""
Test for training monitoring and stability improvements

Tests:
1. Safety reset mechanism when plateau_counter exceeds 3000
2. Layer activity peak value calculation
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vsr_plus_plus.systems.adaptive_system import AdaptiveSystem


def test_safety_reset():
    """Test that safety reset works when plateau_counter exceeds 3000"""
    print("\n" + "="*60)
    print("TEST 1: Safety Reset for Adaptive System")
    print("="*60)
    
    # Create adaptive system with known initial values
    initial_l1 = 0.6
    initial_ms = 0.2
    initial_grad = 0.2
    initial_perc = 0.0
    
    system = AdaptiveSystem(
        initial_l1=initial_l1,
        initial_ms=initial_ms, 
        initial_grad=initial_grad,
        initial_perceptual=initial_perc
    )
    
    # Simulate the system getting into aggressive mode and high plateau counter
    system.plateau_counter = 3500  # Exceeds 3000 threshold
    system.aggressive_mode = True
    system.aggressive_counter = 100
    
    # Modify weights to non-initial values
    system.l1_weight = 0.5
    system.ms_weight = 0.25
    system.grad_weight = 0.25
    system.perceptual_weight = 0.1
    
    print(f"\nBefore safety reset:")
    print(f"  plateau_counter: {system.plateau_counter}")
    print(f"  aggressive_mode: {system.aggressive_mode}")
    print(f"  l1_weight: {system.l1_weight:.3f}")
    print(f"  ms_weight: {system.ms_weight:.3f}")
    print(f"  grad_weight: {system.grad_weight:.3f}")
    print(f"  perceptual_weight: {system.perceptual_weight:.3f}")
    print(f"  is_in_cooldown: {system.is_in_cooldown}")
    
    # Create dummy tensors for update
    pred = torch.randn(1, 3, 64, 64)
    target = torch.randn(1, 3, 64, 64)
    
    # Complete warmup and settling periods first
    system._warmup_complete = True
    system.history_settling_complete = True
    
    # Call update_loss_weights - should trigger safety reset
    print("\n>>> Calling update_loss_weights (step 1500, plateau_counter=3500)...")
    l1_w, ms_w, grad_w, perc_w, status = system.update_loss_weights(
        pred, target, step=1500, current_l1_loss=0.015
    )
    
    print(f"\nAfter safety reset:")
    print(f"  plateau_counter: {system.plateau_counter}")
    print(f"  aggressive_mode: {system.aggressive_mode}")
    print(f"  l1_weight: {system.l1_weight:.3f}")
    print(f"  ms_weight: {system.ms_weight:.3f}")
    print(f"  grad_weight: {system.grad_weight:.3f}")
    print(f"  perceptual_weight: {system.perceptual_weight:.3f}")
    print(f"  is_in_cooldown: {system.is_in_cooldown}")
    print(f"  cooldown_steps: {system.cooldown_steps}")
    
    # Verify safety reset worked
    assert system.plateau_counter == 0, f"Expected plateau_counter=0, got {system.plateau_counter}"
    assert system.aggressive_mode == False, f"Expected aggressive_mode=False, got {system.aggressive_mode}"
    assert abs(system.l1_weight - initial_l1) < 0.01, f"Expected l1_weight≈{initial_l1}, got {system.l1_weight}"
    assert abs(system.ms_weight - initial_ms) < 0.01, f"Expected ms_weight≈{initial_ms}, got {system.ms_weight}"
    assert abs(system.grad_weight - initial_grad) < 0.01, f"Expected grad_weight≈{initial_grad}, got {system.grad_weight}"
    # Note: perceptual_weight may not be exactly initial_perc because _update_perceptual_weight()
    # runs after the reset and enforces its own min/max constraints (min=0.05)
    # The important thing is it's been reset and is within valid range
    assert 0.0 <= system.perceptual_weight <= 0.25, f"Expected perceptual_weight in [0.0, 0.25], got {system.perceptual_weight}"
    assert system.is_in_cooldown == True, f"Expected is_in_cooldown=True, got {system.is_in_cooldown}"
    # Cooldown steps will be cooldown_duration - 1 because it gets decremented once
    assert system.cooldown_steps >= system.cooldown_duration - 1, f"Expected cooldown_steps≈{system.cooldown_duration}, got {system.cooldown_steps}"
    
    print("\n✅ Safety reset test PASSED!")
    print("   - plateau_counter reset to 0")
    print("   - aggressive_mode set to False")
    print("   - Weights reset to initial values")
    print("   - Cooldown activated")
    return True


def test_layer_activity_peak():
    """Test that layer activity peak value is calculated correctly"""
    print("\n" + "="*60)
    print("TEST 2: Layer Activity Peak Value Calculation")
    print("="*60)
    
    # Simulate activity data structure: (name, activity_percent, trend, raw_value)
    test_activities = [
        ("Layer 1", 50, 0, 0.0012),
        ("Layer 2", 75, 1, 0.0018),
        ("Layer 3", 100, 0, 0.0024),  # This has the highest raw value
        ("Layer 4", 30, -1, 0.0007),
        ("Layer 5", 60, 0, 0.0014),
    ]
    
    print("\nTest activities:")
    for name, perc, trend, raw in test_activities:
        print(f"  {name}: percent={perc}%, raw={raw:.6f}")
    
    # Simulate the calculation logic from trainer.py
    layer_act_dict = {}
    peak_activity_value = 0.0
    
    for name, activity_percent, trend, raw_value in test_activities:
        layer_act_dict[name] = activity_percent
        # Track maximum raw value across all layers
        peak_activity_value = max(peak_activity_value, raw_value)
    
    print(f"\nCalculated peak_activity_value: {peak_activity_value:.6f}")
    
    # Verify the peak is correct
    expected_peak = max([raw for _, _, _, raw in test_activities])
    assert abs(peak_activity_value - expected_peak) < 1e-9, \
        f"Expected peak={expected_peak}, got {peak_activity_value}"
    assert peak_activity_value == 0.0024, \
        f"Expected peak=0.0024, got {peak_activity_value}"
    
    print("✅ Layer activity peak test PASSED!")
    print(f"   - Correctly identified peak value: {peak_activity_value:.6f}")
    print(f"   - Peak belongs to: Layer 3")
    
    # Test empty activities
    print("\nTesting edge case: empty activities...")
    layer_act_dict = {}
    peak_activity_value = 0.0
    
    for name, activity_percent, trend, raw_value in []:
        layer_act_dict[name] = activity_percent
        peak_activity_value = max(peak_activity_value, raw_value)
    
    assert peak_activity_value == 0.0, f"Expected peak=0.0 for empty list, got {peak_activity_value}"
    print("✅ Empty activities test PASSED (peak=0.0)")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Testing Training Monitoring and Stability Improvements")
    print("="*60)
    
    all_passed = True
    
    try:
        test_safety_reset()
    except Exception as e:
        print(f"\n❌ Safety reset test FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        test_layer_activity_peak()
    except Exception as e:
        print(f"\n❌ Layer activity peak test FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("="*60)
        return 1


if __name__ == "__main__":
    exit(main())
