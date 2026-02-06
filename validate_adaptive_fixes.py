#!/usr/bin/env python3
"""
Validation script for the 4 critical adaptive system bug fixes.

Tests:
1. Aggressive Mode requires Plateau (not just sharpness)
2. Cooldown doesn't reset permanently
3. Perceptual moves independently of cooldown
4. LR Boost mechanism works
"""

import os
import sys
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vsr_plus_plus.systems.adaptive_system import AdaptiveSystem
from vsr_plus_plus.training.lr_scheduler import AdaptiveLRScheduler


def test_aggressive_requires_plateau():
    """Test 1: Aggressive Mode requires Plateau"""
    print("\n" + "="*70)
    print("TEST 1: Aggressive Mode Requires Plateau (Fix 1)")
    print("="*70)
    
    adaptive = AdaptiveSystem(initial_l1=0.6, initial_ms=0.2, initial_grad=0.2, initial_perceptual=0.05)
    adaptive.plateau_counter = 100  # Not enough
    adaptive.history_settling_complete = True
    adaptive._warmup_complete = True
    
    # Create blurry prediction
    pred = torch.rand(1, 3, 64, 64) * 0.5  # Blurry
    target = torch.rand(1, 3, 64, 64)
    
    sharpness = adaptive.detect_extreme_conditions(pred, target)
    assert not adaptive.aggressive_mode, "Should NOT trigger without plateau > 300"
    print(f"✓ Plateau counter {adaptive.plateau_counter}: Aggressive mode NOT triggered (correct)")
    
    # Now set plateau counter high enough
    adaptive.plateau_counter = 350  # Enough!
    sharpness = adaptive.detect_extreme_conditions(pred, target)
    assert adaptive.aggressive_mode, "Should trigger with plateau > 300"
    print(f"✓ Plateau counter {adaptive.plateau_counter}: Aggressive mode TRIGGERED (correct)")
    
    print("\n✅ Test 1 passed: Aggressive Mode requires Plateau")


def test_cooldown_no_reset():
    """Test 2: Cooldown doesn't reset permanently"""
    print("\n" + "="*70)
    print("TEST 2: Cooldown Doesn't Reset (Fix 2)")
    print("="*70)
    
    adaptive = AdaptiveSystem(initial_l1=0.6, initial_ms=0.2, initial_grad=0.2, initial_perceptual=0.05)
    adaptive.history_settling_complete = True
    adaptive._warmup_complete = True
    
    pred = torch.rand(1, 3, 64, 64) * 0.5  # Blurry
    target = torch.rand(1, 3, 64, 64)
    
    initial_cooldown = None
    cooldown_at_step_10 = None
    
    for step in range(1100, 1300):
        adaptive.update_loss_weights(pred, target, step, current_l1_loss=0.012)
        
        if adaptive.is_in_cooldown:
            if initial_cooldown is None:
                initial_cooldown = adaptive.cooldown_steps
                print(f"✓ Step {step}: Cooldown started with {initial_cooldown} steps")
            
            if step == 1110 and cooldown_at_step_10 is None:
                cooldown_at_step_10 = adaptive.cooldown_steps
                print(f"✓ Step {step}: Cooldown at {cooldown_at_step_10} steps")
                # Should have decreased
                assert cooldown_at_step_10 < initial_cooldown, \
                    f"Cooldown should decrease ({initial_cooldown} -> {cooldown_at_step_10}), not stay same!"
    
    print("\n✅ Test 2 passed: Cooldown runs down correctly")


def test_perceptual_independent():
    """Test 3: Perceptual moves independently of cooldown"""
    print("\n" + "="*70)
    print("TEST 3: Perceptual Independent of Cooldown (Fix 3)")
    print("="*70)
    
    adaptive = AdaptiveSystem(initial_l1=0.6, initial_ms=0.2, initial_grad=0.2, initial_perceptual=0.05)
    adaptive.history_settling_complete = True
    adaptive._warmup_complete = True
    adaptive.is_in_cooldown = True  # Force cooldown
    adaptive.ema_l1_loss = 0.008  # Stable -> should increase perceptual
    
    initial_perc = adaptive.perceptual_weight
    print(f"✓ Initial perceptual weight: {initial_perc:.4f} (cooldown active)")
    
    for i in range(100):
        adaptive._update_perceptual_weight()
    
    print(f"✓ Final perceptual weight: {adaptive.perceptual_weight:.4f} (after 100 updates)")
    
    assert adaptive.perceptual_weight > initial_perc, \
        f"Perceptual should increase ({initial_perc:.4f} -> {adaptive.perceptual_weight:.4f}) despite cooldown"
    
    print(f"✓ Perceptual increased by {(adaptive.perceptual_weight - initial_perc)*100:.2f}%")
    print("\n✅ Test 3 passed: Perceptual independent of cooldown")


def test_lr_boost_mechanism():
    """Test 4: LR Boost mechanism"""
    print("\n" + "="*70)
    print("TEST 4: LR Plateau Boost Mechanism (Fix 4)")
    print("="*70)
    
    # Create dummy optimizer
    import torch.optim as optim
    dummy_params = [torch.nn.Parameter(torch.randn(10))]
    optimizer = optim.Adam(dummy_params, lr=1e-6)  # Very low LR (dead)
    
    scheduler = AdaptiveLRScheduler(
        optimizer, 
        warmup_steps=1000, 
        max_steps=100000,
        max_lr=1e-4,
        min_lr=1e-6
    )
    
    # Simulate being at step 7500 with plateau detected
    global_step = 7500
    
    print(f"✓ Initial LR: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"✓ Boost available: {scheduler.plateau_boost_available}")
    
    # Trigger plateau boost
    new_lr, phase = scheduler.step(global_step, plateau_detected=True)
    
    print(f"✓ After boost LR: {new_lr:.2e}")
    print(f"✓ Phase: {phase}")
    print(f"✓ Boost available: {scheduler.plateau_boost_available}")
    
    assert phase == 'plateau_boost', f"Should be in plateau_boost phase, got {phase}"
    assert new_lr > 1e-6, f"LR should have increased from 1e-6 to {new_lr:.2e}"
    assert not scheduler.plateau_boost_available, "Boost should be disabled after use"
    
    # Test that boost becomes available again after cooldown
    scheduler.last_boost_step = global_step - 1001  # Simulate 1001 steps ago
    scheduler.step(global_step, plateau_detected=False)
    assert scheduler.plateau_boost_available, "Boost should be available after cooldown"
    
    print("\n✅ Test 4 passed: LR Boost mechanism works")


def run_all_tests():
    """Run all validation tests"""
    print("\n" + "="*70)
    print("ADAPTIVE SYSTEM BUG FIXES - VALIDATION SUITE")
    print("="*70)
    
    try:
        test_aggressive_requires_plateau()
        test_cooldown_no_reset()
        test_perceptual_independent()
        test_lr_boost_mechanism()
        
        print("\n" + "="*70)
        print("ALL VALIDATION TESTS PASSED! ✅")
        print("="*70)
        print("\nAll 4 critical bugs are fixed:")
        print("  1. ✅ Aggressive Mode requires plateau")
        print("  2. ✅ Cooldown doesn't reset permanently")
        print("  3. ✅ Perceptual independent of cooldown")
        print("  4. ✅ LR Boost mechanism active")
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
