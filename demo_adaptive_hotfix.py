#!/usr/bin/env python3
"""
Demonstration of Adaptive System Hotfix
Shows how weights behave during different phases
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from vsr_plus_plus.systems.adaptive_system import AdaptiveSystem


def demonstrate_new_training():
    """Demonstrate behavior for new training (step 0 -> 1200)"""
    print("\n" + "="*80)
    print("SCENARIO 1: New Training (step 0 -> 1200)")
    print("="*80)
    
    adaptive = AdaptiveSystem(initial_l1=0.6, initial_ms=0.2, initial_grad=0.2)
    
    # Create dummy tensors
    pred = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    
    print("\n┌─ Phase 1: Warmup (step < 1000)")
    print("│  Expected: Weights stay at 0.6/0.2/0.2")
    print("└─")
    
    for step in [0, 250, 500, 750, 999]:
        l1, ms, grad, perc, status = adaptive.update_loss_weights(pred, target, step)
        mode_emoji = "🚀" if status['mode'] == 'Warmup' else "⚙️"
        print(f"  Step {step:4d}: {mode_emoji} {status['mode']:8s} | "
              f"L1={l1:.3f}, MS={ms:.3f}, Grad={grad:.3f}")
    
    print("\n┌─ Phase 2: Settling (step >= 1000, collecting history)")
    print("│  Expected: Weights stay at 0.6/0.2/0.2 for 100 steps")
    print("└─")
    
    for step in [1000, 1025, 1050, 1075, 1099]:
        l1, ms, grad, perc, status = adaptive.update_loss_weights(pred, target, step)
        mode_emoji = "📊" if status['mode'] == 'Settling' else "✅"
        progress = status.get('settling_progress', '')
        print(f"  Step {step:4d}: {mode_emoji} {status['mode']:8s} | "
              f"L1={l1:.3f}, MS={ms:.3f}, Grad={grad:.3f} | {progress}")
    
    print("\n┌─ Phase 3: Adaptive (after settling)")
    print("│  Expected: Weights can now adapt gradually")
    print("└─")
    
    for step in [1100, 1150, 1200]:
        l1, ms, grad, perc, status = adaptive.update_loss_weights(pred, target, step, current_l1_loss=0.01)
        mode_emoji = "⚡" if status['mode'] == 'Aggressive' else "🔄"
        print(f"  Step {step:4d}: {mode_emoji} {status['mode']:8s} | "
              f"L1={l1:.3f}, MS={ms:.3f}, Grad={grad:.3f}")


def demonstrate_resumed_training():
    """Demonstrate behavior when resuming from checkpoint"""
    print("\n\n" + "="*80)
    print("SCENARIO 2: Resumed Training (checkpoint at step 5000)")
    print("="*80)
    
    adaptive = AdaptiveSystem(initial_l1=0.6, initial_ms=0.2, initial_grad=0.2)
    
    # Create dummy tensors
    pred = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    
    print("\n┌─ Resuming from checkpoint at step 5000")
    print("│  Expected: Settling period kicks in (100 steps)")
    print("│  Weights stay at 0.6/0.2/0.2 during settling")
    print("└─")
    
    for i, step in enumerate([5000, 5020, 5040, 5060, 5080, 5099]):
        l1, ms, grad, perc, status = adaptive.update_loss_weights(pred, target, step)
        mode_emoji = "📊"
        progress = status.get('settling_progress', '')
        print(f"  Step {step:4d}: {mode_emoji} {status['mode']:8s} | "
              f"L1={l1:.3f}, MS={ms:.3f}, Grad={grad:.3f} | {progress}")
    
    print("\n┌─ After settling period")
    print("│  Expected: Adaptive behavior resumes")
    print("└─")
    
    for step in [5100, 5200]:
        l1, ms, grad, perc, status = adaptive.update_loss_weights(pred, target, step, current_l1_loss=0.01)
        mode_emoji = "⚡" if status['mode'] == 'Aggressive' else "🔄"
        print(f"  Step {step:4d}: {mode_emoji} {status['mode']:8s} | "
              f"L1={l1:.3f}, MS={ms:.3f}, Grad={grad:.3f}")


def demonstrate_safety_guards():
    """Demonstrate safety guards in action"""
    print("\n\n" + "="*80)
    print("SCENARIO 3: Safety Guards Enforcement")
    print("="*80)
    
    adaptive = AdaptiveSystem(initial_l1=0.6, initial_ms=0.2, initial_grad=0.2)
    
    # Skip to adaptive phase
    adaptive.history_settling_complete = True
    adaptive._warmup_complete = True
    
    # Create very blurry prediction to trigger aggressive adjustments
    pred = torch.ones(1, 3, 64, 64) * 0.5  # Flat image
    target = torch.rand(1, 3, 64, 64)  # Sharp image
    
    print("\n┌─ Testing with extremely blurry predictions")
    print("│  Expected: MS and Grad never go below 0.05")
    print("│  Expected: L1 never exceeds 0.9")
    print("└─")
    
    for step in range(1000, 1300, 50):
        l1, ms, grad, perc, status = adaptive.update_loss_weights(pred, target, step, current_l1_loss=0.05)
        
        # Check violations
        violations = []
        if ms < 0.05:
            violations.append(f"MS={ms:.3f}")
        if grad < 0.05:
            violations.append(f"Grad={grad:.3f}")
        if l1 > 0.9:
            violations.append(f"L1={l1:.3f}")
        
        status_emoji = "✅" if not violations else "❌"
        violation_str = ", ".join(violations) if violations else "All guards OK"
        
        print(f"  Step {step:4d}: {status_emoji} L1={l1:.3f}, MS={ms:.3f}, Grad={grad:.3f} | {violation_str}")


def main():
    print("\n" + "="*80)
    print("  ADAPTIVE SYSTEM HOTFIX - DEMONSTRATION")
    print("  Showing the fix for extreme weight initialization bug")
    print("="*80)
    
    demonstrate_new_training()
    demonstrate_resumed_training()
    demonstrate_safety_guards()
    
    print("\n\n" + "="*80)
    print("  SUMMARY")
    print("="*80)
    print("""
✅ Weights respect config values (0.6/0.2/0.2 by default)
✅ Early warmup prevents changes during first 1000 steps
✅ Settling period prevents premature adaptation when resuming
✅ Safety guards ensure MS >= 0.05, Grad >= 0.05, L1 <= 0.9
✅ No more "w:0.00" in the GUI!
    """)


if __name__ == '__main__':
    main()
