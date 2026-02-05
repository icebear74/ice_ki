#!/usr/bin/env python3
"""
Debug script to diagnose adaptive system weight issue
Run this to check if the adaptive system is working correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from vsr_plus_plus.systems.adaptive_system import AdaptiveSystem


def test_initialization():
    print("\n" + "="*80)
    print("TEST 1: Check Initialization")
    print("="*80)
    
    # Create adaptive system with default values
    adaptive = AdaptiveSystem(initial_l1=0.6, initial_ms=0.2, initial_grad=0.2)
    
    print(f"\nInitial values stored:")
    print(f"  initial_l1:   {adaptive.initial_l1}")
    print(f"  initial_ms:   {adaptive.initial_ms}")
    print(f"  initial_grad: {adaptive.initial_grad}")
    
    print(f"\nInternal weights:")
    print(f"  l1_weight:   {adaptive.l1_weight}")
    print(f"  ms_weight:   {adaptive.ms_weight}")
    print(f"  grad_weight: {adaptive.grad_weight}")
    
    status = adaptive.get_status()
    print(f"\nget_status() returns:")
    print(f"  l1_weight:   {status['l1_weight']}")
    print(f"  ms_weight:   {status['ms_weight']}")
    print(f"  grad_weight: {status['grad_weight']}")
    
    if (adaptive.initial_l1 == 0.6 and adaptive.initial_ms == 0.2 and adaptive.initial_grad == 0.2 and
        status['l1_weight'] == 0.6 and status['ms_weight'] == 0.2 and status['grad_weight'] == 0.2):
        print("\n✅ Initialization OK")
        return True
    else:
        print("\n❌ Initialization FAILED!")
        return False


def test_warmup_phase():
    print("\n" + "="*80)
    print("TEST 2: Check Warmup Phase (step < 1000)")
    print("="*80)
    
    adaptive = AdaptiveSystem(initial_l1=0.6, initial_ms=0.2, initial_grad=0.2)
    
    # Create dummy tensors
    pred = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    
    print("\nCalling update_loss_weights at step 0:")
    l1, ms, grad, perc, status_dict = adaptive.update_loss_weights(pred, target, 0)
    
    print(f"  Returned: L1={l1:.2f}, MS={ms:.2f}, Grad={grad:.2f}")
    print(f"  Mode: {status_dict['mode']}")
    
    print(f"\nInternal weights after call:")
    print(f"  l1_weight:   {adaptive.l1_weight}")
    print(f"  ms_weight:   {adaptive.ms_weight}")
    print(f"  grad_weight: {adaptive.grad_weight}")
    
    status = adaptive.get_status()
    print(f"\nget_status() returns:")
    print(f"  l1_weight:   {status['l1_weight']}")
    print(f"  ms_weight:   {status['ms_weight']}")
    print(f"  grad_weight: {status['grad_weight']}")
    
    if (l1 == 0.6 and ms == 0.2 and grad == 0.2 and
        status['l1_weight'] == 0.6 and status['ms_weight'] == 0.2 and status['grad_weight'] == 0.2):
        print("\n✅ Warmup phase OK")
        return True
    else:
        print("\n❌ Warmup phase FAILED!")
        print(f"   Expected: L1=0.60, MS=0.20, Grad=0.20")
        print(f"   Got:      L1={status['l1_weight']:.2f}, MS={status['ms_weight']:.2f}, Grad={status['grad_weight']:.2f}")
        return False


def test_settling_phase():
    print("\n" + "="*80)
    print("TEST 3: Check Settling Phase (step >= 1000, no history)")
    print("="*80)
    
    adaptive = AdaptiveSystem(initial_l1=0.6, initial_ms=0.2, initial_grad=0.2)
    
    # Create dummy tensors
    pred = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    
    print("\nCalling update_loss_weights at step 5000 (simulating resume):")
    l1, ms, grad, perc, status_dict = adaptive.update_loss_weights(pred, target, 5000)
    
    print(f"  Returned: L1={l1:.2f}, MS={ms:.2f}, Grad={grad:.2f}")
    print(f"  Mode: {status_dict['mode']}")
    
    status = adaptive.get_status()
    print(f"\nget_status() returns:")
    print(f"  l1_weight:   {status['l1_weight']}")
    print(f"  ms_weight:   {status['ms_weight']}")
    print(f"  grad_weight: {status['grad_weight']}")
    
    if (l1 == 0.6 and ms == 0.2 and grad == 0.2 and
        status['l1_weight'] == 0.6 and status['ms_weight'] == 0.2 and status['grad_weight'] == 0.2):
        print("\n✅ Settling phase OK")
        return True
    else:
        print("\n❌ Settling phase FAILED!")
        print(f"   Expected: L1=0.60, MS=0.20, Grad=0.20")
        print(f"   Got:      L1={status['l1_weight']:.2f}, MS={status['ms_weight']:.2f}, Grad={status['grad_weight']:.2f}")
        return False


def test_gui_display():
    print("\n" + "="*80)
    print("TEST 4: Simulate GUI Display")
    print("="*80)
    
    adaptive = AdaptiveSystem(initial_l1=0.6, initial_ms=0.2, initial_grad=0.2)
    pred = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    
    # Simulate what happens in training
    print("\nStep 0 (warmup):")
    adaptive.update_loss_weights(pred, target, 0)
    status = adaptive.get_status()
    print(f"  GUI would show: L1 (w:{status['l1_weight']:.2f}), MS (w:{status['ms_weight']:.2f}), Grad (w:{status['grad_weight']:.2f})")
    
    print("\nStep 500 (still warmup):")
    adaptive.update_loss_weights(pred, target, 500)
    status = adaptive.get_status()
    print(f"  GUI would show: L1 (w:{status['l1_weight']:.2f}), MS (w:{status['ms_weight']:.2f}), Grad (w:{status['grad_weight']:.2f})")
    
    # New adaptive system for resume scenario
    print("\n--- Simulating resume from checkpoint at step 5000 ---")
    adaptive2 = AdaptiveSystem(initial_l1=0.6, initial_ms=0.2, initial_grad=0.2)
    
    print("\nStep 5000 (just resumed, settling):")
    adaptive2.update_loss_weights(pred, target, 5000)
    status = adaptive2.get_status()
    print(f"  GUI would show: L1 (w:{status['l1_weight']:.2f}), MS (w:{status['ms_weight']:.2f}), Grad (w:{status['grad_weight']:.2f})")
    
    expected_display = "L1 (w:0.60), MS (w:0.20), Grad (w:0.20)"
    print(f"\n✅ Expected display: {expected_display}")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("ADAPTIVE SYSTEM DEBUG SCRIPT")
    print("="*80)
    
    results = []
    results.append(("Initialization", test_initialization()))
    results.append(("Warmup Phase", test_warmup_phase()))
    results.append(("Settling Phase", test_settling_phase()))
    
    test_gui_display()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_pass = all(result for _, result in results)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
    
    if all_pass:
        print("\n✅ All tests passed - adaptive system is working correctly")
        print("\nIf you're still seeing w:1.00/0.00/0.00 in the GUI, then:")
        print("  1. Make sure you've restarted the training process")
        print("  2. Check that you're using the latest code (git pull)")
        print("  3. Delete any old checkpoint files that might have cached state")
    else:
        print("\n❌ Some tests failed - there's a problem with the code")
    
    print("="*80 + "\n")
    
    sys.exit(0 if all_pass else 1)
