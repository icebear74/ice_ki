#!/usr/bin/env python3
"""
Quick check: Is the fix in the code?
Run this to verify the fix is present WITHOUT stopping training
"""

import sys
import os

def check_fix_present():
    print("\n" + "="*80)
    print("CHECKING IF FIX IS PRESENT IN CODE")
    print("="*80)
    
    file_path = "vsr_plus_plus/systems/adaptive_system.py"
    
    if not os.path.exists(file_path):
        print(f"\n❌ ERROR: File not found: {file_path}")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check for the synchronization code
    checks = [
        ("Initial value storage", "self.initial_l1 = initial_l1"),
        ("Warmup sync (L1)", "if step < 1000:" in content and "self.l1_weight = self.initial_l1"),
        ("Warmup sync (MS)", "self.ms_weight = self.initial_ms"),
        ("Warmup sync (Grad)", "self.grad_weight = self.initial_grad"),
        ("Settling sync", "if not self.history_settling_complete:" in content and "self.l1_weight = self.initial_l1"),
        ("Safety guards", "self.ms_weight = max(0.05, self.ms_weight)"),
    ]
    
    print("\nChecking for fix components:")
    all_present = True
    
    for name, search_str in checks:
        if isinstance(search_str, str):
            present = search_str in content
        else:
            present = search_str  # Already a boolean
        
        status = "✅" if present else "❌"
        print(f"  {status} {name}")
        
        if not present:
            all_present = False
    
    print("\n" + "="*80)
    
    if all_present:
        print("✅ FIX IS PRESENT IN CODE!")
        print("\nIf you're still seeing w:1.00/0.00/0.00, then:")
        print("  → Your training process is using OLD code from memory")
        print("  → You MUST restart the training process")
        print("\nSteps to restart:")
        print("  1. Stop training (Ctrl+C)")
        print("  2. Restart: python vsr_plus_plus/train.py")
        print("  3. Check GUI: should show w:0.60/0.20/0.20")
    else:
        print("❌ FIX IS NOT PRESENT!")
        print("\nYou need to pull the latest code:")
        print("  git pull origin copilot/hotfix-adaptive-system-weights")
    
    print("="*80 + "\n")
    
    return all_present


if __name__ == '__main__':
    success = check_fix_present()
    sys.exit(0 if success else 1)
