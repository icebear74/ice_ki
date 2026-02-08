#!/usr/bin/env python3
"""
Real-time diagnostic: Add logging to adaptive system to see what's happening
This will help us understand why w:1.00 is still showing
"""

import sys
import os

def add_debug_logging():
    """Add debug logging to adaptive_system.py"""
    
    file_path = "vsr_plus_plus/systems/adaptive_system.py"
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the __init__ method and add logging after initialization
    modified = False
    new_lines = []
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        # Add logging right after setting initial weights
        if "self.perceptual_weight = initial_perceptual" in line and not modified:
            new_lines.append("\n")
            new_lines.append("        # DEBUG: Log initialization\n")
            new_lines.append("        print(f\"\\n{'='*80}\")\n")
            new_lines.append("        print(f\"ADAPTIVE SYSTEM INITIALIZED:\")\n")
            new_lines.append("        print(f\"  initial_l1={initial_l1}, initial_ms={initial_ms}, initial_grad={initial_grad}\")\n")
            new_lines.append("        print(f\"  self.initial_l1={self.initial_l1}, self.initial_ms={self.initial_ms}, self.initial_grad={self.initial_grad}\")\n")
            new_lines.append("        print(f\"  self.l1_weight={self.l1_weight}, self.ms_weight={self.ms_weight}, self.grad_weight={self.grad_weight}\")\n")
            new_lines.append("        print(f\"{'='*80}\\n\")\n")
            modified = True
    
    if modified:
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
        print("✅ Added debug logging to __init__")
    else:
        print("❌ Could not add debug logging")
    
    # Also add logging to update_loss_weights
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    log_added = False
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        # Add logging at the start of warmup phase return
        if "return self.initial_l1, self.initial_ms, self.initial_grad, self.initial_perceptual, status" in line and "step < 1000" in ''.join(lines[max(0, i-10):i]) and not log_added:
            # Insert BEFORE the return
            new_lines.pop()  # Remove the return line we just added
            new_lines.append("            # DEBUG: Log what we're returning\n")
            new_lines.append("            if step % 100 == 0:  # Log every 100 steps\n")
            new_lines.append("                print(f\"Step {step}: Warmup phase\")\n")
            new_lines.append("                print(f\"  Returning: L1={self.initial_l1:.2f}, MS={self.initial_ms:.2f}, Grad={self.initial_grad:.2f}\")\n")
            new_lines.append("                print(f\"  Internal: L1={self.l1_weight:.2f}, MS={self.ms_weight:.2f}, Grad={self.grad_weight:.2f}\")\n")
            new_lines.append(line)  # Add back the return line
            log_added = True
    
    if log_added:
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
        print("✅ Added debug logging to update_loss_weights warmup")
    else:
        print("❌ Could not add warmup logging")


def create_test_script():
    """Create a test script that mimics what training does"""
    
    script = '''#!/usr/bin/env python3
"""
Test: What values are actually being passed to AdaptiveSystem?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to load the config
try:
    from vsr_plus_plus import config
    print("\\n" + "="*80)
    print("CONFIG VALUES:")
    print("="*80)
    print(f"L1_WEIGHT = {config.L1_WEIGHT}")
    print(f"MS_WEIGHT = {config.MS_WEIGHT}")
    print(f"GRAD_WEIGHT = {config.GRAD_WEIGHT}")
    print(f"Sum = {config.L1_WEIGHT + config.MS_WEIGHT + config.GRAD_WEIGHT}")
    print("="*80)
except Exception as e:
    print(f"Could not load config: {e}")
    print("Using example config...")
    try:
        with open('vsr_plus_plus/config.py.example', 'r') as f:
            content = f.read()
            # Extract values
            import re
            l1 = re.search(r'L1_WEIGHT = ([0-9.]+)', content)
            ms = re.search(r'MS_WEIGHT = ([0-9.]+)', content)
            grad = re.search(r'GRAD_WEIGHT = ([0-9.]+)', content)
            if l1 and ms and grad:
                print(f"\\nExample config values:")
                print(f"L1_WEIGHT = {l1.group(1)}")
                print(f"MS_WEIGHT = {ms.group(1)}")
                print(f"GRAD_WEIGHT = {grad.group(1)}")
    except Exception as e2:
        print(f"Could not read example config: {e2}")

# Now test the adaptive system
import torch
from vsr_plus_plus.systems.adaptive_system import AdaptiveSystem

print("\\n" + "="*80)
print("TEST: Creating AdaptiveSystem")
print("="*80)

adaptive = AdaptiveSystem(initial_l1=0.6, initial_ms=0.2, initial_grad=0.2)

print("\\nAfter creation:")
status = adaptive.get_status()
print(f"  get_status(): L1={status['l1_weight']}, MS={status['ms_weight']}, Grad={status['grad_weight']}")

print("\\nCalling update_loss_weights at step 0...")
pred = torch.rand(1, 3, 64, 64)
target = torch.rand(1, 3, 64, 64)
l1, ms, grad, perc, status_dict = adaptive.update_loss_weights(pred, target, 0)

print(f"  Returned: L1={l1}, MS={ms}, Grad={grad}")

status = adaptive.get_status()
print(f"  get_status() after: L1={status['l1_weight']}, MS={status['ms_weight']}, Grad={status['grad_weight']}")

if status['l1_weight'] == 0.6 and status['ms_weight'] == 0.2 and status['grad_weight'] == 0.2:
    print("\\n✅ Values are CORRECT (0.6/0.2/0.2)")
else:
    print(f"\\n❌ Values are WRONG! Expected 0.6/0.2/0.2, got {status['l1_weight']}/{status['ms_weight']}/{status['grad_weight']}")
'''
    
    with open('test_actual_values.py', 'w') as f:
        f.write(script)
    
    print("✅ Created test_actual_values.py")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("DIAGNOSTIC TOOL: Adding Debug Logging")
    print("="*80)
    
    add_debug_logging()
    create_test_script()
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Run: python test_actual_values.py")
    print("   This will show what values are being used")
    print()
    print("2. Start training and watch the debug output")
    print("   You'll see what values are actually being initialized")
    print()
    print("3. Look for the initialization message and warmup logs")
    print("="*80)
