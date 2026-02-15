#!/usr/bin/env python3
"""
Static verification of checkpoint saving code

This script verifies that all checkpoint saving calls in trainer.py
include the runtime_config parameter.
"""

import os
import re

def check_checkpoint_calls():
    """Check that all checkpoint saving calls include runtime_config"""
    
    trainer_path = os.path.join('vsr_plusplus_NEU', 'training', 'trainer.py')
    
    with open(trainer_path, 'r') as f:
        content = f.read()
    
    print("="*70)
    print("Checking checkpoint saving calls in trainer.py")
    print("="*70)
    
    # Find all checkpoint save calls
    patterns = [
        (r'checkpoint_mgr\.save_checkpoint\([^)]+\)', 'save_checkpoint'),
        (r'checkpoint_mgr\.update_best_checkpoint\([^)]+\)', 'update_best_checkpoint'),
        (r'checkpoint_mgr\.save_emergency_checkpoint\([^)]+\)', 'save_emergency_checkpoint'),
    ]
    
    all_good = True
    
    for pattern, method_name in patterns:
        print(f"\nChecking {method_name} calls...")
        matches = re.finditer(pattern, content, re.DOTALL)
        
        call_count = 0
        missing_runtime_config = 0
        
        for match in matches:
            call_count += 1
            call_text = match.group(0)
            
            # Check if runtime_config is in the call
            if 'runtime_config' not in call_text and 'self.runtime_config' not in call_text:
                missing_runtime_config += 1
                # Get line number
                line_num = content[:match.start()].count('\n') + 1
                print(f"  ❌ Line {line_num}: Missing runtime_config parameter")
                all_good = False
            else:
                line_num = content[:match.start()].count('\n') + 1
                print(f"  ✓ Line {line_num}: Has runtime_config parameter")
        
        if call_count == 0:
            print(f"  ℹ️  No {method_name} calls found")
        elif missing_runtime_config == 0:
            print(f"  ✅ All {call_count} {method_name} calls have runtime_config")
        else:
            print(f"  ❌ {missing_runtime_config}/{call_count} calls missing runtime_config")
    
    print("\n" + "="*70)
    if all_good:
        print("✅ ALL CHECKPOINT CALLS INCLUDE runtime_config")
        return True
    else:
        print("❌ SOME CHECKPOINT CALLS ARE MISSING runtime_config")
        return False


if __name__ == '__main__':
    import sys
    success = check_checkpoint_calls()
    sys.exit(0 if success else 1)
