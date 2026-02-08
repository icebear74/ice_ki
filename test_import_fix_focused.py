#!/usr/bin/env python3
"""
Test that the specific import statements in make_dataset_multi.py work.
This simulates the exact import pattern used in the file.
"""

import sys
import os

# Simulate the sys.path setup in make_dataset_multi.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dataset_generator_v2'))

def test_imports():
    """Test all the imports that were changed from relative to absolute"""
    
    print("Testing import fixes...\n")
    
    # Test 1: Line 937
    print("1. Testing: from utils.ui_terminal import clear_and_home, hide_cursor")
    try:
        from utils.ui_terminal import clear_and_home, hide_cursor
        print("   ✓ Success\n")
    except ImportError as e:
        print(f"   ✗ Failed: {e}\n")
        return False
    
    # Test 2: Line 562 and 985
    print("2. Testing: from utils.ui_display import draw_dataset_generator_ui")
    try:
        from utils.ui_display import draw_dataset_generator_ui
        print("   ✓ Success\n")
    except ImportError as e:
        print(f"   ✗ Failed: {e}\n")
        return False
    
    # Test 3: Line 997
    print("3. Testing: from utils.ui_terminal import show_cursor")
    try:
        from utils.ui_terminal import show_cursor
        print("   ✓ Success\n")
    except ImportError as e:
        print(f"   ✗ Failed: {e}\n")
        return False
    
    return True

def verify_no_relative_imports():
    """Verify that no relative imports remain in make_dataset_multi.py"""
    
    print("Verifying no relative imports remain...\n")
    
    file_path = os.path.join(os.path.dirname(__file__), 
                             'dataset_generator_v2', 
                             'make_dataset_multi.py')
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check for relative imports from .utils
    if 'from .utils.' in content:
        print("   ✗ Found relative imports (from .utils.) in file!\n")
        return False
    
    print("   ✓ No relative imports found\n")
    return True

if __name__ == "__main__":
    print("="*60)
    print("Import Fix Verification Test")
    print("="*60 + "\n")
    
    test1 = test_imports()
    test2 = verify_no_relative_imports()
    
    print("="*60)
    if test1 and test2:
        print("✅ All tests passed!")
        print("\nThe ImportError is fixed:")
        print("  - All relative imports changed to absolute imports")
        print("  - Imports work correctly when script is run directly")
        print("="*60)
    else:
        print("❌ Some tests failed")
        print("="*60)
        sys.exit(1)
