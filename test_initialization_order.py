#!/usr/bin/env python3
"""
Simple test to verify the initialization order fix
Tests that base_dir is set before logger is created
"""

import re

def test_initialization_order():
    """Test that base_dir is initialized before _setup_logger is called"""
    print("=" * 60)
    print("TEST: Initialization Order in make_dataset_v2_uhd.py")
    print("=" * 60)
    
    # Read the file
    with open('dataset_generator_v2/make_dataset_v2_uhd.py', 'r') as f:
        content = f.read()
    
    # Find the __init__ method
    init_match = re.search(r'def __init__\(self.*?\n(.*?)(?=\n    def )', content, re.DOTALL)
    if not init_match:
        print("❌ FAILED: Could not find __init__ method")
        return False
    
    init_body = init_match.group(1)
    
    # Find line numbers (approximate)
    lines = init_body.split('\n')
    
    base_dir_line = None
    logger_line = None
    
    for i, line in enumerate(lines):
        if 'self.base_dir' in line and '=' in line:
            base_dir_line = i
            print(f"   Found self.base_dir assignment at line {i}: {line.strip()}")
        if 'self.logger = self._setup_logger()' in line:
            logger_line = i
            print(f"   Found logger setup at line {i}: {line.strip()}")
    
    if base_dir_line is None:
        print("❌ FAILED: Could not find self.base_dir assignment")
        return False
    
    if logger_line is None:
        print("❌ FAILED: Could not find self.logger = self._setup_logger()")
        return False
    
    # Check that base_dir comes before logger
    if base_dir_line < logger_line:
        print("✅ SUCCESS: self.base_dir is set BEFORE _setup_logger() is called")
        print(f"   Order is correct: base_dir (line {base_dir_line}) before logger (line {logger_line})")
        return True
    else:
        print("❌ FAILED: self.base_dir is set AFTER _setup_logger() is called")
        print(f"   Order is wrong: base_dir (line {base_dir_line}) after logger (line {logger_line})")
        return False


def test_setup_logger_uses_base_dir():
    """Test that _setup_logger method uses self.base_dir"""
    print("\n" + "=" * 60)
    print("TEST: _setup_logger Uses self.base_dir")
    print("=" * 60)
    
    with open('dataset_generator_v2/make_dataset_v2_uhd.py', 'r') as f:
        content = f.read()
    
    # Find _setup_logger method
    logger_match = re.search(r'def _setup_logger\(self\):(.*?)(?=\n    def )', content, re.DOTALL)
    if not logger_match:
        print("❌ FAILED: Could not find _setup_logger method")
        return False
    
    logger_body = logger_match.group(1)
    
    if 'self.base_dir' in logger_body:
        print("✅ SUCCESS: _setup_logger uses self.base_dir")
        # Show the line
        for line in logger_body.split('\n'):
            if 'self.base_dir' in line:
                print(f"   Found: {line.strip()}")
        return True
    else:
        print("❌ FAILED: _setup_logger does not use self.base_dir")
        return False


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Initialization Order Fix Verification".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    results = []
    
    # Test 1: Initialization order
    results.append(("Initialization Order", test_initialization_order()))
    
    # Test 2: Logger uses base_dir
    results.append(("Logger Uses base_dir", test_setup_logger_uses_base_dir()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    import sys
    sys.exit(0 if passed == total else 1)
