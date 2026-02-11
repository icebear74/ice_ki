#!/usr/bin/env python3
"""
Test to verify make_dataset_v2_uhd.py has no syntax errors.

This test ensures the file can be parsed and compiled by Python.
It was created after fixing a SyntaxError caused by a premature
docstring closing at line 1050.
"""

import ast
import sys
from pathlib import Path

print("=" * 70)
print("SYNTAX VERIFICATION TEST - make_dataset_v2_uhd.py")
print("=" * 70)

script_path = Path(__file__).parent / "make_dataset_v2_uhd.py"

# Test 1: File can be compiled
print("\nTest 1: Python Compilation")
print("-" * 70)
try:
    import py_compile
    py_compile.compile(str(script_path), doraise=True)
    print("✓ PASS: File compiles without syntax errors")
except SyntaxError as e:
    print(f"✗ FAIL: SyntaxError at line {e.lineno}: {e.msg}")
    sys.exit(1)

# Test 2: File can be parsed as AST
print("\nTest 2: AST Parsing")
print("-" * 70)
try:
    with open(script_path, 'r') as f:
        code = f.read()
    ast.parse(code)
    print("✓ PASS: File parses as valid Python AST")
except SyntaxError as e:
    print(f"✗ FAIL: SyntaxError at line {e.lineno}: {e.msg}")
    print(f"       Text: {e.text}")
    sys.exit(1)

# Test 3: Check for common docstring issues
print("\nTest 3: Docstring Balance Check")
print("-" * 70)
with open(script_path, 'r') as f:
    lines = f.readlines()

triple_double_count = 0
for i, line in enumerate(lines, 1):
    count = line.count('"""')
    triple_double_count += count
    
if triple_double_count % 2 == 0:
    print(f"✓ PASS: Triple-double-quotes are balanced ({triple_double_count} total)")
else:
    print(f"✗ FAIL: Triple-double-quotes are unbalanced ({triple_double_count} total)")
    sys.exit(1)

# Test 4: Verify specific fix
print("\nTest 4: Specific Fix Verification (Line 1050)")
print("-" * 70)
# Line 1050 should NOT have a standalone '"""' anymore
line_1050 = lines[1049].strip()  # 0-indexed
if line_1050 == '"""':
    print(f"✗ FAIL: Line 1050 still has standalone docstring closing")
    print(f"       This was the bug that caused the SyntaxError")
    sys.exit(1)
else:
    print(f"✓ PASS: Line 1050 does not have standalone '\"\"\"'")
    print(f"       Current content: {repr(line_1050[:50])}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
All syntax verification tests passed!

The file make_dataset_v2_uhd.py:
  ✓ Compiles without syntax errors
  ✓ Parses as valid Python AST
  ✓ Has balanced docstring quotes
  ✓ Does not have the premature docstring closing bug

The SyntaxError has been successfully fixed.
""")

print("✓ All syntax verification tests passed")
