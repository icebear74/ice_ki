#!/usr/bin/env python3
"""
Test that file-based extraction is ALWAYS used (no chunked fallback)
"""

import os
import sys

def test_file_based_always():
    """Verify file-based extraction is always used"""
    
    filepath = os.path.join(os.path.dirname(__file__), 'make_dataset_v2_uhd.py')
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    print("Testing file-based extraction implementation...")
    
    # 1. Check that _extract_frames_with_file exists (renamed from _with_stride)
    if '_extract_frames_with_file' not in content:
        print("❌ FAIL: _extract_frames_with_file method not found")
        return False
    print("✓ PASS: _extract_frames_with_file method exists")
    
    # 2. Check that _extract_frames_chunked is REMOVED
    if '_extract_frames_chunked' in content:
        print("❌ FAIL: _extract_frames_chunked method still exists (should be deleted)")
        return False
    print("✓ PASS: _extract_frames_chunked method removed")
    
    # 3. Check that extract_frames_batch_uhd always calls file-based method
    # Should NOT have uniform/non-uniform branching
    if 'len(set(intervals)) == 1' in content and 'extract_frames_batch_uhd' in content:
        print("❌ FAIL: Still has uniform interval check (should be removed)")
        return False
    print("✓ PASS: No uniform/non-uniform branching")
    
    # 4. Check for FILE-BASED logging message
    if '📄 Using FILE-BASED frame extraction' not in content:
        print("❌ FAIL: FILE-BASED logging message not found")
        return False
    print("✓ PASS: FILE-BASED logging message present")
    
    # 5. Check for -discard nokey flag
    if "'-discard', 'nokey'" not in content:
        print("❌ FAIL: -discard nokey flag not found")
        return False
    print("✓ PASS: -discard nokey flag added")
    
    # 6. Check that sendcmd is still used
    if 'sendcmd=f=' not in content:
        print("❌ FAIL: sendcmd filter not found")
        return False
    print("✓ PASS: sendcmd filter present")
    
    # 7. Check that commands file is created
    if 'frame_select_commands.txt' not in content:
        print("❌ FAIL: commands file creation not found")
        return False
    print("✓ PASS: Commands file creation present")
    
    print("\n✅ All file-based extraction tests passed!")
    print("\nSummary:")
    print("  - File-based extraction (_extract_frames_with_file) always used")
    print("  - Chunked extraction method removed")
    print("  - No uniform/non-uniform branching")
    print("  - -discard nokey for faster seeking")
    print("  - sendcmd with external file for all cases")
    return True

if __name__ == '__main__':
    success = test_file_based_always()
    sys.exit(0 if success else 1)
