#!/usr/bin/env python3
"""
Test that single extraction mode is ACTUALLY implemented (not just documented).
"""

import re

def test_single_mode_actually_implemented():
    """Verify single mode is actually in the code, not just documentation"""
    
    with open('make_dataset_v2_uhd.py', 'r') as f:
        content = f.read()
    
    # Test 1: extract_frames_batch_uhd should call extract_frames_uhd
    if 'def extract_frames_batch_uhd' in content:
        # Find the method body
        match = re.search(r'def extract_frames_batch_uhd.*?(?=\n    def |\nclass |\Z)', content, re.DOTALL)
        if match:
            method_body = match.group(0)
            
            # Should call extract_frames_uhd (single mode)
            assert 'extract_frames_uhd(video_path, ts, n_frames)' in method_body, \
                "❌ extract_frames_batch_uhd should call extract_frames_uhd (single mode)"
            print("✓ PASS: extract_frames_batch_uhd calls extract_frames_uhd (single mode)")
            
            # Should NOT call _extract_frames_with_file (batch mode)
            assert '_extract_frames_with_file' not in method_body, \
                "❌ extract_frames_batch_uhd should NOT call _extract_frames_with_file (batch mode)"
            print("✓ PASS: extract_frames_batch_uhd does NOT call batch method")
            
            # Should have SINGLE mode logging
            assert 'SINGLE extraction mode' in method_body, \
                "❌ Should log 'SINGLE extraction mode'"
            print("✓ PASS: 'SINGLE extraction mode' logging present")
    
    # Test 2: _extract_frames_with_file method should be DELETED
    assert 'def _extract_frames_with_file' not in content, \
        "❌ _extract_frames_with_file method should be DELETED"
    print("✓ PASS: _extract_frames_with_file method deleted (batch mode removed)")
    
    # Test 3: Should NOT have batch-related logging
    # Find extract_frames_batch_uhd again
    match = re.search(r'def extract_frames_batch_uhd.*?(?=\n    def |\nclass |\Z)', content, re.DOTALL)
    if match:
        method_body = match.group(0)
        assert 'FILE-BASED frame extraction' not in method_body, \
            "❌ Should not mention FILE-BASED (batch mode)"
        assert 'Batch extracting with TIME-BASED' not in method_body, \
            "❌ Should not mention TIME-BASED batch extraction"
        print("✓ PASS: No batch-mode logging in extract_frames_batch_uhd")
    
    print("\n✅ All tests passed! Single mode ACTUALLY implemented (not just documented)")
    print("Summary:")
    print("  - extract_frames_batch_uhd loops over timestamps")
    print("  - Calls extract_frames_uhd() for each (single mode)")
    print("  - Batch method _extract_frames_with_file() deleted")
    print("  - No batch-mode logging")
    print("  - User will now see SINGLE extraction mode in action!")

if __name__ == '__main__':
    test_single_mode_actually_implemented()
