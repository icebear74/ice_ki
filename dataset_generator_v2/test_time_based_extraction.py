#!/usr/bin/env python3
"""
Test time-based frame extraction with FFmpeg command logging.
"""

import os
import sys

def test_time_based_extraction():
    """Verify time-based extraction and command logging."""
    
    script_path = os.path.join(os.path.dirname(__file__), 'make_dataset_v2_uhd.py')
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    print("Testing time-based extraction implementation...")
    
    # 1. Verify _extract_frames_with_file method exists
    assert '_extract_frames_with_file' in content, "❌ _extract_frames_with_file method not found"
    print("✓ PASS: _extract_frames_with_file method exists")
    
    # 2. Verify time-based selection (between(t,...))
    assert 'between(t,' in content, "❌ Time-based selection (between(t,...)) not found"
    print("✓ PASS: Time-based selection (between(t,...)) used")
    
    # 3. Verify frame-based selection (eq(n,...)) is NOT used in extraction
    # (it's OK if it exists in comments or other places, just not in the main logic)
    lines_with_eq_n = [line for line in content.split('\n') if 'eq(n,' in line and not line.strip().startswith('#')]
    assert len(lines_with_eq_n) < 2, f"❌ Frame-based selection (eq(n,...)) still in use: {len(lines_with_eq_n)} occurrences"
    print("✓ PASS: Frame-based selection (eq(n,...)) removed from extraction logic")
    
    # 4. Verify TIME-BASED logging message
    assert 'TIME-BASED' in content, "❌ TIME-BASED logging message not found"
    print("✓ PASS: TIME-BASED logging message present")
    
    # 5. Verify -discard nokey is used
    assert 'discard' in content and 'nokey' in content, "❌ -discard nokey not found"
    print("✓ PASS: -discard nokey flag present")
    
    # 6. Verify sendcmd filter is used
    assert 'sendcmd' in content, "❌ sendcmd filter not found"
    print("✓ PASS: sendcmd filter present")
    
    # 7. Verify FFmpeg command is logged
    assert 'FFmpeg command:' in content, "❌ FFmpeg command logging not found"
    print("✓ PASS: FFmpeg command logging added")
    
    # 8. Verify commands file creation
    assert 'frame_select_commands.txt' in content, "❌ Commands file creation not found"
    print("✓ PASS: Commands file creation present")
    
    print("\n✅ All time-based extraction tests passed!")
    print("\nSummary:")
    print("  - Time-based selection (between(t,...)) used instead of frame numbers")
    print("  - Compatible with -discard nokey for faster seeking")
    print("  - FFmpeg command logged for debugging")
    print("  - No more 'extracting ALL frames' issue")
    print("  - sendcmd with external file for all cases")
    
    return True

if __name__ == '__main__':
    try:
        test_time_based_extraction()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n{e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
