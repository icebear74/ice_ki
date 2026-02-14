#!/usr/bin/env python3
"""
Test to verify that subprocess.run calls in run_video_inference.py
are correctly configured without the capture_output + stderr conflict.
"""

import subprocess
import sys


def test_subprocess_capture_output_usage():
    """
    Verify that subprocess.run with capture_output=True works correctly
    without stderr=subprocess.PIPE
    """
    print("Testing subprocess.run with capture_output=True...")
    
    # This should work without error
    try:
        result = subprocess.run(
            ['echo', 'test'],
            check=True,
            capture_output=True
        )
        print("✅ subprocess.run with capture_output=True works correctly")
        print(f"   stdout: {result.stdout.decode().strip()}")
        assert result.returncode == 0
        assert result.stdout == b'test\n'
    except ValueError as e:
        print(f"❌ Error: {e}")
        return False
    
    # This should raise ValueError (the bug we fixed)
    print("\nVerifying that capture_output + stderr raises ValueError...")
    try:
        result = subprocess.run(
            ['echo', 'test'],
            check=True,
            capture_output=True,
            stderr=subprocess.PIPE  # This should raise ValueError
        )
        print("❌ Should have raised ValueError but didn't!")
        return False
    except ValueError as e:
        print(f"✅ Correctly raises ValueError: {e}")
    
    # Test that stderr is accessible in CalledProcessError when using capture_output
    print("\nVerifying that stderr is accessible in exceptions...")
    try:
        result = subprocess.run(
            ['ls', '/nonexistent_path_12345'],
            check=True,
            capture_output=True
        )
        print("❌ Should have raised CalledProcessError but didn't!")
        return False
    except subprocess.CalledProcessError as e:
        print(f"✅ CalledProcessError raised as expected")
        print(f"   stderr is accessible: {e.stderr is not None}")
        assert e.stderr is not None
    
    print("\n✅ All tests passed!")
    return True


if __name__ == '__main__':
    success = test_subprocess_capture_output_usage()
    sys.exit(0 if success else 1)
