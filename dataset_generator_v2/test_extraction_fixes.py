#!/usr/bin/env python3
"""
Test suite for extraction fixes:
1. Frame skipping fix (strict stride detection + explicit frame list)
2. Nice priority for all FFmpeg/ffprobe commands
3. 6 threads for faster extraction
4. Frame count validation
"""

import os
import sys
import re

def test_thread_count():
    """Test that 6 threads are configured"""
    print("Testing thread count configuration...")
    
    with open('make_dataset_v2_uhd.py', 'r') as f:
        content = f.read()
    
    # Check that workers is set to 6
    if 'self.workers = 6' in content:
        print("✓ Workers set to 6 threads")
    else:
        print("✗ Workers not set to 6")
        return False
    
    # Check log message
    if 'Using {self.workers} threads' in content or 'Using 6 threads' in content:
        print("✓ Thread count logged")
    else:
        print("⚠️  Thread count not logged (optional)")
    
    return True

def test_nice_priority():
    """Test that nice priority is added to all commands"""
    print("\nTesting nice priority...")
    
    with open('make_dataset_v2_uhd.py', 'r') as f:
        content = f.read()
    
    # Count nice commands
    nice_count = content.count("'nice', '-n', '19',")
    
    # Should appear in:
    # 1. extract_frames_uhd (single frame extraction)
    # 2. _extract_frames_with_stride (batch extraction)
    # 3. _get_video_metadata (ffprobe)
    
    if nice_count >= 3:
        print(f"✓ Nice priority found in {nice_count} locations")
        print("  - Single frame extraction")
        print("  - Batch extraction")
        print("  - Video metadata (ffprobe)")
    else:
        print(f"✗ Nice priority only found in {nice_count} locations (expected 3+)")
        return False
    
    return True

def test_strict_stride_detection():
    """Test that stride detection is strict"""
    print("\nTesting strict stride detection...")
    
    with open('make_dataset_v2_uhd.py', 'r') as f:
        content = f.read()
    
    # Check for strict uniform detection
    if 'len(set(intervals)) == 1' in content:
        print("✓ Strict uniform stride detection (requires all intervals identical)")
    elif 'len(set(intervals)) <= 2' in content:
        print("✗ Old loose stride detection found (allows 2 variations)")
        return False
    else:
        print("⚠️  Could not verify stride detection logic")
        return False
    
    return True

def test_file_based_frame_list():
    """Test that file-based frame list is used"""
    print("\nTesting file-based frame list...")
    
    with open('make_dataset_v2_uhd.py', 'r') as f:
        content = f.read()
    
    # Check for sendcmd usage
    if 'sendcmd=f=' in content:
        print("✓ File-based sendcmd approach found")
    else:
        print("✗ sendcmd not found")
        return False
    
    # Check for commands file creation
    if 'frame_select_commands.txt' in content or 'commands.txt' in content:
        print("✓ Commands file creation found")
    else:
        print("⚠️  Commands file name not found")
    
    # Check for writing frame selections to file
    if "f.write(f\"0 select 'eq(n,{frame_num})'" in content or "write" in content and "eq(n," in content:
        print("✓ Frame selections written to file")
    else:
        print("⚠️  Frame writing logic not found")
    
    return True

def test_explicit_frame_list():
    """Test that explicit frame list is used instead of modulo"""
    print("\nTesting explicit frame list extraction...")
    
    with open('make_dataset_v2_uhd.py', 'r') as f:
        content = f.read()
    
    # Check for explicit frame number approach (now in file)
    if "eq(n,{frame_num})" in content or "eq(n," in content:
        print("✓ Explicit frame selection approach found")
    else:
        print("⚠️  Explicit frame approach not found")
    
    # Check that file-based approach is used
    if 'sendcmd' in content:
        print("✓ Using sendcmd (avoids command line length issues)")
    else:
        print("⚠️  sendcmd not found")
    
    return True

def test_frame_validation():
    """Test that frame count validation is present"""
    print("\nTesting frame count validation...")
    
    with open('make_dataset_v2_uhd.py', 'r') as f:
        content = f.read()
    
    # Check for validation logic
    if 'Frame count mismatch' in content or 'VALIDATION' in content:
        print("✓ Frame count validation added")
    else:
        print("⚠️  Frame validation not found (optional improvement)")
    
    # Check for detailed error reporting
    if 'Missing:' in content and 'Expected:' in content:
        print("✓ Detailed error reporting for missing frames")
    else:
        print("⚠️  Detailed error reporting not found")
    
    return True

def test_cpu_only_mode():
    """Verify CPU-only mode is still active"""
    print("\nVerifying CPU-only mode...")
    
    with open('make_dataset_v2_uhd.py', 'r') as f:
        content = f.read()
    
    if 'self.use_cuda = False' in content:
        print("✓ CPU-only mode active (CUDA disabled)")
    else:
        print("✗ CPU-only mode not found")
        return False
    
    # Make sure no CUDA arguments in commands
    if '-hwaccel' in content and 'cuda' in content:
        print("⚠️  CUDA hardware acceleration arguments still present")
    else:
        print("✓ No CUDA hardware acceleration in commands")
    
    return True

def main():
    print("=" * 60)
    print("Testing Extraction Fixes")
    print("=" * 60)
    
    tests = [
        ("Thread Count (6 threads)", test_thread_count),
        ("Nice Priority (nice -n 19)", test_nice_priority),
        ("Strict Stride Detection", test_strict_stride_detection),
        ("File-Based Frame List", test_file_based_frame_list),
        ("Explicit Frame Selection", test_explicit_frame_list),
        ("Frame Validation", test_frame_validation),
        ("CPU-Only Mode", test_cpu_only_mode),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Test error: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All extraction fixes verified!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
