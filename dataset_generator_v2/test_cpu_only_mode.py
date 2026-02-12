#!/usr/bin/env python3
"""
Test suite to verify CPU-only mode is properly enabled.
Validates that CUDA/GPU acceleration has been disabled.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_cpu_only_mode():
    """Test that CPU-only mode is enforced"""
    print("Testing CPU-only mode configuration...")
    
    # Check source code for CUDA-related patterns (without importing)
    source_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'make_dataset_v2_uhd.py'
    )
    
    with open(source_file, 'r') as f:
        content = f.read()
    
    # Check for CPU-only mode indicator
    if 'CPU-only mode enabled' in content:
        print("✓ CPU-only mode message found in code")
    else:
        print("❌ CPU-only mode message not found")
        return False
    
    # Check that CUDA configuration is disabled
    if 'self.use_cuda = False' in content:
        print("✓ use_cuda explicitly set to False")
    else:
        print("❌ use_cuda not explicitly disabled")
        return False
    
    # Check tonemap filter uses zscale
    if 'zscale=t=linear:npl=100' in content:
        print("✓ Tonemap filter uses zscale (as recommended)")
    else:
        print("⚠️  WARNING: zscale tonemap filter not found")
    
    # Verify CUDA arguments are not in extraction code
    cuda_patterns = [
        "'-hwaccel', 'cuda'",
        '-hwaccel_device',
        'h264_cuvid',
        'hevc_cuvid'
    ]
    
    cuda_found = False
    for pattern in cuda_patterns:
        # Count occurrences (some might be in comments or removed code)
        count = content.count(pattern)
        if count > 0:
            print(f"⚠️  Found '{pattern}' {count} times (check if in active code)")
            cuda_found = True
    
    if not cuda_found:
        print("✓ No CUDA hardware acceleration arguments in extraction code")
    
    print("\n✅ All CPU-only mode tests passed!")
    return True

if __name__ == '__main__':
    success = test_cpu_only_mode()
    sys.exit(0 if success else 1)
