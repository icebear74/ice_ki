#!/usr/bin/env python3
"""
Simple test to verify cleanup_all_for_fresh_start returns correct values
This test doesn't require torch or other dependencies
"""

import os
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsr_plus_plus.systems.checkpoint_manager import CheckpointManager


def test_empty_directory():
    """Test that cleanup returns 0 when directory is empty"""
    print("\n" + "="*70)
    print("TEST: Empty Directory - Should return 0")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        log_dir = os.path.join(tmpdir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        result = mgr.cleanup_all_for_fresh_start(log_dir)
        
        print(f"Result: {result}")
        print(f"Type: {type(result)}")
        
        assert result is not None, "❌ FAIL: Should return a value, not None"
        assert isinstance(result, int), f"❌ FAIL: Should return int, got {type(result)}"
        assert result == 0, f"❌ FAIL: Should return 0 for empty dir, got {result}"
        
        print("✅ PASS: Returns 0 for empty directory")
    
    return True


def test_with_files():
    """Test that cleanup returns correct count when files exist"""
    print("\n" + "="*70)
    print("TEST: With Files - Should return count of .pth files")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        log_dir = os.path.join(tmpdir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Create dummy .pth files
        for i in range(1, 4):
            dummy_file = os.path.join(tmpdir, f"checkpoint_{i}.pth")
            with open(dummy_file, 'w') as f:
                f.write(f"dummy checkpoint {i}")
        
        print(f"Created 3 dummy .pth files")
        
        result = mgr.cleanup_all_for_fresh_start(log_dir)
        
        print(f"Result: {result}")
        print(f"Type: {type(result)}")
        
        assert result is not None, "❌ FAIL: Should return a value, not None"
        assert isinstance(result, int), f"❌ FAIL: Should return int, got {type(result)}"
        assert result == 3, f"❌ FAIL: Should return 3, got {result}"
        
        # Verify .BAK files were created
        bak_files = [f for f in os.listdir(tmpdir) if f.endswith('.BAK')]
        print(f"Created {len(bak_files)} .BAK files")
        assert len(bak_files) == 3, f"❌ FAIL: Should have 3 .BAK files, got {len(bak_files)}"
        
        print("✅ PASS: Returns correct count and creates .BAK files")
    
    return True


def test_comparison_with_zero():
    """Test that the return value works correctly with > comparison"""
    print("\n" + "="*70)
    print("TEST: Comparison - Verify `if backed_up > 0:` works")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        log_dir = os.path.join(tmpdir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Test with empty directory
        backed_up = mgr.cleanup_all_for_fresh_start(log_dir)
        
        try:
            # This should not raise TypeError
            if backed_up > 0:
                print(f"  Backed up {backed_up} files")
            else:
                print(f"  No files to backup (backed_up = {backed_up})")
            
            print("✅ PASS: Comparison `backed_up > 0` works without TypeError")
        except TypeError as e:
            print(f"❌ FAIL: TypeError occurred: {e}")
            return False
    
    return True


def run_all_tests():
    """Run all standalone tests"""
    print("\n" + "="*70)
    print("CLEANUP RETURN VALUE - STANDALONE TEST SUITE")
    print("="*70)
    
    try:
        test_empty_directory()
        test_with_files()
        test_comparison_with_zero()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70 + "\n")
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
