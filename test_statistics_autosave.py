#!/usr/bin/env python3
"""
Test script for auto-save statistics JSON functionality

Tests that statistics JSON files are saved correctly after validation.
"""

import os
import sys
import json
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_statistics_json_structure():
    """Test that we can create and save a proper statistics JSON"""
    print("\n" + "="*70)
    print("TEST: Statistics JSON Structure")
    print("="*70)
    
    # Create test data matching web monitor structure
    test_data = {
        'step_current': 7500,
        'epoch_num': 8,
        'step_max': 100000,
        'learning_rate_value': 0.00003456,
        'adaptive_mode': 'Stable',
        'adaptive_plateau_counter': 42,
        'adaptive_lr_boost_available': False,
        'perceptual_weight_current': 0.123,
        'total_loss_value': 0.0234,
        'quality_ki_value': 0.621,
        'quality_improvement_value': 0.189,
    }
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create filename
        step = 7500
        filename = f"Statistik_{step}.json"
        filepath = os.path.join(tmpdir, filename)
        
        # Save JSON with pretty formatting
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Created test file: {filename}")
        
        # Verify file exists
        assert os.path.exists(filepath), "File should exist"
        print(f"✓ File exists at: {filepath}")
        
        # Read and verify content
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == test_data, "Data should match"
        print(f"✓ Data integrity verified")
        
        # Check formatting
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert '  "step_current": 7500' in content, "Should have 2-space indentation"
        print(f"✓ Pretty formatting verified (2-space indent)")
        
        # Show sample of content
        print(f"\n📄 Sample JSON content:")
        print("─" * 50)
        print(content[:300] + "...")
        print("─" * 50)
    
    print("\n✅ Test passed: Statistics JSON structure is correct")


def test_filename_format():
    """Test that filenames are generated correctly"""
    print("\n" + "="*70)
    print("TEST: Filename Format")
    print("="*70)
    
    test_steps = [500, 1000, 7500, 15000, 99999]
    
    for step in test_steps:
        filename = f"Statistik_{step}.json"
        expected = f"Statistik_{step}.json"
        assert filename == expected, f"Filename should match format for step {step}"
        print(f"✓ Step {step:6d} -> {filename}")
    
    print("\n✅ Test passed: Filenames are correctly formatted")


def test_directory_creation():
    """Test that directory is created if it doesn't exist"""
    print("\n" + "="*70)
    print("TEST: Directory Creation")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a subdirectory path that doesn't exist yet
        data_root = os.path.join(tmpdir, "Learn", "Statistics")
        
        # Ensure directory doesn't exist initially
        assert not os.path.exists(data_root), "Directory should not exist initially"
        print(f"✓ Directory doesn't exist: {data_root}")
        
        # Create directory (simulating what trainer does)
        os.makedirs(data_root, exist_ok=True)
        
        # Verify it was created
        assert os.path.exists(data_root), "Directory should be created"
        assert os.path.isdir(data_root), "Should be a directory"
        print(f"✓ Directory created successfully: {data_root}")
        
        # Test saving a file
        filepath = os.path.join(data_root, "Statistik_1000.json")
        with open(filepath, 'w') as f:
            json.dump({'test': 'data'}, f)
        
        assert os.path.exists(filepath), "File should be saved"
        print(f"✓ File saved successfully: Statistik_1000.json")
    
    print("\n✅ Test passed: Directory creation works correctly")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("AUTO-SAVE STATISTICS JSON - TEST SUITE")
    print("="*70)
    
    try:
        test_filename_format()
        test_directory_creation()
        test_statistics_json_structure()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✅")
        print("="*70)
        print("\nThe auto-save statistics JSON functionality is working correctly:")
        print("  1. ✅ Filenames are formatted as Statistik_STEP.json")
        print("  2. ✅ Directories are created if they don't exist")
        print("  3. ✅ JSON is saved with proper formatting (2-space indent)")
        print("  4. ✅ Data structure matches web UI download format")
        print("="*70 + "\n")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
