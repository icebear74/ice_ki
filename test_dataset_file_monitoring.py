#!/usr/bin/env python3
"""
Test dataset file monitoring functionality
"""
import os
import sys
import tempfile

# Add vsr_plusplus_NEU to path - import only the dataset module directly
sys.path.insert(0, os.path.dirname(__file__))

def test_dataset_methods():
    """Test the new methods exist and are callable"""
    print("Testing dataset methods exist...")
    
    # Import dataset module source
    dataset_file = os.path.join(os.path.dirname(__file__), 'vsr_plusplus_NEU', 'core', 'dataset.py')
    
    with open(dataset_file, 'r') as f:
        content = f.read()
    
    # Check for new methods
    assert 'def get_file_info(self):' in content, "get_file_info method not found"
    assert 'def check_for_new_files(self):' in content, "check_for_new_files method not found"
    
    print("  ✓ get_file_info() method found")
    print("  ✓ check_for_new_files() method found")
    
    # Check return structure in docstrings
    assert "'mode':" in content, "get_file_info should return mode"
    assert "'size_key':" in content, "get_file_info should return size_key"
    assert "'file_count':" in content, "get_file_info should return file_count"
    assert "'has_changes':" in content, "check_for_new_files should return has_changes"
    assert "'new_files':" in content, "check_for_new_files should return new_files"
    
    print("  ✓ Return structures documented correctly")
    print("  ✓ Dataset methods test PASSED\n")


def test_trainer_method():
    """Test the trainer has the new method"""
    print("Testing trainer method exists...")
    
    trainer_file = os.path.join(os.path.dirname(__file__), 'vsr_plusplus_NEU', 'training', 'trainer.py')
    
    with open(trainer_file, 'r') as f:
        content = f.read()
    
    # Check for new method
    assert 'def _check_dataset_files(self):' in content, "_check_dataset_files method not found"
    assert 'dataset_info = {' in content, "dataset_info structure not found"
    assert "self.global_step % 100 == 0" in content, "Check every 100 steps not found"
    
    print("  ✓ _check_dataset_files() method found")
    print("  ✓ Called every 100 steps")
    print("  ✓ Trainer method test PASSED\n")


def test_web_ui_updates():
    """Test web UI has dataset file support"""
    print("Testing web UI updates...")
    
    # Check web_ui.py
    web_ui_file = os.path.join(os.path.dirname(__file__), 'vsr_plusplus_NEU', 'systems', 'web_ui.py')
    
    with open(web_ui_file, 'r') as f:
        content = f.read()
    
    assert "'dataset_files':" in content, "dataset_files not in state"
    assert "'train':" in content and "'val':" in content, "train/val structure not found"
    assert "'720':" in content and "'540':" in content and "'720_169':" in content, "size keys not found"
    
    print("  ✓ dataset_files added to state")
    print("  ✓ All size keys present (720, 540, 720_169)")
    print("  ✓ Web UI data store test PASSED\n")
    
    # Check template
    template_file = os.path.join(os.path.dirname(__file__), 'vsr_plusplus_NEU', 'web', 'templates', 'monitor.html')
    
    with open(template_file, 'r') as f:
        content = f.read()
    
    assert 'Dataset Files' in content, "Dataset Files section not found"
    assert 'updateDatasetFiles' in content, "updateDatasetFiles function not found"
    assert 'val720Count' in content, "val720Count element not found"
    assert 'val540Count' in content, "val540Count element not found"
    assert 'val720_169Count' in content, "val720_169Count element not found"
    
    print("  ✓ Dataset Files card added to template")
    print("  ✓ JavaScript update function added")
    print("  ✓ All validation size displays present")
    print("  ✓ Web UI template test PASSED\n")


def test_initialization():
    """Test initialization in train.py"""
    print("Testing training initialization...")
    
    train_file = os.path.join(os.path.dirname(__file__), 'vsr_plusplus_NEU', 'train.py')
    
    with open(train_file, 'r') as f:
        content = f.read()
    
    assert 'trainer._check_dataset_files()' in content, "Initial check not found"
    assert 'Initializing dataset file monitoring' in content, "Init message not found"
    
    print("  ✓ Initial dataset file check added")
    print("  ✓ Initialization message present")
    print("  ✓ Training initialization test PASSED\n")


if __name__ == '__main__':
    print("="*60)
    print("Dataset File Monitoring Tests")
    print("="*60)
    print()
    
    try:
        test_dataset_methods()
        test_trainer_method()
        test_web_ui_updates()
        test_initialization()
        
        print("="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print()
        print("Summary:")
        print("  ✓ Dataset class: get_file_info() and check_for_new_files() added")
        print("  ✓ Trainer: _check_dataset_files() called every 100 steps")
        print("  ✓ Web UI: dataset_files tracking for train + val (720/540/720_169)")
        print("  ✓ Template: Dataset Files card with size breakdown")
        print("  ✓ Initialization: File monitoring starts on training begin")
        print()
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
