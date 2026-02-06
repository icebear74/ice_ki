#!/usr/bin/env python3
"""
Test suite for VSR++ training improvements
Tests checkpoint manager, web UI, and checkpoint selection
"""

import os
import sys
import tempfile
import shutil
import torch
import re

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsr_plus_plus.systems.checkpoint_manager import CheckpointManager


def test_checkpoint_naming_scheme():
    """Test that new zero-padded naming scheme works correctly"""
    print("\n" + "="*70)
    print("TEST 1: Checkpoint Naming Scheme")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        
        # Test new format naming
        test_steps = [123, 4567, 100000, 9999999]
        expected_names = [
            "checkpoint_step_0000123.pth",
            "checkpoint_step_0004567.pth",
            "checkpoint_step_0100000.pth",
            "checkpoint_step_9999999.pth"
        ]
        
        print("\n✓ Testing new naming format:")
        for step, expected in zip(test_steps, expected_names):
            # Create a dummy checkpoint
            dummy_model = torch.nn.Linear(10, 10)
            dummy_optimizer = torch.optim.Adam(dummy_model.parameters())
            
            saved_path = mgr.save_checkpoint(
                dummy_model, dummy_optimizer, None, step, {}, None
            )
            
            actual_name = os.path.basename(saved_path)
            status = "✓" if actual_name == expected else "✗"
            print(f"  {status} Step {step:7d} -> {actual_name} (expected: {expected})")
            
            assert actual_name == expected, f"Naming mismatch for step {step}"
        
        print("\n✓ All naming tests passed!")


def test_regex_step_extraction():
    """Test regex-based step extraction from filenames"""
    print("\n" + "="*70)
    print("TEST 2: Regex Step Extraction")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        
        test_cases = [
            ("checkpoint_step_0000123.pth", 123),
            ("checkpoint_step_0004567.pth", 4567),
            ("checkpoint_step_0000001_emergency.pth", 1),
            ("checkpoint_step_1234567_emergency.pth", 1234567),
            ("checkpoint_step_0010000.pth", 10000),
            # Old format (backward compatibility)
            ("checkpoint_step_123.pth", 123),
            ("checkpoint_step_4567.pth", 4567),
        ]
        
        print("\n✓ Testing step extraction:")
        for filename, expected_step in test_cases:
            extracted = mgr._parse_step_from_filename(filename)
            status = "✓" if extracted == expected_step else "✗"
            print(f"  {status} {filename:45s} -> {extracted:7d} (expected: {expected_step})")
            
            assert extracted == expected_step, f"Extraction failed for {filename}"
        
        print("\n✓ All extraction tests passed!")


def test_backward_compatibility():
    """Test that manager can read both old and new checkpoint formats"""
    print("\n" + "="*70)
    print("TEST 3: Backward Compatibility")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        
        # Create mix of old and new format checkpoints
        dummy_model = torch.nn.Linear(10, 10)
        dummy_optimizer = torch.optim.Adam(dummy_model.parameters())
        
        # Manually create old-format checkpoint
        old_checkpoint = {
            'step': 1000,
            'model_state_dict': dummy_model.state_dict(),
            'optimizer_state_dict': dummy_optimizer.state_dict(),
            'scheduler_state_dict': None,
            'metrics': {'ki_quality': 0.75},
            'timestamp': '2024-01-01T12:00:00'
        }
        old_path = os.path.join(tmpdir, "checkpoint_step_1000.pth")
        torch.save(old_checkpoint, old_path)
        
        # Create new-format checkpoint using manager
        mgr.save_checkpoint(dummy_model, dummy_optimizer, None, 2000, {'ki_quality': 0.80}, None)
        
        # Test that both are found
        all_ckpts = mgr.list_checkpoints()
        
        print(f"\n✓ Found {len(all_ckpts)} checkpoints:")
        for ckpt in all_ckpts:
            print(f"  ✓ Step {ckpt['step']:7d} - {ckpt['type']:10s} - {ckpt['filename']}")
        
        assert len(all_ckpts) == 2, "Should find both old and new format checkpoints"
        
        # Test get_latest_checkpoint
        latest_path, latest_step = mgr.get_latest_checkpoint()
        print(f"\n✓ Latest checkpoint: Step {latest_step}")
        
        assert latest_step == 2000, "Should return newest checkpoint (step 2000)"
        assert "checkpoint_step_0002000.pth" in latest_path, "Should use new format name"
        
        print("\n✓ Backward compatibility test passed!")


def test_emergency_checkpoint_naming():
    """Test emergency checkpoint naming with real step numbers"""
    print("\n" + "="*70)
    print("TEST 4: Emergency Checkpoint Naming")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        
        dummy_model = torch.nn.Linear(10, 10)
        dummy_optimizer = torch.optim.Adam(dummy_model.parameters())
        
        # Save emergency checkpoint at step 4144
        emergency_path = mgr.save_emergency_checkpoint(
            dummy_model, dummy_optimizer, None, 4144, {}, None
        )
        
        emergency_name = os.path.basename(emergency_path)
        expected_name = "checkpoint_step_0004144_emergency.pth"
        
        print(f"\n✓ Emergency checkpoint saved: {emergency_name}")
        assert emergency_name == expected_name, f"Expected {expected_name}, got {emergency_name}"
        
        # Verify step extraction works
        extracted_step = mgr._parse_step_from_filename(emergency_name)
        print(f"✓ Extracted step from emergency checkpoint: {extracted_step}")
        assert extracted_step == 4144, f"Should extract step 4144, got {extracted_step}"
        
        print("\n✓ Emergency checkpoint test passed!")


def test_web_ui_state_holder():
    """Test web UI state holder"""
    print("\n" + "="*70)
    print("TEST 5: Web UI Complete Data Store")
    print("="*70)
    
    from vsr_plus_plus.systems.web_ui import CompleteTrainingDataStore
    
    store = CompleteTrainingDataStore()
    
    # Update with sample metrics - ALLE Daten
    store.update_all_metrics(
        step_current=1000,
        total_loss_value=0.0123,
        learning_rate_value=0.0001,
        eta_total_formatted="01:23:45",
        iteration_duration=0.5,
        vram_usage_gb=8.5,
        best_quality_ever=0.85,
        layer_activity_map={'Layer1': 0.75, 'Layer2': 0.82}
    )
    
    # Fetch snapshot
    snapshot = store.get_complete_snapshot()
    
    print("\n✓ Complete data store snapshot:")
    print(f"  step_current: {snapshot['step_current']}")
    print(f"  total_loss_value: {snapshot['total_loss_value']}")
    print(f"  best_quality_ever: {snapshot['best_quality_ever']}")
    print(f"  layer_activity_map: {snapshot['layer_activity_map']}")
    
    assert snapshot['step_current'] == 1000, "Step mismatch"
    assert snapshot['total_loss_value'] == 0.0123, "Loss mismatch"
    assert snapshot['best_quality_ever'] == 0.85, "Best score mismatch"
    assert 'Layer1' in snapshot['layer_activity_map'], "Layer activities missing"
    
    print("\n✓ Web UI complete data store test passed!")


def test_cleanup_all_for_fresh_start_return_value():
    """Test that cleanup_all_for_fresh_start returns the number of backed up files"""
    print("\n" + "="*70)
    print("TEST 6: cleanup_all_for_fresh_start Return Value")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        log_dir = os.path.join(tmpdir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Create some dummy checkpoint files
        dummy_model = torch.nn.Linear(10, 10)
        dummy_optimizer = torch.optim.Adam(dummy_model.parameters())
        
        # Save 3 checkpoints
        mgr.save_checkpoint(dummy_model, dummy_optimizer, None, 1000, {}, None)
        mgr.save_checkpoint(dummy_model, dummy_optimizer, None, 2000, {}, None)
        mgr.save_checkpoint(dummy_model, dummy_optimizer, None, 3000, {}, None)
        
        print("\n✓ Created 3 checkpoint files")
        
        # Call cleanup_all_for_fresh_start and verify it returns the count
        backed_up_count = mgr.cleanup_all_for_fresh_start(log_dir)
        
        print(f"✓ cleanup_all_for_fresh_start returned: {backed_up_count}")
        
        # Verify the return value is correct
        assert backed_up_count is not None, "Should return a value, not None"
        assert backed_up_count == 3, f"Should backup 3 files, got {backed_up_count}"
        
        # Verify .BAK files exist
        bak_files = [f for f in os.listdir(tmpdir) if f.endswith('.BAK')]
        print(f"✓ Found {len(bak_files)} .BAK backup files")
        assert len(bak_files) == 3, f"Should have 3 .BAK files, got {len(bak_files)}"
        
        print("\n✓ cleanup_all_for_fresh_start return value test passed!")


def test_cleanup_with_empty_directory():
    """Test that cleanup_all_for_fresh_start returns 0 when there are no files to backup"""
    print("\n" + "="*70)
    print("TEST 7: cleanup_all_for_fresh_start With Empty Directory")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        log_dir = os.path.join(tmpdir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        print("\n✓ Created empty checkpoint directory")
        
        # Call cleanup_all_for_fresh_start on empty directory
        backed_up_count = mgr.cleanup_all_for_fresh_start(log_dir)
        
        print(f"✓ cleanup_all_for_fresh_start returned: {backed_up_count}")
        
        # Verify the return value is 0 (not None)
        assert backed_up_count is not None, "Should return a value, not None"
        assert backed_up_count == 0, f"Should return 0 when no files to backup, got {backed_up_count}"
        assert isinstance(backed_up_count, int), f"Should return an integer, got {type(backed_up_count)}"
        
        print("\n✓ Empty directory test passed - correctly returns 0!")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("VSR++ TRAINING IMPROVEMENTS - TEST SUITE")
    print("="*70)
    
    try:
        test_checkpoint_naming_scheme()
        test_regex_step_extraction()
        test_backward_compatibility()
        test_emergency_checkpoint_naming()
        test_web_ui_state_holder()
        test_cleanup_all_for_fresh_start_return_value()
        test_cleanup_with_empty_directory()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
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
