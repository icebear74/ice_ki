#!/usr/bin/env python3
"""
Test checkpoint saving with runtime_config parameter

This test verifies that checkpoints are saved with all necessary parameters,
including runtime_config for proper restoration.
"""

import os
import sys
import tempfile
import shutil
import torch
import torch.nn as nn
import torch.optim as optim

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vsr_plusplus_NEU'))

from vsr_plusplus_NEU.systems.checkpoint_manager import CheckpointManager
from vsr_plusplus_NEU.systems.runtime_config import RuntimeConfigManager


class DummyModel(nn.Module):
    """Simple model for testing"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, 1, 1)
    
    def forward(self, x):
        return self.conv(x)


def test_checkpoint_with_runtime_config():
    """Test that checkpoints save and load runtime_config correctly"""
    
    # Create temporary directory for test
    test_dir = tempfile.mkdtemp(prefix='checkpoint_test_')
    
    try:
        print("Testing checkpoint saving with runtime_config...")
        
        # Create dummy model and optimizer
        model = DummyModel()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = None
        
        # Create checkpoint manager
        checkpoint_mgr = CheckpointManager(test_dir)
        
        # Create runtime config
        runtime_config_path = os.path.join(test_dir, "runtime_config.json")
        runtime_config = RuntimeConfigManager(
            config_path=runtime_config_path,
            base_config={'test_param': 123}
        )
        
        # Save checkpoint with runtime_config
        step = 1000
        metrics = {'val_loss': 0.5, 'ki_quality': 0.85}
        log_file = os.path.join(test_dir, "test.log")
        
        checkpoint_path = checkpoint_mgr.save_checkpoint(
            model, optimizer, scheduler, step, metrics, log_file,
            runtime_config=runtime_config
        )
        
        print(f"✓ Checkpoint saved to: {checkpoint_path}")
        
        # Verify checkpoint file exists
        assert os.path.exists(checkpoint_path), "Checkpoint file not created"
        print(f"✓ Checkpoint file exists")
        
        # Verify config snapshot was created
        config_ref_path = os.path.join(test_dir, f"checkpoint_step_{step:07d}_config_ref.json")
        assert os.path.exists(config_ref_path), "Config reference not created"
        print(f"✓ Config reference created")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Verify checkpoint contents
        assert 'step' in checkpoint, "Step not in checkpoint"
        assert checkpoint['step'] == step, f"Step mismatch: {checkpoint['step']} != {step}"
        print(f"✓ Checkpoint step correct: {step}")
        
        assert 'model_state_dict' in checkpoint, "Model state not in checkpoint"
        print(f"✓ Model state dict present")
        
        assert 'optimizer_state_dict' in checkpoint, "Optimizer state not in checkpoint"
        print(f"✓ Optimizer state dict present")
        
        assert 'metrics' in checkpoint, "Metrics not in checkpoint"
        assert checkpoint['metrics'] == metrics, "Metrics mismatch"
        print(f"✓ Metrics correct")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temporary directory
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        print(f"\nCleaned up test directory")


def test_best_checkpoint_with_runtime_config():
    """Test that best checkpoints save runtime_config correctly"""
    
    # Create temporary directory for test
    test_dir = tempfile.mkdtemp(prefix='checkpoint_test_best_')
    
    try:
        print("\nTesting best checkpoint saving with runtime_config...")
        
        # Create dummy model and optimizer
        model = DummyModel()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = None
        
        # Create checkpoint manager
        checkpoint_mgr = CheckpointManager(test_dir)
        
        # Create runtime config
        runtime_config_path = os.path.join(test_dir, "runtime_config.json")
        runtime_config = RuntimeConfigManager(
            config_path=runtime_config_path,
            base_config={'test_param': 456}
        )
        
        # Save best checkpoint with runtime_config
        step = 5000
        quality = 0.92
        metrics = {'val_loss': 0.3, 'ki_quality': quality}
        log_file = os.path.join(test_dir, "test.log")
        
        is_new_best = checkpoint_mgr.update_best_checkpoint(
            model, optimizer, scheduler, step, quality, metrics, log_file,
            runtime_config=runtime_config
        )
        
        print(f"✓ Best checkpoint saved (is_new_best={is_new_best})")
        
        # Verify checkpoint file exists
        checkpoint_path = os.path.join(test_dir, f"checkpoint_step_{step:07d}.pth")
        assert os.path.exists(checkpoint_path), "Best checkpoint file not created"
        print(f"✓ Best checkpoint file exists")
        
        # Verify config snapshot was created
        config_ref_path = os.path.join(test_dir, f"checkpoint_step_{step:07d}_config_ref.json")
        assert os.path.exists(config_ref_path), "Config reference not created for best checkpoint"
        print(f"✓ Config reference created for best checkpoint")
        
        # Verify best symlink exists
        best_link = os.path.join(test_dir, "checkpoint_best.pth")
        assert os.path.islink(best_link), "Best symlink not created"
        print(f"✓ Best symlink created")
        
        print("\n✅ Best checkpoint tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temporary directory
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        print(f"\nCleaned up test directory")


if __name__ == '__main__':
    print("="*70)
    print("Testing Checkpoint Saving with runtime_config")
    print("="*70)
    
    test1_passed = test_checkpoint_with_runtime_config()
    test2_passed = test_best_checkpoint_with_runtime_config()
    
    print("\n" + "="*70)
    if test1_passed and test2_passed:
        print("✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
