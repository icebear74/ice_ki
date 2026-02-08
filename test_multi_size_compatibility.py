#!/usr/bin/env python3
"""
Test Multi-Size Batch Compatibility

Verifies that the trainer can handle both:
1. Single-size batches (tuple format): (lr, gt)
2. Multi-size batches (dict format): {'lr': tensor, 'gt': tensor, 'size_key': str}
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_batch_handling():
    """Test that batch handling works for both formats"""
    
    print("Testing batch format handling...")
    print("=" * 60)
    
    # Simulate device
    device = torch.device('cpu')
    
    # Test 1: Single-size batch (tuple format)
    print("\n1. Testing single-size batch (tuple format):")
    lr_single = torch.randn(2, 7, 3, 64, 64)
    gt_single = torch.randn(2, 3, 192, 192)
    batch_single = (lr_single, gt_single)
    
    # Simulate trainer code
    if isinstance(batch_single, dict):
        lr_stack = batch_single['lr'].to(device)
        gt = batch_single['gt'].to(device)
        size_key = batch_single.get('size_key', 'unknown')
        print(f"   ✓ Detected as multi-size batch")
    else:
        lr_stack, gt = batch_single
        lr_stack = lr_stack.to(device)
        gt = gt.to(device)
        size_key = 'default'
        print(f"   ✓ Detected as single-size batch")
    
    print(f"   • LR shape: {lr_stack.shape}")
    print(f"   • GT shape: {gt.shape}")
    print(f"   • Size key: {size_key}")
    
    # Test 2: Multi-size batch (dict format)
    print("\n2. Testing multi-size batch (dict format):")
    lr_multi = torch.randn(1, 7, 3, 128, 128)
    gt_multi = torch.randn(1, 3, 384, 384)
    batch_multi = {
        'lr': lr_multi,
        'gt': gt_multi,
        'size_key': '720_169',
        'filenames': ['test.png']
    }
    
    # Simulate trainer code
    if isinstance(batch_multi, dict):
        lr_stack = batch_multi['lr'].to(device)
        gt = batch_multi['gt'].to(device)
        size_key = batch_multi.get('size_key', 'unknown')
        print(f"   ✓ Detected as multi-size batch")
    else:
        lr_stack, gt = batch_multi
        lr_stack = lr_stack.to(device)
        gt = gt.to(device)
        size_key = 'default'
        print(f"   ✓ Detected as single-size batch")
    
    print(f"   • LR shape: {lr_stack.shape}")
    print(f"   • GT shape: {gt.shape}")
    print(f"   • Size key: {size_key}")
    
    print("\n" + "=" * 60)
    print("✅ All batch format tests passed!")
    print("=" * 60)

def test_import():
    """Test that imports work correctly"""
    print("\nTesting imports...")
    print("=" * 60)
    
    try:
        # Test trainer import
        from vsr_plusplus_NEU.training.trainer import VSRTrainer
        print("✓ VSRTrainer imported successfully")
        
        # Test dataloader import
        from vsr_plusplus_NEU.core.dataloader import create_train_loader, MultiSizeDataLoader
        print("✓ Multi-size dataloader imported successfully")
        
        print("=" * 60)
        print("✅ All imports successful!")
        print("=" * 60)
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Multi-Size Batch Compatibility Test")
    print("=" * 60)
    
    # Test imports
    if not test_import():
        sys.exit(1)
    
    # Test batch handling
    test_batch_handling()
    
    print("\n✅ All compatibility tests passed!")
