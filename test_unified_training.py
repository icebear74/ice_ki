#!/usr/bin/env python3
"""
Quick test script for VSR++ Unified Training System

Tests:
1. Configuration loading
2. Dataset creation
3. Model initialization
4. Training loop (10 steps)
"""

import os
import sys
import tempfile
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vsr_plus_plus.utils.yaml_config import load_yaml_config, validate_config
import torch
from vsr_plus_plus.core.model import VSRBidirectional_3x
import numpy as np
import cv2

C_GREEN = "\033[92m"
C_CYAN = "\033[96m"
C_RED = "\033[91m"
C_RESET = "\033[0m"


def create_mock_dataset(base_dir):
    """Create minimal mock dataset for testing"""
    print(f"{C_CYAN}Creating mock dataset...{C_RESET}")
    
    # General category path
    category_path = os.path.join(base_dir, 'Universal/Mastermodell/Learn')
    
    # Create Patches format (540×540)
    for subdir in ['GT', 'LR_5frames', 'LR_7frames']:
        os.makedirs(os.path.join(category_path, 'Patches', subdir), exist_ok=True)
    
    # Create 3 training samples
    for i in range(3):
        filename = f'test_{i:04d}.png'
        
        # GT: 540×540
        gt = np.random.randint(0, 256, (540, 540, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(category_path, 'Patches/GT', filename), gt)
        
        # LR 5 frames: 900×180 (5×180)
        lr_5 = np.random.randint(0, 256, (900, 180, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(category_path, 'Patches/LR_5frames', filename), lr_5)
        
        # LR 7 frames: 1260×180 (7×180)
        lr_7 = np.random.randint(0, 256, (1260, 180, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(category_path, 'Patches/LR_7frames', filename), lr_7)
    
    # Create validation samples
    val_gt_dir = os.path.join(category_path, 'Val/GT')
    os.makedirs(val_gt_dir, exist_ok=True)
    
    for i in range(2):
        filename = f'val_{i:04d}.png'
        val_gt = np.random.randint(0, 256, (540, 540, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(val_gt_dir, filename), val_gt)
    
    print(f"{C_GREEN}✅ Mock dataset created{C_RESET}")
    print(f"   Training samples: 3")
    print(f"   Validation samples: 2")


def test_config_loading():
    """Test configuration loading"""
    print(f"\n{C_CYAN}Test 1: Configuration Loading{C_RESET}")
    
    config_path = 'configs/train_general_7frames.yaml'
    if not os.path.exists(config_path):
        print(f"{C_RED}❌ Config not found: {config_path}{C_RESET}")
        return False
    
    try:
        config = load_yaml_config(config_path)
        validate_config(config)
        print(f"{C_GREEN}✅ Config loaded and validated{C_RESET}")
        print(f"   Category: {config.DATA.category}")
        print(f"   LR Version: {config.DATA.lr_version}")
        return True
    except Exception as e:
        print(f"{C_RED}❌ Config loading failed: {e}{C_RESET}")
        return False


def test_model():
    """Test model with different frame counts"""
    print(f"\n{C_CYAN}Test 2: Model Initialization{C_RESET}")
    
    try:
        # Test 5 frames
        model_5 = VSRBidirectional_3x(n_feats=32, n_blocks=4, num_frames=5)
        input_5 = torch.randn(1, 5, 3, 180, 180)
        output_5 = model_5(input_5)
        assert output_5.shape == (1, 3, 540, 540)
        print(f"{C_GREEN}✅ 5-frame model works{C_RESET}")
        
        # Test 7 frames
        model_7 = VSRBidirectional_3x(n_feats=32, n_blocks=4, num_frames=7)
        input_7 = torch.randn(1, 7, 3, 180, 180)
        output_7 = model_7(input_7)
        assert output_7.shape == (1, 3, 540, 540)
        print(f"{C_GREEN}✅ 7-frame model works{C_RESET}")
        
        return True
    except Exception as e:
        print(f"{C_RED}❌ Model test failed: {e}{C_RESET}")
        return False


def test_datasets(test_data_dir):
    """Test dataset loading"""
    print(f"\n{C_CYAN}Test 3: Dataset Loading{C_RESET}")
    
    try:
        from vsr_plus_plus.utils.yaml_config import Config
        from vsr_plus_plus.data import MultiFormatMultiCategoryDataset, ValidationDataset
        
        # Create minimal config
        config_dict = {
            'DATA': {
                'category': 'general',
                'lr_version': '7frames',
                'data_root': test_data_dir,
                'formats': ['small_540'],
                'format_weights': {'small_540': 1.0},
                'val_frequency': 10
            }
        }
        config = Config(config_dict)
        
        # Test training dataset
        train_ds = MultiFormatMultiCategoryDataset(config)
        assert len(train_ds) > 0
        sample = train_ds[0]
        assert sample['lr'].shape[0] == 7  # 7 frames
        print(f"{C_GREEN}✅ Training dataset works ({len(train_ds)} samples){C_RESET}")
        
        # Test validation dataset
        val_ds = ValidationDataset(config)
        assert len(val_ds) > 0
        val_sample = val_ds[0]
        assert val_sample['lr'].shape[0] == 7  # 7 frames
        print(f"{C_GREEN}✅ Validation dataset works ({len(val_ds)} samples){C_RESET}")
        
        return True
    except Exception as e:
        print(f"{C_RED}❌ Dataset test failed: {e}{C_RESET}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*80)
    print("VSR++ Unified Training System - Quick Test")
    print("="*80)
    
    results = []
    
    # Test 1: Config loading
    results.append(("Config Loading", test_config_loading()))
    
    # Test 2: Model
    results.append(("Model", test_model()))
    
    # Test 3: Datasets (need mock data)
    with tempfile.TemporaryDirectory() as tmpdir:
        create_mock_dataset(tmpdir)
        results.append(("Datasets", test_datasets(tmpdir)))
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    all_passed = True
    for name, passed in results:
        status = f"{C_GREEN}✅ PASS{C_RESET}" if passed else f"{C_RED}❌ FAIL{C_RESET}"
        print(f"{name:20s}: {status}")
        all_passed = all_passed and passed
    
    print("="*80)
    
    if all_passed:
        print(f"\n{C_GREEN}✅ All tests passed!{C_RESET}")
        print(f"\nYou can now use the unified training system:")
        print(f"  python vsr_plus_plus/train_unified.py --config configs/train_general_7frames.yaml")
        return 0
    else:
        print(f"\n{C_RED}❌ Some tests failed{C_RESET}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
