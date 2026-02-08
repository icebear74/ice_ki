#!/usr/bin/env python3
"""
Automated Config Finder for VSR++ Training
Tests all combinations of parameters to find optimal config.

IMPORTANT: This script uses HybridLoss (including VGG16 perceptual loss) to accurately
measure memory usage as it will be in actual training. The perceptual loss component
adds approximately 400-650MB of VRAM usage, which is critical for realistic testing.

Memory components tested:
- Model parameters and gradients
- Optimizer state (Adam: 2x parameters)
- Input/output tensors
- HybridLoss with VGG16 perceptual network
- Multi-scale and gradient loss computations
- Gradient accumulation
"""

import os
import sys
import torch
import torch.nn as nn
import time
import logging
from datetime import datetime

# Add module to path
sys.path.insert(0, os.path.dirname(__file__))

from core.model_5frame import VSRBidirectional_5frames_3x
from core.model_7frame import VSRBidirectional_7frames_3x
from core.loss import HybridLoss

# Setup dual logging (screen + file)
LOG_FILE = "config_test_results.log"
RESULTS_FILE = "config_test_results.txt"

class DualLogger:
    """Logs to both console and file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w')
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def info(self, message):
        self.logger.info(message)
    
    def close(self):
        self.log.close()

# Test parameters
TEST_CONFIGS = {
    'frames': [5, 7],
    'batch_size': [1, 2],
    'n_blocks': [24, 26],
    'n_feats': [60, 72],
    'gt_sizes': [(540, 540), (720, 405), (720, 720)],
    'precision': ['float16', 'float32']
}

# Accumulation steps to reach effective batch size of 8
ACCUMULATION_MAP = {1: 8, 2: 4}

# Dataset path from generator_config.json
DATASET_PATH = "/mnt/data/training/datasetNeu/master"

def get_lr_size(gt_size):
    """Map GT size to LR size (3x downscale)."""
    if gt_size == (540, 540):
        return (180, 180)
    elif gt_size == (720, 405):
        return (240, 135)
    elif gt_size == (720, 720):
        return (240, 240)
    else:
        raise ValueError(f"Unknown GT size: {gt_size}")

def create_model(frames, n_feats, n_blocks, precision):
    """Create model instance."""
    if frames == 5:
        model = VSRBidirectional_5frames_3x(n_feats=n_feats, n_blocks=n_blocks)
    elif frames == 7:
        model = VSRBidirectional_7frames_3x(n_feats=n_feats, n_blocks=n_blocks)
    else:
        raise ValueError(f"Invalid frames: {frames}")
    
    model = model.cuda()
    
    # Set precision
    if precision == 'float16':
        model = model.half()
    
    return model

def create_dummy_batch(frames, batch_size, lr_size, gt_size, precision):
    """
    Create dummy input/target tensors matching real training format.
    
    Args:
        frames: Number of frames (5 or 7)
        batch_size: Batch size
        lr_size: LR frame size (H, W)
        gt_size: GT frame size (H*3, W*3)
        precision: 'float16' or 'float32'
    
    Returns:
        lr_input: [B, T, 3, H, W] - T frames
        gt_target: [B, 3, H*3, W*3] - upscaled center frame
    """
    lr_h, lr_w = lr_size
    gt_h, gt_w = gt_size
    
    # LR: [B, T, C, H, W] format (MATCHES REAL TRAINING!)
    lr_input = torch.randn(batch_size, frames, 3, lr_h, lr_w).cuda()
    
    # GT: single upscaled frame [B, 3, H*3, W*3]
    gt_target = torch.randn(batch_size, 3, gt_h, gt_w).cuda()
    
    if precision == 'float16':
        lr_input = lr_input.half()
        gt_target = gt_target.half()
    
    return lr_input, gt_target

def test_config(frames, batch_size, n_blocks, n_feats, gt_size, precision, logger):
    """Test a single configuration."""
    lr_size = get_lr_size(gt_size)
    accumulation = ACCUMULATION_MAP[batch_size]
    
    config_name = f"{frames}f | B{batch_size}×A{accumulation} | {n_blocks}b | {n_feats}f | {gt_size[0]}×{gt_size[1]} | {precision.upper()}"
    
    logger.info(f"\nTesting: {config_name}")
    
    try:
        # Clear CUDA cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create model
        model = create_model(frames, n_feats, n_blocks, precision)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Use HybridLoss to match actual training (includes VGG16 perceptual loss)
        # This significantly increases memory usage compared to simple L1Loss
        criterion = HybridLoss(
            l1_weight=0.6,
            ms_weight=0.2,
            grad_weight=0.2,
            perceptual_weight=0.1  # Enable perceptual loss (VGG16) for realistic memory test
        )
        if precision == 'float16':
            # VGG stays in FP32 for stability
            criterion = criterion.cuda()
        else:
            criterion = criterion.cuda()
        
        # Warmup (1 iteration)
        lr_input, gt_target = create_dummy_batch(frames, batch_size, lr_size, gt_size, precision)
        output = model(lr_input)
        loss_dict = criterion(output, gt_target)
        loss = loss_dict['total']  # HybridLoss returns a dict
        loss.backward()
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        # Timed iterations (10 iterations with accumulation)
        timings = []
        
        for iter_idx in range(10):
            iter_start = time.time()
            
            # Gradient accumulation loop
            for acc_step in range(accumulation):
                lr_input, gt_target = create_dummy_batch(frames, batch_size, lr_size, gt_size, precision)
                
                output = model(lr_input)
                loss_dict = criterion(output, gt_target)
                loss = loss_dict['total'] / accumulation  # HybridLoss returns a dict
                loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            torch.cuda.synchronize()
            iter_time = time.time() - iter_start
            timings.append(iter_time)
        
        # Get VRAM usage
        vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        vram_gb = vram_mb / 1024
        
        # Average timing
        avg_time = sum(timings) / len(timings)
        
        # Cleanup
        del model, optimizer, criterion
        torch.cuda.empty_cache()
        
        logger.info(f"  ✅ OK | {vram_gb:.2f} GB VRAM | {avg_time:.3f} s/iter")
        
        return {
            'config': config_name,
            'success': True,
            'vram_gb': vram_gb,
            'time_per_iter': avg_time,
            'frames': frames,
            'batch_size': batch_size,
            'accumulation': accumulation,
            'n_blocks': n_blocks,
            'n_feats': n_feats,
            'gt_size': gt_size,
            'precision': precision
        }
        
    except RuntimeError as e:
        if 'out of memory' in str(e):
            logger.info(f"  ❌ OOM!")
            torch.cuda.empty_cache()
            return {
                'config': config_name,
                'success': False,
                'error': 'OOM',
                'frames': frames,
                'batch_size': batch_size,
                'accumulation': accumulation,
                'n_blocks': n_blocks,
                'n_feats': n_feats,
                'gt_size': gt_size,
                'precision': precision
            }
        else:
            logger.info(f"  ❌ ERROR: {e}")
            torch.cuda.empty_cache()
            return {
                'config': config_name,
                'success': False,
                'error': str(e),
                'frames': frames,
                'batch_size': batch_size,
                'accumulation': accumulation,
                'n_blocks': n_blocks,
                'n_feats': n_feats,
                'gt_size': gt_size,
                'precision': precision
            }

def main():
    """Run all config tests."""
    logger = DualLogger(LOG_FILE)
    
    logger.info("=" * 80)
    logger.info("VSR++ Config Finder - REALISTIC Memory & Timing Test")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    logger.info("Testing with FULL training components:")
    logger.info("  ✓ Model architecture matches original VSRBidirectional_3x")
    logger.info("  ✓ HybridLoss with VGG16 perceptual network")
    logger.info("  ✓ Multi-scale and gradient loss components")
    logger.info("  ✓ Adam optimizer with full state")
    logger.info("  ✓ Gradient accumulation")
    logger.info("  ✓ FP16/FP32 precision testing")
    logger.info("")
    logger.info("Memory measurements include:")
    logger.info("  • Model parameters + gradients")
    logger.info("  • Optimizer state (2x parameters for Adam)")
    logger.info("  • VGG16 perceptual network (~400-650 MB)")
    logger.info("  • Intermediate tensors (fusion, gradients, etc.)")
    logger.info("")
    logger.info("⚠️  Accuracy: ±0.2 GB memory, ±1 second timing")
    logger.info("")
    
    # Calculate total combinations
    total_configs = (
        len(TEST_CONFIGS['frames']) *
        len(TEST_CONFIGS['batch_size']) *
        len(TEST_CONFIGS['n_blocks']) *
        len(TEST_CONFIGS['n_feats']) *
        len(TEST_CONFIGS['gt_sizes']) *
        len(TEST_CONFIGS['precision'])
    )
    
    logger.info(f"Total configurations to test: {total_configs}")
    logger.info("")
    
    results = []
    config_idx = 0
    
    # Test all combinations
    for frames in TEST_CONFIGS['frames']:
        for batch_size in TEST_CONFIGS['batch_size']:
            for n_blocks in TEST_CONFIGS['n_blocks']:
                for n_feats in TEST_CONFIGS['n_feats']:
                    for gt_size in TEST_CONFIGS['gt_sizes']:
                        for precision in TEST_CONFIGS['precision']:
                            config_idx += 1
                            logger.info(f"\n[{config_idx}/{total_configs}]")
                            
                            result = test_config(
                                frames, batch_size, n_blocks, n_feats,
                                gt_size, precision, logger
                            )
                            results.append(result)
    
    # Generate report
    logger.info("\n" + "=" * 80)
    logger.info("TESTING COMPLETE")
    logger.info("=" * 80)
    
    # Separate successful and failed configs
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    logger.info(f"\nSuccessful configs: {len(successful)}/{total_configs}")
    logger.info(f"Failed configs: {len(failed)}/{total_configs}")
    
    # Sort successful by VRAM (descending - highest capacity first)
    successful_sorted = sorted(successful, key=lambda x: x['vram_gb'], reverse=True)
    
    # Write detailed results file
    with open(RESULTS_FILE, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("VSR++ CONFIG FINDER - RESULTS\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"Total Configs Tested: {total_configs}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed (OOM): {len(failed)}\n\n")
        
        f.write("=" * 100 + "\n")
        f.write("TOP 10 CONFIGS (by VRAM usage - highest capacity without OOM)\n")
        f.write("=" * 100 + "\n\n")
        
        for idx, r in enumerate(successful_sorted[:10], 1):
            f.write(f"{idx}. {r['config']}\n")
            f.write(f"   VRAM: {r['vram_gb']:.2f} GB | Time: {r['time_per_iter']:.3f} s/iter\n\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("ALL SUCCESSFUL CONFIGS (sorted by VRAM)\n")
        f.write("=" * 100 + "\n\n")
        
        for r in successful_sorted:
            f.write(f"✅ {r['config']}\n")
            f.write(f"   VRAM: {r['vram_gb']:.2f} GB | Time: {r['time_per_iter']:.3f} s/iter\n\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("FAILED CONFIGS\n")
        f.write("=" * 100 + "\n\n")
        
        for r in failed:
            f.write(f"❌ {r['config']}\n")
            f.write(f"   Error: {r.get('error', 'Unknown')}\n\n")
    
    logger.info(f"\nResults saved to: {RESULTS_FILE}")
    logger.info(f"Full log saved to: {LOG_FILE}")
    logger.info(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    logger.close()

if __name__ == "__main__":
    main()
