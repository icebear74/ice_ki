#!/usr/bin/env python3
"""
Automated Config Finder for VSR++ Training
Tests all combinations of parameters to find optimal config.
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
    """Create dummy input/target tensors."""
    lr_h, lr_w = lr_size
    gt_h, gt_w = gt_size
    
    # LR: stacked frames horizontally
    lr_input = torch.randn(batch_size, 3, lr_h, lr_w * frames).cuda()
    
    # GT: single upscaled frame
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
        criterion = nn.L1Loss()
        
        # Warmup (1 iteration)
        lr_input, gt_target = create_dummy_batch(frames, batch_size, lr_size, gt_size, precision)
        output = model(lr_input)
        loss = criterion(output, gt_target)
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
                loss = criterion(output, gt_target) / accumulation
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
    logger.info("VSR++ Config Finder - Automated Testing")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
