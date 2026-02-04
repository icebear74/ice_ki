"""
Auto-Tune System - Automatically find optimal configuration

Tests different model configurations to find the best balance of:
- VRAM usage
- Training speed
- Model capacity
"""

import torch
import time
import sys
sys.path.append('/home/runner/work/ice_ki/ice_ki')

from vsr_plus_plus.core.model import VSRBidirectional_3x


def auto_tune_config(target_speed=4.0, max_vram_gb=6.0, min_effective_batch=4):
    """
    Auto-tune configuration for optimal VRAM/speed balance
    
    Args:
        target_speed: Target speed in seconds per iteration
        max_vram_gb: Maximum VRAM in GB
        min_effective_batch: Minimum effective batch size (with accumulation)
        
    Returns:
        Dict with optimal configuration:
        {
            'n_feats': int,
            'n_blocks': int,
            'batch_size': int,
            'accumulation_steps': int,
            'measured_speed': float,
            'measured_vram': float,
            'total_params': int
        }
    """
    print("\n" + "╔" + "═" * 71 + "╗")
    print("║" + " " * 20 + "🔧 AUTO-TUNING SYSTEM" + " " * 30 + "║")
    print("╠" + "═" * 71 + "╣")
    print("║ Testing Configurations..." + " " * 44 + "║")
    print("║" + " " * 71 + "║")
    
    # Test configurations in priority order
    configs = [
        {'n_feats': 192, 'batch': 3, 'n_blocks': 32},
        {'n_feats': 160, 'batch': 4, 'n_blocks': 32},
        {'n_feats': 128, 'batch': 4, 'n_blocks': 32},
        {'n_feats': 192, 'batch': 2, 'n_blocks': 32},
        {'n_feats': 160, 'batch': 3, 'n_blocks': 24},
        {'n_feats': 128, 'batch': 3, 'n_blocks': 24},
        {'n_feats': 96, 'batch': 4, 'n_blocks': 24},
        {'n_feats': 64, 'batch': 4, 'n_blocks': 20},
    ]
    
    # 80% safety margin for VRAM
    effective_vram_budget = max_vram_gb * 0.8
    
    optimal_config = None
    
    for idx, cfg in enumerate(configs, 1):
        n_feats = cfg['n_feats']
        batch = cfg['batch']
        n_blocks = cfg['n_blocks']
        
        # Display current test
        config_str = f"n_feats={n_feats}, batch={batch}, blocks={n_blocks}"
        print(f"║ [{idx}/8] {config_str:<59}║")
        
        try:
            # Create model
            model = VSRBidirectional_3x(n_feats=n_feats, n_blocks=n_blocks).cuda()
            model.train()
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            
            # Create dummy input
            dummy_input = torch.randn(batch, 5, 3, 180, 180).cuda()
            
            # Warmup
            with torch.no_grad():
                _ = model(dummy_input)
            
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            
            # Measure speed (5 forward passes)
            start_time = time.time()
            for _ in range(5):
                output = model(dummy_input)
                torch.cuda.synchronize()
            
            elapsed = time.time() - start_time
            avg_time = elapsed / 5
            
            # Measure VRAM
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
            
            # Check if configuration passes
            passes = avg_time <= target_speed and peak_vram <= effective_vram_budget
            
            status = "✅ PASSED" if passes else "❌ FAILED"
            print(f"║       ⏱️  {avg_time:.1f}s/iter │ 💾 {peak_vram:.1f}GB │ {status:<25}║")
            print("║" + " " * 71 + "║")
            
            # Clean up
            del model, dummy_input, output
            torch.cuda.empty_cache()
            
            # If this config passes, use it
            if passes:
                # Calculate accumulation steps
                accumulation_steps = max(1, min_effective_batch // batch)
                
                optimal_config = {
                    'n_feats': n_feats,
                    'n_blocks': n_blocks,
                    'batch_size': batch,
                    'accumulation_steps': accumulation_steps,
                    'measured_speed': avg_time,
                    'measured_vram': peak_vram,
                    'total_params': total_params
                }
                break
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"║       💾 OOM - Skipping" + " " * 46 + "║")
                print("║" + " " * 71 + "║")
                torch.cuda.empty_cache()
            else:
                raise
    
    if optimal_config is None:
        print("╠" + "═" * 71 + "╣")
        print("║ ❌ NO SUITABLE CONFIGURATION FOUND" + " " * 36 + "║")
        print("║ Please reduce target_speed or increase max_vram_gb" + " " * 20 + "║")
        print("╚" + "═" * 71 + "╝")
        raise RuntimeError("Auto-tune failed to find suitable configuration")
    
    # Display results
    print("╠" + "═" * 71 + "╣")
    print("║ ✅ OPTIMAL CONFIGURATION FOUND" + " " * 40 + "║")
    
    feat_str = f"Features: {optimal_config['n_feats']}"
    batch_str = f"Batch: {optimal_config['batch_size']}"
    blocks_str = f"Blocks: {optimal_config['n_blocks']}"
    config_line = f"   {feat_str} | {batch_str} | {blocks_str}"
    padding = 71 - len(config_line)
    print("║" + config_line + " " * padding + "║")
    
    accum_str = f"Accumulation: {optimal_config['accumulation_steps']}"
    eff_batch_str = f"Effective Batch: {optimal_config['batch_size'] * optimal_config['accumulation_steps']}"
    accum_line = f"   {accum_str} | {eff_batch_str}"
    padding = 71 - len(accum_line)
    print("║" + accum_line + " " * padding + "║")
    
    speed_str = f"Speed: {optimal_config['measured_speed']:.2f}s/iter"
    vram_str = f"VRAM: {optimal_config['measured_vram']:.2f}GB"
    params_str = f"Params: {optimal_config['total_params']/1e6:.2f}M"
    perf_line = f"   {speed_str} | {vram_str} | {params_str}"
    padding = 71 - len(perf_line)
    print("║" + perf_line + " " * padding + "║")
    
    print("╠" + "═" * 71 + "╣")
    print("║" + " " * 15 + "📸 Press ENTER to continue training..." + " " * 18 + "║")
    print("╚" + "═" * 71 + "╝")
    
    # Wait for user
    input()
    
    return optimal_config


if __name__ == '__main__':
    # Test auto-tune
    config = auto_tune_config()
    print(f"\nOptimal config: {config}")
