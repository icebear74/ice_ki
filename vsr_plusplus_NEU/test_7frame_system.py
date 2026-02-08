#!/usr/bin/env python3
"""
Test script for 7-frame VSR training system components
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_adaptive_batch():
    """Test adaptive batch calculator"""
    print("\n" + "="*80)
    print("Testing Adaptive Batch Calculator")
    print("="*80 + "\n")
    
    # Direct import to avoid torch dependency
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "adaptive_batch",
        "vsr_plusplus_NEU/systems/adaptive_batch.py"
    )
    adaptive_batch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adaptive_batch)
    AdaptiveBatchCalculator = adaptive_batch.AdaptiveBatchCalculator
    
    calculator = AdaptiveBatchCalculator()
    
    # Test with different effective batch sizes
    for effective_batch in [4, 6, 8, 12]:
        print(f"Testing effective_batch_size = {effective_batch}:")
        configs = calculator.calculate_all_configs(effective_batch)
        calculator.print_config_table(configs)
        
        # Validate
        is_valid, errors = calculator.validate_all_configs(configs)
        if is_valid:
            print(f"✅ All configurations valid for effective_batch={effective_batch}\n")
        else:
            print(f"❌ Validation errors:")
            for size, error_list in errors.items():
                for error in error_list:
                    print(f"  {size}: {error}")
            print()


def test_runtime_config():
    """Test runtime configuration manager"""
    print("\n" + "="*80)
    print("Testing Runtime Configuration Manager")
    print("="*80 + "\n")
    
    # Direct import to avoid torch dependency
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "runtime_config",
        "vsr_plusplus_NEU/systems/runtime_config.py"
    )
    runtime_config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime_config_module)
    EnhancedRuntimeConfigManager = runtime_config_module.EnhancedRuntimeConfigManager
    
    import tempfile
    import json
    
    # Create temp config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config_path = f.name
    
    try:
        # Create config manager
        manager = EnhancedRuntimeConfigManager(config_path, use_new_structure=True)
        
        print("Initial configuration:")
        config = manager.get_all()
        print(json.dumps(config, indent=2))
        print()
        
        # Validate
        is_valid, errors = manager.validate()
        if is_valid:
            print("✅ Configuration is valid!\n")
        else:
            print("❌ Validation errors:")
            for error in errors:
                print(f"  - {error}")
            print()
        
        # Update effective batch size
        print("Updating effective batch size to 8...")
        success = manager.update_effective_batch_size(8)
        if success:
            print("✅ Updated successfully\n")
        else:
            print("❌ Update failed\n")
        
        # Update size distribution
        print("Updating size distribution...")
        success = manager.update_size_distribution({
            'small_540': 0.70,
            'medium_169': 0.30,
            'large_720': 0.00
        })
        if success:
            print("✅ Updated successfully\n")
        else:
            print("❌ Update failed\n")
        
        # Final validation
        is_valid, errors = manager.validate()
        if is_valid:
            print("✅ Final configuration is valid!\n")
        else:
            print("❌ Final validation errors:")
            for error in errors:
                print(f"  - {error}")
            print()
    
    finally:
        # Cleanup
        if os.path.exists(config_path):
            os.unlink(config_path)


def test_size_tracking():
    """Test size tracking system"""
    print("\n" + "="*80)
    print("Testing Size Tracking System")
    print("="*80 + "\n")
    
    # Direct import to avoid torch dependency
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "size_tracking",
        "vsr_plusplus_NEU/systems/size_tracking.py"
    )
    size_tracking_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(size_tracking_module)
    SizeTracker = size_tracking_module.SizeTracker
    
    import tempfile
    
    # Create temp tracking file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        tracking_path = f.name
    
    try:
        # Create tracker
        tracker = SizeTracker(save_path=tracking_path)
        
        # Set targets
        distribution = {
            'small_540': 0.65,
            'medium_169': 0.35,
            'large_720': 0.00,
        }
        tracker.update_targets(distribution, total_target=1000)
        
        print("Simulating training...")
        # Simulate training
        for step in range(1, 51):
            if step % 2 == 0:
                tracker.record_batch('small_540', batch_size=1, step=step)
            else:
                tracker.record_batch('medium_169', batch_size=1, step=step)
        
        # Print summary
        print(tracker.get_summary())
        print()
        
        # Test checkpoint integration
        checkpoint_data = tracker.to_checkpoint_dict()
        print("✅ Checkpoint data created")
        print(f"   - Total images: {checkpoint_data['size_tracking']['total_images_trained']}")
        print()
        
        # Test restoration
        new_tracker = SizeTracker(save_path=tracking_path + '.new')
        success = new_tracker.from_checkpoint_dict(checkpoint_data)
        if success:
            print("✅ Successfully restored from checkpoint\n")
        else:
            print("❌ Failed to restore from checkpoint\n")
    
    finally:
        # Cleanup
        if os.path.exists(tracking_path):
            os.unlink(tracking_path)
        if os.path.exists(tracking_path + '.new'):
            os.unlink(tracking_path + '.new')


def test_terminal_ui():
    """Test terminal UI functions"""
    print("\n" + "="*80)
    print("Testing Terminal UI Functions")
    print("="*80 + "\n")
    
    # Direct import to avoid torch dependency
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ui_terminal",
        "vsr_plusplus_NEU/utils/ui_terminal.py"
    )
    ui_terminal_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui_terminal_module)
    make_size_bar = ui_terminal_module.make_size_bar
    format_size_stats_compact = ui_terminal_module.format_size_stats_compact
    
    # Test size bar
    print("Size progress bars:")
    for trained, target in [(0, 100), (50, 100), (90, 100), (100, 100)]:
        bar = make_size_bar(trained, target, width=30)
        print(f"  {trained:3d}/{target:3d}: {bar}")
    print()
    
    # Test compact stats
    size_stats = {
        'size_stats': {
            'small_540': {'images_trained': 650, 'target_images': 650, 'percentage_complete': 100.0},
            'medium_169': {'images_trained': 200, 'target_images': 350, 'percentage_complete': 57.1},
            'large_720': {'images_trained': 0, 'target_images': 0, 'percentage_complete': 0.0},
        }
    }
    
    compact = format_size_stats_compact(size_stats)
    print(f"Compact stats: {compact}")
    print()
    
    print("✅ Terminal UI functions working\n")


def test_models():
    """Test model imports"""
    print("\n" + "="*80)
    print("Testing Model Imports")
    print("="*80 + "\n")
    
    try:
        from vsr_plusplus_NEU.core.model_5frame import VSRBidirectional_5frames_3x
        from vsr_plusplus_NEU.core.model_7frame import VSRBidirectional_7frames_3x
        
        # Create models with correct parameters
        model_5 = VSRBidirectional_5frames_3x(n_feats=72, n_blocks=26)
        model_7 = VSRBidirectional_7frames_3x(n_feats=72, n_blocks=26)
        
        print(f"✅ 5-frame model created: {model_5.n_feats} features, {model_5.n_blocks} blocks")
        print(f"✅ 7-frame model created: {model_7.n_feats} features, {model_7.n_blocks} blocks")
        print()
        
        # Count parameters
        params_5 = sum(p.numel() for p in model_5.parameters())
        params_7 = sum(p.numel() for p in model_7.parameters())
        
        print(f"5-frame model parameters: {params_5:,}")
        print(f"7-frame model parameters: {params_7:,}")
        print()
        
    except Exception as e:
        print(f"❌ Error importing models: {e}\n")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("7-Frame VSR Training System - Component Tests")
    print("="*80)
    
    # Run tests that don't require torch
    test_adaptive_batch()
    test_runtime_config()
    test_size_tracking()
    test_terminal_ui()
    
    # Try model tests (may fail if torch not installed)
    try:
        test_models()
    except Exception as e:
        print(f"\nℹ️  Skipping model tests (torch not available): {e}\n")
    
    print("="*80)
    print("All Tests Complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
