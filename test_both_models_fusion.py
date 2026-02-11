#!/usr/bin/env python3
"""
Test script for FusionBlock enhancement in both 5-frame and 7-frame VSR models
"""

import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_5frame_model():
    """Test that the 5-frame model has FusionBlocks"""
    print("\n" + "="*80)
    print("Testing 5-Frame Model (vsr_plus_plus)")
    print("="*80 + "\n")
    
    try:
        from vsr_plus_plus.core.model import VSRBidirectional_3x, FusionBlock
        
        # Create model with default parameters
        model = VSRBidirectional_3x(n_feats=128, n_blocks=32)
        
        print("✅ 5-frame model created successfully")
        print(f"   - Features: {model.n_feats}")
        print(f"   - Blocks: {model.n_blocks}")
        
        # Check that fusion layers are FusionBlocks
        assert isinstance(model.backward_fuse, FusionBlock), "backward_fuse should be a FusionBlock"
        assert isinstance(model.forward_fuse, FusionBlock), "forward_fuse should be a FusionBlock"
        assert isinstance(model.fusion, FusionBlock), "fusion should be a FusionBlock"
        
        print("✅ All fusion layers are FusionBlocks")
        
        # Test forward pass
        model.eval()
        dummy_input = torch.randn(1, 5, 3, 64, 64)  # 5 frames
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ Forward pass successful")
        print(f"   - Input shape: {list(dummy_input.shape)}")
        print(f"   - Output shape: {list(output.shape)}")
        
        # Test activity tracking
        activities = model.get_layer_activity()
        print(f"✅ Activity tracking working")
        print(f"   - backward_fuse: {activities['backward_fuse']:.6f}")
        print(f"   - forward_fuse: {activities['forward_fuse']:.6f}")
        print(f"   - fusion: {activities['fusion']:.6f}")
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"   - Total parameters: {params:,}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing 5-frame model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_7frame_model():
    """Test that the 7-frame model has FusionBlocks"""
    print("\n" + "="*80)
    print("Testing 7-Frame Model (vsr_plusplus_NEU)")
    print("="*80 + "\n")
    
    try:
        from vsr_plusplus_NEU.core.model_7frame import VSRBidirectional_7frames_3x, FusionBlock
        
        # Create model with default parameters
        model = VSRBidirectional_7frames_3x(n_feats=72, n_blocks=26)
        
        print("✅ 7-frame model created successfully")
        print(f"   - Features: {model.n_feats}")
        print(f"   - Blocks: {model.n_blocks}")
        
        # Check that fusion layers are FusionBlocks
        assert isinstance(model.backward_fuse, FusionBlock), "backward_fuse should be a FusionBlock"
        assert isinstance(model.forward_fuse, FusionBlock), "forward_fuse should be a FusionBlock"
        assert isinstance(model.fusion, FusionBlock), "fusion should be a FusionBlock"
        
        print("✅ All fusion layers are FusionBlocks")
        
        # Test forward pass
        model.eval()
        dummy_input = torch.randn(1, 7, 3, 64, 64)  # 7 frames
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ Forward pass successful")
        print(f"   - Input shape: {list(dummy_input.shape)}")
        print(f"   - Output shape: {list(output.shape)}")
        
        # Test activity tracking
        activities = model.get_layer_activity()
        print(f"✅ Activity tracking working")
        print(f"   - backward_fuse: {activities['backward_fuse']:.6f}")
        print(f"   - forward_fuse: {activities['forward_fuse']:.6f}")
        print(f"   - fusion: {activities['fusion']:.6f}")
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"   - Total parameters: {params:,}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing 7-frame model: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_fusion_blocks():
    """Compare FusionBlock implementations"""
    print("\n" + "="*80)
    print("Comparing FusionBlock Implementations")
    print("="*80 + "\n")
    
    try:
        from vsr_plus_plus.core.model import FusionBlock as FusionBlock5
        from vsr_plusplus_NEU.core.model_7frame import FusionBlock as FusionBlock7
        
        # Create instances
        fb5 = FusionBlock5(in_feats=144, out_feats=72)
        fb7 = FusionBlock7(in_feats=144, out_feats=72)
        
        # Test with same input
        dummy_input = torch.randn(1, 144, 32, 32)
        with torch.no_grad():
            out5 = fb5(dummy_input)
            out7 = fb7(dummy_input)
        
        print("✅ Both FusionBlocks work correctly")
        print(f"   - 5-frame FusionBlock output shape: {list(out5.shape)}")
        print(f"   - 7-frame FusionBlock output shape: {list(out7.shape)}")
        print(f"   - 5-frame activity: {fb5.last_activity:.6f}")
        print(f"   - 7-frame activity: {fb7.last_activity:.6f}")
        
        # Check structure
        print("\n   FusionBlock structure check:")
        print(f"   - 5-frame has conv3x3: {hasattr(fb5, 'conv3x3')}")
        print(f"   - 5-frame has relu: {hasattr(fb5, 'relu')}")
        print(f"   - 5-frame has conv1x1: {hasattr(fb5, 'conv1x1')}")
        print(f"   - 7-frame has conv3x3: {hasattr(fb7, 'conv3x3')}")
        print(f"   - 7-frame has relu: {hasattr(fb7, 'relu')}")
        print(f"   - 7-frame has conv1x1: {hasattr(fb7, 'conv1x1')}")
        
        return True
    except Exception as e:
        print(f"❌ Error comparing FusionBlocks: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("FusionBlock Enhancement - Both Models Test")
    print("="*80)
    
    results = []
    
    results.append(("5-Frame Model", test_5frame_model()))
    results.append(("7-Frame Model", test_7frame_model()))
    results.append(("FusionBlock Comparison", compare_fusion_blocks()))
    
    print("\n" + "="*80)
    print("Test Results Summary")
    print("="*80 + "\n")
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ All tests PASSED! Both models have FusionBlocks.")
    else:
        print("❌ Some tests FAILED!")
    print("="*80 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
