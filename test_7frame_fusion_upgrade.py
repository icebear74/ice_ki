#!/usr/bin/env python3
"""
Test script for enhanced fusion layers in 7-frame VSR model
Tests activity tracking and FusionBlock functionality
"""

import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_instantiation():
    """Test that the model can be instantiated with FusionBlocks"""
    print("\n" + "="*80)
    print("Testing Model Instantiation")
    print("="*80 + "\n")
    
    try:
        from vsr_plusplus_NEU.core.model_7frame import VSRBidirectional_7frames_3x
        
        # Create model with default parameters
        model = VSRBidirectional_7frames_3x(n_feats=72, n_blocks=26)
        
        print("✅ Model created successfully")
        print(f"   - Features: {model.n_feats}")
        print(f"   - Blocks: {model.n_blocks}")
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"   - Total parameters: {params:,}")
        
        return True
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fusion_blocks():
    """Test that FusionBlocks are properly initialized"""
    print("\n" + "="*80)
    print("Testing FusionBlock Structure")
    print("="*80 + "\n")
    
    try:
        from vsr_plusplus_NEU.core.model_7frame import VSRBidirectional_7frames_3x, FusionBlock
        
        model = VSRBidirectional_7frames_3x(n_feats=72, n_blocks=26)
        
        # Check that fusion layers are FusionBlocks
        assert isinstance(model.backward_fuse, FusionBlock), "backward_fuse should be a FusionBlock"
        assert isinstance(model.forward_fuse, FusionBlock), "forward_fuse should be a FusionBlock"
        assert isinstance(model.fusion, FusionBlock), "fusion should be a FusionBlock"
        
        print("✅ All fusion layers are FusionBlocks")
        
        # Check FusionBlock structure
        print("\n   Checking FusionBlock layers:")
        print(f"   - backward_fuse has conv3x3: {hasattr(model.backward_fuse, 'conv3x3')}")
        print(f"   - backward_fuse has relu: {hasattr(model.backward_fuse, 'relu')}")
        print(f"   - backward_fuse has conv1x1: {hasattr(model.backward_fuse, 'conv1x1')}")
        print(f"   - backward_fuse has last_activity: {hasattr(model.backward_fuse, 'last_activity')}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing FusionBlocks: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_activity_tracking():
    """Test that activity tracking works correctly"""
    print("\n" + "="*80)
    print("Testing Activity Tracking")
    print("="*80 + "\n")
    
    try:
        from vsr_plusplus_NEU.core.model_7frame import VSRBidirectional_7frames_3x
        
        model = VSRBidirectional_7frames_3x(n_feats=72, n_blocks=26)
        model.eval()
        
        # Create dummy input (batch_size=1, 7 frames, 3 channels, 64x64)
        dummy_input = torch.randn(1, 7, 3, 64, 64)
        
        # Initial activities should be 0
        initial_activity = model.get_layer_activity()
        print("Initial activities:")
        print(f"   - backward_fuse: {initial_activity['backward_fuse']}")
        print(f"   - forward_fuse: {initial_activity['forward_fuse']}")
        print(f"   - fusion: {initial_activity['fusion']}")
        print(f"   - backward_trunk blocks: {len(initial_activity['backward_trunk'])}")
        print(f"   - forward_trunk blocks: {len(initial_activity['forward_trunk'])}")
        
        # Run forward pass
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"\n✅ Forward pass completed")
        print(f"   - Input shape: {dummy_input.shape}")
        print(f"   - Output shape: {output.shape}")
        
        # Check activities after forward pass
        post_activity = model.get_layer_activity()
        print("\nActivities after forward pass:")
        print(f"   - backward_fuse: {post_activity['backward_fuse']:.6f}")
        print(f"   - forward_fuse: {post_activity['forward_fuse']:.6f}")
        print(f"   - fusion: {post_activity['fusion']:.6f}")
        
        # Check that activities are non-zero
        assert post_activity['backward_fuse'] > 0, "backward_fuse activity should be > 0"
        assert post_activity['forward_fuse'] > 0, "forward_fuse activity should be > 0"
        assert post_activity['fusion'] > 0, "fusion activity should be > 0"
        
        # Check backward trunk activities
        print(f"\n   Backward trunk activities (first 3):")
        for i, act in enumerate(post_activity['backward_trunk'][:3]):
            print(f"      Block {i}: {act:.6f}")
            assert act > 0, f"backward_trunk block {i} activity should be > 0"
        
        # Check forward trunk activities
        print(f"\n   Forward trunk activities (first 3):")
        for i, act in enumerate(post_activity['forward_trunk'][:3]):
            print(f"      Block {i}: {act:.6f}")
            assert act > 0, f"forward_trunk block {i} activity should be > 0"
        
        print("\n✅ All activity tracking working correctly")
        
        return True
    except Exception as e:
        print(f"❌ Error testing activity tracking: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_get_layer_activity_method():
    """Test that get_layer_activity returns correct structure"""
    print("\n" + "="*80)
    print("Testing get_layer_activity() Method")
    print("="*80 + "\n")
    
    try:
        from vsr_plusplus_NEU.core.model_7frame import VSRBidirectional_7frames_3x
        
        model = VSRBidirectional_7frames_3x(n_feats=72, n_blocks=26)
        
        # Check method exists
        assert hasattr(model, 'get_layer_activity'), "Model should have get_layer_activity method"
        
        # Get activity
        activity = model.get_layer_activity()
        
        # Check structure
        assert isinstance(activity, dict), "Activity should be a dictionary"
        assert 'backward_trunk' in activity, "Activity should contain backward_trunk"
        assert 'backward_fuse' in activity, "Activity should contain backward_fuse"
        assert 'forward_trunk' in activity, "Activity should contain forward_trunk"
        assert 'forward_fuse' in activity, "Activity should contain forward_fuse"
        assert 'fusion' in activity, "Activity should contain fusion"
        
        # Check types
        assert isinstance(activity['backward_trunk'], list), "backward_trunk should be a list"
        assert isinstance(activity['forward_trunk'], list), "forward_trunk should be a list"
        assert isinstance(activity['backward_fuse'], float), "backward_fuse should be a float"
        assert isinstance(activity['forward_fuse'], float), "forward_fuse should be a float"
        assert isinstance(activity['fusion'], float), "fusion should be a float"
        
        # Check list lengths (should be half_blocks = 13 for n_blocks=26)
        expected_trunk_length = max(1, 26 // 2)
        assert len(activity['backward_trunk']) == expected_trunk_length, \
            f"backward_trunk should have {expected_trunk_length} blocks"
        assert len(activity['forward_trunk']) == expected_trunk_length, \
            f"forward_trunk should have {expected_trunk_length} blocks"
        
        print("✅ get_layer_activity() method structure is correct")
        print(f"   - backward_trunk blocks: {len(activity['backward_trunk'])}")
        print(f"   - forward_trunk blocks: {len(activity['forward_trunk'])}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing get_layer_activity: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_residual_block_activity():
    """Test that ResidualBlock tracks activity"""
    print("\n" + "="*80)
    print("Testing ResidualBlock Activity Tracking")
    print("="*80 + "\n")
    
    try:
        from vsr_plusplus_NEU.core.model_7frame import ResidualBlock
        
        block = ResidualBlock(n_feats=64)
        
        # Check initial activity
        assert hasattr(block, 'last_activity'), "ResidualBlock should have last_activity"
        assert block.last_activity == 0.0, "Initial activity should be 0.0"
        
        # Run forward pass
        dummy_input = torch.randn(1, 64, 32, 32)
        with torch.no_grad():
            output = block(dummy_input)
        
        # Check activity updated
        assert block.last_activity > 0, "Activity should be updated after forward pass"
        
        print("✅ ResidualBlock activity tracking works")
        print(f"   - Initial activity: 0.0")
        print(f"   - Activity after forward: {block.last_activity:.6f}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing ResidualBlock activity: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fusion_block_standalone():
    """Test FusionBlock as a standalone component"""
    print("\n" + "="*80)
    print("Testing FusionBlock Standalone")
    print("="*80 + "\n")
    
    try:
        from vsr_plusplus_NEU.core.model_7frame import FusionBlock
        
        # Create FusionBlock with 144 -> 72 features (like in the model)
        fusion = FusionBlock(in_feats=144, out_feats=72)
        
        # Check structure
        assert hasattr(fusion, 'conv3x3'), "FusionBlock should have conv3x3"
        assert hasattr(fusion, 'relu'), "FusionBlock should have relu"
        assert hasattr(fusion, 'conv1x1'), "FusionBlock should have conv1x1"
        assert hasattr(fusion, 'last_activity'), "FusionBlock should have last_activity"
        
        # Check layer types and sizes
        assert fusion.conv3x3.kernel_size == (3, 3), "conv3x3 should have 3x3 kernel"
        assert fusion.conv3x3.in_channels == 144, "conv3x3 input should be 144"
        assert fusion.conv3x3.out_channels == 72, "conv3x3 output should be 72"
        assert fusion.conv1x1.kernel_size == (1, 1), "conv1x1 should have 1x1 kernel"
        assert fusion.conv1x1.in_channels == 72, "conv1x1 input should be 72"
        assert fusion.conv1x1.out_channels == 72, "conv1x1 output should be 72"
        
        # Test forward pass
        dummy_input = torch.randn(1, 144, 32, 32)
        with torch.no_grad():
            output = fusion(dummy_input)
        
        assert output.shape == (1, 72, 32, 32), "Output shape should be (1, 72, 32, 32)"
        assert fusion.last_activity > 0, "Activity should be tracked"
        
        print("✅ FusionBlock standalone test passed")
        print(f"   - Input shape: {dummy_input.shape}")
        print(f"   - Output shape: {output.shape}")
        print(f"   - Activity: {fusion.last_activity:.6f}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing FusionBlock standalone: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("7-Frame VSR Fusion Layer Enhancement Tests")
    print("="*80)
    
    results = []
    
    results.append(("Model Instantiation", test_model_instantiation()))
    results.append(("FusionBlock Structure", test_fusion_blocks()))
    results.append(("FusionBlock Standalone", test_fusion_block_standalone()))
    results.append(("ResidualBlock Activity", test_residual_block_activity()))
    results.append(("get_layer_activity Method", test_get_layer_activity_method()))
    results.append(("Activity Tracking", test_activity_tracking()))
    
    print("\n" + "="*80)
    print("Test Results Summary")
    print("="*80 + "\n")
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ All tests PASSED!")
    else:
        print("❌ Some tests FAILED!")
    print("="*80 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
