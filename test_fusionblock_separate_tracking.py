#!/usr/bin/env python3
"""
Test FusionBlock separate layer tracking (3x3 and 1x1)
Verifies that each FusionBlock tracks both internal layers separately
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_fusionblock_separate_tracking():
    """Test that FusionBlock tracks 3x3 and 1x1 activities separately"""
    print("\n" + "="*80)
    print("Testing FusionBlock Separate Layer Tracking")
    print("="*80 + "\n")
    
    try:
        import torch
        from vsr_plusplus_NEU.core.model_7frame import VSRBidirectional_7frames_3x, FusionBlock
        
        # Test FusionBlock directly
        print("1. Testing FusionBlock attributes...")
        fusion = FusionBlock(in_feats=144, out_feats=72)
        
        # Check attributes exist
        assert hasattr(fusion, 'last_activity_3x3'), "FusionBlock should have last_activity_3x3"
        assert hasattr(fusion, 'last_activity_1x1'), "FusionBlock should have last_activity_1x1"
        assert not hasattr(fusion, 'last_activity'), "FusionBlock should NOT have old last_activity"
        
        # Check initial values
        assert fusion.last_activity_3x3 == 0.0, "Initial 3x3 activity should be 0.0"
        assert fusion.last_activity_1x1 == 0.0, "Initial 1x1 activity should be 0.0"
        
        print("   ✅ FusionBlock has separate activity attributes")
        
        # Test forward pass
        print("\n2. Testing FusionBlock forward pass...")
        dummy_input = torch.randn(1, 144, 32, 32)
        with torch.no_grad():
            output = fusion(dummy_input)
        
        # Check activities were updated
        assert fusion.last_activity_3x3 > 0, "3x3 activity should be updated after forward"
        assert fusion.last_activity_1x1 > 0, "1x1 activity should be updated after forward"
        
        print(f"   3x3 activity: {fusion.last_activity_3x3:.6f}")
        print(f"   1x1 activity: {fusion.last_activity_1x1:.6f}")
        print("   ✅ Both activities tracked separately")
        
        # Test full model
        print("\n3. Testing full 7-frame model...")
        model = VSRBidirectional_7frames_3x(n_feats=72, n_blocks=26)
        model.eval()
        
        # Run forward pass
        dummy_input = torch.randn(1, 7, 3, 64, 64)
        with torch.no_grad():
            output = model(dummy_input)
        
        print("   ✅ Model forward pass successful")
        
        # Test get_layer_activity
        print("\n4. Testing get_layer_activity() structure...")
        activities = model.get_layer_activity()
        
        # Check structure
        assert 'backward_fuse' in activities, "Should have backward_fuse"
        assert 'forward_fuse' in activities, "Should have forward_fuse"
        assert 'fusion' in activities, "Should have fusion"
        
        # Check that fusion activities are lists with 2 elements
        assert isinstance(activities['backward_fuse'], list), "backward_fuse should be a list"
        assert len(activities['backward_fuse']) == 2, "backward_fuse should have 2 elements"
        
        assert isinstance(activities['forward_fuse'], list), "forward_fuse should be a list"
        assert len(activities['forward_fuse']) == 2, "forward_fuse should have 2 elements"
        
        assert isinstance(activities['fusion'], list), "fusion should be a list"
        assert len(activities['fusion']) == 2, "fusion should have 2 elements"
        
        print(f"   backward_fuse: [{activities['backward_fuse'][0]:.6f}, {activities['backward_fuse'][1]:.6f}]")
        print(f"   forward_fuse: [{activities['forward_fuse'][0]:.6f}, {activities['forward_fuse'][1]:.6f}]")
        print(f"   fusion: [{activities['fusion'][0]:.6f}, {activities['fusion'][1]:.6f}]")
        print("   ✅ All fusion blocks return 2 activities (3x3, 1x1)")
        
        # Count total layers
        print("\n5. Counting total layers...")
        total_layers = 0
        total_layers += len(activities['backward_trunk'])  # 13
        total_layers += len(activities['backward_fuse'])   # 2
        total_layers += len(activities['forward_trunk'])   # 13
        total_layers += len(activities['forward_fuse'])    # 2
        total_layers += len(activities['fusion'])          # 2
        
        print(f"   Backward trunk: {len(activities['backward_trunk'])} layers")
        print(f"   Backward fuse: {len(activities['backward_fuse'])} layers (3x3 + 1x1)")
        print(f"   Forward trunk: {len(activities['forward_trunk'])} layers")
        print(f"   Forward fuse: {len(activities['forward_fuse'])} layers (3x3 + 1x1)")
        print(f"   Fusion: {len(activities['fusion'])} layers (3x3 + 1x1)")
        print(f"   Total: {total_layers} layers")
        
        assert total_layers == 32, f"Total should be 32 layers, got {total_layers}"
        print("   ✅ Correct total of 32 layers")
        
        # Verify all activities are positive
        print("\n6. Verifying all activities are positive...")
        assert all(act > 0 for act in activities['backward_trunk']), "All backward_trunk should be > 0"
        assert all(act > 0 for act in activities['backward_fuse']), "All backward_fuse should be > 0"
        assert all(act > 0 for act in activities['forward_trunk']), "All forward_trunk should be > 0"
        assert all(act > 0 for act in activities['forward_fuse']), "All forward_fuse should be > 0"
        assert all(act > 0 for act in activities['fusion']), "All fusion should be > 0"
        print("   ✅ All activities are positive")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run test"""
    print("\n" + "="*80)
    print("FusionBlock Separate Layer Tracking Test")
    print("="*80)
    
    success = test_fusionblock_separate_tracking()
    
    print("\n" + "="*80)
    if success:
        print("✅ ALL TESTS PASSED - 32 layers with separate tracking")
    else:
        print("❌ TESTS FAILED")
    print("="*80 + "\n")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
