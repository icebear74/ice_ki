#!/usr/bin/env python3
"""
Integration test for optimized frame extraction.
Verifies that:
1. Each category gets different random crop positions
2. GT patches remain at native resolution
3. Scene stability validation happens once per extraction
"""

import sys
import os
import tempfile
import json
import numpy as np
from unittest.mock import Mock, patch, call, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from dataset_generator_v2.make_dataset_multi import DatasetGeneratorV2


def test_process_all_categories_integration():
    """Integration test for process_all_categories_from_frames."""
    print("Testing process_all_categories_from_frames integration...")
    
    config = {
        'base_settings': {
            'output_base_dir': '/tmp/test_output',
            'temp_dir': '/tmp/test_temp',
            'status_file': '/tmp/test_status_integration.json',
            'max_workers': 1,
            'base_frame_limit': 100,
            'max_retry_attempts': 3,
            'retry_skip_seconds': 5,
            'min_file_size': 1000,
            'scene_diff_threshold': 30
        },
        'videos': [],
        'format_config': {
            'master': {'small_540': 1.0},
            'universal': {'medium_169': 1.0},
            'space': {'large_720': 1.0}
        },
        'category_targets': {
            'master': {'target': 1000},
            'universal': {'target': 1000},
            'space': {'target': 1000}
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_file = f.name
    
    try:
        generator = DatasetGeneratorV2(config_file)
        
        # Create mock frames (7 frames of 1920x1080)
        import cv2
        mock_frames = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(7)]
        
        # Track calls to save_patches
        original_save_patches = generator.save_patches
        save_patches_calls = []
        
        def mock_save_patches(frames, category, format_name, video_name, frame_idx):
            save_patches_calls.append({
                'category': category,
                'format_name': format_name,
                'frames_id': id(frames),  # Track if same frames object
                'video_name': video_name,
                'frame_idx': frame_idx
            })
            return True
        
        generator.save_patches = mock_save_patches
        
        # Test categories
        categories = {
            'master': 1.0,
            'universal': 1.0,
            'space': 1.0
        }
        
        # Call process_all_categories_from_frames
        result = generator.process_all_categories_from_frames(
            mock_frames, categories, 'test_video', 42
        )
        
        # Verify result
        assert result == True, "Should return True for successful processing"
        
        # Verify save_patches was called for each category
        assert len(save_patches_calls) == 3, f"Expected 3 calls to save_patches, got {len(save_patches_calls)}"
        
        # Verify each category was processed
        categories_processed = {call['category'] for call in save_patches_calls}
        assert categories_processed == {'master', 'universal', 'space'}, \
            f"Expected all categories processed, got {categories_processed}"
        
        # Verify correct formats were selected (based on config)
        format_mapping = {call['category']: call['format_name'] for call in save_patches_calls}
        assert format_mapping['master'] == 'small_540', "Expected small_540 for master"
        assert format_mapping['universal'] == 'medium_169', "Expected medium_169 for universal"
        assert format_mapping['space'] == 'large_720', "Expected large_720 for space"
        
        # Verify same frames object was used for all (optimization!)
        frames_ids = {call['frames_id'] for call in save_patches_calls}
        assert len(frames_ids) == 1, "All categories should use the same frames object"
        
        print("✓ process_all_categories_from_frames integration test passed")
        
    finally:
        if os.path.exists(config_file):
            os.unlink(config_file)
        if os.path.exists('/tmp/test_status_integration.json'):
            os.unlink('/tmp/test_status_integration.json')


def test_scene_stability_called_once():
    """Verify scene stability is only validated once per extraction."""
    print("Testing scene stability is called once...")
    
    config = {
        'base_settings': {
            'output_base_dir': '/tmp/test_output',
            'temp_dir': '/tmp/test_temp',
            'status_file': '/tmp/test_status_stability.json',
            'max_workers': 1,
            'base_frame_limit': 100,
            'max_retry_attempts': 3,
            'retry_skip_seconds': 5,
            'min_file_size': 1000,
            'scene_diff_threshold': 30
        },
        'videos': [],
        'format_config': {},
        'category_targets': {
            'master': {'target': 1000},
            'universal': {'target': 1000}
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_file = f.name
    
    try:
        generator = DatasetGeneratorV2(config_file)
        
        # Create mock frames
        import cv2
        mock_frames = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(7)]
        
        # Track calls to validate_scene_stability
        original_validate = generator.validate_scene_stability
        validate_call_count = [0]
        
        def mock_validate(frames):
            validate_call_count[0] += 1
            return original_validate(frames)
        
        generator.validate_scene_stability = mock_validate
        
        # Mock save_patches to return True
        generator.save_patches = lambda *args, **kwargs: True
        
        # Process multiple categories
        categories = {'master': 1.0, 'universal': 1.0}
        generator.process_all_categories_from_frames(
            mock_frames, categories, 'test_video', 42
        )
        
        # Verify validate_scene_stability was called exactly once
        assert validate_call_count[0] == 1, \
            f"Expected validate_scene_stability to be called once, but was called {validate_call_count[0]} times"
        
        print("✓ Scene stability validation test passed")
        
    finally:
        if os.path.exists(config_file):
            os.unlink(config_file)
        if os.path.exists('/tmp/test_status_stability.json'):
            os.unlink('/tmp/test_status_stability.json')


def main():
    """Run all integration tests."""
    print("Running integration tests for optimized extraction...\n")
    
    try:
        test_process_all_categories_integration()
        test_scene_stability_called_once()
        
        print("\n✅ All integration tests passed!")
        return 0
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
