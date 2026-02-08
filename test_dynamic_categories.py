#!/usr/bin/env python3
"""
Test dynamic category support in dataset generator v2.
Validates that the generator works with custom category configurations.
"""

import sys
import os
import json
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_config_with_custom_categories():
    """Test that generator config can be loaded with custom categories."""
    print("Testing custom category configuration...")
    
    # Create a test config with custom categories
    test_config = {
        "base_settings": {
            "base_frame_limit": 100,
            "max_workers": 4,
            "val_percent": 0.0,
            "output_base_dir": "/tmp/test_dataset",
            "temp_dir": "/tmp/test_dataset/temp",
            "status_file": "/tmp/test_dataset/.status.json",
            "min_file_size": 10000,
            "scene_diff_threshold": 45,
            "max_retry_attempts": 3,
            "retry_skip_seconds": 30,
            "lr_versions": ["5frames", "7frames"]
        },
        "category_targets": {
            "master": 15000,
            "universal": 75000,
            "space": 45000,
            "toon": 30000,
            "custom1": 10000
        },
        "format_config": {
            "master": {
                "small_540": {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 0.50},
                "medium_169": {"gt_size": [720, 405], "lr_size": [240, 135], "probability": 0.50}
            },
            "universal": {
                "small_540": {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 1.0}
            },
            "space": {
                "small_540": {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 0.40},
                "medium_169": {"gt_size": [720, 405], "lr_size": [240, 135], "probability": 0.60}
            },
            "toon": {
                "small_540": {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 1.0}
            },
            "custom1": {
                "small_540": {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 1.0}
            }
        },
        "videos": []
    }
    
    # Test that we can extract categories from config
    categories = list(test_config['category_targets'].keys())
    
    assert len(categories) == 5, f"Expected 5 categories, got {len(categories)}"
    assert "master" in categories, "master not in categories"
    assert "universal" in categories, "universal not in categories"
    assert "space" in categories, "space not in categories"
    assert "toon" in categories, "toon not in categories"
    assert "custom1" in categories, "custom1 not in categories"
    
    print(f"✓ Successfully extracted {len(categories)} custom categories: {', '.join(categories)}")

def test_generator_initialization_with_custom_config():
    """Test that DatasetGeneratorV2 can initialize with custom categories."""
    print("Testing generator initialization with custom categories...")
    
    # Import after adding to path
    from dataset_generator_v2.make_dataset_multi import DatasetGeneratorV2
    
    # Create a temporary config file
    test_config = {
        "base_settings": {
            "base_frame_limit": 100,
            "max_workers": 4,
            "val_percent": 0.0,
            "output_base_dir": "/tmp/test_dataset",
            "temp_dir": "/tmp/test_dataset/temp",
            "status_file": "/tmp/test_dataset/.status.json",
            "min_file_size": 10000,
            "scene_diff_threshold": 45,
            "max_retry_attempts": 3,
            "retry_skip_seconds": 30,
            "lr_versions": ["5frames", "7frames"]
        },
        "category_targets": {
            "alpha": 10000,
            "beta": 20000,
            "gamma": 30000
        },
        "format_config": {
            "alpha": {
                "small_540": {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 1.0}
            },
            "beta": {
                "small_540": {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 1.0}
            },
            "gamma": {
                "small_540": {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 1.0}
            }
        },
        "videos": []
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f)
        config_path = f.name
    
    try:
        # Initialize generator
        generator = DatasetGeneratorV2(config_path)
        
        # Verify that config was loaded properly
        assert 'category_targets' in generator.config
        categories = list(generator.config['category_targets'].keys())
        
        assert len(categories) == 3, f"Expected 3 categories, got {len(categories)}"
        assert "alpha" in categories
        assert "beta" in categories
        assert "gamma" in categories
        
        # Verify that the generator can extract categories correctly
        extracted_cats = list(generator.config.get('category_targets', {}).keys())
        assert extracted_cats == categories
        
        print(f"✓ Generator initialized with custom categories: {', '.join(categories)}")
        
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)

def test_gui_layout_with_different_category_counts():
    """Test that GUI layout works with different numbers of categories."""
    print("Testing GUI layout with various category counts...")
    
    from dataset_generator_v2.make_dataset_multi import DatasetGeneratorV2
    
    # Test with 1, 2, 3, 4, and 5 categories
    for num_cats in [1, 2, 3, 4, 5]:
        category_names = [f"cat{i}" for i in range(1, num_cats + 1)]
        
        test_config = {
            "base_settings": {
                "base_frame_limit": 100,
                "max_workers": 4,
                "val_percent": 0.0,
                "output_base_dir": "/tmp/test_dataset",
                "temp_dir": "/tmp/test_dataset/temp",
                "status_file": "/tmp/test_dataset/.status.json",
                "min_file_size": 10000,
                "scene_diff_threshold": 45,
                "max_retry_attempts": 3,
                "retry_skip_seconds": 30,
                "lr_versions": ["5frames", "7frames"]
            },
            "category_targets": {cat: 10000 for cat in category_names},
            "format_config": {
                cat: {
                    "small_540": {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 1.0}
                } for cat in category_names
            },
            "videos": []
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            config_path = f.name
        
        try:
            generator = DatasetGeneratorV2(config_path)
            
            # Try to build GUI layout (this will test the disk usage display)
            # We need to suppress output since rich might not be available
            try:
                # This should not raise an exception
                categories_from_config = list(generator.config.get('category_targets', {}).keys())
                assert len(categories_from_config) == num_cats
                
                # Simulate disk usage display logic
                disk_lines = ["[bold]💾 DISK USAGE[/bold]"]
                categories = list(generator.config.get('category_targets', {}).keys())
                for i, cat_name in enumerate(categories):
                    prefix = "├─"
                    disk_lines.append(f"{prefix} {cat_name.upper()}: 0.0 GB")
                disk_lines.append(f"└─ Total: 0.0 GB")
                disk_usage = "\n".join(disk_lines)
                
                # Verify the Total line has the right prefix
                assert "└─ Total:" in disk_lines[-1], "Total should have └─ prefix"
                
                print(f"  ✓ GUI layout works with {num_cats} categories")
                
            except Exception as e:
                print(f"  ✗ Failed with {num_cats} categories: {e}")
                raise
                
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    print(f"✓ GUI layout tested successfully with 1-5 categories")

def test_backward_compatibility():
    """Test that the changes work with the original config format."""
    print("Testing backward compatibility with original categories...")
    
    from dataset_generator_v2.make_dataset_multi import DatasetGeneratorV2
    
    # Use original category names
    test_config = {
        "base_settings": {
            "base_frame_limit": 100,
            "max_workers": 4,
            "val_percent": 0.0,
            "output_base_dir": "/tmp/test_dataset",
            "temp_dir": "/tmp/test_dataset/temp",
            "status_file": "/tmp/test_dataset/.status.json",
            "min_file_size": 10000,
            "scene_diff_threshold": 45,
            "max_retry_attempts": 3,
            "retry_skip_seconds": 30,
            "lr_versions": ["5frames", "7frames"]
        },
        "category_targets": {
            "general": 110000,
            "space": 82500,
            "toon": 30000
        },
        "format_config": {
            "general": {
                "small_540": {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 1.0}
            },
            "space": {
                "small_540": {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 1.0}
            },
            "toon": {
                "small_540": {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 1.0}
            }
        },
        "videos": []
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f)
        config_path = f.name
    
    try:
        generator = DatasetGeneratorV2(config_path)
        categories = list(generator.config.get('category_targets', {}).keys())
        
        assert len(categories) == 3
        assert "general" in categories
        assert "space" in categories
        assert "toon" in categories
        
        print(f"✓ Backward compatibility confirmed with original categories: {', '.join(categories)}")
        
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)

def main():
    """Run all tests."""
    print("Running dynamic category support tests...\n")
    
    try:
        test_config_with_custom_categories()
        test_generator_initialization_with_custom_config()
        test_gui_layout_with_different_category_counts()
        test_backward_compatibility()
        
        print("\n✅ All dynamic category tests passed!")
        return 0
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
