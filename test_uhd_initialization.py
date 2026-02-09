#!/usr/bin/env python3
"""
Test for make_dataset_v2_uhd.py initialization fix
Verifies that the AttributeError is fixed
"""

import sys
import os
import tempfile
import json

# Add dataset_generator_v2 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dataset_generator_v2'))

def test_initialization_fix():
    """Test that DatasetGeneratorV2UHD initializes without AttributeError"""
    print("=" * 60)
    print("TEST: DatasetGeneratorV2UHD Initialization")
    print("=" * 60)
    
    # Create a minimal test config
    test_config = {
        "base_settings": {
            "base_frame_limit": 3000,
            "max_workers": 1,
            "val_percent": 0.0,
            "output_base_dir": "/tmp/test_dataset",
            "temp_dir": "/tmp/test_dataset/temp",
            "status_file": "/tmp/test_dataset/.status.json",
            "min_file_size": 10000,
            "scene_diff_threshold": 45,
            "lr_versions": ["7frames"]
        },
        "category_targets": {
            "master": 100
        },
        "format_config": {
            "master": {
                "small_540": {
                    "gt_size": [540, 540],
                    "lr_size": [180, 180],
                    "probability": 1.0
                }
            }
        },
        "videos": [
            {
                "name": "Test Video",
                "path": "/tmp/test.mkv",
                "categories": {
                    "master": 1.0
                }
            }
        ]
    }
    
    # Write test config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f)
        config_path = f.name
    
    try:
        # Try to initialize - this should NOT raise AttributeError
        from make_dataset_v2_uhd import DatasetGeneratorV2UHD
        
        print("Creating DatasetGeneratorV2UHD instance...")
        generator = DatasetGeneratorV2UHD(config_path)
        
        # Verify attributes are set correctly
        assert hasattr(generator, 'base_dir'), "Missing base_dir attribute"
        assert generator.base_dir == "/tmp/test_dataset", f"Wrong base_dir: {generator.base_dir}"
        
        assert hasattr(generator, 'logger'), "Missing logger attribute"
        assert generator.logger is not None, "Logger is None"
        
        assert hasattr(generator, 'temp_dir'), "Missing temp_dir attribute"
        assert generator.temp_dir == "/tmp/test_dataset/temp", f"Wrong temp_dir: {generator.temp_dir}"
        
        assert hasattr(generator, 'status_file'), "Missing status_file attribute"
        assert generator.status_file == "/tmp/test_dataset/.status.json", f"Wrong status_file: {generator.status_file}"
        
        assert len(generator.videos) == 1, f"Wrong number of videos: {len(generator.videos)}"
        
        print("✅ SUCCESS: All attributes initialized correctly")
        print(f"   base_dir: {generator.base_dir}")
        print(f"   temp_dir: {generator.temp_dir}")
        print(f"   status_file: {generator.status_file}")
        print(f"   videos: {len(generator.videos)}")
        print(f"   logger: {generator.logger}")
        
        return True
        
    except AttributeError as e:
        print(f"❌ FAILED: AttributeError occurred: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    except Exception as e:
        print(f"❌ FAILED: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        if os.path.exists(config_path):
            os.unlink(config_path)


def test_with_original_config():
    """Test with the original generator_config.json if it exists"""
    print("\n" + "=" * 60)
    print("TEST: With Original Config (if available)")
    print("=" * 60)
    
    config_path = "generator_config.json"
    if not os.path.exists(config_path):
        print("⚠️  SKIPPED: generator_config.json not found")
        return True
    
    try:
        from make_dataset_v2_uhd import DatasetGeneratorV2UHD
        
        print("Loading original config...")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"  Videos in config: {len(config.get('videos', []))}")
        print(f"  Categories: {list(config.get('category_targets', {}).keys())}")
        
        print("Creating DatasetGeneratorV2UHD instance...")
        generator = DatasetGeneratorV2UHD(config_path)
        
        print("✅ SUCCESS: Original config loaded successfully")
        print(f"   Videos loaded: {len(generator.videos)}")
        print(f"   base_dir: {generator.base_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  DatasetGeneratorV2UHD Initialization Test".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    results = []
    
    # Test 1: Basic initialization
    results.append(("Basic Initialization", test_initialization_fix()))
    
    # Test 2: Original config (if available)
    results.append(("Original Config", test_with_original_config()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    sys.exit(0 if passed == total else 1)
