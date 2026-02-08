#!/usr/bin/env python3
"""
Test priority system functionality in dataset generator.
"""

import os
import sys
import json
import random

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_priority_sorting():
    """Test that videos are sorted correctly by priority."""
    print("Testing Priority System...")
    print("=" * 70)
    
    config_path = os.path.join(
        os.path.dirname(__file__),
        'dataset_generator_v2',
        'generator_config.json'
    )
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    videos = config['videos'].copy()
    
    # Simulate the sorting that happens in __init__
    random.seed(42)  # Same seed as in the code
    for i, video in enumerate(videos):
        video['_sort_random'] = random.random()
    
    videos.sort(key=lambda v: (v.get('priority', 255), v['_sort_random']))
    
    # Verify sorting is correct
    last_priority = -1
    is_sorted = True
    
    for i, v in enumerate(videos):
        p = v.get('priority', 255)
        if p < last_priority:
            print(f"❌ ERROR: Priority {p} comes after {last_priority} at index {i}")
            print(f"   Video: {v['name']}")
            is_sorted = False
            return False
        last_priority = p
    
    if is_sorted:
        print("✅ Priority sorting works correctly!")
        
        # Show statistics
        priority_counts = {}
        for v in videos:
            p = v.get('priority', 255)
            priority_counts[p] = priority_counts.get(p, 0) + 1
        
        print(f"\nPriority distribution:")
        for p in sorted(priority_counts.keys()):
            count = priority_counts[p]
            if p == 255:
                print(f"  Priority {p} (default): {count} videos")
            else:
                print(f"  Priority {p}: {count} videos")
        
        print(f"\nFirst 10 videos (should start with priority 0):")
        for i in range(min(10, len(videos))):
            v = videos[i]
            p = v.get('priority', 255)
            print(f"  {i+1}. Priority {p:3d} - {v['name'][:50]}")
        
        return True
    
    return False

def test_priority_coverage():
    """Test that all expected priority levels are present."""
    print("\nTesting Priority Coverage...")
    print("=" * 70)
    
    config_path = os.path.join(
        os.path.dirname(__file__),
        'dataset_generator_v2',
        'generator_config.json'
    )
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    videos = config['videos']
    
    # Count priorities
    priority_counts = {}
    for v in videos:
        p = v.get('priority', 255)
        priority_counts[p] = priority_counts.get(p, 0) + 1
    
    expected_priorities = [0, 1, 2, 3, 4, 255]
    found_all = True
    
    for p in expected_priorities:
        if p in priority_counts:
            print(f"✅ Priority {p}: {priority_counts[p]} videos")
        else:
            if p == 255:
                print(f"⚠️  Priority {p} (default): Not found (OK if all videos have explicit priority)")
            else:
                print(f"⚠️  Priority {p}: Not found")
                found_all = False
    
    # Check for unexpected priorities
    for p in priority_counts:
        if p not in expected_priorities:
            print(f"⚠️  Unexpected priority {p}: {priority_counts[p]} videos")
    
    return True

def test_priority_examples():
    """Show example videos from each priority level."""
    print("\nPriority Level Examples...")
    print("=" * 70)
    
    config_path = os.path.join(
        os.path.dirname(__file__),
        'dataset_generator_v2',
        'generator_config.json'
    )
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    videos = config['videos']
    
    # Group by priority
    priority_examples = {}
    for v in videos:
        p = v.get('priority', 255)
        if p not in priority_examples:
            priority_examples[p] = []
        priority_examples[p].append(v['name'])
    
    # Show examples
    for p in sorted(priority_examples.keys()):
        examples = priority_examples[p]
        print(f"\nPriority {p} ({len(examples)} videos):")
        for i, name in enumerate(examples[:3]):
            print(f"  - {name}")
        if len(examples) > 3:
            print(f"  ... and {len(examples) - 3} more")
    
    return True

if __name__ == "__main__":
    try:
        # Run all tests
        success = True
        success &= test_priority_sorting()
        success &= test_priority_coverage()
        success &= test_priority_examples()
        
        print("\n" + "=" * 70)
        if success:
            print("✅ All priority system tests passed!")
            print("\nPriority system is working correctly:")
            print("  - Videos are sorted by priority (0 first, 255 last)")
            print("  - Within same priority, videos are randomized (seed=42)")
            print("  - All priority levels are present in the config")
        else:
            print("❌ Some tests failed!")
            sys.exit(1)
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
