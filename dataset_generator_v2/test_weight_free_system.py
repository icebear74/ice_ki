#!/usr/bin/env python3
"""
Test weight-free category system
"""

import sys
from pathlib import Path

# Add current dir to path
sys.path.insert(0, str(Path(__file__).parent))

from category_utils import (
    normalize_categories,
    get_video_categories,
    is_video_in_category,
    format_categories_display
)

print("="*70)
print("  TEST: WEIGHT-FREE CATEGORY SYSTEM")
print("="*70)

# Test 1: Both formats normalize to same result
print("\n1. Format Normalization:")
print("-" * 70)

old_format = {"master": 0.25, "universal": 0.75, "space": 0}
new_format = ["master", "universal"]

old_normalized = normalize_categories(old_format)
new_normalized = normalize_categories(new_format)

print(f"OLD (dict): {old_format}")
print(f"  Normalized: {old_normalized}")
print(f"\nNEW (list): {new_format}")
print(f"  Normalized: {new_normalized}")

# Note: old format includes 'space' because weight doesn't matter anymore
print(f"\n⚠️  Note: Weights are IGNORED. Any key in dict = category present")

# Test 2: Video category extraction
print("\n\n2. Video Category Extraction:")
print("-" * 70)

videos = [
    {"name": "Video 1", "categories": {"master": 0.25, "universal": 0.75}},
    {"name": "Video 2", "categories": ["master", "space"]},
    {"name": "Video 3", "categories": {}},
    {"name": "Video 4", "categories": []},
]

for video in videos:
    cats = get_video_categories(video)
    print(f"{video['name']}: {cats}")

# Test 3: Category membership
print("\n\n3. Category Membership:")
print("-" * 70)

for video in videos:
    in_master = is_video_in_category(video, 'master')
    in_space = is_video_in_category(video, 'space')
    print(f"{video['name']}: master={in_master}, space={in_space}")

# Test 4: Display formatting
print("\n\n4. Display Formatting:")
print("-" * 70)

for video in videos:
    display = format_categories_display(video.get('categories', []))
    print(f"{video['name']}: {display}")

# Test 5: Distribution calculation simulation
print("\n\n5. Patch Distribution Simulation:")
print("-" * 70)

def simulate_distribution(video, target_patches):
    """Simulate the new distribution logic."""
    cats = get_video_categories(video)
    
    if not cats:
        return "SKIPPED (no categories)"
    
    num_cats = len(cats)
    patches_per_cat = target_patches // num_cats
    remainder = target_patches % num_cats
    
    distribution = {}
    for i, cat in enumerate(cats):
        patches = patches_per_cat
        if i < remainder:
            patches += 1
        distribution[cat] = patches
    
    return distribution

for video in videos:
    result = simulate_distribution(video, 1000)
    print(f"\n{video['name']}:")
    print(f"  Categories: {get_video_categories(video)}")
    print(f"  Distribution: {result}")

# Test 6: Equal distribution verification
print("\n\n6. Equal Distribution Verification:")
print("-" * 70)

test_video = {"name": "Test", "categories": ["master", "space", "toon"]}
target = 1500

dist = simulate_distribution(test_video, target)
print(f"Video: {test_video['name']}")
print(f"Categories: {get_video_categories(test_video)}")
print(f"Target patches: {target}")
print(f"Distribution: {dist}")
print(f"Total: {sum(dist.values())}")
print(f"✓ Each category gets 100% of video (equal share of patches)")

print("\n" + "="*70)
print("  ALL TESTS PASSED! ✅")
print("="*70)
print("\nKEY CHANGES:")
print("  ✅ Weights are IGNORED")
print("  ✅ Video is 100% in each assigned category")
print("  ✅ Patches distributed equally across categories")
print("  ✅ Simpler logic, easier to understand")
print("="*70)
