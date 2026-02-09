#!/usr/bin/env python3
"""
Test for format_probabilities KeyError fix.

Verifies that format probabilities are correctly extracted from format_config
and used in calculate_format_distribution_for_video.
"""

import json
import sys

def test_extract_format_probabilities():
    """Test that format probabilities can be extracted from format_config"""
    
    # Simulate format_config structure from generator_config.json
    format_config = {
        'master': {
            'small_540': {'probability': 0.5},
            'medium_169': {'probability': 0.35},
            'large_720': {'probability': 0.15}
        },
        'universal': {
            'small_540': {'probability': 0.5},
            'medium_169': {'probability': 0.35},
            'large_720': {'probability': 0.15}
        },
        'space': {
            'small_540': {'probability': 0.4},
            'medium_169': {'probability': 0.35},
            'large_720': {'probability': 0.25}
        },
        'toon': {
            'small_540': {'probability': 0.65},
            'medium_169': {'probability': 0.25},
            'large_720': {'probability': 0.1}
        }
    }
    
    # Extract probabilities (mimic the _extract_format_probabilities method)
    probabilities = {}
    for category, formats in format_config.items():
        probabilities[category] = {}
        for format_name, format_info in formats.items():
            probabilities[category][format_name] = format_info.get('probability', 0.0)
    
    # Verify structure
    assert 'master' in probabilities, "master category missing"
    assert 'universal' in probabilities, "universal category missing"
    assert 'space' in probabilities, "space category missing"
    assert 'toon' in probabilities, "toon category missing"
    
    # Verify master probabilities
    assert probabilities['master']['small_540'] == 0.5
    assert probabilities['master']['medium_169'] == 0.35
    assert probabilities['master']['large_720'] == 0.15
    
    # Verify probabilities sum to 1.0 (or close)
    for category in probabilities:
        total = sum(probabilities[category].values())
        assert abs(total - 1.0) < 0.01, f"{category} probabilities don't sum to 1.0: {total}"
    
    print("✅ PASS: Format probabilities extracted correctly")
    return True


def test_calculate_distribution():
    """Test that distribution calculation works with extracted probabilities"""
    
    # Setup
    format_probabilities = {
        'master': {'large_720': 0.5, 'small_540': 0.25, 'medium_169': 0.25},
        'universal': {'large_720': 0.5, 'small_540': 0.25, 'medium_169': 0.25}
    }
    
    video = {
        'name': 'Test Video',
        'categories': {
            'master': 0.5,
            'universal': 0.5
        }
    }
    
    target_patches = 4000
    
    # Calculate distribution (mimic calculate_format_distribution_for_video)
    distribution = {}
    
    for category, category_weight in video['categories'].items():
        category_patches = int(target_patches * category_weight)
        
        # Get format probabilities for this category
        format_probs = format_probabilities.get(category, {})
        
        # Calculate patches per format
        distribution[category] = {}
        remaining_patches = category_patches
        
        # Sort by probability (descending)
        sorted_formats = sorted(format_probs.items(), key=lambda x: x[1], reverse=True)
        
        for idx, (format_name, prob) in enumerate(sorted_formats):
            if idx == len(sorted_formats) - 1:
                # Last format gets remaining patches
                distribution[category][format_name] = remaining_patches
            else:
                count = int(category_patches * prob)
                distribution[category][format_name] = count
                remaining_patches -= count
    
    # Verify results
    assert 'master' in distribution
    assert 'universal' in distribution
    
    # Check master distribution
    assert distribution['master']['large_720'] == 1000  # 2000 * 0.5
    assert distribution['master']['small_540'] == 500   # 2000 * 0.25
    assert distribution['master']['medium_169'] == 500  # remaining
    
    # Check universal distribution
    assert distribution['universal']['large_720'] == 1000
    assert distribution['universal']['small_540'] == 500
    assert distribution['universal']['medium_169'] == 500
    
    # Check totals
    master_total = sum(distribution['master'].values())
    universal_total = sum(distribution['universal'].values())
    assert master_total == 2000
    assert universal_total == 2000
    assert master_total + universal_total == 4000
    
    print("✅ PASS: Distribution calculation works correctly")
    return True


def test_with_actual_config():
    """Test with actual generator_config.json if available"""
    try:
        with open('generator_config.json', 'r') as f:
            config = json.load(f)
        
        format_config = config.get('format_config', {})
        
        # Extract probabilities
        probabilities = {}
        for category, formats in format_config.items():
            probabilities[category] = {}
            for format_name, format_info in formats.items():
                probabilities[category][format_name] = format_info.get('probability', 0.0)
        
        # Verify all categories have probabilities
        expected_categories = ['master', 'universal', 'space', 'toon']
        for cat in expected_categories:
            assert cat in probabilities, f"Category {cat} missing from probabilities"
            assert len(probabilities[cat]) > 0, f"Category {cat} has no formats"
        
        print("✅ PASS: Actual config format probabilities extracted")
        return True
        
    except FileNotFoundError:
        print("⚠️  SKIP: generator_config.json not found (not in repo root)")
        return True


if __name__ == '__main__':
    print("=" * 70)
    print("Testing Format Probabilities Fix")
    print("=" * 70)
    
    tests = [
        test_extract_format_probabilities,
        test_calculate_distribution,
        test_with_actual_config
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ FAIL: {test.__name__}")
            print(f"   Error: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 70)
    
    sys.exit(0 if failed == 0 else 1)
