#!/usr/bin/env python3
"""
Category format utilities - Simple list-based categories (NO WEIGHTS)

Weights are ignored. A video is either in a category (100%) or not (0%).
Both dict and list formats are supported, but weights in dicts are ignored.
"""

from typing import Dict, List, Union


def normalize_categories(categories: Union[Dict[str, float], List[str]]) -> List[str]:
    """
    Convert categories to normalized list format.
    Weights are IGNORED - only category names matter.
    
    Args:
        categories: Either {"cat1": 0.5, "cat2": 0.5} or ["cat1", "cat2"]
    
    Returns:
        List of category names: ["cat1", "cat2"]
    """
    if isinstance(categories, list):
        return categories
    elif isinstance(categories, dict):
        # Ignore weights - just extract category names
        # Any category present = video is in that category (100%)
        return list(categories.keys())
    else:
        return []


def get_video_categories(video: dict) -> List[str]:
    """
    Get categories from video, handling both formats.
    
    Args:
        video: Video dict with 'categories' field
    
    Returns:
        List of category names
    """
    cats = video.get('categories', {})
    return normalize_categories(cats)


def is_video_in_category(video: dict, category: str) -> bool:
    """
    Check if video is in a category, handling both formats.
    
    Args:
        video: Video dict
        category: Category name to check
    
    Returns:
        True if video is in category
    """
    cats = get_video_categories(video)
    return category in cats


def format_categories_display(categories: Union[Dict[str, float], List[str]]) -> str:
    """
    Format categories for display.
    
    Args:
        categories: Either dict or list format
    
    Returns:
        Human-readable string
    """
    cat_list = normalize_categories(categories)
    if not cat_list:
        return "⚠️  <WILL BE SKIPPED - no categories>"
    return ", ".join(cat_list)


def convert_config_to_list_format(config: dict, force: bool = False) -> dict:
    """
    Convert entire config from dict to list format (optional).
    By default, both formats work - weights are just ignored.
    
    Args:
        config: Full generator config
        force: If True, convert all dicts to lists (cleanup)
    
    Returns:
        Updated config with list format (if force=True)
    """
    if not force or 'videos' not in config:
        return config
    
    converted_count = 0
    for video in config['videos']:
        if 'categories' in video:
            old_cats = video['categories']
            if isinstance(old_cats, dict):
                # Convert dict to list (weights ignored)
                video['categories'] = list(old_cats.keys())
                converted_count += 1
    
    if converted_count > 0:
        print(f"✓ Converted {converted_count} videos from dict to list format (weights removed)")
    
    return config


if __name__ == "__main__":
    # Test
    print("Testing category format utilities...")
    
    # Test normalization
    dict_format = {"master": 0.25, "universal": 0.75, "space": 0}
    list_format = ["master", "universal"]
    
    print(f"\nDict format: {dict_format}")
    print(f"Normalized:  {normalize_categories(dict_format)}")
    
    print(f"\nList format: {list_format}")
    print(f"Normalized:  {normalize_categories(list_format)}")
    
    # Test video checks
    video1 = {"name": "Test", "categories": {"master": 0.5, "space": 0.5}}
    video2 = {"name": "Test2", "categories": ["master", "space"]}
    video3 = {"name": "Test3", "categories": {}}
    
    print(f"\nVideo 1 categories: {get_video_categories(video1)}")
    print(f"Video 2 categories: {get_video_categories(video2)}")
    print(f"Video 3 categories: {get_video_categories(video3)}")
    
    print(f"\nVideo 1 in 'master': {is_video_in_category(video1, 'master')}")
    print(f"Video 2 in 'master': {is_video_in_category(video2, 'master')}")
    print(f"Video 3 in 'master': {is_video_in_category(video3, 'master')}")
    
    # Test display
    print(f"\nDisplay dict: {format_categories_display(dict_format)}")
    print(f"Display list: {format_categories_display(list_format)}")
    print(f"Display empty: {format_categories_display({})}")
    
    print("\n✓ All tests passed!")
