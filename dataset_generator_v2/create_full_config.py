#!/usr/bin/env python3
"""
Create complete generator_config.json with 4-category model structure.

Categories:
- master: ALL videos (baseline, never used in production)
- universal: Real-action films (non-space, non-toon)
- space: Sci-Fi/Space content
- toon: Animation/CGI-heavy content

Usage:
    python create_full_config.py
"""

import json
import os
import re

# Category keywords for automatic classification
SPACE_KEYWORDS = [
    'star trek', 'star wars', 'dune', 'interstellar', 'alien', 'avatar',
    'oblivion', 'passengers', 'elysium', 'gattaca', 'arrival', 'marsianer',
    'ad astra', 'event horizon', 'independence day', 'welten', 'galaxy',
    'cosmic', 'moonfall', 'space', 'unheimliche begegnung', 'e.t.', 'creator',
    'ahsoka', 'obi-wan', 'obiwan', 'section 31'
]

TOON_KEYWORDS = [
    'shrek', 'minions', 'ice age', 'dragons', 'drachenzähmen', 'emoji',
    'zoomania', 'grinch', 'mario', 'minecraft', 'trolls', 'pinocchio',
    'arielle', 'asterix', 'jim knopf', 'wonka', 'pets', 'dolittle',
    'unverbesserlich', 'despicable'
]

# Multi-category special cases
MULTI_CATEGORY = {
    'avatar': {'master': 0.2, 'universal': 0.2, 'space': 0.3, 'toon': 0.3},
    'ready player one': {'master': 0.2, 'universal': 0.3, 'space': 0.2, 'toon': 0.3},
    'barbie': {'master': 0.2, 'universal': 0.5, 'toon': 0.3},
    'jumanji': {'master': 0.2, 'universal': 0.5, 'toon': 0.3},
}

def categorize_video(name, path):
    """
    Automatically categorize a video based on its name.
    
    Returns:
        dict: Category weights (e.g., {'master': 0.2, 'space': 0.8})
    """
    name_lower = name.lower()
    path_lower = path.lower()
    
    # Check multi-category special cases first
    for key, cats in MULTI_CATEGORY.items():
        if key in name_lower or key in path_lower:
            return cats
    
    # Check for space
    is_space = any(kw in name_lower or kw in path_lower for kw in SPACE_KEYWORDS)
    
    # Check for toon
    is_toon = any(kw in name_lower or kw in path_lower for kw in TOON_KEYWORDS)
    
    # Assign categories
    if is_space and is_toon:
        # Rare: both space and toon (e.g., some animated sci-fi)
        return {'master': 0.2, 'space': 0.4, 'toon': 0.4}
    elif is_space:
        return {'master': 0.2, 'space': 0.8}
    elif is_toon:
        return {'master': 0.2, 'toon': 0.8}
    else:
        # Universal (default)
        return {'master': 0.25, 'universal': 0.75}

def extract_name_from_path(path):
    """Extract a clean video name from the path."""
    # Get directory name before the file
    parts = path.split('/')
    
    # For series: Use series name + episode info
    if '/SerieUHD/' in path:
        series_name = parts[-3]  # Series directory
        episode_file = parts[-1].replace('.mkv', '')
        
        # Clean up episode name
        episode_clean = re.sub(r'_t\d+$', '', episode_file)
        return f"{series_name} - {episode_clean}"
    
    # For movies: Use directory name
    else:
        movie_dir = parts[-2]
        # Remove year info and clean up
        clean = re.sub(r'\s*\(\d{4}\)\s*', '', movie_dir)
        return clean

def main():
    """Generate complete generator_config.json from uhd_liste.txt"""
    
    # Read uhd_liste.txt
    uhd_list_path = 'uhd_liste.txt'
    if not os.path.exists(uhd_list_path):
        print(f"❌ Error: {uhd_list_path} not found!")
        print(f"   Make sure you're in the dataset_generator_v2/ directory")
        return
    
    print("📖 Reading uhd_liste.txt...")
    
    videos = []
    with open(uhd_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse line format: "N| /path/to/video.mkv"
            if '|' in line:
                parts = line.split('|', 1)
                if len(parts) == 2:
                    path = parts[1].strip()
                    
                    # Extract name
                    name = extract_name_from_path(path)
                    
                    # Categorize
                    categories = categorize_video(name, path)
                    
                    videos.append({
                        'name': name,
                        'path': path,
                        'categories': categories
                    })
    
    print(f"✅ Found {len(videos)} videos")
    
    # Count categories
    category_counts = {'master': 0, 'universal': 0, 'space': 0, 'toon': 0}
    for v in videos:
        for cat in v['categories'].keys():
            if cat != 'master':  # Don't count master (it has all videos)
                category_counts[cat] += 1
    
    category_counts['master'] = len(videos)  # Master has ALL videos
    
    print(f"\n📊 Category Distribution:")
    print(f"   Master:    {category_counts['master']} videos (ALL)")
    print(f"   Universal: {category_counts['universal']} videos")
    print(f"   Space:     {category_counts['space']} videos")
    print(f"   Toon:      {category_counts['toon']} videos")
    
    # Show samples
    print(f"\n📝 Sample Categorizations:")
    
    sample_universal = next((v for v in videos if 'universal' in v['categories']), None)
    if sample_universal:
        print(f"   Universal: {sample_universal['name']}")
        print(f"              {sample_universal['categories']}")
    
    sample_space = next((v for v in videos if 'space' in v['categories']), None)
    if sample_space:
        print(f"   Space:     {sample_space['name']}")
        print(f"              {sample_space['categories']}")
    
    sample_toon = next((v for v in videos if 'toon' in v['categories']), None)
    if sample_toon:
        print(f"   Toon:      {sample_toon['name']}")
        print(f"              {sample_toon['categories']}")
    
    # Create config structure
    config = {
        "base_settings": {
            "base_frame_limit": 3000,
            "max_workers": 12,
            "val_percent": 0.0,
            "output_base_dir": "/mnt/data/training/dataset",
            "temp_dir": "/mnt/data/training/dataset/temp",
            "status_file": "/mnt/data/training/dataset/.generator_status.json",
            "min_file_size": 10000,
            "scene_diff_threshold": 45,
            "max_retry_attempts": 10,
            "retry_skip_seconds": 60,
            "lr_versions": ["5frames", "7frames"]
        },
        
        "category_targets": {
            "master": 300000,
            "universal": 350000,
            "space": 160000,
            "toon": 90000
        },
        
        "format_config": {
            "master": {
                "small_540": {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 0.50},
                "medium_169": {"gt_size": [720, 405], "lr_size": [240, 135], "probability": 0.35},
                "large_720": {"gt_size": [720, 720], "lr_size": [240, 240], "probability": 0.15}
            },
            "universal": {
                "small_540": {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 0.50},
                "medium_169": {"gt_size": [720, 405], "lr_size": [240, 135], "probability": 0.35},
                "large_720": {"gt_size": [720, 720], "lr_size": [240, 240], "probability": 0.15}
            },
            "space": {
                "small_540": {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 0.40},
                "medium_169": {"gt_size": [720, 405], "lr_size": [240, 135], "probability": 0.35},
                "large_720": {"gt_size": [720, 720], "lr_size": [240, 240], "probability": 0.25}
            },
            "toon": {
                "small_540": {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 0.65},
                "medium_169": {"gt_size": [720, 405], "lr_size": [240, 135], "probability": 0.35}
            }
        },
        
        "videos": videos
    }
    
    # Write config
    output_path = 'generator_config.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Generated: {output_path}")
    print(f"   Total size: {len(json.dumps(config))} bytes")
    print(f"   Total videos: {len(videos)}")
    print(f"\n🚀 Next steps:")
    print(f"   1. Review the config: cat generator_config.json | less")
    print(f"   2. Start generation: python make_dataset_multi.py")
    print(f"   3. Monitor progress: python monitor_generator.py")

if __name__ == '__main__':
    main()