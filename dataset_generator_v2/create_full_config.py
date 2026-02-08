#!/usr/bin/env python3
"""
Create complete generator_config.json with 4-category model structure.
"""

import json
import os
import re
import random

# Priority videos - processed FIRST for early training start
PRIORITY_KEYWORDS = [
    'planet earth', 'frozen planet', 'blue planet', 'our planet',
    'life', 'africa', 'dynasties', 'seven worlds', 'green planet',
    'attenborough', 'perfect planet', 'wild',
]

# Category keywords
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

MULTI_CATEGORY = {
    'avatar': {'master': 0.2, 'universal': 0.2, 'space': 0.3, 'toon': 0.3},
    'ready player one': {'master': 0.2, 'universal': 0.3, 'space': 0.2, 'toon': 0.3},
    'barbie': {'master': 0.2, 'universal': 0.5, 'toon': 0.3},
    'jumanji': {'master': 0.2, 'universal': 0.5, 'toon': 0.3},
}

def get_video_priority(name, path):
    """
    Determine video priority for processing order.
    
    Returns:
        0: High priority (nature documentaries - process FIRST)
        1: Normal priority (random order)
    """
    name_lower = name.lower()
    path_lower = path.lower()
    
    # Check if it's a priority video
    for keyword in PRIORITY_KEYWORDS:
        if keyword in name_lower or keyword in path_lower:
            return 0  # Process FIRST
    
    return 1  # Normal priority

def categorize_video(name, path):
    name_lower = name.lower()
    path_lower = path.lower()
    
    for key, cats in MULTI_CATEGORY.items():
        if key in name_lower or key in path_lower:
            return cats
    
    is_space = any(kw in name_lower or kw in path_lower for kw in SPACE_KEYWORDS)
    is_toon = any(kw in name_lower or kw in path_lower for kw in TOON_KEYWORDS)
    
    if is_space and is_toon:
        return {'master': 0.2, 'space': 0.4, 'toon': 0.4}
    elif is_space:
        return {'master': 0.2, 'space': 0.8}
    elif is_toon:
        return {'master': 0.2, 'toon': 0.8}
    else:
        return {'master': 0.25, 'universal': 0.75}

def extract_name_from_path(path):
    parts = path.split('/')
    
    if '/SerieUHD/' in path:
        series_name = parts[-3]
        episode_file = parts[-1].replace('.mkv', '')
        episode_clean = re.sub(r'_t\d+$', '', episode_file)
        return f"{series_name} - {episode_clean}"
    else:
        movie_dir = parts[-2]
        clean = re.sub(r'\s*\(\d{4}\)\s*', '', movie_dir)
        return clean

def main():
    uhd_list_path = 'uhd_liste.txt'
    
    print("📖 Reading uhd_liste.txt...")
    
    videos = []
    with open(uhd_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line:
                continue
            
            path = line
            
            if path.endswith('.mkv'):
                name = extract_name_from_path(path)
                categories = categorize_video(name, path)
                priority = get_video_priority(name, path)
                
                videos.append({
                    'name': name,
                    'path': path,
                    'categories': categories,
                    'priority': priority
                })
    
    print(f"✅ Found {len(videos)} videos\n")
    
    # Sort videos: Priority first, then randomize
    random.seed(42)
    videos.sort(key=lambda v: (v['priority'], random.random()))
    
    priority_count = sum(1 for v in videos if v['priority'] == 0)
    print(f"🌍 Priority videos (nature docs): {priority_count}")
    print(f"🎬 Regular videos: {len(videos) - priority_count}\n")
    
    category_counts = {'master': len(videos), 'universal': 0, 'space': 0, 'toon': 0}
    for v in videos:
        for cat in v['categories'].keys():
            if cat != 'master':
                category_counts[cat] += 1
    
    print("📊 Category Distribution:")
    print(f"   Master:    {category_counts['master']} videos (ALL)")
    print(f"   Universal: {category_counts['universal']} videos")
    print(f"   Space:     {category_counts['space']} videos")
    print(f"   Toon:      {category_counts['toon']} videos\n")
    
    print("🎯 First 5 videos to process:")
    for i, v in enumerate(videos[:5]):
        priority_marker = "🌍" if v['priority'] == 0 else "🎬"
        print(f"   {i+1}. {priority_marker} {v['name']}")
    
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
    
    with open('generator_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Generated: generator_config.json")
    print(f"   Total videos: {len(videos)}")
    print(f"\n🚀 Processing order:")
    print(f"   1. 🌍 Priority videos first (nature documentaries)")
    print(f"   2. 🎬 Remaining videos in random order")
    print(f"\n💡 You can start training after ~30-60 minutes!")

if __name__ == '__main__':
    main()