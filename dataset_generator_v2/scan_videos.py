#!/usr/bin/env python3
"""
Scan actual video files and generate config.
Usage: python scan_videos.py /path/to/your/videos
"""

import os
import sys
import json
import re

def categorize_video(filename, path):
    """Auto-categorize based on filename/path."""
    name_lower = filename.lower()
    path_lower = path.lower()
    
    categories = {}
    
    # SPACE indicators
    space_keywords = [
        'star trek', 'star wars', 'trek', 'wars', 'space', 'alien', 
        'interstellar', 'gravity', 'martian', 'dune', 'avatar',
        'guardians', 'foundation', 'expanse', 'mandalorian', 
        'prometheus', 'covenant', 'arrival', 'contact', 'moon'
    ]
    
    # TOON indicators  
    toon_keywords = [
        'shrek', 'toy story', 'pixar', 'disney', 'dreamworks',
        'frozen', 'moana', 'finding', 'incredibles', 'cars',
        'kung fu panda', 'madagascar', 'despicable', 'minions',
        'animation', 'animated', 'cartoon'
    ]
    
    # GENERAL is default
    has_space = any(k in name_lower or k in path_lower for k in space_keywords)
    has_toon = any(k in name_lower or k in path_lower for k in toon_keywords)
    
    if has_space and has_toon:
        # Multi-category
        categories = {"general": 0.3, "space": 0.4, "toon": 0.3}
    elif has_space:
        categories = {"space": 1.0}
    elif has_toon:
        categories = {"toon": 1.0}
    else:
        categories = {"general": 1.0}
    
    return categories

def scan_videos(root_path):
    """Scan directory for video files."""
    videos = []
    
    print(f"Scanning {root_path} for videos...")
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith(('.mkv', '.mp4', '.avi', '.mov')):
                full_path = os.path.join(dirpath, filename)
                
                # Clean name
                name = os.path.splitext(filename)[0]
                name = re.sub(r'\{edition-[^}]+\}', '', name)  # Remove edition tags
                name = name.strip()
                
                # Auto-categorize
                categories = categorize_video(filename, dirpath)
                
                videos.append({
                    "name": name,
                    "path": full_path,
                    "categories": categories
                })
    
    return videos

def main():
    if len(sys.argv) < 2:
        print("Usage: python scan_videos.py /path/to/videos")
        print("\nOder gib den Pfad ein:")
        video_path = input("Video Pfad: ").strip()
        if not video_path:
            print("Kein Pfad angegeben!")
            return
    else:
        video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"FEHLER: Pfad {video_path} existiert nicht!")
        return
    
    # Scan videos
    videos = scan_videos(video_path)
    
    if not videos:
        print(f"Keine Videos gefunden in {video_path}")
        return
    
    print(f"\n✓ {len(videos)} Videos gefunden!")
    
    # Create config
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
            "retry_skip_seconds": 60
        },
        "videos": videos
    }
    
    # Save
    output_file = "generator_config_REAL.json"
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Config gespeichert: {output_file}")
    
    # Show statistics
    cat_counts = {'general': 0, 'space': 0, 'toon': 0}
    for v in videos:
        for cat in v['categories']:
            cat_counts[cat] += 1
    
    print(f"\nKategorie Verteilung:")
    for cat, count in cat_counts.items():
        print(f"  • {cat.upper()}: {count}")
    
    # Show sample
    print(f"\nErste 5 Videos:")
    for v in videos[:5]:
        cats = ', '.join([f'{k}({v})' for k, v in v['categories'].items()])
        print(f"  • {v['name'][:50]}: {cats}")
    
    print(f"\n💡 Prüfe die Datei {output_file} und passe die Kategorien an!")
    print(f"   Dann: mv {output_file} generator_config.json")

if __name__ == "__main__":
    main()
