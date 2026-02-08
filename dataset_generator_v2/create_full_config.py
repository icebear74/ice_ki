#!/usr/bin/env python3
"""
Helper script for adding priority field to existing generator_config.json
This is mainly for backwards compatibility - new configs created by scan_videos.py
will automatically include priority=255 for all videos.

Usage: python create_full_config.py [config_file]
"""

import json
import sys

def add_priority_to_videos(videos):
    """Add default priority=255 to videos that don't have it."""
    updated_count = 0
    for video in videos:
        if 'priority' not in video:
            video['priority'] = 255
            updated_count += 1
    return updated_count

def main():
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'generator_config.json'
    
    try:
        # Load config
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Add priority field
        updated = add_priority_to_videos(config.get('videos', []))
        
        # Save updated config
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Updated {updated} videos with priority=255")
        print(f"✓ Config saved to {config_file}")
        print("\n💡 You can now manually edit priorities in the JSON file")
        print("   Lower numbers (0) will be processed first, higher numbers (255) last")
        
    except FileNotFoundError:
        print(f"Error: {config_file} not found")
        print("Run scan_videos.py first to create a config file")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
