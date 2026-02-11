#!/usr/bin/env python3
"""
Patch generator_config.json to remove weights from categories.

Converts:
  "categories": {"master": 0.25, "universal": 0.75}
To:
  "categories": ["master", "universal"]
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def patch_config(config_path: Path):
    """Patch config file to remove weights."""
    
    # Load config
    print(f"Loading {config_path}...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    if 'videos' not in config:
        print("❌ No videos found in config")
        return
    
    # Create backup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = config_path.parent / f"{config_path.stem}_backup_{timestamp}.json"
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"✓ Backup saved to {backup_path}")
    
    # Convert categories
    converted_count = 0
    skipped_count = 0
    
    for video in config['videos']:
        if 'categories' not in video:
            continue
        
        cats = video['categories']
        
        if isinstance(cats, dict):
            # Convert dict to list - weights are removed
            category_list = list(cats.keys())
            video['categories'] = category_list
            converted_count += 1
        elif isinstance(cats, list):
            # Already in list format
            skipped_count += 1
        else:
            # Unknown format
            print(f"⚠️  Unknown category format for {video['name']}: {type(cats)}")
    
    print(f"\n📊 Conversion Summary:")
    print(f"  Total videos:        {len(config['videos'])}")
    print(f"  Converted (dict→list): {converted_count}")
    print(f"  Already list format:   {skipped_count}")
    print(f"  No categories:         {len(config['videos']) - converted_count - skipped_count}")
    
    if converted_count > 0:
        # Save patched config
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Config patched successfully!")
        print(f"✅ Weights removed from {converted_count} videos")
        print(f"\nOLD format: \"categories\": {{\"master\": 0.25, \"universal\": 0.75}}")
        print(f"NEW format: \"categories\": [\"master\", \"universal\"]")
    else:
        print(f"\n✓ No conversion needed - all categories already in list format")


def main():
    config_path = Path(__file__).parent / 'generator_config.json'
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    print("="*70)
    print("  PATCH GENERATOR CONFIG - REMOVE CATEGORY WEIGHTS")
    print("="*70)
    print("\nThis will convert category dicts to simple lists:")
    print("  OLD: {\"master\": 0.25, \"universal\": 0.75}")
    print("  NEW: [\"master\", \"universal\"]")
    print("\nA backup will be created automatically.")
    print("="*70)
    
    confirm = input("\nProceed? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("❌ Aborted")
        sys.exit(0)
    
    patch_config(config_path)


if __name__ == "__main__":
    main()
