#!/usr/bin/env python3
"""
Create a new empty default config for the dataset generator.

The new config is based on the generator_config.json template and contains:
  - base_settings    (same defaults as generator_config.json, freely editable)
  - category_targets (master / universal / space / toon, freely editable)
  - format_config    (patch formats per category, freely editable)
  - source_dirs      (empty list – add directories via video_manager.py option 13)
  - videos           (empty list – populated by Rescan, option 16)

Usage:
  python create_default_config.py                   # prompts for filename
  python create_default_config.py my_config.json    # writes directly
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Built-in defaults (mirrors generator_config.json)
_DEFAULT_BASE_SETTINGS = {
    "base_frame_limit": 3000,
    "max_workers": 4,
    "val_percent": 0.0,
    "output_base_dir": "/mnt/data/training/datasetNeu",
    "temp_dir": "/mnt/data/training/datasetNeu/temp",
    "status_file": "/mnt/data/training/datasetNeu/.generator_status.json",
    "min_file_size": 10000,
    "scene_diff_threshold": 45,
    "max_retry_attempts": 10,
    "retry_skip_seconds": 60,
    "lr_versions": ["5frames", "7frames"]
}

_DEFAULT_CATEGORY_TARGETS = {
    "master":    100000,
    "universal":  50000,
    "space":      50000,
    "toon":       40000
}

_DEFAULT_FORMAT_CONFIG = {
    "master": {
        "small_540":  {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 0.6},
        "medium_169": {"gt_size": [405, 720], "lr_size": [135, 240], "probability": 0.2},
        "large_720":  {"gt_size": [720, 720], "lr_size": [240, 240], "probability": 0.2}
    },
    "universal": {
        "small_540":  {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 0.6},
        "medium_169": {"gt_size": [405, 720], "lr_size": [135, 240], "probability": 0.2},
        "large_720":  {"gt_size": [720, 720], "lr_size": [240, 240], "probability": 0.2}
    },
    "space": {
        "small_540":  {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 0.6},
        "medium_169": {"gt_size": [405, 720], "lr_size": [135, 240], "probability": 0.2},
        "large_720":  {"gt_size": [720, 720], "lr_size": [240, 240], "probability": 0.2}
    },
    "toon": {
        "small_540":  {"gt_size": [540, 540], "lr_size": [180, 180], "probability": 0.6},
        "medium_169": {"gt_size": [405, 720], "lr_size": [135, 240], "probability": 0.2},
        "large_720":  {"gt_size": [720, 720], "lr_size": [240, 240], "probability": 0.2}
    }
}


def build_default_config(template_path: str = None) -> dict:
    """
    Build a default config dict.

    If *template_path* points to an existing JSON file (e.g. generator_config.json),
    its base_settings, category_targets and format_config values are used instead of
    the built-in defaults.
    """
    base_settings    = dict(_DEFAULT_BASE_SETTINGS)
    category_targets = dict(_DEFAULT_CATEGORY_TARGETS)
    format_config    = {k: dict(v) for k, v in _DEFAULT_FORMAT_CONFIG.items()}

    if template_path and os.path.exists(template_path):
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                tmpl = json.load(f)
            if 'base_settings' in tmpl:
                base_settings = tmpl['base_settings']
            if 'category_targets' in tmpl:
                category_targets = tmpl['category_targets']
            if 'format_config' in tmpl:
                format_config = tmpl['format_config']
            print(f"✓ Used template values from {template_path}")
        except Exception as exc:
            print(f"⚠️  Could not read template {template_path}: {exc} – using built-in defaults")

    return {
        "_comment_usage": "=== DEFAULT CONFIG – created by create_default_config.py ===",
        "_comment_workflow": (
            "1. Edit base_settings / category_targets / format_config as needed.  "
            "2. Add source directories via video_manager.py option 13.  "
            "3. Rescan (option 16) to build the videos list.  "
            "4. Assign categories with options 5 / 6 / 7."
        ),
        "base_settings":    base_settings,
        "category_targets": category_targets,
        "format_config":    format_config,
        "_comment_source_dirs": (
            "Directories to scan for video files.  "
            "Independent of categories.  "
            "Add/edit/remove via video_manager.py options 13-15."
        ),
        "source_dirs": [],
        "_comment_videos": (
            "Video list populated by 'Rescan' (video_manager.py option 16).  "
            "Assign categories with options 5 / 6 / 7."
        ),
        "videos": []
    }


def create_default_config(output_path: str, template_path: str = None) -> None:
    """Write a fresh default config to *output_path*."""
    config = build_default_config(template_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"✅ Created: {output_path}")


def main():
    script_dir    = Path(__file__).parent
    template_path = str(script_dir / 'generator_config.json')

    # Determine output path
    if len(sys.argv) >= 2:
        output_path = sys.argv[1]
    else:
        ts           = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_name = f'generator_config_new_{ts}.json'
        val = input(f"Output filename [{default_name}]: ").strip()
        output_path = str(script_dir / (val or default_name))

    if os.path.exists(output_path):
        overwrite = input(f"⚠️  '{output_path}' already exists. Overwrite? (yes/no): ").strip().lower()
        if overwrite != 'yes':
            print("Cancelled.")
            return

    try:
        create_default_config(output_path, template_path=template_path)
        print("\nNext steps:")
        print("  1. Open the file and adjust base_settings / category_targets as needed")
        print("  2. python video_manager.py  →  option 13 to add source directories")
        print("  3. option 16 to scan and build the video list")
        print("  4. options 5 / 6 / 7 to assign categories to videos")
    except Exception as exc:
        print(f"❌ Error: {exc}")
        sys.exit(1)


if __name__ == '__main__':
    main()
