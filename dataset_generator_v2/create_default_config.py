#!/usr/bin/env python3
"""
Create a new empty default config for the dataset generator.

The new config is based on the generator_config.json template and contains:
  - base_settings    (only keys read by make_dataset_v2_uhd.py, freely editable)
  - category_targets (master / universal / space / toon, freely editable)
  - format_config    (patch formats per category, freely editable)
  - ffmpeg_timeout   / ffprobe_timeout (top-level timeouts for make_dataset_v2_uhd.py)
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

# Only the keys actually read by make_dataset_v2_uhd.py
_DEFAULT_BASE_SETTINGS = {
    "output_base_dir": "/mnt/data/training/datasetNeu",
    "temp_dir": "/mnt/data/training/datasetNeu/temp",
    "status_file": "/mnt/data/training/datasetNeu/.generator_status.json",
    "lr_versions": ["5frames", "7frames"],
    "min_detail_threshold": 80.0
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
    Build a default config dict containing only the keys read by make_dataset_v2_uhd.py.

    If *template_path* points to an existing JSON file (e.g. generator_config.json),
    its base_settings, category_targets, format_config, ffmpeg_timeout and
    ffprobe_timeout values are used instead of the built-in defaults.
    """
    base_settings    = dict(_DEFAULT_BASE_SETTINGS)
    category_targets = dict(_DEFAULT_CATEGORY_TARGETS)
    format_config    = {k: dict(v) for k, v in _DEFAULT_FORMAT_CONFIG.items()}
    ffmpeg_timeout   = 120
    ffprobe_timeout  = 60

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
            if 'ffmpeg_timeout' in tmpl:
                ffmpeg_timeout = tmpl['ffmpeg_timeout']
            if 'ffprobe_timeout' in tmpl:
                ffprobe_timeout = tmpl['ffprobe_timeout']
            print(f"✓ Used template values from {template_path}")
        except Exception as exc:
            print(f"⚠️  Could not read template {template_path}: {exc} – using built-in defaults")

    return {
        "base_settings":    base_settings,
        "category_targets": category_targets,
        "format_config":    format_config,
        "ffmpeg_timeout":   ffmpeg_timeout,
        "ffprobe_timeout":  ffprobe_timeout,
        "source_dirs": [],
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
