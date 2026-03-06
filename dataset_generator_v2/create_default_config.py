#!/usr/bin/env python3
"""
Create a new empty default config for the dataset generator.

The new config matches the V2 structure produced by video_manager.py:
  - root_path         (output directory – adjust before use)
  - source_dirs       (empty list – add via video_manager.py option 13)
  - videos            (empty list – populated by Rescan, option 16)
  - category_patches  (master / universal / space / toon patch counts)
  - output_patches    (540 / 720 / 720_169 enabled by default)
  - processing        (n_frames, scene settings)
  - quality           (blur / sharpness thresholds)
  - workers           (parallel extraction threads)
  - ffmpeg_timeout / ffprobe_timeout

Usage:
  python create_default_config.py                   # prompts for filename
  python create_default_config.py my_config.json    # writes directly
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

_DEFAULT_CONFIG = {
    "root_path": "/mnt/data/training/datasetNeu",
    "source_dirs": [],
    "videos": [],
    "category_patches": {
        "master":    100000,
        "universal":  50000,
        "space":      50000,
        "toon":       40000
    },
    "output_patches": {
        "540":     {"enabled": True,  "gt_size": [540, 540], "scale": 3, "weight": 35},
        "720":     {"enabled": True,  "gt_size": [720, 720], "scale": 3, "weight": 40},
        "720_169": {"enabled": True,  "gt_size": [720, 405], "scale": 3, "weight": 25}
    },
    "processing": {
        "n_frames":         7,
        "min_scene_length": 21,
        "scene_threshold":  30.0,
        "stride":           3,
        "scale":            3
    },
    "quality": {
        "blur_threshold": 80.0,
        "min_sharpness":  30.0,
        "jpeg_quality":   95,
        "lr_degrade_prob":       0.6,
        "lr_dark_boost":         True,
        "lr_dark_threshold":     60.0,
        "lr_dark_boost_prob":    0.8,
        "lr_jpeg_quality_range": [55, 75],
        "lr_noise_sigma":        [0.5, 2.5],
        "lr_blur_sigma":         [0.2, 0.7]
    },
    "workers":         8,
    "batch_size":      4,
    "random_seed":     None,
    "validation":      {"enabled": False},
    "ffmpeg_timeout":  120,
    "ffprobe_timeout": 60
}


def build_default_config(template_path: str = None) -> dict:
    """
    Build a default V2 config dict.

    If *template_path* points to an existing V2 JSON file, its values are used
    instead of the built-in defaults where available.
    """
    config = {k: (dict(v) if isinstance(v, dict) else v)
              for k, v in _DEFAULT_CONFIG.items()}

    if template_path and os.path.exists(template_path):
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                tmpl = json.load(f)
            for key in ('root_path', 'category_patches', 'output_patches',
                        'processing', 'quality', 'workers', 'batch_size',
                        'ffmpeg_timeout', 'ffprobe_timeout'):
                if key in tmpl:
                    config[key] = tmpl[key]
            print(f"✓ Used template values from {template_path}")
        except Exception as exc:
            print(f"⚠️  Could not read template {template_path}: {exc} – using built-in defaults")

    return config


def create_default_config(output_path: str, template_path: str = None) -> None:
    """Write a fresh default V2 config to *output_path*."""
    config = build_default_config(template_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"✅ Created: {output_path}")


def main():
    script_dir    = Path(__file__).parent
    template_path = str(script_dir / 'generator_config_v2.json')

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
        print("  1. Open the file and adjust root_path / category_patches as needed")
        print("  2. python video_manager.py  →  option 13 to add source directories")
        print("  3. option 16 to scan and build the video list")
        print("  4. options 5 / 6 / 7 to assign categories to videos")
    except Exception as exc:
        print(f"❌ Error: {exc}")
        sys.exit(1)


if __name__ == '__main__':
    main()
