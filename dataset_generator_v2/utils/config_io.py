#!/usr/bin/env python3
"""
Shared config IO and validation layer for dataset_generator_v2.

Provides load/save/validate for templates.json and generator_config_v2.active.json.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

VALID_SOURCE_MODES = {"resize", "crop"}


# ── Load / Save ───────────────────────────────────────────────────────────────

def load_templates(path: str) -> dict:
    """Load templates.json from disk."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_templates(templates: dict, path: str) -> None:
    """Save templates dict to disk."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(templates, f, indent=2, ensure_ascii=False)


def load_active_config(path: str) -> dict:
    """Load generator_config_v2.active.json from disk."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_active_config(config: dict, path: str) -> None:
    """Save active config dict to disk (with optional backup)."""
    backup_path = path + '.backup'
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            old = f.read()
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(old)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


# ── Defaults ──────────────────────────────────────────────────────────────────

def create_default_templates() -> dict:
    """Return a default templates dict (format_templates + degradation_templates)."""
    return {
        "_format": "templates_v1",
        "_description": "Reusable format and degradation templates for dataset generation",
        "format_templates": {
            "960x540_169": {
                "gt_size": [960, 540],
                "scale": 3,
                "aspect_ratio": "16:9",
                "description": "960x540 16:9 Landscape – standard resize target"
            },
            "1152x648_169": {
                "gt_size": [1152, 648],
                "scale": 3,
                "aspect_ratio": "16:9",
                "description": "1152x648 16:9 Landscape – higher detail resize/crop target"
            },
            "720x540_43": {
                "gt_size": [720, 540],
                "scale": 3,
                "aspect_ratio": "4:3",
                "description": "720x540 4:3 – classic TV ratio"
            },
            "720x720_11": {
                "gt_size": [720, 720],
                "scale": 3,
                "aspect_ratio": "1:1",
                "description": "720x720 Square – detail crop"
            }
        },
        "degradation_templates": {
            "classic_sitcom_sd": {
                "description": "Classic SD sitcom / series look: VHS-era sharpness loss, analog noise, warm color cast",
                "blur": {"sigma_range": [0.4, 1.2], "prob": 0.85},
                "compression": {"jpeg_quality_range": [55, 78], "prob": 0.90, "ringing_strength": 0.3},
                "noise": {"sigma_range": [2.0, 8.0], "prob": 0.80, "color_noise_prob": 0.4},
                "chroma": {"saturation_range": [0.85, 1.10], "chroma_bleed_prob": 0.35, "chroma_bleed_strength": 0.2},
                "color": {"contrast_range": [0.90, 1.05], "brightness_range": [-0.02, 0.04], "black_lift": 0.03, "gamma_range": [0.95, 1.10]}
            },
            "classic_scifi_sd": {
                "description": "Classic sci-fi SD: slightly over-sharpened, film grain, cool desaturated look",
                "blur": {"sigma_range": [0.2, 0.8], "prob": 0.70},
                "compression": {"jpeg_quality_range": [62, 82], "prob": 0.80, "ringing_strength": 0.4},
                "noise": {"sigma_range": [1.5, 6.0], "prob": 0.75, "color_noise_prob": 0.25},
                "chroma": {"saturation_range": [0.75, 0.95], "chroma_bleed_prob": 0.20, "chroma_bleed_strength": 0.1},
                "color": {"contrast_range": [0.95, 1.10], "brightness_range": [-0.03, 0.02], "black_lift": 0.01, "gamma_range": [0.90, 1.05]}
            },
            "dvd_film_balanced": {
                "description": "Balanced DVD/film look: moderate compression, slight softening, neutral color",
                "blur": {"sigma_range": [0.3, 0.9], "prob": 0.75},
                "compression": {"jpeg_quality_range": [65, 85], "prob": 0.85, "ringing_strength": 0.2},
                "noise": {"sigma_range": [1.0, 4.5], "prob": 0.65, "color_noise_prob": 0.20},
                "chroma": {"saturation_range": [0.90, 1.05], "chroma_bleed_prob": 0.15, "chroma_bleed_strength": 0.08},
                "color": {"contrast_range": [0.95, 1.05], "brightness_range": [-0.02, 0.03], "black_lift": 0.01, "gamma_range": [0.97, 1.03]}
            },
            "toon_sd": {
                "description": "Animated / cartoon SD: flat color loss, strong compression artifacts, cel-like sharpness",
                "blur": {"sigma_range": [0.1, 0.6], "prob": 0.60},
                "compression": {"jpeg_quality_range": [50, 72], "prob": 0.95, "ringing_strength": 0.5},
                "noise": {"sigma_range": [0.5, 3.0], "prob": 0.50, "color_noise_prob": 0.15},
                "chroma": {"saturation_range": [0.80, 1.15], "chroma_bleed_prob": 0.40, "chroma_bleed_strength": 0.25},
                "color": {"contrast_range": [0.92, 1.08], "brightness_range": [-0.01, 0.05], "black_lift": 0.02, "gamma_range": [0.95, 1.08]}
            }
        }
    }


def create_default_active_config() -> dict:
    """Return a minimal default active config."""
    return {
        "_format": "generator_config_v2",
        "_version": "2.0",
        "_description": "Active dataset project config. Edit via video_manager.py",
        "root_path": "/mnt/data/training/datasetNeu",
        "source_dirs": [],
        "videos": [],
        "categories": {
            "master": {
                "target_total": 100000,
                "formats": [
                    {
                        "template": "960x540_169",
                        "weight": 60,
                        "source_mode": "resize",
                        "degradation_mix": {
                            "dvd_film_balanced": 50,
                            "classic_sitcom_sd": 50
                        }
                    }
                ]
            }
        },
        "processing": {
            "n_frames": 7,
            "min_scene_length": 21,
            "scene_threshold": 30.0,
            "stride": 3,
            "scale": 3
        },
        "quality": {
            "blur_threshold": 80.0,
            "min_sharpness": 30.0,
            "jpeg_quality": 95
        },
        "workers": 8,
        "batch_size": 4,
        "random_seed": None,
        "ffmpeg_timeout": 120,
        "ffprobe_timeout": 60
    }


def ensure_templates_file(path: str) -> dict:
    """Create templates.json if missing, then load and return it."""
    if not os.path.exists(path):
        defaults = create_default_templates()
        save_templates(defaults, path)
        print(f"✓ Created default templates file: {path}")
    return load_templates(path)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_templates(templates: dict) -> List[str]:
    """
    Validate templates dict structure.

    Returns a list of error strings (empty = OK).
    """
    errors: List[str] = []

    if not isinstance(templates, dict):
        errors.append("templates must be a JSON object")
        return errors

    fmt_tmpls = templates.get("format_templates")
    if not isinstance(fmt_tmpls, dict):
        errors.append("templates.format_templates must be a JSON object")
    else:
        for name, spec in fmt_tmpls.items():
            if not isinstance(spec, dict):
                errors.append(f"format_templates.{name}: must be a JSON object")
                continue
            gt = spec.get("gt_size")
            if not (isinstance(gt, (list, tuple)) and len(gt) == 2 and all(isinstance(x, int) and x > 0 for x in gt)):
                errors.append(f"format_templates.{name}.gt_size: must be [width, height] positive ints")
            scale = spec.get("scale")
            if not (isinstance(scale, int) and scale > 0):
                errors.append(f"format_templates.{name}.scale: must be a positive int")

    deg_tmpls = templates.get("degradation_templates")
    if not isinstance(deg_tmpls, dict):
        errors.append("templates.degradation_templates must be a JSON object")

    return errors


def validate_active_config(config: dict, templates: dict) -> List[str]:
    """
    Validate active config against templates.

    Checks cross-file references (format templates, degradation templates).
    Returns a list of error strings (empty = OK).
    """
    errors: List[str] = []

    if not isinstance(config, dict):
        errors.append("config must be a JSON object")
        return errors

    fmt_tmpls = templates.get("format_templates", {}) if isinstance(templates, dict) else {}
    deg_tmpls = templates.get("degradation_templates", {}) if isinstance(templates, dict) else {}

    categories = config.get("categories")
    if not isinstance(categories, dict):
        errors.append("config.categories must be a JSON object")
        return errors

    for cat_name, cat in categories.items():
        if not isinstance(cat, dict):
            errors.append(f"categories.{cat_name}: must be a JSON object")
            continue

        target = cat.get("target_total")
        if not (isinstance(target, int) and target > 0):
            errors.append(f"categories.{cat_name}.target_total: must be a positive integer (got {target!r})")

        formats = cat.get("formats")
        if not isinstance(formats, list) or len(formats) == 0:
            errors.append(f"categories.{cat_name}.formats: must be a non-empty list")
            continue

        for fi, fmt in enumerate(formats):
            prefix = f"categories.{cat_name}.formats[{fi}]"
            if not isinstance(fmt, dict):
                errors.append(f"{prefix}: must be a JSON object")
                continue

            tmpl_name = fmt.get("template")
            if tmpl_name not in fmt_tmpls:
                errors.append(f"{prefix}.template: '{tmpl_name}' not found in format_templates")

            weight = fmt.get("weight")
            if not (isinstance(weight, (int, float)) and weight > 0):
                errors.append(f"{prefix}.weight: must be a positive number (got {weight!r})")

            source_mode = fmt.get("source_mode")
            if source_mode not in VALID_SOURCE_MODES:
                errors.append(
                    f"{prefix}.source_mode: '{source_mode}' is not valid "
                    f"(must be one of {sorted(VALID_SOURCE_MODES)})"
                )

            deg_mix = fmt.get("degradation_mix")
            if not isinstance(deg_mix, dict) or len(deg_mix) == 0:
                errors.append(f"{prefix}.degradation_mix: must be a non-empty JSON object")
                continue

            for dname, dweight in deg_mix.items():
                if dname not in deg_tmpls:
                    errors.append(
                        f"{prefix}.degradation_mix.{dname}: "
                        f"not found in degradation_templates"
                    )
                if not (isinstance(dweight, (int, float)) and dweight > 0):
                    errors.append(
                        f"{prefix}.degradation_mix.{dname}: "
                        f"weight must be a positive number (got {dweight!r})"
                    )

    return errors
