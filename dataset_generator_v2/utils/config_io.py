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

# Supported aspect ratios: name → (numerator, denominator)
ASPECT_RATIOS = {
    "16:9": (16, 9),
    "4:3":  (4,  3),
    "1:1":  (1,  1),
}

# Common base_x preset values for the format-template wizard.
# These are all multiples of 96 and divisible by both 3 and 4, which ensures
# clean integer gt_size and lr_size for 16:9, 4:3, and 1:1 aspect ratios at
# scale factors 2, 3, and 4. Custom values are still accepted.
BASE_X_PRESETS = [768, 864, 960, 1024, 1152, 1280]


# ── Format-size helpers ───────────────────────────────────────────────────────

def compute_format_sizes(base_x: int, aspect_ratio: str, scale: int):
    """
    Compute gt_size and lr_size from base_x, aspect_ratio, and scale.

    Returns (gt_size, lr_size) where each is a [width, height] list of ints.
    Raises ValueError for any inconsistency:
      - unsupported aspect_ratio
      - height would not be an integer
      - gt_size dimensions not divisible by scale
    """
    if aspect_ratio not in ASPECT_RATIOS:
        raise ValueError(
            f"Unsupported aspect_ratio '{aspect_ratio}'. "
            f"Supported: {', '.join(sorted(ASPECT_RATIOS))}"
        )
    num, den = ASPECT_RATIOS[aspect_ratio]
    if (base_x * den) % num != 0:
        raise ValueError(
            f"base_x={base_x} does not produce an integer height for {aspect_ratio}. "
            f"({base_x}*{den}/{num} = {base_x * den / num:.4f} – not integer)"
        )
    gt_w = base_x
    gt_h = (base_x * den) // num
    if gt_w % scale != 0:
        raise ValueError(
            f"gt_size width {gt_w} is not divisible by scale {scale}"
        )
    if gt_h % scale != 0:
        raise ValueError(
            f"gt_size height {gt_h} is not divisible by scale {scale}"
        )
    lr_w = gt_w // scale
    lr_h = gt_h // scale
    return [gt_w, gt_h], [lr_w, lr_h]


def build_format_template(
    base_x: int,
    aspect_ratio: str,
    scale: int,
    description: str = "",
) -> dict:
    """
    Build a complete format-template dict from declarative parameters.

    Stores both the source parameters (base_x, aspect_ratio, scale) and the
    derived sizes (gt_size, lr_size) so that:
      - the template is self-documenting and easy to edit
      - the generator can read gt_size / lr_size directly without recalculation

    Raises ValueError if the combination is invalid.
    """
    gt_size, lr_size = compute_format_sizes(base_x, aspect_ratio, scale)
    return {
        "base_x":       base_x,
        "aspect_ratio": aspect_ratio,
        "scale":        scale,
        "gt_size":      gt_size,
        "lr_size":      lr_size,
        "description":  description,
    }


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
            # 1152_169 – main 16:9 format for clean UHD resize
            # base_x=1152, 16:9 → gt 1152x648, lr 384x216
            "1152_169": build_format_template(
                1152, "16:9", 3,
                "1152x648 16:9 – main resize target (UHD→HD quality step)"
            ),
            # 960_169 – secondary 16:9 for crop use or lighter resize
            # base_x=960, 16:9 → gt 960x540, lr 320x180
            "960_169": build_format_template(
                960, "16:9", 3,
                "960x540 16:9 – secondary resize / crop target"
            ),
            # 960_43 – 4:3 format for classic TV / sitcom / older sci-fi content
            # base_x=960, 4:3 → gt 960x720, lr 320x240
            "960_43": build_format_template(
                960, "4:3", 3,
                "960x720 4:3 – classic TV / sitcom / older sci-fi (crop from 16:9 source)"
            ),
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
                        "template": "1152_169",
                        "weight": 60,
                        "source_mode": "resize",
                        "degradation_mix": {
                            "dvd_film_balanced": 50,
                            "classic_sitcom_sd": 50
                        }
                    },
                    {
                        "template": "960_169",
                        "weight": 40,
                        "source_mode": "crop",
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

    Checks:
    - format_templates entries have valid gt_size, scale
    - gt_size dimensions are divisible by scale (lr_size must be integer)
    - if lr_size is stored, it matches gt_size // scale
    - if base_x / aspect_ratio are stored, derived gt_size matches stored gt_size
    - degradation_templates is a dict

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
            scale = spec.get("scale")

            gt_ok = (
                isinstance(gt, (list, tuple))
                and len(gt) == 2
                and all(isinstance(x, int) and x > 0 for x in gt)
            )
            if not gt_ok:
                errors.append(
                    f"format_templates.{name}.gt_size: must be [width, height] positive ints"
                )

            scale_ok = isinstance(scale, int) and scale > 0
            if not scale_ok:
                errors.append(
                    f"format_templates.{name}.scale: must be a positive int"
                )

            if gt_ok and scale_ok:
                gt_w, gt_h = gt[0], gt[1]
                if gt_w % scale != 0:
                    errors.append(
                        f"format_templates.{name}: gt_size width {gt_w} "
                        f"is not divisible by scale {scale}"
                    )
                if gt_h % scale != 0:
                    errors.append(
                        f"format_templates.{name}: gt_size height {gt_h} "
                        f"is not divisible by scale {scale}"
                    )

                # Validate stored lr_size if present
                stored_lr = spec.get("lr_size")
                if stored_lr is not None:
                    expected_lr = [gt_w // scale, gt_h // scale]
                    if stored_lr != expected_lr:
                        errors.append(
                            f"format_templates.{name}.lr_size: stored {stored_lr} "
                            f"does not match gt_size // scale = {expected_lr}"
                        )

            # Validate that stored base_x / aspect_ratio are consistent with gt_size
            base_x = spec.get("base_x")
            ar = spec.get("aspect_ratio")
            if base_x is not None and ar is not None and gt_ok and scale_ok:
                try:
                    derived_gt, _ = compute_format_sizes(base_x, ar, scale)
                    if derived_gt != list(gt):
                        errors.append(
                            f"format_templates.{name}: base_x={base_x} + "
                            f"aspect_ratio={ar} + scale={scale} would yield "
                            f"gt_size={derived_gt}, but stored gt_size is {list(gt)}"
                        )
                except ValueError as exc:
                    errors.append(f"format_templates.{name}: {exc}")

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
