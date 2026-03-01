"""
config_normalizer.py
──────────────────────────────────────────────────────────────────────────────
Converts a V2-format config (flat structure written by video_manager.py) into
the internal structure expected by DatasetGeneratorV2UHD.

Mapping (V2 → internal)
───────────────────────
base_settings.output_base_dir      ← root_path
base_settings.temp_dir             ← root_path/temp
base_settings.status_file          ← root_path/.generator_status.json
base_settings.lr_versions          ← ['7frames'] if processing.n_frames == 7
                                       else ['5frames']
base_settings.min_detail_threshold ← quality.blur_threshold  (default 80.0)

category_targets                   ← category_patches  (patch count per category)

format_config                      ← enabled output_patches applied per-category.
                                      Probability for each size is derived from its
                                      'weight' field (integer %).  When 'weight' is
                                      absent, all enabled sizes get equal probability.
"""

import os


def normalize_config(config: dict) -> dict:
    """Convert a V2-format config to the internal structure used by the generator."""
    processing = config.get('processing', {})
    quality    = config.get('quality', {})
    root_path  = config.get('root_path', '')
    n_frames   = processing.get('n_frames', 7)

    normalized = dict(config)

    # Build base_settings
    normalized['base_settings'] = {
        'output_base_dir':      root_path,
        'temp_dir':             os.path.join(root_path, 'temp'),
        'status_file':          os.path.join(root_path, '.generator_status.json'),
        'lr_versions':          ['7frames'] if n_frames == 7 else ['5frames'],
        'min_detail_threshold': quality.get('blur_threshold', 80.0),
    }

    # category_patches maps directly to category_targets (patch count per category)
    category_patches = config.get('category_patches', {})
    if category_patches:
        normalized['category_targets'] = dict(category_patches)
    elif 'category_targets' not in normalized:
        normalized['category_targets'] = {}

    # Build format_config: enabled output_patches applied per-category.
    # Each size's probability comes from its 'weight' field (integer %).
    # If no weight is set, all sizes share equal probability.
    output_patches = config.get('output_patches', {})
    enabled = {k: v for k, v in output_patches.items() if v.get('enabled', True)}
    if enabled:
        total_weight = sum(v.get('weight', 1) for v in enabled.values())
        if total_weight <= 0:
            total_weight = len(enabled)
        format_entry = {
            fmt_key: {
                'gt_size':     fmt_val['gt_size'],
                'lr_size':     fmt_val['lr_size'],
                'probability': round(fmt_val.get('weight', 1) / total_weight, 6),
            }
            for fmt_key, fmt_val in enabled.items()
        }
        categories = list(normalized.get('category_targets', {}).keys())
        normalized['format_config'] = {cat: format_entry for cat in categories}
    elif 'format_config' not in normalized:
        normalized['format_config'] = {}

    return normalized
