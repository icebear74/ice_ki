"""Format definitions for multi-category dataset generator."""

# Format specifications for different patch sizes
FORMATS = {
    'small_540': {
        'gt_size': (540, 540),
        'lr_size': (180, 180),
        'output_dir': 'Patches',
        'suffix': '',
        'aspect_ratio': '1:1'
    },
    'medium_169': {
        'gt_size': (720, 405),
        'lr_size': (240, 135),
        'output_dir': 'Patches_Medium169',
        'suffix': '_med169',
        'aspect_ratio': '16:9'
    },
    'large_720': {
        'gt_size': (720, 720),
        'lr_size': (240, 240),
        'output_dir': 'Patches_Large',
        'suffix': '_large',
        'aspect_ratio': '1:1'
    },
    'xlarge_1440': {
        'gt_size': (1440, 810),
        'lr_size': (480, 270),
        'output_dir': 'Patches_XLarge169',
        'suffix': '_xl169',
        'aspect_ratio': '16:9'
    },
    'fullhd_1920': {
        'gt_size': (1920, 1080),
        'lr_size': (640, 360),
        'output_dir': 'Patches_FullHD',
        'suffix': '_fullhd',
        'aspect_ratio': '16:9'
    }
}

# Category-specific format distribution (probabilities must sum to 1.0)
CATEGORY_FORMAT_DISTRIBUTION = {
    'general': {
        'small_540': 0.45,
        'medium_169': 0.35,
        'large_720': 0.20
    },
    'space': {
        'small_540': 0.30,
        'xlarge_1440': 0.45,
        'fullhd_1920': 0.25
    },
    'toon': {
        'small_540': 0.65,
        'medium_169': 0.35
    }
}

# Base paths for each category
CATEGORY_PATHS = {
    'general': 'Universal/Mastermodell/Learn',
    'space': 'Space/SpaceModel/Learn',
    'toon': 'Toon/ToonModel/Learn'
}

def get_format_for_category(category):
    """Get list of available formats for a category."""
    return list(CATEGORY_FORMAT_DISTRIBUTION.get(category, {}).keys())

def select_random_format(category):
    """Select a random format based on category distribution."""
    import random
    distribution = CATEGORY_FORMAT_DISTRIBUTION.get(category, {})
    if not distribution:
        return 'small_540'
    
    formats = list(distribution.keys())
    weights = list(distribution.values())
    return random.choices(formats, weights=weights, k=1)[0]

def get_output_dirs_for_format(base_path, category, format_name):
    """Get output directory paths for a specific format."""
    category_path = CATEGORY_PATHS[category]
    format_spec = FORMATS[format_name]
    base_format_dir = format_spec['output_dir']
    
    return {
        'gt': f"{base_path}/{category_path}/{base_format_dir}/GT",
        'lr_5frames': f"{base_path}/{category_path}/{base_format_dir}/LR_5frames",
        'lr_7frames': f"{base_path}/{category_path}/{base_format_dir}/LR_7frames",
        'val_gt': f"{base_path}/{category_path}/Val/GT"
    }
