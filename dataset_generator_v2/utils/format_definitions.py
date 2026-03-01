"""Format definitions for multi-category dataset generator."""

# Format specifications for different patch sizes
# New V2 structure: uses simple size keys (720, 540, 720_169)
FORMATS = {
    '540': {
        'gt_size': (540, 540),
        'scale': 3,
        'output_dir': 'patches/540',
        'suffix': '',
        'aspect_ratio': '1:1'
    },
    '720_169': {
        'gt_size': (720, 405),  # (width, height) = 720 wide, 405 tall = 16:9 landscape
        'scale': 3,
        'output_dir': 'patches/720_169',
        'suffix': '',
        'aspect_ratio': '16:9'
    },
    '720': {
        'gt_size': (720, 720),
        'scale': 3,
        'output_dir': 'patches/720',
        'suffix': '',
        'aspect_ratio': '1:1'
    },
    # Legacy format names for backward compatibility
    'small_540': {
        'gt_size': (540, 540),
        'scale': 3,
        'output_dir': 'patches/540',
        'suffix': '',
        'aspect_ratio': '1:1'
    },
    'medium_169': {
        'gt_size': (720, 405),  # (width, height) = 720 wide, 405 tall = 16:9 landscape
        'scale': 3,
        'output_dir': 'patches/720_169',
        'suffix': '',
        'aspect_ratio': '16:9'
    },
    'large_720': {
        'gt_size': (720, 720),
        'scale': 3,
        'output_dir': 'patches/720',
        'suffix': '',
        'aspect_ratio': '1:1'
    }
}

# Category-specific format distribution (probabilities must sum to 1.0)
CATEGORY_FORMAT_DISTRIBUTION = {
    'master': {
        'small_540': 0.50,
        'medium_169': 0.35,
        'large_720': 0.15
    },
    'universal': {
        'small_540': 0.50,
        'medium_169': 0.35,
        'large_720': 0.15
    },
    'space': {
        'small_540': 0.40,
        'medium_169': 0.35,
        'large_720': 0.25
    },
    'toon': {
        'small_540': 0.65,
        'medium_169': 0.35
    }
}

# Base paths for each category - NEW V2 STRUCTURE: flat category names
CATEGORY_PATHS = {
    'master': 'master',
    'universal': 'universal',
    'space': 'space',
    'toon': 'toon'
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

def get_output_dirs_for_format(base_path, category, format_name, lr_frames=5):
    """
    Get output directory paths for a specific format.
    
    Args:
        base_path: Base dataset directory
        category: Category (master/universal/space/toon)
        format_name: Format name (small_540, etc.)
        lr_frames: Number of LR frames to use (5 or 7)
                  5 = VSR++ compatible (default)
                  7 = Extended version
    
    Returns:
        Dictionary with 'gt', 'lr', 'val_gt' paths
    
    VSR++ Training expects:
        - dataset_root/Patches/GT/
        - dataset_root/Patches/LR/  (5-frame stack: 180×900)
        - dataset_root/Val/GT/
        - dataset_root/Val/LR/ (optional, falls back to Patches/LR)
    """
    category_path = CATEGORY_PATHS.get(category, f'{category.capitalize()}/{category.capitalize()}Model/Learn')
    format_spec = FORMATS[format_name]
    base_format_dir = format_spec['output_dir']
    
    # VSR++ compatible: Use 'LR' for 5-frame, 'LR_7frames' for extended
    lr_dir_name = 'LR' if lr_frames == 5 else 'LR_7frames'
    
    return {
        'gt': f"{base_path}/{category_path}/{base_format_dir}/GT",
        'lr': f"{base_path}/{category_path}/{base_format_dir}/{lr_dir_name}",
        'val_gt': f"{base_path}/{category_path}/Val/GT",
        'val_lr': f"{base_path}/{category_path}/Val/LR"
    }