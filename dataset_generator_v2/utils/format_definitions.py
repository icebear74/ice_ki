"""Format definitions for multi-category dataset generator."""

# Format specifications for different patch sizes
FORMATS = {
    '540': {
        'gt_size': (540, 540),
        'scale': 3,
        'output_dir': '540',
        'suffix': '',
        'aspect_ratio': '1:1'
    },
    '720_169': {
        'gt_size': (720, 405),
        'scale': 3,
        'output_dir': '720_169',
        'suffix': '',
        'aspect_ratio': '16:9'
    },
    '720': {
        'gt_size': (720, 720),
        'scale': 3,
        'output_dir': '720',
        'suffix': '',
        'aspect_ratio': '1:1'
    },
}

# Base paths for each category – flat V2 structure
CATEGORY_PATHS = {
    'master': 'master',
    'universal': 'universal',
    'space': 'space',
    'toon': 'toon'
}


def get_output_dirs_for_format(base_path, category, format_name, lr_frames=5):
    """
    Get output directory paths for a specific format.

    Args:
        base_path: Base dataset directory
        category: Category name
        format_name: Format key in FORMATS (or any template name)
        lr_frames: 5 = VSR++ compatible LR dir, 7 = extended LR_7frames dir

    Returns:
        Dictionary with 'gt', 'lr', 'val_gt', 'val_lr' paths
    """
    category_path = CATEGORY_PATHS.get(category, category)
    format_spec = FORMATS.get(format_name, {'output_dir': format_name, 'aspect_ratio': '1:1'})
    base_format_dir = format_spec['output_dir']

    lr_dir_name = 'LR' if lr_frames == 5 else 'LR_7frames'

    return {
        'gt': f"{base_path}/{category_path}/patches/{base_format_dir}/GT",
        'lr': f"{base_path}/{category_path}/patches/{base_format_dir}/{lr_dir_name}",
        'val_gt': f"{base_path}/{category_path}/val/{base_format_dir}/GT",
        'val_lr': f"{base_path}/{category_path}/val/{base_format_dir}/{lr_dir_name}"
    }