"""Format definitions for multi-category dataset generator."""

import os as _os

# ---------------------------------------------------------------------------
# Bucket layout
# ---------------------------------------------------------------------------
# To avoid filesystem performance degradation with hundreds of thousands of
# files in a single directory, patches are stored in 4-digit zero-padded
# subdirectories ("buckets"):
#
#   master/patches/720/GT/0000/   ← up to BUCKET_SIZE PNG files
#   master/patches/720/GT/0001/   ← next bucket, created when 0000 is full
#   …
#
# GT and LR buckets always share the same bucket name so filenames match
# across the two trees.  The bucket index is determined ONCE per video
# (see get_synced_bucket_dirs) so that a single video's patches are never
# split across two different buckets.
BUCKET_SIZE: int = 10_000


def get_synced_bucket_dirs(gt_base: str, lr_base: str,
                            bucket_size: int = BUCKET_SIZE) -> tuple:
    """Return ``(gt_bucket_dir, lr_bucket_dir)`` with a matching bucket index.

    Buckets are 4-digit zero-padded subdirectories (``0000``, ``0001``, …).
    The current bucket is identified by looking at the highest-numbered GT
    bucket and counting the PNG files it already contains.  Both GT and LR
    receive the **same** bucket name so filenames stay aligned.

    Call this function **once per video** before any patch from that video is
    written.  That guarantees that all patches from a single video land in the
    same bucket even if the bucket would overflow mid-video.

    The returned paths are *not* created on disk; the caller must
    ``os.makedirs(path, exist_ok=True)`` them.

    Args:
        gt_base:     Base GT directory  (e.g. ``.../master/patches/720/GT``).
        lr_base:     Base LR directory  (e.g. ``.../master/patches/720/LR_7frames``).
        bucket_size: Maximum PNG files per bucket (default :data:`BUCKET_SIZE`).

    Returns:
        ``(gt_bucket_dir, lr_bucket_dir)``
    """
    existing: list = []
    if _os.path.isdir(gt_base):
        existing = sorted(
            d for d in _os.listdir(gt_base)
            if len(d) == 4 and d.isdigit()
            and _os.path.isdir(_os.path.join(gt_base, d))
        )

    if existing:
        last = existing[-1]
        last_gt = _os.path.join(gt_base, last)
        try:
            count = sum(
                1 for f in _os.listdir(last_gt) if f.lower().endswith('.png')
            )
        except OSError:
            count = 0
        bucket_name = last if count < bucket_size else f"{int(last) + 1:04d}"
    else:
        bucket_name = "0000"

    return _os.path.join(gt_base, bucket_name), _os.path.join(lr_base, bucket_name)


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