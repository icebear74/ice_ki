"""Format definitions for multi-category dataset generator.

Fully dynamic — no hardcoded format names or category names.
All paths are derived from the active template config at runtime.
"""

import os as _os

# ---------------------------------------------------------------------------
# Bucket layout
# ---------------------------------------------------------------------------
# To avoid filesystem performance degradation with hundreds of thousands of
# files in a single directory, patches are stored in 4-digit zero-padded
# subdirectories ("buckets"):
#
#   master/patches/1152_169/GT/0000/   ← up to BUCKET_SIZE image files
#   master/patches/1152_169/GT/0001/   ← next bucket, created when 0000 is full
#   …
#
# GT and LR buckets always share the same bucket name so filenames match
# across the two trees.  The bucket index is determined ONCE per video
# (see get_synced_bucket_dirs) so that a single video's patches are never
# split across two different buckets.
BUCKET_SIZE: int = 10_000

# All image extensions that count toward bucket fullness.
# Supports BMP (default), PNG, and common alternatives.
_IMAGE_EXTS: frozenset = frozenset({
    '.png', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff', '.webp'
})


def get_synced_bucket_dirs(gt_base: str, lr_base: str,
                            bucket_size: int = BUCKET_SIZE) -> tuple:
    """Return ``(gt_bucket_dir, lr_bucket_dir)`` with a matching bucket index.

    Buckets are 4-digit zero-padded subdirectories (``0000``, ``0001``, …).
    The current bucket is identified by looking at the highest-numbered GT
    bucket and counting the image files it already contains.  Both GT and LR
    receive the **same** bucket name so filenames stay aligned.

    Supports any output image format (BMP, PNG, JPEG, …) — counts all image
    files regardless of extension so that mixed-format datasets are handled
    correctly.

    Call this function **once per video** before any patch from that video is
    written.  That guarantees that all patches from a single video land in the
    same bucket even if the bucket would overflow mid-video.

    The returned paths are *not* created on disk; the caller must
    ``os.makedirs(path, exist_ok=True)`` them.

    Args:
        gt_base:     Base GT directory  (e.g. ``.../master/patches/1152_169/GT``).
        lr_base:     Base LR directory  (e.g. ``.../master/patches/1152_169/LR_7frames``).
        bucket_size: Maximum image files per bucket (default :data:`BUCKET_SIZE`).

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
                1 for f in _os.listdir(last_gt)
                if _os.path.splitext(f)[1].lower() in _IMAGE_EXTS
            )
        except OSError:
            count = 0
        bucket_name = last if count < bucket_size else f"{int(last) + 1:04d}"
    else:
        bucket_name = "0000"

    return _os.path.join(gt_base, bucket_name), _os.path.join(lr_base, bucket_name)


def get_output_dirs_for_format(base_path, category, format_name, lr_frames=5):
    """
    Get output directory paths for a specific format.

    Fully dynamic: category and format_name are used as-is from the active
    template config.  No hardcoded format or category lookups.

    Args:
        base_path:   Base dataset directory
        category:    Category name (from config, e.g. 'master', 'space')
        format_name: Template name (from config, e.g. '1152_169', '960_43')
        lr_frames:   Number of LR frames — controls the LR subdirectory name

    Returns:
        Dictionary with 'gt', 'lr', 'val_gt', 'val_lr' paths
    """
    lr_dir_name = 'LR' if lr_frames == 5 else f'LR_{lr_frames}frames'

    return {
        'gt':     f"{base_path}/{category}/patches/{format_name}/GT",
        'lr':     f"{base_path}/{category}/patches/{format_name}/{lr_dir_name}",
        'val_gt': f"{base_path}/{category}/val/{format_name}/GT",
        'val_lr': f"{base_path}/{category}/val/{format_name}/{lr_dir_name}",
    }