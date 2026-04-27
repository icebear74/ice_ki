"""
VSRDataset - Video Super-Resolution Dataset Loader

Loads VSR training data with Dataset Generator V2 structure:
- Dataset root: configurable (not hardcoded).
- Category (e.g. master, space) is configurable.
- Architecture metadata is auto-detected from
  ``{dataset_root}/dataset_architecture.json`` when present.

Directory layout (V2 – bucket subdirs):
    {root}/{category}/patches/{template}/GT/0000/<file>.(bmp|png)
    {root}/{category}/patches/{template}/LR_{n}frames/0000/<file>.(bmp|png)
    {root}/{category}/val/{template}/GT/<file>.(bmp|png)  ← GT only; LR found from patches

Legacy flat layout (no bucket dirs) is also supported for backward compatibility.

Key features:
- Supports both BMP (default) and PNG; extension driven by ``img_ext`` param or
  auto-detected from ``dataset_architecture.json``.
- n_frames driven by ``n_frames`` param or auto-detected from architecture JSON.
- Val mode: LR is located in patches/{template}/LR_{n}frames by basename match
  using a precomputed index (no per-sample directory scan).
- Bucket-aware file scanning via ``_collect_image_files()``.
"""

import os
import cv2
import json
import time
import torch
import random
import numpy as np
import threading
from torch.utils.data import Dataset

# Index cache schema version — bump this whenever the stored format changes.
# Bumped to 2 to invalidate caches from the pre-bucket / PNG-only era.
_INDEX_CACHE_VERSION = 2

# Tolerance (seconds) for directory mtime comparison.
# FAT32/SMB shares round mtime to 2 s; 1.0 s covers most local filesystems.
_MTIME_TOLERANCE_SEC = 1.0

# Supported image extensions (in priority order for auto-detection).
_SUPPORTED_EXTS = (".bmp", ".png")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _collect_image_files(base_dir: str, img_ext: str = "") -> list:
    """Return a sorted list of image file relative paths under *base_dir*.

    Supports two layouts:
    * **Bucket layout** (V2): 4-digit zero-padded subdirectories
      (``0000/``, ``0001/``, …) that contain the image files.
      Returned paths are ``"0000/foo.bmp"``, ``"0001/bar.bmp"``, etc.
    * **Flat layout** (legacy): image files directly inside *base_dir*.
      Returned paths are just the bare filename ``"foo.bmp"``.

    GT and LR bucket directories always share the same bucket name so a
    relative key from the GT side can be looked up directly in the LR tree.

    Args:
        base_dir: Directory to scan (e.g. ``.../master/patches/1152_169/GT``).
        img_ext:  File extension to filter by, **including** the leading dot
                  (e.g. ``".bmp"`` or ``".png"``).  When empty or ``""`` the
                  function accepts *all* extensions in :data:`_SUPPORTED_EXTS`.

    Returns:
        Sorted list of relative paths (strings).
    """
    files: list = []
    if not os.path.isdir(base_dir):
        return files

    # Normalise extension filter
    ext_filter: tuple
    if img_ext:
        ext_filter = (img_ext.lower(),)
    else:
        ext_filter = _SUPPORTED_EXTS

    def _matches(name: str) -> bool:
        return any(name.lower().endswith(e) for e in ext_filter)

    entries = os.listdir(base_dir)

    # Detect bucket layout: subdirs whose names are exactly 4 decimal digits.
    bucket_dirs = sorted(
        e for e in entries
        if len(e) == 4 and e.isdigit()
        and os.path.isdir(os.path.join(base_dir, e))
    )

    if bucket_dirs:
        # New V2 bucket layout
        for bucket in bucket_dirs:
            bucket_path = os.path.join(base_dir, bucket)
            try:
                for f in sorted(os.listdir(bucket_path)):
                    if _matches(f):
                        files.append(os.path.join(bucket, f))  # "0000/foo.bmp"
            except OSError:
                pass
    else:
        # Legacy flat layout
        for f in sorted(entries):
            if _matches(f):
                files.append(f)

    return files


def _auto_detect_ext(base_dir: str) -> str:
    """Probe *base_dir* (or its first bucket subdir) to find the image format.

    Returns the first matching extension from :data:`_SUPPORTED_EXTS`, or
    ``".bmp"`` as a safe default when nothing is found.
    """
    if not os.path.isdir(base_dir):
        return ".bmp"

    # If bucket layout, look inside the first bucket
    entries = os.listdir(base_dir)
    bucket_dirs = sorted(
        e for e in entries
        if len(e) == 4 and e.isdigit()
        and os.path.isdir(os.path.join(base_dir, e))
    )
    search_dir = os.path.join(base_dir, bucket_dirs[0]) if bucket_dirs else base_dir

    try:
        names = os.listdir(search_dir)
    except OSError:
        return ".bmp"

    for ext in _SUPPORTED_EXTS:
        if any(n.lower().endswith(ext) for n in names):
            return ext
    return ".bmp"


class VSRDataset(Dataset):
    """
    VSR Dataset for training and validation
    
    Args:
        root: Root directory (e.g., /mnt/data/training/Dataset)
        dataset_name: Dataset category name (e.g., 'master', 'space')
        size_key: Format template key (e.g., '720', '540', '720_169', '1152_169')
        mode: 'train' or 'val'
        augment: Ignored – augmentation is permanently disabled.
                 With 350k+ diverse scenes the variance gain is negligible
                 while the copy overhead (~5-15 ms/sample) is real.
        n_frames: Number of LR frames stacked vertically in each LR image.
                  Defaults to 7.  Must match how the dataset was generated.
        img_ext: Image file extension including dot, e.g. ``".bmp"`` or
                 ``".png"``.  Pass ``""`` (default) to auto-detect from the
                 first file found in the GT directory.
        paths_config: Optional dict with path patterns:
            - train_gt: Pattern for training GT (default: 'patches/{size_key}/GT')
            - train_lr: Pattern for training LR (default: 'patches/{size_key}/LR_{n_frames}frames')
            - val_gt: Pattern for validation GT (default: 'val/{size_key}/GT')
            - val_lr: Pattern for validation LR (default: 'patches/{size_key}/LR_{n_frames}frames')
    """
    
    def __init__(self, root, dataset_name='master', size_key='720', mode='train', augment=True,
                 n_frames=7, img_ext="", paths_config=None, validate_upfront=False):
        self.root = root
        self.dataset_name = dataset_name
        self.size_key = size_key
        self.mode = mode
        self.n_frames = int(n_frames)
        # Augmentation permanently disabled: with 350k+ diverse scenes the
        # regularisation gain is negligible while the copy overhead is real.
        self.augment = False
        self.validate_upfront = validate_upfront

        # LR subdirectory name derived from n_frames
        lr_dir_name = 'LR' if self.n_frames == 5 else f'LR_{self.n_frames}frames'

        # Path patterns (configurable or defaults)
        if paths_config is None:
            paths_config = {}
        self.train_gt_pattern = paths_config.get('train_gt', 'patches/{size_key}/GT')
        self.train_lr_pattern = paths_config.get('train_lr', f'patches/{{size_key}}/{lr_dir_name}')
        self.val_gt_pattern = paths_config.get('val_gt', 'val/{size_key}/GT')
        self.val_lr_pattern = paths_config.get('val_lr', f'patches/{{size_key}}/{lr_dir_name}')

        # Thread lock for safe reloading during training
        self.reload_lock = threading.Lock()
        
        # Build paths based on mode
        dataset_path = os.path.join(root, dataset_name)
        
        if mode == 'train':
            # Training: use train_gt_pattern and train_lr_pattern
            gt_path = self.train_gt_pattern.replace('{size_key}', size_key)
            lr_path = self.train_lr_pattern.replace('{size_key}', size_key)
            
            self.gt_dir = os.path.join(dataset_path, gt_path)
            self.lr_dir = os.path.join(dataset_path, lr_path)
            self.patch_lr_dir = None  # Not needed for training
        elif mode == 'val':
            # Validation: GT only — LR found from patches via precomputed index
            gt_path = self.val_gt_pattern.replace('{size_key}', size_key)
            lr_path = self.val_lr_pattern.replace('{size_key}', size_key)
            
            self.gt_dir = os.path.join(dataset_path, gt_path)
            self.lr_dir = None  # Will use patch_lr_dir
            self.patch_lr_dir = os.path.join(dataset_path, lr_path)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'val'")
        
        # Determine image extension — caller-supplied wins, else auto-detect.
        if img_ext:
            self.img_ext = img_ext.lower() if img_ext.startswith(".") else f".{img_ext.lower()}"
        else:
            self.img_ext = _auto_detect_ext(self.gt_dir)

        # Get all GT files (bucket-aware, extension-aware)
        if not os.path.exists(self.gt_dir):
            raise ValueError(f"GT directory not found: {self.gt_dir}")
        
        all_gt_files = _collect_image_files(self.gt_dir, self.img_ext)
        
        if not all_gt_files:
            raise ValueError(
                f"No {self.img_ext} files found in {self.gt_dir} "
                f"(checked bucket and flat layout)"
            )
        
        # ------------------------------------------------------------------
        # Fast path: try to restore the file index from the on-disk cache.
        # The cache is skipped when validate_upfront=True because image-level
        # validation results cannot be reliably reproduced from mtime alone.
        # ------------------------------------------------------------------
        _loaded_from_cache = False
        if not self.validate_upfront:
            cached = self._load_index()
            if cached is not None:
                self.gt_files, self.lr_paths = cached
                _loaded_from_cache = True
                print(f"\n⚡ Index-Cache geladen: {len(self.gt_files)} Dateien für {mode} ({size_key})")
                # Still run the quick sample validation even on cache hit
                self._validate_samples()
                return

        # ------------------------------------------------------------------
        # Slow path: full directory scan (result is cached for next time)
        # ------------------------------------------------------------------
        self.gt_files = []
        self.lr_paths = {}  # key: relative path (may include bucket prefix), value: full lr dir
        skipped_files = []
        matched_val_lr = 0
        matched_patches_lr = 0

        # ------------------------------------------------------------------
        # For VAL mode: build a precomputed basename → full LR path index
        # from patches/LR_{n}frames so we never scan directories per sample.
        # ------------------------------------------------------------------
        patch_lr_basename_index: dict = {}
        if mode == 'val' and self.patch_lr_dir:
            patch_lr_files = _collect_image_files(self.patch_lr_dir, self.img_ext)
            for rel_path in patch_lr_files:
                basename = os.path.basename(rel_path)
                lr_bucket_dir = os.path.join(self.patch_lr_dir, os.path.dirname(rel_path)) if os.path.dirname(rel_path) else self.patch_lr_dir
                # Store first occurrence only (multiple buckets may have same stem — very unlikely)
                patch_lr_basename_index.setdefault(basename, lr_bucket_dir)
        
        # Expected shapes for validation (well-known legacy size_keys)
        expected_gt_shapes = {
            '720': (720, 720, 3),
            '540': (540, 540, 3),
            '720_169': (405, 720, 3)
        }
        expected_gt_shape = expected_gt_shapes.get(self.size_key)
        
        invalid_dimension_files = []
        
        # Progress tracking for validation
        total_files = len(all_gt_files)
        validated_count = 0
        if self.validate_upfront and total_files > 100:
            print(f"   Validating {total_files} files... (this may take a moment)")
        
        for gt_file in all_gt_files:
            # Show progress every 100 files when validating
            if self.validate_upfront and total_files > 100:
                validated_count += 1
                if validated_count % 100 == 0:
                    print(f"   Progress: {validated_count}/{total_files} files validated...")
            
            basename = os.path.basename(gt_file)

            if mode == 'train':
                # Training: LR file must exist under lr_dir with same relative path
                # (bucket prefix is shared between GT and LR)
                lr_bucket_dir = os.path.join(self.lr_dir, os.path.dirname(gt_file)) if os.path.dirname(gt_file) else self.lr_dir
                lr_path = os.path.join(lr_bucket_dir, basename)
                if os.path.exists(lr_path):
                    if self.validate_upfront and not self._validate_file_dimensions(
                            gt_file, self.gt_dir, lr_bucket_dir, expected_gt_shape, invalid_dimension_files):
                        pass
                    else:
                        self.gt_files.append(gt_file)
                        self.lr_paths[gt_file] = lr_bucket_dir
                        matched_val_lr += 1
                else:
                    skipped_files.append(gt_file)
            else:
                # Val mode: LR is in patches/{template}/LR_{n}frames, not in val/
                # Use the precomputed basename index for O(1) lookup.
                lr_bucket_dir = patch_lr_basename_index.get(basename)
                if lr_bucket_dir is not None:
                    if self.validate_upfront and not self._validate_file_dimensions(
                            gt_file, self.gt_dir, lr_bucket_dir, expected_gt_shape, invalid_dimension_files):
                        pass
                    else:
                        self.gt_files.append(gt_file)
                        self.lr_paths[gt_file] = lr_bucket_dir
                        matched_patches_lr += 1
                else:
                    skipped_files.append(gt_file)
        
        # Show detailed statistics for val mode
        if mode == 'val':
            print("\n" + "="*60)
            print(f"📂 VALIDATION DATASET LOADING ({size_key})")
            print("="*60)
            print(f"  GT files found:           {len(all_gt_files)}")
            print(f"  Matched in patches/LR:    {matched_patches_lr}")
            print(f"  ───────────────────────────────────")
            print(f"  Skipped (no LR):          {len(skipped_files)}")
            if self.validate_upfront:
                print(f"  Skipped (invalid dims):   {len(invalid_dimension_files)}")
            else:
                print(f"  Upfront validation:       SKIPPED (faster startup)")
            print(f"  Final samples loaded:     {len(self.gt_files)}")
            print("="*60)
            
            if invalid_dimension_files and self.validate_upfront:
                print(f"\n⚠️  {len(invalid_dimension_files)} files skipped due to invalid dimensions:")
                for i, (f, reason) in enumerate(invalid_dimension_files[:5]):  # Show first 5
                    print(f"  - {f}: {reason}")
                if len(invalid_dimension_files) > 5:
                    print(f"  ... and {len(invalid_dimension_files) - 5} more")
                print(f"\n💡 Expected dimensions for size_key '{size_key}':")
                if expected_gt_shape:
                    print(f"   GT: {expected_gt_shape}")
                print()
            
            if skipped_files:
                print(f"\n⚠️  {len(skipped_files)} GT files skipped (no matching LR file):")
                for i, f in enumerate(skipped_files[:15]):  # Show first 15
                    print(f"  - {f}")
                if len(skipped_files) > 15:
                    print(f"  ... and {len(skipped_files) - 15} more")
                print("\n💡 To include these files, ensure LR versions exist in:")
                if self.patch_lr_dir:
                    print(f"     {self.patch_lr_dir}")
                print()
        elif invalid_dimension_files:
            # For training mode, only warn about dimension issues (GT-LR filename mismatches are silently skipped)
            if invalid_dimension_files and self.validate_upfront:
                print(f"⚠️  Skipped {len(invalid_dimension_files)} files with invalid dimensions in {mode} (size_key={size_key})")
            print()
        
        if not self.gt_files:
            raise ValueError(
                f"No valid GT-LR pairs found. GT dir: {self.gt_dir}  "
                f"LR dir: {self.lr_dir or self.patch_lr_dir}"
            )
        
        # Report invalid dimension files for training mode too
        if invalid_dimension_files and mode == 'train' and self.validate_upfront:
            print(f"\n⚠️  Skipped {len(invalid_dimension_files)} files with invalid dimensions in {mode} (size_key={size_key})")
            if len(invalid_dimension_files) <= 3:
                for f, reason in invalid_dimension_files:
                    print(f"  - {f}: {reason}")
            print()
        elif mode == 'train' and not self.validate_upfront:
            print(f"\n💡 Upfront validation SKIPPED for faster startup (runtime validation active)")
            print(f"   Loaded {len(self.gt_files)} files for {mode} ({size_key})\n")
        
        # Persist the index so the next startup can skip this scan
        if not self.validate_upfront:
            self._save_index(self.gt_files, self.lr_paths)

        # Validate a few samples
        self._validate_samples()
    
    def _validate_file_dimensions(self, gt_file, gt_dir, lr_dir, expected_gt_shape, invalid_list):
        """
        Validate that a GT/LR file pair has correct dimensions
        
        Args:
            gt_file: Filename to validate
            gt_dir: GT directory path
            lr_dir: LR directory path
            expected_gt_shape: Expected GT shape tuple (H, W, C) or None to skip
            invalid_list: List to append (filename, reason) tuples for invalid files
            
        Returns:
            bool: True if valid, False if invalid
        """
        if not expected_gt_shape:
            # No validation if shape unknown
            return True
        
        try:
            gt_path = os.path.join(self.gt_dir, gt_file)
            lr_path = os.path.join(lr_dir, os.path.basename(gt_file))
            
            # Quick dimension check without loading full image
            gt = cv2.imread(gt_path)
            lr = cv2.imread(lr_path)
            
            if gt is None:
                invalid_list.append((gt_file, "GT image failed to load"))
                return False
            
            if lr is None:
                invalid_list.append((gt_file, "LR image failed to load"))
                return False
            
            # Validate GT dimensions
            if gt.shape != expected_gt_shape:
                invalid_list.append((gt_file, f"GT shape {gt.shape} != expected {expected_gt_shape}"))
                return False
            
            # Validate LR can be split into n_frames
            if lr.shape[0] % self.n_frames != 0:
                invalid_list.append((gt_file, f"LR height {lr.shape[0]} not divisible by {self.n_frames}"))
                return False
            
            return True
            
        except Exception as e:
            invalid_list.append((gt_file, f"Validation error: {str(e)}"))
            return False
    
    def _validate_samples(self):
        """Validate dataset integrity by checking a few samples"""
        samples_to_check = min(5, len(self.gt_files))
        
        issues_found = []
        
        # Expected shapes based on well-known legacy size_keys.
        # For V2 dynamic templates (e.g. "1152_169") we skip shape validation
        # since we don't know the expected size without the architecture file.
        expected_gt_shapes = {
            '720': (720, 720, 3),      # 720×720 square patches
            '540': (540, 540, 3),      # 540×540 square patches
            '720_169': (405, 720, 3)   # 720×405 (16:9 aspect ratio)
        }
        
        expected_gt_shape = expected_gt_shapes.get(self.size_key)
        if not expected_gt_shape:
            # V2 dynamic template — skip fixed-shape validation
            return
        
        # LR should be height*n_frames, same width (n_frames stacked vertically)
        # Assume scale=3 for legacy size_keys (the only ones we know shapes for)
        expected_lr_width = expected_gt_shape[1] // 3
        expected_lr_height = (expected_gt_shape[0] * self.n_frames) // 3
        expected_lr_shape = (expected_lr_height, expected_lr_width, 3)
        
        for i in range(samples_to_check):
            gt_file = self.gt_files[i]
            gt_path = os.path.join(self.gt_dir, gt_file)
            # lr_paths value is the full directory containing the LR file
            lr_dir = self.lr_paths[gt_file]
            lr_path = os.path.join(lr_dir, os.path.basename(gt_file))
            
            # Check if files exist (should exist since we filtered them)
            if not os.path.exists(gt_path):
                issues_found.append(f"GT file not found: {gt_path}")
                continue
            if not os.path.exists(lr_path):
                issues_found.append(f"LR file not found: {lr_path}")
                continue
            
            # Load and validate shapes
            gt = cv2.imread(gt_path)
            lr = cv2.imread(lr_path)
            
            if gt is None:
                issues_found.append(f"Corrupted GT image: {gt_path}")
                continue
            if lr is None:
                issues_found.append(f"Corrupted LR image: {lr_path}")
                continue
            
            if gt.shape != expected_gt_shape:
                issues_found.append(f"Invalid GT shape {gt.shape}, expected {expected_gt_shape}: {gt_path}")
            # Allow ±2px tolerance for LR height to account for rounding in downscaling operations
            if lr.shape[1] != expected_lr_shape[1] or lr.shape[2] != expected_lr_shape[2]:
                issues_found.append(f"Invalid LR shape {lr.shape}, expected {expected_lr_shape}: {lr_path}")
            elif abs(lr.shape[0] - expected_lr_shape[0]) > 2:
                issues_found.append(f"Invalid LR height {lr.shape[0]}, expected {expected_lr_shape[0]} (±2px): {lr_path}")
        
        # Report issues as warnings instead of errors
        if issues_found:
            print(f"\n⚠️  Dataset validation warnings in {self.mode} (size_key={self.size_key}):")
            for issue in issues_found:
                print(f"  - {issue}")
            print()
    
    # ------------------------------------------------------------------
    # Index cache helpers
    # ------------------------------------------------------------------

    def _get_index_path(self):
        """Return the path of the JSON index cache file for this dataset.

        The cache filename includes the image extension so that regenerating
        a dataset with a different format (BMP vs PNG) automatically
        invalidates the old cache.
        """
        cache_dir = os.path.join(self.root, self.dataset_name, '.vsr_index')
        os.makedirs(cache_dir, exist_ok=True)
        ext_tag = self.img_ext.lstrip(".")  # e.g. "bmp" or "png"
        return os.path.join(cache_dir, f'{self.mode}_{self.size_key}_{ext_tag}.json')

    def _load_index(self):
        """
        Try to load a previously cached file index.

        The cache is valid when:
        * Schema version matches.
        * Stored GT file count equals the current count on disk.
        * GT and LR directory mtimes match (as a secondary freshness check
          for flat layouts where mtime is reliable).

        Returns:
            (gt_files, lr_paths) tuple on cache hit, or None on miss/error.
        """
        index_path = self._get_index_path()
        if not os.path.exists(index_path):
            return None
        try:
            with open(index_path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)

            if data.get('version') != _INDEX_CACHE_VERSION:
                return None

            # Extension must match current loader config
            if data.get('img_ext') != self.img_ext:
                return None

            # -- File-count based invalidation (works for bucket layout too) --
            # Re-scan GT directory and compare count
            cached_count = data.get('gt_file_count')
            if cached_count is None:
                return None
            current_gt_files = _collect_image_files(self.gt_dir, self.img_ext)
            if len(current_gt_files) != cached_count:
                return None  # files were added or removed

            # -- mtime based secondary check (optional, handles flat layout) --
            gt_mtime_cached = data.get('gt_dir_mtime')
            if gt_mtime_cached is not None and os.path.isdir(self.gt_dir):
                if abs(os.path.getmtime(self.gt_dir) - gt_mtime_cached) > _MTIME_TOLERANCE_SEC:
                    return None  # directory mtime changed

            active_lr_dir = self.lr_dir if self.lr_dir else self.patch_lr_dir
            lr_mtime_cached = data.get('lr_dir_mtime')
            if active_lr_dir and os.path.isdir(active_lr_dir) and lr_mtime_cached is not None:
                if abs(os.path.getmtime(active_lr_dir) - lr_mtime_cached) > _MTIME_TOLERANCE_SEC:
                    return None  # directory changed

            gt_files = data['gt_files']
            lr_paths = data['lr_paths']
            return gt_files, lr_paths

        except Exception:
            # Any read/parse error → treat as cache miss
            return None

    def _save_index(self, gt_files, lr_paths):
        """
        Persist the current file index to disk so the next startup is instant.

        Not called when ``validate_upfront=True`` because the stored result
        would then be filtered by image-level validation which is not
        reproducible purely from directory mtimes.
        """
        try:
            index_path = self._get_index_path()
            active_lr_dir = self.lr_dir if self.lr_dir else self.patch_lr_dir
            data = {
                'version': _INDEX_CACHE_VERSION,
                'img_ext': self.img_ext,
                'gt_dir': self.gt_dir,
                'lr_dir': active_lr_dir,
                'gt_file_count': len(gt_files),
                'gt_dir_mtime': os.path.getmtime(self.gt_dir) if os.path.isdir(self.gt_dir) else None,
                'lr_dir_mtime': os.path.getmtime(active_lr_dir) if active_lr_dir and os.path.isdir(active_lr_dir) else None,
                'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'gt_files': gt_files,
                'lr_paths': lr_paths,
            }
            with open(index_path, 'w', encoding='utf-8') as fh:
                json.dump(data, fh)
        except Exception:
            # Writing the cache is best-effort; never crash training
            pass

    def _invalidate_index(self):
        """Delete the on-disk index so the next startup triggers a fresh scan."""
        try:
            index_path = self._get_index_path()
            if os.path.exists(index_path):
                os.remove(index_path)
        except Exception:
            pass

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.gt_files)
    
    def get_file_info(self):
        """
        Get information about dataset files
        
        Returns:
            dict with file counts and paths
        """
        return {
            'mode': self.mode,
            'size_key': self.size_key,
            'dataset_name': self.dataset_name,
            'file_count': len(self.gt_files),
            'gt_dir': self.gt_dir,
            'lr_dir': self.lr_dir if self.lr_dir else self.patch_lr_dir
        }
    
    def check_for_new_files(self):
        """
        Check if files have been added to or removed from the dataset directories.

        Uses an internal ``_last_gt_scan_count`` counter so that *both* additions
        and deletions are detected on every periodic check.  On the very first
        call the counter is bootstrapped from the current on-disk GT count, so a
        permanently-missing LR file does not cause spurious reload loops.

        Returns:
            dict with:
                - has_new: bool  – True when the GT directory count changed
                - new_gt_count: int (total GT files in directory now)
                - current_loaded: int (files currently loaded)
                - new_files: int (difference vs. current_loaded; negative = deletions)
        """
        if not os.path.exists(self.gt_dir):
            self._last_gt_scan_count = 0
            return {
                'has_new': False,
                'new_gt_count': 0,
                'current_loaded': len(self.gt_files),
                'new_files': 0
            }

        # Count all GT image files (bucket-aware, extension-aware)
        all_gt_files = _collect_image_files(self.gt_dir, self.img_ext)
        new_gt_count = len(all_gt_files)
        current_loaded = len(self.gt_files)
        new_files = new_gt_count - current_loaded

        # Determine whether a reload is needed.
        # Compare against the last *scanned* GT count rather than the loaded count.
        last_gt_scan = getattr(self, '_last_gt_scan_count', None)
        if last_gt_scan is None:
            has_new = new_gt_count > current_loaded
        else:
            has_new = new_gt_count != last_gt_scan

        self._last_gt_scan_count = new_gt_count

        return {
            'has_new': has_new,
            'new_gt_count': new_gt_count,
            'current_loaded': current_loaded,
            'new_files': new_files
        }
    
    def reload_files(self):
        """
        Reload dataset files from disk — picks up new files added during training.
        
        This method is called when new files are detected in the dataset directories.
        It safely reloads the file list while training is running.
        
        Returns:
            dict with:
                - success: bool
                - files_before: int
                - files_after: int
                - new_files_loaded: int
        """
        with self.reload_lock:
            files_before = len(self.gt_files)
            
            try:
                if not os.path.exists(self.gt_dir):
                    return {
                        'success': False,
                        'files_before': files_before,
                        'files_after': files_before,
                        'new_files_loaded': 0,
                        'error': 'GT directory not found'
                    }
                
                # Get all GT files (bucket-aware, extension-aware)
                all_gt_files = _collect_image_files(self.gt_dir, self.img_ext)
                
                if not all_gt_files:
                    return {
                        'success': False,
                        'files_before': files_before,
                        'files_after': files_before,
                        'new_files_loaded': 0,
                        'error': f'No {self.img_ext} files found'
                    }

                # ------------------------------------------------------------------
                # For VAL mode: rebuild LR basename index from patches/LR_{n}frames
                # ------------------------------------------------------------------
                patch_lr_basename_index: dict = {}
                if self.mode == 'val' and self.patch_lr_dir:
                    patch_lr_files = _collect_image_files(self.patch_lr_dir, self.img_ext)
                    for rel_path in patch_lr_files:
                        bn = os.path.basename(rel_path)
                        lr_bucket = (
                            os.path.join(self.patch_lr_dir, os.path.dirname(rel_path))
                            if os.path.dirname(rel_path) else self.patch_lr_dir
                        )
                        patch_lr_basename_index.setdefault(bn, lr_bucket)
                
                # Build new file lists - only check LR file existence (fast reload)
                new_gt_files = []
                new_lr_paths = {}
                missing_lr_count = 0
                
                for gt_file in all_gt_files:
                    basename = os.path.basename(gt_file)
                    if self.mode == 'train' and self.lr_dir:
                        lr_bucket_dir = (
                            os.path.join(self.lr_dir, os.path.dirname(gt_file))
                            if os.path.dirname(gt_file) else self.lr_dir
                        )
                        lr_path = os.path.join(lr_bucket_dir, basename)
                        if os.path.exists(lr_path):
                            new_gt_files.append(gt_file)
                            new_lr_paths[gt_file] = lr_bucket_dir
                        else:
                            missing_lr_count += 1
                    elif self.mode == 'val':
                        lr_bucket_dir = patch_lr_basename_index.get(basename)
                        if lr_bucket_dir is not None:
                            new_gt_files.append(gt_file)
                            new_lr_paths[gt_file] = lr_bucket_dir
                        else:
                            missing_lr_count += 1
                    else:
                        missing_lr_count += 1
                
                # GT files without a matching LR file are silently skipped
                
                # Update the dataset atomically
                self.gt_files = new_gt_files
                self.lr_paths = new_lr_paths
                
                # Persist updated index
                self._save_index(self.gt_files, self.lr_paths)
                
                files_after = len(self.gt_files)
                new_files_loaded = files_after - files_before
                
                return {
                    'success': True,
                    'files_before': files_before,
                    'files_after': files_after,
                    'new_files_loaded': new_files_loaded
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'files_before': files_before,
                    'files_after': files_before,
                    'new_files_loaded': 0,
                    'error': str(e)
                }
    
    def __getitem__(self, idx):
        """
        Load and process a single sample.

        Returns:
            lr_stack: [n_frames, 3, H_lr, W_lr] — n_frames LR frames
            gt: [3, H_gt, W_gt] — GT frame
            gt_file: relative path string (for logging)
        """
        # Try to load the current index, but handle errors gracefully
        max_attempts = 3  # Try current index, then 2 random fallbacks
        
        for attempt in range(max_attempts):
            try:
                # Use current index on first attempt, random on subsequent attempts
                current_idx = idx if attempt == 0 else random.randint(0, len(self.gt_files) - 1)
                
                gt_file = self.gt_files[current_idx]
                gt_path = os.path.join(self.gt_dir, gt_file)
                # lr_paths value is the full directory that contains the LR file.
                # For bucket layout gt_file = "0000/foo.bmp", lr_paths[gt_file]
                # already points to the matching bucket dir, so we use basename.
                lr_dir = self.lr_paths[gt_file]
                lr_path = os.path.join(lr_dir, os.path.basename(gt_file))
                
                # Load images
                gt = cv2.imread(gt_path)
                lr = cv2.imread(lr_path)
                
                # Validate
                if gt is None or lr is None:
                    raise ValueError(f"Failed to load images: GT={gt_path}, LR={lr_path}")
                
                # Validate dimensions match expected size_key
                expected_gt_shapes = {
                    '720': (720, 720, 3),
                    '540': (540, 540, 3),
                    '720_169': (405, 720, 3)
                }
                expected_gt_shape = expected_gt_shapes.get(self.size_key)
                
                if expected_gt_shape and gt.shape != expected_gt_shape:
                    raise ValueError(f"Invalid GT dimensions {gt.shape}, expected {expected_gt_shape} for size_key '{self.size_key}': {gt_file}")
                
                # Validate LR can be split into n_frames
                if lr.shape[0] % self.n_frames != 0:
                    raise ValueError(f"LR height {lr.shape[0]} not divisible by {self.n_frames}: {gt_file}")
                
                # Convert BGR to RGB
                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
                lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
                
                # Split LR into n_frames (stacked vertically: H_total = H_frame * n_frames)
                lr_height_total = lr.shape[0]
                lr_height_per_frame = lr_height_total // self.n_frames
                
                lr_frames = []
                for i in range(self.n_frames):
                    # Slice vertically (by height dimension)
                    frame = lr[i*lr_height_per_frame:(i+1)*lr_height_per_frame, :, :]
                    lr_frames.append(frame)
                
                # Augmentation is permanently disabled (self.augment is always False).
                # With 350k+ diverse scenes the regularisation gain is negligible.
                
                # Convert to tensors and normalize to [0, 1]
                gt = torch.from_numpy(gt).permute(2, 0, 1).float() / 255.0
                lr_stack = torch.stack([
                    torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
                    for f in lr_frames
                ])
                
                return lr_stack, gt, gt_file
                
            except Exception as e:
                # Log the error but try to recover
                if attempt == 0:
                    print(f"\n⚠️  ERROR loading sample {idx} ({gt_file}): {str(e)}")
                    print(f"   Attempting to use random fallback sample...")
                elif attempt < max_attempts - 1:
                    print(f"   Fallback attempt {attempt} failed, trying another...")
                else:
                    # Last attempt failed - this is critical
                    print(f"\n❌ CRITICAL: All {max_attempts} attempts to load a valid sample failed!")
                    print(f"   Last error: {str(e)}")
                    print(f"   Dataset may have serious issues. Please check your data!")
                    raise RuntimeError(f"Failed to load any valid sample after {max_attempts} attempts. Last file: {gt_file}")
        
        # Should never reach here due to raise above, but just in case
        raise RuntimeError(f"Unexpected error in __getitem__ for index {idx}")
