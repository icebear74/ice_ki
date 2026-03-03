"""
VSRDataset - Video Super-Resolution Dataset Loader

Loads VSR training data with new dataset structure:
- Dataset structure: root/dataset_name/patches/{size_key}/GT/ and LR/
- Validation structure: root/dataset_name/val/{size_key}/GT/ (GT) + patches/{size_key}/LR/ (LR)
- GT images: Variable size based on size_key (e.g., 720×720 for '720', 540×540 for '540')
- LR stack: 7 frames stacked vertically (e.g., H*7 x W x 3)
- Supported size_keys: '720', '540', '720_169' (16:9 aspect ratio variants)
"""

import os
import cv2
import torch
import random
import numpy as np
import threading
from torch.utils.data import Dataset


class VSRDataset(Dataset):
    """
    VSR Dataset for training and validation
    
    Args:
        root: Root directory (e.g., /mnt/data/training/datasetNeu)
        dataset_name: Dataset name (e.g., 'master')
        size_key: Size variant ('720', '540', or '720_169')
        mode: 'train' or 'val'
        augment: Whether to apply augmentations (flip, rotate)
        paths_config: Optional dict with path patterns:
            - train_gt: Pattern for training GT (default: 'patches/{size_key}/GT')
            - train_lr: Pattern for training LR (default: 'patches/{size_key}/LR_7frames')
            - val_gt: Pattern for validation GT (default: 'val/{size_key}/GT')
            - val_lr: Pattern for validation LR (default: 'patches/{size_key}/LR_7frames')
    """
    
    def __init__(self, root, dataset_name='master', size_key='720', mode='train', augment=True, paths_config=None, validate_upfront=False):
        self.root = root
        self.dataset_name = dataset_name
        self.size_key = size_key
        self.mode = mode
        self.augment = augment and (mode == 'train')
        self.validate_upfront = validate_upfront
        
        # Thread lock for safe reloading during training
        self.reload_lock = threading.Lock()
        
        # Path patterns (configurable or defaults)
        if paths_config is None:
            paths_config = {}
        self.train_gt_pattern = paths_config.get('train_gt', 'patches/{size_key}/GT')
        self.train_lr_pattern = paths_config.get('train_lr', 'patches/{size_key}/LR_7frames')
        self.val_gt_pattern = paths_config.get('val_gt', 'val/{size_key}/GT')
        self.val_lr_pattern = paths_config.get('val_lr', 'patches/{size_key}/LR_7frames')
        
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
            # Validation: use val_gt_pattern and val_lr_pattern
            gt_path = self.val_gt_pattern.replace('{size_key}', size_key)
            lr_path = self.val_lr_pattern.replace('{size_key}', size_key)
            
            self.gt_dir = os.path.join(dataset_path, gt_path)
            self.lr_dir = None  # Will use patch_lr_dir
            self.patch_lr_dir = os.path.join(dataset_path, lr_path)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'val'")
        
        # Get all GT files
        if not os.path.exists(self.gt_dir):
            raise ValueError(f"GT directory not found: {self.gt_dir}")
        
        all_gt_files = sorted([f for f in os.listdir(self.gt_dir) if f.endswith('.png')])
        
        if not all_gt_files:
            raise ValueError(f"No PNG files found in {self.gt_dir}")
        
        # Filter to only keep GT files that have corresponding LR files
        # For Val mode, check both Val/LR and Patches/LR (like original)
        self.gt_files = []
        self.lr_paths = {}  # Map filename to actual LR directory
        skipped_files = []
        matched_val_lr = 0
        matched_patches_lr = 0
        
        # Expected shapes for validation
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
            
            # For training, check lr_dir. For validation, always use patch_lr_dir
            if self.lr_dir:
                lr_path = os.path.join(self.lr_dir, gt_file)
                
                if os.path.exists(lr_path):
                    # Validate file dimensions before adding (optional, controlled by validate_upfront)
                    if self.validate_upfront and not self._validate_file_dimensions(gt_file, self.gt_dir, self.lr_dir, expected_gt_shape, invalid_dimension_files):
                        # File invalid, skip it
                        pass
                    else:
                        # File valid or validation skipped - add it
                        self.gt_files.append(gt_file)
                        self.lr_paths[gt_file] = self.lr_dir
                        matched_val_lr += 1
                elif mode == 'val' and self.patch_lr_dir:
                    # For validation, fallback to patches/LR
                    patch_lr_path = os.path.join(self.patch_lr_dir, gt_file)
                    if os.path.exists(patch_lr_path):
                        if self.validate_upfront and not self._validate_file_dimensions(gt_file, self.gt_dir, self.patch_lr_dir, expected_gt_shape, invalid_dimension_files):
                            # File invalid, skip it
                            pass
                        else:
                            # File valid or validation skipped - add it
                            self.gt_files.append(gt_file)
                            self.lr_paths[gt_file] = self.patch_lr_dir
                            matched_patches_lr += 1
                    else:
                        skipped_files.append(gt_file)
                else:
                    skipped_files.append(gt_file)
            elif mode == 'val' and self.patch_lr_dir:
                # For validation with no val LR dir, always use patches
                patch_lr_path = os.path.join(self.patch_lr_dir, gt_file)
                if os.path.exists(patch_lr_path):
                    if self.validate_upfront and not self._validate_file_dimensions(gt_file, self.gt_dir, self.patch_lr_dir, expected_gt_shape, invalid_dimension_files):
                        # File invalid, skip it
                        pass
                    else:
                        # File valid or validation skipped - add it
                        self.gt_files.append(gt_file)
                        self.lr_paths[gt_file] = self.patch_lr_dir
                        matched_patches_lr += 1
                else:
                    skipped_files.append(gt_file)
            else:
                skipped_files.append(gt_file)
        
        # Show detailed statistics for val mode
        if mode == 'val':
            print("\n" + "="*60)
            print(f"📂 VALIDATION DATASET LOADING ({size_key})")
            print("="*60)
            print(f"  GT files found:           {len(all_gt_files)}")
            print(f"  Matched in val/LR:        {matched_val_lr}")
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
                print(f"     {self.lr_dir}")
                if self.patch_lr_dir:
                    print(f"  OR {self.patch_lr_dir}")
                print()
        elif skipped_files or invalid_dimension_files:
            # For training mode, just show count
            if skipped_files:
                print(f"\n⚠️  Skipped {len(skipped_files)} GT files without matching LR files in {mode}")
            if invalid_dimension_files and self.validate_upfront:
                print(f"⚠️  Skipped {len(invalid_dimension_files)} files with invalid dimensions in {mode} (size_key={size_key})")
            print()
        
        if not self.gt_files:
            raise ValueError(f"No valid GT-LR pairs found in {self.gt_dir} and {self.lr_dir}")
        
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
            gt_path = os.path.join(gt_dir, gt_file)
            lr_path = os.path.join(lr_dir, gt_file)
            
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
            
            # Validate LR can be split into 7 frames
            if lr.shape[0] % 7 != 0:
                invalid_list.append((gt_file, f"LR height {lr.shape[0]} not divisible by 7"))
                return False
            
            return True
            
        except Exception as e:
            invalid_list.append((gt_file, f"Validation error: {str(e)}"))
            return False
    
    def _validate_samples(self):
        """Validate dataset integrity by checking a few samples"""
        samples_to_check = min(5, len(self.gt_files))
        
        issues_found = []
        
        # Expected shapes based on size_key
        expected_gt_shapes = {
            '720': (720, 720, 3),      # 720×720 square patches
            '540': (540, 540, 3),      # 540×540 square patches
            '720_169': (405, 720, 3)   # 720×405 (16:9 aspect ratio)
        }
        
        expected_gt_shape = expected_gt_shapes.get(self.size_key)
        if not expected_gt_shape:
            print(f"\n⚠️  Unknown size_key '{self.size_key}', skipping shape validation")
            return
        
        # LR should be height*7, same width (7 frames stacked vertically)
        expected_lr_width = expected_gt_shape[1] // 3  # 3x downscale
        # Calculate LR height: (GT_height / scale) * n_frames
        # Mathematically equivalent: (GT_height * 7) / 3 for precision
        expected_lr_height = (expected_gt_shape[0] * 7) // 3  # 7 frames stacked vertically, downscaled 3x
        expected_lr_shape = (expected_lr_height, expected_lr_width, 3)
        
        for i in range(samples_to_check):
            gt_file = self.gt_files[i]
            gt_path = os.path.join(self.gt_dir, gt_file)
            # Use the correct LR directory from lr_paths mapping
            lr_dir = self.lr_paths[gt_file]
            lr_path = os.path.join(lr_dir, gt_file)
            
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
        Check if new files have been added to the dataset directories
        
        Returns:
            dict with:
                - has_new: bool
                - new_gt_count: int (total GT files in directory now)
                - current_loaded: int (files currently loaded)
                - new_files: int (difference)
        """
        if not os.path.exists(self.gt_dir):
            return {
                'has_new': False,
                'new_gt_count': 0,
                'current_loaded': len(self.gt_files),
                'new_files': 0
            }
        
        # Count all GT files in directory
        all_gt_files = sorted([f for f in os.listdir(self.gt_dir) if f.endswith('.png')])
        new_gt_count = len(all_gt_files)
        current_loaded = len(self.gt_files)
        new_files = new_gt_count - current_loaded
        
        return {
            'has_new': new_files > 0,
            'new_gt_count': new_gt_count,
            'current_loaded': current_loaded,
            'new_files': new_files
        }
    
    def reload_files(self):
        """
        Reload dataset files from disk - picks up new files added during training
        
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
                
                # Get all GT files
                all_gt_files = sorted([f for f in os.listdir(self.gt_dir) if f.endswith('.png')])
                
                if not all_gt_files:
                    return {
                        'success': False,
                        'files_before': files_before,
                        'files_after': files_before,
                        'new_files_loaded': 0,
                        'error': 'No PNG files found'
                    }
                
                # Build new file lists - only check LR file existence (fast reload)
                new_gt_files = []
                new_lr_paths = {}
                missing_lr_count = 0
                
                for gt_file in all_gt_files:
                    # For training, check lr_dir. For validation, use patch_lr_dir
                    if self.lr_dir:
                        lr_path = os.path.join(self.lr_dir, gt_file)
                        
                        if os.path.exists(lr_path):
                            # Only check file existence - no dimension validation during reload
                            new_gt_files.append(gt_file)
                            new_lr_paths[gt_file] = self.lr_dir
                        elif self.mode == 'val' and self.patch_lr_dir:
                            # For validation, fallback to patches/LR
                            patch_lr_path = os.path.join(self.patch_lr_dir, gt_file)
                            if os.path.exists(patch_lr_path):
                                new_gt_files.append(gt_file)
                                new_lr_paths[gt_file] = self.patch_lr_dir
                            else:
                                missing_lr_count += 1
                        else:
                            missing_lr_count += 1
                    elif self.mode == 'val' and self.patch_lr_dir:
                        # For validation with no val LR dir, always use patches
                        patch_lr_path = os.path.join(self.patch_lr_dir, gt_file)
                        if os.path.exists(patch_lr_path):
                            new_gt_files.append(gt_file)
                            new_lr_paths[gt_file] = self.patch_lr_dir
                        else:
                            missing_lr_count += 1
                    else:
                        missing_lr_count += 1
                
                # Report skipped files (only if any were skipped)
                if missing_lr_count > 0:
                    print(f"\n⚠️  Reload: Skipped {missing_lr_count} GT files without matching LR files ({self.mode}, size_key={self.size_key})")
                
                # Update the dataset atomically
                self.gt_files = new_gt_files
                self.lr_paths = new_lr_paths
                
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
        Load and process a single sample
        
        Returns:
            lr_stack: [7, 3, H, W] - 7 LR frames
            gt: [3, H*3, W*3] - GT frame (3x upscale)
        """
        # Try to load the current index, but handle errors gracefully
        max_attempts = 3  # Try current index, then 2 random fallbacks
        
        for attempt in range(max_attempts):
            try:
                # Use current index on first attempt, random on subsequent attempts
                current_idx = idx if attempt == 0 else random.randint(0, len(self.gt_files) - 1)
                
                gt_file = self.gt_files[current_idx]
                gt_path = os.path.join(self.gt_dir, gt_file)
                # Use the correct LR directory from lr_paths mapping
                lr_dir = self.lr_paths[gt_file]
                lr_path = os.path.join(lr_dir, gt_file)
                
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
                
                # Validate LR can be split into 7 frames
                if lr.shape[0] % 7 != 0:
                    raise ValueError(f"LR height {lr.shape[0]} not divisible by 7: {gt_file}")
                
                # Convert BGR to RGB
                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
                lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
                
                # Split LR into 7 frames (stacked vertically: H_total = H_frame * 7)
                lr_height_total = lr.shape[0]
                lr_width = lr.shape[1]
                lr_height_per_frame = lr_height_total // 7
                
                lr_frames = []
                for i in range(7):
                    # Slice vertically (by height dimension)
                    frame = lr[i*lr_height_per_frame:(i+1)*lr_height_per_frame, :, :]
                    lr_frames.append(frame)
                
                # Apply augmentations (only for training)
                if self.augment:
                    # Random horizontal flip
                    if random.random() > 0.5:
                        gt = np.flip(gt, axis=1).copy()
                        lr_frames = [np.flip(f, axis=1).copy() for f in lr_frames]
                    
                    # Random vertical flip
                    if random.random() > 0.5:
                        gt = np.flip(gt, axis=0).copy()
                        lr_frames = [np.flip(f, axis=0).copy() for f in lr_frames]
                    
                    # Random rotation (0, 90, 180, 270)
                    k = random.randint(0, 3)
                    if k > 0:
                        gt = np.rot90(gt, k).copy()
                        lr_frames = [np.rot90(f, k).copy() for f in lr_frames]
                
                # Convert to tensors and normalize to [0, 1]
                gt = torch.from_numpy(gt).permute(2, 0, 1).float() / 255.0
                lr_stack = torch.stack([
                    torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
                    for f in lr_frames
                ])
                
                return lr_stack, gt
                
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
