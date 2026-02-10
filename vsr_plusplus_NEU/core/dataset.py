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
    """
    
    def __init__(self, root, dataset_name='master', size_key='720', mode='train', augment=True):
        self.root = root
        self.dataset_name = dataset_name
        self.size_key = size_key
        self.mode = mode
        self.augment = augment and (mode == 'train')
        
        # Build paths based on mode
        dataset_path = os.path.join(root, dataset_name)
        
        if mode == 'train':
            # Training: root/dataset_name/patches/size_key/GT and LR_7frames
            patches_path = os.path.join(dataset_path, 'patches', size_key)
            self.gt_dir = os.path.join(patches_path, 'GT')
            self.lr_dir = os.path.join(patches_path, 'LR_7frames')
            self.patch_lr_dir = None  # Not needed for training
        elif mode == 'val':
            # Validation: GT from Val/GT/size_key, LR from patches/size_key/LR_7frames
            val_gt_path = os.path.join(dataset_path, 'Val', 'GT', size_key)
            self.gt_dir = val_gt_path
            # LR always comes from patches (no separate val LR directory)
            self.lr_dir = None  # Will use patch_lr_dir
            # Fallback to patches/LR_7frames for validation
            self.patch_lr_dir = os.path.join(dataset_path, 'patches', size_key, 'LR_7frames')
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
        
        for gt_file in all_gt_files:
            # For training, check lr_dir. For validation, always use patch_lr_dir
            if self.lr_dir:
                lr_path = os.path.join(self.lr_dir, gt_file)
                
                if os.path.exists(lr_path):
                    self.gt_files.append(gt_file)
                    self.lr_paths[gt_file] = self.lr_dir
                    matched_val_lr += 1
                elif mode == 'val' and self.patch_lr_dir:
                    # For validation, fallback to patches/LR
                    patch_lr_path = os.path.join(self.patch_lr_dir, gt_file)
                    if os.path.exists(patch_lr_path):
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
            print(f"  Final samples loaded:     {len(self.gt_files)}")
            print("="*60)
            
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
        elif skipped_files:
            # For training mode, just show count
            print(f"\n⚠️  Skipped {len(skipped_files)} GT files without matching LR files in {mode}")
            print()
        
        if not self.gt_files:
            raise ValueError(f"No valid GT-LR pairs found in {self.gt_dir} and {self.lr_dir}")
        
        # Validate a few samples
        self._validate_samples()
    
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
    
    def __getitem__(self, idx):
        """
        Load and process a single sample
        
        Returns:
            lr_stack: [7, 3, H, W] - 7 LR frames
            gt: [3, H*3, W*3] - GT frame (3x upscale)
        """
        gt_file = self.gt_files[idx]
        gt_path = os.path.join(self.gt_dir, gt_file)
        # Use the correct LR directory from lr_paths mapping
        lr_dir = self.lr_paths[gt_file]
        lr_path = os.path.join(lr_dir, gt_file)
        
        # Load images
        gt = cv2.imread(gt_path)
        lr = cv2.imread(lr_path)
        
        # Validate
        if gt is None or lr is None:
            raise ValueError(f"Failed to load images for index {idx}")
        
        # Convert BGR to RGB
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        
        # Split LR into 7 frames (stacked horizontally: W_total = W_frame * 7)
        lr_height = lr.shape[0]
        lr_width_total = lr.shape[1]
        lr_width_per_frame = lr_width_total // 7
        
        lr_frames = []
        for i in range(7):
            frame = lr[:, i*lr_width_per_frame:(i+1)*lr_width_per_frame, :]
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
