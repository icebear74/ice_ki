"""
VSRDataset - Video Super-Resolution Dataset Loader

Loads VSR training data:
- GT images: 540x540 (single frame)
- LR stack: 180x900 (5 frames stacked vertically)
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
        dataset_root: Root directory of dataset
        mode: 'Patches' (training) or 'Val' (validation)
        augment: Whether to apply augmentations (flip, rotate)
    """
    
    def __init__(self, dataset_root, mode='Patches', augment=True):
        self.dataset_root = dataset_root
        self.mode = mode
        self.augment = augment and (mode == 'Patches')
        
        # Build paths
        self.gt_dir = os.path.join(dataset_root, mode, 'GT')
        self.lr_dir = os.path.join(dataset_root, mode, 'LR')
        # For validation, also check Patches/LR as fallback (like original train.py)
        self.patch_lr_dir = os.path.join(dataset_root, 'Patches', 'LR')
        
        # Get all GT files
        all_gt_files = sorted([f for f in os.listdir(self.gt_dir) if f.endswith('.png')])
        
        if not all_gt_files:
            raise ValueError(f"No PNG files found in {self.gt_dir}")
        
        # Filter to only keep GT files that have corresponding LR files
        # For Val mode, check both Val/LR and Patches/LR (like original)
        self.gt_files = []
        self.lr_paths = {}  # Map filename to actual LR directory
        skipped_files = []
        
        for gt_file in all_gt_files:
            lr_path = os.path.join(self.lr_dir, gt_file)
            patch_lr_path = os.path.join(self.patch_lr_dir, gt_file)
            
            if os.path.exists(lr_path):
                self.gt_files.append(gt_file)
                self.lr_paths[gt_file] = self.lr_dir
            elif mode == 'Val' and os.path.exists(patch_lr_path):
                # For validation, fallback to Patches/LR
                self.gt_files.append(gt_file)
                self.lr_paths[gt_file] = self.patch_lr_dir
            else:
                skipped_files.append(gt_file)
        
        # Report skipped files
        if skipped_files:
            print(f"\n⚠️  Skipped {len(skipped_files)} GT files without matching LR files in {mode}:")
            for i, f in enumerate(skipped_files[:10]):  # Show first 10
                print(f"  - {f}")
            if len(skipped_files) > 10:
                print(f"  ... and {len(skipped_files) - 10} more")
            print()
        
        if not self.gt_files:
            raise ValueError(f"No valid GT-LR pairs found in {self.gt_dir} and {self.lr_dir}")
        
        # Validate a few samples
        self._validate_samples()
    
    def _validate_samples(self):
        """Validate dataset integrity by checking a few samples"""
        samples_to_check = min(5, len(self.gt_files))
        
        issues_found = []
        
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
            
            if gt.shape != (540, 540, 3):
                issues_found.append(f"Invalid GT shape {gt.shape}, expected (540, 540, 3): {gt_path}")
            if lr.shape != (900, 180, 3):
                issues_found.append(f"Invalid LR shape {lr.shape}, expected (900, 180, 3): {lr_path}")
        
        # Report issues as warnings instead of errors
        if issues_found:
            print(f"\n⚠️  Dataset validation warnings in {self.mode}:")
            for issue in issues_found:
                print(f"  - {issue}")
            print()
    
    def __len__(self):
        return len(self.gt_files)
    
    def __getitem__(self, idx):
        """
        Load and process a single sample
        
        Returns:
            lr_stack: [5, 3, 180, 180] - 5 LR frames
            gt: [3, 540, 540] - GT frame
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
        
        # Split LR into 5 frames (stacked vertically: 900 = 5 * 180)
        lr_frames = []
        for i in range(5):
            frame = lr[i*180:(i+1)*180, :, :]  # Extract 180x180 frame
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
