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
        
        # Get file list
        self.gt_files = sorted([f for f in os.listdir(self.gt_dir) if f.endswith('.png')])
        
        if not self.gt_files:
            raise ValueError(f"No PNG files found in {self.gt_dir}")
        
        # Validate a few samples
        self._validate_samples()
    
    def _validate_samples(self):
        """Validate dataset integrity by checking a few samples"""
        samples_to_check = min(5, len(self.gt_files))
        
        for i in range(samples_to_check):
            gt_path = os.path.join(self.gt_dir, self.gt_files[i])
            lr_path = os.path.join(self.lr_dir, self.gt_files[i])
            
            # Check if files exist
            if not os.path.exists(gt_path):
                raise FileNotFoundError(f"GT file not found: {gt_path}")
            if not os.path.exists(lr_path):
                raise FileNotFoundError(f"LR file not found: {lr_path}")
            
            # Load and validate shapes
            gt = cv2.imread(gt_path)
            lr = cv2.imread(lr_path)
            
            if gt is None:
                raise ValueError(f"Corrupted GT image: {gt_path}")
            if lr is None:
                raise ValueError(f"Corrupted LR image: {lr_path}")
            
            if gt.shape != (540, 540, 3):
                raise ValueError(f"Invalid GT shape {gt.shape}, expected (540, 540, 3): {gt_path}")
            if lr.shape != (900, 180, 3):
                raise ValueError(f"Invalid LR shape {lr.shape}, expected (900, 180, 3): {lr_path}")
    
    def __len__(self):
        return len(self.gt_files)
    
    def __getitem__(self, idx):
        """
        Load and process a single sample
        
        Returns:
            lr_stack: [5, 3, 180, 180] - 5 LR frames
            gt: [3, 540, 540] - GT frame
        """
        gt_path = os.path.join(self.gt_dir, self.gt_files[idx])
        lr_path = os.path.join(self.lr_dir, self.gt_files[idx])
        
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
