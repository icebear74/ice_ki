"""
ValidationDataset - Validation with On-the-Fly LR Generation

Loads GT images from Val/GT/ and generates LR frames on-the-fly
with slight frame variations to simulate camera motion.
"""

import os
import cv2
import torch
import numpy as np
import glob
from torch.utils.data import Dataset


class ValidationDataset(Dataset):
    """
    Validation dataset
    - Loads GT images from Val/GT/
    - Generates LR on-the-fly with slight frame variations
    
    Args:
        config: Configuration object with DATA section
    """
    
    def __init__(self, config):
        self.category = config.DATA.category
        self.lr_version = config.DATA.lr_version
        self.num_frames = 7 if self.lr_version == "7frames" else 5
        
        # Determine val path
        category_paths = {
            'general': 'Universal/Mastermodell/Learn',
            'space': 'Space/SpaceModel/Learn',
            'toon': 'Toon/ToonModel/Learn'
        }
        
        base_path = os.path.join(
            config.DATA.data_root,
            category_paths[self.category]
        )
        
        self.val_gt_dir = os.path.join(base_path, "Val", "GT")
        
        if not os.path.exists(self.val_gt_dir):
            raise ValueError(f"Validation GT directory not found: {self.val_gt_dir}")
        
        self.gt_images = sorted(glob.glob(os.path.join(self.val_gt_dir, "*.png")))
        
        if not self.gt_images:
            raise ValueError(f"No PNG images found in {self.val_gt_dir}")
        
        print(f"\n{'='*80}")
        print(f"📊 VALIDATION DATASET LOADED")
        print(f"{'='*80}")
        print(f"   Category:          {self.category}")
        print(f"   LR Version:        {self.lr_version} ({self.num_frames} frames)")
        print(f"   Val images:        {len(self.gt_images):,}")
        print(f"   Source:            {self.val_gt_dir}")
        print(f"{'='*80}\n")
    
    def __len__(self):
        return len(self.gt_images)
    
    def __getitem__(self, idx):
        """
        Load GT and generate LR frames on-the-fly
        
        Returns:
            dict with:
                'lr': Tensor [T, 3, H, W] where T = 5 or 7
                'gt': Tensor [3, H, W]
                'filename': str filename
        """
        # Load GT
        gt = cv2.imread(self.gt_images[idx])
        if gt is None:
            raise ValueError(f"Failed to load GT: {self.gt_images[idx]}")
        
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        h, w = gt.shape[:2]
        
        # Generate LR frames on-the-fly (3x downscale)
        lr_h, lr_w = h // 3, w // 3
        lr_middle = cv2.resize(gt, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
        
        # Create frames with slight shifts (simulate camera motion)
        lr_frames = []
        
        if self.num_frames == 5:
            shifts = [-1.0, -0.5, 0.0, 0.5, 1.0]
        else:  # 7 frames
            shifts = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
        
        for shift in shifts:
            # Create affine transformation matrix for horizontal shift
            M = np.float32([[1, 0, shift], [0, 1, 0]])
            shifted = cv2.warpAffine(
                lr_middle, M, (lr_w, lr_h), 
                borderMode=cv2.BORDER_REFLECT
            )
            lr_frames.append(shifted)
        
        # Convert to tensors and normalize to [0, 1]
        lr_tensor = torch.stack([
            torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
            for f in lr_frames
        ])
        
        gt_tensor = torch.from_numpy(gt).permute(2, 0, 1).float() / 255.0
        
        return {
            'lr': lr_tensor,      # (5 or 7, 3, H/3, W/3)
            'gt': gt_tensor,      # (3, H, W)
            'filename': os.path.basename(self.gt_images[idx])
        }
