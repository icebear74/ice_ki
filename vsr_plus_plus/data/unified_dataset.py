"""
MultiFormatMultiCategoryDataset - Unified Dataset Loader

Loads patches from multiple formats and categories.
Supports both 5-frame and 7-frame LR versions.
"""

import os
import cv2
import torch
import random
import numpy as np
import glob
from torch.utils.data import Dataset


class MultiFormatMultiCategoryDataset(Dataset):
    """
    Loads patches from multiple formats and categories
    Supports both 5-frame and 7-frame LR versions
    
    Args:
        config: Configuration object with DATA section
    """
    
    def __init__(self, config):
        self.category = config.DATA.category
        self.lr_version = config.DATA.lr_version
        self.data_root = config.DATA.data_root
        self.formats = config.DATA.formats
        self.format_weights = config.DATA.format_weights
        
        # Determine category path
        self.category_paths = {
            'general': 'Universal/Mastermodell/Learn',
            'space': 'Space/SpaceModel/Learn',
            'toon': 'Toon/ToonModel/Learn'
        }
        
        self.base_path = os.path.join(
            self.data_root, 
            self.category_paths[self.category]
        )
        
        # Expected LR frames
        self.num_frames = 7 if self.lr_version == "7frames" else 5
        
        # Scan all formats
        self.image_pairs = []
        self._scan_datasets()
        
        print(f"\n{'='*80}")
        print(f"📊 DATASET LOADED")
        print(f"{'='*80}")
        print(f"   Category:          {self.category}")
        print(f"   LR Version:        {self.lr_version} ({self.num_frames} frames)")
        print(f"   Total samples:     {len(self.image_pairs):,}")
        print(f"   {'─'*76}")
        for fmt in self.formats:
            count = sum(1 for p in self.image_pairs if p['format'] == fmt)
            print(f"   - {fmt:20s}: {count:,} samples")
        print(f"{'='*80}\n")
    
    def _scan_datasets(self):
        """Scan all format directories for GT + LR pairs"""
        
        # Directory name mapping
        format_dir_map = {
            "small_540": "Patches",
            "medium_169": "Patches_Medium169",
            "large_720": "Patches_Large",
            "xlarge_1440": "Patches_XLarge169",
            "fullhd_1920": "Patches_FullHD"
        }
        
        for format_name in self.formats:
            # Determine directory name
            dir_name = format_dir_map.get(format_name)
            
            if dir_name is None:
                print(f"⚠️  Warning: Unknown format {format_name}, skipping")
                continue
            
            gt_dir = os.path.join(self.base_path, dir_name, "GT")
            lr_dir = os.path.join(self.base_path, dir_name, f"LR_{self.lr_version}")
            
            if not os.path.exists(gt_dir):
                print(f"⚠️  Warning: GT directory not found for {format_name}: {gt_dir}")
                continue
                
            if not os.path.exists(lr_dir):
                print(f"⚠️  Warning: LR directory not found for {format_name}: {lr_dir}")
                continue
            
            # Find all GT images
            gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
            
            if not gt_files:
                print(f"⚠️  Warning: No PNG files found in {gt_dir}")
                continue
            
            matched_count = 0
            for gt_path in gt_files:
                filename = os.path.basename(gt_path)
                lr_path = os.path.join(lr_dir, filename)
                
                if os.path.exists(lr_path):
                    self.image_pairs.append({
                        'gt': gt_path,
                        'lr': lr_path,
                        'format': format_name
                    })
                    matched_count += 1
            
            if matched_count < len(gt_files):
                print(f"⚠️  Warning: Only {matched_count}/{len(gt_files)} GT files have matching LR for {format_name}")
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        """
        Load and process a single sample
        
        Returns:
            dict with:
                'lr': Tensor [T, 3, H, W] where T = 5 or 7
                'gt': Tensor [3, H, W]
                'format': str format name
        """
        pair = self.image_pairs[idx]
        
        # Load GT
        gt = cv2.imread(pair['gt'])
        if gt is None:
            raise ValueError(f"Failed to load GT: {pair['gt']}")
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        
        # Load LR stack
        lr_stacked = cv2.imread(pair['lr'])
        if lr_stacked is None:
            raise ValueError(f"Failed to load LR: {pair['lr']}")
        lr_stacked = cv2.cvtColor(lr_stacked, cv2.COLOR_BGR2RGB)
        
        # Split into individual frames (stacked vertically)
        lr_h = lr_stacked.shape[0] // self.num_frames
        lr_frames = [
            lr_stacked[i*lr_h:(i+1)*lr_h, :, :] 
            for i in range(self.num_frames)
        ]
        
        # Data augmentation (random flip/rotation)
        if random.random() > 0.5:
            # Horizontal flip
            lr_frames = [np.fliplr(f).copy() for f in lr_frames]
            gt = np.fliplr(gt).copy()
        
        if random.random() > 0.5:
            # Vertical flip
            lr_frames = [np.flipud(f).copy() for f in lr_frames]
            gt = np.flipud(gt).copy()
        
        # Random rotation (0, 90, 180, 270)
        k = random.randint(0, 3)
        if k > 0:
            lr_frames = [np.rot90(f, k).copy() for f in lr_frames]
            gt = np.rot90(gt, k).copy()
        
        # Convert to tensors and normalize to [0, 1]
        lr_tensor = torch.stack([
            torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
            for f in lr_frames
        ])
        
        gt_tensor = torch.from_numpy(gt).permute(2, 0, 1).float() / 255.0
        
        return {
            'lr': lr_tensor,      # (5 or 7, 3, H, W)
            'gt': gt_tensor,      # (3, H, W)
            'format': pair['format']
        }
