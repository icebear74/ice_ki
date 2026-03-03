"""
Multi-Size DataLoader with Grouped Sampling

Enables training on multiple resolution variants simultaneously:
- Samples from different resolution datasets based on distribution weights
- Groups samples by size to maintain batch consistency
- Supports custom batch sizes per resolution
"""

import torch
from torch.utils.data import Sampler
import random
import numpy as np


class SizeGroupedSampler(Sampler):
    """
    Sampler that yields batches grouped by size key.
    
    Samples from different size groups proportionally to their file counts.
    
    IMPORTANT: Dataset extraction is pre-weighted, so files on disk already reflect
    the desired distribution. This sampler simply samples proportionally to the
    number of files in each size category (no additional weighting needed).
    
    Args:
        datasets_dict: Dict mapping size_key to dataset
                      Example: {'720': dataset1, '540': dataset2, '720_169': dataset3}
        size_distribution: Dict mapping size_key to probability weight
                          (ONLY used to determine which sizes to load, NOT for sampling)
                          Example: {'720': 0.4, '540': 0.4, '720_169': 0.2}
        batch_sizes: Dict mapping size_key to batch size
                    Example: {'720': 1, '540': 1, '720_169': 1}
        shuffle: Whether to shuffle indices within each size group
    
    Note: size_distribution is now only used to filter active sizes.
          Actual sampling is proportional to file counts (files are pre-weighted during extraction).
    """
    
    def __init__(self, datasets_dict, size_distribution, batch_sizes, shuffle=True):
        self.datasets_dict = datasets_dict
        self.size_distribution = size_distribution
        self.batch_sizes = batch_sizes
        self.shuffle = shuffle
        
        # Use all sizes from datasets_dict (size_distribution no longer filters)
        # All datasets in datasets_dict are already enabled/detected
        self.active_sizes = list(datasets_dict.keys())
        
        if not self.active_sizes:
            raise ValueError("No datasets provided")
        
        # NOTE: We NO LONGER normalize/weight by distribution!
        # Files are already pre-weighted during dataset generation.
        # We simply sample proportionally to actual file counts.
        
        # Pre-compute total batches per size (based on actual file counts, not distribution)
        self.num_batches_per_size = {
            size_key: len(datasets_dict[size_key]) // batch_sizes[size_key]
            for size_key in self.active_sizes
        }
        
        # Total number of batches across all sizes
        self.total_batches = sum(self.num_batches_per_size.values())
    
    def __iter__(self):
        """
        Yields (size_key, batch_indices) tuples.
        
        Each iteration:
        1. Shuffles indices for each size group (if shuffle=True)
        2. Creates batch schedule proportional to file counts (NOT distribution weights)
        3. Yields batches in random order
        
        Note: Sampling is now purely based on actual file counts, not distribution config.
        """
        # Create shuffled indices for each size group
        indices_per_size = {}
        for size_key in self.active_sizes:
            dataset_size = len(self.datasets_dict[size_key])
            indices = list(range(dataset_size))
            
            if self.shuffle:
                random.shuffle(indices)
            
            indices_per_size[size_key] = indices
        
        # Create batch schedule: list of (size_key, batch_idx) based on file counts
        # (NO distribution weighting - files are already pre-weighted!)
        batch_schedule = []
        for size_key in self.active_sizes:
            num_batches = self.num_batches_per_size[size_key]
            batch_schedule.extend([(size_key, i) for i in range(num_batches)])
        
        # Shuffle the batch schedule to mix sizes
        if self.shuffle:
            random.shuffle(batch_schedule)
        
        # Yield batches according to schedule
        for size_key, batch_idx in batch_schedule:
            batch_size = self.batch_sizes[size_key]
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            batch_indices = indices_per_size[size_key][start_idx:end_idx]
            
            yield (size_key, batch_indices)
    
    def __len__(self):
        """Total number of batches across all size groups"""
        return self.total_batches


class MultiSizeDataLoader:
    """
    DataLoader that handles multiple dataset sizes with grouped sampling.
    
    Iterates over batches from different resolution datasets, yielding
    batches with their corresponding size key and metadata.
    
    Args:
        datasets_dict: Dict mapping size_key to VSRDataset instance
        sampler: SizeGroupedSampler instance
    """
    
    def __init__(self, datasets_dict, sampler):
        self.datasets_dict = datasets_dict
        self.sampler = sampler
    
    def __iter__(self):
        """
        Yields batches containing:
        - 'lr': [B, 7, 3, H, W] - LR frames tensor
        - 'gt': [B, 3, H*3, W*3] - GT frames tensor
        - 'size_key': str - Size identifier
        - 'filenames': List[str] - Filenames for this batch
        """
        for size_key, batch_indices in self.sampler:
            dataset = self.datasets_dict[size_key]
            
            # Load samples for this batch
            lr_list = []
            gt_list = []
            filename_list = []
            
            for idx in batch_indices:
                lr, gt, filename = dataset[idx]
                lr_list.append(lr)
                gt_list.append(gt)
                filename_list.append(filename)
            
            # Stack into batch tensors
            lr_batch = torch.stack(lr_list, dim=0)  # [B, 7, 3, H, W]
            gt_batch = torch.stack(gt_list, dim=0)  # [B, 3, H, W]
            
            yield {
                'lr': lr_batch,
                'gt': gt_batch,
                'size_key': size_key,
                'filenames': filename_list
            }
    
    def __len__(self):
        """Total number of batches"""
        return len(self.sampler)


def create_train_loader(config):
    """
    Create multi-size training dataloader from config.
    
    IMPORTANT: Dataset files are pre-weighted during extraction!
    The 'distribution' values are ONLY used to determine which sizes to load.
    Actual training samples ALL files proportionally (no additional weighting).
    
    Args:
        config: Dict containing:
            - 'data_root': Root directory for datasets
            - 'dataset_name': Name of dataset (default: 'master')
            - 'sizes': Dict with size configs, e.g.:
                {
                    '720': {'enabled': True, 'distribution': 0.4, 'batch_size': 1},
                    '540': {'enabled': True, 'distribution': 0.4, 'batch_size': 1},
                    '720_169': {'enabled': True, 'distribution': 0.2, 'batch_size': 1}
                }
                Note: 'distribution' > 0 means "load this size", the value itself
                      is only informational (files on disk determine actual ratio)
            - 'augment': Whether to use augmentations (default: True)
            - 'shuffle': Whether to shuffle batches (default: True)
    
    Returns:
        MultiSizeDataLoader instance
    """
    from .dataset import VSRDataset
    
    data_root = config.get('data_root')
    dataset_name = config.get('dataset_name', 'master')
    sizes_config = config.get('sizes', {})
    augment = config.get('augment', True)
    shuffle = config.get('shuffle', True)
    paths_config = config.get('paths', None)  # NEW: Get paths config
    
    if not data_root:
        raise ValueError("config must contain 'data_root'")
    
    # Create datasets for enabled sizes
    datasets_dict = {}
    size_distribution = {}
    batch_sizes = {}
    
    for size_key, size_cfg in sizes_config.items():
        if not size_cfg.get('enabled', False):
            continue
        
        distribution = size_cfg.get('distribution', 0.0)
        if distribution <= 0.0:
            continue
        
        # Create dataset — skip this size gracefully if it fails
        try:
            dataset = VSRDataset(
                root=data_root,
                dataset_name=dataset_name,
                size_key=size_key,
                mode='train',
                augment=augment,
                paths_config=paths_config  # NEW: Pass paths config
            )
        except Exception as e:
            print(f"⚠️  Warning: Could not load training dataset for size '{size_key}': {e}")
            print(f"   Skipping size '{size_key}' — check GT/LR directories and file extensions.")
            continue
        
        datasets_dict[size_key] = dataset
        size_distribution[size_key] = distribution
        batch_sizes[size_key] = size_cfg.get('batch_size', 1)
    
    if not datasets_dict:
        raise ValueError("No training datasets could be loaded for any size. Check GT/LR directories and file extensions.")
    
    # Create sampler
    sampler = SizeGroupedSampler(
        datasets_dict=datasets_dict,
        size_distribution=size_distribution,
        batch_sizes=batch_sizes,
        shuffle=shuffle
    )
    
    # Create dataloader
    loader = MultiSizeDataLoader(
        datasets_dict=datasets_dict,
        sampler=sampler
    )
    
    return loader
