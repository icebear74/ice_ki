"""
Multi-Size DataLoader with Grouped Sampling

Enables training on multiple resolution variants simultaneously:
- Samples from different resolution datasets based on distribution weights
- Groups samples by size to maintain batch consistency
- Supports custom batch sizes per resolution
- Background prefetch queue: loads the next N batches while the GPU processes
  the current one, hiding disk I/O latency.
"""

import queue
import threading
import torch
from torch.utils.data import Sampler
import random
import numpy as np

from .data_strategy import DataStrategyScheduler  # noqa: F401 – re-exported for callers


class SizeGroupedSampler(Sampler):
    """
    Sampler that yields batches grouped by size key.

    Samples from different size groups proportionally to their file counts
    by default.  When a distribution override is set via
    ``set_distribution()``, sampling proportions follow those weights
    instead of raw file counts.

    IMPORTANT: Dataset extraction is pre-weighted, so files on disk already
    reflect the desired long-run distribution.  The distribution override is
    used only during the graduated warmup strategy.

    Accumulation grouping ensures that exactly ``accum_steps[size_key]``
    consecutive batches of the same resolution are always yielded together,
    so that ``optimizer.step()`` fires only at resolution-block boundaries.

    Args:
        datasets_dict:     Dict mapping size_key to dataset
                           Example: {'720': ds1, '540': ds2, '720_169': ds3}
        size_distribution: Dict mapping size_key to probability weight
                           (legacy param – no longer used for sampling weights;
                           kept for backwards compatibility)
        batch_sizes:       Dict mapping size_key to batch size
        shuffle:           Whether to shuffle indices within each size group
        accum_steps:       Dict mapping size_key to gradient accumulation steps
                           (default: 4 für '720', 4 für '720_169', 3 für '540')
    """

    def __init__(self, datasets_dict, size_distribution, batch_sizes, shuffle=True,
                 accum_steps=None):
        self.datasets_dict = datasets_dict
        self.size_distribution = size_distribution
        self.batch_sizes = batch_sizes
        self.shuffle = shuffle

        # All loaded datasets are active
        self.active_sizes = list(datasets_dict.keys())

        if not self.active_sizes:
            raise ValueError("No datasets provided")

        # Per-size accumulation steps — fixed values matching ADAPTIVE_BATCH_CONFIG:
        #   '720_169': 4,  '720': 4,  '540': 3
        # Any unknown size_key gets a safe default of 4.
        if accum_steps is None:
            accum_steps = {}
        self.accum_steps = {
            sk: accum_steps.get(sk, 4)
            for sk in self.active_sizes
        }

        # Distribution override set by DataStrategyScheduler (None = file-count mode)
        self._distribution_override = None

        # Compute initial batch counts (file-count proportional)
        self._compute_batch_counts()

    # ------------------------------------------------------------------
    # Distribution override
    # ------------------------------------------------------------------

    def set_distribution(self, distribution_dict):
        """
        Override sampling proportions with explicit weights.

        Called by ``DataStrategyScheduler`` at the start of each epoch to
        implement the graduated training schedule.

        Args:
            distribution_dict: Dict mapping size_key → weight (need not sum
                                to 1; will be normalized internally).
                                Sizes with weight 0.0 are excluded from
                                the current epoch's batch schedule.
                                Pass ``None`` to restore file-count mode.
        """
        self._distribution_override = distribution_dict
        self._compute_batch_counts()

    def _compute_batch_counts(self):
        """
        (Re-)compute ``num_batches_per_size`` and ``total_batches``.

        When no distribution override is active, batches are proportional
        to file counts (original behaviour).  When an override is active,
        batches are allocated according to the normalized weights, capped
        at the number of available batches per size to avoid cycling.

        In both cases the final batch count per size is rounded *down* to
        the nearest multiple of ``accum_steps[size_key]`` so that full
        accumulation blocks are always emitted.
        """
        if self._distribution_override is None:
            # Original: proportional to file counts
            self.num_batches_per_size = {
                sk: self._round_to_accum(
                    len(self.datasets_dict[sk]) // self.batch_sizes[sk], sk
                )
                for sk in self.active_sizes
            }
            self.total_batches = sum(self.num_batches_per_size.values())
            return

        # Distribution-weighted mode
        # Active sizes: those with a positive weight that are loaded
        active = {
            sk: w
            for sk, w in self._distribution_override.items()
            if sk in self.active_sizes and w > 0
        }
        total_w = sum(active.values())

        if total_w == 0:
            # Fallback: use file-count proportional
            self.num_batches_per_size = {
                sk: self._round_to_accum(
                    len(self.datasets_dict[sk]) // self.batch_sizes[sk], sk
                )
                for sk in self.active_sizes
            }
            self.total_batches = sum(self.num_batches_per_size.values())
            return

        normalized = {sk: w / total_w for sk, w in active.items()}

        # Base epoch length: total available batches across all sizes
        base_total = sum(
            len(self.datasets_dict[sk]) // self.batch_sizes[sk]
            for sk in self.active_sizes
        )

        self.num_batches_per_size = {}
        for sk in self.active_sizes:
            w = normalized.get(sk, 0.0)
            if w > 0:
                available = len(self.datasets_dict[sk]) // self.batch_sizes[sk]
                target = max(1, round(base_total * w))
                # Cap at available to avoid cycling identical samples, then
                # round down to a full accumulation block
                self.num_batches_per_size[sk] = self._round_to_accum(
                    min(target, available), sk
                )
            else:
                self.num_batches_per_size[sk] = 0

        self.total_batches = sum(self.num_batches_per_size.values())

    def _round_to_accum(self, n_batches, size_key):
        """Round *n_batches* down to the nearest multiple of accum_steps."""
        steps = self.accum_steps.get(size_key, 1)
        if steps <= 1:
            return n_batches
        return (n_batches // steps) * steps

    # ------------------------------------------------------------------
    # Iterator
    # ------------------------------------------------------------------

    def __iter__(self):
        """
        Yields (size_key, batch_indices) tuples.

        Each iteration:
        1. Shuffles indices for each size group (if shuffle=True)
        2. Creates batch schedule according to current batch counts
           (file-count proportional by default; distribution-weighted when
           ``set_distribution()`` has been called)
        3. Groups per-size batches into accumulation blocks of
           ``accum_steps[size_key]`` so that the optimizer step always fires
           at a resolution-block boundary.
        4. Shuffles the accumulation blocks (not individual batches) to
           interleave sizes without breaking accumulation alignment.
        """
        # Create shuffled indices for each active size group
        indices_per_size = {}
        for size_key in self.active_sizes:
            if self.num_batches_per_size.get(size_key, 0) == 0:
                continue
            dataset_size = len(self.datasets_dict[size_key])
            indices = list(range(dataset_size))
            if self.shuffle:
                random.shuffle(indices)
            indices_per_size[size_key] = indices

        # Build accumulation blocks: each block is a list of
        # accum_steps consecutive batch indices for the same size.
        accum_blocks = []
        for size_key in self.active_sizes:
            num_batches = self.num_batches_per_size.get(size_key, 0)
            if num_batches == 0:
                continue
            steps = self.accum_steps.get(size_key, 1)
            # Slice the flat batch index list into blocks of `steps`
            batch_indices = list(range(num_batches))
            for block_start in range(0, num_batches, steps):
                block = [(size_key, i) for i in batch_indices[block_start:block_start + steps]]
                if len(block) == steps:  # only emit complete blocks
                    accum_blocks.append(block)

        # Shuffle at block level to interleave sizes
        if self.shuffle:
            random.shuffle(accum_blocks)

        # Yield batches in block order
        for block in accum_blocks:
            for size_key, batch_idx in block:
                batch_size = self.batch_sizes[size_key]
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                yield (size_key, indices_per_size[size_key][start_idx:end_idx])

    def __len__(self):
        """Total number of batches (forward passes) across all active size groups."""
        return self.total_batches

    @property
    def optimizer_steps(self):
        """Total number of optimizer steps in one epoch.

        Each size group fires an optimizer step every accum_steps[sk] forward
        passes.  Summing over all active sizes gives the correct per-epoch
        count instead of dividing total_batches by a single global value.
        """
        return sum(
            self.num_batches_per_size[sk] // self.accum_steps.get(sk, 1)
            for sk in self.active_sizes
            if self.num_batches_per_size.get(sk, 0) > 0
        )

    @property
    def total_files(self):
        """Total number of individual training images in one epoch.

        Computed as the sum of (batches × batch_size) per size, which accounts
        for the fact that different sizes may have different physical batch sizes
        (e.g. BS=2 for 540/720_169, BS=1 for 720).
        """
        return sum(
            self.num_batches_per_size[sk] * self.batch_sizes.get(sk, 1)
            for sk in self.active_sizes
            if self.num_batches_per_size.get(sk, 0) > 0
        )


class MultiSizeDataLoader:
    """
    DataLoader that handles multiple dataset sizes with grouped sampling.
    
    Iterates over batches from different resolution datasets, yielding
    batches with their corresponding size key and metadata.

    Background prefetch threads load the next ``prefetch_count`` batches
    while the training loop processes the current one, eliminating disk I/O
    from the critical path.
    
    Args:
        datasets_dict: Dict mapping size_key to VSRDataset instance
        sampler: SizeGroupedSampler instance
        prefetch_count: Number of batches to buffer ahead (0 = no prefetch)
        prefetch_workers: Number of background loader threads
    """
    
    def __init__(self, datasets_dict, sampler, prefetch_count=8, prefetch_workers=2):
        self.datasets_dict = datasets_dict
        self.sampler = sampler
        self.prefetch_count = max(0, int(prefetch_count))
        self.prefetch_workers = max(1, int(prefetch_workers))
    
    def _load_batch(self, size_key, batch_indices):
        """Load a single batch and return the packed dict."""
        dataset = self.datasets_dict[size_key]
        lr_list = []
        gt_list = []
        filename_list = []
        for idx in batch_indices:
            lr, gt, filename = dataset[idx]
            lr_list.append(lr)
            gt_list.append(gt)
            filename_list.append(filename)
        lr_batch = torch.stack(lr_list, dim=0)   # [B, 7, 3, H, W]
        gt_batch = torch.stack(gt_list, dim=0)   # [B, 3, H, W]
        return {
            'lr': lr_batch,
            'gt': gt_batch,
            'size_key': size_key,
            'filenames': filename_list
        }

    def __iter__(self):
        """
        Yields batches containing:
        - 'lr': [B, 7, 3, H, W] - LR frames tensor
        - 'gt': [B, 3, H*3, W*3] - GT frames tensor
        - 'size_key': str - Size identifier
        - 'filenames': List[str] - Filenames for this batch

        When prefetch_count > 0 a pool of daemon threads pre-loads batches
        into a bounded queue so that GPU compute and disk I/O overlap.
        Exceptions raised in worker threads are re-raised in the main thread.
        """
        if self.prefetch_count <= 0:
            # Simple synchronous path (no prefetch)
            for size_key, batch_indices in self.sampler:
                yield self._load_batch(size_key, batch_indices)
            return

        # --- Asynchronous prefetch path ---
        result_queue: queue.Queue = queue.Queue(maxsize=self.prefetch_count)
        _SENTINEL = object()  # unique end-of-stream marker

        def producer():
            try:
                for size_key, batch_indices in self.sampler:
                    batch = self._load_batch(size_key, batch_indices)
                    result_queue.put(batch)
                result_queue.put(_SENTINEL)
            except Exception as exc:
                result_queue.put(exc)

        thread = threading.Thread(target=producer, daemon=True,
                                  name="vsr-prefetch-producer")
        thread.start()

        while True:
            item = result_queue.get()
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield item

        thread.join()
    
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
                    '720': {'enabled': True, 'distribution': 0.4, 'batch_size': 1, 'accum': 4},
                    '540': {'enabled': True, 'distribution': 0.4, 'batch_size': 2, 'accum': 3},
                    '720_169': {'enabled': True, 'distribution': 0.2, 'batch_size': 2, 'accum': 4}
                }
                Note: 'distribution' > 0 means "load this size", the value itself
                      is only informational (files on disk determine actual ratio).
                      'accum' sets gradient accumulation steps for this size
                      (defaults: 4 for '720', 4 for '720_169', 3 for '540').
            - 'augment': Ignored – augmentation is permanently disabled.
            - 'shuffle': Whether to shuffle batches (default: True)
            - 'cache_max_items': LRU cache size per dataset (default: 3000, 0=disabled)
            - 'prefetch_count': Batches to buffer ahead in background (default: 8)
            - 'prefetch_workers': Background loader threads (default: 2, currently
                                   the implementation uses a single producer thread)
    
    Returns:
        MultiSizeDataLoader instance
    """
    from .dataset import VSRDataset
    
    data_root = config.get('data_root')
    dataset_name = config.get('dataset_name', 'master')
    sizes_config = config.get('sizes', {})
    augment = config.get('augment', True)
    shuffle = config.get('shuffle', True)
    paths_config = config.get('paths', None)
    cache_max_items = int(config.get('cache_max_items', 3000))
    prefetch_count = int(config.get('prefetch_count', 8))
    prefetch_workers = int(config.get('prefetch_workers', 2))
    
    if not data_root:
        raise ValueError("config must contain 'data_root'")
    
    # Create datasets for enabled sizes
    datasets_dict = {}
    size_distribution = {}
    batch_sizes = {}
    accum_steps = {}
    
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
                paths_config=paths_config,
                cache_max_items=cache_max_items,
            )
        except Exception as e:
            import traceback as _tb
            print(f"⚠️  Warning: Could not load training dataset for size '{size_key}': {e}")
            _tb.print_exc()
            print(f"   Skipping size '{size_key}' — check GT/LR directories and file extensions.")
            continue
        
        datasets_dict[size_key] = dataset
        size_distribution[size_key] = distribution
        # batch_size and accum must be explicitly set in sizes_config — they come
        # from ADAPTIVE_BATCH_CONFIG in config.py (fixed values, no guessing).
        batch_sizes[size_key] = size_cfg['batch_size']
        accum_steps[size_key] = size_cfg['accum']
    
    if not datasets_dict:
        raise ValueError("No training datasets could be loaded for any size. Check GT/LR directories and file extensions.")
    
    # Create sampler
    sampler = SizeGroupedSampler(
        datasets_dict=datasets_dict,
        size_distribution=size_distribution,
        batch_sizes=batch_sizes,
        shuffle=shuffle,
        accum_steps=accum_steps
    )
    
    # Create dataloader with prefetch queue
    loader = MultiSizeDataLoader(
        datasets_dict=datasets_dict,
        sampler=sampler,
        prefetch_count=prefetch_count,
        prefetch_workers=prefetch_workers,
    )
    
    return loader
