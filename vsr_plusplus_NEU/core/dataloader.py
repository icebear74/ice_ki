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

    Implements a **two-stage asynchronous pipeline** to hide disk I/O latency:

      Stage 1 – Producer  (``prefetch_workers`` threads)
          Disk  →  cv2.imread / tensor conversion  →  *raw_queue*

      Stage 2 – Pinner  (``pin_workers`` threads)
          *raw_queue*  →  ``.pin_memory()``  →  *ready_queue*

      Consumer  (training loop)
          *ready_queue*  →  ``.to(device, non_blocking=True)``  →  GPU

    The queues are bounded, so producers block whenever the consumer is slow
    and the buffer is full, preventing unbounded RAM growth.

    The live fill-levels of both queues are exposed via :attr:`prefetch_stats`
    and are pushed to the WebUI on every training step.

    Args:
        datasets_dict:    Dict mapping size_key to VSRDataset instance.
        sampler:          SizeGroupedSampler instance.
        prefetch_count:   Capacity of the raw (disk) queue in batches.
                          0 disables async prefetch entirely.
        prefetch_workers: Number of parallel disk-loading threads (Stage 1).
        pin_workers:      Number of pin_memory threads (Stage 2).
                          Set to 0 to skip pinning (useful without CUDA).
    """

    def __init__(self, datasets_dict, sampler,
                 prefetch_count: int = 10,
                 prefetch_workers: int = 1,
                 pin_workers: int = 1):
        self.datasets_dict = datasets_dict
        self.sampler = sampler
        self.prefetch_count  = max(0, int(prefetch_count))
        self.prefetch_workers = max(1, int(prefetch_workers))
        self.pin_workers     = max(0, int(pin_workers))

        # Queue references – set during __iter__, None between epochs
        self._raw_queue: 'queue.Queue | None'   = None
        self._ready_queue: 'queue.Queue | None' = None
        self._ready_queue_max: int = 0

    # ------------------------------------------------------------------
    # Public stats property (readable from trainer / WebUI at any time)
    # ------------------------------------------------------------------

    @property
    def prefetch_stats(self) -> dict:
        """Return a snapshot of both pipeline queue fill levels.

        Safe to call from any thread at any time (no locking needed because
        ``queue.Queue.qsize()`` is atomic on CPython / Linux).
        """
        raw_q   = self._raw_queue
        ready_q = self._ready_queue

        raw_current   = raw_q.qsize()   if raw_q   is not None else 0
        ready_current = ready_q.qsize() if ready_q is not None else 0
        raw_max       = self.prefetch_count
        ready_max     = self._ready_queue_max

        total_current = raw_current + ready_current
        total_max     = raw_max + ready_max
        fill_pct      = (total_current / total_max * 100.0) if total_max > 0 else 0.0

        return {
            'enabled':       self.prefetch_count > 0,
            'raw_current':   raw_current,
            'raw_max':       raw_max,
            'ready_current': ready_current,
            'ready_max':     ready_max,
            'total_current': total_current,
            'total_max':     total_max,
            'fill_pct':      round(fill_pct, 1),
        }

    # ------------------------------------------------------------------
    # Internal batch loader (called from producer threads)
    # ------------------------------------------------------------------

    def _load_batch(self, size_key, batch_indices):
        """Load a single batch and return the packed dict."""
        dataset = self.datasets_dict[size_key]
        lr_list, gt_list, filename_list = [], [], []
        for idx in batch_indices:
            lr, gt, filename = dataset[idx]
            lr_list.append(lr)
            gt_list.append(gt)
            filename_list.append(filename)
        return {
            'lr':        torch.stack(lr_list, dim=0),   # [B, 7, 3, H, W]
            'gt':        torch.stack(gt_list, dim=0),   # [B, 3, H, W]
            'size_key':  size_key,
            'filenames': filename_list,
        }

    # ------------------------------------------------------------------
    # Iterator
    # ------------------------------------------------------------------

    def __iter__(self):
        """
        Yields batches with keys: ``lr``, ``gt``, ``size_key``, ``filenames``.

        When ``prefetch_count > 0`` the two-stage async pipeline is active.
        Workers are daemon threads that are killed automatically if the main
        process exits.  Exceptions in any worker thread are re-raised in the
        consumer (training loop).
        """
        if self.prefetch_count <= 0:
            # Synchronous fallback – useful for debugging
            for size_key, batch_indices in self.sampler:
                yield self._load_batch(size_key, batch_indices)
            return

        _SENTINEL = object()  # unique end-of-stream marker
        _use_pin  = self.pin_workers > 0 and torch.cuda.is_available()

        # ---- Stop event: set by consumer to unblock stuck threads ----
        # Threads must check this flag on every queue.put() so they do not
        # block forever when the consumer exits early (GeneratorExit or break).
        _stop = threading.Event()

        # ---- Stage 1: raw queue (disk I/O) ---------------------------
        raw_queue: queue.Queue = queue.Queue(maxsize=self.prefetch_count)
        self._raw_queue = raw_queue

        # ---- Stage 2: ready queue (pinned) ---------------------------
        ready_max = max(2, self.pin_workers * 2) if _use_pin else 0
        if _use_pin:
            ready_queue: queue.Queue = queue.Queue(maxsize=ready_max)
        else:
            ready_queue = raw_queue   # bypass – ready == raw
        self._ready_queue     = ready_queue
        self._ready_queue_max = ready_max

        # ---- Helper: put with stop-event check -----------------------
        def _put(q: queue.Queue, item, timeout: float = 0.05):
            """Put item into q; return False if _stop was set before success."""
            while not _stop.is_set():
                try:
                    q.put(item, timeout=timeout)
                    return True
                except queue.Full:
                    pass
            return False  # stop signal received, item dropped

        # ---- Producer thread(s) (disk → CPU tensor) ------------------
        work_queue: queue.Queue = queue.Queue()
        for item in self.sampler:
            work_queue.put(item)
        # One sentinel per producer so each knows when work is exhausted
        for _ in range(self.prefetch_workers):
            work_queue.put(_SENTINEL)

        finished_producers = [0]
        finished_lock = threading.Lock()

        def producer():
            while True:
                item = work_queue.get()
                if item is _SENTINEL:
                    with finished_lock:
                        finished_producers[0] += 1
                        if finished_producers[0] == self.prefetch_workers:
                            # Last producer finished: emit one sentinel per pinner
                            # (or one directly to ready_queue when no pinners).
                            sentinels = self.pin_workers if _use_pin else 1
                            for _ in range(sentinels):
                                _put(raw_queue, _SENTINEL)
                    return
                if _stop.is_set():
                    return
                size_key, batch_indices = item
                try:
                    batch = self._load_batch(size_key, batch_indices)
                    if not _put(raw_queue, batch):
                        return  # consumer is gone
                except Exception as exc:
                    _put(raw_queue, exc)
                    return

        producer_threads = [
            threading.Thread(target=producer, daemon=True,
                             name=f"vsr-producer-{i}")
            for i in range(self.prefetch_workers)
        ]

        # ---- Pinner thread(s) (CPU tensor → pinned) ------------------
        pinner_threads = []
        if _use_pin:
            finished_pinners = [0]
            finished_pinner_lock = threading.Lock()

            def pinner():
                while True:
                    try:
                        item = raw_queue.get(timeout=0.05)
                    except queue.Empty:
                        if _stop.is_set():
                            return
                        continue
                    if item is _SENTINEL:
                        with finished_pinner_lock:
                            finished_pinners[0] += 1
                            if finished_pinners[0] == self.pin_workers:
                                _put(ready_queue, _SENTINEL)
                        return
                    if isinstance(item, Exception):
                        _put(ready_queue, item)
                        return
                    if _stop.is_set():
                        return
                    try:
                        _put(ready_queue, {
                            'lr':        item['lr'].pin_memory(),
                            'gt':        item['gt'].pin_memory(),
                            'size_key':  item['size_key'],
                            'filenames': item['filenames'],
                        })
                    except Exception as exc:
                        _put(ready_queue, exc)
                        return

            pinner_threads = [
                threading.Thread(target=pinner, daemon=True,
                                 name=f"vsr-pinner-{i}")
                for i in range(self.pin_workers)
            ]

        # ---- Start all threads ---------------------------------------
        for t in producer_threads:
            t.start()
        for t in pinner_threads:
            t.start()

        # ---- Consume -------------------------------------------------
        try:
            while True:
                item = ready_queue.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            # Signal all worker threads to stop (handles GeneratorExit, break,
            # exceptions – any exit path from the consumer).
            _stop.set()

            # Drain both queues so any thread blocked on put() can unblock,
            # notice _stop, and exit its loop.
            for q in ([raw_queue, ready_queue] if _use_pin else [raw_queue]):
                while True:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break

            # Join with a generous timeout; daemon=True ensures process exit
            # is not blocked if a thread hangs unexpectedly.
            for t in producer_threads + pinner_threads:
                t.join(timeout=5.0)

            # Clear queue references so prefetch_stats shows zeros
            self._raw_queue   = None
            self._ready_queue = None
            self._ready_queue_max = 0

    def __len__(self):
        """Total number of batches"""
        return len(self.sampler)


def create_train_loader(config):
    """
    Create multi-size training dataloader from config.
    
    Args:
        config: Dict containing:
            - 'data_root': Root directory for datasets
            - 'dataset_name': Name of dataset (default: 'master')
            - 'sizes': Dict with size configs
            - 'augment': Ignored – augmentation is permanently disabled.
            - 'shuffle': Whether to shuffle batches (default: True)
            - 'prefetch_count':  Raw-queue capacity in batches (default: 10).
                                 0 = synchronous / no prefetch.
            - 'prefetch_workers': Parallel disk-loading threads (default: 1).
            - 'pin_workers':     pin_memory threads for GPU-ready queue (default: 1).
                                 0 = skip pinning (e.g. CPU-only machine).
    
    Returns:
        MultiSizeDataLoader instance
    """
    from .dataset import VSRDataset
    
    data_root        = config.get('data_root')
    dataset_name     = config.get('dataset_name', 'master')
    sizes_config     = config.get('sizes', {})
    augment          = config.get('augment', True)
    shuffle          = config.get('shuffle', True)
    paths_config     = config.get('paths', None)
    prefetch_count   = int(config.get('prefetch_count',   10))
    prefetch_workers = int(config.get('prefetch_workers',  1))
    pin_workers      = int(config.get('pin_workers',       1))
    
    if not data_root:
        raise ValueError("config must contain 'data_root'")
    
    # Create datasets for enabled sizes
    datasets_dict    = {}
    size_distribution = {}
    batch_sizes      = {}
    accum_steps      = {}
    
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
            )
        except Exception as e:
            import traceback as _tb
            print(f"⚠️  Warning: Could not load training dataset for size '{size_key}': {e}")
            _tb.print_exc()
            print(f"   Skipping size '{size_key}' — check GT/LR directories and file extensions.")
            continue
        
        datasets_dict[size_key]    = dataset
        size_distribution[size_key] = distribution
        batch_sizes[size_key]      = size_cfg['batch_size']
        accum_steps[size_key]      = size_cfg['accum']
    
    if not datasets_dict:
        raise ValueError("No training datasets could be loaded for any size. "
                         "Check GT/LR directories and file extensions.")
    
    # Create sampler
    sampler = SizeGroupedSampler(
        datasets_dict=datasets_dict,
        size_distribution=size_distribution,
        batch_sizes=batch_sizes,
        shuffle=shuffle,
        accum_steps=accum_steps,
    )
    
    # Create dataloader with 2-stage prefetch pipeline
    loader = MultiSizeDataLoader(
        datasets_dict=datasets_dict,
        sampler=sampler,
        prefetch_count=prefetch_count,
        prefetch_workers=prefetch_workers,
        pin_workers=pin_workers,
    )
    
    return loader
