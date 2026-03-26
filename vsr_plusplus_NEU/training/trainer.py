"""
VSRTrainer - Main training loop orchestration

Coordinates all training components:
- Model training
- Validation
- Checkpointing
- Logging
- Adaptive systems
- Interactive GUI
"""

import time
import os
import json
from collections import deque
import torch
from torch.amp import autocast
from ..utils.ui_display import draw_ui, get_activity_data
from ..utils.keyboard_handler import KeyboardHandler
from ..utils.ui_terminal import C_GREEN, C_CYAN, C_YELLOW, C_RESET, show_cursor
from ..core.data_strategy import DataStrategyScheduler



class VSRTrainer:
    """
    Main training orchestrator
    
    Args:
        model: VSR model
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        train_loader: Training data loader
        val_loader: Validation data loader
        loss_fn: Loss function
        validator: Validator instance
        checkpoint_mgr: Checkpoint manager
        train_logger: Training logger
        tb_logger: TensorBoard logger
        adaptive_system: Adaptive training system
        config: Training configuration
        device: Device to use
    """
    
    def __init__(self, model, optimizer, lr_scheduler, train_loader, val_loader, loss_fn,
                 validator, checkpoint_mgr, train_logger, tb_logger, adaptive_system,
                 config, device='cuda', scaler=None, use_amp=False):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.validator = validator
        self.checkpoint_mgr = checkpoint_mgr
        self.train_logger = train_logger
        self.tb_logger = tb_logger
        self.adaptive_system = adaptive_system
        self.config = config
        self.device = device
        self.scaler = scaler
        self.use_amp = use_amp
        
        self.global_step = 0
        self.start_step = 0
        
        # Metrics tracking
        self.last_metrics = None
        self.last_activities = None
        self.loss_history = []
        
        # For validation snapshots
        self.last_total_loss = None
        self.last_l1_loss = None
        self.last_validation_quality = None
        
        # Performance tracking
        # Rolling window of the last 200 optimizer-step durations (seconds each).
        # Using 200 steps gives a stable, representative average without
        # over-weighting stale measurements from many steps ago.
        # deque(maxlen=200) gives O(1) append + automatic eviction of old entries.
        self.step_times = deque(maxlen=200)
        
        # Rolling window of per-sample validation durations (seconds per sample).
        # Validation speed includes the full cycle: model forward pass + metric
        # computation + TensorBoard image logging + disk writes.
        # 1 entry = 1 GT/LR pair processed.  Kept to the last 200 samples.
        self._val_sample_timings = deque(maxlen=200)
        self.last_val_iter_per_sec = 0.0  # published to UI / WebMonitor
        
        # EMA for GUI smoothing (factor 0.95)
        self.ema_loss = None
        self.ema_factor = 0.95
        
        # UI state
        self.paused = False
        self.do_manual_val = False

        # Crop-wait state: training is blocked when step >= WARMUP_END but
        # not enough crop GT images exist yet.  Checked every 5 minutes.
        self.waiting_for_crops = False
        self._crop_wait_next_check = 0.0   # monotonic timestamp of next auto-rescan
        self._crop_wait_current_count = 0  # last known combined 540+720 count
        
        # Adaptive mode change tracking for TensorBoard phase transition logging
        self._last_adaptive_mode = 'Stable'
        
        # Graduated data/loss strategy scheduler (optional)
        # Set via trainer.data_strategy_scheduler = DataStrategyScheduler(...)
        self.data_strategy_scheduler = None
        
        # Per-size VRAM tracker: running EMA of memory_reserved() measured
        # right after each optimizer step, keyed by size_key.  Sent to the
        # WebUI so the batch-config table shows live values instead of
        # hardcoded placeholders.
        self._vram_per_size: dict = {}
        
        # Keyboard handler for interactive training control
        self.keyboard = KeyboardHandler()

        # Web interface for remote monitoring
        from ..systems.web_ui import WebMonitoringInterface
        self.web_monitor = WebMonitoringInterface(port_num=5050, refresh_seconds=5)
        # Push ADAPTIVE_BATCH_CONFIG from config into the WebUI state so the
        # batch-config table reflects what is actually configured, not the
        # hardcoded placeholder that web_ui.py uses as its initial state.
        adaptive_cfg = self.config.get('ADAPTIVE_BATCH_CONFIG', {})
        if adaptive_cfg:
            ui_batch_cfg = {}
            for sk, v in adaptive_cfg.items():
                entry = {'batch': v['batch'], 'accum': v['accum'],
                         'effective': v['batch'] * v['accum']}
                if 'vram_gb' in v:
                    entry['vram_gb'] = v['vram_gb']
                ui_batch_cfg[sk] = entry
            self.web_monitor.data_store.update_all_metrics(
                adaptive_batch_config=ui_batch_cfg
            )
    
    def set_start_step(self, step):
        """Set starting step (for resume)"""
        self.start_step = step
        self.global_step = step
    
    def _reload_val_datasets_if_needed(self):
        """
        Check every validation dataset for file changes and reload immediately if found.

        Called unconditionally before every validation run so that each validation
        always uses the most current image data — regardless of whether the periodic
        100-step check has already fired for this step.

        Covers:
          - multi-size loaders  (self.val_loaders list of (size_key, loader) tuples)
          - single-size loader  (self.val_loader)
        """
        # ── Multi-size validation loaders ────────────────────────────────────
        val_loaders = getattr(self, 'val_loaders', None)
        if val_loaders and isinstance(val_loaders, list):
            for size_key, val_loader in val_loaders:
                try:
                    if not hasattr(val_loader, 'dataset'):
                        continue
                    val_ds = val_loader.dataset
                    if not hasattr(val_ds, 'check_for_new_files') or not hasattr(val_ds, 'reload_files'):
                        continue
                    val_changes = val_ds.check_for_new_files()
                    if val_changes['has_new']:
                        delta = val_changes['new_files']
                        delta_str = f"+{delta}" if delta >= 0 else str(delta)
                        print(f"\n📂 Pre-validation check: {size_key} changed ({delta_str} files). Reloading...")
                        reload_result = val_ds.reload_files()
                        if reload_result['success']:
                            print(f"   ✅ {size_key}: {reload_result['files_before']} → {reload_result['files_after']} files")
                            if hasattr(self, 'train_logger') and self.train_logger:
                                self.train_logger.log_event(
                                    f"Pre-val reload {size_key}: {reload_result['files_before']} → {reload_result['files_after']} files"
                                )
                        else:
                            print(f"   ❌ {size_key} reload failed: {reload_result.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"⚠️  Pre-validation reload error for {size_key}: {e}")

        # ── Single validation loader ──────────────────────────────────────────
        elif hasattr(self, 'val_loader') and hasattr(self.val_loader, 'dataset'):
            try:
                val_ds = self.val_loader.dataset
                if hasattr(val_ds, 'check_for_new_files') and hasattr(val_ds, 'reload_files'):
                    val_changes = val_ds.check_for_new_files()
                    if val_changes['has_new']:
                        delta = val_changes['new_files']
                        delta_str = f"+{delta}" if delta >= 0 else str(delta)
                        print(f"\n📂 Pre-validation check: validation dataset changed ({delta_str} files). Reloading...")
                        reload_result = val_ds.reload_files()
                        if reload_result['success']:
                            print(f"   ✅ {reload_result['files_before']} → {reload_result['files_after']} files")
                            if hasattr(self, 'train_logger') and self.train_logger:
                                self.train_logger.log_event(
                                    f"Pre-val reload: {reload_result['files_before']} → {reload_result['files_after']} files"
                                )
                        else:
                            print(f"   ❌ Reload failed: {reload_result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"⚠️  Pre-validation reload error: {e}")

    def _run_multi_size_validation(self):
        """
        Run validation on all configured sizes
        Returns combined metrics averaging across all sizes
        """
        # Always refresh validation datasets before running so every validation
        # uses the most current images (covers periodic, manual, web-UI and snapshot calls).
        self._reload_val_datasets_if_needed()

        # ── Signal validation start to WebUI ─────────────────────────────────
        self.web_monitor.data_store.update_all_metrics(
            validation_running=True,
            val_status={'running': True, 'phase': 'validating',
                        'done': 0, 'total': 0, 'pct': 0.0, 'size_key': ''},
        )

        try:
            if not hasattr(self, 'val_loaders') or not self.val_loaders:
                # Fallback to single-size validation
                total_batches = len(self.validator.val_loader) if self.validator.val_loader else 0
                self.web_monitor.data_store.update_all_metrics(
                    val_status={'running': True, 'phase': 'validating',
                                'done': 0, 'total': total_batches, 'pct': 0.0, 'size_key': 'default'},
                )
                def _cb(done, total):
                    pct = done / total * 100 if total else 0.0
                    self.web_monitor.data_store.update_all_metrics(
                        val_status={'running': True, 'phase': 'validating',
                                    'done': done, 'total': total, 'pct': pct, 'size_key': 'default'},
                    )
                return self.validator.validate(self.global_step, progress_callback=_cb)

            print(f"\n{C_CYAN}Running multi-size validation on {len(self.val_loaders)} sizes...{C_RESET}")

            # Pre-calculate total batches across all sizes for global progress
            size_batch_counts = [(sk, len(loader)) for sk, loader in self.val_loaders]
            global_total = sum(n for _, n in size_batch_counts)
            global_done = 0

            # Run validation on each size
            all_metrics = []
            all_labeled_images = []

            for size_key, val_loader in self.val_loaders:
                print(f"  Validating {size_key}...")
                size_total = len(val_loader)
                size_done_offset = global_done

                # Build per-size callback that updates the global counter
                def _make_cb(sk, offset):
                    def _cb(done, _total):
                        nonlocal global_done
                        # global_done = offset + done (avoid double-counting)
                        global_done = offset + done
                        pct = global_done / global_total * 100 if global_total else 0.0
                        self.web_monitor.data_store.update_all_metrics(
                            val_status={'running': True, 'phase': 'validating',
                                        'done': global_done, 'total': global_total,
                                        'pct': pct, 'size_key': sk},
                        )
                    return _cb

                progress_cb = _make_cb(size_key, size_done_offset)

                # Temporarily swap loader
                original_loader = self.validator.val_loader
                self.validator.val_loader = val_loader

                # Run validation
                metrics = self.validator.validate(self.global_step, progress_callback=progress_cb)

                # Restore original loader
                self.validator.val_loader = original_loader

                global_done = size_done_offset + size_total  # mark size as fully done

                # Collect labeled images with size prefix
                if 'labeled_images' in metrics and metrics['labeled_images'] is not None:
                    # Build tag as val_{size_key}/{filename_stem}
                    for name, img in metrics['labeled_images']:
                        all_labeled_images.append((f"val_{size_key}/{name}", img))

                # Store metrics
                all_metrics.append((size_key, metrics))
                print(f"    ✓ {size_key}: KI Quality {metrics['ki_quality']*100:.1f}%, PSNR {metrics['ki_psnr']:.2f}dB")

            # ── Signal "saving" phase (TensorBoard image write) ───────────────
            self.web_monitor.data_store.update_all_metrics(
                val_status={'running': True, 'phase': 'saving',
                            'done': global_total, 'total': global_total,
                            'pct': 100.0, 'size_key': ''},
            )

            # Combine metrics by averaging
            combined_metrics = {
                'val_loss': sum(m['val_loss'] for _, m in all_metrics) / len(all_metrics),
                'lr_quality': sum(m['lr_quality'] for _, m in all_metrics) / len(all_metrics),
                'ki_quality': sum(m['ki_quality'] for _, m in all_metrics) / len(all_metrics),
                'improvement': sum(m['improvement'] for _, m in all_metrics) / len(all_metrics),
                'lr_psnr': sum(m['lr_psnr'] for _, m in all_metrics) / len(all_metrics),
                'lr_ssim': sum(m['lr_ssim'] for _, m in all_metrics) / len(all_metrics),
                'ki_psnr': sum(m['ki_psnr'] for _, m in all_metrics) / len(all_metrics),
                'ki_ssim': sum(m['ki_ssim'] for _, m in all_metrics) / len(all_metrics),
                'ki_to_gt': sum(m.get('ki_to_gt', 0) for _, m in all_metrics) / len(all_metrics),
                'lr_to_gt': sum(m.get('lr_to_gt', 0) for _, m in all_metrics) / len(all_metrics),
            }

            # Include all labeled images from all sizes (list of (tag, tensor) tuples)
            if all_labeled_images:
                combined_metrics['labeled_images'] = all_labeled_images

            # Store per-size metrics for detailed logging
            combined_metrics['per_size_metrics'] = {size_key: m for size_key, m in all_metrics}

            print(f"{C_GREEN}✅ Multi-size validation complete - Average KI Quality: {combined_metrics['ki_quality']*100:.1f}%{C_RESET}\n")

            return combined_metrics

        finally:
            # ── Always clear validation-running flag ──────────────────────────
            self.web_monitor.data_store.update_all_metrics(
                validation_running=False,
                val_status={'running': False, 'phase': 'idle',
                            'done': 0, 'total': 0, 'pct': 0.0, 'size_key': ''},
            )
    
    def train_epoch(self, epoch):
        """
        Train one epoch

        Args:
            epoch: Current epoch number
        """
        self.model.train()

        # ── Graduated data strategy ──────────────────────────────────────────
        # Update the sampler's distribution at the start of each epoch so that
        # Phase 1/2/3 transitions take effect without restarting training.
        if self.data_strategy_scheduler is not None:
            scheduler = self.data_strategy_scheduler
            sampler = getattr(self.train_loader, 'sampler', None)

            # Build crop file count map so Phase 2 is only entered once enough
            # crop files actually exist on disk (independent of step number).
            _crop_counts = self._get_crop_file_counts()

            if sampler is not None and hasattr(sampler, 'set_distribution'):
                dist = scheduler.get_distribution(
                    self.global_step,
                    available_sizes=sampler.active_sizes,
                    crop_file_counts=_crop_counts
                )
                sampler.set_distribution(dist)

            # Log phase transitions
            scheduler.check_phase_transition(
                self.global_step,
                log_fn=self.train_logger.log_event,
                crop_file_counts=_crop_counts
            )
        # ── End graduated data strategy ──────────────────────────────────────
        
        default_accum_steps = self.config.get('ACCUMULATION_STEPS', 4)
        # Prefer the sampler's exact optimizer-step count (accounts for per-size
        # accum_steps).  Fall back to dividing total forward passes by the global
        # default only when the sampler is unavailable (single-size DataLoader).
        _sampler = getattr(self.train_loader, 'sampler', None)
        if _sampler is not None and hasattr(_sampler, 'optimizer_steps'):
            steps_per_epoch = _sampler.optimizer_steps
        else:
            steps_per_epoch = len(self.train_loader) // default_accum_steps
        current_epoch_step = 0
        accum_counter = 0  # Running counter for dynamic accumulation
        prev_size_key = None  # Track size-key changes for clean boundary enforcement

        # Per-optimizer-step file tracking: collects batches for the CURRENT
        # accumulation window (reset at each optimizer step and on size-key change).
        current_window_batches = []  # list[dict]: each entry has 'size_key' and 'files'

        # Snapshot of the last complete accumulation window for WebUI display.
        # Updated only when current_window_batches reaches current_accum_steps,
        # so the display always shows a full set of same-resolution files.
        display_files = []   # list[str]: file paths from the last complete window
        display_fps = {'720': 0, '540': 0, '720_169': 0}

        # Cumulative per-size file counter for the current epoch (used by WebUI).
        # This is a local variable – reset automatically on each call to train_epoch.
        epoch_files_per_size = {'720': 0, '540': 0, '720_169': 0}
        
        # Initialize loop timing
        loop_start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Handle pause state
            while self.paused:
                self._update_gui(epoch, {}, 0.1, steps_per_epoch, current_epoch_step, paused=True)
                time.sleep(0.5)
                self._check_keyboard_input(epoch, steps_per_epoch, current_epoch_step)

            # ── Crop availability guard ──────────────────────────────────────
            # When we have reached WARMUP_END the system wants to introduce
            # 540/720 crops.  Block here until at least MIN_CROP_FILES_TRAINING
            # combined crop GT images are available.  Rescans every 5 minutes;
            # the user can also trigger an immediate re-check from the WebGUI.
            if (self.data_strategy_scheduler is not None
                    and self.global_step >= DataStrategyScheduler.WARMUP_END):
                # Fast path: check in-memory counts first (no I/O).
                if not self._check_crop_readiness(force_rescan=False):
                    # Trigger an immediate rescan before entering the wait loop.
                    self._check_crop_readiness(force_rescan=True)

                if self.waiting_for_crops:
                    needed = DataStrategyScheduler.MIN_CROP_FILES_TRAINING
                    self.train_logger.log_event(
                        f"⏳ Crop-Wait: Nur {self._crop_wait_current_count:,}/{needed:,} "
                        f"Crop-Bilder vorhanden. Training pausiert."
                    )
                    # Schedule the first periodic rescan 5 minutes from now.
                    self._crop_wait_next_check = time.time() + 300.0

                while self.waiting_for_crops:
                    now = time.time()

                    # Periodic auto-rescan every 5 minutes.
                    if now >= self._crop_wait_next_check:
                        self._check_crop_readiness(force_rescan=True)
                        if not self.waiting_for_crops:
                            break  # Enough crops found – resume training.
                        needed = DataStrategyScheduler.MIN_CROP_FILES_TRAINING
                        self.train_logger.log_event(
                            f"⏳ Crop-Wait: {self._crop_wait_current_count:,}/{needed:,} "
                            f"vorhanden. Nächste Prüfung in 5 Minuten."
                        )
                        self._crop_wait_next_check = time.time() + 300.0

                    secs_until = max(0, int(self._crop_wait_next_check - time.time()))
                    needed = DataStrategyScheduler.MIN_CROP_FILES_TRAINING
                    self.web_monitor.data_store.update_all_metrics(
                        crop_wait_active=True,
                        crop_wait_current_count=self._crop_wait_current_count,
                        crop_wait_needed_count=needed,
                        crop_wait_next_check_secs=secs_until,
                        training_paused=True,
                        training_active=False,
                    )
                    self._update_gui(epoch, {}, 0.1, steps_per_epoch, current_epoch_step, paused=True)
                    time.sleep(0.5)
                    self._check_keyboard_input(epoch, steps_per_epoch, current_epoch_step)

                # Clear crop-wait banner once resolved.
                if not self.waiting_for_crops:
                    self.web_monitor.data_store.update_all_metrics(crop_wait_active=False)
            # ── End crop availability guard ──────────────────────────────────
            
            # Check keyboard input
            self._check_keyboard_input(epoch, steps_per_epoch, current_epoch_step)

            
            # Manual validation trigger
            if self.do_manual_val:
                if getattr(self, 'use_async_validation', False):
                    # Async path: just like the scheduled trigger – saves a
                    # weights-only checkpoint and signals the async validator.
                    # Training continues unblocked after this call.
                    self._request_async_validation()
                else:
                    # Synchronous path (default or no second GPU): blocks until
                    # the full validation cycle (incl. TensorBoard writes) is done.
                    self._run_validation()
                    # Reset timing after the blocking validation
                    loop_start_time = time.time()
                self.do_manual_val = False
            
            # Handle both single-size (tuple) and multi-size (dict) batches
            if isinstance(batch, dict):
                # Multi-size batch
                lr_stack = batch['lr'].to(self.device)
                gt = batch['gt'].to(self.device)
                size_key = batch.get('size_key', 'unknown')
                batch_filenames = batch.get('filenames', [])
            else:
                # Traditional single-size batch (tuple: lr, gt, filenames)
                lr_stack, gt, batch_filenames = batch
                lr_stack = lr_stack.to(self.device)
                gt = gt.to(self.device)
                # Get the actual resolution key from the dataset if possible
                if hasattr(self.train_loader, 'dataset') and hasattr(self.train_loader.dataset, 'size_key'):
                    size_key = self.train_loader.dataset.size_key
                else:
                    size_key = 'default'
            
            # Accumulation steps come from ADAPTIVE_BATCH_CONFIG in config.
            _batch_cfg = self.config.get('ADAPTIVE_BATCH_CONFIG', {}).get(size_key)
            current_accum_steps = _batch_cfg['accum'] if _batch_cfg is not None else self.config.get('ACCUMULATION_STEPS', 4)

            # ── Size-key transition: enforce clean accumulation boundaries ────
            # If the resolution block changes mid-accumulation (e.g. due to a
            # crash-resume or an imperfect sampler block), discard any partially
            # accumulated gradients and reset the display buffer so the WebUI
            # never shows files from different resolutions in the same window.
            if size_key != prev_size_key:
                if accum_counter > 0:
                    # Discard partial gradients — never mix resolutions
                    self.optimizer.zero_grad()
                    accum_counter = 0
                current_window_batches = []
                display_files = []
                display_fps = {'720': 0, '540': 0, '720_169': 0}
            prev_size_key = size_key
            # ── End size-key transition ───────────────────────────────────────
            
            # Track batch files for WebUI display — always update counters, filenames when available
            if hasattr(self, 'web_monitor') and self.web_monitor:
                batch_size_val = lr_stack.size(0)
                # Accumulate per-size file usage for the current epoch
                epoch_files_per_size[size_key] = epoch_files_per_size.get(size_key, 0) + batch_size_val
                # files_used: sum of all images seen so far this epoch across all
                # forward passes (batch_idx counts forward passes, not optimizer steps)
                files_used_in_epoch = (batch_idx + 1) * batch_size_val
                
                # total_files: use the sampler's exact count when available so that
                # different physical batch sizes per size are accounted for correctly
                # (e.g. BS=2 for 540/720_169, BS=1 for 720).
                _sampler = getattr(self.train_loader, 'sampler', None)
                if _sampler is not None and hasattr(_sampler, 'total_files'):
                    total_files_in_epoch = _sampler.total_files
                elif _sampler is not None:
                    total_files_in_epoch = len(_sampler) * batch_size_val
                elif hasattr(self.train_loader, '__len__'):
                    total_files_in_epoch = len(self.train_loader) * batch_size_val
                else:
                    total_files_in_epoch = steps_per_epoch * batch_size_val
                
                # Accumulate filenames for the current optimizer-step window
                if batch_filenames:
                    formatted_files = [f"{size_key}/{fn}" for fn in batch_filenames]
                    current_window_batches.append({
                        'size_key': size_key,
                        'files': formatted_files,
                    })
                    # When the window is complete, snapshot it as the *committed*
                    # display and reset so the next window starts fresh.
                    if len(current_window_batches) >= current_accum_steps:
                        display_files = [
                            f for item in current_window_batches for f in item['files']
                        ]
                        display_fps = {'720': 0, '540': 0, '720_169': 0}
                        for item in current_window_batches:
                            sk = item['size_key']
                            display_fps[sk] = display_fps.get(sk, 0) + len(item['files'])
                        current_window_batches = []  # ready for next window

                # Always show the CURRENT iteration's files: merge the committed
                # display_files with any in-progress batches from the current
                # (not-yet-complete) accumulation window so that every forward
                # pass appears in the WebUI immediately, regardless of batch size
                # or accumulation depth.
                live_files = display_files + [
                    f for item in current_window_batches for f in item['files']
                ]
                # Build per-size counts in a single pass (O(n)) instead of
                # re-scanning the list once per size key (O(n·m)).
                live_fps = {'720': 0, '540': 0, '720_169': 0}
                for f in display_files:
                    sk_prefix = f.split('/', 1)[0]
                    if sk_prefix in live_fps:
                        live_fps[sk_prefix] += 1
                for item in current_window_batches:
                    sk_item = item['size_key']
                    live_fps[sk_item] = live_fps.get(sk_item, 0) + len(item['files'])

                # Update web_monitor with current batch info.
                # live_files always reflects ALL files of the current iteration
                # (complete windows + the in-progress window), so the display is
                # never blank mid-accumulation.
                self.web_monitor.data_store.update_all_metrics(
                    current_batch={
                        'files': live_files,
                        'size_key': size_key,
                        'batch_size': batch_size_val,
                        'files_used_in_epoch': files_used_in_epoch,
                        'total_files_in_epoch': total_files_in_epoch,
                        'files_per_size': live_fps,
                        'epoch_files_per_size': dict(epoch_files_per_size),
                        'accumulation_steps': current_accum_steps,
                        'accum_step': accum_counter + 1,
                    }
                )
            
            # Forward pass with mixed precision
            with autocast('cuda', enabled=self.use_amp):
                output = self.model(lr_stack)
                
                # Compute L1 loss for adaptive system
                with torch.no_grad():
                    current_l1 = torch.abs(output - gt).mean().item()
                
                # Get adaptive weights (now returns 5 values including status)
                l1_w, ms_w, grad_w, perceptual_w, adaptive_status = self.adaptive_system.update_loss_weights(
                    output, gt, self.global_step, 
                    current_l1_loss=current_l1
                )
                
                # ── Graduated perceptual loss scheduling ────────────────────
                # Phase 1/2: override adaptive weight with the scheduled ramp.
                # Phase 3: get_perceptual_weight returns None, so the
                # AdaptiveSystem's dynamic weight is used unchanged.
                # crop_file_counts is passed so Phase 2 stays locked while
                # crops don't exist yet (see DataStrategyScheduler.can_introduce_crops).
                if self.data_strategy_scheduler is not None:
                    scheduled_perceptual_w = self.data_strategy_scheduler.get_perceptual_weight(
                        self.global_step,
                        crop_file_counts=self._get_crop_file_counts()
                    )
                    if scheduled_perceptual_w is not None:
                        perceptual_w = scheduled_perceptual_w
                        # Keep adaptive_status in sync for logging
                        adaptive_status['perceptual_weight'] = perceptual_w
                        # Suppress AdaptiveSystem's 0.05 floor so the scheduled
                        # value (which may be 0.0 in Phase 1) is fully respected.
                        self.adaptive_system.set_perceptual_floor(0.0)
                    else:
                        # Phase 3: restore AdaptiveSystem's autonomous floor.
                        self.adaptive_system.set_perceptual_floor(0.05)
                # ── End graduated perceptual loss scheduling ─────────────────
                
                # Compute loss
                loss_dict = self.loss_fn(output, gt, l1_w, ms_w, grad_w, perceptual_w)
                loss = loss_dict['total']
                
                # Scale loss for accumulation
                loss = loss / current_accum_steps
            
            # Backward pass with gradient scaling
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update optimizer (every current_accum_steps using running counter)
            accum_counter += 1
            if accum_counter >= current_accum_steps:
                # Unscale gradients before clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                
                # Clip gradients
                grad_norm, clip_val = self.adaptive_system.clip_gradients(self.model)
                
                # Update optimizer with scaler
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Extract AdamW momentum (exp_avg) for GUI
                adam_momentum = self._get_adam_momentum()
                
                self.optimizer.zero_grad()
                accum_counter = 0  # Reset after successful optimizer step
                
                # Update LR scheduler (every LR_UPDATE_EVERY steps)
                lr_update_every = self.config.get('LR_UPDATE_EVERY', 10)
                if self.global_step % lr_update_every == 0:
                    # Bug 5 fix: combine train-plateau and val-plateau signals.
                    # Val-plateau uses a gentler ×0.7 LR reduction (vs ×0.5 for train-plateau).
                    plateau_detected = self.adaptive_system.is_plateau()
                    val_plateau_detected = self.adaptive_system.is_val_plateau()
                    current_lr, lr_phase = self.lr_scheduler.step(
                        self.global_step,
                        plateau_detected=plateau_detected,
                        val_plateau_detected=val_plateau_detected,
                    )
                    
                    # Log LR Boost events
                    lr_status = self.lr_scheduler.get_status()
                    if lr_phase == 'plateau_boost':
                        self.tb_logger.log_event(
                            self.global_step, 
                            'LR_Boost', 
                            f"LR boosted at step {self.global_step}"
                        )
                        self.train_logger.log_event(f"⚡ LR BOOST triggered at step {self.global_step}")
                    elif lr_phase == 'val_plateau_hold':
                        self.train_logger.log_event(f"🔽 LR reduced (val plateau) at step {self.global_step}")
                else:
                    # Keep current LR
                    current_lr = self.lr_scheduler.get_current_lr()
                    lr_phase = self.lr_scheduler.get_current_phase()
                
                # Update plateau tracker.
                # Only block during the LR warmup ramp (~1000 steps).
                # The DataStrategy Phase-1 duration (WARMUP_END) is intentionally
                # NOT included here: the plateau tracker must detect stagnation
                # on full-frame data so that aggressive_mode can fire well before
                # crops are introduced.  If we also blocked until WARMUP_END the
                # tracker would stay frozen until step 10 000 and aggressive mode
                # could never intervene during Phase 1.
                _lr_warmup = getattr(self.lr_scheduler, 'warmup_steps', 1000)
                _effective_warmup = _lr_warmup

                self.adaptive_system.update_plateau_tracker(
                    loss_dict['total'].item() if torch.is_tensor(loss_dict['total']) else loss_dict['total'],
                    quality=self.last_validation_quality,
                    step=self.global_step,
                    warmup_steps=_effective_warmup
                )
                
                # Get activity
                self.last_activities = self.model.get_layer_activity()
                
                # Measure performance (sync GPU first to capture async operations)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_time = time.time() - loop_start_time
                self.step_times.append(step_time)
                
                avg_time = sum(self.step_times) / len(self.step_times)
                
                # Reset loop timer for next iteration
                loop_start_time = time.time()
                # memory_reserved() = memory that PyTorch's caching allocator holds from
                # the CUDA driver.  This matches what nvidia-smi reports for the Python
                # process, unlike memory_allocated() (only live tensors, ~0 between steps)
                # or mem_get_info() total-free (sums ALL GPU processes, overshoots).
                vram = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0.0

                # Update per-size VRAM tracker with EMA (α=0.1) so the WebUI
                # batch-config table shows a stable measured value per resolution.
                if vram > 0.0 and size_key not in ('unknown', 'default'):
                    prev = self._vram_per_size.get(size_key)
                    self._vram_per_size[size_key] = (
                        0.9 * prev + 0.1 * vram if prev is not None else vram
                    )
                
                # Track loss history (raw values)
                raw_total_loss = loss_dict['total'].item() if torch.is_tensor(loss_dict['total']) else loss_dict['total']
                self.loss_history.append(raw_total_loss)
                if len(self.loss_history) > 1000:
                    self.loss_history.pop(0)
                
                # Apply EMA smoothing for GUI
                smoothed_loss_dict = self._apply_ema_smoothing(loss_dict)
                
                # Increment step
                self.global_step += 1
                current_epoch_step += 1
                
                # Update GUI with smoothed values
                self._update_gui(epoch, smoothed_loss_dict, avg_time, steps_per_epoch, current_epoch_step, adam_momentum=adam_momentum)
                
                # TensorBoard logging (use RAW values, not smoothed)
                if self.global_step % self.config.get('LOG_TBOARD_EVERY', 100) == 0:
                    self.tb_logger.log_losses(self.global_step, loss_dict)
                    self.tb_logger.log_lr(self.global_step, current_lr)
                    
                    # Get adaptive status and add LR boost availability
                    adaptive_status = self.adaptive_system.get_status()
                    lr_status = self.lr_scheduler.get_status()
                    adaptive_status['lr_boost_available'] = lr_status['plateau_boost_available']
                    
                    self.tb_logger.log_adaptive(self.global_step, adaptive_status)
                    self.tb_logger.log_system(self.global_step, avg_time, vram)
                    self.tb_logger.log_gradients(self.global_step, grad_norm, self.last_activities)
                    self.tb_logger.log_arch_block_metrics(self.global_step, self.last_activities)
                    self.tb_logger.log_lr_phase(self.global_step, lr_phase)
                    
                    # Log adaptive mode/phase transitions
                    current_adaptive_mode = adaptive_status.get('mode', 'Stable')
                    phase_changed = current_adaptive_mode.lower() != self._last_adaptive_mode.lower()
                    self.tb_logger.log_training_phase(self.global_step, {
                        'phase': current_adaptive_mode.lower(),
                        'phase_changed': phase_changed,
                    })
                    if phase_changed:
                        self.train_logger.log_event(
                            f"Adaptive mode changed: {self._last_adaptive_mode} → {current_adaptive_mode} at step {self.global_step}"
                        )
                        self._last_adaptive_mode = current_adaptive_mode
                    
                    # Log data strategy scheduler info to TensorBoard
                    if self.data_strategy_scheduler is not None:
                        sched = self.data_strategy_scheduler
                        _crop_counts_tb = self._get_crop_file_counts()
                        sched_perc_w = sched.get_perceptual_weight(
                            self.global_step, crop_file_counts=_crop_counts_tb
                        )
                        # Only log the scheduled weight during Phase 1/2 (not None)
                        if sched_perc_w is not None:
                            self.tb_logger.writer.add_scalar(
                                'DataStrategy/PerceptualWeight', sched_perc_w, self.global_step
                            )
                        sampler = getattr(self.train_loader, 'sampler', None)
                        if sampler is not None:
                            dist = sched.get_distribution(
                                self.global_step,
                                available_sizes=getattr(sampler, 'active_sizes', None),
                                crop_file_counts=_crop_counts_tb
                            )
                            # dist is None in Phase 3 (natural file-count sampling)
                            if dist is not None:
                                for sk, w in dist.items():
                                    self.tb_logger.writer.add_scalar(
                                        f'DataStrategy/Weight_{sk}', w, self.global_step
                                    )
                    
                    # Log plateau state details
                    if hasattr(self.adaptive_system, 'get_plateau_info'):
                        plateau_info = self.adaptive_system.get_plateau_info()
                        self.tb_logger.log_plateau_state(self.global_step, plateau_info)
                    
                    # Log weight statistics
                    weights = {
                        'l1': adaptive_status.get('loss_weights', (0.6, 0.2, 0.2))[0],
                        'ms': adaptive_status.get('loss_weights', (0.6, 0.2, 0.2))[1],
                        'grad': adaptive_status.get('loss_weights', (0.6, 0.2, 0.2))[2],
                        'perceptual': adaptive_status.get('perceptual_weight', 0.0)
                    }
                    self.tb_logger.log_weight_statistics(self.global_step, weights)
                    
                    # Log VRAM usage every 100 steps
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
                        
                        # Log to TensorBoard
                        self.tb_logger.writer.add_scalar('Memory/Allocated_GB', allocated, self.global_step)
                        self.tb_logger.writer.add_scalar('Memory/Reserved_GB', reserved, self.global_step)
                        self.tb_logger.writer.add_scalar('Memory/Peak_GB', max_allocated, self.global_step)
                        
                        # Print to console every 500 steps
                        if self.global_step % 500 == 0:
                            print(f"  📊 VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {max_allocated:.2f}GB peak")

                    # ── Mid-epoch data strategy update (every 100 steps) ─────────
                    # Re-apply the sampler distribution so that phase transitions
                    # (warmup → crop_introduction → stable) take effect immediately
                    # within the epoch instead of waiting until the next epoch start.
                    if self.data_strategy_scheduler is not None:
                        _mid_sampler = getattr(self.train_loader, 'sampler', None)
                        _mid_crop_counts = self._get_crop_file_counts()
                        if _mid_sampler is not None and hasattr(_mid_sampler, 'set_distribution'):
                            _mid_dist = self.data_strategy_scheduler.get_distribution(
                                self.global_step,
                                available_sizes=getattr(_mid_sampler, 'active_sizes', None),
                                crop_file_counts=_mid_crop_counts,
                            )
                            _mid_sampler.set_distribution(_mid_dist)
                        # Log phase transition if one occurred
                        self.data_strategy_scheduler.check_phase_transition(
                            self.global_step,
                            log_fn=self.train_logger.log_event,
                            crop_file_counts=_mid_crop_counts,
                        )
                        # Push current phase name to WebUI
                        _current_phase = self.data_strategy_scheduler.get_phase(
                            self.global_step, crop_file_counts=_mid_crop_counts
                        )
                        if hasattr(self, 'web_monitor') and self.web_monitor is not None:
                            self.web_monitor.data_store.update_all_metrics(
                                data_strategy_phase=_current_phase,
                            )
                    # ── End mid-epoch data strategy update ───────────────────────
                
                # Status file update (every 5 steps)
                if self.global_step % 5 == 0:
                    self.train_logger.update_status(
                        self.global_step, epoch, loss_dict, current_lr, avg_time, vram,
                        self.config.get('MODEL_CONFIG', {}), self.last_metrics, 
                        self.adaptive_system.get_status()
                    )
                
                # Validation
                if self.global_step % self.config.get('VAL_STEP_EVERY', 500) == 0:
                    self.train_logger.log_event(f"Running validation at step {self.global_step}")
                    
                    if getattr(self, 'use_async_validation', False):
                        # Async path: save weights-only checkpoint and signal the
                        # async validator process.  Training continues immediately.
                        self._request_async_validation()
                    else:
                        # Synchronous path (default): block until validation completes.
                        # --- timing start: covers forward pass + all I/O incl. TensorBoard ---
                        _val_cycle_start = time.time()
                        
                        metrics = self._run_multi_size_validation()
                        self.last_metrics = metrics

                        # Keep last_validation_quality in sync so the plateau
                        # tracker receives the real quality on every update call.
                        self.last_validation_quality = metrics.get('ki_quality', None)

                        # Feed validation loss into the validation plateau tracker.
                        self.adaptive_system.update_validation_tracker(
                            metrics.get('val_loss'),
                            metrics.get('ki_quality')
                        )
                        
                        # Pass improvement to adaptive system for logging
                        adaptive_status = self.adaptive_system.get_status()
                        adaptive_status['ki_improvement'] = metrics.get('improvement', 0)
                        
                        # Log to TensorBoard with dashboards
                        self.tb_logger.log_quality(self.global_step, metrics)
                        self.tb_logger.log_metrics(self.global_step, metrics)
                        self.tb_logger.log_validation_loss(self.global_step, metrics.get('val_loss', 0.0))
                        self.tb_logger.log_adaptive(self.global_step, adaptive_status)
                        
                        # Log validation event
                        self.tb_logger.log_validation_event(self.global_step, metrics)
                        
                        # Log ALL images (like in original)
                        labeled_images = metrics.get('labeled_images')
                        if labeled_images is not None and len(labeled_images) > 0:
                            print(f"📊 Logging {len(labeled_images)} validation images to TensorBoard...")
                            logged_count = 0
                            failed_count = 0
                            
                            for tag, img_tensor in labeled_images:
                                try:
                                    # Ensure tensor is in correct format for TensorBoard
                                    if img_tensor.device.type != 'cpu':
                                        img_tensor = img_tensor.cpu()
                                    if not img_tensor.is_contiguous():
                                        img_tensor = img_tensor.contiguous()
                                    
                                    self.tb_logger.writer.add_image(
                                        tag, 
                                        img_tensor, 
                                        self.global_step
                                    )
                                    logged_count += 1
                                except Exception as e:
                                    failed_count += 1
                                    print(f"⚠️  Failed to log validation image {tag}: {e}")
                                    self.train_logger.log_event(
                                        f"Warning: Failed to log validation image {tag}: {e}"
                                    )
                                    continue
                            
                            # Flush to ensure images are written
                            self.tb_logger.writer.flush()
                            
                            if failed_count == 0:
                                print(f"✅ Successfully logged all {logged_count} validation images to TensorBoard")
                            else:
                                print(f"⚠️  Logged {logged_count}/{len(labeled_images)} images ({failed_count} failed)")
                            
                            # CRITICAL: Remove labeled_images from metrics to prevent memory leak
                            del labeled_images
                            metrics.pop('labeled_images', None)
                            
                            import gc
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        else:
                            print("⚠️  No labeled images to log to TensorBoard")
                        
                        # --- timing end: full cycle including TensorBoard flush ---
                        _val_cycle_elapsed = time.time() - _val_cycle_start
                        self._update_val_speed(_val_cycle_elapsed)

                        self.train_logger.log_event(
                            f"Step {self.global_step} | Validation | "
                            f"KI Quality: {metrics['ki_quality']*100:.1f}%"
                        )
                        
                        # Best-checkpoint check (must precede _push_val_metrics_to_store
                        # so best_quality_ever is final before the atomic write).
                        if self.checkpoint_mgr.should_check_best(self.global_step):
                            print(f"\n💾 Checking if this is a new best checkpoint...")
                            is_new_best = self.checkpoint_mgr.update_best_checkpoint(
                                self.model, self.optimizer, self.lr_scheduler, 
                                self.global_step, metrics['ki_quality'], metrics,
                                self.train_logger.log_file
                            )
                            if is_new_best:
                                print(f"✅ New best checkpoint saved!")
                                self.tb_logger.log_checkpoint(self.global_step, 'best')
                            else:
                                print(f"   (Not better than current best)")

                        # Single atomic write of all quality metrics to the data store.
                        # Must happen BEFORE _update_gui() (which is called with no
                        # loss_dict and would zero out loss values in the store).
                        self._push_val_metrics_to_store(metrics, self.global_step)

                        # Write Statistik_<step>.json immediately – while the data
                        # store still holds the real training losses from the last
                        # training step.  The no-arg _update_gui() call below would
                        # overwrite those losses with 0.0, which is why we save here.
                        self._save_statistics_json(self.global_step)

                        # Reset timing after validation
                        loop_start_time = time.time()
                        
                        # Redraw terminal UI (called with no loss_dict → losses show
                        # as 0 in the terminal display until the next real step, but
                        # the data store / JSON are already correct).
                        self._update_gui()

                # Poll for async validation results (every step, cheap file-existence check)
                if getattr(self, 'use_async_validation', False):
                    self._poll_async_val_result()
                
                # Check dataset files every 100 steps
                if self.global_step % 100 == 0:
                    self._check_dataset_files()
                
                # Auto-continue timer for manual validation
                    if self.do_manual_val:
                        import select
                        from vsr_plusplus_NEU.utils.ui_terminal import C_CYAN, C_BOLD, C_GREEN, C_RESET, C_YELLOW
                        
                        # Show results
                        val_duration = time.time() - self.last_val_time if hasattr(self, 'last_val_time') else 0
                        print(f"\n{C_CYAN}{'='*80}{C_RESET}")
                        print(f"{C_BOLD}📊 VALIDATION RESULTS{C_RESET}")
                        print(f"{C_CYAN}{'-'*80}{C_RESET}")
                        print(f"  Loss:           {C_GREEN}{metrics['val_loss']:.6f}{C_RESET}")
                        print(f"  Duration:       {val_duration:.2f}s")
                        print(f"{C_CYAN}{'-'*80}{C_RESET}")
                        print(f"  {C_BOLD}QUALITY SCORES:{C_RESET}")
                        print(f"  LR Quality:     {C_YELLOW}{metrics['lr_quality']*100:.1f}%{C_RESET}  (PSNR: {metrics['lr_psnr']:.2f} dB, SSIM: {metrics['lr_ssim']*100:.1f}%)")
                        print(f"  KI Quality:     {C_GREEN}{metrics['ki_quality']*100:.1f}%{C_RESET}  (PSNR: {metrics['ki_psnr']:.2f} dB, SSIM: {metrics['ki_ssim']*100:.1f}%)")
                        
                        # Display improvement (sum of per-image KI-LR)
                        imp = metrics['improvement'] * 100
                        imp_sign = "+" if imp >= 0 else ""
                        imp_color = C_GREEN if imp >= 0 else C_RED
                        print(f"  Improvement (Sum): {C_BOLD}{imp_color}{imp_sign}{imp:.1f}%{C_RESET}")
                        
                        # Display GT differences if available
                        if 'ki_to_gt' in metrics and 'lr_to_gt' in metrics:
                            ki_gt = metrics['ki_to_gt'] * 100
                            lr_gt = metrics['lr_to_gt'] * 100
                            print(f"  KI to GT (Sum): {C_CYAN}{ki_gt:+.1f}%{C_RESET}")
                            print(f"  LR to GT (Sum): {C_CYAN}{lr_gt:+.1f}%{C_RESET}")
                        
                        print(f"{C_CYAN}{'='*80}{C_RESET}\n")
                        
                        # Auto-continue timer (10 seconds)
                        import sys
                        print(f"{C_YELLOW}Auto-continue in 10s (Press ENTER to skip)...{C_RESET}", end='', flush=True)
                        start_wait = time.time()
                        while time.time() - start_wait < 10.0:
                            if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                                sys.stdin.read(1)  # Enter pressed
                                break
                            remaining = int(10.0 - (time.time() - start_wait))
                            if remaining >= 0:
                                print(f"\r{C_YELLOW}Auto-continue in {remaining}s (Press ENTER to skip)...{C_RESET}", end='', flush=True)
                        print()  # New line
                        
                        # Reset flag
                        self.do_manual_val = False
                        
                        # Redraw UI
                        self._update_gui(epoch, loss_dict, avg_time, steps_per_epoch, current_epoch_step, self.paused)
                
                # Regular checkpoint
                if self.checkpoint_mgr.should_save_regular(self.global_step):
                    print(f"\n💾 Saving regular checkpoint at step {self.global_step:,}...")
                    self.checkpoint_mgr.save_checkpoint(
                        self.model, self.optimizer, self.lr_scheduler,
                        self.global_step, self.last_metrics or {},
                        self.train_logger.log_file
                    )
                    print(f"✅ Regular checkpoint saved!")
                    self.tb_logger.log_checkpoint(self.global_step, 'regular')
                    
                    # Redraw UI after save
                    self._update_gui()
                
                # Check if training complete
                if self.global_step >= self.config.get('MAX_STEPS', 100000):
                    return
        
        # End of epoch - check for new dataset files
        # This ensures video list is updated at natural epoch boundaries
        print(f"\n📊 End of epoch {epoch} - checking for new dataset files...")
        self._check_dataset_files()
    
    def _update_gui(self, epoch=1, loss_dict=None, avg_time=0.1, steps_per_epoch=1, current_epoch_step=0, paused=False, adam_momentum=0.0):
        """Update the GUI display"""
        # Get activities
        activities = get_activity_data(self.model)
        
        # Prepare loss dict
        losses = {
            'l1': loss_dict.get('l1', 0.0) if loss_dict else 0.0,
            'ms': loss_dict.get('ms', 0.0) if loss_dict else 0.0,
            'grad': loss_dict.get('grad', 0.0) if loss_dict else 0.0,
            'perceptual': loss_dict.get('perceptual', 0.0) if loss_dict else 0.0,
            'total': loss_dict.get('total', 0.0) if loss_dict else 0.0,
        }
        
        # Convert tensor to float if needed
        for k, v in losses.items():
            if torch.is_tensor(v):
                losses[k] = v.item()
        
        # Get LR info
        current_lr = self.optimizer.param_groups[0]['lr']
        lr_phase = getattr(self.lr_scheduler, 'current_phase', 'unknown')
        lr_info = {'lr': current_lr, 'phase': lr_phase}
        
        # Quality metrics
        quality_metrics = None
        if self.last_metrics:
            quality_metrics = {
                'lr_quality': self.last_metrics.get('lr_quality', 0.0) * 100,  # Convert to %
                'ki_quality': self.last_metrics.get('ki_quality', 0.0) * 100,
                'improvement': self.last_metrics.get('improvement', 0.0) * 100,
            }
            # Add GT difference metrics if available
            if 'ki_to_gt' in self.last_metrics:
                quality_metrics['ki_to_gt'] = self.last_metrics.get('ki_to_gt', 0.0) * 100
            if 'lr_to_gt' in self.last_metrics:
                quality_metrics['lr_to_gt'] = self.last_metrics.get('lr_to_gt', 0.0) * 100
        
        # Adaptive status
        adaptive_status = self.adaptive_system.get_status()
        
        # Number of training images
        # Handle both single-size DataLoader and MultiSizeDataLoader
        if hasattr(self.train_loader, 'dataset'):
            # Old single-size DataLoader
            num_images = len(self.train_loader.dataset)
        else:
            # New MultiSizeDataLoader - sum all datasets
            num_images = sum(len(ds) for ds in self.train_loader.datasets_dict.values())
        
        # Calculate ETAs
        from ..utils.ui_terminal import format_time
        
        if paused:
            total_eta = "PAUSED"
            epoch_eta = "PAUSED"
        else:
            # Total ETA
            remaining_steps = self.config['MAX_STEPS'] - self.global_step
            total_eta = format_time(remaining_steps * avg_time)
            
            # Epoch ETA
            remaining_epoch_steps = steps_per_epoch - current_epoch_step
            epoch_eta = format_time(remaining_epoch_steps * avg_time)
        
        # Draw UI
        draw_ui(
            step=self.global_step,
            epoch=epoch,
            losses=losses,
            it_time=avg_time,
            activities=activities,
            config=self.config,
            num_images=num_images,
            steps_per_epoch=steps_per_epoch,
            current_epoch_step=current_epoch_step,
            adaptive_status=adaptive_status,
            paused=paused,
            quality_metrics=quality_metrics,
            lr_info=lr_info,
            total_eta=total_eta,
            epoch_eta=epoch_eta,
            adam_momentum=adam_momentum,
            val_iter_per_sec=self.last_val_iter_per_sec
        )
        
        # Update web monitor with COMPLETE training state (ALL data)
        best_quality = self.checkpoint_mgr.best_quality if self.checkpoint_mgr.best_quality > 0 else 0.0
        # memory_reserved() = memory held by PyTorch's caching allocator from the CUDA
        # driver.  This matches nvidia-smi's per-process column, unlike mem_get_info()
        # total-free which sums ALL GPU processes and therefore overshoots.
        gpu_mem = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0.0
        
        # Konvertiere Layer-Aktivitäten in Dict-Format
        layer_act_dict = {}
        peak_activity_value = 0.0
        if activities:
            for name, activity_percent, trend, raw_value in activities:
                # Send RAW VALUES to web UI (not normalized percentages)
                layer_act_dict[name] = raw_value
                # Track maximum raw value across all layers
                peak_activity_value = max(peak_activity_value, raw_value)
        
        # Perceptual trend: +1 wenn EMA quality steigt, -1 wenn fallend, 0 sonst
        _ema_q = getattr(self.adaptive_system, 'ema_quality', None)
        _best_q = getattr(self.adaptive_system, 'best_quality', None)
        if _ema_q is not None and _best_q is not None and _best_q > 0:
            _trend_ratio = _ema_q / _best_q
            if _trend_ratio > 1.001:   # >0.1% above best → improving
                _perceptual_trend = 1
            elif _trend_ratio < 0.998:  # >0.2% below best → declining
                _perceptual_trend = -1
            else:
                _perceptual_trend = 0
        else:
            _perceptual_trend = 0
        
        # Debug: Print first update to verify data flow
        if self.global_step == 1:
            print(f"\n🔍 Web UI Debug - First Update:")
            print(f"   Step: {self.global_step}")
            print(f"   Total Loss: {losses['total']}")
            print(f"   LR: {current_lr}")
            print(f"   VRAM: {gpu_mem:.2f} GB")
            print(f"   Layer activities: {len(layer_act_dict)} layers")
        
        try:
            self.web_monitor.update(
                # Grundlegende Metriken
                step_current=self.global_step,
                epoch_num=epoch,
                step_max=self.config.get('MAX_STEPS', 100000),
                epoch_step_current=current_epoch_step,
                epoch_step_total=steps_per_epoch,
                
                # Verluste
                total_loss_value=losses['total'],
                l1_loss_value=losses['l1'],
                ms_loss_value=losses['ms'],
                gradient_loss_value=losses['grad'],
                perceptual_loss_value=losses['perceptual'],
                
                # Adaptive Gewichte
                l1_weight_current=adaptive_status.get('l1_weight', 1.0),
                ms_weight_current=adaptive_status.get('ms_weight', 1.0),
                gradient_weight_current=adaptive_status.get('grad_weight', 1.0),
                perceptual_weight_current=adaptive_status.get('perceptual_weight', 0.0),
                gradient_clip_val=adaptive_status.get('grad_clip', 1.0),
                
                # Adaptive Status (NEW)
                adaptive_mode=adaptive_status.get('mode', 'Stable'),
                adaptive_is_cooldown=adaptive_status.get('is_cooldown', False),
                adaptive_cooldown_remaining=adaptive_status.get('cooldown_remaining', 0),
                adaptive_plateau_counter=adaptive_status.get('plateau_counter', 0),
                adaptive_plateau_patience=adaptive_status.get('plateau_patience', 100),
                adaptive_lr_boost_available=adaptive_status.get('lr_boost_available', False),
                adaptive_perceptual_trend=_perceptual_trend,
                # Validation plateau tracking (Bug 5 fix)
                adaptive_val_no_improve_count=adaptive_status.get('val_no_improve_count', 0),
                adaptive_val_plateau_patience=adaptive_status.get('val_plateau_patience', 5),
                adaptive_best_val_loss=adaptive_status.get('best_val_loss', None),
                adaptive_ema_val_loss=adaptive_status.get('ema_val_loss', None),
                adaptive_is_val_plateau=adaptive_status.get('is_val_plateau', False),
                
                # Lernrate
                learning_rate_value=current_lr,
                lr_phase_name=lr_phase,
                
                # Performance
                iteration_duration=avg_time,
                vram_usage_gb=gpu_mem,
                adam_momentum_avg=adam_momentum,
                val_iter_per_sec=self.last_val_iter_per_sec,
                
                # Zeitschätzungen
                eta_total_formatted=total_eta,
                eta_epoch_formatted=epoch_eta,
                
                # Quality-Metriken
                quality_lr_value=quality_metrics.get('lr_quality', 0.0) / 100.0 if quality_metrics else 0.0,
                quality_ki_value=quality_metrics.get('ki_quality', 0.0) / 100.0 if quality_metrics else 0.0,
                quality_improvement_value=quality_metrics.get('improvement', 0.0) / 100.0 if quality_metrics else 0.0,
                quality_ki_to_gt_value=quality_metrics.get('ki_to_gt', 0.0) / 100.0 if quality_metrics else 0.0,
                quality_lr_to_gt_value=quality_metrics.get('lr_to_gt', 0.0) / 100.0 if quality_metrics else 0.0,
                validation_loss_value=self.last_metrics.get('val_loss', 0.0) if self.last_metrics else 0.0,
                best_quality_ever=best_quality,
                
                # Layer-Aktivitäten
                layer_activity_map=layer_act_dict,
                layer_activity_peak_value=peak_activity_value,
                # TemporalAlign flow magnitudes – extracted from layer_act_dict so the
                # dedicated top-level JSON fields always match what the web UI shows.
                align_backward_flow=layer_act_dict.get('Backward Align', 0.0),
                align_forward_flow=layer_act_dict.get('Forward Align', 0.0),
                
                # Batch-Konfiguration mit gemessenen VRAM-Werten
                adaptive_batch_config=self._build_ui_batch_config(),
                
                # Prefetch pipeline queue statistics
                prefetch_stats=self._collect_prefetch_stats(),

                # Crop-wait status (system-level pause waiting for crop images)
                crop_wait_active=self.waiting_for_crops,
                crop_wait_current_count=self._crop_wait_current_count,
                crop_wait_needed_count=DataStrategyScheduler.MIN_CROP_FILES_TRAINING
                    if self.data_strategy_scheduler is not None else 0,
                crop_wait_next_check_secs=max(0, int(self._crop_wait_next_check - time.time()))
                    if self.waiting_for_crops else 0,

                # Status
                training_active=not (paused or self.waiting_for_crops),
                validation_running=False,
                training_paused=paused or self.waiting_for_crops
            )
        except Exception as e:
            # Log error but don't crash training
            print(f"\n⚠️  Web UI update failed: {e}")
            import traceback
            traceback.print_exc()

    def _build_ui_batch_config(self) -> dict:
        """Build the adaptive_batch_config dict for the WebUI.

        Merges the static ADAPTIVE_BATCH_CONFIG from self.config with the
        live per-size VRAM measurements collected in self._vram_per_size.
        The WebUI table therefore always shows the actual measured values
        instead of the hardcoded placeholders from web_ui.py.
        """
        result = {}
        for sk, v in self.config.get('ADAPTIVE_BATCH_CONFIG', {}).items():
            entry = {
                'batch': v['batch'],
                'accum': v['accum'],
                'effective': v['batch'] * v['accum'],
            }
            measured = self._vram_per_size.get(sk)
            if measured is not None:
                entry['vram_gb'] = round(measured, 2)
            elif 'vram_gb' in v:
                entry['vram_gb'] = v['vram_gb']
            result[sk] = entry
        return result

    def _collect_prefetch_stats(self) -> dict:
        """Return the current prefetch pipeline queue fill levels.

        Reads ``MultiSizeDataLoader.prefetch_stats`` and returns a dict ready
        for the WebUI.  Returns a disabled-state dict when the loader is not
        yet initialised or prefetch is turned off.
        """
        if self.train_loader is None:
            return {'enabled': False}
        stats = getattr(self.train_loader, 'prefetch_stats', None)
        if stats is None:
            return {'enabled': False}
        return stats

    def _apply_ema_smoothing(self, loss_dict):
        """
        Apply exponential moving average smoothing to losses for GUI display
        
        Args:
            loss_dict: Dictionary of current loss values
        
        Returns:
            Dictionary of smoothed loss values
        """
        # Initialize EMA on first call
        if self.ema_loss is None:
            self.ema_loss = {}
            for key in loss_dict:
                val = loss_dict[key]
                self.ema_loss[key] = val.item() if torch.is_tensor(val) else val
        
        # Update EMA
        smoothed = {}
        for key in loss_dict:
            val = loss_dict[key]
            raw_val = val.item() if torch.is_tensor(val) else val
            
            # EMA formula: smoothed = alpha * smoothed_prev + (1 - alpha) * current
            self.ema_loss[key] = self.ema_factor * self.ema_loss[key] + (1 - self.ema_factor) * raw_val
            smoothed[key] = self.ema_loss[key]
        
        return smoothed

    def _get_crop_file_counts(self):
        """
        Return a dict mapping size_key → number of files currently loaded
        for that size, derived from the train-loader's sampler.

        Used by DataStrategyScheduler.can_introduce_crops() to gate Phase 2
        on crop files actually existing on disk (not just on step count).

        Returns:
            dict or None if the sampler does not expose ``datasets_dict``.
        """
        sampler = getattr(self.train_loader, 'sampler', None)
        if sampler is None or not hasattr(sampler, 'datasets_dict'):
            return None
        return {sk: len(sampler.datasets_dict[sk]) for sk in sampler.active_sizes}

    def _check_crop_readiness(self, force_rescan=False):
        """Check whether enough crop GT images exist to proceed with Phase 2.

        Optionally triggers a full filesystem rescan before the check.
        Updates ``self.waiting_for_crops`` and ``self._crop_wait_current_count``.

        Args:
            force_rescan: When True, call ``_check_dataset_files()`` first so
                          newly generated crop files are detected.

        Returns:
            True  – enough crops available (or crop check doesn't apply).
            False – still waiting; ``self.waiting_for_crops`` is set to True.
        """
        if self.data_strategy_scheduler is None:
            self.waiting_for_crops = False
            return True

        # The crop-wait guard only activates at/past WARMUP_END.
        if self.global_step < DataStrategyScheduler.WARMUP_END:
            self.waiting_for_crops = False
            return True

        if force_rescan:
            self._check_dataset_files()

        crop_counts = self._get_crop_file_counts()
        total = DataStrategyScheduler.get_crop_total_count(crop_counts)
        self._crop_wait_current_count = total

        if DataStrategyScheduler.has_enough_training_crops(crop_counts):
            if self.waiting_for_crops:
                self.train_logger.log_event(
                    f"✅ Crop-Wait beendet: {total:,} Crop-Bilder vorhanden "
                    f"(Mindest: {DataStrategyScheduler.MIN_CROP_FILES_TRAINING:,})"
                )
            self.waiting_for_crops = False
            return True
        else:
            self.waiting_for_crops = True
            return False
    
    def _get_adam_momentum(self):
        """
        Extract average momentum (exp_avg) from AdamW optimizer state.

        Returns:
            float: Average L2-norm of the first moment (exp_avg) across all
                   parameter tensors that currently have a gradient.

        Why is this value typically very small (e.g. 0.0002)?
        -------------------------------------------------------
        AdamW stores exp_avg[i] = β₁ · exp_avg[i-1] + (1-β₁) · grad[i],
        i.e. a per-parameter exponential moving average of raw gradients.
        The L2 norm of a single parameter tensor's exp_avg reflects the
        *magnitude* of the gradients for that tensor only.

        In a stable training phase the raw gradients are small (the model
        has converged to a local minimum and makes only tiny corrections),
        so their EMA is likewise small.  Averaging the L2 norm over all
        parameter tensors (many of which are large weight matrices with
        many near-zero entries) produces a value that is typically several
        orders of magnitude below 1.  A reading of ~0.0003 is therefore
        expected and correct — it indicates that training is stable, NOT
        that the optimizer is broken or that momentum is not being tracked.
        """
        total_momentum = 0.0
        count = 0
        
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.optimizer.state[p]
                if 'exp_avg' in state:
                    # Get the exponential moving average of gradients (momentum)
                    exp_avg = state['exp_avg']
                    # Calculate magnitude (L2 norm)
                    momentum_mag = exp_avg.norm().item()
                    total_momentum += momentum_mag
                    count += 1
        
        # Return average momentum magnitude
        return total_momentum / count if count > 0 else 0.0
    
    def _check_keyboard_input(self, epoch, steps_per_epoch, current_epoch_step):
        """Check for keyboard input and web commands"""
        key = self.keyboard.check_key_pressed(timeout=0)
        
        if key:
            key_lower = key.lower()
            
            if key_lower == '\r' or key_lower == '\n':  # ENTER
                # Show live config menu
                self.config = self.keyboard.show_live_menu(self.config, self.optimizer, self)
            
            elif key_lower == 's':  # Switch display mode
                current_mode = self.config.get('DISPLAY_MODE', 0)
                self.config['DISPLAY_MODE'] = (current_mode + 1) % 4
            
            elif key_lower == 'p':  # Pause/Resume
                self.paused = not self.paused
                # Note: Timing will be reset by loop_start_time assignment at line 95
            
            elif key_lower == 'v':  # Manual validation
                self.do_manual_val = True
            
            elif key_lower == 'c':  # Manual checkpoint save
                self._save_checkpoint()
        
        # Check for web UI commands (new method name)
        web_cmd = self.web_monitor.poll_commands()
        if web_cmd == 'validate':
            self.do_manual_val = True
        elif web_cmd == 'save_checkpoint':
            # Trigger immediate checkpoint save
            self._save_checkpoint()
        elif web_cmd == 'toggle_pause':
            if self.waiting_for_crops:
                # User pressed "Resume" while the crop-wait guard is active.
                # Do an immediate rescan; only lift the wait if crops are ready.
                ready = self._check_crop_readiness(force_rescan=True)
                if ready:
                    self.paused = False
                    self.train_logger.log_event(
                        f"✅ Crop-Wait beendet nach manueller Prüfung: "
                        f"{self._crop_wait_current_count:,} Bilder verfügbar."
                    )
                else:
                    needed = DataStrategyScheduler.MIN_CROP_FILES_TRAINING
                    self.train_logger.log_event(
                        f"⚠️ Fortsetzen nicht möglich: nur "
                        f"{self._crop_wait_current_count:,}/{needed:,} "
                        f"Crop-Bilder vorhanden."
                    )
            else:
                self.paused = not self.paused
                status = "paused" if self.paused else "resumed"
                self.train_logger.log_event(f"Training {status} at step {self.global_step}")
        elif web_cmd == 'check_crops_now':
            # Trigger an immediate crop rescan (fired by "Jetzt prüfen" in WebGUI).
            self._crop_wait_next_check = 0.0  # causes the loop to rescan on next iteration
        elif web_cmd == 'run_video_test':
            # Trigger video inference
            self._run_video_inference()
    
    def _save_checkpoint(self):
        """Save checkpoint immediately"""
        try:
            metrics = self.last_metrics or {}
            self.checkpoint_mgr.save_checkpoint(
                self.model,
                self.optimizer,
                self.lr_scheduler,
                self.global_step,
                metrics,
                self.train_logger.log_file,
            )
            self.train_logger.log_event(f"Manual checkpoint saved at step {self.global_step}")
        except Exception as e:
            self.train_logger.log_event(f"Failed to save checkpoint: {str(e)}")

    # ------------------------------------------------------------------
    # Async validation helpers
    # ------------------------------------------------------------------

    def enable_async_validation(self, checkpoint_dir, val_sizes, log_dir):
        """
        Switch the trainer to asynchronous validation mode.

        In this mode the scheduled validation (every VAL_STEP_EVERY steps)
        no longer blocks the training loop.  Instead the trainer:
          1. Saves a lightweight model-weights-only checkpoint.
          2. Writes an ``async_val_request.json`` sentinel file.
          3. Continues training immediately.

        The separate AsyncValidationProcess (running on a second GPU) picks up
        the sentinel, runs full validation (including TensorBoard image writes),
        and writes ``async_val_result.json``.  The trainer polls for that file
        at every step and ingests the results when they arrive.

        Manual validation (key 'v' / WebUI button) always runs synchronously so
        the user can get immediate feedback on demand.

        Args:
            checkpoint_dir: Directory shared with the async validator process
                            (used for IPC sentinel files and weights checkpoints).
            val_sizes:      List of size-key strings that the async validator
                            should validate (e.g. ['540', '720']).
            log_dir:        TensorBoard log directory forwarded to the async
                            validator so it can write events to the same run.
        """
        self.use_async_validation = True
        self._async_val_checkpoint_dir = checkpoint_dir
        self._async_val_sizes = val_sizes
        self._async_val_log_dir = log_dir
        self._async_val_last_ingested_step = -1
        self.train_logger.log_event(
            f"Async validation enabled (checkpoint_dir={checkpoint_dir})"
        )

    def _request_async_validation(self):
        """
        Save a model-weights-only checkpoint and write the async_val_request
        sentinel so the secondary validation process picks it up.

        This is a fast operation (just one torch.save + one JSON write) and
        returns immediately so the training loop is not blocked.
        """
        checkpoint_dir = self._async_val_checkpoint_dir
        step = self.global_step

        # Save weights-only checkpoint (much smaller than full checkpoint)
        weights_path = os.path.join(checkpoint_dir, f'async_val_weights_{step:07d}.pth')
        try:
            torch.save(self.model.state_dict(), weights_path)
        except Exception as e:
            self.train_logger.log_event(
                f"⚠ Async val: failed to save weights checkpoint: {e}"
            )
            return

        # Determine data root and dataset name from config
        data_root = self.config.get('DATA_ROOT', '')
        dataset_name = self.config.get('DEFAULT_DATASET_NAME', 'master')

        # Build config snapshot (only the fields needed by the async validator)
        config_snapshot = {
            'N_FEATS':            self.config.get('N_FEATS', 72),
            'N_BLOCKS':           self.config.get('N_BLOCKS', 28),
            'USE_CHECKPOINTING':  self.config.get('USE_CHECKPOINTING', False),
            'L1_WEIGHT':          self.config.get('L1_WEIGHT', 0.60),
            'MS_WEIGHT':          self.config.get('MS_WEIGHT', 0.20),
            'GRAD_WEIGHT':        self.config.get('GRAD_WEIGHT', 0.20),
            'PERCEPTUAL_WEIGHT':  self.config.get('PERCEPTUAL_WEIGHT', 0.0),
        }

        request = {
            'step':            step,
            'checkpoint_path': weights_path,
            'data_root':       data_root,
            'dataset_name':    dataset_name,
            'val_sizes':       getattr(self, '_async_val_sizes', ['540']),
            'log_dir':         getattr(self, '_async_val_log_dir', ''),
            'config_snapshot': config_snapshot,
        }

        request_file = os.path.join(checkpoint_dir, 'async_val_request.json')
        tmp_file = request_file + '.tmp'
        try:
            with open(tmp_file, 'w') as f:
                json.dump(request, f)
            os.replace(tmp_file, request_file)
            self.train_logger.log_event(
                f"Async val request written for step {step}"
            )
        except Exception as e:
            self.train_logger.log_event(
                f"⚠ Async val: failed to write request file: {e}"
            )

    def _poll_async_val_result(self):
        """
        Check for a completed async validation result and ingest it.

        Reads ``async_val_result.json`` from the checkpoint directory.  If the
        file is present and contains results for a step that has not yet been
        ingested, the metrics are applied to the adaptive system and logged to
        TensorBoard — exactly as the synchronous path would do.

        If the result file contains an ``'error'`` key (written by the async
        validator when model loading or inference fails), the error is logged
        visibly so the user knows what went wrong.  Quality metrics are NOT
        updated in the error case so the data store retains the last valid
        values.

        The result file is removed after ingestion to avoid double-processing.
        """
        checkpoint_dir = getattr(self, '_async_val_checkpoint_dir', None)
        if checkpoint_dir is None:
            return

        result_file = os.path.join(checkpoint_dir, 'async_val_result.json')
        if not os.path.exists(result_file):
            return

        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
        except (json.JSONDecodeError, OSError):
            return  # File might still be written — retry next step

        step = result.get('step', -1)
        if step <= self._async_val_last_ingested_step:
            return  # Already processed

        # Consume the file immediately
        try:
            os.unlink(result_file)
        except OSError:
            pass

        self._async_val_last_ingested_step = step

        # ── Error result: the async validator failed — log and bail out ───────
        if 'error' in result:
            error_msg = result['error']
            print(f"\n❌ [AsyncVal] Validation for step {step} FAILED: {error_msg}")
            self.train_logger.log_event(
                f"[AsyncVal] ERROR at step {step}: {error_msg}"
            )
            # Do NOT update quality metrics — keep previous valid values.
            return

        # Clean up the weights-only checkpoint for this step (no longer needed)
        weights_path = os.path.join(checkpoint_dir, f'async_val_weights_{step:07d}.pth')
        if os.path.exists(weights_path):
            try:
                os.unlink(weights_path)
            except OSError:
                pass

        # --- Feed metrics into adaptive system (same as synchronous path) ---
        self.last_metrics = result
        ki_quality = result.get('ki_quality')
        self.last_validation_quality = ki_quality

        self.adaptive_system.update_validation_tracker(
            result.get('val_loss'),
            ki_quality
        )

        # Log to TensorBoard (images are already written by the async process)
        self.tb_logger.log_quality(step, result)
        self.tb_logger.log_metrics(step, result)
        self.tb_logger.log_validation_loss(step, result.get('val_loss', 0.0))

        adaptive_status = self.adaptive_system.get_status()
        adaptive_status['ki_improvement'] = result.get('improvement', 0)
        self.tb_logger.log_adaptive(step, adaptive_status)

        # Best-checkpoint check (must precede _push_val_metrics_to_store so
        # best_quality_ever is final before the atomic write to the data store).
        if self.checkpoint_mgr.should_check_best(step):
            if ki_quality is not None:
                is_new_best = self.checkpoint_mgr.update_best_checkpoint(
                    self.model, self.optimizer, self.lr_scheduler,
                    step, ki_quality, result,
                    self.train_logger.log_file
                )
                if is_new_best:
                    print(f"\n✅ [AsyncVal] New best checkpoint at step {step}!")
                    self.tb_logger.log_checkpoint(step, 'best')

        # Update val-speed tracker.
        val_elapsed = result.get('val_elapsed_seconds')
        if val_elapsed is not None and val_elapsed > 0:
            self._update_val_speed(val_elapsed)

        # Single atomic write of all quality metrics to the data store.
        # Called here – after the best-checkpoint check – so best_quality_ever
        # is already the final value.  Using `step` (the original validation
        # step, e.g. 15000) as last_validation_step so the export snapshot
        # correctly identifies which model the quality numbers belong to.
        self._push_val_metrics_to_store(result, step)

        # Write Statistik_<step>.json immediately.
        # No delay needed: _push_val_metrics_to_store already flushed all quality
        # data to the store, and _update_gui (which runs at the TOP of each loop
        # iteration, before _poll_async_val_result) has already refreshed the
        # training-loss fields with this iteration's real values.
        self._save_statistics_json(step)

        ki_pct = ki_quality * 100 if ki_quality is not None else 0.0
        self.train_logger.log_event(
            f"[AsyncVal] Ingested step {step} | KI Quality: {ki_pct:.1f}%"
        )
        print(f"\n📥 [AsyncVal] Results ingested for step {step} | "
              f"KI Quality: {ki_pct:.1f}%")

    def _count_val_samples(self):
        """Return total number of validation samples across all loaded val datasets."""
        total = 0
        val_loaders = getattr(self, 'val_loaders', None)
        if val_loaders:
            for _, loader in val_loaders:
                if hasattr(loader, 'dataset'):
                    total += len(loader.dataset)
        elif hasattr(self, 'val_loader') and self.val_loader is not None:
            if hasattr(self.val_loader, 'dataset'):
                total += len(self.val_loader.dataset)
        return max(total, 1)

    def _update_val_speed(self, elapsed_seconds):
        """
        Update the rolling validation-speed tracker.

        elapsed_seconds is the wall-clock time for a full validation cycle
        (forward pass + metric computation + TensorBoard writes).
        We spread it evenly over all validated samples and append one
        per-sample duration to the rolling window of 200 entries.

        The resulting ``last_val_iter_per_sec`` gives the average throughput
        over the last 200 GT/LR pairs processed, *including* all I/O time.
        """
        total_samples = self._count_val_samples()
        if elapsed_seconds <= 0 or total_samples <= 0:
            return

        per_sample_sec = elapsed_seconds / total_samples

        # Extend the rolling window.  We add one entry per sample (capped at 200)
        # so each cycle contributes weight proportional to how many samples it
        # processed.  deque(maxlen=200) evicts old entries automatically.
        # Adding the same per_sample_sec value multiple times is intentional:
        # it represents the measured throughput for all validated samples in this
        # cycle and weights the rolling average accordingly.
        entries_to_add = min(total_samples, 200)
        for _ in range(entries_to_add):
            self._val_sample_timings.append(per_sample_sec)

        # Compute speed from the window
        if self._val_sample_timings:
            avg_per_sample = sum(self._val_sample_timings) / len(self._val_sample_timings)
            self.last_val_iter_per_sec = 1.0 / avg_per_sample if avg_per_sample > 0 else 0.0

    def _push_val_metrics_to_store(self, metrics: dict, val_step: int) -> None:
        """
        Atomically write completed validation metrics into the web-monitor data store.

        This is the **single authoritative write point** for all quality / validation
        fields.  It must be called only *after* all post-validation processing has
        finished (including ``checkpoint_mgr.update_best_checkpoint``) so that
        ``best_quality_ever`` is already up to date before the atomic write.

        Scaling convention (matches ``_update_gui``):
          - ``quality_*_value`` : raw 0-1 fraction as returned by the validator.
          - ``quality_improvement_value`` / ``*_to_gt_value`` : raw sums from validator.
          - ``validation_loss_value`` : raw float loss.
          - ``best_quality_ever`` : read from ``checkpoint_mgr`` after checkpoint check.

        The call also sets ``has_validation_data = True`` and
        ``last_validation_step = val_step`` so that ``get_export_snapshot()``
        can expose a ``validation_step`` field that clearly shows which model
        snapshot the quality numbers belong to.
        """
        best_quality = (
            self.checkpoint_mgr.best_quality
            if self.checkpoint_mgr.best_quality > 0 else 0.0
        )
        self.web_monitor.data_store.update_all_metrics(
            # Quality fractions (0-1, raw validator output)
            quality_lr_value=metrics.get('lr_quality', 0.0),
            quality_ki_value=metrics.get('ki_quality', 0.0),
            # Raw sums (not per-image averages – match existing web_monitor convention)
            quality_improvement_value=metrics.get('improvement', 0.0),
            quality_ki_to_gt_value=metrics.get('ki_to_gt', 0.0),
            quality_lr_to_gt_value=metrics.get('lr_to_gt', 0.0),
            # Scalar loss
            validation_loss_value=metrics.get('val_loss', 0.0),
            # Best-ever quality (read after checkpoint update so it is final)
            best_quality_ever=best_quality,
            # Provenance: mark that real data is present and record the step
            has_validation_data=True,
            last_validation_step=val_step,
        )

    def _run_validation(self):
        """Run validation immediately"""
        self.train_logger.log_event(f"Manual validation triggered at step {self.global_step}")
        
        # --- timing start: covers forward pass + all I/O incl. TensorBoard ---
        _val_cycle_start = time.time()

        metrics = self._run_multi_size_validation()

        # Bug 1 fix: update last_validation_quality so the plateau tracker uses it.
        self.last_validation_quality = metrics.get('ki_quality', None)

        # Bug 5 fix: feed validation loss into the validation plateau tracker.
        self.adaptive_system.update_validation_tracker(
            metrics.get('val_loss'),
            metrics.get('ki_quality')
        )
        
        # Log to TensorBoard
        self.tb_logger.log_quality(self.global_step, metrics)
        self.tb_logger.log_metrics(self.global_step, metrics)
        self.tb_logger.log_validation_loss(self.global_step, metrics.get('val_loss', 0.0))
        
        # Log ALL images (like in original)
        labeled_images = metrics.get('labeled_images')
        if labeled_images is not None:
            for tag, img_tensor in labeled_images:
                self.tb_logger.writer.add_image(
                    tag, 
                    img_tensor, 
                    self.global_step
                )
            self.tb_logger.writer.flush()
            
            # CRITICAL: Remove labeled_images to prevent memory leak
            del labeled_images
            metrics.pop('labeled_images', None)
            
            # Force cleanup
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # --- timing end: full cycle including TensorBoard flush ---
        _val_cycle_elapsed = time.time() - _val_cycle_start
        self._update_val_speed(_val_cycle_elapsed)

        # Store metrics WITHOUT labeled_images
        self.last_metrics = metrics

        # Best checkpoint check (must happen before _push_val_metrics_to_store
        # so that best_quality_ever is up to date when we do the atomic write).
        if self.checkpoint_mgr.should_check_best(self.global_step):
            self.checkpoint_mgr.update_best_checkpoint(
                self.model, self.optimizer, self.lr_scheduler,
                self.global_step, metrics['ki_quality'], metrics,
                self.train_logger.log_file
            )

        # Single atomic write of all validation metrics to the data store.
        # Called AFTER the best-checkpoint check so best_quality_ever is final.
        self._push_val_metrics_to_store(metrics, self.global_step)

        # Write Statistik_<step>.json immediately.
        # The JSON is written BEFORE the no-arg _update_gui() call that follows
        # in the training loop so the snapshot still contains the real training
        # losses from the last training step (the no-arg call would zero them out).
        self._save_statistics_json(self.global_step)
        
        self.train_logger.log_event(
            f"Manual Validation | KI Quality: {metrics['ki_quality']*100:.1f}%"
        )
        
        self.model.train()  # Back to training mode
    
    def _check_dataset_files(self):
        """
        Check for new files in training and validation datasets and reload if found.
        
        This method is called every 100 steps to detect new files added by parallel extraction.
        When new files are detected, datasets are reloaded to include them in training.
        
        Updates web monitor with current file counts per size and distribution.
        """
        try:
            dataset_info = {
                'train_per_size': {},  # Per-size breakdown for training
                'val': {},
                'distribution': {},  # Current distribution from config
                'last_check': self.global_step
            }
            
            # Distribution will be calculated from actual file counts after gathering data
            # (No longer using size_distribution from config)
            
            # Check training dataset
            if hasattr(self.train_loader, 'dataset'):
                # Standard DataLoader with single dataset
                try:
                    train_ds = self.train_loader.dataset
                    
                    # Verify dataset has required methods
                    if not hasattr(train_ds, 'get_file_info') or not hasattr(train_ds, 'check_for_new_files'):
                        print(f"⚠️  Warning: Training dataset missing file monitoring methods")
                    else:
                        train_info = train_ds.get_file_info()
                        train_changes = train_ds.check_for_new_files()
                        
                        size_key = train_info['size_key']
                        dataset_info['train_per_size'][size_key] = {
                            'count': train_info['file_count'],
                            'has_new': train_changes['has_new'],
                            'new_count': train_changes['new_files']
                        }
                        
                        if train_changes['has_new']:
                            print(f"\n📂 New training files detected for {size_key}: +{train_changes['new_files']} files")
                            print(f"   Total files in directory: {train_changes['new_gt_count']}")
                            print(f"   Currently loaded: {train_changes['current_loaded']}")
                            print(f"   🔄 Reloading dataset...")
                            
                            if hasattr(train_ds, 'reload_files'):
                                reload_result = train_ds.reload_files()
                                if reload_result['success']:
                                    print(f"   ✅ Reload successful: {reload_result['files_before']} → {reload_result['files_after']} files")
                                    dataset_info['train_per_size'][size_key]['count'] = reload_result['files_after']
                                    if hasattr(self, 'train_logger') and self.train_logger:
                                        self.train_logger.log_event(
                                            f"Reloaded {size_key} training: +{reload_result['new_files_loaded']} files"
                                        )
                                else:
                                    print(f"   ❌ Reload failed: {reload_result.get('error', 'Unknown error')}")
                            else:
                                print(f"   ⚠️  Dataset does not support reload_files()")
                except Exception as e:
                    print(f"⚠️  Error checking training dataset: {e}")
                    import traceback
                    traceback.print_exc()
                        
            elif hasattr(self.train_loader, 'datasets_dict'):
                # MultiSizeDataLoader with multiple datasets
                # Check and reload each size separately
                try:
                    for size_key, train_ds in self.train_loader.datasets_dict.items():
                        try:
                            # Verify dataset has required methods
                            if not hasattr(train_ds, 'get_file_info') or not hasattr(train_ds, 'check_for_new_files'):
                                print(f"⚠️  Warning: Training dataset {size_key} missing file monitoring methods")
                                continue
                                
                            train_info = train_ds.get_file_info()
                            train_changes = train_ds.check_for_new_files()
                            
                            dataset_info['train_per_size'][size_key] = {
                                'count': train_info['file_count'],
                                'has_new': train_changes['has_new'],
                                'new_count': train_changes['new_files']
                            }
                            
                            if train_changes['has_new']:
                                print(f"\n📂 New training files detected for {size_key}: +{train_changes['new_files']} files")
                                print(f"   Total files in directory: {train_changes['new_gt_count']}")
                                print(f"   Currently loaded: {train_changes['current_loaded']}")
                                print(f"   🔄 Reloading {size_key} dataset...")
                                
                                if hasattr(train_ds, 'reload_files'):
                                    reload_result = train_ds.reload_files()
                                    if reload_result['success']:
                                        print(f"   ✅ Reload successful: {reload_result['files_before']} → {reload_result['files_after']} files")
                                        dataset_info['train_per_size'][size_key]['count'] = reload_result['files_after']
                                        if hasattr(self, 'train_logger') and self.train_logger:
                                            self.train_logger.log_event(
                                                f"Reloaded {size_key} training: +{reload_result['new_files_loaded']} files"
                                            )
                                    else:
                                        print(f"   ❌ Reload failed: {reload_result.get('error', 'Unknown error')}")
                                else:
                                    print(f"   ⚠️  Dataset {size_key} does not support reload_files()")
                        except Exception as e:
                            print(f"⚠️  Error checking training dataset {size_key}: {e}")
                except Exception as e:
                    print(f"⚠️  Error iterating training datasets: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Check validation datasets
            val_loaders = getattr(self, 'val_loaders', None)
            if val_loaders and isinstance(val_loaders, list):
                # Multi-size validation
                try:
                    for size_key, val_loader in val_loaders:
                        try:
                            if hasattr(val_loader, 'dataset'):
                                val_ds = val_loader.dataset
                                
                                # Verify dataset has required methods
                                if not hasattr(val_ds, 'get_file_info') or not hasattr(val_ds, 'check_for_new_files'):
                                    print(f"⚠️  Warning: Validation dataset {size_key} missing file monitoring methods")
                                    continue
                                    
                                val_info = val_ds.get_file_info()
                                val_changes = val_ds.check_for_new_files()
                                
                                dataset_info['val'][size_key] = {
                                    'count': val_info['file_count'],
                                    'has_new': val_changes['has_new'],
                                    'new_count': val_changes['new_files']
                                }
                                
                                if val_changes['has_new']:
                                    delta = val_changes['new_files']
                                    delta_str = f"+{delta}" if delta >= 0 else str(delta)
                                    print(f"\n📂 Validation dataset changed for {size_key}: {delta_str} files (GT dir: {val_changes['new_gt_count']}, loaded: {val_changes['current_loaded']})")
                                    print(f"   🔄 Reloading {size_key} validation dataset...")

                                    if hasattr(val_ds, 'reload_files'):
                                        reload_result = val_ds.reload_files()
                                        if reload_result['success']:
                                            print(f"   ✅ Reload successful: {reload_result['files_before']} → {reload_result['files_after']} files")
                                            dataset_info['val'][size_key]['count'] = reload_result['files_after']
                                            if hasattr(self, 'train_logger') and self.train_logger:
                                                self.train_logger.log_event(
                                                    f"Reloaded {size_key} validation: {reload_result['files_before']} → {reload_result['files_after']} files"
                                                )
                                        else:
                                            print(f"   ❌ Reload failed: {reload_result.get('error', 'Unknown error')}")
                                    else:
                                        print(f"   ⚠️  Validation dataset {size_key} does not support reload_files()")
                        except Exception as e:
                            print(f"⚠️  Error checking validation dataset {size_key}: {e}")
                except Exception as e:
                    print(f"⚠️  Error iterating validation datasets: {e}")
                    import traceback
                    traceback.print_exc()
                                
            elif hasattr(self, 'val_loader') and hasattr(self.val_loader, 'dataset'):
                # Single validation loader
                try:
                    val_ds = self.val_loader.dataset
                    
                    # Verify dataset has required methods
                    if not hasattr(val_ds, 'get_file_info') or not hasattr(val_ds, 'check_for_new_files'):
                        print(f"⚠️  Warning: Validation dataset missing file monitoring methods")
                    else:
                        val_info = val_ds.get_file_info()
                        val_changes = val_ds.check_for_new_files()
                        
                        size_key = val_info.get('size_key', '540')
                        dataset_info['val'][size_key] = {
                            'count': val_info['file_count'],
                            'has_new': val_changes['has_new'],
                            'new_count': val_changes['new_files']
                        }
                        
                        if val_changes['has_new']:
                            delta = val_changes['new_files']
                            delta_str = f"+{delta}" if delta >= 0 else str(delta)
                            print(f"\n📂 Validation dataset changed for {size_key}: {delta_str} files (GT dir: {val_changes['new_gt_count']}, loaded: {val_changes['current_loaded']})")
                            print(f"   🔄 Reloading validation dataset...")

                            if hasattr(val_ds, 'reload_files'):
                                reload_result = val_ds.reload_files()
                                if reload_result['success']:
                                    print(f"   ✅ Reload successful: {reload_result['files_before']} → {reload_result['files_after']} files")
                                    dataset_info['val'][size_key]['count'] = reload_result['files_after']
                                    if hasattr(self, 'train_logger') and self.train_logger:
                                        self.train_logger.log_event(
                                            f"Reloaded validation: {reload_result['files_before']} → {reload_result['files_after']} files"
                                        )
                                else:
                                    print(f"   ❌ Reload failed: {reload_result.get('error', 'Unknown error')}")
                            else:
                                print(f"   ⚠️  Validation dataset does not support reload_files()")
                except Exception as e:
                    print(f"⚠️  Error checking validation dataset: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Calculate distribution from actual file counts
            total_train_files = sum(info['count'] for info in dataset_info['train_per_size'].values())
            if total_train_files > 0:
                for size_key, info in dataset_info['train_per_size'].items():
                    dataset_info['distribution'][size_key] = info['count'] / total_train_files
            
            # Update web monitor
            if hasattr(self, 'web_monitor') and self.web_monitor:
                try:
                    self.web_monitor.data_store.update_all_metrics(dataset_files=dataset_info)
                except Exception as e:
                    print(f"⚠️  Error updating web monitor: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️  Web monitor not available")
            
        except Exception as e:
            print(f"⚠️  Error checking dataset files: {e}")
            import traceback
            traceback.print_exc()
    
    def _run_video_inference(self):
        """
        Run video inference on test video using sliding window approach
        
        Processes testvideo.mkv from DATA_ROOT and saves output as testvideo_step_[STEP].mkv
        Uses FFmpeg to extract frames, processes with model, and merges audio back in
        """
        import subprocess
        import tempfile
        import shutil
        import cv2
        import numpy as np
        
        # Get DATA_ROOT from config
        data_root = self.config.get('DATA_ROOT', './Learn')
        input_video_path = os.path.join(data_root, 'testvideo.mkv')
        output_video_path = os.path.join(data_root, f'testvideo_step_{self.global_step}.mkv')
        
        # Check if input file exists
        if not os.path.exists(input_video_path):
            self.train_logger.log_event(f"⚠️  Video test skipped: {input_video_path} not found")
            return
        
        self.train_logger.log_event(f"🎬 Starting video test run at step {self.global_step}")
        
        # Save safety checkpoint before processing
        try:
            self._save_checkpoint()
            self.train_logger.log_event("💾 Safety checkpoint saved before video test")
        except Exception as e:
            self.train_logger.log_event(f"⚠️  Warning: Could not save safety checkpoint: {e}")
        
        # Set model to eval mode
        was_training = self.model.training
        self.model.eval()
        
        try:
            # Create temporary directory for frames
            with tempfile.TemporaryDirectory() as temp_dir:
                frames_dir = os.path.join(temp_dir, 'frames')
                output_frames_dir = os.path.join(temp_dir, 'output_frames')
                os.makedirs(frames_dir, exist_ok=True)
                os.makedirs(output_frames_dir, exist_ok=True)
                
                # Extract frames from video using FFmpeg
                self.train_logger.log_event("📹 Extracting frames from video...")
                extract_cmd = [
                    'ffmpeg', '-i', input_video_path,
                    '-vf', 'scale=180:180',  # Scale to LR size (180x180)
                    '-q:v', '1',  # High quality
                    os.path.join(frames_dir, 'frame_%05d.png')
                ]
                
                try:
                    subprocess.run(extract_cmd, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    self.train_logger.log_event(f"❌ FFmpeg extraction failed: {e.stderr.decode()}")
                    return
                
                # Get list of extracted frames
                frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
                total_frames = len(frame_files)
                
                if total_frames < 7:
                    self.train_logger.log_event(f"❌ Not enough frames ({total_frames} < 7)")
                    return
                
                self.train_logger.log_event(f"✅ Extracted {total_frames} frames")
                
                # Process frames with sliding window (7 frames)
                self.train_logger.log_event("🔄 Processing frames with 7-frame model...")
                processed_count = 0
                
                with torch.no_grad():
                    # We need 7 frames for the sliding window, with center frame (index 3) being upscaled
                    # Process from frame 3 to frame (total-3) to always have context
                    for i in range(3, total_frames - 3):
                        # Load 7 consecutive frames (i-3 to i+3, with i being center)
                        window_frames = []
                        for offset in range(-3, 4):  # -3, -2, -1, 0, 1, 2, 3
                            frame_path = os.path.join(frames_dir, frame_files[i + offset])
                            frame = cv2.imread(frame_path)
                            if frame is None:
                                self.train_logger.log_event(f"❌ Failed to load frame {frame_files[i + offset]}")
                                break
                            # Convert BGR to RGB and normalize to [0, 1]
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame = frame.astype(np.float32) / 255.0
                            window_frames.append(frame)
                        
                        if len(window_frames) != 7:
                            continue
                        
                        # Stack frames and convert to tensor [1, 7, 3, H, W]
                        frames_tensor = torch.from_numpy(np.stack(window_frames)).permute(0, 3, 1, 2).unsqueeze(0)
                        frames_tensor = frames_tensor.to(self.device)
                        
                        # Process through model
                        output = self.model(frames_tensor)  # [1, 3, 540, 540]
                        
                        # Convert output back to image
                        output_img = output[0].cpu().permute(1, 2, 0).numpy()
                        output_img = np.clip(output_img * 255.0, 0, 255).astype(np.uint8)
                        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
                        
                        # Save output frame (first iteration at i=3 produces frame_00000.png)
                        output_path = os.path.join(output_frames_dir, f'frame_{i-3:05d}.png')
                        cv2.imwrite(output_path, output_img)
                        
                        processed_count += 1
                        
                        # Log progress every 30 frames
                        if processed_count % 30 == 0:
                            self.train_logger.log_event(f"  Processed {processed_count}/{total_frames-6} frames...")
                
                self.train_logger.log_event(f"✅ Processed {processed_count} frames")
                
                # Create output video with FFmpeg and merge audio
                self.train_logger.log_event("🎞️  Creating output video with audio...")
                
                # First, create video from frames
                temp_video = os.path.join(temp_dir, 'temp_video.mkv')
                create_video_cmd = [
                    'ffmpeg', '-framerate', '24',
                    '-i', os.path.join(output_frames_dir, 'frame_%05d.png'),
                    '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                    '-pix_fmt', 'yuv420p',
                    '-y', temp_video
                ]
                
                try:
                    subprocess.run(create_video_cmd, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    self.train_logger.log_event(f"❌ Video creation failed: {e.stderr.decode()}")
                    return
                
                # Then merge with audio and metadata from original
                merge_cmd = [
                    'ffmpeg', '-i', temp_video,
                    '-i', input_video_path,
                    '-map', '0:v:0',  # Video from processed
                    '-map', '1:a?',   # Audio from original (if exists)
                    '-c:v', 'copy',   # Copy video codec
                    '-c:a', 'copy',   # Copy audio codec
                    '-y', output_video_path
                ]
                
                try:
                    subprocess.run(merge_cmd, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    # If audio merge fails, just save video without audio
                    self.train_logger.log_event(f"⚠️  Audio merge failed, saving video only: {e.stderr.decode()}")
                    shutil.copy(temp_video, output_video_path)
                
                self.train_logger.log_event(f"✅ Video test complete! Saved to: {output_video_path}")
                
        except Exception as e:
            self.train_logger.log_event(f"❌ Video test failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Restore model mode
            if was_training:
                self.model.train()
            self.train_logger.log_event("🔄 Training mode restored, continuing training...")
    
    def _save_statistics_json(self, step):
        """
        Save complete training statistics as JSON file.

        Saves to DATA_ROOT/Statistik_STEP.json.  The snapshot is taken from the
        web-monitor data store via ``get_export_snapshot()``, which:

          - Strips transient runtime fields (``val_status``, ``validation_running``)
            that are meaningless in a persisted file and could confuse analysis
            scripts that read the Statistik files.
          - Overrides ``step_current`` with *step* so the filename
            ``Statistik_{step}.json`` and the ``step_current`` field inside the
            file are always consistent (without the override the field would be
            2 steps ahead because of the 2-step save delay).

        Args:
            step: The validation step this file belongs to.
        """
        try:
            # Use the export snapshot (strips transient fields, aligns step_current)
            data_snapshot = self.web_monitor.data_store.get_export_snapshot(
                override_step=step
            )
            
            # Get DATA_ROOT from config (Learning directory)
            data_root = self.config.get('DATA_ROOT', './Learn')
            
            # Ensure directory exists
            os.makedirs(data_root, exist_ok=True)
            
            # Create filename: Statistik_STEP.json
            filename = f"Statistik_{step}.json"
            filepath = os.path.join(data_root, filename)
            
            # Save JSON with pretty formatting (same as web download)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_snapshot, f, indent=2, ensure_ascii=False)
            
            print(f"  📊 Statistics saved: {filename}")
            self.train_logger.log_event(f"Statistics JSON saved: {filename}")
            
        except Exception as e:
            print(f"  ⚠️  Failed to save statistics JSON: {e}")
            self.train_logger.log_event(f"Warning: Failed to save statistics JSON: {e}")
    
    def get_current_state(self):
        """Capture current training state for comparison"""
        return {
            'step': self.global_step,
            'total_loss': getattr(self, 'last_total_loss', None),
            'l1_loss': getattr(self, 'last_l1_loss', None),
            'quality_ki': getattr(self, 'last_validation_quality', None),
            'learning_rate': self.lr_scheduler.optimizer.param_groups[0]['lr'] if hasattr(self.lr_scheduler, 'optimizer') else 0.0,
            'plateau_counter': self.adaptive_system.plateau_counter,
            'timestamp': time.time()
        }
    
    def run_validation_snapshot(self, snapshot_name=None):
        """
        Run validation and save snapshot
        Used BEFORE config changes to capture baseline
        
        Args:
            snapshot_name: Optional name suffix (e.g., 'before_change')
        
        Returns:
            Validation results dict
        """
        # Run validation (use multi-size if available)
        val_results = self._run_multi_size_validation()
        
        # Capture current state
        state = self.get_current_state()
        state.update(val_results)
        
        # Save snapshot
        data_root = self.config.get('DATA_ROOT', './Learn')
        if snapshot_name:
            filename = f"Statistik_{self.global_step}_{snapshot_name}.json"
        else:
            filename = f"Statistik_{self.global_step}.json"
        
        filepath = os.path.join(data_root, filename)
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"📸 Validation snapshot saved: {filename}")
        return state

    def run(self):
        """
        Main training loop
        """
        self.train_logger.log_event("🚀 TRAINING STARTED")
        
        # Log initial configuration snapshot to TensorBoard — merge in
        # adaptive system params which are not stored in self.config
        snapshot_config = dict(self.config)
        if self.adaptive_system:
            snapshot_config['plateau_patience'] = self.adaptive_system.plateau_patience
            snapshot_config['plateau_safety_threshold'] = self.adaptive_system.plateau_safety_threshold
            snapshot_config['cooldown_duration'] = self.adaptive_system.cooldown_duration
        self.tb_logger.log_config_snapshot(snapshot_config)
        
        # Log initial hyperparameters if at step 0
        if self.global_step == 0:
            hparams = {
                'n_feats': self.config.get('N_FEATS', 128),
                'n_blocks': self.config.get('N_BLOCKS', 32),
                'batch_size': self.config.get('BATCH_SIZE', 4),
                'max_lr': self.config.get('MAX_LR', 1.5e-4),
                'min_lr': self.config.get('MIN_LR', 1e-6),
                'plateau_patience': snapshot_config.get('plateau_patience', 250),
            }
            # Will update metrics as training progresses
            initial_metrics = {'initial_step': 0}
            try:
                self.tb_logger.log_hyperparameters(hparams, initial_metrics)
            except Exception as e:
                # Hyperparameters might fail if already logged, continue anyway
                pass
        
        # Setup keyboard handler
        self.keyboard.setup_raw_mode()
        
        try:
            for epoch in range(1, 100000):
                self.train_epoch(epoch)
                
                if self.global_step >= self.config.get('MAX_STEPS', 100000):
                    self.train_logger.log_event("✅ TRAINING COMPLETED")
                    break
        
        except KeyboardInterrupt:
            print("\n")  # New line after ^C
            self.train_logger.log_event("⚠️  Training interrupted by user")
            
            # Restore terminal
            self.keyboard.restore_normal_mode()
            
            # Ask user if they want to save checkpoint
            save_choice = input(f"{C_YELLOW}Checkpoint speichern? (y/n): {C_RESET}").lower()
            
            if save_choice == 'y':
                print(f"{C_CYAN}💾 Saving interrupt checkpoint...{C_RESET}")
                self.checkpoint_mgr.save_emergency_checkpoint(
                    self.model, self.optimizer, self.lr_scheduler,
                    self.global_step, self.last_metrics or {},
                    self.train_logger.log_file
                )
                self.tb_logger.log_checkpoint(self.global_step, 'emergency')
                print(f"{C_GREEN}✅ Checkpoint saved!{C_RESET}")
            else:
                print(f"{C_YELLOW}Checkpoint not saved.{C_RESET}")
        
        except Exception as e:
            self.train_logger.log_event(f"❌ Training crashed: {e}")
            self.checkpoint_mgr.save_emergency_checkpoint(
                self.model, self.optimizer, self.lr_scheduler,
                self.global_step, self.last_metrics or {},
                self.train_logger.log_file
            )
            self.tb_logger.log_checkpoint(self.global_step, 'emergency')
            raise
        
        finally:
            # Restore terminal mode
            self.keyboard.restore_normal_mode()
            self.tb_logger.close()
            # Stop the web-monitoring HTTP server so its thread doesn't keep
            # the process alive (or trigger std::terminate during teardown).
            try:
                self.web_monitor.terminate()
            except Exception:
                pass

