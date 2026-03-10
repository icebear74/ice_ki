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
import torch
from torch.cuda.amp import autocast
from ..utils.ui_display import draw_ui, get_activity_data
from ..utils.keyboard_handler import KeyboardHandler
from ..utils.ui_terminal import C_GREEN, C_CYAN, C_YELLOW, C_RESET, show_cursor
from ..core.data_strategy import DataStrategyScheduler

# Fixed per-size batch and gradient accumulation configuration.
# These values are hardcoded to guarantee correct GPU memory usage.
# They are NOT read from config.py, runtime_config.json, or anywhere else.
# This is the single source of truth for batch/accum decisions inside the trainer.
#
#   720_169 (720×405) – 16:9 full frames:  BS=2, accum=4 → eff=8  (~5.14 GB)
#   540     (540×540) – 1080p crops:       BS=2, accum=3 → eff=6  (~5.15 GB)
#   720     (720×720) – 4K crops:          BS=1, accum=4 → eff=4  (~6.14 GB, BS=2 = OOM)
FIXED_BATCH_CONFIG = {
    '720_169': {'batch': 2, 'accum': 4},
    '540':     {'batch': 2, 'accum': 3},
    '720':     {'batch': 1, 'accum': 4},
}


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
        self.step_times = []
        
        # EMA for GUI smoothing (factor 0.95)
        self.ema_loss = None
        self.ema_factor = 0.95
        
        # UI state
        self.paused = False
        self.do_manual_val = False
        
        # Pending JSON save tracking (save after validation + N steps)
        self.pending_json_save_step = None
        
        # Adaptive mode change tracking for TensorBoard phase transition logging
        self._last_adaptive_mode = 'Stable'
        
        # Graduated data/loss strategy scheduler (optional)
        # Set via trainer.data_strategy_scheduler = DataStrategyScheduler(...)
        self.data_strategy_scheduler = None
        
        # Keyboard handler
        self.keyboard = KeyboardHandler()
        
        # Web interface for remote monitoring
        from ..systems.web_ui import WebMonitoringInterface
        self.web_monitor = WebMonitoringInterface(port_num=5050, refresh_seconds=5)
    
    def set_start_step(self, step):
        """Set starting step (for resume)"""
        self.start_step = step
        self.global_step = step
    
    def _run_multi_size_validation(self):
        """
        Run validation on all configured sizes
        Returns combined metrics averaging across all sizes
        """
        if not hasattr(self, 'val_loaders') or not self.val_loaders:
            # Fallback to single-size validation
            return self.validator.validate(self.global_step)
        
        print(f"\n{C_CYAN}Running multi-size validation on {len(self.val_loaders)} sizes...{C_RESET}")
        
        # Run validation on each size
        all_metrics = []
        all_labeled_images = []
        
        for size_key, val_loader in self.val_loaders:
            print(f"  Validating {size_key}...")
            
            # Temporarily swap loader
            original_loader = self.validator.val_loader
            self.validator.val_loader = val_loader
            
            # Run validation
            metrics = self.validator.validate(self.global_step)
            
            # Restore original loader
            self.validator.val_loader = original_loader
            
            # Collect labeled images with size prefix
            if 'labeled_images' in metrics and metrics['labeled_images'] is not None:
                # Build tag as val_{size_key}/{filename_stem}
                for name, img in metrics['labeled_images']:
                    all_labeled_images.append((f"val_{size_key}/{name}", img))
            
            # Store metrics
            all_metrics.append((size_key, metrics))
            print(f"    ✓ {size_key}: KI Quality {metrics['ki_quality']*100:.1f}%, PSNR {metrics['ki_psnr']:.2f}dB")
        
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

            if sampler is not None and hasattr(sampler, 'set_distribution'):
                dist = scheduler.get_distribution(
                    self.global_step,
                    available_sizes=sampler.active_sizes
                )
                sampler.set_distribution(dist)

            # Log phase transitions
            scheduler.check_phase_transition(
                self.global_step,
                log_fn=self.train_logger.log_event
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
        
        # Initialize loop timing
        loop_start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Handle pause state
            while self.paused:
                self._update_gui(epoch, {}, 0.1, steps_per_epoch, current_epoch_step, paused=True)
                time.sleep(0.5)
                self._check_keyboard_input(epoch, steps_per_epoch, current_epoch_step)
            
            # Check keyboard input
            self._check_keyboard_input(epoch, steps_per_epoch, current_epoch_step)

            
            # Manual validation trigger
            if self.do_manual_val:
                self._run_validation()
                self.do_manual_val = False
                # Reset timing after validation
                loop_start_time = time.time()
            
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
            
            # Accumulation steps come from FIXED_BATCH_CONFIG (module-level constant).
            # Never from self.config, runtime_config, or any external source.
            _fixed = FIXED_BATCH_CONFIG.get(size_key)
            current_accum_steps = _fixed['accum'] if _fixed is not None else 4

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
                    # When the window is complete, snapshot it for display and
                    # reset so the next window starts fresh.
                    if len(current_window_batches) >= current_accum_steps:
                        display_files = [
                            f for item in current_window_batches for f in item['files']
                        ]
                        display_fps = {'720': 0, '540': 0, '720_169': 0}
                        for item in current_window_batches:
                            sk = item['size_key']
                            display_fps[sk] = display_fps.get(sk, 0) + len(item['files'])
                        current_window_batches = []  # ready for next window
                
                # Update web_monitor with current batch info.
                # display_files/display_fps hold the last complete window so the
                # list is always either empty (epoch start, before first complete
                # window) or exactly current_accum_steps files of the same size.
                self.web_monitor.data_store.update_all_metrics(
                    current_batch={
                        'files': display_files,
                        'size_key': size_key,
                        'batch_size': batch_size_val,
                        'files_used_in_epoch': files_used_in_epoch,
                        'total_files_in_epoch': total_files_in_epoch,
                        'files_per_size': display_fps,
                        'accumulation_steps': current_accum_steps,
                        'accum_step': accum_counter + 1,
                    }
                )
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
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
                if self.data_strategy_scheduler is not None:
                    scheduled_perceptual_w = self.data_strategy_scheduler.get_perceptual_weight(
                        self.global_step
                    )
                    if scheduled_perceptual_w is not None:
                        perceptual_w = scheduled_perceptual_w
                        # Keep adaptive_status in sync for logging
                        adaptive_status['perceptual_weight'] = perceptual_w
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
                    plateau_detected = self.adaptive_system.is_plateau()
                    current_lr, lr_phase = self.lr_scheduler.step(self.global_step, plateau_detected)
                    
                    # Log LR Boost events
                    lr_status = self.lr_scheduler.get_status()
                    if lr_phase == 'plateau_boost':
                        self.tb_logger.log_event(
                            self.global_step, 
                            'LR_Boost', 
                            f"LR boosted at step {self.global_step}"
                        )
                        self.train_logger.log_event(f"⚡ LR BOOST triggered at step {self.global_step}")
                else:
                    # Keep current LR
                    current_lr = self.lr_scheduler.get_current_lr()
                    lr_phase = self.lr_scheduler.get_current_phase()
                
                # Update plateau tracker (only meaningful after BOTH warmup phases):
                # 1. LR Warmup (~2000 Steps) AND
                # 2. DataStrategy Phase 1 (WARMUP_END = 15000 Steps)
                _lr_warmup = getattr(self.lr_scheduler, 'warmup_steps', 1000)
                _data_warmup = (
                    DataStrategyScheduler.WARMUP_END
                    if self.data_strategy_scheduler is not None
                    else 0
                )
                _effective_warmup = max(_lr_warmup, _data_warmup)

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
                if len(self.step_times) > 100:
                    self.step_times.pop(0)
                
                avg_time = sum(self.step_times) / len(self.step_times)
                
                # Reset loop timer for next iteration
                loop_start_time = time.time()
                # Use current allocated memory (not peak) for per-step VRAM display.
                # Peak (max_memory_allocated) is logged separately to TensorBoard.
                vram = torch.cuda.memory_allocated() / (1024**3)
                
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
                
                # Check if we need to save JSON (delayed save after validation)
                if self.pending_json_save_step is not None and self.global_step >= self.pending_json_save_step:
                    self._save_statistics_json(self.pending_json_save_step - 2)  # Save with original validation step
                    self.pending_json_save_step = None  # Clear pending save
                
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
                        sched_perc_w = sched.get_perceptual_weight(self.global_step)
                        # Only log the scheduled weight during Phase 1/2 (not None)
                        if sched_perc_w is not None:
                            self.tb_logger.writer.add_scalar(
                                'DataStrategy/PerceptualWeight', sched_perc_w, self.global_step
                            )
                        sampler = getattr(self.train_loader, 'sampler', None)
                        if sampler is not None:
                            dist = sched.get_distribution(
                                self.global_step,
                                available_sizes=getattr(sampler, 'active_sizes', None)
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
                    
                    metrics = self._run_multi_size_validation()
                    self.last_metrics = metrics
                    
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
                                # Continue with other images even if one fails
                                continue
                        
                        # Flush to ensure images are written
                        self.tb_logger.writer.flush()
                        
                        # Summary
                        if failed_count == 0:
                            print(f"✅ Successfully logged all {logged_count} validation images to TensorBoard")
                        else:
                            print(f"⚠️  Logged {logged_count}/{len(labeled_images)} images ({failed_count} failed)")
                        
                        # CRITICAL: Remove labeled_images from metrics to prevent memory leak
                        # These images can be 2GB+ and were being held in self.last_metrics
                        del labeled_images
                        metrics.pop('labeled_images', None)
                        
                        # Force garbage collection and GPU memory cleanup
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        print("⚠️  No labeled images to log to TensorBoard")
                    
                    self.train_logger.log_event(
                        f"Step {self.global_step} | Validation | "
                        f"KI Quality: {metrics['ki_quality']*100:.1f}%"
                    )
                    
                    # Check for best checkpoint
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
                    
                    # Reset timing after validation
                    loop_start_time = time.time()
                    
                    # Redraw UI after validation completes
                    self._update_gui()
                    
                    # Schedule JSON save for 2 steps later (so web_monitor gets updated with fresh loss data)
                    self.pending_json_save_step = self.global_step + 2
                
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
            adam_momentum=adam_momentum
        )
        
        # Update web monitor with COMPLETE training state (ALL data)
        best_quality = self.checkpoint_mgr.best_quality if self.checkpoint_mgr.best_quality > 0 else 0.0
        # Use current allocated memory (not peak) so the web monitor shows
        # live VRAM, not the maximum since training start.
        gpu_mem = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        
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
                
                # Lernrate
                learning_rate_value=current_lr,
                lr_phase_name=lr_phase,
                
                # Performance
                iteration_duration=avg_time,
                vram_usage_gb=gpu_mem,
                adam_momentum_avg=adam_momentum,
                
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
                
                # Status
                training_active=not paused,
                validation_running=False,
                training_paused=paused
            )
        except Exception as e:
            # Log error but don't crash training
            print(f"\n⚠️  Web UI update failed: {e}")
            import traceback
            traceback.print_exc()
    
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
    
    def _get_adam_momentum(self):
        """
        Extract average momentum (exp_avg) from AdamW optimizer state
        
        Returns:
            float: Average momentum magnitude across all parameters
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
        
        # Return average momentum
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
            # Toggle pause state
            self.paused = not self.paused
            status = "paused" if self.paused else "resumed"
            self.train_logger.log_event(f"Training {status} at step {self.global_step}")
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
    
    def _run_validation(self):
        """Run validation immediately"""
        self.train_logger.log_event(f"Manual validation triggered at step {self.global_step}")
        
        metrics = self._run_multi_size_validation()
        
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
            
            # CRITICAL: Remove labeled_images to prevent memory leak
            del labeled_images
            metrics.pop('labeled_images', None)
            
            # Force cleanup
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Store metrics WITHOUT labeled_images
        self.last_metrics = metrics
        
        # Update web_monitor with validation metrics before scheduling JSON save
        # This ensures the saved JSON includes the fresh validation data
        if self.last_metrics:
            self.web_monitor.update(
                quality_lr_value=self.last_metrics.get('lr_quality', 0.0),
                quality_ki_value=self.last_metrics.get('ki_quality', 0.0),
                quality_improvement_value=self.last_metrics.get('improvement', 0.0),
                quality_ki_to_gt_value=self.last_metrics.get('ki_to_gt', 0.0),
                quality_lr_to_gt_value=self.last_metrics.get('lr_to_gt', 0.0),
                validation_loss_value=self.last_metrics.get('val_loss', 0.0),
            )
        
        # Schedule JSON save for 2 steps later (so web_monitor gets updated with fresh loss data)
        self.pending_json_save_step = self.global_step + 2
        
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
                                    print(f"\n📂 New validation files detected for {size_key}: +{val_changes['new_files']} files")
                                    print(f"   🔄 Reloading {size_key} validation dataset...")
                                    
                                    if hasattr(val_ds, 'reload_files'):
                                        reload_result = val_ds.reload_files()
                                        if reload_result['success']:
                                            print(f"   ✅ Reload successful: {reload_result['files_before']} → {reload_result['files_after']} files")
                                            dataset_info['val'][size_key]['count'] = reload_result['files_after']
                                            if hasattr(self, 'train_logger') and self.train_logger:
                                                self.train_logger.log_event(
                                                    f"Reloaded {size_key} validation: +{reload_result['new_files_loaded']} files"
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
                            print(f"\n📂 New validation files detected for {size_key}: +{val_changes['new_files']} files")
                            print(f"   🔄 Reloading validation dataset...")
                            
                            if hasattr(val_ds, 'reload_files'):
                                reload_result = val_ds.reload_files()
                                if reload_result['success']:
                                    print(f"   ✅ Reload successful: {reload_result['files_before']} → {reload_result['files_after']} files")
                                    dataset_info['val'][size_key]['count'] = reload_result['files_after']
                                    if hasattr(self, 'train_logger') and self.train_logger:
                                        self.train_logger.log_event(
                                            f"Reloaded validation: +{reload_result['new_files_loaded']} files"
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
        Save complete training statistics as JSON file
        
        Saves to DATA_ROOT/Statistik_STEP.json with all data from web monitor
        
        Args:
            step: Current training step
        """
        try:
            # Get complete data snapshot from web monitor (same as web UI download)
            data_snapshot = self.web_monitor.data_store.get_complete_snapshot()
            
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

