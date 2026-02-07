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
from ..utils.ui_display import draw_ui, get_activity_data
from ..utils.keyboard_handler import KeyboardHandler
from ..utils.ui_terminal import C_GREEN, C_CYAN, C_YELLOW, C_RESET, show_cursor


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
                 config, device='cuda', runtime_config=None):
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
        self.runtime_config = runtime_config
        
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
        
        # Keyboard handler
        self.keyboard = KeyboardHandler()
        
        # Web interface for remote monitoring - COMPLETE data
        from ..systems.web_ui import WebMonitoringInterface
        self.web_monitor = WebMonitoringInterface(port_num=5050, refresh_seconds=5)
    
    def set_start_step(self, step):
        """Set starting step (for resume)"""
        self.start_step = step
        self.global_step = step
    
    def train_epoch(self, epoch):
        """
        Train one epoch
        
        Args:
            epoch: Current epoch number
        """
        self.model.train()
        
        accumulation_steps = self.config.get('ACCUMULATION_STEPS', 1)
        steps_per_epoch = len(self.train_loader) // accumulation_steps
        current_epoch_step = 0
        
        # Initialize loop timing
        loop_start_time = time.time()
        
        for batch_idx, (lr_stack, gt) in enumerate(self.train_loader):
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
            
            # Move to device
            lr_stack = lr_stack.to(self.device)
            gt = gt.to(self.device)
            
            # Forward pass
            output = self.model(lr_stack)
            
            # Compute L1 loss for adaptive system
            with torch.no_grad():
                current_l1 = torch.abs(output - gt).mean().item()
            
            # Get adaptive weights (now returns 5 values including status)
            l1_w, ms_w, grad_w, perceptual_w, adaptive_status = self.adaptive_system.update_loss_weights(
                output, gt, self.global_step, 
                current_l1_loss=current_l1
            )
            
            # Compute loss
            loss_dict = self.loss_fn(output, gt, l1_w, ms_w, grad_w, perceptual_w)
            loss = loss_dict['total']
            
            # Backward pass (with accumulation)
            loss = loss / accumulation_steps
            loss.backward()
            
            # Update optimizer (every accumulation_steps)
            if (batch_idx + 1) % accumulation_steps == 0:
                # Clip gradients
                grad_norm, clip_val = self.adaptive_system.clip_gradients(self.model)
                
                # Update optimizer (every accumulation_steps)
                self.optimizer.step()
                
                # Extract AdamW momentum (exp_avg) for GUI
                adam_momentum = self._get_adam_momentum()
                
                self.optimizer.zero_grad()
                
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
                
                # Update plateau tracker
                self.adaptive_system.update_plateau_tracker(loss_dict['total'].item() if torch.is_tensor(loss_dict['total']) else loss_dict['total'])
                
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
                vram = torch.cuda.max_memory_allocated() / (1024**3)
                
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
                
                # Check for runtime config updates every 10 steps
                if self.runtime_config is not None and self.global_step % 10 == 0:
                    if self.runtime_config.check_for_updates():
                        self._apply_config_changes()
                
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
                    
                    metrics = self.validator.validate(self.global_step)
                    self.last_metrics = metrics
                    
                    # Pass improvement to adaptive system for logging
                    adaptive_status = self.adaptive_system.get_status()
                    adaptive_status['ki_improvement'] = metrics.get('improvement', 0)
                    
                    # Log to TensorBoard with dashboards
                    self.tb_logger.log_quality(self.global_step, metrics)
                    self.tb_logger.log_metrics(self.global_step, metrics)
                    self.tb_logger.log_validation_loss(self.global_step, metrics.get('val_loss', 0.0))
                    self.tb_logger.log_adaptive(self.global_step, adaptive_status)
                    
                    # Log ALL images (like in original)
                    labeled_images = metrics.get('labeled_images')
                    if labeled_images is not None and len(labeled_images) > 0:
                        print(f"📊 Logging {len(labeled_images)} validation images to TensorBoard...")
                        logged_count = 0
                        failed_count = 0
                        
                        for idx, img_tensor in enumerate(labeled_images):
                            try:
                                # Ensure tensor is in correct format for TensorBoard
                                if img_tensor.device.type != 'cpu':
                                    img_tensor = img_tensor.cpu()
                                if not img_tensor.is_contiguous():
                                    img_tensor = img_tensor.contiguous()
                                
                                self.tb_logger.writer.add_image(
                                    f"Val/sample_{idx:04d}", 
                                    img_tensor, 
                                    self.global_step
                                )
                                logged_count += 1
                            except Exception as e:
                                failed_count += 1
                                print(f"⚠️  Failed to log validation image {idx}: {e}")
                                self.train_logger.log_event(
                                    f"Warning: Failed to log validation image {idx}: {e}"
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
                    
                    # Auto-continue timer for manual validation
                    if self.do_manual_val:
                        import select
                        from vsr_plus_plus.utils.ui_terminal import C_CYAN, C_BOLD, C_GREEN, C_RESET, C_YELLOW
                        
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
        num_images = len(self.train_loader.dataset)
        
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
        gpu_mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        
        # Konvertiere Layer-Aktivitäten in Dict-Format
        layer_act_dict = {}
        peak_activity_value = 0.0
        if activities:
            for name, activity_percent, trend, raw_value in activities:
                layer_act_dict[name] = activity_percent
                # Track maximum raw value across all layers
                peak_activity_value = max(peak_activity_value, raw_value)
        
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
            adaptive_lr_boost_available=adaptive_status.get('lr_boost_available', False),
            adaptive_perceptual_trend=0,  # TODO: calculate trend
            
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
        
        # Check for web UI commands (new method name)
        web_cmd = self.web_monitor.poll_commands()
        if web_cmd == 'validate':
            self.do_manual_val = True
    
    def _run_validation(self):
        """Run validation immediately"""
        self.train_logger.log_event(f"Manual validation triggered at step {self.global_step}")
        
        metrics = self.validator.validate(self.global_step)
        
        # Log to TensorBoard
        self.tb_logger.log_quality(self.global_step, metrics)
        self.tb_logger.log_metrics(self.global_step, metrics)
        self.tb_logger.log_validation_loss(self.global_step, metrics.get('val_loss', 0.0))
        
        # Log ALL images (like in original)
        labeled_images = metrics.get('labeled_images')
        if labeled_images is not None:
            for idx, img_tensor in enumerate(labeled_images):
                self.tb_logger.writer.add_image(
                    f"Val/sample_{idx:04d}", 
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
        # Run validation
        val_results = self.validator.validate(self.model, self.val_loader)
        
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
    
    def _apply_config_changes(self):
        """Apply runtime config changes to live systems"""
        if self.runtime_config is None:
            return
        
        # Update Adaptive System
        new_threshold = self.runtime_config.get('plateau_safety_threshold')
        if new_threshold is not None and new_threshold != self.adaptive_system.plateau_safety_threshold:
            old = self.adaptive_system.plateau_safety_threshold
            self.adaptive_system.plateau_safety_threshold = new_threshold
            print(f"⚙️  Config Update: plateau_safety_threshold {old} → {new_threshold}")
        
        new_patience = self.runtime_config.get('plateau_patience')
        if new_patience is not None and new_patience != self.adaptive_system.plateau_patience:
            old = self.adaptive_system.plateau_patience
            self.adaptive_system.plateau_patience = new_patience
            print(f"⚙️  Config Update: plateau_patience {old} → {new_patience}")
        
        new_cooldown = self.runtime_config.get('cooldown_duration')
        if new_cooldown is not None and new_cooldown != self.adaptive_system.cooldown_duration:
            old = self.adaptive_system.cooldown_duration
            self.adaptive_system.cooldown_duration = new_cooldown
            print(f"⚙️  Config Update: cooldown_duration {old} → {new_cooldown}")
        
        # Update LR Scheduler
        new_max_lr = self.runtime_config.get('max_lr')
        if new_max_lr is not None and hasattr(self.lr_scheduler, 'max_lr'):
            if new_max_lr != self.lr_scheduler.max_lr:
                old = self.lr_scheduler.max_lr
                self.lr_scheduler.max_lr = new_max_lr
                print(f"⚙️  Config Update: max_lr {old:.2e} → {new_max_lr:.2e}")
        
        new_min_lr = self.runtime_config.get('min_lr')
        if new_min_lr is not None and hasattr(self.lr_scheduler, 'min_lr'):
            if new_min_lr != self.lr_scheduler.min_lr:
                old = self.lr_scheduler.min_lr
                self.lr_scheduler.min_lr = new_min_lr
                print(f"⚙️  Config Update: min_lr {old:.2e} → {new_min_lr:.2e}")
        
        # Update gradient clipping
        new_grad_clip = self.runtime_config.get('initial_grad_clip')
        if new_grad_clip is not None and new_grad_clip != self.adaptive_system.clip_value:
            old = self.adaptive_system.clip_value
            self.adaptive_system.clip_value = new_grad_clip
            print(f"⚙️  Config Update: initial_grad_clip {old:.2f} → {new_grad_clip:.2f}")
    
    def run(self):
        """
        Main training loop
        """
        self.train_logger.log_event("🚀 TRAINING STARTED")
        
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

