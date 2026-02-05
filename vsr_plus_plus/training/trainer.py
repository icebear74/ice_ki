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
                 config, device='cuda'):
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
        
        self.global_step = 0
        self.start_step = 0
        
        # Metrics tracking
        self.last_metrics = None
        self.last_activities = None
        self.loss_history = []
        
        # Performance tracking
        self.step_times = []
        
        # EMA for GUI smoothing (factor 0.95)
        self.ema_loss = None
        self.ema_factor = 0.95
        
        # UI state
        self.paused = False
        self.do_manual_val = False
        
        # Keyboard handler
        self.keyboard = KeyboardHandler()
        
        # Web interface for remote monitoring
        from ..systems.web_ui import WebInterface
        self.web_monitor = WebInterface(port_number=5050)
    
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
            
            # Get adaptive weights
            l1_w, ms_w, grad_w = self.adaptive_system.update_loss_weights(
                output, gt, self.global_step, 
                current_grad_loss=None  # Could compute this if needed
            )
            
            # Get perceptual weight (fixed, not adaptive)
            perceptual_w = self.adaptive_system.perceptual_weight
            
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
                
                # Update GUI with smoothed values
                self._update_gui(epoch, smoothed_loss_dict, avg_time, steps_per_epoch, current_epoch_step, adam_momentum=adam_momentum)
                
                # TensorBoard logging (use RAW values, not smoothed)
                if self.global_step % self.config.get('LOG_TBOARD_EVERY', 100) == 0:
                    self.tb_logger.log_losses(self.global_step, loss_dict)
                    self.tb_logger.log_lr(self.global_step, current_lr)
                    self.tb_logger.log_adaptive(self.global_step, self.adaptive_system.get_status())
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
                    
                    # Log to TensorBoard
                    self.tb_logger.log_quality(self.global_step, metrics)
                    self.tb_logger.log_metrics(self.global_step, metrics)
                    self.tb_logger.log_validation_loss(self.global_step, metrics.get('val_loss', 0.0))
                    
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
        
        # Update web monitor with current training state
        best_quality = self.checkpoint_mgr.best_quality if self.checkpoint_mgr.best_quality > 0 else 0.0
        gpu_mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        
        self.web_monitor.update(
            iteration=self.global_step,
            total_loss=losses['total'],
            learn_rate=current_lr,
            time_remaining=total_eta,
            iter_speed=avg_time,
            gpu_memory=gpu_mem,
            best_score=best_quality,
            is_validating=False
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
        
        # Check for web UI commands
        web_cmd = self.web_monitor.check_commands()
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
        
        self.train_logger.log_event(
            f"Manual Validation | KI Quality: {metrics['ki_quality']*100:.1f}%"
        )
        
        self.model.train()  # Back to training mode
    
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

