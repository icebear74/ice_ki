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
        
        # UI state
        self.paused = False
        self.do_manual_val = False
        
        # Keyboard handler
        self.keyboard = KeyboardHandler()
    
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
            
            step_start_time = time.time()
            
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
            
            # Compute loss
            loss_dict = self.loss_fn(output, gt, l1_w, ms_w, grad_w)
            loss = loss_dict['total']
            
            # Backward pass (with accumulation)
            loss = loss / accumulation_steps
            loss.backward()
            
            # Update optimizer (every accumulation_steps)
            if (batch_idx + 1) % accumulation_steps == 0:
                # Clip gradients
                grad_norm, clip_val = self.adaptive_system.clip_gradients(self.model)
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update LR scheduler
                plateau_detected = self.adaptive_system.is_plateau()
                current_lr, lr_phase = self.lr_scheduler.step(self.global_step, plateau_detected)
                
                # Update plateau tracker
                self.adaptive_system.update_plateau_tracker(loss_dict['total'].item() if torch.is_tensor(loss_dict['total']) else loss_dict['total'])
                
                # Get activity
                self.last_activities = self.model.get_layer_activity()
                
                # Measure performance
                step_time = time.time() - step_start_time
                self.step_times.append(step_time)
                if len(self.step_times) > 100:
                    self.step_times.pop(0)
                
                avg_time = sum(self.step_times) / len(self.step_times)
                vram = torch.cuda.max_memory_allocated() / (1024**3)
                
                # Track loss history
                self.loss_history.append(loss_dict['total'].item() if torch.is_tensor(loss_dict['total']) else loss_dict['total'])
                if len(self.loss_history) > 1000:
                    self.loss_history.pop(0)
                
                # Increment step
                self.global_step += 1
                current_epoch_step += 1
                
                # Update GUI
                self._update_gui(epoch, loss_dict, avg_time, steps_per_epoch, current_epoch_step)
                
                # TensorBoard logging
                if self.global_step % self.config.get('LOG_TBOARD_EVERY', 100) == 0:
                    self.tb_logger.log_losses(self.global_step, loss_dict)
                    self.tb_logger.log_lr(self.global_step, current_lr)
                    self.tb_logger.log_adaptive(self.global_step, self.adaptive_system.get_status())
                    self.tb_logger.log_system(self.global_step, avg_time, vram)
                    self.tb_logger.log_gradients(self.global_step, grad_norm, self.last_activities)
                    self.tb_logger.log_lr_phase(self.global_step, lr_phase)
                
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
                    
                    # Log images
                    if metrics.get('sample_lr') is not None:
                        self.tb_logger.log_images(
                            self.global_step,
                            metrics['sample_lr'],
                            metrics['sample_ki'],
                            metrics['sample_gt']
                        )
                    
                    self.train_logger.log_event(
                        f"Step {self.global_step} | Validation | "
                        f"KI Quality: {metrics['ki_quality']*100:.1f}%"
                    )
                    
                    # Check for best checkpoint
                    if self.checkpoint_mgr.should_check_best(self.global_step):
                        is_new_best = self.checkpoint_mgr.update_best_checkpoint(
                            self.model, self.optimizer, self.lr_scheduler, 
                            self.global_step, metrics['ki_quality'], metrics,
                            self.train_logger.log_file
                        )
                        
                        if is_new_best:
                            self.tb_logger.log_checkpoint(self.global_step, 'best')
                    
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
                        print(f"  Improvement:    {C_BOLD}{C_GREEN}+{metrics['improvement']*100:.1f}%{C_RESET}")
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
                    self.checkpoint_mgr.save_checkpoint(
                        self.model, self.optimizer, self.lr_scheduler,
                        self.global_step, self.last_metrics or {},
                        self.train_logger.log_file
                    )
                    self.tb_logger.log_checkpoint(self.global_step, 'regular')
                    self.train_logger.log_event(f"Regular checkpoint saved at step {self.global_step}")
                
                # Check if training complete
                if self.global_step >= self.config.get('MAX_STEPS', 100000):
                    return
    
    def _update_gui(self, epoch, loss_dict, avg_time, steps_per_epoch, current_epoch_step, paused=False):
        """Update the GUI display"""
        # Get activities
        activities = get_activity_data(self.model)
        
        # Prepare loss dict
        losses = {
            'l1': loss_dict.get('l1', 0.0) if loss_dict else 0.0,
            'ms': loss_dict.get('ms', 0.0) if loss_dict else 0.0,
            'grad': loss_dict.get('grad', 0.0) if loss_dict else 0.0,
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
        
        # Adaptive status
        adaptive_status = self.adaptive_system.get_status()
        
        # Number of training images
        num_images = len(self.train_loader.dataset)
        
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
            lr_info=lr_info
        )
    
    def _check_keyboard_input(self, epoch, steps_per_epoch, current_epoch_step):
        """Check for keyboard input and handle commands"""
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
                if not self.paused:
                    # Reset step timer when resuming
                    self.step_times = []
            
            elif key_lower == 'v':  # Manual validation
                self.do_manual_val = True
    
    def _run_validation(self):
        """Run validation immediately"""
        self.train_logger.log_event(f"Manual validation triggered at step {self.global_step}")
        
        metrics = self.validator.validate(self.global_step)
        self.last_metrics = metrics
        
        # Log to TensorBoard
        self.tb_logger.log_quality(self.global_step, metrics)
        self.tb_logger.log_metrics(self.global_step, metrics)
        
        # Log images
        if metrics.get('sample_lr') is not None:
            self.tb_logger.log_images(
                self.global_step,
                metrics['sample_lr'],
                metrics['sample_ki'],
                metrics['sample_gt']
            )
        
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

