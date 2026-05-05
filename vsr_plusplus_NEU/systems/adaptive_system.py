"""
Adaptive Training System - "Smooth Operator"

Enhanced adaptive system with:
- EMA smoothing for stable decisions
- Momentum-based weight updates (max 1% change per step)
- Intelligent perceptual loss coupling to L1 loss
- Cooldown mechanism to prevent oscillations
- Adaptive loss weights (L1, MS, Grad, Perceptual)
- Adaptive gradient clipping
- Aggressive mode detection
- Plateau detection
"""

import math
import torch
import numpy as np


class AdaptiveSystem:
    """
    Complete adaptive training system with smooth, stable control
    
    Features:
    - EMA smoothing over loss values
    - Momentum-limited weight adjustments
    - Intelligent perceptual weight control
    - Cooldown periods after adjustments
    """
    
    def __init__(self, initial_l1=0.6, initial_ms=0.2, initial_grad=0.2, initial_perceptual=0.0):
        # Store initial weights to always respect config values
        self.initial_l1 = initial_l1
        self.initial_ms = initial_ms
        self.initial_grad = initial_grad
        self.initial_perceptual = initial_perceptual
        
        # Loss weights - start with config values
        self.l1_weight = initial_l1
        self.ms_weight = initial_ms
        self.grad_weight = initial_grad
        self.perceptual_weight = initial_perceptual
        
        # EMA for smooth loss tracking (50-step window, alpha = 2/(N+1))
        self.ema_l1_loss = None
        self.ema_window = 50
        self.ema_alpha = 2.0 / (self.ema_window + 1)
        
        # Momentum: maximum change per step (0.3% = 0.003, was 1% = 0.01)
        # Reduced to prevent wild oscillation (max ~15% drift over 500 steps).
        self.max_weight_change = 0.003
        
        # Cooldown mechanism
        self.cooldown_steps = 0
        self.cooldown_duration = 30  # Reduced from 80 to 30 for faster recovery
        self.is_in_cooldown = False
        
        # Gradient clipping
        # Start closer to the config INITIAL_GRAD_CLIP value instead of 3.0
        self.clip_value = 1.5
        self.grad_norms = []
        
        # Sharpness tracking
        self.sharpness_history = []
        self.adjustment_step = 0
        
        # Aggressive mode
        self.aggressive_mode = False
        self.aggressive_counter = 0
        self.aggressive_max_steps = 5000  # Keep aggressive phase short; a small model collapses under prolonged pressure
        
        # Thresholds
        self.extreme_grad_threshold = 0.025
        self.extreme_sharpness_threshold = 0.70
        self.aggressive_blur_threshold = 0.72
        self.aggressive_stabilization_threshold = 0.75
        self.normal_blur_threshold = 0.75
        
        # L1 loss thresholds for perceptual weight control
        self.l1_stable_threshold = 0.025  # L1 below this is "stable and good" (realistischer für Phase 3)
        self.l1_unstable_threshold = 0.045  # L1 above this is "unstable"
        
        # Update frequencies — increased to reduce oscillation frequency
        self.aggressive_update_frequency = 50   # was 25; slower weight changes prevent gradient explosion on small models
        self.normal_update_frequency = 150      # was 100; gentler normal-mode updates
        
        # Plateau detection
        self.best_loss = float('inf')
        self.plateau_counter = 0
        self.plateau_patience = 1000  # Raised from 500: a small model converges slowly; triggering Aggressive Mode at 500 steps is premature and leads to gradient explosion
        self.plateau_safety_threshold = 1500  # Lowered from 2000: trigger safety reset sooner when genuinely stuck
        
        # Enhanced plateau detection
        self.best_quality = 0.0
        self.ema_loss = None
        self.ema_quality = None
        self.ema_alpha = 0.1
        
        # Adaptive thresholds based on loss level
        self.plateau_threshold_map = {
            0.015: 0.999,  # 0.1% when loss very good
            0.020: 0.998,  # 0.2% when loss good
            0.030: 0.997,  # 0.3% when loss ok
            0.050: 0.995,  # 0.5% when loss early
        }
        
        # Grace period
        self.grace_enabled = True
        
        # History settling period: when resuming training at step >= 1000,
        # wait to collect history before making changes
        self.history_settling_steps = 100
        self.history_steps_collected = 0
        self.history_settling_complete = False

        # Validation-based plateau tracking (Bug 5 fix)
        # Tracks validation loss trend to detect overfitting independent of train loss.
        self.best_val_loss = float('inf')
        self.ema_val_loss = None
        self.val_no_improve_count = 0
        self.val_plateau_patience = 5  # After 5 consecutive validations without improvement

        # Cached mode — updated by update_loss_weights so get_status() always
        # returns the true current mode (Warmup / Settling / Stable / Aggressive)
        self._cached_mode = 'Warmup'

        # Perceptual floor: minimum value _update_perceptual_weight() will enforce.
        # Set to 0.0 during DataStrategy Phase 1/2 so the scheduler's suppression
        # is respected; set back to 0.05 in Phase 3 (AdaptiveSystem controls weight).
        self._perceptual_floor = 0.05
    
    def _update_ema_loss(self, l1_loss_value):
        """Update EMA of L1 loss for smooth tracking"""
        # NaN/Inf guard: a bad loss value must never contaminate the EMA.
        # Once NaN enters (NaN * alpha + finite * (1-alpha) = NaN), it can
        # never recover.  Simply skip the update and keep the previous value.
        if not math.isfinite(l1_loss_value):
            return
        if self.ema_l1_loss is None:
            self.ema_l1_loss = l1_loss_value
        else:
            self.ema_l1_loss = self.ema_alpha * l1_loss_value + (1 - self.ema_alpha) * self.ema_l1_loss

    def set_perceptual_floor(self, value):
        """Set the minimum perceptual weight enforced by _update_perceptual_weight().

        Call with 0.0 during DataStrategy Phase 1/2 so the scheduler's
        suppression is fully respected.  Call with 0.05 (the default) when
        entering Phase 3 to restore autonomous AdaptiveSystem control.

        Args:
            value: float – new floor value (typically 0.0 or 0.05).
        """
        self._perceptual_floor = float(value)
    
    def _apply_momentum(self, current_value, target_value):
        """Apply momentum constraint: limit change to max_weight_change"""
        delta = target_value - current_value
        # Clip delta to maximum allowed change
        delta = np.clip(delta, -self.max_weight_change, self.max_weight_change)
        return current_value + delta
    
    def _update_perceptual_weight(self):
        """
        Update perceptual weight based on L1 stability
        
        Perceptual runs INDEPENDENTLY of cooldown (separate, slow process)
        """
        if self.ema_l1_loss is None:
            return
        
        # FIX 3: REMOVED cooldown check - Perceptual runs independently
        # Perceptual is a slow, independent process that should NOT be blocked
        
        # Dynamic maximum based on L1 quality (universal for all image types)
        if self.ema_l1_loss < 0.008:
            max_perc = 0.20  # L1 very stable -> allow details
        elif self.ema_l1_loss < 0.012:
            max_perc = 0.15  # L1 stable -> moderate details
        else:
            max_perc = 0.10  # L1 unstable -> structure first
        
        min_perc = self._perceptual_floor  # Respects DataStrategy phase suppression
        
        target_perc = self.perceptual_weight
        
        if self.ema_l1_loss < 0.010:  # L1 stable -> increase perceptual
            target_perc = min(max_perc, self.perceptual_weight + 0.0015)  # 0.15% per update
        elif self.ema_l1_loss > 0.018:  # L1 unstable -> decrease perceptual
            target_perc = max(min_perc, self.perceptual_weight - 0.002)   # 0.2% per update
        
        # Apply momentum (smooth change)
        self.perceptual_weight = self._apply_momentum(self.perceptual_weight, target_perc)
        
        # Hard limits (safety net)
        self.perceptual_weight = max(min_perc, min(0.25, self.perceptual_weight))
    
    def detect_extreme_conditions(self, pred, target, current_l1_loss=None):
        """Check if immediate intervention needed"""
        with torch.no_grad():
            # Compute sharpness
            pred_grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
            target_grad_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
            pred_grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
            target_grad_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
            
            pred_sharpness = (pred_grad_x.mean() + pred_grad_y.mean()) / 2
            target_sharpness = (target_grad_x.mean() + target_grad_y.mean()) / 2
            
            if target_sharpness > 0:
                sharpness_ratio = (pred_sharpness / target_sharpness).item()
            else:
                sharpness_ratio = 1.0
        
        # NaN/Inf sharpness_ratio (from NaN model output) must not trigger
        # aggressive mode or update the L1 EMA.
        if not math.isfinite(sharpness_ratio):
            return 1.0  # treat as neutral (no intervention)

        # Update EMA with L1 loss if provided
        if current_l1_loss is not None:
            self._update_ema_loss(current_l1_loss)
        
        # Check extreme conditions
        extreme = False
        
        # Don't trigger aggressive mode during warmup (first 100 steps)
        # This prevents premature weight changes when model is still initializing
        if not hasattr(self, '_warmup_complete'):
            self._warmup_steps = 0
            self._warmup_complete = False
        
        if not self._warmup_complete:
            self._warmup_steps += 1
            # Allow aggressive mode evaluation after 2000 internal steps.
            # This is decoupled from the DataStrategy Phase-1 duration so the
            # system can react to L1-plateau stagnation on full-frame data
            # well before crops are introduced.
            if self._warmup_steps >= 2000:
                self._warmup_complete = True
            # During warmup, don't trigger aggressive mode
            return sharpness_ratio
        
        # SAFETY: Don't trigger aggressive mode if we don't have enough history yet
        # This prevents extreme weight changes when resuming from checkpoint
        if not self.history_settling_complete:
            return sharpness_ratio
        
        # FIX 1: Only trigger aggressive mode if BOTH conditions are true:
        # 1. Sharpness is poor (< 0.70)
        # 2. Training is stuck (plateau > plateau_patience)
        if sharpness_ratio < self.extreme_sharpness_threshold and self.plateau_counter > self.plateau_patience:
            extreme = True
        
        if extreme and not self.aggressive_mode:
            self.aggressive_mode = True
            self.aggressive_counter = 0
            # Start cooldown
            self.is_in_cooldown = True
            self.cooldown_steps = self.cooldown_duration
            # Boost gradient and perceptual to break L1-dominance, but keep the
            # push moderate so a small/reduced model doesn't suffer gradient explosion.
            # grad=0.30 + ms=0.15 + perc=0.08 = 0.53 → l1=0.47 (still dominant).
            # If these targets are ever changed, ensure their sum stays < 0.90
            # so that l1_weight remains positive after the floor clamp below.
            target_grad = 0.30
            target_ms = 0.15
            target_perc = 0.08
            assert target_grad + target_ms + target_perc < 1.0, (
                f"Aggressive-mode loss targets sum to "
                f"{target_grad + target_ms + target_perc:.2f} >= 1.0; "
                "l1_weight would be zero or negative."
            )
            target_l1 = max(0.1, 1.0 - target_grad - target_ms - target_perc)
            self.grad_weight = self._apply_momentum(self.grad_weight, target_grad)
            self.l1_weight = self._apply_momentum(self.l1_weight, target_l1)
            self.ms_weight = self._apply_momentum(self.ms_weight, target_ms)
            self.perceptual_weight = self._apply_momentum(self.perceptual_weight, target_perc)

            # Apply minimum guards even in aggressive mode
            self.ms_weight = max(0.05, self.ms_weight)
            self.grad_weight = max(0.05, self.grad_weight)
            self.perceptual_weight = max(0.05, self.perceptual_weight)
            self.l1_weight = min(0.9, 1.0 - self.ms_weight - self.grad_weight - self.perceptual_weight)
        
        return sharpness_ratio
    
    def update_loss_weights(self, pred, target, step, current_l1_loss=None):
        """
        Update loss weights based on image quality with smooth control
        
        Args:
            pred: Predicted image
            target: Target image
            step: Current training step
            current_l1_loss: Current L1 loss value for EMA tracking
            
        Returns:
            Tuple of (l1_weight, ms_weight, grad_weight, perceptual_weight, status_dict)
            status_dict contains: {
                'is_cooldown': bool,
                'cooldown_remaining': int,
                'mode': str ('Warmup', 'Settling', 'Aggressive' or 'Stable')
            }
        """
        # Track last step to prevent double decrement when called multiple times per step
        if not hasattr(self, '_last_step'):
            self._last_step = -1
        
        # PHASE 1: Early warmup (step < 1000) - return initial weights unchanged
        # This gives the model time to stabilize before any adaptive changes
        if step < 1000:
            # Sync internal weights with initial values for consistent GUI display
            self.l1_weight = self.initial_l1
            self.ms_weight = self.initial_ms
            self.grad_weight = self.initial_grad
            self.perceptual_weight = self.initial_perceptual
            
            self._cached_mode = 'Warmup'
            status = {
                'is_cooldown': False,
                'cooldown_remaining': 0,
                'mode': 'Warmup'
            }
            return self.initial_l1, self.initial_ms, self.initial_grad, self.initial_perceptual, status
        
        # PHASE 2: History settling period (step >= 1000, but no history yet)
        # When resuming from checkpoint or after warmup, collect history before adapting
        if not self.history_settling_complete:
            # Only increment once per step, not per batch
            if step != self._last_step:
                self.history_steps_collected += 1
                if self.history_steps_collected >= self.history_settling_steps:
                    self.history_settling_complete = True
            
            # Sync internal weights with initial values for consistent GUI display
            self.l1_weight = self.initial_l1
            self.ms_weight = self.initial_ms
            self.grad_weight = self.initial_grad
            self.perceptual_weight = self.initial_perceptual
            
            self._cached_mode = 'Settling'
            # Return initial weights during settling
            status = {
                'is_cooldown': False,
                'cooldown_remaining': 0,
                'mode': 'Settling',
                'settling_progress': f"{self.history_steps_collected}/{self.history_settling_steps}"
            }
            return self.initial_l1, self.initial_ms, self.initial_grad, self.initial_perceptual, status
        
        # SAFETY VALVE: Force reset if plateau counter exceeds threshold
        if self.plateau_counter > self.plateau_safety_threshold:
            print(f"[AdaptiveSystem] SAFETY RESET: plateau_counter={self.plateau_counter} exceeded {self.plateau_safety_threshold} steps")
            print(f"[AdaptiveSystem] Resetting to Stable mode with initial weights")
            # Reset to stable mode
            self.aggressive_mode = False
            self.plateau_counter = 0
            # Reset weights to initial values
            self.l1_weight = self.initial_l1
            self.ms_weight = self.initial_ms
            self.grad_weight = self.initial_grad
            self.perceptual_weight = self.initial_perceptual
            # Activate cooldown
            self.is_in_cooldown = True
            self.cooldown_steps = self.cooldown_duration
        
        # Update cooldown counter ONLY ONCE PER STEP (not per batch)
        if self.is_in_cooldown and step != self._last_step:
            self.cooldown_steps -= 1
            if self.cooldown_steps <= 0:
                self.is_in_cooldown = False
        
        # Remember this step to avoid double decrement
        self._last_step = step
        
        # Detect extreme conditions (also updates EMA)
        sharpness_ratio = self.detect_extreme_conditions(pred, target, current_l1_loss)
        
        # Update perceptual weight based on L1 stability
        self._update_perceptual_weight()
        
        # Update frequency based on mode
        if self.aggressive_mode:
            update_freq = self.aggressive_update_frequency
            min_measurements = 2
            adjustment_factor = 1.08  # was 1.15; gentler per-step push prevents rapid weight escalation on small models
            blur_threshold = self.aggressive_blur_threshold
            
            self.aggressive_counter += 1
            
            # Deactivate after max steps or if stabilized
            if self.aggressive_counter >= self.aggressive_max_steps:
                self.aggressive_mode = False
            elif sharpness_ratio > self.aggressive_stabilization_threshold and len(self.sharpness_history) > 50:
                avg_recent = np.mean(self.sharpness_history[-20:])
                if avg_recent > self.aggressive_stabilization_threshold:
                    self.aggressive_mode = False
        else:
            update_freq = self.normal_update_frequency
            min_measurements = 10
            adjustment_factor = 1.05
            blur_threshold = self.normal_blur_threshold
        
        # Check if we should update (skip if in cooldown or not update time)
        if self.is_in_cooldown or step % update_freq != 0:
            self._cached_mode = 'Aggressive' if self.aggressive_mode else 'Stable'
            status = {
                'is_cooldown': self.is_in_cooldown,
                'cooldown_remaining': self.cooldown_steps if self.is_in_cooldown else 0,
                'mode': self._cached_mode
            }
            return self.l1_weight, self.ms_weight, self.grad_weight, self.perceptual_weight, status
        
        # Add to history
        self.sharpness_history.append(sharpness_ratio)
        if len(self.sharpness_history) > 200:
            self.sharpness_history.pop(0)
        
        # Warmup
        if len(self.sharpness_history) < min_measurements:
            self._cached_mode = 'Aggressive' if self.aggressive_mode else 'Stable'
            status = {
                'is_cooldown': False,
                'cooldown_remaining': 0,
                'mode': self._cached_mode
            }
            return self.l1_weight, self.ms_weight, self.grad_weight, self.perceptual_weight, status
        
        # Compute average using EMA concept
        window = min(20, len(self.sharpness_history))
        avg_sharpness = np.mean(self.sharpness_history[-window:])
        
        # Calculate target weights
        target_grad = self.grad_weight
        target_ms = self.ms_weight
        target_l1 = self.l1_weight
        
        # Adjust weights based on sharpness
        if avg_sharpness < blur_threshold:
            # Image is blurry, boost gradient.
            # Cap at 0.35 in aggressive mode (was 0.5) to avoid overwhelming a small model.
            max_grad = 0.35 if self.aggressive_mode else 0.50
            target_grad = min(max_grad, self.grad_weight * adjustment_factor)
            target_ms = min(0.2, self.ms_weight)
            target_l1 = max(0.3, 1.0 - target_grad - target_ms)
            # FIX 2: Start cooldown ONLY if not already in cooldown
            if not self.is_in_cooldown:
                self.is_in_cooldown = True
                self.cooldown_steps = self.cooldown_duration
        elif avg_sharpness > 0.92:
            # Image is sharp enough, reduce gradient
            target_grad = max(0.15, self.grad_weight * 0.95)
            target_ms = min(0.2, self.ms_weight)
            target_l1 = 1.0 - target_grad - target_ms
            # FIX 2: Start cooldown ONLY if not already in cooldown
            if not self.is_in_cooldown:
                self.is_in_cooldown = True
                self.cooldown_steps = self.cooldown_duration
        
        # Apply momentum (smooth transitions)
        self.grad_weight = self._apply_momentum(self.grad_weight, target_grad)
        self.ms_weight = self._apply_momentum(self.ms_weight, target_ms)
        self.l1_weight = self._apply_momentum(self.l1_weight, target_l1)
        
        # HARD SAFETY GUARDS: Prevent MS or Grad from dropping too low
        # This ensures the model always considers structure (MS) and sharpness (Grad)
        self.ms_weight = max(0.05, self.ms_weight)  # Never below 5%
        self.grad_weight = max(0.05, self.grad_weight)  # Never below 5%
        
        # Recalculate L1 as residual to maintain sum = 1.0
        self.l1_weight = 1.0 - self.ms_weight - self.grad_weight
        
        # Ensure L1 doesn't exceed maximum (cap at 0.9)
        if self.l1_weight > 0.9:
            self.l1_weight = 0.9
            # Rebalance MS and Grad proportionally to use remaining budget
            remaining_budget = 1.0 - self.l1_weight  # 0.1 when L1 is at max
            total_other = self.ms_weight + self.grad_weight
            if total_other > 0:
                scale = remaining_budget / total_other
                self.ms_weight *= scale
                self.grad_weight *= scale
            else:
                # Fallback to minimum values if both were zero
                self.ms_weight = 0.05
                self.grad_weight = 0.05
        
        # Ensure L1 doesn't go below minimum either
        if self.l1_weight < 0.1:
            # If L1 would be too low, rebalance all three
            total = self.ms_weight + self.grad_weight
            self.ms_weight = 0.45 * (self.ms_weight / total) if total > 0 else 0.15
            self.grad_weight = 0.45 * (self.grad_weight / total) if total > 0 else 0.30
            self.l1_weight = 0.1
        
        self.adjustment_step += 1
        
        self._cached_mode = 'Aggressive' if self.aggressive_mode else 'Stable'
        status = {
            'is_cooldown': self.is_in_cooldown,
            'cooldown_remaining': self.cooldown_steps if self.is_in_cooldown else 0,
            'mode': self._cached_mode
        }
        
        return self.l1_weight, self.ms_weight, self.grad_weight, self.perceptual_weight, status
    
    def clip_gradients(self, model):
        """
        Clip gradients adaptively
        
        Args:
            model: Model with gradients
            
        Returns:
            Tuple of (total_norm, clip_value)
        """
        # Compute gradient norm, skipping any NaN/inf parameter gradients.
        # NaN gradients arise during AMP training when activations explode;
        # without this guard, total_norm becomes NaN, which poisons clip_value
        # and disables clipping entirely — accelerating the explosion.
        total_norm = 0.0
        has_bad_grads = False
        for p in model.parameters():
            if p.grad is not None:
                pn = p.grad.data.norm(2).item()
                if np.isfinite(pn):
                    total_norm += pn ** 2
                else:
                    has_bad_grads = True
        total_norm = total_norm ** 0.5

        # NaN/Inf guard: if ANY gradient is non-finite, do NOT call clip_grad_norm_.
        # torch.nn.utils.clip_grad_norm_() recomputes the norm internally (without
        # filtering), so if even one gradient is NaN, its internal total_norm becomes
        # NaN → clip_coef = NaN → every parameter.grad *= NaN, spreading corruption
        # to all previously finite gradients.  The GradScaler will detect the bad
        # gradients via found_inf (set during unscale_()) and skip optimizer.step()
        # anyway, so skipping clipping here is safe and avoids the contamination.
        if has_bad_grads:
            return total_norm, self.clip_value

        # Only record finite norms so the history stays clean.
        if np.isfinite(total_norm):
            self.grad_norms.append(total_norm)

        # Keep last 500 norms
        if len(self.grad_norms) > 500:
            self.grad_norms.pop(0)

        # Update clip value after warmup.  Filter out any stale NaN entries
        # (shouldn't occur after the guard above, but defensive nonetheless).
        if len(self.grad_norms) >= 100:
            finite_norms = [v for v in self.grad_norms if np.isfinite(v)]
            if finite_norms:
                new_clip = np.percentile(finite_norms, 95)
                # Smooth update with minimum floor to prevent feedback-loop collapse.
                # 1.0 floor (was 0.5) ensures deep layers keep receiving gradient signal.
                MIN_CLIP_VALUE = 1.0
                self.clip_value = max(MIN_CLIP_VALUE, 0.9 * self.clip_value + 0.1 * new_clip)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
        
        return total_norm, self.clip_value
    
    def update_plateau_tracker(self, loss, quality=None, step=0, warmup_steps=1000):
        """
        ADVANCED plateau detection with multiple signals:
        - Loss improvement (adaptive threshold based on loss level)
        - Quality improvement (if available)
        - EMA trend analysis
        - Grace period for noisy improvements
        
        Args:
            loss:         Current total loss value
            quality:      Optional quality metric (KI quality %)
            step:         Current global training step
            warmup_steps: Effective warmup guard: plateau tracking is frozen while
                          step < warmup_steps.  The caller (trainer) is responsible
                          for passing max(lr_warmup_steps, data_strategy_warmup_end)
                          so that the counter only starts once BOTH the LR ramp-up
                          AND the DataStrategy Phase-1 data curriculum are complete.
        """
        # During warmup the loss is dominated by the LR ramp-up schedule and the
        # homogeneous Phase-1 data distribution, not by genuine convergence.
        # Keep the tracker reset so it starts fresh with a clean slate once both
        # warmup phases end.
        if step < warmup_steps:
            self.plateau_counter = 0
            self.ema_loss = None
            self.best_loss = float('inf')
            self.ema_quality = None
            self.best_quality = 0.0
            return

        # NaN/Inf guard: a non-finite loss (from a bad batch) must never seed or
        # update the EMA.  NaN * alpha + finite * (1-alpha) = NaN forever, and
        # NaN comparisons always return False — making the plateau counter tick up
        # on every step and eventually triggering an unwanted safety reset.
        if not math.isfinite(loss):
            return

        # Initialize EMA on first call
        if self.ema_loss is None:
            self.ema_loss = loss
            self.best_loss = loss
            if quality is not None:
                self.ema_quality = quality
                self.best_quality = quality
            self.plateau_counter = 0
            return
        
        # Update EMAs (alpha = 0.1 for slow adaptation)
        self.ema_loss = self.ema_alpha * loss + (1 - self.ema_alpha) * self.ema_loss
        if quality is not None:
            if self.ema_quality is None:
                self.ema_quality = quality
            else:
                self.ema_quality = self.ema_alpha * quality + (1 - self.ema_alpha) * self.ema_quality
        
        # Determine adaptive threshold based on loss level
        threshold = 0.997  # Default 0.3%
        for loss_level, thresh in sorted(self.plateau_threshold_map.items()):
            if loss < loss_level:
                threshold = thresh
                break
        
        # CHECK 1: Loss improvement
        loss_improved = loss < self.best_loss * threshold
        
        # CHECK 2: EMA trend
        ema_trend_good = self.ema_loss < self.best_loss * (threshold + 0.001)
        
        # CHECK 3: Quality improvement
        quality_improved = False
        ema_quality_trend_good = False
        if quality is not None:
            quality_improved = quality > self.best_quality * 1.001  # 0.1% improvement
            if self.ema_quality is not None:
                ema_quality_trend_good = self.ema_quality > self.best_quality * 0.9995
        
        # DECISION: Reset if any significant signal
        should_reset = (
            loss_improved or
            quality_improved or
            (ema_trend_good and ema_quality_trend_good)
        )
        
        if should_reset:
            if loss < self.best_loss:
                self.best_loss = loss
            if quality is not None and quality > self.best_quality:
                self.best_quality = quality
            self.plateau_counter = 0
        else:
            # Grace period: slower counter increase if slight improvement
            if self.grace_enabled:
                slight_improvement = (
                    loss < self.best_loss * 1.002 or
                    (quality is not None and quality > self.best_quality * 0.999)
                )
                if slight_improvement and self.plateau_counter > 0:
                    self.plateau_counter = max(0, self.plateau_counter - 0.5)
                else:
                    self.plateau_counter += 1
            else:
                self.plateau_counter += 1
    
    def is_plateau(self):
        """Return True if training has plateaued"""
        return self.plateau_counter >= self.plateau_patience
    
    def update_validation_tracker(self, val_loss, val_quality=None):
        """
        Track validation loss trend to detect overfitting.

        Called after every validation run.  Uses EMA smoothing to filter
        noise, then checks whether the smoothed val loss improved by at
        least 0.5%.  If not, the no-improvement counter is incremented.
        Once it reaches val_plateau_patience, is_val_plateau() returns True.

        Args:
            val_loss:    Current validation loss (float).  Ignored if None or <= 0.
            val_quality: Optional KI quality metric (float in [0,1]).  Reserved
                         for future use; not currently used in the decision.
        """
        if val_loss is None or val_loss <= 0:
            return

        # Initialise on first valid call
        if self.ema_val_loss is None:
            self.ema_val_loss = val_loss
            self.best_val_loss = val_loss
            return

        # EMA smoothing: α=0.3 for the new observation, 0.7 for the running mean.
        # Equivalent to a window of ~3 validations.
        self.ema_val_loss = 0.3 * val_loss + 0.7 * self.ema_val_loss

        # Improvement check: require at least 0.5% reduction
        if self.ema_val_loss < self.best_val_loss * 0.995:
            self.best_val_loss = self.ema_val_loss
            self.val_no_improve_count = 0
        else:
            self.val_no_improve_count += 1

    def is_val_plateau(self):
        """Return True if validation loss has plateaued (no improvement for val_plateau_patience runs)."""
        return self.val_no_improve_count >= self.val_plateau_patience

    def get_plateau_info(self):
        """Get detailed plateau status for logging/UI"""
        return {
            'plateau_counter': int(self.plateau_counter),
            'plateau_patience': self.plateau_patience,
            'plateau_threshold': self.plateau_safety_threshold,
            'best_loss': self.best_loss,
            'best_quality': getattr(self, 'best_quality', 0.0),
            'ema_loss': self.ema_loss,
            'ema_quality': getattr(self, 'ema_quality', None),
            'is_plateau': self.is_plateau(),
            'steps_until_reset': max(0, self.plateau_safety_threshold - int(self.plateau_counter)),
            'val_no_improve_count': self.val_no_improve_count,
            'val_plateau_patience': self.val_plateau_patience,
            'best_val_loss': self.best_val_loss if self.best_val_loss != float('inf') else None,
            'ema_val_loss': self.ema_val_loss,
            'is_val_plateau': self.is_val_plateau(),
        }
    
    def get_status(self):
        """
        Get current adaptive system status
        
        Returns:
            Dict with current state including cooldown and mode info
        """
        return {
            'l1_weight': self.l1_weight,
            'ms_weight': self.ms_weight,
            'grad_weight': self.grad_weight,
            'perceptual_weight': self.perceptual_weight,
            'grad_clip': self.clip_value,
            'aggressive_mode': self.aggressive_mode,
            'is_cooldown': self.is_in_cooldown,
            'cooldown_remaining': self.cooldown_steps if self.is_in_cooldown else 0,
            'mode': self._cached_mode,
            'plateau_counter': self.plateau_counter,
            'plateau_patience': self.plateau_patience,
            'best_loss': self.best_loss,
            'ema_l1_loss': self.ema_l1_loss if self.ema_l1_loss is not None else 0.0,
            'val_no_improve_count': self.val_no_improve_count,
            'val_plateau_patience': self.val_plateau_patience,
            'best_val_loss': self.best_val_loss if self.best_val_loss != float('inf') else None,
            'ema_val_loss': self.ema_val_loss,
            'is_val_plateau': self.is_val_plateau(),
        }
