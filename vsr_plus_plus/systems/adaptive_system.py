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
        # Loss weights
        self.l1_weight = initial_l1
        self.ms_weight = initial_ms
        self.grad_weight = initial_grad
        self.perceptual_weight = initial_perceptual
        
        # EMA for smooth loss tracking (50-step window, alpha = 2/(N+1))
        self.ema_l1_loss = None
        self.ema_window = 50
        self.ema_alpha = 2.0 / (self.ema_window + 1)
        
        # Momentum: maximum change per step (1% = 0.01)
        self.max_weight_change = 0.01
        
        # Cooldown mechanism
        self.cooldown_steps = 0
        self.cooldown_duration = 100  # Wait 100 steps after adjustment
        self.is_in_cooldown = False
        
        # Gradient clipping
        self.clip_value = 1.5
        self.grad_norms = []
        
        # Sharpness tracking
        self.sharpness_history = []
        self.adjustment_step = 0
        
        # Aggressive mode
        self.aggressive_mode = False
        self.aggressive_counter = 0
        self.aggressive_max_steps = 5000
        
        # Thresholds
        self.extreme_grad_threshold = 0.025
        self.extreme_sharpness_threshold = 0.70
        self.aggressive_blur_threshold = 0.72
        self.aggressive_stabilization_threshold = 0.75
        self.normal_blur_threshold = 0.75
        
        # L1 loss thresholds for perceptual weight control
        self.l1_stable_threshold = 0.010  # L1 below this is "stable and good"
        self.l1_unstable_threshold = 0.020  # L1 above this is "unstable"
        
        # Update frequencies
        self.aggressive_update_frequency = 10
        self.normal_update_frequency = 50
        
        # Plateau detection
        self.best_loss = float('inf')
        self.plateau_counter = 0
        self.plateau_patience = 300
    
    def _update_ema_loss(self, l1_loss_value):
        """Update EMA of L1 loss for smooth tracking"""
        if self.ema_l1_loss is None:
            self.ema_l1_loss = l1_loss_value
        else:
            self.ema_l1_loss = self.ema_alpha * l1_loss_value + (1 - self.ema_alpha) * self.ema_l1_loss
    
    def _apply_momentum(self, current_value, target_value):
        """Apply momentum constraint: limit change to max_weight_change"""
        delta = target_value - current_value
        # Clip delta to maximum allowed change
        delta = np.clip(delta, -self.max_weight_change, self.max_weight_change)
        return current_value + delta
    
    def _update_perceptual_weight(self):
        """
        Intelligently adjust perceptual weight based on L1 loss stability
        
        Logic:
        - If L1 is stable and low -> allow perceptual weight to increase (more details)
        - If L1 is unstable/high -> decrease perceptual weight (focus on structure)
        """
        if self.ema_l1_loss is None:
            return
        
        # Skip during cooldown
        if self.is_in_cooldown:
            return
        
        target_perc = self.perceptual_weight
        
        if self.ema_l1_loss < self.l1_stable_threshold:
            # L1 is stable and good -> slowly increase perceptual
            target_perc = min(0.15, self.perceptual_weight + 0.001)
        elif self.ema_l1_loss > self.l1_unstable_threshold:
            # L1 is unstable -> decrease perceptual
            target_perc = max(0.0, self.perceptual_weight - 0.002)
        
        # Apply momentum
        self.perceptual_weight = self._apply_momentum(self.perceptual_weight, target_perc)
    
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
        
        # Update EMA with L1 loss if provided
        if current_l1_loss is not None:
            self._update_ema_loss(current_l1_loss)
        
        # Check extreme conditions
        extreme = False
        
        if sharpness_ratio < self.extreme_sharpness_threshold:
            extreme = True
        
        if extreme and not self.aggressive_mode:
            self.aggressive_mode = True
            self.aggressive_counter = 0
            # Start cooldown
            self.is_in_cooldown = True
            self.cooldown_steps = self.cooldown_duration
            # Smooth boost with momentum
            target_grad = 0.30
            target_l1 = 0.55
            target_ms = 0.15
            self.grad_weight = self._apply_momentum(self.grad_weight, target_grad)
            self.l1_weight = self._apply_momentum(self.l1_weight, target_l1)
            self.ms_weight = self._apply_momentum(self.ms_weight, target_ms)
        
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
                'mode': str ('Aggressive' or 'Stable')
            }
        """
        # Update cooldown counter
        if self.is_in_cooldown:
            self.cooldown_steps -= 1
            if self.cooldown_steps <= 0:
                self.is_in_cooldown = False
        
        # Detect extreme conditions (also updates EMA)
        sharpness_ratio = self.detect_extreme_conditions(pred, target, current_l1_loss)
        
        # Update perceptual weight based on L1 stability
        self._update_perceptual_weight()
        
        # Update frequency based on mode
        if self.aggressive_mode:
            update_freq = self.aggressive_update_frequency
            min_measurements = 2
            adjustment_factor = 1.15
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
            status = {
                'is_cooldown': self.is_in_cooldown,
                'cooldown_remaining': self.cooldown_steps if self.is_in_cooldown else 0,
                'mode': 'Aggressive' if self.aggressive_mode else 'Stable'
            }
            return self.l1_weight, self.ms_weight, self.grad_weight, self.perceptual_weight, status
        
        # Add to history
        self.sharpness_history.append(sharpness_ratio)
        if len(self.sharpness_history) > 200:
            self.sharpness_history.pop(0)
        
        # Warmup
        if len(self.sharpness_history) < min_measurements:
            status = {
                'is_cooldown': False,
                'cooldown_remaining': 0,
                'mode': 'Aggressive' if self.aggressive_mode else 'Stable'
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
            # Image is blurry, boost gradient
            target_grad = min(0.5, self.grad_weight * adjustment_factor)
            target_ms = min(0.2, self.ms_weight)
            target_l1 = max(0.3, 1.0 - target_grad - target_ms)
            # Start cooldown after adjustment
            self.is_in_cooldown = True
            self.cooldown_steps = self.cooldown_duration
        elif avg_sharpness > 0.92:
            # Image is sharp enough, reduce gradient
            target_grad = max(0.15, self.grad_weight * 0.95)
            target_ms = min(0.2, self.ms_weight)
            target_l1 = 1.0 - target_grad - target_ms
            # Start cooldown after adjustment
            self.is_in_cooldown = True
            self.cooldown_steps = self.cooldown_duration
        
        # Apply momentum (smooth transitions)
        self.grad_weight = self._apply_momentum(self.grad_weight, target_grad)
        self.ms_weight = self._apply_momentum(self.ms_weight, target_ms)
        self.l1_weight = self._apply_momentum(self.l1_weight, target_l1)
        
        self.adjustment_step += 1
        
        status = {
            'is_cooldown': self.is_in_cooldown,
            'cooldown_remaining': self.cooldown_steps if self.is_in_cooldown else 0,
            'mode': 'Aggressive' if self.aggressive_mode else 'Stable'
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
        # Compute gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        self.grad_norms.append(total_norm)
        
        # Keep last 500 norms
        if len(self.grad_norms) > 500:
            self.grad_norms.pop(0)
        
        # Update clip value after warmup
        if len(self.grad_norms) >= 100:
            new_clip = np.percentile(self.grad_norms, 95)
            # Smooth update
            self.clip_value = 0.9 * self.clip_value + 0.1 * new_clip
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
        
        return total_norm, self.clip_value
    
    def update_plateau_tracker(self, loss):
        """
        Update plateau tracking
        
        Args:
            loss: Current loss value
        """
        # Check for improvement
        if loss < self.best_loss * 0.997:
            self.best_loss = loss
            self.plateau_counter = 0
        else:
            self.plateau_counter += 1
    
    def is_plateau(self):
        """Return True if training has plateaued"""
        return self.plateau_counter >= self.plateau_patience
    
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
            'mode': 'Aggressive' if self.aggressive_mode else 'Stable',
            'plateau_counter': self.plateau_counter,
            'best_loss': self.best_loss,
            'ema_l1_loss': self.ema_l1_loss if self.ema_l1_loss is not None else 0.0
        }
