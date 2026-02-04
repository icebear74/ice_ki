"""
Adaptive Training System

Based on the existing adaptive_system.py but adapted for VSR++
- Adaptive loss weights (L1, MS, Grad)
- Adaptive gradient clipping
- Aggressive mode detection
- Plateau detection
"""

import torch
import numpy as np


class AdaptiveSystem:
    """
    Complete adaptive training system
    
    Combines:
    - Dynamic loss weights
    - Adaptive gradient clipping
    - Aggressive mode for blur correction
    """
    
    def __init__(self, initial_l1=0.6, initial_ms=0.2, initial_grad=0.2):
        # Loss weights
        self.l1_weight = initial_l1
        self.ms_weight = initial_ms
        self.grad_weight = initial_grad
        
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
        
        # Update frequencies
        self.aggressive_update_frequency = 10
        self.normal_update_frequency = 50
        
        # Plateau detection
        self.best_loss = float('inf')
        self.plateau_counter = 0
        self.plateau_patience = 300
    
    def detect_extreme_conditions(self, pred, target, current_grad_loss=None):
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
        
        # Check extreme conditions
        extreme = False
        
        if sharpness_ratio < self.extreme_sharpness_threshold:
            extreme = True
            
        if current_grad_loss and current_grad_loss > self.extreme_grad_threshold:
            extreme = True
        
        if extreme and not self.aggressive_mode:
            self.aggressive_mode = True
            self.aggressive_counter = 0
            # Immediate boost
            self.grad_weight = 0.30
            self.l1_weight = 0.55
            self.ms_weight = 0.15
        
        return sharpness_ratio
    
    def update_loss_weights(self, pred, target, step, current_grad_loss=None):
        """
        Update loss weights based on image quality
        
        Args:
            pred: Predicted image
            target: Target image
            step: Current training step
            current_grad_loss: Current gradient loss value
            
        Returns:
            Tuple of (l1_weight, ms_weight, grad_weight)
        """
        # Detect extreme conditions
        sharpness_ratio = self.detect_extreme_conditions(pred, target, current_grad_loss)
        
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
        
        # Update check
        if step % update_freq != 0:
            return self.l1_weight, self.ms_weight, self.grad_weight
        
        # Add to history
        self.sharpness_history.append(sharpness_ratio)
        if len(self.sharpness_history) > 200:
            self.sharpness_history.pop(0)
        
        # Warmup
        if len(self.sharpness_history) < min_measurements:
            return self.l1_weight, self.ms_weight, self.grad_weight
        
        # Compute average
        window = min(20, len(self.sharpness_history))
        avg_sharpness = np.mean(self.sharpness_history[-window:])
        
        # Adjust weights
        if avg_sharpness < blur_threshold:
            self.grad_weight = min(0.5, self.grad_weight * adjustment_factor)
            self.ms_weight = min(0.2, self.ms_weight)
            self.l1_weight = max(0.3, 1.0 - self.grad_weight - self.ms_weight)
        elif avg_sharpness > 0.92:
            self.grad_weight = max(0.15, self.grad_weight * 0.95)
            self.ms_weight = min(0.2, self.ms_weight)
            self.l1_weight = 1.0 - self.grad_weight - self.ms_weight
        
        self.adjustment_step += 1
        return self.l1_weight, self.ms_weight, self.grad_weight
    
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
            Dict with current state
        """
        return {
            'loss_weights': (self.l1_weight, self.ms_weight, self.grad_weight),
            'grad_clip': self.clip_value,
            'aggressive_mode': self.aggressive_mode,
            'plateau_counter': self.plateau_counter,
            'best_loss': self.best_loss
        }
