"""
Adaptive Training System for VSR
Auto-adjusts LR, Loss Weights, and Gradient Clipping
"""

import torch
import numpy as np

class AdaptiveLRScheduler:
    """
    Adaptive Learning Rate Scheduling
    - Increases LR on plateau
    - Decreases LR on divergence
    """
    def __init__(self, optimizer, patience=300, factor=1.3, 
                 min_lr=1e-6, max_lr=1e-3, cooldown=100):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cooldown = cooldown
        
        self.best_loss = float('inf')
        self.plateau_counter = 0
        self.cooldown_counter = 0
        self.loss_history = []
        self.last_notification = None  # Store last change notification
        
    def step(self, current_loss, global_step=None):
        """Call every training step with current loss"""
        self.loss_history.append(current_loss)
        
        # Keep last 1000 losses
        if len(self.loss_history) > 1000:
            self.loss_history.pop(0)
        
        # Cooldown period after LR change
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return
        
        # Check for improvement (0.3% threshold)
        if current_loss < self.best_loss * 0.997:
            self.best_loss = current_loss
            self.plateau_counter = 0
            return
        
        # Plateau detection
        self.plateau_counter += 1
        
        if self.plateau_counter >= self.patience:
            current_lr = self.optimizer.param_groups[0]['lr']
            new_lr = min(current_lr * self.factor, self.max_lr)
            
            if new_lr > current_lr:
                for pg in self.optimizer.param_groups:
                    pg['lr'] = new_lr
                
                # Store notification instead of printing
                change_pct = ((new_lr/current_lr - 1)*100)
                self.last_notification = {
                    'type': 'plateau',
                    'step': global_step,
                    'message': f"⚡ PLATEAU → LR: {current_lr:.2e}→{new_lr:.2e} (+{change_pct:.1f}%)",
                    'details': f"Loss: {current_loss:.6f} (Best: {self.best_loss:.6f})"
                }
                
                self.plateau_counter = 0
                self.cooldown_counter = self.cooldown
        
        # Divergence detection
        if current_loss > self.best_loss * 1.15 and len(self.loss_history) > 50:
            current_lr = self.optimizer.param_groups[0]['lr']
            new_lr = max(current_lr * 0.6, self.min_lr)
            
            if new_lr < current_lr:
                for pg in self.optimizer.param_groups:
                    pg['lr'] = new_lr
                
                # Store notification instead of printing
                change_pct = ((new_lr/current_lr - 1)*100)
                self.last_notification = {
                    'type': 'divergence',
                    'step': global_step,
                    'message': f"⚠️  DIVERGENCE → LR: {current_lr:.2e}→{new_lr:.2e} ({change_pct:.1f}%)",
                    'details': f"Loss: {current_loss:.6f} (Best: {self.best_loss:.6f})"
                }
                
                self.best_loss = current_loss
                self.plateau_counter = 0
                self.cooldown_counter = self.cooldown


class DynamicLossWeights:
    """
    Dynamic Loss Weight Adjustment
    - Increases gradient loss when blurry
    - Balances L1/MS/Grad automatically
    """
    def __init__(self, initial_l1=0.6, initial_ms=0.2, initial_grad=0.2):
        self.l1_weight = initial_l1
        self.ms_weight = initial_ms
        self.grad_weight = initial_grad
        
        self.sharpness_history = []
        self.adjustment_step = 0
        self.last_notification = None  # Store last change notification
        
        # Aggressive mode
        self.aggressive_mode = False
        self.aggressive_counter = 0
        self.aggressive_max_steps = 5000
        
        # Extreme condition thresholds
        self.extreme_grad_threshold = 0.025
        self.extreme_sharpness_threshold = 0.70
        
        # Aggressive mode parameters (extracted for easier tuning)
        self.aggressive_update_frequency = 10  # Every 10 steps (5x faster than normal)
        self.aggressive_min_measurements = 2  # Only 2 needed
        self.aggressive_adjustment_factor = 1.15  # Aggressive boost
        self.aggressive_blur_threshold = 0.72  # More aggressive threshold
        self.aggressive_stabilization_threshold = 0.75  # Exit threshold
        
        # Fine-tuning mode parameters
        self.normal_update_frequency = 50  # Every 50 steps
        self.normal_min_measurements = 10  # Need 10 measurements
        self.normal_adjustment_factor = 1.05  # Conservative
        self.normal_blur_threshold = 0.75  # Normal threshold
    
    def detect_extreme_conditions(self, pred, target, current_grad_loss=None):
        """Check if immediate intervention needed"""
        # Compute sharpness
        with torch.no_grad():
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
        trigger_reason = []
        
        if sharpness_ratio < self.extreme_sharpness_threshold:
            extreme = True
            trigger_reason.append(f"Blur: {sharpness_ratio:.2%}")
            
        if current_grad_loss and current_grad_loss > self.extreme_grad_threshold:
            extreme = True
            trigger_reason.append(f"GradLoss: {current_grad_loss:.4f}")
        
        if extreme and not self.aggressive_mode:
            self.aggressive_mode = True
            self.aggressive_counter = 0
            # Immediate boost!
            self.grad_weight = 0.30
            self.l1_weight = 0.55
            self.ms_weight = 0.20  # Explicitly set for consistency
            
            # Store notification instead of printing
            self.last_notification = {
                'type': 'aggressive_mode_activated',
                'message': f"⚡ AGGRESSIVE MODE → {', '.join(trigger_reason)}",
                'details': f"Weights: L1=0.55, MS=0.20, Grad=0.30"
            }
        
        return sharpness_ratio
        
    def update(self, pred, target, step, current_grad_loss=None):
        """Update weights - now with aggressive mode!"""
        
        # Detect extreme conditions first
        sharpness_ratio = self.detect_extreme_conditions(pred, target, current_grad_loss)
        
        # Update frequency based on mode
        if self.aggressive_mode:
            update_frequency = self.aggressive_update_frequency
            min_measurements = self.aggressive_min_measurements
            adjustment_factor = self.aggressive_adjustment_factor
            blur_threshold = self.aggressive_blur_threshold
            
            self.aggressive_counter += 1
            
            # Deactivate after max steps or if stabilized
            if self.aggressive_counter >= self.aggressive_max_steps:
                self.aggressive_mode = False
                # Store notification
                self.last_notification = {
                    'type': 'aggressive_mode_completed',
                    'message': f"✅ AGGRESSIVE MODE COMPLETE → {self.aggressive_max_steps} steps",
                    'details': "Switching to fine-tuning mode"
                }
            elif sharpness_ratio > self.aggressive_stabilization_threshold and len(self.sharpness_history) > 50:
                avg_recent = np.mean(self.sharpness_history[-20:])
                if avg_recent > self.aggressive_stabilization_threshold:
                    self.aggressive_mode = False
                    # Store notification
                    self.last_notification = {
                        'type': 'aggressive_mode_stabilized',
                        'message': f"✅ STABILIZED → Sharpness: {avg_recent:.2%}",
                        'details': "Switching to fine-tuning mode"
                    }
        else:
            update_frequency = self.normal_update_frequency
            min_measurements = self.normal_min_measurements
            adjustment_factor = self.normal_adjustment_factor
            blur_threshold = self.normal_blur_threshold
        
        # Update check
        if step % update_frequency != 0:
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
            old_grad = self.grad_weight
            self.grad_weight = min(0.5, self.grad_weight * adjustment_factor)
            self.ms_weight = min(0.2, self.ms_weight)
            self.l1_weight = max(0.3, 1.0 - self.grad_weight - self.ms_weight)
            
            # Store notification (only when weights actually changed)
            if abs(self.grad_weight - old_grad) > 0.001:
                mode = "AGGRESSIVE" if self.aggressive_mode else "NORMAL"
                self.last_notification = {
                    'type': 'blur_correction',
                    'message': f"🔍 {mode} BLUR → Sharpness: {avg_sharpness:.2%}",
                    'details': f"L1={self.l1_weight:.2f}, MS={self.ms_weight:.2f}, Grad={self.grad_weight:.2f}"
                }
        
        elif avg_sharpness > 0.92:
            old_grad = self.grad_weight
            self.grad_weight = max(0.15, self.grad_weight * 0.95)
            self.ms_weight = min(0.2, self.ms_weight)
            self.l1_weight = 1.0 - self.grad_weight - self.ms_weight
            
            # Store notification (only when weights changed)
            if abs(self.grad_weight - old_grad) > 0.001:
                self.last_notification = {
                    'type': 'sharp_reduction',
                    'message': f"✅ SHARP → Sharpness: {avg_sharpness:.2%}",
                    'details': f"L1={self.l1_weight:.2f}, MS={self.ms_weight:.2f}, Grad={self.grad_weight:.2f}"
                }
        
        self.adjustment_step += 1
        return self.l1_weight, self.ms_weight, self.grad_weight


class AdaptiveGradientClipper:
    """
    Adaptive Gradient Clipping
    - Monitors gradient norms
    - Auto-adjusts clip value
    """
    def __init__(self, initial_clip=1.5, percentile=95):
        self.clip_value = initial_clip
        self.percentile = percentile
        self.grad_norms = []
        
    def clip_gradients(self, model):
        """Compute norm, update clip value, and clip"""
        
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
            new_clip = np.percentile(self.grad_norms, self.percentile)
            # Don't change too drastically
            self.clip_value = 0.9 * self.clip_value + 0.1 * new_clip
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
        
        return total_norm, self.clip_value


class FullAdaptiveSystem:
    """
    Complete Adaptive Training System
    Combines all adaptive components
    """
    def __init__(self, optimizer):
        self.lr_scheduler = AdaptiveLRScheduler(optimizer, patience=300, factor=1.3)
        self.loss_weights = DynamicLossWeights(
            initial_l1=0.6, 
            initial_ms=0.2, 
            initial_grad=0.2
        )
        self.grad_clipper = AdaptiveGradientClipper(initial_clip=1.5)
        
        print("\n" + "="*80)
        print("✨ FULL ADAPTIVE SYSTEM INITIALIZED!")
        print("="*80)
        print("📈 Adaptive LR: patience=300, factor=1.3")
        print("📊 Dynamic Loss Weights: L1=0.6, MS=0.2, Grad=0.2 (initial)")
        print("✂️  Adaptive Grad Clip: initial=1.5, auto-adjusting")
        print("="*80 + "\n")
        
    def on_train_step(self, loss, pred, target, step, current_grad_loss=None):
        """Call this every training step"""
        # Update LR based on loss (pass step for logging)
        self.lr_scheduler.step(loss, global_step=step)
        
        # Get dynamic loss weights (now with grad_loss for aggressive mode)
        l1_w, ms_w, grad_w = self.loss_weights.update(pred, target, step, current_grad_loss)
        
        return l1_w, ms_w, grad_w
    
    def on_backward(self, model):
        """Call this after backward, before optimizer.step()"""
        grad_norm, clip_val = self.grad_clipper.clip_gradients(model)
        return grad_norm, clip_val
    
    def get_status(self):
        """Get current adaptive system status"""
        return {
            'lr': self.lr_scheduler.optimizer.param_groups[0]['lr'],
            'loss_weights': (
                self.loss_weights.l1_weight,
                self.loss_weights.ms_weight,
                self.loss_weights.grad_weight
            ),
            'grad_clip': self.grad_clipper.clip_value,
            'best_loss': self.lr_scheduler.best_loss,
            'plateau_counter': self.lr_scheduler.plateau_counter,
            'aggressive_mode': self.loss_weights.aggressive_mode  # NEW!
        }
    
    def get_last_notification(self):
        """Get and clear last notification from any component"""
        # Check LR scheduler first
        if self.lr_scheduler.last_notification:
            notification = self.lr_scheduler.last_notification
            self.lr_scheduler.last_notification = None
            return notification
        
        # Then check loss weights
        if self.loss_weights.last_notification:
            notification = self.loss_weights.last_notification
            self.loss_weights.last_notification = None
            return notification
        
        return None
