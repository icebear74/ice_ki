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
        
    def update(self, pred, target, step):
        """Update weights based on prediction sharpness"""
        
        # Only adjust every 100 steps
        if step % 100 != 0:
            return self.l1_weight, self.ms_weight, self.grad_weight
        
        # Store old weights for comparison
        old_l1 = self.l1_weight
        old_ms = self.ms_weight
        old_grad = self.grad_weight
        
        # Compute sharpness ratio
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
        
        self.sharpness_history.append(sharpness_ratio)
        
        # Keep last 100 measurements
        if len(self.sharpness_history) > 100:
            self.sharpness_history.pop(0)
        
        # Need at least 20 measurements
        if len(self.sharpness_history) < 20:
            return self.l1_weight, self.ms_weight, self.grad_weight
        
        avg_sharpness = np.mean(self.sharpness_history[-20:])
        weights_changed = False
        
        # Too blurry? Increase gradient loss!
        if avg_sharpness < 0.75:
            self.grad_weight = min(0.5, self.grad_weight * 1.05)
            self.ms_weight = min(0.2, self.ms_weight)
            self.l1_weight = max(0.3, 1.0 - self.grad_weight - self.ms_weight)
            weights_changed = True
            
            # Store notification every 10th adjustment
            if self.adjustment_step % 10 == 0:
                self.last_notification = {
                    'type': 'blur',
                    'step': step,
                    'message': f"🔍 BLUR → Weights: L1 {old_l1:.2f}→{self.l1_weight:.2f} MS {old_ms:.2f}→{self.ms_weight:.2f} Grad {old_grad:.2f}→{self.grad_weight:.2f}",
                    'details': f"Sharpness: {avg_sharpness:.1%}"
                }
        
        # Sharp enough? Focus on color accuracy
        elif avg_sharpness > 0.92:
            self.grad_weight = max(0.15, self.grad_weight * 0.95)
            self.ms_weight = min(0.2, self.ms_weight)
            self.l1_weight = 1.0 - self.grad_weight - self.ms_weight
            weights_changed = True
            
            # Store notification every 10th adjustment
            if self.adjustment_step % 10 == 0:
                self.last_notification = {
                    'type': 'sharp',
                    'step': step,
                    'message': f"✅ SHARP → Weights: L1 {old_l1:.2f}→{self.l1_weight:.2f} MS {old_ms:.2f}→{self.ms_weight:.2f} Grad {old_grad:.2f}→{self.grad_weight:.2f}",
                    'details': f"Sharpness: {avg_sharpness:.1%}"
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
        
    def on_train_step(self, loss, pred, target, step):
        """Call this every training step"""
        # Update LR based on loss (pass step for logging)
        self.lr_scheduler.step(loss, global_step=step)
        
        # Get dynamic loss weights
        l1_w, ms_w, grad_w = self.loss_weights.update(pred, target, step)
        
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
            'plateau_counter': self.lr_scheduler.plateau_counter
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
