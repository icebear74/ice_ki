"""
Adaptive LR Scheduler with Warmup

Three phases:
1. Warmup (0-1000 steps): Linear 0 → 1e-4
2. Cosine Annealing (1000-max): 1e-4 → 1e-6
3. Plateau Emergency: Current LR × 0.5
"""

import math


class AdaptiveLRScheduler:
    """
    Learning rate scheduler with warmup and cosine annealing
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps (default: 1000)
        max_steps: Maximum training steps
        max_lr: Maximum learning rate after warmup (default: 1e-4)
        min_lr: Minimum learning rate at end (default: 1e-6)
        initial_lr: Initial learning rate for warmup start (default: None, uses optimizer's lr)
    """
    
    def __init__(self, optimizer, warmup_steps=1000, max_steps=100000, 
                 max_lr=1e-4, min_lr=1e-6, initial_lr=None):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        
        # Get initial LR from optimizer if not specified
        if initial_lr is None:
            self.initial_lr = optimizer.param_groups[0]['lr']
        else:
            self.initial_lr = initial_lr
        
        self.current_phase = 'warmup'
        self.plateau_reductions = 0
        
        # FIX 4: Plateau recovery mechanism
        self.plateau_boost_available = True
        self.last_boost_step = 0
        self.boost_cooldown = 1000  # Wait 1000 steps between boosts
    
    def step(self, global_step, plateau_detected=False):
        """
        Update learning rate
        
        Args:
            global_step: Current training step
            plateau_detected: Whether plateau was detected
            
        Returns:
            Tuple of (current_lr, phase_name)
        """
        # FIX 4: PLATEAU RECOVERY - If stuck for 300+ steps, boost LR
        if plateau_detected and self.plateau_boost_available:
            old_lr = self.optimizer.param_groups[0]['lr']
            # Triple LR (but cap at MAX_LR)
            new_lr = min(old_lr * 3.0, self.max_lr)
            
            # Apply to optimizer
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            
            current_lr = new_lr
            
            # Disable boost for next 1000 steps (prevent spam)
            self.plateau_boost_available = False
            self.last_boost_step = global_step
            
            # Log event
            print(f"\n⚡ LR BOOST TRIGGERED at step {global_step}")
            print(f"   {old_lr:.2e} -> {new_lr:.2e} (×{new_lr/old_lr:.1f})")
            
            return current_lr, 'plateau_boost'
        
        # Re-enable boost after cooldown
        if global_step - self.last_boost_step > self.boost_cooldown:
            self.plateau_boost_available = True
        
        # Phase 1: Warmup
        if global_step < self.warmup_steps:
            # Linear warmup from initial_lr to max_lr
            progress = global_step / self.warmup_steps
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * progress
            self.current_phase = 'warmup'
        
        # Phase 2: Cosine Annealing
        else:
            # Cosine annealing from max_lr to min_lr
            progress = (global_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            progress = min(1.0, progress)
            
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            self.current_phase = 'cosine'
        
        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr, self.current_phase
    
    def get_current_lr(self):
        """Get current learning rate without updating"""
        return self.optimizer.param_groups[0]['lr']
    
    def get_current_phase(self):
        """Get current phase without updating"""
        return self.current_phase
    
    def get_status(self):
        """Return scheduler status for GUI/logging"""
        return {
            'plateau_boost_available': self.plateau_boost_available,
            'steps_since_boost': self.last_boost_step,
        }
