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
    """
    
    def __init__(self, optimizer, warmup_steps=1000, max_steps=100000, 
                 max_lr=1e-4, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        
        self.current_phase = 'warmup'
        self.plateau_reductions = 0
    
    def step(self, global_step, plateau_detected=False):
        """
        Update learning rate
        
        Args:
            global_step: Current training step
            plateau_detected: Whether plateau was detected
            
        Returns:
            Tuple of (current_lr, phase_name)
        """
        # Handle plateau emergency
        if plateau_detected:
            current_lr = self.optimizer.param_groups[0]['lr']
            new_lr = max(current_lr * 0.5, self.min_lr)
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            
            self.plateau_reductions += 1
            return new_lr, 'plateau_reduced'
        
        # Phase 1: Warmup
        if global_step < self.warmup_steps:
            # Linear warmup from 0 to max_lr
            lr = self.max_lr * (global_step / self.warmup_steps)
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
