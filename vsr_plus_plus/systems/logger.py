"""
Logging System - File and TensorBoard logging

Two-part system:
1. TrainingLogger - Human-readable file logging
2. TensorBoardLogger - TensorBoard integration
"""

import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """
    File-based logging system
    
    Manages two files:
    - training.log: Append-only event log
    - training_status.txt: Overwritten status file
    """
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(log_dir, "training.log")
        self.status_file = os.path.join(log_dir, "training_status.txt")
        
        # Create log file if doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write(f"{'='*80}\n")
                f.write(f"VSR++ Training Log - Started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*80}\n\n")
    
    def log_event(self, message):
        """
        Log an event to training.log
        
        Args:
            message: Event message
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_line)
    
    def update_status(self, step, epoch, losses, lr, speed, vram, config, metrics, adaptive_info):
        """
        Update training_status.txt with current status
        
        Args:
            step: Current step
            epoch: Current epoch
            losses: Loss dict
            lr: Learning rate
            speed: Training speed
            vram: VRAM usage
            config: Model config
            metrics: Validation metrics
            adaptive_info: Adaptive system info
        """
        with open(self.status_file, 'w') as f:
            f.write(f"{'='*80}\n")
            f.write(f"VSR++ Training Status - Updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"TRAINING PROGRESS:\n")
            f.write(f"  Step: {step:,}\n")
            f.write(f"  Epoch: {epoch}\n")
            f.write(f"  Learning Rate: {lr:.2e}\n")
            f.write(f"  Speed: {speed:.2f}s/iter\n")
            f.write(f"  VRAM: {vram:.2f}GB\n\n")
            
            f.write(f"LOSS COMPONENTS:\n")
            f.write(f"  Total: {losses.get('total', 0):.6f}\n")
            f.write(f"  L1: {losses.get('l1', 0):.6f}\n")
            f.write(f"  MS: {losses.get('ms', 0):.6f}\n")
            f.write(f"  Grad: {losses.get('grad', 0):.6f}\n\n")
            
            f.write(f"MODEL CONFIG:\n")
            f.write(f"  Features: {config.get('n_feats', 128)}\n")
            f.write(f"  Blocks: {config.get('n_blocks', 32)}\n")
            f.write(f"  Batch Size: {config.get('batch_size', 4)}\n\n")
            
            if metrics:
                f.write(f"VALIDATION METRICS:\n")
                f.write(f"  LR Quality: {metrics.get('lr_quality', 0)*100:.1f}%\n")
                f.write(f"  KI Quality: {metrics.get('ki_quality', 0)*100:.1f}%\n")
                f.write(f"  Improvement (Sum): {metrics.get('improvement', 0)*100:.1f}%\n")
                if 'ki_to_gt' in metrics:
                    f.write(f"  KI to GT (Sum): {metrics.get('ki_to_gt', 0)*100:.1f}%\n")
                if 'lr_to_gt' in metrics:
                    f.write(f"  LR to GT (Sum): {metrics.get('lr_to_gt', 0)*100:.1f}%\n")
                f.write(f"  PSNR: {metrics.get('ki_psnr', 0):.2f}dB\n")
                f.write(f"  SSIM: {metrics.get('ki_ssim', 0):.4f}\n\n")
            
            if adaptive_info:
                f.write(f"ADAPTIVE SYSTEM:\n")
                l1_w, ms_w, grad_w = adaptive_info.get('loss_weights', (0.6, 0.2, 0.2))
                f.write(f"  Loss Weights: L1={l1_w:.2f}, MS={ms_w:.2f}, Grad={grad_w:.2f}\n")
                f.write(f"  Gradient Clip: {adaptive_info.get('grad_clip', 1.5):.3f}\n")
                f.write(f"  Aggressive Mode: {adaptive_info.get('aggressive_mode', False)}\n")


class TensorBoardLogger:
    """
    TensorBoard logging system
    
    Logs all metrics, losses, and system info to TensorBoard
    """
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        # Log to 'active_run' subdirectory (like original train.py)
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "active_run"))
    
    def _to_float(self, value):
        """Convert value to float, handling Tensors"""
        import torch
        if isinstance(value, torch.Tensor):
            return value.item()
        return float(value) if value is not None else 0.0
    
    def log_losses(self, step, losses):
        """Log loss components"""
        self.writer.add_scalar('Loss/L1', self._to_float(losses.get('l1', 0)), step)
        self.writer.add_scalar('Loss/MS', self._to_float(losses.get('ms', 0)), step)
        self.writer.add_scalar('Loss/Grad', self._to_float(losses.get('grad', 0)), step)
        self.writer.add_scalar('Loss/Perceptual', self._to_float(losses.get('perceptual', 0)), step)
        self.writer.add_scalar('Loss/Total', self._to_float(losses.get('total', 0)), step)
    
    def log_lr(self, step, lr):
        """Log learning rate"""
        self.writer.add_scalar('Training/LearningRate', self._to_float(lr), step)
    
    def log_adaptive(self, step, adaptive_info):
        """Log adaptive system info"""
        if not adaptive_info:
            return
        
        l1_w, ms_w, grad_w = adaptive_info.get('loss_weights', (0.6, 0.2, 0.2))
        self.writer.add_scalar('Adaptive/L1_Weight', self._to_float(l1_w), step)
        self.writer.add_scalar('Adaptive/MS_Weight', self._to_float(ms_w), step)
        self.writer.add_scalar('Adaptive/Grad_Weight', self._to_float(grad_w), step)
        self.writer.add_scalar('Adaptive/GradientClip', self._to_float(adaptive_info.get('grad_clip', 1.5)), step)
        self.writer.add_scalar('Adaptive/AggressiveMode', 1 if adaptive_info.get('aggressive_mode') else 0, step)
        
        # Add BestLoss and PlateauCounter (like in original)
        if 'best_loss' in adaptive_info:
            self.writer.add_scalar('Adaptive/BestLoss', self._to_float(adaptive_info['best_loss']), step)
        if 'plateau_counter' in adaptive_info:
            self.writer.add_scalar('Adaptive/PlateauCounter', self._to_float(adaptive_info['plateau_counter']), step)
    
    def log_quality(self, step, metrics):
        """Log quality metrics"""
        if not metrics:
            return
        
        self.writer.add_scalar('Quality/LR_Quality', self._to_float(metrics.get('lr_quality', 0)), step)
        self.writer.add_scalar('Quality/KI_Quality', self._to_float(metrics.get('ki_quality', 0)), step)
        self.writer.add_scalar('Quality/Improvement', self._to_float(metrics.get('improvement', 0)), step)
        
        # Log additional GT difference metrics if available
        if 'ki_to_gt' in metrics:
            self.writer.add_scalar('Quality/KI_to_GT', self._to_float(metrics.get('ki_to_gt', 0)), step)
        if 'lr_to_gt' in metrics:
            self.writer.add_scalar('Quality/LR_to_GT', self._to_float(metrics.get('lr_to_gt', 0)), step)
    
    def log_metrics(self, step, metrics):
        """Log PSNR/SSIM metrics"""
        if not metrics:
            return
        
        self.writer.add_scalar('Metrics/LR_PSNR', self._to_float(metrics.get('lr_psnr', 0)), step)
        self.writer.add_scalar('Metrics/LR_SSIM', self._to_float(metrics.get('lr_ssim', 0)), step)
        self.writer.add_scalar('Metrics/KI_PSNR', self._to_float(metrics.get('ki_psnr', 0)), step)
        self.writer.add_scalar('Metrics/KI_SSIM', self._to_float(metrics.get('ki_ssim', 0)), step)
    
    def log_system(self, step, speed, vram):
        """Log system metrics"""
        self.writer.add_scalar('System/Speed_s_per_iter', self._to_float(speed), step)
        self.writer.add_scalar('System/VRAM_GB', self._to_float(vram), step)
    
    def log_gradients(self, step, grad_norm, activities):
        """Log gradient norms and activity"""
        self.writer.add_scalar('Gradients/TotalNorm', self._to_float(grad_norm), step)
        
        if activities:
            back_acts = activities.get('backward_trunk', [])
            forw_acts = activities.get('forward_trunk', [])
            
            # Log individual layer activities (like in original)
            for idx, act in enumerate(back_acts):
                self.writer.add_scalar(f'Layers/Backward_Block_{idx+1:02d}', float(act), step)
            
            # Log fusion layers
            if 'backward_fuse' in activities:
                self.writer.add_scalar('Layers/Backward_Fuse', float(activities['backward_fuse']), step)
            
            for idx, act in enumerate(forw_acts):
                self.writer.add_scalar(f'Layers/Forward_Block_{idx+1:02d}', float(act), step)
            
            if 'forward_fuse' in activities:
                self.writer.add_scalar('Layers/Forward_Fuse', float(activities['forward_fuse']), step)
            
            if 'fusion' in activities:
                self.writer.add_scalar('Layers/Final_Fusion', float(activities['fusion']), step)
            
            # Log averages
            if back_acts:
                avg_back = sum(back_acts) / len(back_acts)
                self.writer.add_scalar('Activity/Backward_Trunk_Avg', avg_back, step)
            
            if forw_acts:
                avg_forw = sum(forw_acts) / len(forw_acts)
                self.writer.add_scalar('Activity/Forward_Trunk_Avg', avg_forw, step)
    
    def log_lr_phase(self, step, phase):
        """Log LR schedule phase"""
        phase_map = {'warmup': 0, 'cosine': 1, 'plateau_reduced': 2}
        self.writer.add_scalar('LR_Schedule/Phase', phase_map.get(phase, 0), step)
    
    def log_checkpoint(self, step, checkpoint_type):
        """Log checkpoint event"""
        type_map = {'regular': 1, 'best': 2, 'emergency': 3}
        self.writer.add_scalar('Events/Checkpoints', type_map.get(checkpoint_type, 0), step)
    
    def log_images(self, step, lr_img, ki_img, gt_img):
        """Log validation images side by side"""
        import torch
        
        # Concatenate images horizontally [LR | KI | GT]
        combined = torch.cat([lr_img, ki_img, gt_img], dim=2)  # Concatenate along width
        self.writer.add_image('Validation/Comparison', combined, step)
    
    def log_validation_loss(self, step, val_loss):
        """Log validation loss (like in original)"""
        self.writer.add_scalar('Validation/Loss_Total', self._to_float(val_loss), step)
    
    def close(self):
        """Close writer"""
        self.writer.close()
