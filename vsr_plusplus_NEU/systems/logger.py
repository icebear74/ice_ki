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
        """Log loss components with scaling"""
        self.writer.add_scalar('Loss/L1', self._to_float(losses.get('l1', 0)), step)
        self.writer.add_scalar('Loss/MS', self._to_float(losses.get('ms', 0)), step)
        self.writer.add_scalar('Loss/Grad', self._to_float(losses.get('grad', 0)), step)
        self.writer.add_scalar('Loss/Perceptual', self._to_float(losses.get('perceptual', 0)), step)
        self.writer.add_scalar('Loss/Total', self._to_float(losses.get('total', 0)), step)
        
        # Add total loss to CoreMetrics
        total = self._to_float(losses.get('total', 0))
        self.writer.add_scalar('Training/CoreMetrics/Total_Loss', total * 100, step)
        
        # Add EMA L1 loss if available
        if 'ema_l1_loss' in losses:
            self.writer.add_scalar('Training/CoreMetrics/EMA_L1_Loss', self._to_float(losses['ema_l1_loss']) * 1000, step)
    
    def log_lr(self, step, lr):
        """Log learning rate with scaling for visualization"""
        self.writer.add_scalar('Training/LearningRate', self._to_float(lr), step)
        
        # Add to CoreMetrics dashboard (scaled for visibility)
        self.writer.add_scalar('Training/CoreMetrics/Learning_Rate', self._to_float(lr) * 1e6, step)
    
    def log_adaptive(self, step, adaptive_info):
        """Log adaptive system info with correlation dashboards"""
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
        
        # Add Perceptual Weight tracking
        perc_w = adaptive_info.get('perceptual_weight', 0.0)
        self.writer.add_scalar('Adaptive/Perceptual_Weight', self._to_float(perc_w), step)
        
        # DASHBOARD 1: System Health + Performance
        self.writer.add_scalars('Adaptive/SystemHealth', {
            'L1_Weight': self._to_float(l1_w) * 100,
            'MS_Weight': self._to_float(ms_w) * 100,
            'Grad_Weight': self._to_float(grad_w) * 100,
            'Perceptual_Weight': self._to_float(perc_w) * 100,
            'KI_Improvement': self._to_float(adaptive_info.get('ki_improvement', 0)) * 100,  # 0-1 -> 0-100%
        }, step)
        
        # DASHBOARD 2: Interventions + Impact
        self.writer.add_scalars('Adaptive/Interventions', {
            'Plateau_Counter': self._to_float(adaptive_info.get('plateau_counter', 0)) / 10,  # Scale for visibility
            'Cooldown_Active': 50 if adaptive_info.get('is_cooldown') else 0,
            'Aggressive_Mode': 50 if adaptive_info.get('aggressive_mode') else 0,
            'LR_Boost_Available': 50 if adaptive_info.get('lr_boost_available', False) else 0,
            'KI_Improvement': self._to_float(adaptive_info.get('ki_improvement', 0)) * 100,
        }, step)
    
    def log_quality(self, step, metrics):
        """Log quality metrics with correlation to LR/Loss"""
        if not metrics:
            return
        
        self.writer.add_scalar('Quality/LR_Quality', self._to_float(metrics.get('lr_quality', 0)), step)
        self.writer.add_scalar('Quality/KI_Quality', self._to_float(metrics.get('ki_quality', 0)), step)
        self.writer.add_scalar('Quality/Improvement', self._to_float(metrics.get('improvement', 0)), step)
        
        # Add to CoreMetrics dashboard
        self.writer.add_scalars('Training/CoreMetrics', {
            'KI_Quality': self._to_float(metrics.get('ki_quality', 0)) * 100,       # 0-1 -> 0-100%
            'KI_Improvement': self._to_float(metrics.get('improvement', 0)) * 100,   # 0-1 -> 0-100%
        }, step)
        
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
    
    def log_event(self, step, event_type, message):
        """Log important events to timeline"""
        self.writer.add_text('Events/Timeline', f'{event_type}: {message}', step)
        
        # Add event marker (spike for visibility)
        if event_type in ['LR_Boost', 'Aggressive_Mode', 'Cooldown_Start']:
            self.writer.add_scalar(f'Events/{event_type}', 100, step)
    
    def log_config_change(self, step, param_name, old_value, new_value):
        """
        Log runtime configuration changes
        
        Args:
            step: Current training step
            param_name: Parameter that changed
            old_value: Previous value
            new_value: New value
        """
        # Log as text event
        self.writer.add_text(
            'Config/Changes', 
            f'**{param_name}**: {old_value} → {new_value}',
            step
        )
        
        # Log the new value as scalar for tracking
        self.writer.add_scalar(f'Config/Parameters/{param_name}', self._to_float(new_value), step)
        
        # Add event marker
        self.writer.add_scalar('Events/Config_Change', 100, step)
    
    def log_config_snapshot(self, config):
        """
        Log initial training configuration as text
        
        Args:
            config: Configuration dictionary
        """
        # Format configuration as markdown table
        config_text = "# Training Configuration\n\n"
        config_text += "## Model Architecture\n"
        config_text += f"- **Features (n_feats)**: {config.get('n_feats', 'N/A')}\n"
        config_text += f"- **Blocks (n_blocks)**: {config.get('n_blocks', 'N/A')}\n"
        config_text += f"- **Batch Size**: {config.get('batch_size', 'N/A')}\n"
        config_text += f"- **Accumulation Steps**: {config.get('ACCUMULATION_STEPS', 'N/A')}\n\n"
        
        config_text += "## Training Parameters\n"
        config_text += f"- **Max Steps**: {config.get('MAX_STEPS', 'N/A')}\n"
        config_text += f"- **Max LR**: {config.get('max_lr', 'N/A')}\n"
        config_text += f"- **Min LR**: {config.get('min_lr', 'N/A')}\n"
        config_text += f"- **Warmup Steps**: {config.get('WARMUP_STEPS', 'N/A')}\n"
        config_text += f"- **Validation Every**: {config.get('VAL_STEP_EVERY', 'N/A')} steps\n"
        config_text += f"- **Save Every**: {config.get('SAVE_STEP_EVERY', 'N/A')} steps\n\n"
        
        config_text += "## Adaptive System\n"
        config_text += f"- **Plateau Patience**: {config.get('plateau_patience', 'N/A')}\n"
        config_text += f"- **Plateau Safety Threshold**: {config.get('plateau_safety_threshold', 'N/A')}\n"
        config_text += f"- **Cooldown Duration**: {config.get('cooldown_duration', 'N/A')}\n"
        config_text += f"- **Initial Grad Clip**: {config.get('initial_grad_clip', 'N/A')}\n\n"
        
        config_text += "## Loss Weights (Target)\n"
        config_text += f"- **L1 Weight**: {config.get('l1_weight_target', 'N/A')}\n"
        config_text += f"- **MS Weight**: {config.get('ms_weight_target', 'N/A')}\n"
        config_text += f"- **Gradient Weight**: {config.get('grad_weight_target', 'N/A')}\n"
        config_text += f"- **Perceptual Weight**: {config.get('perceptual_weight_target', 'N/A')}\n\n"
        
        config_text += "## Data\n"
        config_text += f"- **Data Root**: {config.get('DATA_ROOT', 'N/A')}\n"
        config_text += f"- **Num Workers**: {config.get('num_workers', 'N/A')}\n"
        
        self.writer.add_text('Config/Initial_Configuration', config_text, 0)
        
        # Also log as scalars for initial values
        if 'plateau_patience' in config:
            self.writer.add_scalar('Config/Parameters/plateau_patience', config['plateau_patience'], 0)
        if 'plateau_safety_threshold' in config:
            self.writer.add_scalar('Config/Parameters/plateau_safety_threshold', config['plateau_safety_threshold'], 0)
        if 'max_lr' in config:
            self.writer.add_scalar('Config/Parameters/max_lr', config['max_lr'], 0)
        if 'min_lr' in config:
            self.writer.add_scalar('Config/Parameters/min_lr', config['min_lr'], 0)
    
    def log_plateau_state(self, step, plateau_info):
        """
        Log detailed plateau detection state
        
        Args:
            step: Current training step
            plateau_info: Dictionary with plateau detection details
        """
        if not plateau_info:
            return
        
        # Log plateau counter and threshold
        counter = plateau_info.get('plateau_counter', 0)
        patience = plateau_info.get('plateau_patience', 0)
        
        self.writer.add_scalar('Plateau/Counter', self._to_float(counter), step)
        self.writer.add_scalar('Plateau/Patience', self._to_float(patience), step)
        self.writer.add_scalar('Plateau/Progress_Percent', 
                              (self._to_float(counter) / max(1, self._to_float(patience))) * 100, step)
        
        # Log best loss and EMA if available
        if 'best_loss' in plateau_info:
            self.writer.add_scalar('Plateau/Best_Loss', self._to_float(plateau_info['best_loss']), step)
        if 'ema_loss' in plateau_info:
            self.writer.add_scalar('Plateau/EMA_Loss', self._to_float(plateau_info['ema_loss']), step)
        
        # Log quality tracking
        if 'best_quality' in plateau_info:
            self.writer.add_scalar('Plateau/Best_Quality', self._to_float(plateau_info['best_quality']), step)
        if 'ema_quality' in plateau_info:
            self.writer.add_scalar('Plateau/EMA_Quality', self._to_float(plateau_info['ema_quality']), step)
        
        # Log plateau state
        is_plateau = plateau_info.get('is_plateau', False)
        self.writer.add_scalar('Plateau/Is_Plateau', 1 if is_plateau else 0, step)
        
        # Steps until potential reset
        if 'steps_until_reset' in plateau_info:
            self.writer.add_scalar('Plateau/Steps_Until_Reset', 
                                  self._to_float(plateau_info['steps_until_reset']), step)
    
    def log_weight_statistics(self, step, weights):
        """
        Log loss weight statistics and distributions
        
        Args:
            step: Current training step
            weights: Dictionary with weight values
        """
        import torch
        
        l1_w = self._to_float(weights.get('l1', 0))
        ms_w = self._to_float(weights.get('ms', 0))
        grad_w = self._to_float(weights.get('grad', 0))
        perc_w = self._to_float(weights.get('perceptual', 0))
        
        total = l1_w + ms_w + grad_w + perc_w
        
        # Log weight distribution as percentages
        if total > 0:
            self.writer.add_scalars('Weights/Distribution', {
                'L1_Percent': (l1_w / total) * 100,
                'MS_Percent': (ms_w / total) * 100,
                'Grad_Percent': (grad_w / total) * 100,
                'Perceptual_Percent': (perc_w / total) * 100,
            }, step)
        
        # Log sum for validation
        self.writer.add_scalar('Weights/Sum', total, step)
        
        # Create histogram of weight distribution
        weight_tensor = torch.tensor([l1_w, ms_w, grad_w, perc_w])
        self.writer.add_histogram('Weights/Distribution_Histogram', weight_tensor, step)
    
    def log_validation_event(self, step, metrics):
        """
        Log validation event with comprehensive metrics
        
        Args:
            step: Current training step
            metrics: Validation metrics dictionary
        """
        # Log as text event
        ki_quality = self._to_float(metrics.get('ki_quality', 0))
        improvement = self._to_float(metrics.get('improvement', 0))
        
        event_text = f"**Validation Run**\n"
        event_text += f"- KI Quality: {ki_quality*100:.2f}%\n"
        event_text += f"- Improvement: {improvement*100:.2f}%\n"
        event_text += f"- PSNR: {self._to_float(metrics.get('ki_psnr', 0)):.2f}dB\n"
        event_text += f"- SSIM: {self._to_float(metrics.get('ki_ssim', 0)):.4f}\n"
        
        self.writer.add_text('Events/Validation', event_text, step)
        
        # Add event marker
        self.writer.add_scalar('Events/Validation_Run', 50, step)
    
    def log_training_phase(self, step, phase_info):
        """
        Log training phase information (aggressive, stable, cooldown)
        
        Args:
            step: Current training step
            phase_info: Dictionary with phase information
        """
        # Encode phases as numbers
        phase_map = {
            'stable': 0,
            'aggressive': 1,
            'cooldown': 2,
            'plateau_reducing': 3
        }
        
        current_phase = phase_info.get('phase', 'stable')
        self.writer.add_scalar('Training/Phase', phase_map.get(current_phase, 0), step)
        
        # Log phase transitions as text
        if phase_info.get('phase_changed', False):
            self.writer.add_text(
                'Events/Phase_Transitions',
                f"Phase changed to: **{current_phase}**",
                step
            )
            self.writer.add_scalar('Events/Phase_Change', 75, step)
    
    def log_hyperparameters(self, hparam_dict, metric_dict):
        """
        Log hyperparameters for comparison in TensorBoard
        
        Args:
            hparam_dict: Dictionary of hyperparameters
            metric_dict: Dictionary of metrics to track
        """
        self.writer.add_hparams(hparam_dict, metric_dict)
    
    def close(self):
        """Close writer"""
        self.writer.close()
