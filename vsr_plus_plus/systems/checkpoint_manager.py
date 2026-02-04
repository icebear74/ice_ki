"""
Checkpoint Manager - Smart checkpoint management with symlinks

Checkpoint Strategy:
1. Regular checkpoints every 10,000 steps (keep all)
2. Best checkpoints between regulars (2k-8k window)
3. Emergency checkpoints on crash
"""

import os
import torch
import glob
from datetime import datetime


class CheckpointManager:
    """
    Manages checkpoint saving and loading with symlinks
    
    Args:
        checkpoint_dir: Directory to save checkpoints
    """
    
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.best_quality = -1.0
        self.best_checkpoint_path = None
    
    def should_save_regular(self, step):
        """Return True if step % 10000 == 0"""
        return step % 10000 == 0 and step > 0
    
    def should_check_best(self, step):
        """Return True if step is in range for best checking (2k-8k window)"""
        step_in_cycle = step % 10000
        return 2000 <= step_in_cycle <= 8000
    
    def save_checkpoint(self, model, optimizer, scheduler, step, metrics, log_file):
        """
        Save checkpoint file
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: LR scheduler state  
            step: Current step
            metrics: Validation metrics dict
            log_file: Path to log file
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Determine filename
        if step > 0:
            filename = f"checkpoint_step_{step}.pth"
        else:
            filename = "checkpoint_emergency.pth"
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        
        # Log to file
        if log_file and os.path.exists(log_file):
            with open(log_file, 'a') as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                       f"💾 Checkpoint saved: {filename}\n")
        
        return filepath
    
    def update_best_checkpoint(self, model, optimizer, scheduler, step, quality, metrics, log_file):
        """
        Check if best, save if needed, update symlinks
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: LR scheduler state
            step: Current step
            quality: Current quality score (0-1)
            metrics: Validation metrics dict
            log_file: Path to log file
            
        Returns:
            True if new best, False otherwise
        """
        # Check if new best
        if quality <= self.best_quality:
            return False
        
        # Save new best checkpoint
        checkpoint_path = self.save_checkpoint(model, optimizer, scheduler, step, metrics, log_file)
        
        # Update best symlinks
        best_link = os.path.join(self.checkpoint_dir, "checkpoint_best.pth")
        best_old_link = os.path.join(self.checkpoint_dir, "checkpoint_best_old.pth")
        
        # Move current best to old (if exists)
        if os.path.islink(best_link):
            # Remove old "best_old" if exists
            if os.path.islink(best_old_link):
                os.unlink(best_old_link)
            
            # Rename current best to best_old
            os.rename(best_link, best_old_link)
        
        # Create new best symlink (relative path for portability)
        checkpoint_filename = os.path.basename(checkpoint_path)
        os.symlink(checkpoint_filename, best_link)
        
        # Update tracking
        self.best_quality = quality
        self.best_checkpoint_path = checkpoint_path
        
        # Log to file
        if log_file and os.path.exists(log_file):
            with open(log_file, 'a') as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                       f"🏆 NEW BEST CHECKPOINT! (Step {step}, Quality {quality*100:.1f}%)\n")
        
        return True
    
    def save_emergency_checkpoint(self, model, optimizer, scheduler, step, metrics, log_file):
        """Save emergency checkpoint on crash/interrupt"""
        checkpoint_path = self.save_checkpoint(model, optimizer, scheduler, 0, metrics, log_file)
        
        if log_file and os.path.exists(log_file):
            with open(log_file, 'a') as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                       f"⚠️  EMERGENCY CHECKPOINT SAVED (Step {step})\n")
        
        return checkpoint_path
    
    def get_latest_checkpoint(self):
        """
        Return path and step of latest checkpoint
        
        Returns:
            Tuple of (path, step) or (None, 0) if no checkpoints
        """
        # Look for regular checkpoints
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_step_*.pth"))
        
        if not checkpoint_files:
            # Check for emergency checkpoint
            emergency = os.path.join(self.checkpoint_dir, "checkpoint_emergency.pth")
            if os.path.exists(emergency):
                return emergency, 0
            return None, 0
        
        # Find latest by step number
        latest_step = 0
        latest_path = None
        
        for path in checkpoint_files:
            try:
                # Extract step from filename
                filename = os.path.basename(path)
                step = int(filename.split('_')[-1].replace('.pth', ''))
                
                if step > latest_step:
                    latest_step = step
                    latest_path = path
            except:
                continue
        
        return latest_path, latest_step
    
    def list_checkpoints(self):
        """
        Return list of checkpoint info dicts
        
        Returns:
            List of dicts with checkpoint information
        """
        checkpoints = []
        
        # Regular checkpoints
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_step_*.pth"))
        
        for path in checkpoint_files:
            try:
                # Load checkpoint metadata
                ckpt = torch.load(path, map_location='cpu')
                
                info = {
                    'path': path,
                    'step': ckpt.get('step', 0),
                    'type': 'regular' if ckpt.get('step', 0) % 10000 == 0 else 'best',
                    'quality': ckpt.get('metrics', {}).get('ki_quality', 0),
                    'loss': ckpt.get('metrics', {}).get('val_loss', 0),
                    'size_mb': os.path.getsize(path) / (1024 * 1024)
                }
                
                checkpoints.append(info)
            except:
                continue
        
        # Sort by step
        checkpoints.sort(key=lambda x: x['step'])
        
        return checkpoints
    
    def cleanup_old_checkpoints(self):
        """Remove old checkpoint formats and broken symlinks"""
        # Remove broken symlinks
        for link_name in ['checkpoint_best.pth', 'checkpoint_best_old.pth']:
            link_path = os.path.join(self.checkpoint_dir, link_name)
            if os.path.islink(link_path):
                target = os.readlink(link_path)
                target_path = os.path.join(self.checkpoint_dir, target)
                if not os.path.exists(target_path):
                    os.unlink(link_path)
    
    def cleanup_all_for_fresh_start(self, log_dir):
        """
        Clean up everything for fresh start (when user chooses 'L')
        - All checkpoints
        - All logs
        - All TensorBoard events
        """
        import shutil
        
        # Remove all checkpoint files
        for f in os.listdir(self.checkpoint_dir):
            path = os.path.join(self.checkpoint_dir, f)
            try:
                if os.path.isfile(path) or os.path.islink(path):
                    os.unlink(path)
            except Exception as e:
                print(f"Warning: Could not remove {path}: {e}")
        
        # Remove TensorBoard events in active_run
        active_run_dir = os.path.join(log_dir, "active_run")
        if os.path.exists(active_run_dir):
            try:
                shutil.rmtree(active_run_dir)
                os.makedirs(active_run_dir, exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not clean TensorBoard logs: {e}")
        
        # Remove log files
        log_files = ['training.log', 'training_status.txt']
        for log_file in log_files:
            log_path = os.path.join(self.checkpoint_dir, log_file)
            if os.path.exists(log_path):
                try:
                    os.unlink(log_path)
                except Exception as e:
                    print(f"Warning: Could not remove {log_file}: {e}")
    
    def show_checkpoint_info(self):
        """Display checkpoint table at startup"""
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            print("\n⚠️  No checkpoints found - starting fresh\n")
            return
        
        print("\n" + "="*80)
        print("Available Checkpoints:")
        print("="*80)
        print(f"{'Step':<12} {'Type':<15} {'Quality':<12} {'Loss':<12}")
        print("-"*80)
        
        for ckpt in checkpoints[-10:]:  # Show last 10
            step = f"{ckpt['step']:,}"
            ckpt_type = ckpt['type']
            quality = f"{ckpt['quality']*100:.1f}%"
            loss = f"{ckpt['loss']:.4f}"
            
            print(f"{step:<12} {ckpt_type:<15} {quality:<12} {loss:<12}")
        
        print("="*80 + "\n")
