"""
Checkpoint Manager - Smart checkpoint management with symlinks

Checkpoint Strategy:
1. Regular checkpoints every 10,000 steps (keep all)
2. Best checkpoints between regulars (2k-8k window)
3. Emergency checkpoints on crash

NEW: Zero-padded naming (7 digits) with regex parsing for robustness
"""

import os
import torch
import glob
import re
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
        
        # Regex pattern for extracting step from checkpoint filenames
        # Matches: checkpoint_step_0001234.pth or checkpoint_step_0001234_emergency.pth
        self.step_extractor = re.compile(r'checkpoint_step_(\d+)(?:_.*)?\.pth')
    
    def should_save_regular(self, step):
        """Return True if step % 10000 == 0"""
        return step % 10000 == 0 and step > 0
    
    def should_check_best(self, step):
        """Return True if step is in range for best checking (2k-8k window)"""
        step_in_cycle = step % 10000
        return 2000 <= step_in_cycle <= 8000
    
    def save_checkpoint(self, model, optimizer, scheduler, step, metrics, log_file):
        """
        Save checkpoint file with new zero-padded naming scheme
        
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
        
        # NEW: Use zero-padded naming (7 digits)
        # Emergency checkpoints should use save_emergency_checkpoint() instead
        filename = f"checkpoint_step_{step:07d}.pth"
        
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
        """Save emergency checkpoint with real step number in filename"""
        # NEW: Emergency checkpoints now include actual step number
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"checkpoint_step_{step:07d}_emergency.pth"
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        
        if log_file and os.path.exists(log_file):
            with open(log_file, 'a') as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                       f"⚠️  EMERGENCY CHECKPOINT SAVED (Step {step})\n")
        
        return filepath
    
    def _parse_step_from_filename(self, filename):
        """
        Extract step number from checkpoint filename using regex
        Supports both old and new naming conventions
        
        Args:
            filename: Checkpoint filename
            
        Returns:
            Step number or None if not parseable
        """
        # Try new format with regex: checkpoint_step_0001234.pth or checkpoint_step_0001234_emergency.pth
        match = self.step_extractor.match(filename)
        if match:
            return int(match.group(1))
        
        # Fallback for old format: checkpoint_step_123.pth
        if filename.startswith('checkpoint_step_') and filename.endswith('.pth'):
            try:
                # Extract number between last underscore and .pth
                parts = filename.replace('.pth', '').split('_')
                # Last part should be the number (possibly with suffix like "emergency")
                for part in reversed(parts):
                    if part.isdigit():
                        return int(part)
            except (ValueError, IndexError):
                pass
        
        # Old emergency format
        if filename == 'checkpoint_emergency.pth':
            # Return 0 for old emergency checkpoints that don't have step info in filename
            return 0
        
        return None
    
    def get_latest_checkpoint(self):
        """
        Return path and step of latest checkpoint
        Uses regex parsing to support both old and new formats
        
        Returns:
            Tuple of (path, step) or (None, 0) if no checkpoints
        """
        # Look for all checkpoint files
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*.pth"))
        
        if not checkpoint_files:
            return None, 0
        
        # Find latest by step number (using regex parser)
        max_step = -1
        selected_path = None
        
        for ckpt_path in checkpoint_files:
            filename = os.path.basename(ckpt_path)
            step_num = self._parse_step_from_filename(filename)
            
            if step_num is not None and step_num > max_step:
                max_step = step_num
                selected_path = ckpt_path
        
        if selected_path is None:
            return None, 0
        
        return selected_path, max_step
    
    def list_checkpoints(self):
        """
        Return list of checkpoint info dicts with enhanced metadata
        Uses regex parser for backward compatibility
        
        Returns:
            List of dicts with checkpoint information sorted by step
        """
        checkpoints = []
        
        # Find all checkpoint files
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*.pth"))
        
        for ckpt_path in checkpoint_files:
            try:
                filename = os.path.basename(ckpt_path)
                
                # Parse step from filename
                step_num = self._parse_step_from_filename(filename)
                if step_num is None:
                    continue
                
                # Load checkpoint metadata
                ckpt_data = torch.load(ckpt_path, map_location='cpu')
                
                # Determine checkpoint type from filename and step
                if '_emergency' in filename:
                    ckpt_type = 'emergency'
                elif step_num % 10000 == 0 and step_num > 0:
                    ckpt_type = 'regular'
                else:
                    ckpt_type = 'best'
                
                # Get file modification time
                file_stat = os.stat(ckpt_path)
                modification_time = datetime.fromtimestamp(file_stat.st_mtime)
                
                info_dict = {
                    'path': ckpt_path,
                    'filename': filename,
                    'step': step_num,
                    'type': ckpt_type,
                    'quality': ckpt_data.get('metrics', {}).get('ki_quality', 0),
                    'loss': ckpt_data.get('metrics', {}).get('val_loss', 0),
                    'size_mb': os.path.getsize(ckpt_path) / (1024 * 1024),
                    'timestamp': modification_time,
                    'date_str': modification_time.strftime('%Y-%m-%d %H:%M')
                }
                
                checkpoints.append(info_dict)
            except Exception as e:
                # Skip corrupted checkpoints
                continue
        
        # Sort by step number
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
        """Display enhanced checkpoint table at startup"""
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            print("\n⚠️  No checkpoints found - starting fresh\n")
            return
        
        print("\n" + "="*90)
        print("Available Checkpoints:")
        print("="*90)
        print(f"{'Step':<12} {'Type':<12} {'Quality':<10} {'Loss':<10} {'Date':<18}")
        print("-"*90)
        
        for ckpt in checkpoints[-10:]:  # Show last 10
            step_str = f"{ckpt['step']:,}"
            type_str = ckpt['type']
            quality_str = f"{ckpt['quality']*100:.1f}%"
            loss_str = f"{ckpt['loss']:.4f}"
            date_str = ckpt['date_str']
            
            print(f"{step_str:<12} {type_str:<12} {quality_str:<10} {loss_str:<10} {date_str:<18}")
        
        print("="*90 + "\n")
