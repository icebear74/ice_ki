"""
Keyboard Input Handler

Handles terminal raw mode and keyboard input for interactive training control.
Ported from original train.py to maintain feature parity.
"""

import sys
import select
import termios
import tty
from .ui_terminal import *


class KeyboardHandler:
    """
    Manages terminal keyboard input in raw mode
    
    Provides:
    - Terminal setup/teardown
    - Non-blocking keyboard input
    - Live config menu
    """
    
    def __init__(self):
        self.old_settings = None
        self.is_raw_mode = False
    
    def setup_raw_mode(self):
        """Setup terminal in raw mode for character-by-character input"""
        if not self.is_raw_mode:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self.is_raw_mode = True
    
    def restore_normal_mode(self):
        """Restore terminal to normal mode"""
        if self.is_raw_mode and self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            self.is_raw_mode = False
            show_cursor()
    
    def check_key_pressed(self, timeout=0):
        """
        Check if a key has been pressed (non-blocking)
        
        Args:
            timeout: Timeout in seconds (0 = non-blocking)
        
        Returns:
            str or None: Pressed key character, or None if no key pressed
        """
        if sys.stdin in select.select([sys.stdin], [], [], timeout)[0]:
            return sys.stdin.read(1)
        return None
    
    def show_live_menu(self, config, optimizer, trainer=None):
        """
        Show interactive live config menu
        
        Args:
            config: Configuration dict
            optimizer: PyTorch optimizer
            trainer: VSRTrainer instance (optional, for special operations)
        
        Returns:
            dict: Updated configuration
        """
        # Temporarily restore normal terminal mode
        self.restore_normal_mode()
        print(ANSI_SHOW_CURSOR + ANSI_CLEAR + ANSI_HOME)
        
        while True:
            print(f"{C_BOLD}🛠️  LIVE CONFIG{C_RESET}\n" + "-"*45)
            
            # Get configurable keys
            keys = [
                'LR_EXPONENT', 'WEIGHT_DECAY', 'VAL_STEP_EVERY', 
                'SAVE_STEP_EVERY', 'LOG_TBOARD_EVERY', 'HIST_STEP_EVERY',
                'ACCUMULATION_STEPS', 'DISPLAY_MODE', 'GRAD_CLIP',
                # Runtime config parameters - SAFE
                'plateau_patience', 'plateau_safety_threshold', 'cooldown_duration',
                'max_lr', 'min_lr', 'initial_grad_clip',
                'log_tboard_every', 'val_step_every', 'save_step_every',
                # Runtime config parameters - CAREFUL (loss weights)
                'l1_weight_target', 'ms_weight_target', 'grad_weight_target', 'perceptual_weight_target'
            ]
            
            # Display current values
            for idx, k in enumerate(keys):
                if k in config:
                    val = config[k]
                    if k == "DISPLAY_MODE":
                        val_display = DISPLAY_MODE_NAMES[val] if val < len(DISPLAY_MODE_NAMES) else val
                    else:
                        val_display = val
                    
                    # Add note for loss weight parameters
                    if k in ['l1_weight_target', 'ms_weight_target', 'grad_weight_target', 'perceptual_weight_target']:
                        print(f" {idx+1}. {k:<20}: {val_display} {C_YELLOW}(must sum ~1.0){C_RESET}")
                    else:
                        print(f" {idx+1}. {k:<20}: {val_display}")
            
            print("-" * 45 + "\n 0. ZURÜCK")
            
            wahl = input("\n Auswahl: ").lower()
            
            if wahl == "0":
                break
            
            try:
                idx = int(wahl) - 1
                if 0 <= idx < len(keys):
                    k_name = keys[idx]
                    if k_name not in config:
                        print(f"⚠️  {k_name} not in config")
                        continue
                    
                    new_val = input(f"Neuer Wert für {k_name}: ")
                    
                    # Handle special cases
                    if k_name == "LR_EXPONENT":
                        val = int(new_val)
                        config[k_name] = val
                        # Update learning rate
                        new_lr = 10**val
                        for pg in optimizer.param_groups:
                            pg['lr'] = new_lr
                        print(f"✅ LR updated to {new_lr:.2e}")
                    
                    elif k_name == "GRAD_CLIP":
                        config[k_name] = float(new_val)
                    
                    elif k_name == "DISPLAY_MODE":
                        config[k_name] = int(new_val) % 4
                    
                    elif k_name == "WEIGHT_DECAY":
                        config[k_name] = float(new_val)
                    
                    # Runtime config parameters - safe interval parameters
                    elif k_name in ['plateau_patience', 'plateau_safety_threshold', 'cooldown_duration',
                                   'log_tboard_every', 'val_step_every', 'save_step_every']:
                        config[k_name] = int(new_val)
                        # Update runtime config if trainer has it
                        if trainer and hasattr(trainer, 'runtime_config') and trainer.runtime_config:
                            trainer.runtime_config.set(k_name, int(new_val))
                            print(f"✅ Runtime config updated via trainer")
                    
                    elif k_name in ['max_lr', 'min_lr', 'initial_grad_clip']:
                        config[k_name] = float(new_val)
                        # Update runtime config if trainer has it
                        if trainer and hasattr(trainer, 'runtime_config') and trainer.runtime_config:
                            trainer.runtime_config.set(k_name, float(new_val))
                            print(f"✅ Runtime config updated via trainer")
                    
                    # Runtime config parameters - careful (loss weights)
                    elif k_name in ['l1_weight_target', 'ms_weight_target', 'grad_weight_target', 'perceptual_weight_target']:
                        config[k_name] = float(new_val)
                        # Update runtime config if trainer has it
                        if trainer and hasattr(trainer, 'runtime_config') and trainer.runtime_config:
                            trainer.runtime_config.set(k_name, float(new_val))
                            print(f"✅ Runtime config updated via trainer")
                            print(f"⚠️  Note: Loss weights must sum to ~1.0 for proper training")
                    
                    else:
                        # Try to preserve type
                        old_val = config[k_name]
                        config[k_name] = type(old_val)(new_val)
                    
                    print(f"✅ {k_name} = {config[k_name]}")
                    
                    # Save to config file if trainer available
                    if trainer and hasattr(trainer, 'config'):
                        trainer.config = config
            
            except ValueError as e:
                print(f"❌ Invalid value: {e}")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Return to raw mode
        self.setup_raw_mode()
        print(ANSI_CLEAR)
        
        return config
    
    def __enter__(self):
        """Context manager entry"""
        self.setup_raw_mode()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.restore_normal_mode()
        return False
