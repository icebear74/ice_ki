"""
Runtime Configuration Manager - Live configuration changes without restart

Manages runtime configuration that can be changed during training without restart.

Config File Structure:
- runtime_config.json - Active config (always current)
- runtime_config_step_XXXX.json - Snapshots at checkpoints

Key Features:
- Live reload (check every 10 steps)
- Validation (ranges, sums, types)
- Thread-safe
- Snapshot management
- Config history
"""

import os
import json
import time
import threading
from typing import Dict, Any, Tuple, List, Optional


# Parameter categories with validation ranges
RUNTIME_SAFE_PARAMS = {
    'plateau_safety_threshold': (100, 5000),
    'plateau_patience': (50, 1000),
    'cooldown_duration': (20, 200),
    'max_lr': (1e-5, 1e-3),
    'min_lr': (1e-8, 1e-4),
    'log_tboard_every': (10, 500),
    'val_step_every': (100, 2000),
    'save_step_every': (1000, 50000),
    'initial_grad_clip': (0.1, 10.0),
}

RUNTIME_CAREFUL_PARAMS = {
    'l1_weight_target': (0.1, 0.9),
    'ms_weight_target': (0.05, 0.5),
    'grad_weight_target': (0.05, 0.5),
    'perceptual_weight_target': (0.0, 0.25),
    # Must sum to ~1.0
}

STARTUP_ONLY_PARAMS = {
    'n_feats', 'n_blocks', 'batch_size', 'num_workers', 'accumulation_steps'
}


class RuntimeConfigManager:
    """
    Manages runtime configuration with validation and snapshots
    
    Args:
        config_path: Path to runtime_config.json
        base_config: Base configuration dict (from startup)
    """
    
    def __init__(self, config_path: str, base_config: Dict[str, Any]):
        self.config_path = config_path
        self.base_config = base_config
        self.config = {}
        self.last_modified = 0
        self.lock = threading.Lock()
        
        # Snapshot directory (same as config file)
        self.snapshot_dir = os.path.dirname(config_path)
        
        # Initialize config file if it doesn't exist
        if os.path.exists(config_path):
            self.load()
        else:
            self._initialize_from_base()
            self.save()
    
    def _initialize_from_base(self):
        """Initialize runtime config from base config"""
        with self.lock:
            self.config = {}
            
            # Safe parameters
            for param in RUNTIME_SAFE_PARAMS:
                if param in self.base_config:
                    self.config[param] = self.base_config[param]
            
            # Careful parameters
            for param in RUNTIME_CAREFUL_PARAMS:
                if param in self.base_config:
                    self.config[param] = self.base_config[param]
            
            # Startup-only (read-only in runtime)
            for param in STARTUP_ONLY_PARAMS:
                if param in self.base_config:
                    self.config[param] = self.base_config[param]
    
    def load(self) -> bool:
        """
        Load configuration from file
        
        Returns:
            True if loaded successfully
        """
        try:
            with self.lock:
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                self.last_modified = os.path.getmtime(self.config_path)
                return True
        except Exception as e:
            print(f"⚠️  Error loading runtime config: {e}")
            return False
    
    def save(self) -> bool:
        """
        Save configuration to file
        
        Returns:
            True if saved successfully
        """
        try:
            with self.lock:
                with open(self.config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
                self.last_modified = os.path.getmtime(self.config_path)
                return True
        except Exception as e:
            print(f"⚠️  Error saving runtime config: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        with self.lock:
            return self.config.get(key, default)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values"""
        with self.lock:
            return self.config.copy()
    
    def set(self, key: str, value: Any, validate: bool = True) -> bool:
        """
        Set configuration value
        
        Args:
            key: Configuration key
            value: New value
            validate: Whether to validate the value
            
        Returns:
            True if set successfully
        """
        # Check if parameter is startup-only
        if key in STARTUP_ONLY_PARAMS:
            print(f"⚠️  Cannot change startup-only parameter '{key}' at runtime")
            return False
        
        # Validate if requested
        if validate:
            if not self._validate_parameter(key, value):
                return False
        
        with self.lock:
            old_value = self.config.get(key)
            self.config[key] = value
            
            # Log the change
            if old_value is not None and old_value != value:
                if isinstance(value, float) and value < 0.01:
                    print(f"⚙️  Config Update: {key} {old_value:.2e} → {value:.2e}")
                else:
                    print(f"⚙️  Config Update: {key} {old_value} → {value}")
        
        # Save to file
        return self.save()
    
    def _validate_parameter(self, key: str, value: Any) -> bool:
        """
        Validate parameter value
        
        Args:
            key: Parameter key
            value: Value to validate
            
        Returns:
            True if valid
        """
        # Check safe parameters
        if key in RUNTIME_SAFE_PARAMS:
            min_val, max_val = RUNTIME_SAFE_PARAMS[key]
            if not (min_val <= value <= max_val):
                print(f"⚠️  Invalid value for '{key}': {value} (must be in range [{min_val}, {max_val}])")
                return False
        
        # Check careful parameters
        if key in RUNTIME_CAREFUL_PARAMS:
            min_val, max_val = RUNTIME_CAREFUL_PARAMS[key]
            if not (min_val <= value <= max_val):
                print(f"⚠️  Invalid value for '{key}': {value} (must be in range [{min_val}, {max_val}])")
                return False
            
            # Validate weight sum if it's a weight parameter
            if key.endswith('_weight_target'):
                return self._validate_weight_sum(key, value)
        
        return True
    
    def _validate_weight_sum(self, changed_key: str, new_value: float) -> bool:
        """
        Validate that weight parameters sum to ~1.0
        
        Args:
            changed_key: Key that was changed
            new_value: New value for the key
            
        Returns:
            True if sum is valid
        """
        # Get all weight targets
        weights = {}
        for key in RUNTIME_CAREFUL_PARAMS:
            if key.endswith('_weight_target'):
                if key == changed_key:
                    weights[key] = new_value
                else:
                    weights[key] = self.get(key, 0.0)
        
        total = sum(weights.values())
        
        # Allow 0.95 to 1.05 range
        if not (0.95 <= total <= 1.05):
            print(f"⚠️  Weight sum validation failed: {total:.3f} (should be 0.95-1.05)")
            print(f"   Current weights: {weights}")
            return False
        
        return True
    
    def check_for_updates(self) -> bool:
        """
        Check if config file was modified externally
        
        Returns:
            True if config was updated
        """
        try:
            if not os.path.exists(self.config_path):
                return False
            
            current_mtime = os.path.getmtime(self.config_path)
            
            if current_mtime > self.last_modified:
                print("🔄 Runtime config file changed externally, reloading...")
                self.load()
                return True
        except Exception as e:
            print(f"⚠️  Error checking for config updates: {e}")
        
        return False
    
    def save_snapshot(self, step: int) -> str:
        """
        Save configuration snapshot for a specific step
        
        Args:
            step: Training step number
            
        Returns:
            Path to snapshot file
        """
        snapshot_path = os.path.join(self.snapshot_dir, f"runtime_config_step_{step:07d}.json")
        
        try:
            with self.lock:
                snapshot_data = {
                    'step': step,
                    'timestamp': time.time(),
                    'config': self.config.copy()
                }
                
                with open(snapshot_path, 'w') as f:
                    json.dump(snapshot_data, f, indent=2)
                
                print(f"📸 Config snapshot saved: runtime_config_step_{step:07d}.json")
                return snapshot_path
        except Exception as e:
            print(f"⚠️  Error saving config snapshot: {e}")
            return ""
    
    def load_snapshot(self, step: int) -> bool:
        """
        Load configuration from a snapshot
        
        Args:
            step: Training step number
            
        Returns:
            True if loaded successfully
        """
        snapshot_path = os.path.join(self.snapshot_dir, f"runtime_config_step_{step:07d}.json")
        
        try:
            with open(snapshot_path, 'r') as f:
                snapshot_data = json.load(f)
            
            with self.lock:
                self.config = snapshot_data['config'].copy()
            
            # Save as active config
            self.save()
            
            print(f"📂 Config restored from snapshot: runtime_config_step_{step:07d}.json")
            return True
        except Exception as e:
            print(f"⚠️  Error loading config snapshot: {e}")
            return False
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """
        List all available config snapshots
        
        Returns:
            List of snapshot info dicts
        """
        snapshots = []
        
        try:
            pattern = os.path.join(self.snapshot_dir, "runtime_config_step_*.json")
            import glob
            
            for filepath in sorted(glob.glob(pattern)):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    snapshots.append({
                        'step': data['step'],
                        'timestamp': data.get('timestamp', 0),
                        'filepath': filepath,
                        'filename': os.path.basename(filepath),
                        'preview': {
                            'plateau_safety_threshold': data['config'].get('plateau_safety_threshold'),
                            'max_lr': data['config'].get('max_lr'),
                            'l1_weight_target': data['config'].get('l1_weight_target'),
                        }
                    })
                except:
                    continue
        except Exception as e:
            print(f"⚠️  Error listing config snapshots: {e}")
        
        return snapshots
    
    def compare_snapshots(self, step1: int, step2: int) -> Dict[str, Any]:
        """
        Compare two config snapshots
        
        Args:
            step1: First snapshot step
            step2: Second snapshot step
            
        Returns:
            Dict with comparison results
        """
        snapshot1_path = os.path.join(self.snapshot_dir, f"runtime_config_step_{step1:07d}.json")
        snapshot2_path = os.path.join(self.snapshot_dir, f"runtime_config_step_{step2:07d}.json")
        
        try:
            with open(snapshot1_path, 'r') as f:
                data1 = json.load(f)
            with open(snapshot2_path, 'r') as f:
                data2 = json.load(f)
            
            config1 = data1['config']
            config2 = data2['config']
            
            differences = {}
            all_keys = set(config1.keys()) | set(config2.keys())
            
            for key in all_keys:
                val1 = config1.get(key)
                val2 = config2.get(key)
                
                if val1 != val2:
                    differences[key] = {
                        'step1_value': val1,
                        'step2_value': val2,
                        'changed': True
                    }
            
            return {
                'step1': step1,
                'step2': step2,
                'differences': differences,
                'num_changes': len(differences)
            }
        except Exception as e:
            print(f"⚠️  Error comparing config snapshots: {e}")
            return {
                'step1': step1,
                'step2': step2,
                'error': str(e)
            }
