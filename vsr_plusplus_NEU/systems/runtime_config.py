"""
Enhanced Runtime Configuration Manager - 7-Frame VSR Training System

NEW Features:
- Model configuration (n_frames, n_feats, n_blocks, precision)
- Training configuration with effective_batch_size
- Adaptive batch configs per size
- Size distribution configuration
- Enhanced validation (sum checks, VRAM limits)

Legacy Features Preserved:
- Live reload (check every 10 steps)
- Thread-safe
- Snapshot management
- Config history
"""

import os
import json
import time
import threading
from typing import Dict, Any, Tuple, List, Optional


# Import adaptive batch calculator for VRAM validation
try:
    from .adaptive_batch import AdaptiveBatchCalculator, VRAM_LIMIT_GB
except ImportError:
    # Fallback for standalone usage
    VRAM_LIMIT_GB = 6.5
    AdaptiveBatchCalculator = None


# Legacy parameter categories
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
}

STARTUP_ONLY_PARAMS = {
    'n_feats', 'n_blocks', 'batch_size', 'num_workers', 'accumulation_steps'
}


# Default configuration structure
DEFAULT_CONFIG = {
    "model": {
        "n_frames": 7,
        "n_feats": 72,
        "n_blocks": 26,
        "precision": "float32"
    },
    "training": {
        "effective_batch_size": 6,
        "adaptive_batch": {
            "small_540": {"batch": 1, "accum": 6},
            "medium_169": {"batch": 1, "accum": 6},
            "large_720": {"batch": 1, "accum": 6}
        }
    },
    "size_distribution": {
        "small_540": 0.65,
        "medium_169": 0.35,
        "large_720": 0.00
    }
}


class EnhancedRuntimeConfigManager:
    """
    Enhanced Runtime Configuration Manager for 7-Frame VSR Training
    
    Args:
        config_path: Path to runtime_config.json
        base_config: Base configuration dict (optional, for legacy support)
        use_new_structure: Use new 7-frame structure (default: True)
    """
    
    def __init__(
        self, 
        config_path: str, 
        base_config: Optional[Dict[str, Any]] = None,
        use_new_structure: bool = True
    ):
        self.config_path = config_path
        self.base_config = base_config or {}
        self.use_new_structure = use_new_structure
        self.config = {}
        self.last_modified = 0
        self.lock = threading.Lock()
        
        # Snapshot directory
        self.snapshot_dir = os.path.dirname(config_path)
        
        # Initialize adaptive batch calculator
        if AdaptiveBatchCalculator:
            self.batch_calculator = AdaptiveBatchCalculator()
        else:
            self.batch_calculator = None
        
        # Initialize config file
        if os.path.exists(config_path):
            self.load()
            # Migrate if needed
            if use_new_structure and not self._is_new_structure():
                self._migrate_to_new_structure()
        else:
            if use_new_structure:
                self._initialize_new_structure()
            else:
                self._initialize_from_base()
            self.save()
    
    def _is_new_structure(self) -> bool:
        """Check if config uses new structure"""
        return 'model' in self.config and 'training' in self.config
    
    def _initialize_new_structure(self):
        """Initialize with new 7-frame structure"""
        with self.lock:
            self.config = DEFAULT_CONFIG.copy()
            
            # Merge with base config if provided
            if self.base_config:
                # Model params
                if 'n_frames' in self.base_config:
                    self.config['model']['n_frames'] = self.base_config['n_frames']
                if 'n_feats' in self.base_config:
                    self.config['model']['n_feats'] = self.base_config['n_feats']
                if 'n_blocks' in self.base_config:
                    self.config['model']['n_blocks'] = self.base_config['n_blocks']
    
    def _migrate_to_new_structure(self):
        """Migrate legacy config to new structure"""
        with self.lock:
            old_config = self.config.copy()
            self.config = DEFAULT_CONFIG.copy()
            
            # Preserve any legacy settings
            for key, value in old_config.items():
                if key not in self.config:
                    self.config[key] = value
    
    def _initialize_from_base(self):
        """Initialize from base config (legacy mode)"""
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
            
            # Startup-only
            for param in STARTUP_ONLY_PARAMS:
                if param in self.base_config:
                    self.config[param] = self.base_config[param]
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate entire configuration
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        with self.lock:
            config = self.config.copy()
        
        # Validate new structure
        if self.use_new_structure:
            # Check size distribution sum
            if 'size_distribution' in config:
                total = sum(config['size_distribution'].values())
                if not (0.99 <= total <= 1.01):  # Allow ±0.01 tolerance
                    errors.append(
                        f"Size distribution sum is {total:.4f}, must be 1.0 (±0.01)"
                    )
            
            # Check VRAM limits for adaptive batch configs
            if 'training' in config and 'adaptive_batch' in config['training']:
                if self.batch_calculator:
                    for size, batch_config in config['training']['adaptive_batch'].items():
                        # Estimate VRAM
                        try:
                            calc_config = self.batch_calculator.calculate_batch_config(
                                size, 
                                batch_config.get('accum', 1)
                            )
                            
                            if calc_config['vram_est'] >= VRAM_LIMIT_GB:
                                errors.append(
                                    f"VRAM limit exceeded for {size}: "
                                    f"{calc_config['vram_est']:.2f} GB >= {VRAM_LIMIT_GB} GB"
                                )
                        except Exception as e:
                            errors.append(f"Error validating {size}: {e}")
            
            # Validate model config
            if 'model' in config:
                model = config['model']
                
                if model.get('n_frames', 0) not in [5, 7]:
                    errors.append(f"n_frames must be 5 or 7, got {model.get('n_frames')}")
                
                if model.get('n_feats', 0) < 32 or model.get('n_feats', 0) > 128:
                    errors.append(f"n_feats should be 32-128, got {model.get('n_feats')}")
                
                if model.get('n_blocks', 0) < 8 or model.get('n_blocks', 0) > 50:
                    errors.append(f"n_blocks should be 8-50, got {model.get('n_blocks')}")
        
        # Legacy validation
        else:
            # Validate weight sums
            weight_keys = [k for k in config.keys() if k.endswith('_weight_target')]
            if weight_keys:
                total = sum(config.get(k, 0.0) for k in weight_keys)
                if total > 0 and not (0.95 <= total <= 1.05):
                    errors.append(
                        f"Weight sum validation failed: {total:.3f} (should be 0.95-1.05)"
                    )
        
        return len(errors) == 0, errors
    
    def load(self) -> bool:
        """Load configuration from file"""
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
        """Save configuration to file"""
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
        """Get configuration value (supports nested keys like 'model.n_frames')"""
        with self.lock:
            # Handle nested keys
            if '.' in key:
                parts = key.split('.')
                value = self.config
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        return default
                return value
            else:
                return self.config.get(key, default)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values"""
        with self.lock:
            return self.config.copy()
    
    def set(self, key: str, value: Any, validate: bool = True) -> bool:
        """
        Set configuration value (supports nested keys)
        
        Args:
            key: Configuration key (use '.' for nested, e.g., 'model.n_frames')
            value: New value
            validate: Whether to validate after setting
        """
        with self.lock:
            # Handle nested keys
            if '.' in key:
                parts = key.split('.')
                config_ref = self.config
                
                # Navigate to parent
                for part in parts[:-1]:
                    if part not in config_ref:
                        config_ref[part] = {}
                    config_ref = config_ref[part]
                
                # Set value
                old_value = config_ref.get(parts[-1])
                config_ref[parts[-1]] = value
                
                # Log change
                if old_value != value:
                    print(f"⚙️  Config Update: {key} {old_value} → {value}")
            else:
                # Check if startup-only
                if key in STARTUP_ONLY_PARAMS:
                    print(f"⚠️  Cannot change startup-only parameter '{key}' at runtime")
                    return False
                
                old_value = self.config.get(key)
                self.config[key] = value
                
                if old_value != value:
                    print(f"⚙️  Config Update: {key} {old_value} → {value}")
        
        # Validate if requested
        if validate:
            is_valid, errors = self.validate()
            if not is_valid:
                print("⚠️  Validation errors after config change:")
                for error in errors:
                    print(f"    - {error}")
                return False
        
        # Save to file
        return self.save()
    
    def update_size_distribution(self, distribution: Dict[str, float]) -> bool:
        """
        Update size distribution configuration
        
        Args:
            distribution: Dict mapping size category to percentage (0.0-1.0)
            
        Returns:
            True if updated successfully
        """
        # Validate sum
        total = sum(distribution.values())
        if not (0.99 <= total <= 1.01):
            print(f"⚠️  Size distribution sum is {total:.4f}, must be 1.0 (±0.01)")
            return False
        
        with self.lock:
            if 'size_distribution' not in self.config:
                self.config['size_distribution'] = {}
            
            self.config['size_distribution'].update(distribution)
        
        return self.save()
    
    def update_effective_batch_size(self, effective_batch: int) -> bool:
        """
        Update effective batch size and recalculate adaptive configs
        
        Args:
            effective_batch: New effective batch size
            
        Returns:
            True if updated successfully
        """
        if effective_batch < 1:
            print(f"⚠️  Effective batch size must be >= 1, got {effective_batch}")
            return False
        
        with self.lock:
            if 'training' not in self.config:
                self.config['training'] = {}
            
            self.config['training']['effective_batch_size'] = effective_batch
            
            # Recalculate adaptive batch configs
            if self.batch_calculator:
                if 'adaptive_batch' not in self.config['training']:
                    self.config['training']['adaptive_batch'] = {}
                
                for size in ['small_540', 'medium_169', 'large_720']:
                    batch_config = self.batch_calculator.calculate_batch_config(
                        size, effective_batch
                    )
                    
                    self.config['training']['adaptive_batch'][size] = {
                        'batch': batch_config['batch'],
                        'accum': batch_config['accum']
                    }
        
        return self.save()
    
    def check_for_updates(self) -> bool:
        """Check if config file was modified externally"""
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
        """Save configuration snapshot for a specific step"""
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
                
                return snapshot_path
        except Exception as e:
            print(f"⚠️  Error saving config snapshot: {e}")
            return ""
    
    def load_snapshot(self, step: int) -> bool:
        """Load configuration from a snapshot"""
        snapshot_path = os.path.join(self.snapshot_dir, f"runtime_config_step_{step:07d}.json")
        
        try:
            with open(snapshot_path, 'r') as f:
                snapshot_data = json.load(f)
            
            with self.lock:
                self.config = snapshot_data['config'].copy()
            
            self.save()
            print(f"📂 Config restored from snapshot: runtime_config_step_{step:07d}.json")
            return True
        except Exception as e:
            print(f"⚠️  Error loading config snapshot: {e}")
            return False


# Maintain backward compatibility
RuntimeConfigManager = EnhancedRuntimeConfigManager


if __name__ == "__main__":
    # Demo usage
    print("Enhanced Runtime Config Manager Demo\n")
    
    # Create config manager with new structure
    config_path = "/tmp/runtime_config_demo.json"
    manager = EnhancedRuntimeConfigManager(config_path, use_new_structure=True)
    
    # Print config
    print("Initial config:")
    print(json.dumps(manager.get_all(), indent=2))
    
    # Validate
    print("\nValidating config...")
    is_valid, errors = manager.validate()
    if is_valid:
        print("✅ Configuration is valid!")
    else:
        print("❌ Validation errors:")
        for error in errors:
            print(f"  - {error}")
    
    # Update effective batch size
    print("\nUpdating effective batch size to 8...")
    manager.update_effective_batch_size(8)
    
    # Update size distribution
    print("\nUpdating size distribution...")
    manager.update_size_distribution({
        'small_540': 0.70,
        'medium_169': 0.30,
        'large_720': 0.00
    })
    
    # Validate again
    print("\nValidating updated config...")
    is_valid, errors = manager.validate()
    if is_valid:
        print("✅ Updated configuration is valid!")
    else:
        print("❌ Validation errors:")
        for error in errors:
            print(f"  - {error}")
