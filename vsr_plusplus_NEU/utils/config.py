"""
Configuration Management
"""

import json
import os


def get_default_config() -> dict:
    """Return default configuration"""
    return {
        "LR_EXPONENT": -5,
        "WEIGHT_DECAY": 0.001,
        "MAX_STEPS": 100000,
        "VAL_STEP_EVERY": 500,
        "SAVE_STEP_EVERY": 10000,  # Regular checkpoints
        "LOG_TBOARD_EVERY": 100,
        "HIST_STEP_EVERY": 500,
        "WARMUP_STEPS": 1000,
        "AUTO_TUNED": False,
        "MODEL_CONFIG": {}
    }


def load_config(path: str) -> dict:
    """
    Load config from JSON, merge with defaults
    
    Args:
        path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    defaults = get_default_config()
    
    if not os.path.exists(path):
        return defaults
    
    try:
        with open(path, 'r') as f:
            user_config = json.load(f)
        
        # Merge with defaults
        config = defaults.copy()
        config.update(user_config)
        
        return config
    except Exception as e:
        print(f"Warning: Failed to load config from {path}: {e}")
        return defaults


def save_config(config: dict, path: str):
    """
    Save config to JSON
    
    Args:
        config: Configuration dictionary
        path: Path to save config file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def validate_config(config: dict) -> bool:
    """
    Validate config has required fields
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid
    """
    required_fields = [
        'LR_EXPONENT', 'WEIGHT_DECAY', 'MAX_STEPS',
        'VAL_STEP_EVERY', 'SAVE_STEP_EVERY', 'LOG_TBOARD_EVERY'
    ]
    
    for field in required_fields:
        if field not in config:
            print(f"Error: Missing required field '{field}' in config")
            return False
    
    return True
