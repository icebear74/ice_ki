"""
YAML Configuration Loader for Unified Training System

Provides utilities to load and parse YAML configuration files
with nested dictionary access support.
"""

import yaml
import os
from typing import Any, Dict


class Config:
    """
    Configuration object with dot notation access
    
    Example:
        config = Config({'MODEL': {'name': 'VSR'}})
        print(config.MODEL.name)  # 'VSR'
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        
        # Convert nested dicts to Config objects
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default"""
        return self._config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access"""
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists"""
        return key in self._config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary"""
        return self._config


def load_yaml_config(config_path: str) -> Config:
    """
    Load YAML configuration file
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Config object with dot notation access
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    if config_dict is None:
        raise ValueError(f"Empty config file: {config_path}")
    
    return Config(config_dict)


def print_config(config: Config, indent: int = 0):
    """
    Pretty print configuration
    
    Args:
        config: Config object
        indent: Current indentation level
    """
    for key, value in config._config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(Config(value), indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


def validate_config(config: Config) -> bool:
    """
    Validate configuration has required fields
    
    Args:
        config: Config object
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If required fields are missing
    """
    required_sections = ['MODEL', 'DATA', 'TRAINING', 'LOSS', 'LOGGING', 'HARDWARE']
    
    for section in required_sections:
        if not hasattr(config, section):
            raise ValueError(f"Missing required section: {section}")
    
    # Check DATA section
    if not hasattr(config.DATA, 'category'):
        raise ValueError("DATA.category is required")
    
    if not hasattr(config.DATA, 'lr_version'):
        raise ValueError("DATA.lr_version is required")
    
    # Validate category
    valid_categories = ['general', 'space', 'toon']
    if config.DATA.category not in valid_categories:
        raise ValueError(f"Invalid category: {config.DATA.category}. Must be one of {valid_categories}")
    
    # Validate LR version
    valid_lr_versions = ['5frames', '7frames']
    if config.DATA.lr_version not in valid_lr_versions:
        raise ValueError(f"Invalid lr_version: {config.DATA.lr_version}. Must be one of {valid_lr_versions}")
    
    return True
