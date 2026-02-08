"""
Configuration API for runtime config management via web interface

This module extends the existing web_ui.py to add configuration management endpoints.
It uses the same HTTP server architecture for consistency.
"""

import json
import urllib.parse
from typing import Dict, Any, Optional


class ConfigAPIHandler:
    """
    Handler for configuration API requests
    
    This class provides methods to handle config-related HTTP requests
    that can be integrated into the existing WebMonitorRequestProcessor
    """
    
    def __init__(self, runtime_config, checkpoint_manager=None, trainer=None):
        """
        Initialize config API handler
        
        Args:
            runtime_config: RuntimeConfigManager instance
            checkpoint_manager: Optional CheckpointManager instance
            trainer: Optional VSRTrainer instance
        """
        self.runtime_config = runtime_config
        self.checkpoint_manager = checkpoint_manager
        self.trainer = trainer
    
    def handle_get_config(self) -> Dict[str, Any]:
        """
        Get all runtime configuration
        
        Returns:
            Dict with all config values
        """
        if self.runtime_config is None:
            return {'error': 'Runtime config not available'}
        
        config = self.runtime_config.get_all()
        
        # Add metadata
        from ..systems.runtime_config import RUNTIME_SAFE_PARAMS, RUNTIME_CAREFUL_PARAMS, STARTUP_ONLY_PARAMS
        
        return {
            'config': config,
            'categories': {
                'safe': list(RUNTIME_SAFE_PARAMS.keys()),
                'careful': list(RUNTIME_CAREFUL_PARAMS.keys()),
                'startup_only': list(STARTUP_ONLY_PARAMS)
            },
            'ranges': {
                **RUNTIME_SAFE_PARAMS,
                **RUNTIME_CAREFUL_PARAMS
            }
        }
    
    def handle_update_config(self, param: str, value: Any) -> Dict[str, Any]:
        """
        Update a single config parameter
        
        Args:
            param: Parameter name
            value: New value
            
        Returns:
            Success/error response
        """
        if self.runtime_config is None:
            return {'success': False, 'error': 'Runtime config not available'}
        
        # Convert value to appropriate type
        try:
            # Try to parse as float first for numeric values
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    pass
        except:
            pass
        
        # Update config
        success = self.runtime_config.set(param, value, validate=True)
        
        if success:
            return {
                'success': True,
                'param': param,
                'value': value,
                'message': f'Successfully updated {param} to {value}'
            }
        else:
            return {
                'success': False,
                'param': param,
                'error': f'Failed to update {param}'
            }
    
    def handle_trigger_checkpoint(self) -> Dict[str, Any]:
        """
        Manually trigger checkpoint save
        
        Returns:
            Success/error response
        """
        if self.checkpoint_manager is None or self.trainer is None:
            return {'success': False, 'error': 'Checkpoint manager or trainer not available'}
        
        try:
            # Save checkpoint
            step = self.trainer.global_step
            metrics = self.trainer.last_metrics or {}
            
            self.checkpoint_manager.save_checkpoint(
                self.trainer.model,
                self.trainer.optimizer,
                self.trainer.lr_scheduler,
                step,
                metrics,
                self.trainer.train_logger.log_file,
                self.runtime_config
            )
            
            return {
                'success': True,
                'step': step,
                'message': f'Checkpoint saved at step {step}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def handle_list_snapshots(self) -> Dict[str, Any]:
        """
        List all config snapshots
        
        Returns:
            List of snapshot info
        """
        if self.runtime_config is None:
            return {'error': 'Runtime config not available'}
        
        snapshots = self.runtime_config.list_snapshots()
        
        return {
            'snapshots': snapshots,
            'count': len(snapshots)
        }
    
    def handle_restore_snapshot(self, step: int) -> Dict[str, Any]:
        """
        Restore config from snapshot
        
        Args:
            step: Snapshot step number
            
        Returns:
            Success/error response
        """
        if self.runtime_config is None:
            return {'success': False, 'error': 'Runtime config not available'}
        
        success = self.runtime_config.load_snapshot(step)
        
        if success:
            return {
                'success': True,
                'step': step,
                'message': f'Config restored from step {step}'
            }
        else:
            return {
                'success': False,
                'error': f'Failed to restore config from step {step}'
            }
    
    def handle_compare_snapshots(self, step1: int, step2: int) -> Dict[str, Any]:
        """
        Compare two config snapshots
        
        Args:
            step1: First snapshot step
            step2: Second snapshot step
            
        Returns:
            Comparison results
        """
        if self.runtime_config is None:
            return {'error': 'Runtime config not available'}
        
        comparison = self.runtime_config.compare_snapshots(step1, step2)
        
        return comparison
    
    def handle_validation_snapshot(self, snapshot_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Trigger validation snapshot
        
        Args:
            snapshot_name: Optional name suffix
            
        Returns:
            Success/error response
        """
        if self.trainer is None:
            return {'success': False, 'error': 'Trainer not available'}
        
        try:
            results = self.trainer.run_validation_snapshot(snapshot_name)
            
            return {
                'success': True,
                'results': results,
                'message': 'Validation snapshot completed'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


def parse_query_params(query_string: str) -> Dict[str, Any]:
    """
    Parse query string into dict
    
    Args:
        query_string: URL query string
        
    Returns:
        Dict of parsed parameters
    """
    if not query_string:
        return {}
    
    params = {}
    for key_value in query_string.split('&'):
        if '=' in key_value:
            key, value = key_value.split('=', 1)
            key = urllib.parse.unquote(key)
            value = urllib.parse.unquote(value)
            params[key] = value
    
    return params
