"""
Configuration API for runtime config management via web interface

This module extends the existing web_ui.py to add configuration management endpoints.
It uses the same HTTP server architecture for consistency.
"""

import json
import urllib.parse
from typing import Dict, Any, Optional


# Constants - shared with runtime_config module
DISTRIBUTION_SUM_TOLERANCE = 0.01  # Allow ±0.01 absolute tolerance for sum validation


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
    
    # NEW: 7-Frame VSR System Endpoints
    
    def handle_update_batch_config(self, effective_batch: int) -> Dict[str, Any]:
        """
        Update effective batch size and recalculate adaptive configs
        
        Args:
            effective_batch: New effective batch size (4-12)
            
        Returns:
            Success/error response with calculated configs
        """
        if self.runtime_config is None:
            return {'success': False, 'error': 'Runtime config not available'}
        
        # Validate range
        if not (4 <= effective_batch <= 12):
            return {
                'success': False,
                'error': f'Effective batch size must be 4-12, got {effective_batch}'
            }
        
        # Update config
        success = self.runtime_config.update_effective_batch_size(effective_batch)
        
        if success:
            # Get updated configs
            config = self.runtime_config.get_all()
            adaptive_batch = config.get('training', {}).get('adaptive_batch', {})
            
            # Calculate VRAM estimates
            from ..systems.adaptive_batch import AdaptiveBatchCalculator
            calculator = AdaptiveBatchCalculator()
            
            vram_estimates = {}
            for size, batch_config in adaptive_batch.items():
                calc_config = calculator.calculate_batch_config(size, effective_batch)
                vram_estimates[size] = {
                    'batch': batch_config['batch'],
                    'accum': batch_config['accum'],
                    'effective': batch_config['batch'] * batch_config['accum'],
                    'vram_est': calc_config['vram_est'],
                    'status': calculator.get_vram_status(calc_config['vram_est'])
                }
            
            return {
                'success': True,
                'effective_batch': effective_batch,
                'adaptive_configs': vram_estimates,
                'message': f'Batch configuration updated to {effective_batch}'
            }
        else:
            return {
                'success': False,
                'error': 'Failed to update batch configuration'
            }
    
    def handle_update_size_distribution(self, distribution: Dict[str, float]) -> Dict[str, Any]:
        """
        Update size distribution configuration
        
        Args:
            distribution: Dict mapping size category to percentage (0.0-1.0)
            
        Returns:
            Success/error response
        """
        if self.runtime_config is None:
            return {'success': False, 'error': 'Runtime config not available'}
        
        # Validate sum
        total = sum(distribution.values())
        tolerance_min = 1.0 - DISTRIBUTION_SUM_TOLERANCE
        tolerance_max = 1.0 + DISTRIBUTION_SUM_TOLERANCE
        
        if not (tolerance_min <= total <= tolerance_max):
            return {
                'success': False,
                'error': f'Size distribution must sum to 1.0 (±{DISTRIBUTION_SUM_TOLERANCE}), got {total:.4f}'
            }
        
        # Update config
        success = self.runtime_config.update_size_distribution(distribution)
        
        if success:
            return {
                'success': True,
                'distribution': distribution,
                'total': total,
                'message': 'Size distribution updated successfully'
            }
        else:
            return {
                'success': False,
                'error': 'Failed to update size distribution'
            }
    
    def handle_size_stats(self, size_tracker) -> Dict[str, Any]:
        """
        Get size tracking statistics
        
        Args:
            size_tracker: SizeTracker instance
            
        Returns:
            Size tracking statistics
        """
        if size_tracker is None:
            return {'error': 'Size tracker not available'}
        
        stats = size_tracker.get_stats()
        
        # Add formatted data for UI
        formatted_stats = []
        if 'size_stats' in stats:
            for category, cat_stats in stats['size_stats'].items():
                formatted_stats.append({
                    'category': category,
                    'trained': cat_stats['images_trained'],
                    'target': cat_stats['target_images'],
                    'percentage': cat_stats['percentage_complete'],
                    'last_step': cat_stats['last_trained_step']
                })
        
        return {
            'total_images': stats.get('total_images_trained', 0),
            'last_step': stats.get('last_step', 0),
            'categories': formatted_stats,
            'last_updated': stats.get('last_updated', '')
        }
    
    def handle_get_batch_preview(self, effective_batch: int) -> Dict[str, Any]:
        """
        Preview batch configuration without saving
        
        Args:
            effective_batch: Effective batch size to preview
            
        Returns:
            Preview of batch configs with VRAM estimates
        """
        from ..systems.adaptive_batch import AdaptiveBatchCalculator
        
        calculator = AdaptiveBatchCalculator()
        configs = calculator.calculate_all_configs(effective_batch)
        
        # Format for UI
        formatted_configs = {}
        for size, config in configs.items():
            formatted_configs[size] = {
                'batch': config['batch'],
                'accum': config['accum'],
                'effective': config['effective'],
                'vram_gb': config['vram_est'],
                'status': calculator.get_vram_status(config['vram_est']),
                'safe': config['safe']
            }
        
        return {
            'effective_batch': effective_batch,
            'configs': formatted_configs
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
