"""
Adaptive Batch Calculator - Automatically calculates batch size and accumulation

Manages VRAM budget < 6.5 GB (Plex transcoding reserve)

From configuration tests:
- ALL sizes use batch=1 (safest approach)
- Accumulation = effective_batch / 1
- 540×540 @ B1: 3.77 GB ✅
- 720×405 @ B1: 3.77 GB ✅  
- 720×720 @ B1×A6: ~6.0 GB ✅
"""

from typing import Dict, Any, Tuple, List


# VRAM estimates from config tests (in GB)
VRAM_ESTIMATES = {
    '540': {  # 540×540 GT, 180×180 LR
        'batch_1': 3.77,
    },
    '720_169': {  # 720×405 GT, 240×135 LR
        'batch_1': 3.77,
    },
    '720': {  # 720×720 GT, 240×240 LR
        'batch_1': 3.77,
    },
}

# VRAM safety limits
VRAM_LIMIT_GB = 6.5  # Total VRAM budget
VRAM_SAFE_GB = 6.0   # Target to stay under


class AdaptiveBatchCalculator:
    """
    Calculate optimal batch size and accumulation steps for each image size
    
    Strategy: Use batch=1 for ALL sizes (safest for VRAM)
    """
    
    def __init__(self, vram_limit_gb: float = VRAM_LIMIT_GB):
        """
        Initialize adaptive batch calculator
        
        Args:
            vram_limit_gb: VRAM limit in GB (default: 6.5)
        """
        self.vram_limit_gb = vram_limit_gb
        self.vram_safe_gb = VRAM_SAFE_GB
    
    def calculate_batch_config(self, gt_size: str, effective_batch: int) -> Dict[str, Any]:
        """
        Calculate batch and accumulation configuration for a given size
        
        Args:
            gt_size: Ground truth size category ('540', '720_169', '720')
            effective_batch: Target effective batch size (e.g., 6, 8, 12)
        
        Returns:
            Dict with:
                - batch: Physical batch size (always 1)
                - accum: Accumulation steps 
                - effective: Effective batch size (batch * accum)
                - vram_est: Estimated VRAM usage in GB
                - safe: Whether config is safe (< VRAM limit)
        """
        # Validate size category
        if gt_size not in VRAM_ESTIMATES:
            raise ValueError(f"Unknown size category: {gt_size}. Must be one of {list(VRAM_ESTIMATES.keys())}")
        
        # Validate effective batch
        if effective_batch < 1:
            raise ValueError(f"effective_batch must be >= 1, got {effective_batch}")
        
        # Fixed batch size = 1 (safest approach)
        batch = 1
        
        # Calculate accumulation steps
        accum = max(1, effective_batch // batch)
        
        # Get VRAM estimate for batch=1
        vram_est = VRAM_ESTIMATES[gt_size]['batch_1']
        
        # Check if configuration is safe
        safe = vram_est < self.vram_limit_gb
        
        # Calculate actual effective batch
        actual_effective = batch * accum
        
        return {
            'batch': batch,
            'accum': accum,
            'effective': actual_effective,
            'vram_est': vram_est,
            'safe': safe,
            'gt_size': gt_size,
        }
    
    def calculate_all_configs(self, effective_batch: int) -> Dict[str, Dict[str, Any]]:
        """
        Calculate batch configurations for all size categories
        
        Args:
            effective_batch: Target effective batch size
            
        Returns:
            Dict mapping size category to batch config
        """
        configs = {}
        
        for gt_size in VRAM_ESTIMATES.keys():
            configs[gt_size] = self.calculate_batch_config(gt_size, effective_batch)
        
        return configs
    
    def validate_config(self, batch_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a batch configuration
        
        Args:
            batch_config: Batch configuration dict
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check VRAM limit
        if batch_config['vram_est'] >= self.vram_limit_gb:
            errors.append(
                f"VRAM estimate {batch_config['vram_est']:.2f} GB exceeds limit {self.vram_limit_gb:.2f} GB"
            )
        
        # Check batch size is 1
        if batch_config['batch'] != 1:
            errors.append(
                f"Batch size must be 1 for safety, got {batch_config['batch']}"
            )
        
        # Check accumulation steps
        if batch_config['accum'] < 1:
            errors.append(
                f"Accumulation steps must be >= 1, got {batch_config['accum']}"
            )
        
        return len(errors) == 0, errors
    
    def validate_all_configs(self, configs: Dict[str, Dict[str, Any]]) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Validate all batch configurations
        
        Args:
            configs: Dict mapping size category to batch config
            
        Returns:
            Tuple of (all_valid, errors_dict)
        """
        all_errors = {}
        all_valid = True
        
        for gt_size, config in configs.items():
            is_valid, errors = self.validate_config(config)
            
            if not is_valid:
                all_valid = False
                all_errors[gt_size] = errors
        
        return all_valid, all_errors
    
    def get_vram_status(self, vram_gb: float) -> str:
        """
        Get VRAM status string with color coding
        
        Args:
            vram_gb: VRAM usage in GB
            
        Returns:
            Status string ('safe', 'warning', 'danger')
        """
        if vram_gb < self.vram_safe_gb:
            return 'safe'  # Green
        elif vram_gb < self.vram_limit_gb:
            return 'warning'  # Yellow
        else:
            return 'danger'  # Red
    
    def print_config_table(self, configs: Dict[str, Dict[str, Any]]):
        """
        Print a formatted table of batch configurations
        
        Args:
            configs: Dict mapping size category to batch config
        """
        print("\n" + "="*80)
        print("Adaptive Batch Configuration")
        print("="*80)
        print(f"{'Size Category':<15} {'Batch':>6} {'Accum':>6} {'Effective':>10} {'VRAM (GB)':>12} {'Status':>10}")
        print("-"*80)
        
        for gt_size, config in configs.items():
            status = self.get_vram_status(config['vram_est'])
            status_symbol = {
                'safe': '✅ Safe',
                'warning': '⚠️  Warning', 
                'danger': '❌ Danger'
            }[status]
            
            print(f"{gt_size:<15} {config['batch']:>6} {config['accum']:>6} {config['effective']:>10} "
                  f"{config['vram_est']:>12.2f} {status_symbol:>10}")
        
        print("-"*80)
        print(f"VRAM Limit: {self.vram_limit_gb:.2f} GB")
        print("="*80 + "\n")


# Convenience functions
def calculate_batch_config(gt_size: str, effective_batch: int) -> Dict[str, Any]:
    """
    Calculate batch configuration for a size category
    
    Args:
        gt_size: Size category
        effective_batch: Target effective batch size
        
    Returns:
        Batch configuration dict
    """
    calculator = AdaptiveBatchCalculator()
    return calculator.calculate_batch_config(gt_size, effective_batch)


def calculate_all_configs(effective_batch: int) -> Dict[str, Dict[str, Any]]:
    """
    Calculate batch configurations for all size categories
    
    Args:
        effective_batch: Target effective batch size
        
    Returns:
        Dict mapping size category to batch config
    """
    calculator = AdaptiveBatchCalculator()
    return calculator.calculate_all_configs(effective_batch)


if __name__ == "__main__":
    # Demo usage
    print("Adaptive Batch Calculator Demo\n")
    
    calculator = AdaptiveBatchCalculator()
    
    # Test with effective batch size = 6
    print("Testing with effective_batch_size = 6:")
    configs = calculator.calculate_all_configs(effective_batch=6)
    calculator.print_config_table(configs)
    
    # Validate
    is_valid, errors = calculator.validate_all_configs(configs)
    if is_valid:
        print("✅ All configurations are valid!\n")
    else:
        print("❌ Configuration errors:")
        for gt_size, error_list in errors.items():
            print(f"  {gt_size}:")
            for error in error_list:
                print(f"    - {error}")
        print()
