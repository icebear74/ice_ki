"""
Adaptive Batch Calculator - Automatically calculates batch size and accumulation

Manages VRAM budget < 6.5 GB (Plex transcoding reserve)

From configuration tests (7f | 26b | 72f | FP32):
- 720_169 BS=2, A=4: ~5.14 GB ✅
- 540     BS=2, A=3: ~5.15 GB ✅
- 720     BS=1, A=4: ~6.14 GB ✅ (BS=2 = OOM!)
"""

from typing import Dict, Any, Tuple, List, Optional


# VRAM estimates from config tests (in GB)
VRAM_ESTIMATES = {
    '540': {  # 540×540 GT, 180×180 LR
        'batch_1': 3.77,
        'batch_2': 5.15,   # 7f | B2×A3 | 26b | 72f | FP32 (gemessen)
    },
    '720_169': {  # 720×405 GT, 240×135 LR
        'batch_1': 3.77,
        'batch_2': 5.14,   # 7f | B2×A4 | 26b | 72f | FP32 (gemessen)
    },
    '720': {  # 720×720 GT, 240×240 LR
        'batch_1': 3.77,
        'batch_2': None,   # OOM bei BS=2! Nicht verwenden.
    },
}

# VRAM safety limits
VRAM_LIMIT_GB = 6.5  # Total VRAM budget
VRAM_SAFE_GB = 6.0   # Target to stay under


class AdaptiveBatchCalculator:
    """
    Calculate optimal batch size and accumulation steps for each image size

    Strategy: Use per-size measured VRAM values to determine safe batch sizes.
    - 720_169 and 540: batch=2 supported (measured < 6.5 GB)
    - 720: batch=1 only (BS=2 causes OOM)
    """
    
    def __init__(self, vram_limit_gb: float = VRAM_LIMIT_GB):
        """
        Initialize adaptive batch calculator
        
        Args:
            vram_limit_gb: VRAM limit in GB (default: 6.5)
        """
        self.vram_limit_gb = vram_limit_gb
        self.vram_safe_gb = VRAM_SAFE_GB
    
    def calculate_batch_config(self, gt_size: str, effective_batch: int,
                               batch_size: int = None) -> Dict[str, Any]:
        """
        Calculate batch and accumulation configuration for a given size

        Args:
            gt_size: Ground truth size category ('540', '720_169', '720')
            effective_batch: Target effective batch size (e.g., 6, 8, 12)
            batch_size: Physical batch size to use. If None, the maximum safe
                        batch size for this gt_size is chosen automatically.

        Returns:
            Dict with:
                - batch: Physical batch size
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

        size_vram = VRAM_ESTIMATES[gt_size]

        # Determine physical batch size
        if batch_size is None:
            # Use batch=2 if VRAM is known and safe, else batch=1
            if size_vram.get('batch_2') is not None and size_vram['batch_2'] < self.vram_limit_gb:
                batch = 2
            else:
                batch = 1
        else:
            batch = batch_size

        # Calculate accumulation steps
        accum = max(1, effective_batch // batch)

        # Get VRAM estimate for the chosen batch size
        vram_key = f'batch_{batch}'
        vram_val = size_vram.get(vram_key)
        vram_est = vram_val if vram_val is not None else size_vram.get('batch_1', 0.0)

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

    def get_vram_for_batch(self, gt_size: str, batch_size: int) -> Optional[float]:
        """
        Get the measured VRAM estimate for a given size and batch size.

        Args:
            gt_size: Size category ('540', '720_169', '720')
            batch_size: Physical batch size (1 or 2)

        Returns:
            VRAM in GB, or None if not measured / would OOM
        """
        if gt_size not in VRAM_ESTIMATES:
            return None
        return VRAM_ESTIMATES[gt_size].get(f'batch_{batch_size}')
    
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
        vram_est = batch_config.get('vram_est')
        if vram_est is None:
            errors.append(
                f"VRAM estimate is not available for this configuration (likely OOM risk)"
            )
        elif vram_est >= self.vram_limit_gb:
            errors.append(
                f"VRAM estimate {vram_est:.2f} GB exceeds limit {self.vram_limit_gb:.2f} GB"
            )

        # Check batch size is positive
        if batch_config['batch'] < 1:
            errors.append(
                f"Batch size must be >= 1, got {batch_config['batch']}"
            )

        # Check that batch=2 is not used for '720' (OOM risk)
        gt_size = batch_config.get('gt_size', '')
        if gt_size == '720' and batch_config['batch'] > 1:
            errors.append(
                f"Batch size > 1 is not supported for '720' (720×720) — OOM risk! "
                f"Got batch={batch_config['batch']}"
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
def calculate_batch_config(gt_size: str, effective_batch: int,
                           batch_size: int = None) -> Dict[str, Any]:
    """
    Calculate batch configuration for a size category

    Args:
        gt_size: Size category
        effective_batch: Target effective batch size
        batch_size: Physical batch size (optional; auto-selected if None)

    Returns:
        Batch configuration dict
    """
    calculator = AdaptiveBatchCalculator()
    return calculator.calculate_batch_config(gt_size, effective_batch, batch_size)


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
