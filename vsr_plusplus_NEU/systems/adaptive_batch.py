"""
Adaptive Batch Calculator - Automatically calculates batch size and accumulation

Manages VRAM budget < 6.5 GB (Plex transcoding reserve)

Strategy:
- Legacy size keys ('540', '720_169', '720') use measured VRAM values.
- Dynamic V2 templates use a pixel-count-based rule:
    GT pixels <= 405×720 (291,600): batch=2, accum=4 → eff=8
    GT pixels  > 405×720          : batch=1, accum=4 → eff=4
  This is conservative but always safe on 8 GB VRAM.

From configuration tests (7f | 24b | 72f | FP32):
- 720_169 BS=2, A=4: ~5.14 GB ✅
- 540     BS=2, A=3: ~5.15 GB ✅
- 720     BS=1, A=4: ~6.14 GB ✅ (BS=2 = OOM!)
"""

from typing import Dict, Any, Tuple, List, Optional


# VRAM estimates from config tests (in GB) for legacy size keys
VRAM_ESTIMATES = {
    '540': {  # 540×540 GT, 180×180 LR
        'batch_1': 3.77,
        'batch_2': 5.15,   # 7f | B2×A3 | 24b | 72f | FP32 (gemessen)
    },
    '720_169': {  # 720×405 GT, 240×135 LR
        'batch_1': 3.77,
        'batch_2': 5.14,   # 7f | B2×A4 | 24b | 72f | FP32 (gemessen)
    },
    '720': {  # 720×720 GT, 240×240 LR
        'batch_1': 3.77,
        'batch_2': None,   # OOM bei BS=2! Nicht verwenden.
    },
}

# Pixel-count threshold for the generic batch rule (405×720 = 291,600 px).
# Templates with GT area <= this threshold safely support batch_size=2.
_PIXEL_THRESHOLD_BATCH2 = 405 * 720  # 291,600 px

# VRAM safety limits
VRAM_LIMIT_GB = 6.5  # Total VRAM budget
VRAM_SAFE_GB = 6.0   # Target to stay under


def _pixel_count_batch_config(gt_w: int, gt_h: int, effective_batch: int) -> Dict[str, Any]:
    """
    Derive a safe batch config from GT pixel count.

    Used for dynamic V2 templates that are not in VRAM_ESTIMATES.

    Args:
        gt_w: GT image width in pixels.
        gt_h: GT image height in pixels.
        effective_batch: Target effective batch size (batch * accum).

    Returns:
        Dict with keys: batch, accum, effective, vram_est, safe, gt_size.
    """
    pixels = gt_w * gt_h
    batch = 2 if pixels <= _PIXEL_THRESHOLD_BATCH2 else 1
    accum = max(1, effective_batch // batch)
    return {
        'batch': batch,
        'accum': accum,
        'effective': batch * accum,
        'vram_est': 5.15 if batch == 2 else 3.77,  # conservative estimate
        'safe': True,
        'gt_size': f'{gt_w}x{gt_h}',
    }


class AdaptiveBatchCalculator:
    """
    Calculate optimal batch size and accumulation steps for each image size.

    For legacy size keys ('540', '720_169', '720') the measured VRAM table
    is used directly.  For dynamic V2 template keys the pixel-count rule is
    applied instead, so no code change is needed when new templates are added.
    """

    def __init__(self, vram_limit_gb: float = VRAM_LIMIT_GB):
        self.vram_limit_gb = vram_limit_gb
        self.vram_safe_gb = VRAM_SAFE_GB

    def calculate_batch_config(self, gt_size: str, effective_batch: int,
                               batch_size: Optional[int] = None,
                               gt_width: int = 0, gt_height: int = 0) -> Dict[str, Any]:
        """
        Calculate batch and accumulation configuration for a given size.

        For known legacy keys the existing measured VRAM table is used.
        For unknown keys (dynamic V2 templates) the pixel-count rule is applied
        when *gt_width* and *gt_height* are provided; otherwise a safe default
        of batch=1 / accum=effective_batch is returned.

        Args:
            gt_size:       Size key or template name.
            effective_batch: Target effective batch size (e.g., 6, 8).
            batch_size:    Physical batch size override (auto if None).
            gt_width:      GT image width (used for pixel-count rule on V2 templates).
            gt_height:     GT image height (used for pixel-count rule on V2 templates).

        Returns:
            Dict with: batch, accum, effective, vram_est, safe, gt_size.
        """
        if effective_batch < 1:
            raise ValueError(f"effective_batch must be >= 1, got {effective_batch}")

        # ── Dynamic V2 template (not in legacy table) ─────────────────────────
        if gt_size not in VRAM_ESTIMATES:
            if gt_width > 0 and gt_height > 0:
                result = _pixel_count_batch_config(gt_width, gt_height, effective_batch)
            else:
                # No pixel dimensions available — safest default
                batch = batch_size if batch_size is not None else 1
                accum = max(1, effective_batch // batch)
                result = {
                    'batch': batch,
                    'accum': accum,
                    'effective': batch * accum,
                    'vram_est': 3.77,
                    'safe': True,
                    'gt_size': gt_size,
                }
            if batch_size is not None:
                result['batch'] = batch_size
                result['accum'] = max(1, effective_batch // batch_size)
                result['effective'] = result['batch'] * result['accum']
            result['gt_size'] = gt_size
            return result

        # ── Legacy size key (measured VRAM table) ─────────────────────────────
        size_vram = VRAM_ESTIMATES[gt_size]

        if batch_size is None:
            if size_vram.get('batch_2') is not None and size_vram['batch_2'] < self.vram_limit_gb:
                batch = 2
            else:
                batch = 1
        else:
            batch = batch_size

        accum = max(1, effective_batch // batch)
        vram_key = f'batch_{batch}'
        vram_val = size_vram.get(vram_key)
        vram_est = vram_val if vram_val is not None else size_vram.get('batch_1', 0.0)
        safe = vram_est < self.vram_limit_gb

        return {
            'batch': batch,
            'accum': accum,
            'effective': batch * accum,
            'vram_est': vram_est,
            'safe': safe,
            'gt_size': gt_size,
        }

    def get_vram_for_batch(self, gt_size: str, batch_size: int) -> Optional[float]:
        """
        Get the measured VRAM estimate for a given size and batch size.

        For dynamic V2 templates (not in the legacy table) returns None.
        """
        if gt_size not in VRAM_ESTIMATES:
            return None
        return VRAM_ESTIMATES[gt_size].get(f'batch_{batch_size}')

    def calculate_all_configs(self, effective_batch: int) -> Dict[str, Dict[str, Any]]:
        """
        Calculate batch configurations for all legacy size categories.

        For dynamic V2 templates use ``calculate_batch_config`` directly.
        """
        configs = {}
        for gt_size in VRAM_ESTIMATES.keys():
            configs[gt_size] = self.calculate_batch_config(gt_size, effective_batch)
        return configs

    def validate_config(self, batch_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a batch configuration dict."""
        errors = []

        vram_est = batch_config.get('vram_est')
        if vram_est is None or vram_est == 0.0:
            errors.append("VRAM estimate is not available for this configuration (likely OOM risk)")
        elif vram_est >= self.vram_limit_gb:
            errors.append(
                f"VRAM estimate {vram_est:.2f} GB exceeds limit {self.vram_limit_gb:.2f} GB"
            )

        if batch_config['batch'] < 1:
            errors.append(f"Batch size must be >= 1, got {batch_config['batch']}")

        # Legacy safety check: '720' (720×720) must use batch=1
        gt_size = batch_config.get('gt_size', '')
        if gt_size == '720' and batch_config['batch'] > 1:
            errors.append(
                f"Batch size > 1 is not supported for '720' (720×720) — OOM risk! "
                f"Got batch={batch_config['batch']}"
            )

        if batch_config['accum'] < 1:
            errors.append(f"Accumulation steps must be >= 1, got {batch_config['accum']}")

        return len(errors) == 0, errors

    def validate_all_configs(self, configs: Dict[str, Dict[str, Any]]) -> Tuple[bool, Dict[str, List[str]]]:
        """Validate all batch configurations."""
        all_errors: Dict[str, List[str]] = {}
        all_valid = True

        for gt_size, config in configs.items():
            is_valid, errors = self.validate_config(config)
            if not is_valid:
                all_valid = False
                all_errors[gt_size] = errors

        return all_valid, all_errors

    def get_vram_status(self, vram_gb: float) -> str:
        """Get VRAM status string ('safe', 'warning', 'danger')."""
        if vram_gb < self.vram_safe_gb:
            return 'safe'
        elif vram_gb < self.vram_limit_gb:
            return 'warning'
        else:
            return 'danger'

    def print_config_table(self, configs: Dict[str, Dict[str, Any]]):
        """Print a formatted table of batch configurations."""
        print("\n" + "="*80)
        print("Adaptive Batch Configuration")
        print("="*80)
        print(f"{'Size Category':<18} {'Batch':>6} {'Accum':>6} {'Effective':>10} {'VRAM (GB)':>12} {'Status':>10}")
        print("-"*80)

        for gt_size, config in configs.items():
            status = self.get_vram_status(config['vram_est'])
            status_symbol = {
                'safe': '✅ Safe',
                'warning': '⚠️  Warning',
                'danger': '❌ Danger'
            }[status]

            print(f"{gt_size:<18} {config['batch']:>6} {config['accum']:>6} {config['effective']:>10} "
                  f"{config['vram_est']:>12.2f} {status_symbol:>10}")

        print("-"*80)
        print(f"VRAM Limit: {self.vram_limit_gb:.2f} GB")
        print("="*80 + "\n")


# Convenience functions
def calculate_batch_config(gt_size: str, effective_batch: int,
                           batch_size: int = None,
                           gt_width: int = 0, gt_height: int = 0) -> Dict[str, Any]:
    """
    Calculate batch configuration for a size category (or dynamic V2 template).

    Pass *gt_width* / *gt_height* for V2 templates not in the legacy table.
    """
    calculator = AdaptiveBatchCalculator()
    return calculator.calculate_batch_config(gt_size, effective_batch, batch_size,
                                             gt_width=gt_width, gt_height=gt_height)


def calculate_all_configs(effective_batch: int) -> Dict[str, Dict[str, Any]]:
    """Calculate batch configurations for all legacy size categories."""
    calculator = AdaptiveBatchCalculator()
    return calculator.calculate_all_configs(effective_batch)


if __name__ == "__main__":
    print("Adaptive Batch Calculator Demo\n")

    calculator = AdaptiveBatchCalculator()

    print("Testing with effective_batch_size = 6 (legacy size keys):")
    configs = calculator.calculate_all_configs(effective_batch=6)
    calculator.print_config_table(configs)

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

    print("Testing pixel-count rule for V2 templates:")
    for (w, h) in [(1152, 648), (960, 540), (960, 720), (720, 405), (540, 540)]:
        cfg = calculate_batch_config(f"{w}x{h}", 8, gt_width=w, gt_height=h)
        print(f"  {w}×{h}: batch={cfg['batch']}, accum={cfg['accum']}, eff={cfg['effective']}")
    print()
