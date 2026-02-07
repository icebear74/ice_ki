# Runtime Configuration System

## Overview

The Runtime Configuration System allows you to change training parameters **during training without restart**.

## Key Features

- Live reload every 10 steps
- Parameter validation (ranges, sums, types)
- Config snapshots with checkpoints
- Thread-safe operations

## Parameter Categories

### Safe Parameters (Change anytime)
- `plateau_safety_threshold` (100-5000)
- `plateau_patience` (50-1000)  
- `max_lr` (1e-5 to 1e-3)
- `min_lr` (1e-8 to 1e-4)

### Careful Parameters (Weight sum must = 0.95-1.05)
- `l1_weight_target` (0.1-0.9)
- `ms_weight_target` (0.05-0.5)
- `grad_weight_target` (0.05-0.5)
- `perceptual_weight_target` (0.0-0.25)

### Startup-Only (Requires restart)
- `n_feats`, `n_blocks`, `batch_size`

## Usage

Edit `runtime_config.json` in your training directory:
```json
{
  "plateau_patience": 300,
  "max_lr": 0.0002
}
```

Changes auto-apply within 10 steps.

## See Also
- Web UI for visual config editing
- Config snapshots saved with checkpoints
