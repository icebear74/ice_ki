# VSR++ Unified Training System

Multi-category and dual LR support for Video Super-Resolution training.

## Features

✅ **Multi-Category Support**
- General (Universal/Mastermodell)
- Space (specialized for space imagery)
- Toon (animated content)

✅ **Dual LR Versions**
- 5 frames (original)
- 7 frames (extended temporal context)

✅ **Multi-Format Training**
- Small (540×540)
- Medium 16:9 (720×405)
- Large (720×720)
- XLarge 16:9 (1440×810)
- FullHD (1920×1080)

✅ **Adaptive Batch Sizes**
- Per-format batch size configuration
- Gradient accumulation for memory efficiency
- Weighted format sampling

✅ **Fresh Training**
- Start from Step 0
- No resume required
- Clean checkpointing

## Quick Start

### 1. Prepare Your Dataset

Organize your data following this structure:

```
/mnt/data/training/dataset/
│
├── Universal/Mastermodell/Learn/     # GENERAL category
│   ├── Patches/                      # 540×540
│   │   ├── GT/                       # Ground truth images
│   │   ├── LR_5frames/              # 5 frames stacked (900px high)
│   │   └── LR_7frames/              # 7 frames stacked (1260px high)
│   │
│   ├── Patches_Medium169/            # 720×405
│   ├── Patches_Large/                # 720×720
│   └── Val/GT/                       # Validation GT only
│
├── Space/SpaceModel/Learn/           # SPACE category
│   ├── Patches/
│   ├── Patches_XLarge169/
│   └── Val/GT/
│
└── Toon/ToonModel/Learn/             # TOON category
    ├── Patches/
    ├── Patches_Medium169/
    └── Val/GT/
```

**Important Notes:**
- GT images are single frames
- LR images are stacked vertically (5 frames = 5×H, 7 frames = 7×H)
- Validation only needs GT images (LR generated on-the-fly)

### 2. Choose a Configuration

Pre-configured YAML files are available:

- `configs/train_general_7frames.yaml` - General model, 7 frames
- `configs/train_general_5frames.yaml` - General model, 5 frames
- `configs/train_space_7frames.yaml` - Space model, 7 frames
- `configs/train_toon_7frames.yaml` - Toon model, 7 frames

### 3. Train!

```bash
# Train General model with 7 frames
python vsr_plus_plus/train_unified.py --config configs/train_general_7frames.yaml

# Train Space model
python vsr_plus_plus/train_unified.py --config configs/train_space_7frames.yaml

# Train Toon model
python vsr_plus_plus/train_unified.py --config configs/train_toon_7frames.yaml
```

## Configuration Guide

### Basic Structure

```yaml
MODEL:
  base_channels: 128      # Feature channels (64-256)
  num_blocks: 32          # Residual blocks (16-48)
  use_checkpointing: true # Gradient checkpointing for VRAM

DATA:
  category: "general"     # "general", "space", or "toon"
  lr_version: "7frames"   # "5frames" or "7frames"
  data_root: "/mnt/data/training/dataset"
  
  formats:                # Which patch sizes to use
    - "small_540"
    - "medium_169"
  
  format_weights:         # Sampling probability
    small_540: 0.6
    medium_169: 0.4

TRAINING:
  total_steps: 100000
  
  batch_size:             # Per format
    small_540: 4
    medium_169: 2
  
  gradient_accumulation:  # Effective batch = batch_size × accumulation
    small_540: 1
    medium_169: 2
  
  learning_rate: 0.0001
  warmup_steps: 1000
```

### Format Names

| Format Name    | Resolution | Aspect Ratio | LR Size (5f) | LR Size (7f) |
|---------------|------------|--------------|--------------|--------------|
| `small_540`   | 540×540    | 1:1          | 180×900      | 180×1260     |
| `medium_169`  | 720×405    | 16:9         | 240×675      | 240×945      |
| `large_720`   | 720×720    | 1:1          | 240×1200     | 240×1680     |
| `xlarge_1440` | 1440×810   | 16:9         | 480×2700     | 480×3780     |
| `fullhd_1920` | 1920×1080  | 16:9         | 640×3240     | 640×4536     |

### Gradient Accumulation

Larger formats need smaller batch sizes due to memory constraints. Use gradient accumulation to maintain effective batch size:

```yaml
batch_size:
  small_540: 4          # Fits 4 samples
  fullhd_1920: 1        # Only 1 sample fits

gradient_accumulation:
  small_540: 1          # Effective batch = 4 × 1 = 4
  fullhd_1920: 4        # Effective batch = 1 × 4 = 4
```

## Model Architecture

The VSR model now supports variable frame counts:

- **5 frames**: Original configuration
  - Center frame: Index 2
  - Backward: 2 → 3 → 4
  - Forward: 2 → 1 → 0

- **7 frames**: Extended temporal context
  - Center frame: Index 3
  - Backward: 3 → 4 → 5 → 6
  - Forward: 3 → 2 → 1 → 0

Both configurations use bidirectional propagation with the center frame as initialization.

## Validation

Validation GT images are stored in `Val/GT/`. LR frames are generated on-the-fly during validation with slight horizontal shifts to simulate camera motion:

- **5 frames**: shifts = [-1.0, -0.5, 0.0, 0.5, 1.0]
- **7 frames**: shifts = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

This ensures consistent validation without storing LR copies.

## Output Structure

```
/mnt/data/training/experiments/general_fresh_7frames/
├── checkpoints/
│   ├── checkpoint_step_005000.pth
│   ├── checkpoint_step_010000.pth
│   └── ...
├── validation_images/
│   ├── step_000500/
│   │   ├── pred_000.png
│   │   ├── gt_000.png
│   │   └── ...
│   └── ...
└── tensorboard/
```

## Performance Tips

1. **Start Small**: Test with `small_540` only first
2. **Mixed Precision**: Enable for 1.5-2× speedup
3. **Gradient Checkpointing**: Enable for large models (saves VRAM)
4. **Worker Count**: Set `num_workers` = CPU cores / 2
5. **Format Weights**: Balance training based on importance

## Troubleshooting

### Out of Memory

```yaml
# Reduce batch size
batch_size:
  small_540: 2  # Was 4

# Increase accumulation
gradient_accumulation:
  small_540: 2  # Was 1

# Enable checkpointing
MODEL:
  use_checkpointing: true
```

### Dataset Not Found

Verify paths:
```bash
# Check structure
ls -R /mnt/data/training/dataset/Universal/Mastermodell/Learn/Patches/

# Should show: GT/, LR_5frames/, LR_7frames/
```

### Wrong Frame Count

LR stacked images must have exact heights:
- 5 frames: `height = LR_height × 5`
- 7 frames: `height = LR_height × 7`

Example for 180×180 LR:
- 5 frames: 180×900
- 7 frames: 180×1260

## Testing

### Quick Validation Test

Run the automated test script to verify your installation:

```bash
# Run quick validation test (creates mock data automatically)
python test_unified_training.py
```

This will verify:
- ✅ Config loading
- ✅ Dataset loading
- ✅ Model creation (5 and 7 frames)
- ✅ All components working

### Full Training Test

Once you have prepared your dataset, test with a real config:

```bash
# Full training (requires prepared dataset at /mnt/data/training/dataset/)
python vsr_plus_plus/train_unified.py --config configs/train_general_7frames.yaml
```

**Note**: Update `data_root` in the config file to match your actual dataset location if different from `/mnt/data/training/dataset/`.

## Migration from Old System

If you're migrating from the original training system:

1. **Dataset**: Organize into new structure (see above)
2. **Config**: Use YAML instead of Python config
3. **Training**: Use `train_unified.py` instead of `train.py`
4. **Resume**: Not supported in fresh start mode (by design)

The old training system (`train.py`) remains available for backward compatibility.

## Advanced Usage

### Custom Format Combinations

```yaml
DATA:
  formats:
    - "small_540"
    - "xlarge_1440"  # Skip medium
  
  format_weights:
    small_540: 0.7
    xlarge_1440: 0.3
```

### Category-Specific Settings

Each category can have different formats:

**General**: All sizes
**Space**: Large formats (xlarge, fullhd)
**Toon**: Smaller formats (small, medium)

See example configs for details.

## Support

For issues or questions:
1. Check configuration files in `configs/`
2. Review this README
3. Test with mock data first
4. Check logs for detailed errors
