# VSR++ - Next Generation Video Super-Resolution Training System

A modular, high-performance VSR training system with automatic configuration tuning, smart checkpoint management, and comprehensive logging.

## Features

- **Bidirectional Propagation**: Frame-3 initialization for optimal temporal information usage
- **Auto-Tuning**: Automatically finds optimal configuration for your hardware
- **Smart Checkpointing**: Regular + best checkpoints with symlink management
- **Adaptive Training**: Dynamic loss weights, gradient clipping, and LR scheduling
- **Comprehensive Logging**: File logs + TensorBoard with 17+ graphs
- **Clean Architecture**: Modular design with clear separation of concerns

## Architecture

```
vsr_plus_plus/
├── core/               # Core ML components
│   ├── model.py        # VSRBidirectional_3x model
│   ├── loss.py         # HybridLoss (L1 + MS + Grad)
│   └── dataset.py      # VSRDataset loader
├── training/           # Training orchestration
│   ├── trainer.py      # Main training loop
│   ├── validator.py    # Validation logic
│   └── lr_scheduler.py # LR scheduling with warmup
├── systems/            # Support systems
│   ├── auto_tune.py    # Auto-configuration tuning
│   ├── checkpoint_manager.py  # Checkpoint management
│   ├── adaptive_system.py     # Adaptive weights/clipping
│   └── logger.py       # File + TensorBoard logging
└── utils/              # Utilities
    ├── metrics.py      # PSNR, SSIM, quality
    ├── ui.py           # GUI functions
    └── config.py       # Config management
```

## Installation

```bash
# Install dependencies
pip install torch torchvision opencv-python tensorboard numpy

# No additional setup needed - just run train.py!
```

## Usage

### Start New Training (with Auto-Tune)

```bash
python vsr_plus_plus/train.py
> L  # Choose "Löschen" to start fresh

# Auto-tune will test configurations and find the best one
# Press ENTER when prompted to continue
```

### Resume Training

```bash
python vsr_plus_plus/train.py
> F  # Choose "Fortsetzen" to resume

# Select checkpoint to resume from
```

### View Training Progress

```bash
# Start TensorBoard
tensorboard --logdir /mnt/data/training/Universal/Mastermodell/Learn/logs --bind_all

# Open browser to http://localhost:6006
```

## Configuration

Configuration is stored in `train_config.json`:

```json
{
  "LR_EXPONENT": -5,           // Learning rate: 10^-5 = 1e-5
  "WEIGHT_DECAY": 0.001,       // AdamW weight decay
  "MAX_STEPS": 100000,         // Total training steps
  "VAL_STEP_EVERY": 500,       // Validation frequency
  "SAVE_STEP_EVERY": 10000,    // Regular checkpoint frequency
  "LOG_TBOARD_EVERY": 100,     // TensorBoard log frequency
  "WARMUP_STEPS": 1000,        // LR warmup steps
  "AUTO_TUNED": true,          // Whether auto-tuned
  "MODEL_CONFIG": {
    "n_feats": 160,
    "n_blocks": 32,
    "batch_size": 4,
    "accumulation_steps": 1
  }
}
```

## Auto-Tuning System

The auto-tuner tests configurations in order of priority to find the best balance of:

- **VRAM usage**: Must fit within 80% of max VRAM (safety margin)
- **Training speed**: Must be ≤ target speed (default 4.0s/iter)
- **Model capacity**: Prefers larger models when possible

Test configurations (in order):

1. `n_feats=192, batch=3, n_blocks=32` (largest)
2. `n_feats=160, batch=4, n_blocks=32`
3. `n_feats=128, batch=4, n_blocks=32`
4. `n_feats=192, batch=2, n_blocks=32`
5. `n_feats=160, batch=3, n_blocks=24`
6. `n_feats=128, batch=3, n_blocks=24`
7. `n_feats=96, batch=4, n_blocks=24`
8. `n_feats=64, batch=4, n_blocks=20` (smallest)

The system automatically adjusts `accumulation_steps` to maintain an effective batch size ≥ 4.

## Checkpoint Strategy

### Regular Checkpoints

- Saved every 10,000 steps
- Filename: `checkpoint_step_10000.pth`, `checkpoint_step_20000.pth`, etc.
- **Never deleted** (kept permanently)

### Best Checkpoints

- Only checked during validation in the 2,000-8,000 step window
- Saved when quality improves: `checkpoint_step_25500.pth`
- Symlinks managed automatically:
  - `checkpoint_best.pth` → latest best
  - `checkpoint_best_old.pth` → previous best
- Uses **relative symlinks** for portability

### Emergency Checkpoints

- Saved on crash or keyboard interrupt
- Filename: `checkpoint_emergency.pth`
- Can be used to resume after unexpected shutdown

## TensorBoard Graphs

The system logs 17+ graphs to TensorBoard:

### Loss Components
1. `Loss/L1` - L1 loss component
2. `Loss/MS` - Multi-scale loss component
3. `Loss/Grad` - Gradient loss component
4. `Loss/Total` - Total weighted loss

### Training Info
5. `Training/LearningRate` - Current learning rate

### Adaptive System
6. `Adaptive/L1_Weight` - L1 loss weight
7. `Adaptive/MS_Weight` - Multi-scale loss weight
8. `Adaptive/Grad_Weight` - Gradient loss weight
9. `Adaptive/GradientClip` - Adaptive gradient clip value
10. `Adaptive/AggressiveMode` - Binary (0/1)

### Quality Metrics
11. `Quality/LR_Quality` - LR quality percentage
12. `Quality/KI_Quality` - KI quality percentage
13. `Quality/Improvement` - KI - LR improvement

### PSNR/SSIM
14. `Metrics/LR_PSNR` - LR PSNR
15. `Metrics/LR_SSIM` - LR SSIM
16. `Metrics/KI_PSNR` - KI PSNR
17. `Metrics/KI_SSIM` - KI SSIM

### System Info
18. `System/VRAM_GB` - VRAM usage in GB
19. `System/Speed_s_per_iter` - Training speed

### Gradients & Activity
20. `Gradients/TotalNorm` - Total gradient norm
21. `Activity/Backward_Trunk_Avg` - Backward trunk activity
22. `Activity/Forward_Trunk_Avg` - Forward trunk activity

### LR Schedule
23. `LR_Schedule/Phase` - Current phase (warmup/cosine/plateau)

### Events
24. `Events/Checkpoints` - Checkpoint timeline (1=regular, 2=best, 3=emergency)

### Validation Images
25. `Validation/Comparison` - Side-by-side LR | KI | GT images

## Logging System

### training.log

Append-only event log:

```
[2026-02-04 15:00:00] 🚀 TRAINING STARTED
[2026-02-04 15:05:12] Step 100 | Loss: 0.3245
[2026-02-04 16:30:45] ✅ WARMUP COMPLETE
[2026-02-04 17:45:23] Step 1500 | Validation | KI Quality: 45.2%
[2026-02-04 20:31:07] 🏆 NEW BEST CHECKPOINT! (Step 25000, Quality 77.8%)
[2026-02-04 22:15:42] ⚠️  Training interrupted by user
```

### training_status.txt

Overwritten every 5 steps with current status:

```
================================================================================
VSR++ Training Status - Updated 2026-02-04 22:15:42
================================================================================

TRAINING PROGRESS:
  Step: 25,000
  Epoch: 15
  Learning Rate: 8.73e-05
  Speed: 3.8s/iter
  VRAM: 5.2GB

LOSS COMPONENTS:
  Total: 0.012345
  L1: 0.008234
  MS: 0.002111
  Grad: 0.002000

...
```

## Model Architecture

### VSRBidirectional_3x

**Input**: `[B, 5, 3, 180, 180]` - 5 LR frames at 180x180

**Output**: `[B, 3, 540, 540]` - 1 HR frame at 540x540 (3x upscale)

**Key Components**:

1. **Feature Extraction**: Conv2d(3, n_feats, 3, 1, 1)
2. **Backward Trunk**: n_blocks/2 ResidualBlocks
3. **Forward Trunk**: n_blocks/2 ResidualBlocks
4. **Fusion**: Combine bidirectional features
5. **Upsampling**: PixelShuffle 3x

**Critical Feature - Frame-3 Initialization**:

The model initializes propagation from Frame 3 (center frame), NOT zeros:

```python
# CORRECT: Start with Frame 3
center_feat = feats[:, 2].clone()
back_prop = center_feat  # Backward: F3 → F4 → F5
forw_prop = center_feat  # Forward: F3 → F2 → F1
```

This ensures the model effectively uses temporal information.

## Adaptive Training System

### Dynamic Loss Weights

Automatically adjusts L1, MS, and Grad loss weights based on output sharpness:

- **Normal Mode**: Adjusts every 50 steps conservatively
- **Aggressive Mode**: Activated when blur detected
  - Adjusts every 10 steps
  - Boosts gradient loss to 0.30
  - Runs for max 5,000 steps or until stabilized

### Adaptive Gradient Clipping

Monitors gradient norms and auto-adjusts clip value:

- Tracks last 500 gradient norms
- Sets clip at 95th percentile
- Smooth updates (90% old, 10% new)

### LR Scheduling

Three-phase learning rate schedule:

1. **Warmup (0-1,000 steps)**: Linear 0 → 1e-4
2. **Cosine Annealing (1,000-max)**: 1e-4 → 1e-6
3. **Plateau Emergency**: Current LR × 0.5 when plateau detected

## Troubleshooting

### Out of Memory (OOM) Errors

If you get OOM errors during auto-tune:

```bash
# Reduce max VRAM in train.py:
model_config = auto_tune_config(
    target_speed=4.0,
    max_vram_gb=4.0,  # Reduce this
    min_effective_batch=4
)
```

### Training Too Slow

If training is slower than expected:

```bash
# Increase target speed in train.py:
model_config = auto_tune_config(
    target_speed=6.0,  # Increase this
    max_vram_gb=6.0,
    min_effective_batch=4
)
```

### Dataset Not Found

Ensure paths are correct in `train.py`:

```python
DATA_ROOT = "/mnt/data/training/Universal/Mastermodell/Learn"
DATASET_ROOT = "/mnt/data/training/Dataset/Universal/Mastermodell"
```

Dataset structure should be:

```
DATASET_ROOT/
├── Patches/  # Training set
│   ├── GT/   # 540x540 PNG files
│   └── LR/   # 180x900 PNG files (5 frames stacked)
└── Val/      # Validation set
    ├── GT/
    └── LR/
```

### Broken Symlinks

Run cleanup:

```bash
cd /mnt/data/training/Universal/Mastermodell/Learn
rm -f checkpoint_best*.pth  # Remove broken symlinks
```

The system will automatically recreate them on next best checkpoint.

## Performance Tips

1. **Use SSD storage** for datasets - HDD will bottleneck training
2. **Adjust num_workers** in DataLoader based on CPU cores
3. **Monitor VRAM** - if consistently under 80%, try higher n_feats
4. **Validation frequency** - Reduce VAL_STEP_EVERY if validation is slow

## Comparison with Original train.py

| Feature | Original | VSR++ |
|---------|----------|-------|
| Model | VSRTriplePlus_3x | VSRBidirectional_3x |
| Auto-tune | No | Yes |
| Checkpoint management | Basic | Smart with symlinks |
| Logging | Single file | File + TensorBoard |
| Adaptive system | External | Integrated |
| LR scheduling | Adaptive only | Warmup + Cosine + Adaptive |
| Code organization | Single file | Modular |
| Frame initialization | Weighted average | Frame-3 initialization |

## License

This code is part of the ice_ki repository.

## Credits

Developed for the ice_ki VSR training system.
