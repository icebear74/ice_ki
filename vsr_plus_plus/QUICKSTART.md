# VSR++ Quick Start Guide

## Prerequisites

```bash
# Install dependencies
pip install torch torchvision opencv-python tensorboard numpy
```

## Dataset Structure

Ensure your dataset is organized as:

```
/mnt/data/training/Dataset/Universal/Mastermodell/
├── Patches/          # Training set
│   ├── GT/           # Ground truth: 540x540 PNG files
│   └── LR/           # Low resolution: 180x900 PNG files (5 frames stacked)
└── Val/              # Validation set
    ├── GT/           # Ground truth: 540x540 PNG files
    └── LR/           # Low resolution: 180x900 PNG files (5 frames stacked)
```

## Starting Fresh Training

```bash
cd /home/runner/work/ice_ki/ice_ki
python vsr_plus_plus/train.py
```

When prompted:
```
⚠️  [L]öschen oder [F]ortsetzen? (L/F): L
```

The system will:
1. Run auto-tune to find optimal configuration
2. Display results in a formatted box
3. Wait for ENTER to continue
4. Start training with optimal settings

## Resuming Training

```bash
python vsr_plus_plus/train.py
```

When prompted:
```
⚠️  [L]öschen oder [F]ortsetzen? (L/F): F
```

The system will:
1. Load existing configuration
2. Show available checkpoints
3. Resume from latest checkpoint

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir /mnt/data/training/Universal/Mastermodell/Learn/logs --bind_all
```

Then open: http://localhost:6006

### Log Files

```bash
# View event log
tail -f /mnt/data/training/Universal/Mastermodell/Learn/training.log

# View current status
cat /mnt/data/training/Universal/Mastermodell/Learn/training_status.txt
```

## Understanding the Auto-Tune Output

```
╔═══════════════════════════════════════════════════════════════════════╗
║                    🔧 AUTO-TUNING SYSTEM                              ║
╠═══════════════════════════════════════════════════════════════════════╣
║ Testing Configurations...                                             ║
║                                                                        ║
║ [1/8] n_feats=192, batch=3, blocks=32                                ║
║       ⏱️  Speed: 3.8s/iter │ 💾 VRAM: 5.8GB │ ✅ PASSED               ║
║                                                                        ║
╠═══════════════════════════════════════════════════════════════════════╣
║ ✅ OPTIMAL CONFIGURATION FOUND                                        ║
║   Features: 192 | Batch: 3 | Blocks: 32                              ║
║   Speed: 3.8s/iter | VRAM: 5.8GB | Params: 12.4M                     ║
╠═══════════════════════════════════════════════════════════════════════╣
║              📸 Press ENTER to continue training...                   ║
╚═══════════════════════════════════════════════════════════════════════╝
```

- **Features**: Number of feature channels (64-256)
- **Batch**: Batch size (2-4)
- **Blocks**: Total ResBlocks (20-32)
- **Speed**: Seconds per training iteration
- **VRAM**: GPU memory usage
- **Params**: Total model parameters

## Checkpoints

### Types

1. **Regular**: Saved every 10,000 steps
   - `checkpoint_step_10000.pth`
   - `checkpoint_step_20000.pth`
   - Never deleted

2. **Best**: Saved when quality improves (during 2k-8k window)
   - `checkpoint_step_25500.pth` (example)
   - Symlink: `checkpoint_best.pth` → latest best
   - Symlink: `checkpoint_best_old.pth` → previous best

3. **Emergency**: Saved on crash or interrupt
   - `checkpoint_emergency.pth`

### Loading Checkpoints

Checkpoints are loaded automatically when resuming. To manually load:

```python
import torch
checkpoint = torch.load('checkpoint_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Configuration

Edit `train_config.json` to customize:

```json
{
  "LR_EXPONENT": -5,          // Initial LR = 10^-5
  "WEIGHT_DECAY": 0.001,      // AdamW weight decay
  "MAX_STEPS": 100000,        // Total training steps
  "VAL_STEP_EVERY": 500,      // Validation frequency
  "SAVE_STEP_EVERY": 10000,   // Checkpoint frequency
  "WARMUP_STEPS": 1000        // LR warmup steps
}
```

## Troubleshooting

### Out of Memory

Reduce VRAM limit in `train.py`:

```python
model_config = auto_tune_config(
    target_speed=4.0,
    max_vram_gb=4.0,  # Reduce from 6.0
    min_effective_batch=4
)
```

### Training Too Slow

Increase target speed in `train.py`:

```python
model_config = auto_tune_config(
    target_speed=6.0,  # Increase from 4.0
    max_vram_gb=6.0,
    min_effective_batch=4
)
```

### Dataset Not Found

Check paths in `train.py`:
```python
DATA_ROOT = "/mnt/data/training/Universal/Mastermodell/Learn"
DATASET_ROOT = "/mnt/data/training/Dataset/Universal/Mastermodell"
```

## Expected Output

### During Training

```
================================================================================
VSR++ Training - Step 15,000 | Epoch 8
================================================================================

Loss Components:
  Total: 0.012345 | L1: 0.008234 | MS: 0.002111 | Grad: 0.002000

Training Info:
  LR: 8.73e-05 | Speed: 3.8s/iter | VRAM: 5.2GB

Model Config:
  Features: 192 | Blocks: 32 | Batch: 3

Quality Metrics:
  LR Quality: 42.5% | KI Quality: 68.3% | Improvement: 25.8%

Adaptive System:
  Loss Weights: L1=0.60 | MS=0.20 | Grad=0.20
  Grad Clip: 1.523

Block Activity:
  Backward Trunk: 0.3245 | Forward Trunk: 0.3156

================================================================================
```

### Log Events

```
[2026-02-04 15:00:00] 🚀 TRAINING STARTED
[2026-02-04 15:05:12] Step 100 | Loss: 0.3245
[2026-02-04 15:30:45] Running validation at step 500
[2026-02-04 15:31:12] Step 500 | Validation | KI Quality: 45.2%
[2026-02-04 17:45:23] 🏆 NEW BEST CHECKPOINT! (Step 25000, Quality 77.8%)
[2026-02-04 20:15:42] Regular checkpoint saved at step 10000
```

## Performance Tips

1. Use SSD storage for datasets (not HDD)
2. Adjust `num_workers` in DataLoader based on CPU cores
3. Monitor VRAM - if consistently under 80%, increase n_feats
4. Reduce `VAL_STEP_EVERY` if validation is fast

## Support

For issues or questions:
1. Check `vsr_plus_plus/README.md` for detailed documentation
2. Review `vsr_plus_plus/ARCHITECTURE.md` for system overview
3. Examine log files for error messages

## Files Location

All training files in: `/mnt/data/training/Universal/Mastermodell/Learn/`
- `train_config.json` - Configuration
- `training.log` - Event log
- `training_status.txt` - Current status
- `checkpoint_*.pth` - Model checkpoints
- `logs/` - TensorBoard logs
