# VSR++ — Next Generation Video Super-Resolution Training System

A modular, high-performance VSR training system with **manual configuration**, smart checkpoint
management, and comprehensive logging.  Fully migrated to the **Generator V2 dynamic dataset
structure** (templates from `dataset_architecture.json`).

## Features

- **Bidirectional Propagation**: Frame-3 initialization for optimal temporal information usage
- **Dynamic Template Discovery**: Templates are discovered from `dataset_architecture.json` — no hardcoded size keys
- **Locked Run Config**: `training_run_locked.json` ensures checkpoint compatibility across restarts
- **Smart Checkpointing**: Regular + best checkpoints with symlink management
- **Adaptive Training**: Dynamic loss weights, gradient clipping, and LR scheduling
- **Comprehensive Logging**: File logs + TensorBoard with 17+ graphs
- **Graduated Data Strategy**: Dynamic Phase 1/2/3 schedule based on architecture metadata

## Default Model Settings (benchmark-validated)

```python
N_FEATS  = 72   # Feature channels — strong quality, within 8 GB VRAM
N_BLOCKS = 24   # Residual blocks — balanced choice (26/28 are increasingly expensive
                #   for larger V2 formats like 960×540, 960×720, 1152×648)
```

**Lighter alternative**: `N_BLOCKS = 20` (reduced VRAM, slightly lower quality).

## Module Structure

```
vsr_plusplus_NEU/
├── config.py.example       # ⭐ Copy to config.py and edit!
├── train.py                # Entry point
├── core/
│   ├── model_7frame.py     # VSRBidirectional_3x (configurable odd n_frames)
│   ├── loss.py             # HybridLoss
│   ├── dataset.py          # VSRDataset (V2 bucket + flat layout)
│   ├── dataloader.py       # MultiSizeDataLoader + DataStrategyScheduler
│   └── data_strategy.py    # Dynamic 3-phase strategy (arch-driven)
├── training/
│   ├── trainer.py          # Main training loop (dynamic size keys)
│   ├── validator.py        # Validation (GT-only, LR from patches/)
│   └── lr_scheduler.py     # Warmup + cosine + plateau
├── systems/
│   ├── run_lock.py         # ⭐ Locked run config (checkpoint compatibility)
│   ├── checkpoint_manager.py
│   ├── adaptive_system.py
│   ├── adaptive_batch.py   # Pixel-count-based batch rules for V2 templates
│   └── size_tracking.py    # Dynamic per-template counters
└── utils/
    ├── dataset_architecture.py  # dataset_architecture.json loader
    └── ...
```

## Quick Start

### 1. Create config.py

```bash
cd vsr_plusplus_NEU
cp config.py.example config.py
# Edit config.py to set DATASET_ROOT and DEFAULT_DATASET_NAME
```

### 2. Start Training

```bash
python train.py
> L  # Fresh start (backs up existing checkpoints)
> F  # Resume from checkpoint
```

On **first start** `training_run_locked.json` is written to the category directory.
On **resume** the locked config is verified — mismatches abort with a clear error.

### 3. Monitor Progress

TensorBoard is started automatically.  Manual access:

```bash
tensorboard --logdir /mnt/data/training/Dataset/master/logs --bind_all
# Open http://localhost:6006
```

## Configuration

### config.py Settings

```python
# Dataset root — new V2 default path
DATASET_ROOT = "/mnt/data/training/Dataset"

# Category (must match dataset_architecture.json)
DEFAULT_DATASET_NAME = "master"   # or 'space', 'toon', 'universal'

# Model defaults (benchmark-validated)
N_FEATS  = 72
N_BLOCKS = 24   # lighter: 20

# Per-template batch config (pixel-count rule applies to unknown V2 templates)
ADAPTIVE_BATCH_CONFIG = {
    '720_169': {'batch': 2, 'accum': 4},   # eff=8
    '540':     {'batch': 2, 'accum': 3},   # eff=6
    '720':     {'batch': 1, 'accum': 4},   # eff=4 (BS=1 required!)
}
```

### Dynamic Template Discovery

Templates are read from `{DATASET_ROOT}/dataset_architecture.json`.  No size keys need to
be hardcoded — add new V2 formats (e.g. `960_169`, `1152_169`) to the architecture file and
the training system picks them up automatically.

## Checkpoint Compatibility (Run Lock)

`training_run_locked.json` is stored in the category run directory and locks:

- `n_feats`, `n_blocks`, `n_frames`, `scale`
- `dataset_root`, `category`
- Template list for the run

If any of these differ on resume, training aborts with a clear error.
To start fresh, choose `L` at the prompt — this backs up checkpoints and removes the lock.

## Validation Workflow

1. Copy GT images to `{DATASET_ROOT}/{category}/val/{template}/GT/`
2. **Never** copy LR to `val/` — LR is always auto-found in `patches/{template}/LR_{n}frames/` via basename index
3. GT and LR basenames must be identical

## Data Strategy Phases

| Phase | Steps | Data | Perceptual |
|-------|-------|------|------------|
| Warmup | 0–3000 | 100 % warmup template (largest GT area) | 0.0 → 0.03 |
| Crop Intro | 3000–8000 | Linear → arch weight distribution | 0.03 → 0.08 |
| Stable | 8000+ | Natural file-count sampling (no override) | AdaptiveSystem |

Phase 2 is gated on `MIN_CROP_FILES_TRAINING = 10000` non-warmup GT images on disk.

## Model Architecture

`n_frames` is loaded from `{DATASET_ROOT}/dataset_architecture.json` (must be odd, >=3).
The model outputs the upscaled center frame (`center = n_frames // 2`) and keeps 3× upscale.

Input:  `[B, n_frames, 3, H_lr, W_lr]` — LR frame window from generator layout
Output: `[B, 3, H_gt, W_gt]`           — 1 HR center frame

## Troubleshooting

**CHECKPOINT COMPATIBILITY ERROR**: Config differs from `training_run_locked.json`.
→ Restore the original config values or start fresh with `L`.

**MODEL ARCHITECTURE MISMATCH**: runtime/checkpoint/run-lock `n_frames` differs.
→ Use a matching dataset/checkpoint or start a fresh run.

**No GT/LR matches**: Basenames in `patches/{template}/GT/` and `LR_{n}frames/` must match.
→ Check `DATASET_ROOT` and `DEFAULT_DATASET_NAME` in `config.py`.

**Out of Memory**: Reduce `N_BLOCKS` to 20, or add a custom entry to `ADAPTIVE_BATCH_CONFIG`
with `batch=1`.

See `DATASET_STRUCTURE.md` for the full dataset layout and validation workflow.
