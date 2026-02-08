# VSR++ Manual Configuration Guide

## Overview

All training parameters are now configured manually in `config.py`. There is **no auto-tuning** - you have complete control over all settings.

## How to Use

1. **Edit `config.py`** - Open `vsr_plus_plus/config.py` and modify the parameters directly
2. **Run training** - Execute `python vsr_plus_plus/train.py`
3. **Choose L (Löschen)** to start fresh or **F (Fortsetzen)** to resume

## Quick Start

```bash
# 1. Edit configuration
nano vsr_plus_plus/config.py  # or your favorite editor

# 2. View current settings
python vsr_plus_plus/config.py

# 3. Start training
python vsr_plus_plus/train.py
```

## Key Parameters to Configure

### Model Architecture

```python
N_FEATS = 128      # Feature channels (64-256)
                   # Higher = more capacity, slower, more VRAM

N_BLOCKS = 32      # Residual blocks (20-32)
                   # Higher = more capacity, slower
```

**Recommendations:**
- **Small/Fast**: `N_FEATS=96`, `N_BLOCKS=24` (~6GB VRAM, 2-3s/iter)
- **Medium**: `N_FEATS=128`, `N_BLOCKS=32` (~8GB VRAM, 3-4s/iter)
- **Large**: `N_FEATS=160`, `N_BLOCKS=32` (~10GB VRAM, 4-5s/iter)
- **Extra Large**: `N_FEATS=192`, `N_BLOCKS=32` (~12GB VRAM, 5-6s/iter)

### Batch Settings

```python
BATCH_SIZE = 4              # Images per iteration
ACCUMULATION_STEPS = 1      # Gradient accumulation
```

**Effective Batch Size** = `BATCH_SIZE × ACCUMULATION_STEPS`

**Tips:**
- If out of memory: Reduce `BATCH_SIZE` or `N_FEATS`
- If you need larger batch but limited VRAM: Reduce `BATCH_SIZE` and increase `ACCUMULATION_STEPS`
- Example: `BATCH_SIZE=2`, `ACCUMULATION_STEPS=2` → Effective batch = 4

### Learning Rate

```python
LR_EXPONENT = -5     # Initial LR = 10^-5 = 0.00001
MAX_LR = 1e-4        # Maximum after warmup
MIN_LR = 1e-6        # Minimum at end
WARMUP_STEPS = 1000  # Warmup duration
```

**Common values:**
- Conservative: `LR_EXPONENT = -6` (slower, more stable)
- Normal: `LR_EXPONENT = -5` (balanced)
- Aggressive: `LR_EXPONENT = -4` (faster, less stable)

### Loss Weights

```python
L1_WEIGHT = 0.6      # Pixel-wise loss
MS_WEIGHT = 0.2      # Multi-scale loss
GRAD_WEIGHT = 0.2    # Gradient loss
```

**Should sum to ~1.0**

**Tips:**
- More detail/sharpness: Increase `GRAD_WEIGHT` to 0.3-0.4
- More color accuracy: Increase `L1_WEIGHT` to 0.7-0.8
- Better structure: Increase `MS_WEIGHT` to 0.3

### Training Duration

```python
MAX_STEPS = 100000       # Total steps
VAL_STEP_EVERY = 500     # Validation frequency
SAVE_STEP_EVERY = 10000  # Checkpoint frequency
```

## Example Configurations

### Fast Training (Limited VRAM)

```python
N_FEATS = 96
N_BLOCKS = 24
BATCH_SIZE = 4
ACCUMULATION_STEPS = 1
LR_EXPONENT = -5
MAX_STEPS = 50000
```

### Quality Training (High VRAM)

```python
N_FEATS = 160
N_BLOCKS = 32
BATCH_SIZE = 3
ACCUMULATION_STEPS = 2
LR_EXPONENT = -5
MAX_STEPS = 150000
```

### Balanced (Recommended Start)

```python
N_FEATS = 128
N_BLOCKS = 32
BATCH_SIZE = 4
ACCUMULATION_STEPS = 1
LR_EXPONENT = -5
MAX_STEPS = 100000
```

## Adaptive Features

```python
ADAPTIVE_LOSS_WEIGHTS = True   # Auto-adjust loss weights
ADAPTIVE_GRAD_CLIP = True      # Auto-adjust gradient clipping
```

If you want **complete manual control**, set both to `False` and the configured weights will remain fixed.

## Testing Your Configuration

Run this to see your current settings:

```bash
python vsr_plus_plus/config.py
```

This will print all parameters in a readable format.

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir /mnt/data/training/Universal/Mastermodell/Learn/logs --bind_all
```

### Log Files

```bash
# Event log
tail -f /mnt/data/training/Universal/Mastermodell/Learn/training.log

# Current status
cat /mnt/data/training/Universal/Mastermodell/Learn/training_status.txt
```

## Common Issues

### Out of Memory

**Solution 1:** Reduce model size
```python
N_FEATS = 96  # or 64
N_BLOCKS = 24  # or 20
```

**Solution 2:** Reduce batch size
```python
BATCH_SIZE = 2  # or 1
```

**Solution 3:** Use gradient accumulation
```python
BATCH_SIZE = 2
ACCUMULATION_STEPS = 2  # Effective batch = 4
```

### Training Too Slow

**Solution 1:** Reduce model size
```python
N_FEATS = 96
N_BLOCKS = 24
```

**Solution 2:** Reduce validation frequency
```python
VAL_STEP_EVERY = 1000  # instead of 500
```

### Loss Not Decreasing

**Solution 1:** Increase learning rate
```python
LR_EXPONENT = -4  # instead of -5
```

**Solution 2:** Adjust loss weights
```python
L1_WEIGHT = 0.7
GRAD_WEIGHT = 0.1
```

## Advanced Tips

1. **Start conservative** - Use default settings first
2. **Monitor VRAM** - Check TensorBoard System/VRAM_GB graph
3. **Check speed** - Aim for 3-5 seconds per iteration
4. **Watch quality** - Monitor Quality graphs in TensorBoard
5. **Adjust gradually** - Change one parameter at a time

## Getting Help

If you need to understand what a parameter does, check the comments in `config.py` - each parameter has an explanation.
