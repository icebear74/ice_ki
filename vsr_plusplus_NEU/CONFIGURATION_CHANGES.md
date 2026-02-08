# VSR++ Configuration Change Summary

## What Changed

### BEFORE (Auto-Tune System)
```
User runs: python vsr_plus_plus/train.py
↓
Choose [L]öschen
↓
Auto-tune runs (tests 8 configurations)
↓
User waits ~2 minutes
↓
User presses ENTER
↓
Training starts with auto-detected config
```

**Problems:**
- ❌ No control over parameters
- ❌ Wait time before training
- ❌ Cannot customize specific values
- ❌ Need to understand auto-tune output

### AFTER (Manual Configuration)
```
User edits: vsr_plus_plus/config.py
↓
User runs: python vsr_plus_plus/train.py
↓
Choose [L]öschen
↓
Training starts IMMEDIATELY with configured parameters
```

**Benefits:**
- ✅ Full manual control
- ✅ No waiting
- ✅ Easy to understand
- ✅ Can set ALL parameters exactly as desired

## Files Changed

### Removed
- ❌ `vsr_plus_plus/systems/auto_tune.py` - Deleted completely

### Added
- ✅ `vsr_plus_plus/config.py` - Manual configuration file (7KB)
- ✅ `vsr_plus_plus/CONFIG_GUIDE.md` - Configuration guide (5KB)

### Modified
- 📝 `vsr_plus_plus/train.py` - Use config.py instead of auto-tune
- 📝 `vsr_plus_plus/systems/__init__.py` - Remove auto_tune import
- 📝 `vsr_plus_plus/README.md` - Update documentation

## New Configuration File

The new `config.py` contains **24 configurable parameters**:

### Model Architecture (2 params)
```python
N_FEATS = 128      # Feature channels
N_BLOCKS = 32      # Residual blocks
```

### Batch Settings (2 params)
```python
BATCH_SIZE = 4
ACCUMULATION_STEPS = 1
```

### Learning Rate (5 params)
```python
LR_EXPONENT = -5    # Initial LR
MAX_LR = 1e-4
MIN_LR = 1e-6
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.001
```

### Loss Weights (3 params)
```python
L1_WEIGHT = 0.6
MS_WEIGHT = 0.2
GRAD_WEIGHT = 0.2
```

### Training Schedule (5 params)
```python
MAX_STEPS = 100000
VAL_STEP_EVERY = 500
SAVE_STEP_EVERY = 10000
LOG_TBOARD_EVERY = 100
HIST_STEP_EVERY = 500
```

### Data Loading (2 params)
```python
NUM_WORKERS = 4
PIN_MEMORY = True
```

### Paths (2 params)
```python
DATA_ROOT = "/mnt/data/training/Universal/Mastermodell/Learn"
DATASET_ROOT = "/mnt/data/training/Dataset/Universal/Mastermodell"
```

### Adaptive System (3 params)
```python
ADAPTIVE_LOSS_WEIGHTS = True
ADAPTIVE_GRAD_CLIP = True
INITIAL_GRAD_CLIP = 1.5
```

## How to Use

### 1. View Current Configuration
```bash
python vsr_plus_plus/config.py
```

Output:
```
================================================================================
CURRENT CONFIGURATION
================================================================================

MODEL ARCHITECTURE:
  Features (n_feats):     128
  Blocks (n_blocks):      32

BATCH SETTINGS:
  Batch Size:             4
  Accumulation Steps:     1
  Effective Batch Size:   4

LEARNING RATE:
  Initial LR:             1.00e-05 (10^-5)
  Max LR:                 1.00e-04
  ...
```

### 2. Edit Configuration
```bash
nano vsr_plus_plus/config.py
# or
vim vsr_plus_plus/config.py
# or use any text editor
```

### 3. Start Training
```bash
python vsr_plus_plus/train.py
> L  # Start fresh
```

Training starts immediately with your configured parameters!

## Recommended Configurations

### Small/Fast (6GB VRAM)
```python
N_FEATS = 96
N_BLOCKS = 24
BATCH_SIZE = 4
```

### Medium (8GB VRAM) ⭐ Default
```python
N_FEATS = 128
N_BLOCKS = 32
BATCH_SIZE = 4
```

### Large (10GB VRAM)
```python
N_FEATS = 160
N_BLOCKS = 32
BATCH_SIZE = 3
```

### Extra Large (12GB VRAM)
```python
N_FEATS = 192
N_BLOCKS = 32
BATCH_SIZE = 2
```

## Common Adjustments

### Out of Memory?
```python
N_FEATS = 96           # Reduce from 128
# or
BATCH_SIZE = 2         # Reduce from 4
ACCUMULATION_STEPS = 2 # Keep effective batch = 4
```

### Want More Sharpness?
```python
GRAD_WEIGHT = 0.3   # Increase from 0.2
L1_WEIGHT = 0.5     # Decrease from 0.6
```

### Training Too Slow?
```python
N_FEATS = 96         # Reduce from 128
N_BLOCKS = 24        # Reduce from 32
VAL_STEP_EVERY = 1000  # Validate less often
```

## Documentation

Three documentation files now available:

1. **config.py** - The configuration file itself with inline comments
2. **CONFIG_GUIDE.md** - Detailed parameter explanations and examples
3. **README.md** - Updated usage instructions

## Migration Notes

If you have old auto-tuned configs, they will no longer be used. You should:

1. Note your previous model settings
2. Set them in `config.py`
3. Start training fresh with `L` choice

The old `train_config.json` is ignored - everything is now in `config.py`.

## Summary

**Auto-tune removed ✅**
- No more waiting
- No more automatic detection
- No more complex auto-tune output

**Manual control added ✅**
- Edit simple Python file
- See all parameters in one place
- Change anything you want
- Start training immediately

**Full transparency ✅**
- All 24 parameters documented
- Clear explanations for each
- Recommended presets included
- Easy to experiment and tune
