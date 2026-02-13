# PyTorch 2.6 Checkpoint Loading Fix

## Problem

Training resume functionality was failing with the error:
```
[DEBUG] Error loading checkpoint_step_0006770_emergency.pth: Weights only load failed.
WeightsUnpickler error: Unsupported global: GLOBAL vsr_plusplus_NEU.training.lr_scheduler.AdaptiveLRScheduler
```

This caused:
- Checkpoint files to fail loading silently
- Empty checkpoint list
- No checkpoint selection menu appearing
- Training always starting from scratch instead of resuming

## Root Cause

**PyTorch 2.6 Breaking Change**: The default value of `weights_only` parameter in `torch.load()` changed from `False` to `True`.

- **Before PyTorch 2.6**: `torch.load(path)` → loads all objects including custom classes
- **PyTorch 2.6+**: `torch.load(path)` → only loads tensors/weights, fails on custom classes

Our checkpoints contain custom class instances:
- `vsr_plusplus_NEU.training.lr_scheduler.AdaptiveLRScheduler`
- Possibly other custom classes

These are not in PyTorch's default safe globals list, causing the load to fail.

## Solution

Add `weights_only=False` parameter to all `torch.load()` calls for checkpoints:

```python
# Before (fails in PyTorch 2.6+)
checkpoint = torch.load(path, map_location='cpu')

# After (works with all PyTorch versions)
checkpoint = torch.load(path, map_location='cpu', weights_only=False)
```

## Files Changed

1. **vsr_plusplus_NEU/systems/checkpoint_manager.py** (line 348)
   - `list_checkpoints()` method
   - Loads checkpoint metadata to display in selection menu

2. **vsr_plusplus_NEU/train.py** (line 678)
   - Main checkpoint loading for resume
   - Loads full checkpoint state for training continuation

## Security Note

Using `weights_only=False` is safe in this case because:
- We're loading our own checkpoints (not from untrusted sources)
- Checkpoints are created by the same codebase
- Custom classes (AdaptiveLRScheduler) are part of our code

## Testing

To verify the fix:
1. Run training script: `python vsr_plusplus_NEU/train.py`
2. Select "F" (Fortsetzen/Resume)
3. Checkpoint selection menu should now appear
4. Checkpoints should load successfully
5. Training should resume from selected checkpoint

## Debug Output

The fix includes debug output that shows:
- Path configuration (DATASET_ROOT, DATASET_SPECIFIC_ROOT)
- Checkpoint search directory and pattern
- List of found checkpoint files
- Any errors during checkpoint loading

This can be disabled by removing the debug print statements after confirming the fix works.

## Alternative Solution (Not Used)

PyTorch 2.6 also supports adding custom classes to safe globals:
```python
import torch.serialization
torch.serialization.add_safe_globals([AdaptiveLRScheduler])
checkpoint = torch.load(path, weights_only=True)
```

We chose `weights_only=False` instead because:
- Simpler solution (one parameter vs. managing global allowlist)
- More maintainable (no need to track which classes are in checkpoints)
- Our checkpoints are trusted
