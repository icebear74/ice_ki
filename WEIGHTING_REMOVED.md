# Weighting Logic Removed from Training

## User Requirement

> "entferne die gewichtung aus dem training. die ist bereits im dataset.. behandele alles gleich im training"

The user correctly identified that **dataset extraction already weights the files**, so additional weighting during training creates **double weighting** and is incorrect.

## Problem: Double Weighting

### Before (INCORRECT)

1. **Dataset Generation** creates files according to distribution:
   - If distribution is 40% 720, 40% 540, 20% 720_169
   - Extraction creates ~400 files for 720, ~400 for 540, ~200 for 720_169

2. **Training Sampler** ALSO weighted by distribution:
   - Tried to sample 40% from 720, 40% from 540, 20% from 720_169
   - **Result**: Double weighting! Wrong proportions!

### Example of the Problem

**Dataset on disk:**
- 720: 400 files (40%)
- 540: 400 files (40%)
- 720_169: 200 files (20%)

**Old sampler behavior:**
- Forced 40% sampling from 720 (already 40% of files!)
- Forced 40% sampling from 540 (already 40% of files!)
- Forced 20% sampling from 720_169 (already 20% of files!)

This is redundant and can cause incorrect behavior.

## Solution: Remove Training Weighting

### After (CORRECT)

1. **Dataset Generation** creates files according to distribution:
   - Same as before: ~400/400/200 files

2. **Training Sampler** samples proportionally to file counts:
   - Simply iterates through ALL files from all sizes
   - Natural proportion is already correct (40%/40%/20%)
   - No additional weighting needed!

### New Sampler Behavior

**Dataset on disk:**
- 720: 400 files
- 540: 400 files
- 720_169: 200 files
- **Total: 1000 files**

**New sampler:**
- Creates schedule with ALL batches from ALL sizes
- 720: 400 batches
- 540: 400 batches
- 720_169: 200 batches
- Shuffles this schedule
- **Result**: Natural 40%/40%/20% distribution ✓

## Changes Made

### 1. SizeGroupedSampler (core/dataloader.py)

**Removed:**
```python
# Normalize distribution to sum to 1.0
total_weight = sum(size_distribution[k] for k in self.active_sizes)
if total_weight == 0:
    raise ValueError("Total distribution weight is 0")

self.normalized_dist = {
    k: size_distribution[k] / total_weight 
    for k in self.active_sizes
}
```

**Added:**
```python
# NOTE: We NO LONGER normalize/weight by distribution!
# Files are already pre-weighted during dataset generation.
# We simply sample proportionally to actual file counts.
```

**Updated docstring:**
- Clarified that `size_distribution` is ONLY for filtering (which sizes to load)
- NOT for weighting during sampling
- Files on disk determine actual proportions

### 2. Documentation Updates

**runtime_config.json:**
- Added German explanation: "Die tatsächliche Gewichtung im Training erfolgt automatisch durch die Anzahl der Dateien auf der Festplatte!"
- Clarified that distribution values only control WHICH sizes to load, not HOW to weight them

**create_train_loader():**
- Updated docstring to explain distribution is informational
- Files on disk determine actual sampling ratio

## What size_distribution Does Now

### Old Behavior (WRONG)
```
distribution = {
    '720': 0.4,      # Load 720 AND weight it at 40% during sampling
    '540': 0.4,      # Load 540 AND weight it at 40% during sampling
    '720_169': 0.2   # Load 169 AND weight it at 20% during sampling
}
```

### New Behavior (CORRECT)
```
distribution = {
    '720': 0.4,      # Load 720 if > 0 (value is informational only)
    '540': 0.4,      # Load 540 if > 0 (value is informational only)
    '720_169': 0.2   # Load 169 if > 0 (value is informational only)
}
```

**The actual sampling ratio comes from file counts on disk!**

## Benefits

### ✅ No Double Weighting
- Files created with correct distribution
- Training uses correct distribution
- No conflict or redundancy

### ✅ Simpler Logic
- Removed normalization code
- Removed distribution-based scheduling
- Just sample all files proportionally

### ✅ More Transparent
- File counts visible in Web UI
- Actual proportions clear from file counts
- No hidden weighting logic

### ✅ Correct Behavior
- If you have 400/400/200 files, you train on 40%/40%/20%
- If you have 600/300/100 files, you train on 60%/30%/10%
- Distribution follows reality!

## Migration Guide

### For Users

**No changes needed!**

If your dataset extraction already created files according to distribution:
- Training will now correctly use those proportions
- No configuration changes required

### For Developers

If you want to change the training distribution:
1. **Don't** change `size_distribution` during training
2. **Do** change it in dataset extraction configuration
3. **Do** re-extract or add/remove files to get desired proportion
4. Restart training to load new file counts

## Technical Details

### Sampling Algorithm

**Old (with distribution weighting):**
```python
# Calculate how many batches to sample from each size based on distribution
batches_720 = total_batches * 0.4
batches_540 = total_batches * 0.4
batches_169 = total_batches * 0.2
```

**New (proportional to files):**
```python
# Sample ALL batches from ALL sizes
batches_720 = num_files_720 // batch_size
batches_540 = num_files_540 // batch_size
batches_169 = num_files_169 // batch_size
total_batches = batches_720 + batches_540 + batches_169
```

### Epoch Definition

An "epoch" now means:
- **One pass through ALL files from ALL sizes**
- Not "one pass weighted by distribution"

Example:
- 400 files in 720, 400 in 540, 200 in 720_169
- 1 epoch = 1000 batches (assuming batch_size=1)
- Naturally 40%/40%/20% split

## Backward Compatibility

### Configuration Files

Old runtime_config.json files work without changes:
- `size_distribution` still read
- Values > 0 still mean "load this size"
- Only the interpretation changed (filtering vs weighting)

### Checkpoints

No changes to checkpoint format needed.

### Existing Training

If resuming training:
- New behavior applies immediately
- File counts determine proportions
- More correct than before!

## Summary

**User was absolutely right!**

Dataset extraction creates weighted files → Training should sample them proportionally → No additional weighting needed.

The fix:
1. Removed distribution normalization from sampler
2. Sample proportionally to actual file counts
3. Updated documentation to clarify
4. `size_distribution` now only filters which sizes to load

**Result: Simpler, more transparent, and actually correct!** ✓
