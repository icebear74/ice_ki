# Error Handling Guide - Dataset Loading

## Overview

The VSRDataset now includes robust error handling to prevent training crashes when invalid files are encountered.

## Problem

Previously, if a file with wrong dimensions was accidentally placed in the dataset folder (e.g., a 720×720 image in the 540×540 folder), the training would crash completely.

## Solution

### 1. Multi-Level Error Handling

#### A. Pre-Loading Validation
When the dataset is first loaded or reloaded, files are validated:

```python
# During __init__() and reload_files()
for each file:
    if _validate_file_dimensions(file):
        add to dataset
    else:
        skip with warning
```

**Checks:**
- GT image loads successfully
- LR image loads successfully  
- GT shape matches expected dimensions for size_key
- LR height is divisible by 7 (for 7-frame stack)

**Expected Shapes:**
- `size_key='720'`: GT = (720, 720, 3)
- `size_key='540'`: GT = (540, 540, 3)
- `size_key='720_169'`: GT = (405, 720, 3)  [16:9 aspect ratio]

#### B. Runtime Fallback in __getitem__()

If a file passes pre-validation but fails during actual loading:

```python
max_attempts = 3

Attempt 1: Try requested index
    ↓ Failed
Attempt 2: Try random fallback index
    ↓ Failed
Attempt 3: Try another random fallback
    ↓ Failed
CRITICAL ERROR: Raise exception
```

**This handles:**
- Corrupted files that appeared valid initially
- Files modified/replaced during training
- Race conditions during file system operations
- Unexpected dimension mismatches

### 2. Error Messages

#### Startup Validation (Pre-Loading)

**Example: Validation Mode**
```
📂 VALIDATION DATASET LOADING (540)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  GT files found:           5
  Matched in val/LR:        0
  Matched in patches/LR:    5
  ───────────────────────────────────
  Skipped (no LR):          0
  Skipped (invalid dims):   2
  Final samples loaded:     3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  2 files skipped due to invalid dimensions:
  - frame_0042.png: GT shape (720, 720, 3) != expected (540, 540, 3)
  - frame_0105.png: LR height 1680 not divisible by 7

💡 Expected dimensions for size_key '540':
   GT: (540, 540, 3)
```

**Example: Training Mode**
```
⚠️  Skipped 1 file with invalid dimensions in train (size_key=540)
  - wrong_size.png: GT shape (720, 720, 3) != expected (540, 540, 3)
```

#### Runtime Error (During Training)

**Example: File Load Error with Fallback**
```
⚠️  ERROR loading sample 42 (frame_0042.png): Invalid GT dimensions (720, 720, 3), expected (540, 540, 3) for size_key '540': frame_0042.png
   Attempting to use random fallback sample...
```

**Example: Critical Error (All Attempts Failed)**
```
⚠️  ERROR loading sample 42 (frame_0042.png): Failed to load images
   Attempting random fallback sample...
   Fallback attempt 1 failed, trying another...
   Fallback attempt 2 failed, trying another...

❌ CRITICAL: All 3 attempts to load a valid sample failed!
   Last error: Corrupted GT image: /path/to/frame_0157.png
   Dataset may have serious issues. Please check your data!

RuntimeError: Failed to load any valid sample after 3 attempts. Last file: frame_0157.png
```

#### Reload Validation (Dynamic Updates)

**Example: New Files Added During Training**
```
🔄 Reloading 540 dataset...
⚠️  Reload: Skipped 1 file with invalid dimensions (train, size_key=540)
  - new_frame_999.png: GT shape (720, 720, 3) != expected (540, 540, 3)
✅ Reload successful: 1,000 → 1,049 files
```

### 3. Best Practices

#### Preventing Issues

1. **Check Files Before Adding**
   ```bash
   # Check GT dimensions
   file /path/to/GT/frame_*.png | grep "720 x 720"  # for 720
   file /path/to/GT/frame_*.png | grep "540 x 540"  # for 540
   file /path/to/GT/frame_*.png | grep "720 x 405"  # for 720_169
   ```

2. **Validate LR Stack Height**
   ```python
   import cv2
   lr = cv2.imread("LR_7frames/frame.png")
   print(f"LR height: {lr.shape[0]} (should be divisible by 7)")
   print(f"Per-frame height: {lr.shape[0] // 7}")
   ```

3. **Use Correct Folders**
   - 720×720 images → `.../patches/720/GT/` and `.../patches/720/LR_7frames/`
   - 540×540 images → `.../patches/540/GT/` and `.../patches/540/LR_7frames/`
   - 720×405 images → `.../patches/720_169/GT/` and `.../patches/720_169/LR_7frames/`

#### Monitoring During Training

1. **Watch for Warnings**
   - Pre-loading warnings appear at startup
   - Runtime warnings appear during training
   - Reload warnings appear when new files are added

2. **Check Logs**
   - All skipped files are logged with reasons
   - Runtime errors show which specific file failed
   - Critical errors indicate dataset corruption

3. **Review Web UI**
   - File counts show how many valid files loaded
   - Compare with actual file counts in directories
   - Large discrepancy = many invalid files

### 4. Troubleshooting

#### Problem: Many Files Skipped

**Symptoms:**
```
⚠️  Skipped 50 files with invalid dimensions in train (size_key=540)
```

**Solution:**
1. Check if files are in correct folder
2. Verify file dimensions match size_key
3. Consider moving wrongly-sized files to correct folder

#### Problem: Training Slows Down

**Symptoms:**
```
⚠️  ERROR loading sample 42 (frame_0042.png): ...
   Attempting to use random fallback sample...
⚠️  ERROR loading sample 157 (frame_0157.png): ...
   Attempting to use random fallback sample...
```

**Cause:** Multiple corrupted files causing frequent fallbacks

**Solution:**
1. Review console for error patterns
2. Identify problematic files
3. Remove or replace corrupted files
4. Restart training with clean dataset

#### Problem: Critical Errors

**Symptoms:**
```
❌ CRITICAL: All 3 attempts to load a valid sample failed!
RuntimeError: Failed to load any valid sample after 3 attempts
```

**Cause:** Dataset severely corrupted or almost all files invalid

**Solution:**
1. **Immediate:** Stop training
2. **Investigate:** Check dataset directories
3. **Validate:** Manually inspect random files
4. **Fix:** Regenerate or repair dataset
5. **Test:** Validate a few files before restarting

### 5. Technical Details

#### Validation Method

```python
def _validate_file_dimensions(self, gt_file, gt_dir, lr_dir, 
                              expected_gt_shape, invalid_list):
    """
    Validate file dimensions without full processing
    
    Args:
        gt_file: Filename to validate
        gt_dir: GT directory path
        lr_dir: LR directory path
        expected_gt_shape: Expected GT shape (H, W, C) or None
        invalid_list: List to append (filename, reason) for invalid files
        
    Returns:
        bool: True if valid, False if invalid
    """
    try:
        gt = cv2.imread(gt_path)
        lr = cv2.imread(lr_path)
        
        # Check load success
        if gt is None or lr is None:
            invalid_list.append((gt_file, "Image failed to load"))
            return False
        
        # Check GT dimensions
        if gt.shape != expected_gt_shape:
            invalid_list.append((gt_file, f"GT shape mismatch"))
            return False
        
        # Check LR can be split into 7 frames
        if lr.shape[0] % 7 != 0:
            invalid_list.append((gt_file, f"LR height not divisible by 7"))
            return False
        
        return True
        
    except Exception as e:
        invalid_list.append((gt_file, f"Validation error: {e}"))
        return False
```

#### Fallback Mechanism

```python
def __getitem__(self, idx):
    max_attempts = 3
    
    for attempt in range(max_attempts):
        try:
            # Use current index on first attempt
            # Use random index on fallback attempts
            current_idx = idx if attempt == 0 else random.randint(0, len-1)
            
            # Load, validate, process
            gt_file = self.gt_files[current_idx]
            gt = cv2.imread(gt_path)
            
            # Validate dimensions
            if gt.shape != expected_shape:
                raise ValueError("Invalid dimensions")
            
            # Process and return
            return lr_stack, gt
            
        except Exception as e:
            # Log and try next attempt
            if attempt == 0:
                print(f"⚠️  ERROR loading sample {idx}: {e}")
                print(f"   Attempting fallback...")
            elif attempt < max_attempts - 1:
                print(f"   Fallback {attempt} failed, trying another...")
            else:
                # All attempts failed - critical
                print(f"❌ CRITICAL: All attempts failed!")
                raise RuntimeError(...)
```

### 6. Summary

**Before:**
- Invalid file → Training crash ❌
- No information about problem
- Manual intervention required
- Lost training progress

**After:**
- Invalid file → Automatic skip ✓
- Detailed error messages ✓
- Training continues ✓
- Fallback mechanism ✓
- Clear troubleshooting info ✓

The dataset loading is now production-ready and handles errors gracefully!
