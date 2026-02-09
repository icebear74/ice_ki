# Unpacking Error Fix - Summary

## Error

```
2026-02-09 01:42:48,123 - ERROR - Error saving patches: too many values to unpack (expected 2)
```

## Root Cause

In `make_dataset_v2_uhd.py`, the `_save_patch_pair` method tried to unpack a dictionary return value into 2 variables:

```python
# WRONG - Dictionary has 4 keys, can't unpack into 2 variables
gt_dir, lr_dir = get_output_dirs_for_format(
    self.base_dir, category, format_name, n_frames
)
```

The function `get_output_dirs_for_format()` returns a **dictionary** with 4 keys:
- `'gt'` - Ground truth directory
- `'lr'` - Low resolution directory  
- `'val_gt'` - Validation GT directory
- `'val_lr'` - Validation LR directory

When Python unpacks a dictionary, it unpacks the **keys**, not values. Since there are 4 keys but only 2 variables, it raises:
```
ValueError: too many values to unpack (expected 2)
```

## Solution

Changed to properly handle the dictionary return:

```python
# CORRECT - Get dictionary, then extract needed values
output_dirs = get_output_dirs_for_format(
    self.base_dir, category, format_name, n_frames
)
gt_dir = output_dirs['gt']
lr_dir = output_dirs['lr']
```

## Files Changed

1. **dataset_generator_v2/make_dataset_v2_uhd.py**
   - Fixed `_save_patch_pair` method (lines 419-424)

2. **test_unpacking_fix.py** (NEW)
   - Verification test
   - 3/3 tests passing

## Testing

Test verifies:
- ✅ Function returns dictionary with correct keys
- ✅ New unpacking pattern works
- ✅ Old pattern fails with the exact error we saw

```
╔══════════════════════════════════════════════════════════╗
║           Unpacking Fix Verification                     ║
╚══════════════════════════════════════════════════════════╝

✅ PASS  Return Type
✅ PASS  Correct Unpacking  
✅ PASS  Wrong Unpacking Fails

Results: 3/3 tests passed
```

## Impact

**Before Fix:**
- ❌ Patch saving failed immediately
- ❌ No patches could be generated
- ❌ Generator unusable

**After Fix:**
- ✅ Patches save successfully
- ✅ Directories created correctly
- ✅ Generator runs without errors

## Status

✅ **FIXED and TESTED**
