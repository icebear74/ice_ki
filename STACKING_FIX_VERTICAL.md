# Stacking Fix - Vertical (Übereinander)

## Problem

User reported: "stacking ist falsch .. ist nebeneinander und nicht untereinander"
- Translation: "stacking is wrong .. is side-by-side and not underneath"

## Requirement

**New requirement:** "stacking muss ÜBEREINANDER nicht NEBENEINANDER"
- Translation: "stacking must be UNDERNEATH not SIDE-BY-SIDE"

## Solution

Changed `np.concatenate(lr_frames, axis=1)` to `np.concatenate(lr_frames, axis=0)`

### Understanding numpy concatenate axes:

- **axis=0**: Concatenates along the **height** dimension (vertical stacking - übereinander)
  - Frames stack underneath each other
  - For 7 frames of 240×240: Result is **1680×240** (height × width)

- **axis=1**: Concatenates along the **width** dimension (horizontal stacking - nebeneinander)
  - Frames stack side-by-side
  - For 7 frames of 240×240: Result is **240×1680** (height × width)

## Changes Made

### File: `dataset_generator_v2/make_dataset_v2_uhd.py`
Line 285:
```python
# Before (WRONG):
lr_stacked = np.concatenate(lr_frames, axis=1)  # Horizontal/nebeneinander

# After (CORRECT):
lr_stacked = np.concatenate(lr_frames, axis=0)  # Vertical/übereinander
```

### File: `dataset_generator_v2/make_dataset_v2_clean.py`
Line 220:
```python
# Before (WRONG):
lr_stacked = np.concatenate(lr_frames, axis=1)  # Horizontal

# After (CORRECT):
lr_stacked = np.concatenate(lr_frames, axis=0)  # Vertical
```

## Result

For 7 frames, each 240×240:
- **Old (axis=1)**: 240 (height) × 1680 (width) - wide horizontal strip
- **New (axis=0)**: 1680 (height) × 240 (width) - tall vertical strip ✓

## Aspect Ratio (16:9) - CORRECT

**Requirement:** "Video muss 16 (Width) : 9 (Height)"

Current format `720_169`:
- `gt_size: (405, 720)` = (height, width)
- Width = 720, Height = 405
- Aspect ratio = 720/405 = 1.778 ≈ 16/9 ✓

This is CORRECT - width (720) is 16/9 times the height (405).
