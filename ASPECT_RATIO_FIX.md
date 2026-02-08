# 16:9 Aspect Ratio Fix

## Problem (German)
"16:9 format ist auch falsch 16 (breite) 9 (höhe) .."

**Translation:** "16:9 format is also wrong - 16 (width) 9 (height)"

## Issue
The 16:9 format had inverted dimensions - it was actually 9:16 (portrait) instead of 16:9 (landscape).

## Root Cause
In OpenCV/numpy, image dimensions are specified as `(height, width)`, not `(width, height)`.

For a 16:9 aspect ratio (landscape):
- Width should be larger than height
- Aspect ratio: width ÷ height = 16 ÷ 9 = 1.778

## Before (WRONG - 9:16 Portrait)

```python
'720_169': {
    'gt_size': (720, 405),  # height=720, width=405
    'lr_size': (240, 135),  # height=240, width=135
}
```

**Aspect ratio:** 405 ÷ 720 = 0.5625 (this is 9:16, portrait) ❌

**Visual representation:**
```
┌────┐
│    │  720 pixels tall
│    │  405 pixels wide
│    │  ← Portrait (taller than wide)
└────┘
```

## After (CORRECT - 16:9 Landscape)

```python
'720_169': {
    'gt_size': (405, 720),  # height=405, width=720
    'lr_size': (135, 240),  # height=135, width=240
}
```

**Aspect ratio:** 720 ÷ 405 = 1.778 (this is 16:9, landscape) ✅

**Visual representation:**
```
┌──────────────┐
│              │  405 pixels tall
│              │  720 pixels wide
└──────────────┘
← Landscape (wider than tall)
```

## Changes Made

### File: `dataset_generator_v2/utils/format_definitions.py`

**Format: `'720_169'`**
- `gt_size`: (720, 405) → (405, 720)
- `lr_size`: (240, 135) → (135, 240)

**Format: `'medium_169'` (legacy name)**
- `gt_size`: (720, 405) → (405, 720)
- `lr_size`: (240, 135) → (135, 240)

## Verification

### Aspect Ratios

| Format | GT Size | LR Size | Aspect Ratio | Expected | Status |
|--------|---------|---------|--------------|----------|--------|
| 540 | 540×540 | 180×180 | 1.000 (1:1) | 1:1 | ✅ |
| **720_169** | **405×720** | **135×240** | **1.778 (16:9)** | **16:9** | ✅ |
| 720 | 720×720 | 240×240 | 1.000 (1:1) | 1:1 | ✅ |

### Vertical Stacking (7 Frames)

With vertical stacking, 7 frames are stacked on top of each other:

| Format | Single LR | Stacked LR (7 frames) | Aspect Ratio |
|--------|-----------|----------------------|--------------|
| 540 | 180×180 | 1260×180 | 1:1 |
| **720_169** | **135×240** | **945×240** | **16:9** ✅ |
| 720 | 240×240 | 1680×240 | 1:1 |

### Mathematical Verification

```
16:9 ratio = 16 ÷ 9 = 1.777...

Our 720_169 format:
  Width ÷ Height = 720 ÷ 405 = 1.777... ✅

Scale factor (GT to LR): 3×
  GT: 405×720
  LR: 135×240 (each dimension ÷ 3) ✅

Vertical stacking (7 frames):
  Single LR: 135×240
  Stacked: 945×240 (135 × 7 = 945) ✅
```

## Dimension Comparison

### Before vs After

| Aspect | Before (Wrong) | After (Correct) |
|--------|----------------|-----------------|
| GT height | 720 | 405 |
| GT width | 405 | 720 |
| GT aspect | 9:16 (0.56) | 16:9 (1.78) ✅ |
| LR height | 240 | 135 |
| LR width | 135 | 240 |
| LR aspect | 9:16 (0.56) | 16:9 (1.78) ✅ |
| LR stacked | 1680×135 | 945×240 |
| Stacked aspect | 9:16 (0.56) | 16:9 (1.78) ✅ |

## All Formats Summary

### 540 Patches (1:1 Square)
- **GT**: 540×540 (height×width)
- **LR**: 180×180 (height×width)
- **LR Stacked (7 frames)**: 1260×180 (height×width)
- **Aspect Ratio**: 1:1 ✅

### 720_169 Patches (16:9 Landscape)
- **GT**: 405×720 (height×width)
- **LR**: 135×240 (height×width)
- **LR Stacked (7 frames)**: 945×240 (height×width)
- **Aspect Ratio**: 16:9 (1.778) ✅

### 720 Patches (1:1 Square)
- **GT**: 720×720 (height×width)
- **LR**: 240×240 (height×width)
- **LR Stacked (7 frames)**: 1680×240 (height×width)
- **Aspect Ratio**: 1:1 ✅

## Files Modified

1. **dataset_generator_v2/utils/format_definitions.py**
   - Fixed dimensions for `'720_169'` format
   - Fixed dimensions for `'medium_169'` format (legacy name)

## Commit

`4ec6f24` - Fix 16:9 aspect ratio dimensions (swap height and width)

## Result

✅ The 16:9 format now creates **landscape** (wider than tall) patches instead of portrait  
✅ All aspect ratios are mathematically correct  
✅ Vertical LR stacking maintains correct aspect ratio  
✅ All other features preserved (GUI, priority system, resume, keyboard controls, etc.)

## Visual Comparison

```
BEFORE (Wrong - Portrait):          AFTER (Correct - Landscape):
┌────┐                              ┌──────────────┐
│    │  720h × 405w                 │              │  405h × 720w
│    │  9:16 ratio                  │              │  16:9 ratio
│    │  ❌                            └──────────────┘  ✅
└────┘
```
