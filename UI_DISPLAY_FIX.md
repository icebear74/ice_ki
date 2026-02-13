# UI Display Fix for 7-Frame FusionBlock Activities

## Issue

Training crashed with `TypeError` when trying to display layer activities in the UI:

```
TypeError: float() argument must be a string or a real number, not 'list'
  File "vsr_plusplus_NEU/utils/ui_display.py", line 141
    activities_with_names.append(("Backward Fuse", float(backward_fuse) ...))
```

## Root Cause

The 7-frame model's `FusionBlock` tracks activities for **two separate convolutions**:
- `conv3x3`: spatial context (3x3 convolution)
- `conv1x1`: feature gating (1x1 convolution)

The `get_layer_activity()` method returns:
```python
{
    'backward_fuse': [conv3x3_activity, conv1x1_activity],  # LIST of 2 values
    'forward_fuse': [conv3x3_activity, conv1x1_activity],
    'fusion': [conv3x3_activity, conv1x1_activity]
}
```

But the UI code expected single float values (from old TrackedConv2d implementation):
```python
activities_with_names.append(("Backward Fuse", float(backward_fuse)))
# ❌ Fails when backward_fuse is a list!
```

## Solution

Added a helper function `add_fusion_activity()` that intelligently handles both formats:

```python
def add_fusion_activity(name, activity):
    """Add fusion layer activity - handles both FusionBlock (list) and TrackedConv2d (float)"""
    if isinstance(activity, list):
        if len(activity) == 2:
            # 7-frame FusionBlock: [conv3x3_act, conv1x1_act]
            activities_with_names.append((f"{name} 3x3", float(activity[0])))
            activities_with_names.append((f"{name} 1x1", float(activity[1])))
        elif len(activity) > 0:
            # Unexpected length - use average
            avg = sum(activity) / len(activity)
            activities_with_names.append((name, avg))
        else:
            # Empty list
            activities_with_names.append((name, 0.0))
    elif activity is not None:
        # Old TrackedConv2d: single float
        activities_with_names.append((name, float(activity)))
    else:
        # None
        activities_with_names.append((name, 0.0))
```

## Benefits

### 1. Shows Both Conv Layers Separately

**Before (would crash):**
```
Backward Fuse: ???
```

**After (works perfectly):**
```
Backward Fuse 3x3: 0.40  ← Spatial context activity
Backward Fuse 1x1: 0.50  ← Gating activity
```

This gives **better visibility** into how each part of the FusionBlock is performing!

### 2. Backward Compatible

Still works with old models that return single float values:
```python
# Old TrackedConv2d format
{'backward_fuse': 0.4}  → "Backward Fuse: 0.40"

# New FusionBlock format
{'backward_fuse': [0.4, 0.5]}  → "Backward Fuse 3x3: 0.40"
                                  "Backward Fuse 1x1: 0.50"
```

### 3. Robust Edge Case Handling

- `None` → 0.0
- Empty list `[]` → 0.0
- Unexpected lengths → average value

## UI Display Result

The training UI now shows **8 fusion activities** instead of 3:

**7-Frame Model (FusionBlock):**
```
Backward 1-14      ← Residual blocks
Backward Fuse 3x3  ← New!
Backward Fuse 1x1  ← New!
Forward 1-14       ← Residual blocks
Forward Fuse 3x3   ← New!
Forward Fuse 1x1   ← New!
Final Fusion 3x3   ← New!
Final Fusion 1x1   ← New!
```

**Old Model (TrackedConv2d):**
```
Backward 1-N
Backward Fuse      ← Single entry
Forward 1-N
Forward Fuse       ← Single entry
Final Fusion       ← Single entry
```

## Testing

All test cases passed:
- ✅ 7-frame FusionBlock with list `[0.4, 0.5]`
- ✅ Old TrackedConv2d with float `0.4`
- ✅ Edge cases: `None`, `[]`, `[0.5]`, `[0.1, 0.2, 0.3]`

## Files Modified

- `vsr_plusplus_NEU/utils/ui_display.py` - Updated `get_activity_data()` function

## Status

✅ **FIXED** - Training UI now works with 7-frame model FusionBlock activities!
