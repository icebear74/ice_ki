# LR Stacking Direction Fix

## Problem (German)
"das Stacking ist falsch rum :( Es muss hochkant gestacked werden nicht seitlich"

**Translation:** "the stacking is the wrong way around :( It must be stacked vertically not horizontally"

## Issue
The LR frames were being stacked horizontally (side-by-side) but needed to be stacked vertically (top-to-bottom, "hochkant" in German).

## Solution

Changed from horizontal concatenation to vertical concatenation:

### Code Change
```python
# Before (WRONG):
cv2.hconcat(lr_frames)  # Horizontal stacking

# After (CORRECT):
cv2.vconcat(lr_frames)  # Vertical stacking
```

### Shape Change

**Before (Horizontal):**
- Shape: `(H, W×7, 3)`
- Example for 540 patches: `(180, 1260, 3)` - 180 height × 1260 width
- Frames arranged: `[F1][F2][F3][F4][F5][F6][F7]` (side-by-side)

**After (Vertical):**
- Shape: `(H×7, W, 3)`
- Example for 540 patches: `(1260, 180, 3)` - 1260 height × 180 width
- Frames arranged:
  ```
  [F1]
  [F2]
  [F3]
  [F4]  ← GT middle frame
  [F5]
  [F6]
  [F7]
  ```

## Dimension Examples

### 540 Patches
- **GT**: 540×540
- **LR Before**: 180×1260 (horizontal)
- **LR After**: 1260×180 (vertical) ✅

### 720_169 Patches
- **GT**: 720×405
- **LR Before**: 240×945 (horizontal)
- **LR After**: 945×240 (vertical) ✅

### 720 Patches
- **GT**: 720×720
- **LR Before**: 240×1680 (horizontal)
- **LR After**: 1680×240 (vertical) ✅

## Files Modified

1. **dataset_generator_v2/make_dataset_multi.py**
   - Changed `cv2.hconcat()` → `cv2.vconcat()` in `create_lr_stack()` method
   - Updated all docstrings and comments:
     - "horizontally" → "vertically"
     - "width × 7" → "height × 7"
     - "(H, W×7)" → "(H×7, W)"
     - "(180, 1260)" → "(1260, 180)"

## Visual Representation

```
HORIZONTAL (Wrong):          VERTICAL (Correct):
┌─┬─┬─┬─┬─┬─┬─┐              ┌─┐
│1│2│3│4│5│6│7│              │1│
└─┴─┴─┴─┴─┴─┴─┘              ├─┤
                             │2│
Width = W × 7                ├─┤
Height = H                   │3│
                             ├─┤
                             │4│  ← GT (middle)
                             ├─┤
                             │5│
                             ├─┤
                             │6│
                             ├─┤
                             │7│
                             └─┘
                             
                             Width = W
                             Height = H × 7
```

## All Features Preserved

✅ GUI with Rich progress bars  
✅ Priority system (0-255, default 255)  
✅ Keyboard controls (Space, +/-, q)  
✅ Resume/checkpoint functionality  
✅ Scene handling (no skipping at cuts)  
✅ FFmpeg HDR tonemapping  
✅ New flat directory structure (master/patches/720/)  

## Commit

`ec89906` - Fix LR stacking: change from horizontal to vertical (hochkant)

## Result

LR frames are now correctly stacked vertically (hochkant) as required. The 7 frames are stacked on top of each other rather than side-by-side, creating tall narrow images instead of wide short images.
