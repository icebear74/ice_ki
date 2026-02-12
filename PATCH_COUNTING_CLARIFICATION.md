# Patch Counting Clarification

## User Question
> "when i say, 200.000 patches for a category .. it means 200.000 in GT ? or 200.000 overall (so div by 7 in GT?)"

## Current Implementation

### What is a "Patch"?

1. **Extraction Phase:**
   - Extract 7 consecutive frames from a timestamp
   - Example: frames 100, 101, 102, 103, 104, 105, 106

2. **Patch Creation Phase:**
   - Combine those 7 frames into 1 multi-frame patch
   - Create GT (Ground Truth) version: 1920×1080 (7 frames)
   - Create LR (Low Resolution) version: downscaled (7 frames)

3. **File Output:**
   - 1 GT file (PNG containing the 7-frame patch)
   - 1 LR file (PNG containing the downscaled 7-frame patch)

### Current Counting: 1 Patch = 1 GT File

**When you say "200,000 patches for master":**
- Creates: **200,000 GT files**
- Creates: **200,000 LR files** (companion files)
- **Total: 400,000 files**
- Each file contains 7 frames

**Code:**
```python
saved, gt_path, lr_path = self._save_patch_pair(gt, lr, ...)
if saved:
    patches_created[category] += 1  # Counts 1 patch = 1 GT+LR pair
```

### The "Divide by 7" Question

You mentioned "div by 7" - this might refer to:

**Scenario A: Frames → Patches**
- Extract 7 frames from each scene
- Those 7 frames = 1 patch
- So 1,400,000 frames ÷ 7 = 200,000 patches ✓ (This is what we do)

**Scenario B: Total Files → GT Files?**
- If 200,000 = total files (GT + LR combined)
- Then GT files = 200,000 ÷ 2 = 100,000
- But this seems unlikely...

## Examples

### Example 1: Current Interpretation (200k = GT count)

**Configuration:**
```json
"category_targets": {
    "master": 200000
}
```

**Result:**
- GT files created: 200,000
- LR files created: 200,000
- Total PNG files: 400,000
- Each file contains 7 frames
- Total frames extracted: 200,000 × 7 = 1,400,000 frames

**Storage:**
```
master/
├── large_1080/
│   ├── GT/
│   │   ├── video1_00010000.png  (7 frames, 1920×1080)
│   │   ├── video1_00020000.png
│   │   └── ... (100,000 files)
│   └── LR_7frames/
│       ├── video1_00010000.png  (7 frames, downscaled)
│       └── ... (100,000 files)
├── medium_720/
│   └── ... (50,000 GT + 50,000 LR)
└── small_540/
    └── ... (50,000 GT + 50,000 LR)

Total: 200,000 GT + 200,000 LR = 400,000 files
```

### Example 2: Alternative Interpretation (200k = total files?)

If you meant 200,000 TOTAL files including both GT and LR:

**Configuration would need to be:**
```json
"category_targets": {
    "master": 100000  // Half of desired total
}
```

**Result:**
- GT files: 100,000
- LR files: 100,000
- Total: 200,000 files
- Each file contains 7 frames

## Comparison

| Interpretation | Target Value | GT Files | LR Files | Total Files |
|---------------|--------------|----------|----------|-------------|
| **Current (A)** | 200,000 | 200,000 | 200,000 | 400,000 |
| Alternative (B) | 200,000 | 100,000 | 100,000 | 200,000 |

## VSR Training Perspective

For VSR (Video Super Resolution) training:
- You train on **GT-LR pairs**
- Each pair is 1 training sample
- So 200,000 patches = 200,000 training samples

**Current implementation gives you 200,000 training samples** ✓

## Recommendation

**I believe the current interpretation (A) is correct:**

1. **Intuitive**: "200,000 patches" = "200,000 training samples"
2. **Standard**: Each GT-LR pair is counted as 1 patch
3. **Practical**: Matches VSR training expectations
4. **Clear**: Target = number of patches you actually get

**But please confirm:**
- Do you want 200,000 GT files (current)?
- Or do you want 200,000 total files (GT+LR combined, so 100k GT)?

## Current Status

**Implementation:** 200,000 = GT count (also creates 200k LR files)

**If this is wrong**, I can easily change it to:
- Count total files instead
- Divide targets by 2 internally
- So 200,000 target → 100k GT + 100k LR

**Awaiting your confirmation!** 🎯
