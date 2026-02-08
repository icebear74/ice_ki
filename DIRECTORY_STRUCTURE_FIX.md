# Directory Structure Fix - Summary

## Problem Identified

The dataset generator was creating the old nested directory structure:
```
/mnt/data/training/datasetNeu/
├── Master/MasterModel/Learn/
│   ├── Patches/
│   │   └── small_540/
│   ├── Patches_Medium169/
│   │   └── medium_169/
│   └── Patches_Large/
│       └── large_720/
```

But the requirement specified a new flat structure:
```
/mnt/data/training/datasetNeu/
└── master/
    ├── patches/
    │   ├── 720/GT+LR
    │   ├── 540/GT+LR
    │   └── 720_169/GT+LR
    └── val/
        ├── 720/GT
        ├── 540/GT
        └── 720_169/GT
```

## Root Causes

1. **`CATEGORY_PATHS`** used deeply nested paths:
   - Old: `'master': 'Master/MasterModel/Learn'`
   - New: `'master': 'master'`

2. **`FORMATS` output_dir** used old directory names:
   - Old: `'output_dir': 'Patches'`, `'Patches_Large'`, etc.
   - New: `'output_dir': 'patches/540'`, `'patches/720'`, etc.

3. **Format names** were inconsistent:
   - Old: `small_540`, `medium_169`, `large_720`
   - Needed: `540`, `720`, `720_169` (with legacy support)

## Solution Implemented

### 1. Updated `format_definitions.py`

**Category Paths:**
```python
CATEGORY_PATHS = {
    'master': 'master',      # was: 'Master/MasterModel/Learn'
    'universal': 'universal', # was: 'Universal/UniversalModel/Learn'
    'space': 'space',        # was: 'Space/SpaceModel/Learn'
    'toon': 'toon'           # was: 'Toon/ToonModel/Learn'
}
```

**Format Definitions:**
```python
FORMATS = {
    # New size keys
    '540': {
        'gt_size': (540, 540),
        'lr_size': (180, 180),
        'output_dir': 'patches/540',  # was: 'Patches'
        ...
    },
    '720_169': {
        'gt_size': (720, 405),
        'lr_size': (240, 135),
        'output_dir': 'patches/720_169',  # was: 'Patches_Medium169'
        ...
    },
    '720': {
        'gt_size': (720, 720),
        'lr_size': (240, 240),
        'output_dir': 'patches/720',  # was: 'Patches_Large'
        ...
    },
    # Legacy format names for backward compatibility
    'small_540': { ... maps to 'patches/540' ... },
    'medium_169': { ... maps to 'patches/720_169' ... },
    'large_720': { ... maps to 'patches/720' ... }
}
```

### 2. Updated `make_dataset_multi.py`

**Validation Directory Structure:**
```python
# Old:
'val_gt': f"{self.base_dir}/{category_path}/Val/GT"

# New:
'val_gt': f"{self.base_dir}/{category_path}/val/{size}/GT"
```

## Result

### Directory Structure for Each Category

For category `master`, the generator now creates:

```
/mnt/data/training/datasetNeu/
└── master/
    ├── patches/
    │   ├── 540/
    │   │   ├── GT/        # 540×540 ground truth
    │   │   └── LR/        # 180×1260 (7-frame horizontal)
    │   ├── 720_169/
    │   │   ├── GT/        # 720×405 ground truth (16:9)
    │   │   └── LR/        # 240×945 (7-frame horizontal)
    │   └── 720/
    │       ├── GT/        # 720×720 ground truth
    │       └── LR/        # 240×1680 (7-frame horizontal)
    └── val/
        ├── 540/
        │   └── GT/        # User copies validation images here
        ├── 720_169/
        │   └── GT/
        └── 720/
            └── GT/
```

Same structure applies to `universal`, `space`, and `toon` categories.

## Backward Compatibility

The existing `generator_config.json` continues to work because:
- Legacy format names (`small_540`, `medium_169`, `large_720`) are mapped to new structure
- All existing video configurations remain valid
- Categories are still used as defined

## LR Image Dimensions

With 7-frame horizontal stacking:
- **540 patches**: GT (540×540) → LR (180×1260) = 180 height × (180×7) width
- **720_169 patches**: GT (720×405) → LR (240×945) = 240 height × (135×7) width  
- **720 patches**: GT (720×720) → LR (240×1680) = 240 height × (240×7) width

## Files Modified

1. `dataset_generator_v2/utils/format_definitions.py`
   - Updated `FORMATS` with new size keys and output directories
   - Simplified `CATEGORY_PATHS` to flat structure
   - Added legacy format name support

2. `dataset_generator_v2/make_dataset_multi.py`
   - Updated validation directory paths
   - Updated comments to reflect new structure

## Testing

Run the verification script to see the full structure:
```bash
cd /home/runner/work/ice_ki/ice_ki/dataset_generator_v2
python3 verify_structure.py
```

## Next Steps

1. Test generator with actual video processing
2. Verify directories are created correctly
3. Check that existing config files work
4. Confirm LR stacking dimensions are correct
