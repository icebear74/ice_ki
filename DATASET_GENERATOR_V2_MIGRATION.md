# Dataset Generator V2 Migration Guide

## Overview

This guide covers the complete migration from the old dataset generator to the new V2 system with 7-frame support and simplified directory structure.

## Key Changes

### 1. Directory Structure

**Old Structure:**
```
/mnt/data/training/dataset/
├── train/
│   ├── 5frames/
│   │   ├── small_540/
│   │   │   ├── GT/
│   │   │   └── LR/
│   │   ├── medium_169/
│   │   └── large_720/
│   └── 7frames/ (if enabled)
└── Val/
```

**New Structure:**
```
/mnt/data/training/datasetNeu/
└── master/                    # Dataset name (configurable)
    ├── patches/               # Training data
    │   ├── 720/              # Size key (was "large_720")
    │   │   ├── GT/
    │   │   └── LR/
    │   ├── 540/              # Size key (was "small_540")
    │   │   ├── GT/
    │   │   └── LR/
    │   └── 720_169/          # Size key (was "medium_169")
    │       ├── GT/
    │       └── LR/
    └── val/                   # Validation data
        ├── 720/
        │   └── GT/           # LR fetched from patches/720/LR
        └── 720_169/
            └── GT/
```

### 2. Size Key Changes

| Old Name | New Name | Description |
|----------|----------|-------------|
| `small_540` | `540` | 540×540 patches |
| `medium_169` | `720_169` | 720×405 (16:9) patches |
| `large_720` | `720` | 720×720 patches |

### 3. Frame Count

- **Old:** Supported both 5-frame and 7-frame
- **New:** **ONLY 7-frame** support

### 4. LR Stacking Direction

- **Old:** Vertical stacking (5 frames × H)
  - Shape: `(900, 180, 3)` for 540 patches
- **New:** Horizontal stacking (W × 7 frames)
  - Shape: `(180, 1260, 3)` for 540 patches

## Migration Steps

### Step 1: Update Generator Config

Create new config file `dataset_generator_v2/generator_config_v2.json`:

```json
{
  "dataset_name": "master",
  "root_path": "/mnt/data/training/datasetNeu",
  
  "source": {
    "video_dir": "/mnt/data/training/source_videos",
    "extensions": [".mp4", ".mkv", ".avi"]
  },
  
  "output_patches": {
    "720": {
      "gt_size": [720, 720],
      "lr_size": [240, 240],
      "enabled": true
    },
    "540": {
      "gt_size": [540, 540],
      "lr_size": [180, 180],
      "enabled": true
    },
    "720_169": {
      "gt_size": [720, 405],
      "lr_size": [240, 135],
      "enabled": true
    }
  },
  
  "processing": {
    "n_frames": 7,
    "scale": 3,
    "stride": 3,
    "min_scene_length": 21,
    "scene_threshold": 30.0
  },
  
  "quality": {
    "jpeg_quality": 95,
    "min_sharpness": 30.0,
    "blur_threshold": 100.0
  },
  
  "workers": 8,
  "batch_size": 4
}
```

### Step 2: Run New Generator

```bash
cd dataset_generator_v2
python3 make_dataset_v2_clean.py
```

### Step 3: Update Trainer Config

Create `vsr_plusplus_NEU/runtime_config.json`:

```json
{
  "data": {
    "root": "/mnt/data/training/datasetNeu",
    "dataset_name": "master"
  },
  
  "model": {
    "n_frames": 7,
    "n_feats": 72,
    "n_blocks": 26,
    "precision": "float32"
  },
  
  "training": {
    "effective_batch_size": 6,
    "adaptive_batch": {
      "720": {"batch": 1, "accum": 6},
      "540": {"batch": 1, "accum": 6},
      "720_169": {"batch": 1, "accum": 6}
    }
  },
  
  "size_distribution": {
    "720": 0.0,
    "540": 0.65,
    "720_169": 0.35
  },
  
  "validation": {
    "sizes": ["720", "720_169"],
    "batch_size": 1
  }
}
```

### Step 4: Update Code References

All code using old size keys needs to be updated:

```python
# Old code
size_key = "small_540"

# New code
size_key = "540"
```

### Step 5: Run Training

The trainer automatically detects `runtime_config.json` and uses multi-size training:

```bash
cd vsr_plusplus_NEU
python3 train.py
```

## Backward Compatibility

The system maintains backward compatibility:

- **If `runtime_config.json` exists:** Uses new multi-size system
- **If `runtime_config.json` missing:** Falls back to old single-size system

## Validation Dataset Setup

The new system expects validation GT in `val/{size}/GT/`:

```bash
# Copy validation images
cp /path/to/val/images/* /mnt/data/training/datasetNeu/master/val/720/GT/
cp /path/to/val/images/* /mnt/data/training/datasetNeu/master/val/720_169/GT/
```

LR pairs are automatically fetched from `patches/{size}/LR/` during validation.

## Troubleshooting

### Issue: "No matching LR files"
- **Cause:** Validation GT doesn't have corresponding LR in patches/
- **Solution:** Ensure all validation GT filenames match patches/ filenames

### Issue: "Invalid LR shape"
- **Cause:** Expecting old vertical stacking instead of new horizontal
- **Solution:** Regenerate dataset with new generator

### Issue: "Size key not found"
- **Cause:** Using old size keys (small_540, etc.)
- **Solution:** Update all references to new keys (540, 720_169, 720)

## Testing

After migration, verify:

1. **Generator creates correct structure:**
   ```bash
   ls /mnt/data/training/datasetNeu/master/patches/
   # Should show: 720/ 540/ 720_169/
   ```

2. **LR images are horizontal:**
   ```bash
   python3 -c "import cv2; img = cv2.imread('...LR/image.png'); print(img.shape)"
   # Should show: (180, 1260, 3) for 540 patches
   ```

3. **Trainer loads multi-size:**
   ```bash
   cd vsr_plusplus_NEU
   python3 train.py
   # Should show: "Multi-size training enabled"
   ```

## Success Criteria

✅ Generator creates new directory structure  
✅ Only 7-frame LR images created  
✅ LR images are horizontally stacked  
✅ Trainer loads from new structure  
✅ Size tracking uses new keys  
✅ All configs updated  
✅ Backward compatibility maintained  

## Files Updated

### Generator
- `dataset_generator_v2/generator_config_v2.json` (new)
- `dataset_generator_v2/make_dataset_v2_clean.py` (new)
- `dataset_generator_v2/make_dataset_multi.py` (bug fix)

### Trainer
- `vsr_plusplus_NEU/core/dataset.py` (updated for new structure)
- `vsr_plusplus_NEU/core/dataloader.py` (new multi-size loader)
- `vsr_plusplus_NEU/training/trainer.py` (multi-size batch support)
- `vsr_plusplus_NEU/train.py` (auto-detect multi-size)
- `vsr_plusplus_NEU/runtime_config.json` (new)

### Size Tracking
- `vsr_plusplus_NEU/systems/size_tracking.py` (new keys)
- `vsr_plusplus_NEU/systems/runtime_config.py` (new keys)
- `vsr_plusplus_NEU/utils/ui_terminal.py` (new keys)

### Tests
- `vsr_plusplus_NEU/test_7frame_system.py` (new keys)
- `test_batch_compatibility.py` (new)

## Summary

The V2 system provides:
- **Cleaner structure:** Flat patches/ directory, no nested train/
- **Simpler keys:** 540, 720, 720_169 instead of small_540, etc.
- **7-frame only:** No 5-frame confusion
- **Multi-size training:** Train on multiple resolutions simultaneously
- **Backward compatibility:** Works with old single-size system too

For questions or issues, refer to:
- `DATASET_UPDATE_SUMMARY.md` - Dataset changes
- `MULTI_SIZE_BATCH_SUPPORT.md` - Trainer changes
