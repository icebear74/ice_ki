# VSRDataset Update Summary - 7-Frame Horizontal Stacking Support

## ✅ Changes Completed

### File Updated: `vsr_plusplus_NEU/core/dataset.py`

The VSRDataset class has been updated to support the new dataset structure with 7-frame horizontal stacking.

---

## 📋 Key Changes

### 1. **Updated Docstring**
- Now documents 7-frame horizontal stacking structure
- Reflects new path structure: `root/dataset_name/patches/{size_key}/GT` and `LR`
- Documents variable GT sizes based on size_key

### 2. **New __init__ Signature**
```python
# OLD:
VSRDataset(dataset_root, mode='Patches', augment=True)

# NEW:
VSRDataset(root, dataset_name='master', size_key='720', mode='train', augment=True)
```

**Parameters:**
- `root`: Root directory (e.g., `/mnt/data/training/datasetNeu`)
- `dataset_name`: Dataset name (e.g., `'master'`)
- `size_key`: Size variant - `'720'`, `'540'`, or `'720_169'`
- `mode`: `'train'` or `'val'` (replaces old 'Patches'/'Val')
- `augment`: Whether to apply augmentations

### 3. **Path Construction**

**Training Mode (`mode='train'`):**
```
root/dataset_name/patches/size_key/GT/
root/dataset_name/patches/size_key/LR/
```

**Validation Mode (`mode='val'`):**
```
GT: root/dataset_name/val/size_key/GT/
LR: root/dataset_name/val/size_key/LR/  (with fallback to patches/size_key/LR/)
```

### 4. **Frame Splitting - Horizontal (7 frames)**

**OLD:** Vertical stacking (5 frames)
- LR shape: `(900, 180, 3)` = 5 frames × 180px height
- Split: `lr[i*180:(i+1)*180, :, :]`

**NEW:** Horizontal stacking (7 frames)
- LR shape: `(H, W×7, 3)` where W depends on size_key
- Split: `lr[:, i*W:(i+1)*W, :]`
- Example for 720p: `(240, 2986, 3)` → 7 frames of `(240, 426, 3)`

### 5. **Shape Validation**

Expected shapes per size_key:

| size_key | GT Shape | LR Height | LR Width/Frame | LR Total Width |
|----------|----------|-----------|----------------|----------------|
| '720' | (720, 1280, 3) | 240 | 426 | 2986 (7×426) |
| '540' | (540, 960, 3) | 180 | 320 | 2240 (7×320) |
| '720_169' | (720, 1280, 3) | 240 | 426 | 2986 (7×426) |

### 6. **Return Format (Unchanged)**

```python
lr_stack: [7, 3, H, W]  # 7 LR frames instead of 5
gt: [3, H*3, W*3]       # GT frame (3x upscale)
```

---

## 🔄 Migration Guide for Existing Code

### Example: Updating `train.py`

**OLD CODE:**
```python
DATASET_ROOT = "/mnt/data/training/Dataset/Universal/Mastermodell"
train_dataset = VSRDataset(DATASET_ROOT, mode='Patches', augment=True)
val_dataset = VSRDataset(DATASET_ROOT, mode='Val', augment=False)
```

**NEW CODE:**
```python
# Option 1: Using default 'master' dataset and '720' size
ROOT = "/mnt/data/training/datasetNeu"
train_dataset = VSRDataset(
    root=ROOT,
    dataset_name='master',
    size_key='720',
    mode='train',
    augment=True
)
val_dataset = VSRDataset(
    root=ROOT,
    dataset_name='master',
    size_key='720',
    mode='val',
    augment=False
)

# Option 2: For 540p training
train_dataset = VSRDataset(
    root=ROOT,
    dataset_name='master',
    size_key='540',
    mode='train',
    augment=True
)
```

---

## ✅ Verification

The update has been verified for:

1. ✅ Correct parameter signature
2. ✅ Horizontal frame splitting (7 frames)
3. ✅ Path construction for train and val modes
4. ✅ Shape validation for different size_keys
5. ✅ Python syntax correctness
6. ✅ Augmentation logic applied to all 7 frames

---

## 📝 Files That May Need Updates

The following files use `VSRDataset` and may need migration:

1. `vsr_plusplus_NEU/train.py` - **Main training script**
   - Update dataset initialization calls
   - Update config to include new parameters

2. `vsr_plusplus_NEU/config_p4_optimized.py` - **Config file**
   - Add `DATASET_NAME` parameter (default: 'master')
   - Add `SIZE_KEY` parameter (default: '720')
   - Update `DATA_ROOT` to point to new structure

---

## 🎯 Next Steps

To use the updated dataset:

1. **Update config file** to add new parameters:
   ```python
   DATASET_NAME = "master"
   SIZE_KEY = "720"  # or '540', '720_169'
   ```

2. **Update train.py** dataset initialization:
   ```python
   train_dataset = VSRDataset(
       root=config.get('DATASET_ROOT'),
       dataset_name=config.get('DATASET_NAME', 'master'),
       size_key=config.get('SIZE_KEY', '720'),
       mode='train',
       augment=True
   )
   val_dataset = VSRDataset(
       root=config.get('DATASET_ROOT'),
       dataset_name=config.get('DATASET_NAME', 'master'),
       size_key=config.get('SIZE_KEY', '720'),
       mode='val',
       augment=False
   )
   ```

3. **Ensure dataset structure** matches expected paths:
   ```
   /mnt/data/training/datasetNeu/
   └── master/
       ├── patches/
       │   ├── 720/
       │   │   ├── GT/
       │   │   └── LR/
       │   ├── 540/
       │   │   ├── GT/
       │   │   └── LR/
       │   └── 720_169/
       │       ├── GT/
       │       └── LR/
       └── val/
           ├── 720/
           │   ├── GT/
           │   └── LR/  (optional, falls back to patches)
           └── 540/
               └── GT/
   ```

---

## 🧪 Testing

Run basic verification:
```bash
python -c "
from vsr_plusplus_NEU.core.dataset import VSRDataset
# Test initialization (will fail if paths don't exist, but tests API)
try:
    ds = VSRDataset(
        root='/mnt/data/training/datasetNeu',
        dataset_name='master',
        size_key='720',
        mode='train'
    )
    print(f'✓ Dataset initialized: {len(ds)} samples')
except Exception as e:
    print(f'Dataset init test: {e}')
"
```

---

## 📚 Reference

- **Frame count:** Changed from 5 to 7 frames
- **Stacking direction:** Changed from vertical to horizontal
- **API parameters:** Now uses `root`, `dataset_name`, `size_key`, `mode`
- **Supported sizes:** '720' (720p), '540' (540p), '720_169' (16:9 720p)
