# Size Distribution Removed - Auto-Detection Implemented

## Summary

**User Request:** "remove size distribution from json. its not longer used ?!"

**Status:** ✅ Complete

The `size_distribution` configuration has been completely removed. Training now automatically detects which dataset sizes are available by scanning the filesystem.

---

## Why Was It Removed?

### Original Purpose (Now Obsolete)

`size_distribution` was originally used for two purposes:
1. **Determine which sizes to load** (e.g., only load 540 and 720_169, skip 720)
2. **Weight sampling during training** (e.g., sample 40% from 720, 40% from 540, 20% from 720_169)

### After Weighting Removal

After we removed the weighting logic (because files are pre-weighted during extraction), `size_distribution` only served purpose #1 - determining which sizes to load.

But this created confusion:
- Users thought it still controlled training ratio
- Values had to sum to 1.0 (validation)
- Documentation keys (`_NOTE`, `_EXPLANATION`) required filtering
- Redundant with actual file existence

### User's Insight

The user correctly identified that `size_distribution` was no longer necessary. If files are already pre-weighted during extraction, why maintain a separate config to say which sizes exist? Just check the filesystem!

---

## New Behavior

### Auto-Detection on Startup

Training now scans the filesystem to detect available sizes:

```python
def detect_available_sizes(data_root, dataset_name):
    """Detect which dataset sizes are available by checking directories."""
    available = []
    for size_key in ['540', '720', '720_169']:
        train_dir = os.path.join(data_root, dataset_name, 'train', size_key, 'GT')
        if os.path.exists(train_dir):
            files = [f for f in os.listdir(train_dir) if f.endswith('.png')]
            if files:
                available.append((size_key, len(files)))
    return available
```

### Startup Output

```
Found 1,456 files for size 540
Found 1,234 files for size 720
Found 753 files for size 720_169
✓ Multi-size training enabled: 540, 720, 720_169
```

### Multi-Size vs Single-Size

- **Multiple sizes with files** → Multi-size training automatically enabled
- **Single size with files** → Single-size training
- **No files found** → Fallback to defaults (540)

---

## Configuration Changes

### Before (runtime_config.json)

```json
{
  "size_distribution": {
    "_NOTE": "WICHTIG: Diese Werte bestimmen nur WELCHE Sizes geladen werden...",
    "_EXPLANATION": "If 720=0.4, 540=0.4, 720_169=0.2...",
    "720": 0.4,
    "540": 0.4,
    "720_169": 0.2
  },
  "validation": {
    "sizes": ["720", "720_169"]
  }
}
```

### After (runtime_config.json)

```json
{
  "training": {
    "_NOTE": "Training automatically uses ALL available dataset sizes found on disk. File counts determine the training ratio (pre-weighted during extraction).",
    "effective_batch_size": 6,
    "adaptive_batch": {
      "720": {"batch": 1, "accum": 6},
      "540": {"batch": 1, "accum": 6},
      "720_169": {"batch": 1, "accum": 6}
    }
  },
  "validation": {
    "_NOTE": "Specify which sizes to use for validation. Training will auto-detect available sizes from disk.",
    "sizes": ["720", "720_169"]
  }
}
```

---

## Code Changes

### 1. train.py

**Added:**
- `detect_available_sizes()` function
- Auto-detection on startup
- File count reporting

**Removed:**
- All `size_distribution` references
- Distribution filtering logic
- Documentation key filtering

### 2. dataloader.py

**Before:**
```python
# Filter to only active size keys (those with non-zero distribution)
self.active_sizes = [k for k, v in size_distribution.items() 
                   if not k.startswith('_') and isinstance(v, (int, float)) and v > 0]
```

**After:**
```python
# Use all sizes from datasets_dict (size_distribution no longer filters)
self.active_sizes = list(datasets_dict.keys())
```

### 3. runtime_config.py

**Removed:**
- `size_distribution` from `DEFAULT_CONFIG`
- Size distribution sum validation
- Distribution-related validation logic

---

## How to Control Training Sizes

### Option 1: File Presence (Automatic)

**Want to train on 540 only?**
- Put files in: `data_root/dataset_name/train/540/`
- Leave empty: `data_root/dataset_name/train/720/`
- Leave empty: `data_root/dataset_name/train/720_169/`

Result: Single-size training on 540 ✓

**Want to train on all sizes?**
- Put files in all: `540/`, `720/`, `720_169/`

Result: Multi-size training on all ✓

### Option 2: Validation Sizes (Explicit)

Validation still uses explicit configuration:

```json
"validation": {
  "sizes": ["720", "720_169"]
}
```

If not specified, falls back to detected training sizes.

---

## Benefits

### Simpler Configuration

**Before:**
- Configure which sizes to load: `size_distribution`
- Configure validation sizes: `validation.sizes`
- Ensure distribution sums to 1.0
- Add documentation keys with `_` prefix
- Filter out documentation keys in code

**After:**
- Configure validation sizes: `validation.sizes`
- Training sizes auto-detected ✓

### No Confusion

**Before:**
- Users thought `size_distribution` controlled training ratio
- "If I change 720 from 0.4 to 0.6, will it sample more 720?"
- "Do I need to restart if I change distribution?"

**After:**
- File counts control training ratio (clear and direct)
- "Want more 720 in training? Add more 720 files during extraction"

### Automatic Behavior

**Before:**
- Add files to 720 folder
- Update `runtime_config.json` to enable 720
- Restart training

**After:**
- Add files to 720 folder
- Restart training (auto-detected) ✓

---

## Validation

### Startup Checks

Training automatically:
1. Scans each size directory (540, 720, 720_169)
2. Counts PNG files in GT folders
3. Reports file counts to console
4. Enables multi-size if multiple sizes found

### No Distribution Validation

No more:
- ❌ Distribution sum must equal 1.0
- ❌ Distribution values must be 0.0-1.0
- ❌ All distributions are 0 error

Simply:
- ✅ Use whatever sizes have files
- ✅ Ratio determined by file counts

---

## Migration Guide

### For Existing Configurations

**Old runtime_config.json with size_distribution:**
```json
{
  "size_distribution": {
    "720": 0.4,
    "540": 0.4,
    "720_169": 0.2
  }
}
```

**Action Required:** None! Old configs still work.

**What Happens:**
- `size_distribution` values are ignored
- Training auto-detects sizes from filesystem
- No errors, no warnings
- Just works! ✓

**Recommendation:** Remove `size_distribution` section for clarity.

### For New Configurations

Don't include `size_distribution` at all. Just:

```json
{
  "data": {
    "root": "/path/to/data",
    "dataset_name": "master"
  },
  "training": {
    "effective_batch_size": 6,
    "adaptive_batch": { ... }
  },
  "validation": {
    "sizes": ["720", "540"]
  }
}
```

---

## Example Scenarios

### Scenario 1: Start with Only 540 Files

**Filesystem:**
```
/data/master/train/
  ├── 540/GT/     (1,500 files)
  ├── 720/GT/     (empty)
  └── 720_169/GT/ (empty)
```

**Result:**
```
Found 1,500 files for size 540
✓ Single-size training: 540
```

### Scenario 2: Add 720 Files Later

**Filesystem:**
```
/data/master/train/
  ├── 540/GT/     (1,500 files)
  ├── 720/GT/     (1,200 files)  ← Added!
  └── 720_169/GT/ (empty)
```

**Action:** Restart training

**Result:**
```
Found 1,500 files for size 540
Found 1,200 files for size 720
✓ Multi-size training enabled: 540, 720
```

Training ratio: 55.6% from 540, 44.4% from 720 (proportional to file counts)

### Scenario 3: All Sizes Available

**Filesystem:**
```
/data/master/train/
  ├── 540/GT/     (1,500 files)
  ├── 720/GT/     (1,200 files)
  └── 720_169/GT/ (750 files)
```

**Result:**
```
Found 1,500 files for size 540
Found 1,200 files for size 720
Found 750 files for size 720_169
✓ Multi-size training enabled: 540, 720, 720_169
```

Training ratio: 43.5% / 34.8% / 21.7% (proportional to file counts)

---

## Technical Details

### Detection Algorithm

```python
for size_key in ['540', '720', '720_169']:
    train_dir = os.path.join(data_root, dataset_name, 'train', size_key, 'GT')
    
    if os.path.exists(train_dir):
        files = [f for f in os.listdir(train_dir) if f.endswith('.png')]
        
        if files:
            available.append((size_key, len(files)))
```

### Size Priority

Checks in order: 540, 720, 720_169

For single-size fallback, uses first available size.

### Validation Fallback

```python
# 1. Try explicit validation.sizes from config
val_sizes = rt_config.get('validation', {}).get('sizes', [])

# 2. Fallback to detected training sizes
if not val_sizes:
    available = detect_available_sizes(data_root, dataset_name)
    val_sizes = [size_key for size_key, _ in available]

# 3. Ultimate fallback
if not val_sizes:
    val_sizes = ['540']
```

---

## Summary

### What Changed

- ✅ Removed `size_distribution` from runtime_config.json
- ✅ Added auto-detection of available sizes
- ✅ Simplified dataloader logic
- ✅ Removed distribution validation
- ✅ Updated documentation

### User Impact

- ✅ Simpler configuration
- ✅ No confusion about training ratio
- ✅ Automatic size detection
- ✅ Backward compatible (old configs work)

### Key Principle

> **Files on disk determine everything.**
> 
> - Which sizes to load: Based on which directories have files
> - Training ratio: Based on file counts (pre-weighted during extraction)
> - No manual configuration needed!

This is the cleanest, simplest, and most intuitive approach. 🎉
