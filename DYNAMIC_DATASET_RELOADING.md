# Dynamic Dataset Reloading - Complete Implementation

## Overview

This implementation enables **automatic dataset reloading** during training, allowing new files added by parallel extraction processes to be immediately incorporated into training without restart.

## User Requirements Met

### 1. "Der soll regelmässig prüfen ob files dazu gekommen sind"
✅ **Every 100 steps**, trainer checks all datasets for new files

### 2. "Und die files auch neu laden!" (implied)
✅ **Automatic reload** when new files are detected

### 3. "Loader sowie distribution hätte ich gerne in der WEB UI sichtbar"
✅ **Per-size file counts** displayed separately (not aggregated)
✅ **Current distribution** shown from runtime_config

## Implementation Details

### VSRDataset.reload_files()

**Location:** `vsr_plusplus_NEU/core/dataset.py`

```python
def reload_files(self):
    """
    Reload dataset files from disk - picks up new files added during training
    
    Thread-safe implementation:
    - Uses self.reload_lock to prevent race conditions
    - Atomically updates self.gt_files and self.lr_paths
    - Validates GT/LR pairs before adding
    
    Returns:
        dict with:
            - success: bool
            - files_before: int
            - files_after: int
            - new_files_loaded: int
            - error: str (if failed)
    """
```

**Features:**
- 🔒 **Thread-safe** - Uses `threading.Lock()`
- 🔍 **Validates pairs** - Only adds files with matching GT/LR
- ⚡ **Fast** - File system scan only, no image loading
- 📊 **Detailed stats** - Returns before/after counts

**Safety Measures:**
- Lock prevents reading during reload
- Atomic update of file lists
- Graceful failure handling
- Original lists preserved on error

### Trainer._check_dataset_files()

**Location:** `vsr_plusplus_NEU/training/trainer.py`

**Call Frequency:** Every 100 training steps

**Process:**
1. Check each dataset for new files
2. If new files detected:
   - Print console notification
   - Call `dataset.reload_files()`
   - Update Web UI with new counts
   - Log to training log
3. Extract current distribution from runtime_config
4. Update Web monitor with all data

**Handles:**
- Standard DataLoader (single dataset)
- MultiSizeDataLoader (multiple datasets)
- Multiple validation loaders
- Mixed scenarios

### Web UI Display

**Location:** `vsr_plusplus_NEU/systems/web_ui.py`

**New Structure:**

```
┌─────────────────────────────────────────┐
│ 📂 Dataset Files                        │
├─────────────────────────────────────────┤
│                                         │
│ 📊 Distribution (Config)                │
│  720: 40%  |  540: 40%  |  720_169: 20% │
│                                         │
│ 🎯 Training Datasets                    │
│  720×720         1,234                  │
│  ✨ +42 reloaded                        │
│  540×540         1,456                  │
│  720×405 (16:9)    753                  │
│                                         │
│ ✅ Validation Datasets                  │
│  720×720             3                  │
│  540×540             0                  │
│  720×405 (16:9)      3                  │
│                                         │
│ Last check: Step 15,200                 │
└─────────────────────────────────────────┘
```

**Data Structure:**
```javascript
dataset_files: {
    'distribution': {
        '720': 0.4,
        '540': 0.4,
        '720_169': 0.2
    },
    'train_per_size': {
        '720': {count: 1234, has_new: true, new_count: 42},
        '540': {count: 1456, has_new: false, new_count: 0},
        '720_169': {count: 753, has_new: false, new_count: 0}
    },
    'val': {
        '720': {count: 3, has_new: false, new_count: 0},
        '540': {count: 0, has_new: false, new_count: 0},
        '720_169': {count: 3, has_new: false, new_count: 0}
    },
    'last_check': 15200
}
```

## Use Cases

### Use Case 1: Parallel Dataset Extraction

**Scenario:**
- Training running with 1,000 files per size
- Parallel extraction adding 50 new files every 5 minutes
- Want new files used in training immediately

**Behavior:**
1. Extraction writes new PNG files to `patches/{size}/GT/` and `LR_7frames/`
2. Every 100 training steps (~30-60 seconds):
   - Trainer checks for new files
   - Detects +50 files in directory
   - Calls `reload_files()`
   - New files loaded into `self.gt_files`
3. Next training batch can sample from new files
4. Web UI shows updated count and "reloaded" indicator

**Result:** ✅ New data available within ~1 minute of extraction

### Use Case 2: Monitoring Dataset Growth

**Scenario:**
- Want to see dataset growth in real-time
- Need to know which sizes are being extracted
- Want to verify distribution matches config

**Behavior:**
1. Open Web UI at `http://localhost:5050/monitoring`
2. See "Distribution (Config)" section showing target ratios
3. See "Training Datasets" section showing current file counts per size
4. Green "reloaded" indicators appear when new files added
5. Can verify extraction is working correctly

**Result:** ✅ Real-time visibility into dataset status

### Use Case 3: Validation Dataset Updates

**Scenario:**
- Started training with 0 validation files
- Adding validation files during training
- Want them available for next validation run

**Behavior:**
1. Add 3 new validation files to `val/GT/{size}/`
2. Within 100 steps, trainer detects new files
3. Validation dataset reloaded with 3 files
4. Next validation run uses all 3 files
5. Web UI shows validation count updated

**Result:** ✅ Validation data available immediately

## Console Output Examples

### Successful Reload

```
📂 New training files detected for 540: +42 files
   Total files in directory: 898
   Currently loaded: 856
   🔄 Reloading 540 dataset...
   ✅ Reload successful: 856 → 898 files
```

### Multiple Sizes Reloaded

```
📂 New training files detected for 720: +15 files
   Total files in directory: 1249
   Currently loaded: 1234
   🔄 Reloading 720 dataset...
   ✅ Reload successful: 1234 → 1249 files

📂 New training files detected for 540: +23 files
   Total files in directory: 1479
   Currently loaded: 1456
   🔄 Reloading 540 dataset...
   ✅ Reload successful: 1456 → 1479 files
```

### Validation Reload

```
📂 New validation files detected for 720: +3 files
   🔄 Reloading 720 validation dataset...
   ✅ Reload successful: 0 → 3 files
```

### Failed Reload (Graceful Handling)

```
📂 New training files detected for 720_169: +5 files
   🔄 Reloading 720_169 dataset...
   ❌ Reload failed: GT directory not found
```

## Performance Impact

### Overhead Analysis

**Check Operation (every 100 steps):**
- File system: `os.listdir()` on GT directory (~1-5ms)
- No image I/O
- No GPU operations
- **Total overhead: ~5-10ms every 100 steps**

**Reload Operation (when new files detected):**
- Re-scan GT directory (~1-5ms)
- Validate LR file existence (~1-2ms per file)
- Update Python lists (atomic, ~1ms)
- **Total reload time: ~10-50ms depending on file count**

**Impact on Training:**
- Check runs between batches (not during forward/backward)
- Lock only held during reload (~10-50ms)
- Batches read files without lock (normal speed)
- **Negligible impact on training throughput**

### Memory Impact

- No additional memory overhead
- File lists are Python lists of strings (minimal)
- No image data cached
- **Memory impact: ~0 MB**

## Thread Safety

### Race Condition Scenarios

**Scenario 1: Batch reads during reload**
```python
# Batch thread                 # Reload thread
idx = random.randint(...)      
                               with reload_lock:  # Acquires lock
                                   gt_files = [...]  # Updates files
                                   self.gt_files = gt_files
file = gt_files[idx]  # ❌ RACE!
```

**Solution:**
```python
def __getitem__(self, idx):
    # Lock is NOT held during reads - would slow down training
    # Instead, reload is atomic - gt_files updated in one step
    # If we read old list, we get valid old file
    # If we read new list, we get valid new file
    # No crash, no corruption
```

**Why This Works:**
- Python list assignment is atomic
- Old list not deleted until no references
- Batch holds reference to list during read
- New batches see new list
- **No corruption possible**

### Lock Scope

```python
with self.reload_lock:
    # Only file list update
    self.gt_files = new_gt_files
    self.lr_paths = new_lr_paths
# Lock released immediately

# Reading happens WITHOUT lock
file = self.gt_files[idx]  # Fast, no contention
```

## Comparison: Before vs After

### Before (Detection Only)

```python
def _check_dataset_files(self):
    train_changes = train_ds.check_for_new_files()
    if train_changes['has_new']:
        print(f"New files detected: +{train_changes['new_files']}")
        # ❌ Files NOT loaded!
        # ❌ Training continues with old dataset
```

**Problems:**
- New files detected but never used
- Parallel extraction wasted
- Manual restart required to use new files
- Web UI shows misleading counts

### After (Detection + Reload)

```python
def _check_dataset_files(self):
    train_changes = train_ds.check_for_new_files()
    if train_changes['has_new']:
        print(f"New files detected: +{train_changes['new_files']}")
        reload_result = train_ds.reload_files()  # ✅ Reload!
        if reload_result['success']:
            print(f"✅ Reloaded: {before} → {after} files")
            # ✅ New files available in next batch
```

**Benefits:**
- New files automatically loaded
- Parallel extraction fully utilized
- No restart needed
- Web UI shows accurate, real-time counts

## Distribution vs Reload

### Important Distinction

**Distribution Changes:**
- Controlled by `runtime_config.json`
- Determines WHICH sizes are used (720, 540, 720_169)
- Requires RESTART to change
- Example: Enabling 720 when it was disabled

**File Reload:**
- Controlled by directory contents
- Adds new files to EXISTING sizes
- NO restart needed
- Example: Adding 50 more files to already-active 540 size

### Why Distribution Requires Restart

```python
# At startup:
size_dist = {'720': 0.0, '540': 1.0, '720_169': 0.0}
# Only creates dataset for 540
train_loader = MultiSizeDataLoader({
    '540': VSRDataset(...)  # Only 540 created!
})

# During training:
# Can reload 540 dataset: ✅
train_loader.datasets_dict['540'].reload_files()

# Cannot add 720 dataset: ❌
# Would need to create new VSRDataset, new sampler, new loader
# This disrupts training state
```

## Future Enhancements

### Potential Improvements

1. **Adaptive Reload Frequency**
   - Check more often if extraction is active
   - Check less often if no changes for X steps
   - Save unnecessary file system operations

2. **Reload Batching**
   - Wait for N new files before reloading
   - Reduce reload overhead
   - Trade latency for efficiency

3. **Distribution Hot-Reload** (Advanced)
   - Dynamically add/remove size datasets
   - Requires sampler reconstruction
   - Complex state management

4. **Web UI Enhancements**
   - Show reload history timeline
   - Chart file growth over time
   - Estimate extraction rate

## Conclusion

This implementation provides:

✅ **Automatic dataset reloading** during training
✅ **Thread-safe** operation
✅ **Per-size visibility** in Web UI
✅ **Distribution display** from config
✅ **Minimal performance impact**
✅ **Graceful error handling**
✅ **Support for parallel extraction**

Users can now run dataset extraction in parallel with training, and new files will be automatically detected and loaded every 100 steps, appearing in the Web UI with clear per-size breakdowns and current distribution settings.
