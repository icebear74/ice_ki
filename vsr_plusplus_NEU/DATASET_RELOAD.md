# VSR Trainer: Automatic Dataset Reloading

## Overview

The VSR trainer (`vsr_plusplus_NEU`) automatically reloads the video list while training is running. This allows continuous training even when new videos are being added to the dataset by parallel extraction processes.

## How It Works

### Dual-Strategy Checking

The trainer employs two complementary strategies to detect and reload new dataset files:

#### 1. Periodic Checks (Every 100 Steps)
```python
# In train_epoch() method, during the batch processing loop:
if self.global_step % 100 == 0:
    self._check_dataset_files()
```

**Purpose**: Detect new files quickly during long-running epochs
**Frequency**: Every 100 training steps
**Location**: Inside the batch processing loop

#### 2. End-of-Epoch Checks
```python
# At the end of train_epoch() method, after all batches processed:
print(f"\n📊 End of epoch {epoch} - checking for new dataset files...")
self._check_dataset_files()
```

**Purpose**: Ensure new files are available for the next epoch
**Frequency**: Once per epoch
**Location**: After the batch loop completes, before next epoch starts

### The `_check_dataset_files()` Method

This method handles all the file detection and reloading logic:

1. **Scans dataset directories** for new files
2. **Compares** with currently loaded files
3. **Reloads datasets** if new files are found
4. **Updates metrics** for web monitoring
5. **Logs events** for tracking

#### Supported Dataset Types

**Single-Size Datasets:**
```python
train_loader.dataset  # Standard DataLoader
```

**Multi-Size Datasets:**
```python
train_loader.datasets_dict['540']   # 540p dataset
train_loader.datasets_dict['1080']  # 1080p dataset
train_loader.datasets_dict['2160']  # 2160p dataset
```

**Validation Datasets:**
```python
val_loader.dataset              # Single validation
val_loaders[('540', loader)]   # Multi-size validation
```

## Use Case: Parallel Dataset Generation

### Scenario
You want to:
1. Start training with an initial dataset (e.g., 100 videos)
2. Continue generating more dataset videos while training
3. Have the trainer automatically pick up new videos

### Workflow

**Step 1: Start Training**
```bash
cd vsr_plusplus_NEU
python3 train.py
```

Initial state:
- Training starts with 100 videos
- Trainer shows: "Training: 540p (100 files)"

**Step 2: Generate More Data (in parallel)**
```bash
# In another terminal
cd dataset_generator_v2
python3 make_dataset_v2_uhd.py
```

The dataset generator adds:
- Epoch 1: +50 new videos
- Epoch 2: +50 new videos
- Epoch 3: +50 new videos

**Step 3: Automatic Detection**

The trainer automatically detects and reloads:

```
[Step 100] Checking for new files...
[Step 200] Checking for new files...
[Step 300] Checking for new files...

📊 End of epoch 1 - checking for new dataset files...
📂 New training files detected for 540: +50 files
   Total files in directory: 150
   Currently loaded: 100
   🔄 Reloading dataset...
   ✅ Reload successful: 100 → 150 files

[Epoch 2 starts]
Training: 540p (150 files) ← Updated!

📊 End of epoch 2 - checking for new dataset files...
📂 New training files detected for 540: +50 files
   🔄 Reloading dataset...
   ✅ Reload successful: 150 → 200 files

[Epoch 3 starts]
Training: 540p (200 files) ← Updated again!
```

## Benefits

### ✅ Continuous Operation
- No need to stop and restart training
- Training continues seamlessly with new data
- Maximizes GPU utilization

### ✅ Natural Boundaries
- Checks happen at epoch boundaries (no mid-epoch disruption)
- Additional checks every 100 steps for long epochs
- Batch processing never interrupted

### ✅ Automatic Integration
- New videos picked up automatically
- No manual intervention required
- Training logs show reload events

### ✅ Quality Improvement
- Model sees more diverse data over time
- Dataset expands during training
- Better generalization from increased variety

## Technical Details

### File Detection Logic

The dataset classes (in `core/dataset_loader.py`) implement these methods:

```python
class VSRDataset:
    def get_file_info(self):
        """Get current file count and size info"""
        return {
            'size_key': self.size_key,
            'file_count': len(self.file_list)
        }
    
    def check_for_new_files(self):
        """Check if new files exist in the directory"""
        # Scan directory for GT files
        current_files = glob.glob(f"{self.gt_dir}/*.png")
        
        # Compare with loaded files
        has_new = len(current_files) > len(self.file_list)
        
        return {
            'has_new': has_new,
            'new_files': len(current_files) - len(self.file_list),
            'new_gt_count': len(current_files),
            'current_loaded': len(self.file_list)
        }
    
    def reload_files(self):
        """Reload the file list from disk"""
        old_count = len(self.file_list)
        
        # Re-scan directory
        self._build_file_list()
        
        new_count = len(self.file_list)
        
        return {
            'success': True,
            'files_before': old_count,
            'files_after': new_count,
            'new_files_loaded': new_count - old_count
        }
```

### Error Handling

The reload process is robust:

```python
try:
    if train_changes['has_new']:
        print(f"📂 New training files detected: +{train_changes['new_files']}")
        
        if hasattr(train_ds, 'reload_files'):
            reload_result = train_ds.reload_files()
            if reload_result['success']:
                print(f"✅ Reload successful")
            else:
                print(f"❌ Reload failed: {reload_result.get('error')}")
        else:
            print(f"⚠️  Dataset does not support reload_files()")
except Exception as e:
    print(f"⚠️  Error checking dataset: {e}")
    traceback.print_exc()
```

If reload fails:
- Training continues with existing files
- Error is logged but doesn't crash
- Next check will retry

## Configuration

### Check Frequency

The 100-step check interval is hardcoded but can be modified:

```python
# In trainer.py, change this line:
if self.global_step % 100 == 0:  # ← Change 100 to desired interval
    self._check_dataset_files()
```

Recommendations:
- **Fast epochs** (< 500 steps): Use 50 or 100
- **Normal epochs** (500-2000 steps): Use 100 or 200
- **Long epochs** (> 2000 steps): Use 200 or 500

The epoch-end check always runs regardless of step interval.

### Disabling Reload (If Needed)

To disable automatic reloading:

```python
# Comment out the checks in trainer.py:

# if self.global_step % 100 == 0:
#     self._check_dataset_files()

# self._check_dataset_files()  # End of epoch check
```

## Monitoring

### Console Output

During training, you'll see:

```
[Step 100] 
[Step 200] 
[Step 300] 

📊 End of epoch 1 - checking for new dataset files...
```

When new files are found:
```
📂 New training files detected for 1080: +25 files
   Total files in directory: 525
   Currently loaded: 500
   🔄 Reloading 1080 dataset...
   ✅ Reload successful: 500 → 525 files
```

### Web Monitor

The web interface (port 5050) shows:
- Current file counts per size
- Distribution across sizes
- Last check timestamp

```
Training Files:
  540p:  150 files (30%)
  1080p: 250 files (50%)
  2160p: 100 files (20%)

Last checked: Step 3400
```

### Log File

The training log records all reload events:

```
2024-02-11 12:34:56 - Reloaded 1080 training: +25 files
2024-02-11 13:15:22 - Reloaded 540 training: +50 files
2024-02-11 14:45:10 - Reloaded 2160 validation: +10 files
```

## Best Practices

### 1. Start with Initial Dataset
Always start training with at least some data:
- Minimum: 50-100 videos per size category
- Recommended: 200-500 videos for stable early training

### 2. Generate in Batches
Add videos in reasonable batches:
- Too small (< 10): Frequent reloads, overhead
- Too large (> 100): Long wait for new data
- Sweet spot: 20-50 videos per batch

### 3. Monitor Reload Events
Check that reloads are happening:
```bash
# Watch for reload messages
tail -f training_log.txt | grep "Reloaded"
```

### 4. Balance Generation Speed
- If generation is slower than training: Train with existing data
- If generation is faster: Reloads happen frequently, which is fine

## Troubleshooting

### "Dataset does not support reload_files()"

**Cause**: Old dataset implementation without reload support

**Solution**: Update dataset class to implement:
- `get_file_info()`
- `check_for_new_files()`
- `reload_files()`

### No New Files Detected

**Cause**: Files in wrong directory or wrong format

**Check**:
```bash
# Verify files are in correct location
ls -l data/train/540_train/GT/*.png | wc -l

# Check dataset config points to right directory
grep "gt_dir" config.py
```

### Reload Failed

**Cause**: Permissions, disk space, or corrupted files

**Check**:
```bash
# Check permissions
ls -la data/train/540_train/GT/

# Check disk space
df -h

# Check for corrupted files
find data/ -name "*.png" -size 0
```

## Performance Impact

### Minimal Overhead
- File scanning: ~10-100ms (depends on file count)
- Only happens every 100 steps + end of epoch
- Negligible compared to training time per step

### Example Timing
```
Step time: 150-300ms (GPU training)
Reload check: 10-20ms (file system scan)
Overhead: < 7% (only when checking)
```

For a 1000-step epoch:
- Checks: 10 times (every 100 steps) + 1 (end of epoch) = 11 checks
- Total overhead: 11 × 20ms = 220ms
- Training time: 1000 × 200ms = 200,000ms
- Overhead percentage: 0.11% ← Negligible!

## Summary

The automatic dataset reloading feature enables:

✅ **Continuous training** during dataset generation
✅ **Automatic detection** of new files
✅ **Zero manual intervention** required
✅ **Epoch-safe** reloading (no disruption)
✅ **Multi-size support** (540p, 1080p, 2160p)
✅ **Robust error handling** (training continues on failure)
✅ **Comprehensive logging** (console, file, web monitor)

This is perfect for the common workflow:
1. Start training with initial dataset
2. Continue generating more data in parallel
3. Let trainer automatically pick up new videos
4. Enjoy continuous improvement with expanding dataset
