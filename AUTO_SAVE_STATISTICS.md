# Auto-Save Training Statistics - Documentation

## Overview
The training system now automatically saves complete training statistics as JSON files after each validation. This provides a complete historical record of training progress without requiring manual downloads.

---

## Feature Details

### When Statistics Are Saved
Statistics JSON files are saved **automatically** after:
1. **Automatic Validation** - Every VAL_STEP_EVERY steps (default: 500)
2. **Manual Validation** - When triggered via keyboard ('v') or web UI

### File Location
Files are saved to the **Learning directory** (DATA_ROOT from config):
```
/mnt/data/training/Universal/Mastermodell/Learn/
```

### File Naming Convention
```
Statistik_STEP.json
```

Where `STEP` is the current training step.

**Examples:**
- `Statistik_500.json` - First validation at step 500
- `Statistik_1000.json` - Second validation at step 1000
- `Statistik_7500.json` - Validation at step 7500
- etc.

---

## Data Contents

Each JSON file contains **all data** from the web UI monitoring interface:

### Basic Training Metrics
```json
{
  "step_current": 7500,
  "epoch_num": 8,
  "step_max": 100000,
  "epoch_step_current": 235,
  "epoch_step_total": 1000
}
```

### Learning Rate Information
```json
{
  "learning_rate_value": 0.00003456,
  "lr_phase_name": "cosine"
}
```

### Loss Values
```json
{
  "total_loss_value": 0.0234,
  "l1_loss_value": 0.0156,
  "ms_loss_value": 0.0045,
  "gradient_loss_value": 0.0023,
  "perceptual_loss_value": 0.0010
}
```

### Adaptive System Weights
```json
{
  "l1_weight_current": 0.6,
  "ms_weight_current": 0.2,
  "gradient_weight_current": 0.2,
  "perceptual_weight_current": 0.123,
  "gradient_clip_val": 1.5
}
```

### Adaptive System Status
```json
{
  "adaptive_mode": "Stable",
  "adaptive_is_cooldown": false,
  "adaptive_cooldown_remaining": 0,
  "adaptive_plateau_counter": 42,
  "adaptive_lr_boost_available": false,
  "adaptive_perceptual_trend": 0.0015
}
```

### Quality Metrics (✨ Including Improvement!)
```json
{
  "quality_lr_value": 0.543,
  "quality_ki_value": 0.621,
  "quality_improvement_value": 0.189,
  "quality_ki_to_gt_value": 0.234,
  "quality_lr_to_gt_value": 0.456,
  "validation_loss_value": 0.0145,
  "best_quality_ever": 0.658
}
```

### Performance Metrics
```json
{
  "iteration_duration": 0.234,
  "vram_usage_gb": 8.45,
  "adam_momentum_avg": 0.9123,
  "eta_total_formatted": "12:34:56",
  "eta_epoch_formatted": "00:45:30"
}
```

### Layer Activities
```json
{
  "layer_activity_map": {
    "Backward_Block_01": 34.5,
    "Backward_Block_02": 45.2,
    "Forward_Block_01": 38.7,
    "Final_Fusion": 67.3
  }
}
```

### Training Status
```json
{
  "training_active": true,
  "validation_running": false,
  "training_paused": false,
  "last_update_time": 1707213600.123
}
```

---

## Usage Examples

### Tracking Improvement Over Time
```python
import json
import glob

# Load all statistics files
stats_files = sorted(glob.glob('Learn/Statistik_*.json'))

for filepath in stats_files:
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    step = data['step_current']
    improvement = data['quality_improvement_value'] * 100  # Convert to %
    ki_quality = data['quality_ki_value'] * 100
    
    print(f"Step {step:6d}: Improvement={improvement:5.1f}%, KI Quality={ki_quality:5.1f}%")
```

**Output:**
```
Step    500: Improvement= 12.3%, KI Quality= 54.2%
Step   1000: Improvement= 15.7%, KI Quality= 57.8%
Step   1500: Improvement= 18.2%, KI Quality= 61.3%
Step   2000: Improvement= 21.5%, KI Quality= 64.7%
```

### Analyzing Plateau Behavior
```python
import json

# Load a specific statistics file
with open('Learn/Statistik_7500.json', 'r') as f:
    data = json.load(f)

# Check for plateau warning
plateau = data['adaptive_plateau_counter']
lr_boost = data['adaptive_lr_boost_available']

if plateau > 300:
    print(f"⚠️  WARNING: Plateau detected for {plateau} steps!")
    if lr_boost:
        print("   LR Boost is available and will trigger soon")
    else:
        print("   LR Boost is on cooldown")
else:
    print(f"✅ Training is progressing (plateau: {plateau} steps)")
```

### Comparing Different Training Runs
```python
import json

def compare_runs(run1_path, run2_path, step):
    # Load statistics from two different runs at same step
    with open(f"{run1_path}/Statistik_{step}.json") as f:
        run1 = json.load(f)
    
    with open(f"{run2_path}/Statistik_{step}.json") as f:
        run2 = json.load(f)
    
    print(f"Comparison at Step {step}:")
    print(f"  Run 1: Improvement={run1['quality_improvement_value']*100:.1f}%")
    print(f"  Run 2: Improvement={run2['quality_improvement_value']*100:.1f}%")
```

---

## Log Output

When statistics are saved, you'll see:
```
📊 Statistics saved: Statistik_7500.json
```

This message appears:
- In the terminal console
- In the training log file (`training.log`)

If saving fails (e.g., permission error):
```
⚠️  Failed to save statistics JSON: [error details]
```

---

## File Format

Files are saved with **pretty formatting** for readability:
- 2-space indentation
- Sorted keys (optional, depends on Python version)
- UTF-8 encoding
- Human-readable numbers (scientific notation where appropriate)

**Example File Content:**
```json
{
  "step_current": 7500,
  "epoch_num": 8,
  "step_max": 100000,
  "learning_rate_value": 3.456e-05,
  "adaptive_mode": "Stable",
  "quality_improvement_value": 0.189,
  "quality_ki_value": 0.621,
  "total_loss_value": 0.0234,
  ...
}
```

---

## Integration with Training System

### Automatic Validation
```python
# In trainer.py, line ~256
if self.global_step % self.config.get('VAL_STEP_EVERY', 500) == 0:
    metrics = self.validator.validate(self.global_step)
    # ... TensorBoard logging ...
    
    # Auto-save statistics JSON
    self._save_statistics_json(self.global_step)
```

### Manual Validation
```python
# In trainer.py, line ~655
def _run_validation(self):
    metrics = self.validator.validate(self.global_step)
    # ... logging ...
    
    # Auto-save statistics JSON
    self._save_statistics_json(self.global_step)
```

---

## Benefits

### For Training Monitoring
- ✅ Complete historical record of training progress
- ✅ No manual downloads needed
- ✅ Can track improvement trends over time
- ✅ Easy to spot when training gets stuck

### For Analysis
- ✅ Machine-readable format for automated analysis
- ✅ Can compare different training runs
- ✅ Can identify optimal training configurations
- ✅ Can detect anomalies or issues

### For Debugging
- ✅ Full system state at each validation point
- ✅ Can trace back when issues started
- ✅ Can see correlation between metrics
- ✅ Can share with others for troubleshooting

### For Documentation
- ✅ Permanent record of training experiments
- ✅ Can recreate training graphs from JSON
- ✅ Can include in research papers/reports
- ✅ Can archive successful configurations

---

## Storage Considerations

### File Size
Each JSON file is approximately **5-15 KB** depending on:
- Number of layers (affects layer_activity_map size)
- Precision of floating-point numbers
- Pretty formatting vs compact

### Disk Usage Estimates
For a 100,000 step training run with validation every 500 steps:
- Number of files: 200
- Total size: ~1-3 MB
- Negligible compared to model checkpoints (500+ MB each)

### Cleanup (Optional)
If you want to reduce storage, you can:
1. Keep only milestone statistics (every 5000 steps)
2. Compress old statistics: `gzip Statistik_*.json`
3. Delete statistics older than X days

---

## Troubleshooting

### File Not Created
**Problem:** Statistics file doesn't appear after validation

**Check:**
1. Verify DATA_ROOT path exists and is writable
2. Check training log for error messages
3. Verify validation actually ran (check console output)

### Permission Error
**Problem:** `PermissionError: [Errno 13] Permission denied`

**Solution:**
```bash
# Make sure directory is writable
chmod 755 /path/to/Learn/
```

### Incomplete Data
**Problem:** Some fields are missing or zero

**Possible causes:**
- First validation (some metrics not yet available)
- Web monitor not fully initialized
- Validation ran before first training step

**Solution:** This is normal for very early validations. Data will be complete after a few hundred steps.

---

## Advanced Usage

### Export to CSV
```python
import json
import csv
import glob

# Collect all statistics
stats = []
for filepath in sorted(glob.glob('Learn/Statistik_*.json')):
    with open(filepath, 'r') as f:
        stats.append(json.load(f))

# Write to CSV
with open('training_history.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=stats[0].keys())
    writer.writeheader()
    writer.writerows(stats)
```

### Plot Training Progress
```python
import json
import glob
import matplotlib.pyplot as plt

steps = []
improvements = []
ki_qualities = []

for filepath in sorted(glob.glob('Learn/Statistik_*.json')):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    steps.append(data['step_current'])
    improvements.append(data['quality_improvement_value'] * 100)
    ki_qualities.append(data['quality_ki_value'] * 100)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(steps, improvements, 'b-', linewidth=2)
plt.xlabel('Training Step')
plt.ylabel('Improvement (%)')
plt.title('Quality Improvement Over Time')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(steps, ki_qualities, 'g-', linewidth=2)
plt.xlabel('Training Step')
plt.ylabel('KI Quality (%)')
plt.title('KI Quality Over Time')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_progress.png')
```

---

## Summary

The auto-save statistics feature provides:
- ✅ **Automatic** archiving of training data
- ✅ **Complete** snapshot of all metrics
- ✅ **Sequential** naming for easy tracking
- ✅ **Same data** as manual web UI download
- ✅ **Machine-readable** JSON format
- ✅ **Minimal overhead** (~10KB per file)

All statistics are saved to the Learning directory as `Statistik_STEP.json` after each validation, providing a complete historical record of your training progress! 📊
