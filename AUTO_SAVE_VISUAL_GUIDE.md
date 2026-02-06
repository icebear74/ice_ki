# Auto-Save Statistics - Visual Guide

## What You'll See During Training

### Console Output

When validation completes, you'll now see:
```
═══════════════════════════════════════════════════════════════════
🔍 VALIDATION
═══════════════════════════════════════════════════════════════════

Progress: [████████████████████████████] 100.0% | Batch 10/10 (40 samples) | ETA: 0.0s

Validation Results:
  LR Quality:    54.2% (avg PSNR/SSIM vs GT)
  KI Quality:    62.7% (avg PSNR/SSIM vs GT)
  Improvement:   18.5% (KI better than LR)
  Val Loss:      0.0145

📊 Logging validation metrics to TensorBoard...
✅ Successfully logged all 10 validation images to TensorBoard
📊 Statistics saved: Statistik_7500.json          ← NEW! 🎉
```

The new line `📊 Statistics saved: Statistik_7500.json` confirms the file was saved.

---

## File Structure in Learning Directory

Before implementing the feature:
```
/mnt/data/training/Universal/Mastermodell/Learn/
├── training.log
├── training_status.txt
├── checkpoints/
│   ├── checkpoint_latest.pth
│   └── checkpoint_best.pth
└── active_run/
    └── (TensorBoard logs)
```

After implementing the feature:
```
/mnt/data/training/Universal/Mastermodell/Learn/
├── training.log
├── training_status.txt
├── Statistik_500.json      ← NEW! First validation
├── Statistik_1000.json     ← NEW! Second validation
├── Statistik_1500.json     ← NEW! Third validation
├── Statistik_2000.json     ← NEW! ...and so on
├── Statistik_2500.json
├── Statistik_3000.json
├── ...
├── checkpoints/
│   ├── checkpoint_latest.pth
│   └── checkpoint_best.pth
└── active_run/
    └── (TensorBoard logs)
```

---

## Sample Statistics File

**File:** `Statistik_7500.json`

**Size:** ~10-15 KB (small!)

**Content Preview:**
```json
{
  "step_current": 7500,
  "epoch_num": 8,
  "step_max": 100000,
  "epoch_step_current": 235,
  "epoch_step_total": 1000,
  
  "learning_rate_value": 3.456e-05,
  "lr_phase_name": "cosine",
  
  "total_loss_value": 0.0234,
  "l1_loss_value": 0.0156,
  "ms_loss_value": 0.0045,
  "gradient_loss_value": 0.0023,
  "perceptual_loss_value": 0.001,
  
  "l1_weight_current": 0.6,
  "ms_weight_current": 0.2,
  "gradient_weight_current": 0.2,
  "perceptual_weight_current": 0.123,
  "gradient_clip_val": 1.5,
  
  "adaptive_mode": "Stable",
  "adaptive_is_cooldown": false,
  "adaptive_cooldown_remaining": 0,
  "adaptive_plateau_counter": 42,
  "adaptive_lr_boost_available": false,
  "adaptive_perceptual_trend": 0.0015,
  
  "quality_lr_value": 0.543,
  "quality_ki_value": 0.621,
  "quality_improvement_value": 0.189,    ← The improvement percentage!
  "quality_ki_to_gt_value": 0.234,
  "quality_lr_to_gt_value": 0.456,
  "validation_loss_value": 0.0145,
  "best_quality_ever": 0.658,
  
  "iteration_duration": 0.234,
  "vram_usage_gb": 8.45,
  "adam_momentum_avg": 0.9123,
  "eta_total_formatted": "12:34:56",
  "eta_epoch_formatted": "00:45:30",
  
  "layer_activity_map": {
    "Backward_Block_01": 34.5,
    "Backward_Block_02": 45.2,
    "Backward_Block_03": 38.9,
    "...": "...",
    "Forward_Block_01": 38.7,
    "Forward_Block_02": 42.3,
    "...": "...",
    "Final_Fusion": 67.3
  },
  
  "training_active": true,
  "validation_running": false,
  "training_paused": false,
  "last_update_time": 1707213600.123
}
```

---

## Comparison: Manual Download vs Auto-Save

### Before (Manual Download Only)

**To get training data:**
1. Open web browser
2. Navigate to `http://localhost:5050/monitoring`
3. Click "📥 Download Data (JSON)" button
4. Save file manually
5. Rename file to something meaningful
6. Remember to do this after every validation!

**Problems:**
- ❌ Easy to forget
- ❌ Manual process
- ❌ Loses historical data if not downloaded
- ❌ No automatic archiving

### After (Auto-Save)

**To get training data:**
- ✅ Nothing! It happens automatically!

**Benefits:**
- ✅ Automatic after every validation
- ✅ Complete training history preserved
- ✅ Sequential naming (Statistik_500, Statistik_1000, etc.)
- ✅ Can still manually download if needed

---

## Timeline Example

Here's what happens during a typical training run:

```
Step 0      Training starts
            └─ No statistics yet

Step 500    First validation
            ├─ Validation runs
            ├─ Improvement calculated: 12.3%
            └─ 📊 Statistik_500.json saved

Step 1000   Second validation
            ├─ Validation runs
            ├─ Improvement calculated: 15.7%
            └─ 📊 Statistik_1000.json saved

Step 1500   Third validation
            ├─ Validation runs
            ├─ Improvement calculated: 18.2%
            └─ 📊 Statistik_1500.json saved

Step 2000   Fourth validation
            ├─ Validation runs
            ├─ Improvement calculated: 21.5%
            └─ 📊 Statistik_2000.json saved

... and so on, every 500 steps!

Step 99500  Near end of training
            ├─ Validation runs
            ├─ Improvement calculated: 45.8%
            └─ 📊 Statistik_99500.json saved

Step 100000 Training complete
            └─ 200 statistics files saved!
```

---

## Error Handling

### If Saving Fails

You'll see an error message (but training continues):

```
⚠️  Failed to save statistics JSON: [Error details]
```

Common errors and solutions:

**Permission Denied:**
```
⚠️  Failed to save statistics JSON: PermissionError: Permission denied
```
→ Fix: `chmod 755 /path/to/Learn/`

**Disk Full:**
```
⚠️  Failed to save statistics JSON: OSError: No space left on device
```
→ Fix: Free up disk space

**Directory Doesn't Exist:**
```
⚠️  Failed to save statistics JSON: FileNotFoundError: No such directory
```
→ Fix: This shouldn't happen (directory is auto-created), but verify DATA_ROOT path

---

## Quick Start Guide

### 1. Start Training
```bash
cd /home/runner/work/ice_ki/ice_ki/vsr_plus_plus
python train.py
```

### 2. Wait for First Validation
Training runs for 500 steps, then first validation happens.

### 3. Check for Statistics File
```bash
ls -lh /mnt/data/training/Universal/Mastermodell/Learn/Statistik_*.json
```

Expected output:
```
-rw-r--r-- 1 user user 12K Feb  6 10:30 Statistik_500.json
```

### 4. View Statistics
```bash
cat /mnt/data/training/Universal/Mastermodell/Learn/Statistik_500.json | jq
```

Or with Python:
```python
import json

with open('Statistik_500.json', 'r') as f:
    data = json.load(f)

print(f"Step: {data['step_current']}")
print(f"Improvement: {data['quality_improvement_value']*100:.1f}%")
print(f"KI Quality: {data['quality_ki_value']*100:.1f}%")
```

---

## Analysis Examples

### Track Improvement Over Time
```bash
# List all statistics files
ls -1 Learn/Statistik_*.json | while read file; do
    step=$(basename "$file" .json | cut -d_ -f2)
    improvement=$(jq '.quality_improvement_value * 100' "$file")
    echo "Step $step: Improvement=$improvement%"
done
```

Output:
```
Step 500: Improvement=12.3%
Step 1000: Improvement=15.7%
Step 1500: Improvement=18.2%
Step 2000: Improvement=21.5%
```

### Find Best Quality
```bash
# Find the step with best quality
for file in Learn/Statistik_*.json; do
    step=$(basename "$file" .json | cut -d_ -f2)
    quality=$(jq '.quality_ki_value * 100' "$file")
    echo "$quality $step"
done | sort -rn | head -1
```

Output:
```
67.8 7500
```
→ Best quality (67.8%) was at step 7500!

---

## Summary

### What Changed
1. ✅ Added auto-save after every validation
2. ✅ Files saved to Learning directory
3. ✅ Named as `Statistik_STEP.json`
4. ✅ Contains ALL web UI data (including improvement)

### What You Get
- 📊 Complete training history
- 📈 Easy progress tracking
- 💾 Automatic archiving
- 📉 Data for analysis/plotting
- 🔍 Debugging information

### What You Need to Do
- ✅ Nothing! It works automatically!
- ✅ (Optional) Analyze the saved files
- ✅ (Optional) Create plots from data
- ✅ (Optional) Compare different training runs

---

The auto-save feature provides a complete, automatic archive of your training progress. Every validation automatically saves a snapshot of all training data to the Learning directory! 🎉
