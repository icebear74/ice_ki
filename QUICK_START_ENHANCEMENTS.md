# VSR++ Enhancements - Quick Start Guide

## What's New?

This enhancement package adds **runtime configuration management** and **advanced monitoring** to your VSR++ training system.

---

## 🔥 Immediate Benefits

### 1. Change Settings While Training
**No more restarts!** Edit `runtime_config.json` and changes apply automatically within 10 steps.

```bash
# Example: Increase learning rate mid-training
nano /mnt/data/training/.../Learn/runtime_config.json
# Change "max_lr": 0.00015 to "max_lr": 0.0002
# Save and wait ~10 steps - done!
```

### 2. Monitor Loss Distribution
**New Web UI** shows stacked bar charts visualizing:
- Which loss components are weighted how much
- Which losses are contributing most to total loss
- Real-time updates every 3 seconds

Access at: `http://localhost:5050` (or your configured port)

### 3. Smarter Plateau Detection
**No more false plateaus!** New system considers:
- Loss improvements (adaptive thresholds)
- Quality improvements (KI quality %)
- EMA trends (noise reduction)
- Grace periods (temporary dips)

---

## 📋 Quick Commands

### View Current Config
```bash
cat /mnt/data/training/.../Learn/runtime_config.json
```

### Change Plateau Patience
```bash
# Edit the file
nano /mnt/data/training/.../Learn/runtime_config.json

# Find and change this line:
"plateau_patience": 350   # Increase from default 250
```

### Save Manual Checkpoint
Via Web UI: Click "💾 Save Checkpoint Now" button

### List Config Snapshots
```python
from vsr_plus_plus.systems.runtime_config import RuntimeConfigManager
config = RuntimeConfigManager('/path/to/runtime_config.json', base_config)
snapshots = config.list_snapshots()
print(f"Found {len(snapshots)} config snapshots")
```

---

## 🎯 Common Use Cases

### Case 1: Training Too Conservative
**Symptom:** Loss decreasing but very slowly
**Solution:**
```json
{
  "max_lr": 0.0002,        // Increase from 0.00015
  "plateau_patience": 400  // Give it more time
}
```

### Case 2: Too Many False Plateaus
**Symptom:** Frequent plateau resets when loss is still improving
**Solution:**
```json
{
  "plateau_patience": 350,        // Increase from 250
  "plateau_safety_threshold": 1000  // Increase from 800
}
```

### Case 3: Rebalance Loss Weights
**Symptom:** Want more gradient emphasis
**Solution:**
```json
{
  "l1_weight_target": 0.55,    // Reduce from 0.6
  "grad_weight_target": 0.25   // Increase from 0.2
}
// Sum must equal 0.95-1.05
```

---

## 🛡️ Safety Features

### Automatic Validation
All changes validated before applying:
- ✅ Range checks (each parameter has min/max)
- ✅ Type checks (numbers are numbers, etc.)
- ✅ Sum checks (weights must sum to ~1.0)

### Config Snapshots
Every checkpoint includes config snapshot:
- Roll back to any previous configuration
- Compare configs between checkpoints
- Full reproducibility

### Pre-Change Validation
Save validation snapshot before risky changes:
```python
# Via Web UI: Click "🔍 Run Validation Snapshot"
# Creates: Statistik_STEP_before_change.json
```

---

## 📊 Web UI Features

### Stacked Bar Charts
**Two bars side-by-side:**

1. **Weight Distribution** - Shows configured weights (L1: 60%, MS: 20%, etc.)
2. **Loss Distribution** - Shows actual loss values (which contribute most)

**Colors:**
- 🔴 Red = L1 Loss
- 🟠 Orange = Multi-Scale Loss
- 🟣 Purple = Gradient Loss
- 🔵 Cyan = Perceptual Loss

### Peak Activity Monitor
Shows peak layer activity on 0.0-2.0 scale with color zones:
- 🟢 Green (0.0-0.5): Normal
- 🟡 Yellow (0.5-1.0): Moderate
- 🟠 Orange (1.0-1.5): High
- 🔴 Red (1.5-2.0+): Extreme (check stability!)

---

## ⚠️ Important Notes

### Can Change Anytime (Safe)
- `plateau_patience`, `plateau_safety_threshold`
- `max_lr`, `min_lr`
- `cooldown_duration`
- `initial_grad_clip`

### Change Carefully (Need Balance)
- `l1_weight_target`, `ms_weight_target`
- `grad_weight_target`, `perceptual_weight_target`
- **Must sum to 0.95-1.05**

### Cannot Change (Need Restart)
- `n_feats`, `n_blocks`
- `batch_size`, `num_workers`
- `accumulation_steps`

---

## 🆘 Troubleshooting

### Changes Not Applying?
1. Check file has valid JSON syntax
2. Wait at least 10 steps
3. Check terminal for error messages
4. Verify file permissions (must be writable)

### Validation Error?
```
⚠️  Invalid value for 'plateau_patience': 2000 (must be in range [50, 1000])
```
**Solution:** Use a value within the allowed range

### Weights Don't Sum to 1.0?
```
⚠️  Weight sum validation failed: 1.3 (should be 0.95-1.05)
```
**Solution:** Adjust weights so they sum between 0.95 and 1.05

---

## 📚 Full Documentation

- **Complete Guide:** [docs/RUNTIME_CONFIG.md](docs/RUNTIME_CONFIG.md)
- **Implementation Details:** [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Tests & Examples:** [tests/test_runtime_config.py](tests/test_runtime_config.py)

---

## 🎓 Best Practices

1. **One change at a time** - Easier to identify effects
2. **Small increments** - Don't jump from 250 to 1000, go 250→300→350
3. **Save snapshots before big changes** - Use validation snapshot feature
4. **Monitor the stacked bars** - Watch how changes affect loss distribution
5. **Document your changes** - Add comments in JSON explaining why

---

## ✅ Quick Validation

Test your setup:
```python
# Import and test
from vsr_plus_plus.systems.runtime_config import RuntimeConfigManager

# Should work without errors
config = RuntimeConfigManager('/tmp/test_config.json', {
    'plateau_patience': 250,
    'max_lr': 1.5e-4
})

print("✅ Runtime config system ready!")
```

---

**Version:** 1.0.0  
**Last Updated:** February 2026  
**Status:** Production Ready
