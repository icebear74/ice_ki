# Auto-Detection Debugging Guide

## Problem: Only Some Dataset Sizes Being Loaded

If you see a message like:
```
✓ Single-size training: 540
```

But you expect:
```
✓ Multi-size training enabled: 540, 720, 720_169
```

This guide will help you debug the issue.

---

## Diagnostic Output

With the new verbose logging, you'll see exactly what's being checked:

```
Checking for dataset sizes in: /mnt/data/training/datasetNeu/master/train
  Checking 540: /mnt/data/training/datasetNeu/master/train/540/GT
    ✓ Found 1456 files for size 540
  Checking 720: /mnt/data/training/datasetNeu/master/train/720/GT
    ⚠ Directory does not exist
  Checking 720_169: /mnt/data/training/datasetNeu/master/train/720_169/GT
    ⚠ Directory exists but no .png files found
```

---

## Common Issues and Solutions

### Issue 1: Directory Does Not Exist

**Message:**
```
  Checking 720: /mnt/data/training/datasetNeu/master/train/720/GT
    ⚠ Directory does not exist
```

**Cause:** The directory is missing from your filesystem.

**Solution:**
1. Check if the directory exists:
   ```bash
   ls -la /mnt/data/training/datasetNeu/master/train/
   ```

2. Expected structure:
   ```
   train/
   ├── 540/
   │   ├── GT/         # Ground truth (high-res) images
   │   └── LR/         # Low-res input images
   ├── 720/
   │   ├── GT/
   │   └── LR/
   └── 720_169/
       ├── GT/
       └── LR/
   ```

3. Create missing directories:
   ```bash
   mkdir -p /mnt/data/training/datasetNeu/master/train/720/GT
   mkdir -p /mnt/data/training/datasetNeu/master/train/720/LR
   mkdir -p /mnt/data/training/datasetNeu/master/train/720_169/GT
   mkdir -p /mnt/data/training/datasetNeu/master/train/720_169/LR
   ```

4. Run dataset extraction to populate the directories

---

### Issue 2: Directory Exists But No Files

**Message:**
```
  Checking 720_169: /mnt/data/training/datasetNeu/master/train/720_169/GT
    ⚠ Directory exists but no .png files found
```

**Cause:** Directory exists but contains no PNG files.

**Solution:**
1. Check directory contents:
   ```bash
   ls /mnt/data/training/datasetNeu/master/train/720_169/GT/
   ```

2. If empty, run dataset extraction for this size:
   ```bash
   # Run your dataset extraction script
   # It should populate 720_169/GT/ with PNG files
   ```

3. If files exist but with different extension:
   ```bash
   # Check for other extensions
   ls /mnt/data/training/datasetNeu/master/train/720_169/GT/*.jpg
   ls /mnt/data/training/datasetNeu/master/train/720_169/GT/*.PNG  # uppercase
   ```

4. Convert if needed:
   ```bash
   # Convert JPG to PNG if necessary
   for f in *.jpg; do convert "$f" "${f%.jpg}.png"; done
   ```

---

### Issue 3: Wrong Path in Configuration

**Message:**
```
Checking for dataset sizes in: /wrong/path/master/train
  Checking 540: /wrong/path/master/train/540/GT
    ⚠ Directory does not exist
  Checking 720: /wrong/path/master/train/720/GT
    ⚠ Directory does not exist
```

**Cause:** `runtime_config.json` has incorrect `data.root` or `data.dataset_name`.

**Solution:**
1. Check your `runtime_config.json`:
   ```json
   {
     "data": {
       "root": "/mnt/data/training/datasetNeu",  // Check this path!
       "dataset_name": "master"                   // Check this name!
     }
   }
   ```

2. Verify the path exists:
   ```bash
   ls -la /mnt/data/training/datasetNeu/master/train/
   ```

3. Update `runtime_config.json` with correct paths

4. Restart training (config changes require restart)

---

### Issue 4: Files Being Extracted During Training

**Scenario:** Dataset extraction running in parallel with training.

**What You'll See:**
```
# Initial startup:
  Checking 720: .../train/720/GT
    ⚠ Directory exists but no .png files found

# Later (after extraction completes):
📂 New training files detected for 720: +1234 files
🔄 Reloading 720 dataset...
✅ Reload successful: 0 → 1,234 files
```

**This is NORMAL!** The dynamic reload feature will pick up new files automatically every 100 steps.

---

## Verification Checklist

Use this checklist to verify your setup:

### 1. Check Directory Structure
```bash
# Should show 540, 720, 720_169 directories
ls -la /mnt/data/training/datasetNeu/master/train/
```

### 2. Check GT Files Exist
```bash
# Each should show PNG files
ls /mnt/data/training/datasetNeu/master/train/540/GT/*.png | head
ls /mnt/data/training/datasetNeu/master/train/720/GT/*.png | head
ls /mnt/data/training/datasetNeu/master/train/720_169/GT/*.png | head
```

### 3. Check LR Files Exist
```bash
# Each should show PNG files
ls /mnt/data/training/datasetNeu/master/train/540/LR/*.png | head
ls /mnt/data/training/datasetNeu/master/train/720/LR/*.png | head
ls /mnt/data/training/datasetNeu/master/train/720_169/LR/*.png | head
```

### 4. Count Files Per Size
```bash
echo "540 GT files: $(ls /mnt/data/training/datasetNeu/master/train/540/GT/*.png 2>/dev/null | wc -l)"
echo "720 GT files: $(ls /mnt/data/training/datasetNeu/master/train/720/GT/*.png 2>/dev/null | wc -l)"
echo "720_169 GT files: $(ls /mnt/data/training/datasetNeu/master/train/720_169/GT/*.png 2>/dev/null | wc -l)"
```

### 5. Check Configuration
```bash
# Verify paths in config
cat vsr_plusplus_NEU/runtime_config.json | grep -A 3 '"data"'
```

---

## Expected Output When Everything Works

When auto-detection finds all sizes correctly:

```
Checking for dataset sizes in: /mnt/data/training/datasetNeu/master/train
  Checking 540: /mnt/data/training/datasetNeu/master/train/540/GT
    ✓ Found 1456 files for size 540
  Checking 720: /mnt/data/training/datasetNeu/master/train/720/GT
    ✓ Found 1234 files for size 720
  Checking 720_169: /mnt/data/training/datasetNeu/master/train/720_169/GT
    ✓ Found 753 files for size 720_169
✓ Multi-size training enabled: 540, 720, 720_169

📊 Distribution (Automatic from file counts):
   540: 42.3%  |  720: 35.8%  |  720_169: 21.9%
```

---

## Troubleshooting Steps

If you still have issues after checking the above:

1. **Enable maximum verbosity:**
   - The new logging shows exactly what's being checked
   - Look for the "⚠" warning symbols
   - Copy the exact paths shown

2. **Manually verify paths:**
   ```bash
   # Copy the exact path from the log output and check it
   ls -la /mnt/data/training/datasetNeu/master/train/720/GT/
   ```

3. **Check permissions:**
   ```bash
   # Ensure you have read access to the directories
   ls -ld /mnt/data/training/datasetNeu/master/train/*/GT/
   ```

4. **Check for symbolic links:**
   ```bash
   # Follow symlinks to verify they point to valid locations
   ls -laL /mnt/data/training/datasetNeu/master/train/
   ```

5. **Try absolute paths in config:**
   ```json
   {
     "data": {
       "root": "/mnt/data/training/datasetNeu",  // Use absolute path
       "dataset_name": "master"
     }
   }
   ```

---

## Still Having Issues?

If you've checked everything above and still only see one size being loaded:

1. **Share the complete output** from training startup
2. **Share the output** of:
   ```bash
   tree -L 3 /mnt/data/training/datasetNeu/master/train/
   ```
3. **Share your runtime_config.json** (data section)

This will help identify the exact issue.

---

## Summary

The new verbose logging tells you exactly:
- ✅ Which paths are being checked
- ✅ Whether directories exist
- ✅ How many files were found
- ✅ Why a size was skipped

Use the color-coded output:
- 🟢 **Green (✓)** = Success, files found
- 🟡 **Yellow (⚠)** = Warning, check this

Most common fix: **Run dataset extraction** to populate the missing directories!
