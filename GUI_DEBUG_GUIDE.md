# GUI Debug Guide

## Current Status

You have commit `749a4c4` but report the GUI is not updating.

## What Changed (Latest Commit: f5d2be6)

Added diagnostic logging to help debug the issue. **You need to pull and restart!**

## Steps to Debug

### 1. Pull Latest Code

```bash
cd /mnt/data/ice_ki
git pull origin copilot/fix-import-error-category-utils
```

### 2. Stop Any Running Process

If the dataset generator is already running, **STOP IT** (Ctrl+C).

### 3. Start Fresh

```bash
python3 dataset_generator_v2/make_dataset_v2_uhd.py
```

### 4. Watch for Diagnostic Messages

**At startup, you should see:**
```
================================================================================
🎨 TERMINAL GUI ENABLED - Real-time progress display active
   GUI will update every ~2 seconds during processing
================================================================================
```

**When starting a video:**
```
================================================================================
🎬 STARTING VIDEO: Avatar (2009).mkv (38/466)
   Category targets: {'master': 5000, 'space': 1500}
================================================================================
```

**During processing (every ~2 seconds):**
```
[GUI UPDATE #1] Patches: 0
[DRAWING GUI...]
[GUI DRAWN]

[GUI UPDATE #2] Patches: 7
[DRAWING GUI...]
[GUI DRAWN]
```

## What to Report

Please tell me:

1. ✅ **Do you see "🎨 TERMINAL GUI ENABLED"?**
   - YES → GUI is initialized
   - NO → GUI not enabled (problem found!)

2. ✅ **Do you see "GUI UPDATE #1, #2, #3..."?**
   - YES → Update method is being called
   - NO → Update method not called (problem found!)

3. ✅ **Do you see "DRAWING GUI..." and "GUI DRAWN"?**
   - YES → Draw function is executing
   - NO → Draw function not executing (problem found!)

4. ✅ **Does the GUI actually update on screen?**
   - YES → Everything working!
   - NO → Terminal display issue (different problem)

5. ✅ **Do you see any "⚠️ GUI UPDATE ERROR" messages?**
   - YES → Share the error message
   - NO → Good, no errors

## Expected Result

**If working correctly:**
```
╔════════════════════════════════════════════════════════╗
║          DATASET GENERATOR - FORTSCHRITT               ║
╚════════════════════════════════════════════════════════╝

▸ AKTUELLER FILM
────────────────────────────────────────────────────────
  Film 38 / 466
  Avatar (2009).mkv

  Master        125 /  5000 (  2.5%)  ███░░░░░░░░░░░░░░░
  Space          45 /  1500 (  3.0%)  ███░░░░░░░░░░░░░░░
```

**Numbers should increase every few seconds!**

## Common Issues

### Issue: Old code still running
**Solution:** Stop and restart after pulling latest code

### Issue: Terminal doesn't support ANSI codes
**Solution:** Use a terminal that supports colors (most modern terminals do)

### Issue: Output buffering
**Solution:** Already fixed with multiple flush() calls

## Next Steps

Run the code and report back what diagnostic messages you see. This will help us pinpoint exactly where the problem is!
