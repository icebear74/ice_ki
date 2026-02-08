# Dataset Generator v2.0 - GUI Preview

## Main Generator Display

When running `python make_dataset_multi.py`, you'll see:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    DATASET GENERATOR v2.0 - MULTI-CATEGORY                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 OVERALL PROGRESS
├─ Total Videos: 116
├─ Completed: 42 (36.2%)
├─ Current: Planet Earth 2 - S01E04 - Deserts
├─ Remaining: 74 videos
├─ Elapsed: 1d 8h 32m
├─ ETA: 2d 14h 18m
└─ Workers: 12

🎬 CURRENT VIDEO
├─ Path: /mnt/data/video/SerieUHD/Planet Earth 2/S01E04.mkv
├─ Extractions: 1847 / 3000 (61.6%)
├─ Success Rate: 94.3%
└─ Status: Running

📦 CATEGORY PROGRESS
┌─────────────┬──────────┬──────────┬─────────┬──────────────────────┐
│  Category   │  Videos  │  Images  │  Target │      Progress        │
├─────────────┼──────────┼──────────┼─────────┼──────────────────────┤
│ GENERAL     │  28/68   │  18,456  │  80,000 │ ██░░░░░░ 23.1%       │
│ SPACE       │  12/35   │   8,893  │  55,000 │ █░░░░░░░ 16.2%       │
│ TOON        │   5/18   │   2,102  │  30,000 │ ░░░░░░░░  7.0%       │
└─────────────┴──────────┴──────────┴─────────┴──────────────────────┘

💾 DISK USAGE
├─ GENERAL: 28.2 GB
├─ SPACE: 38.8 GB
├─ TOON: 3.1 GB
└─ Total: 70.1 GB

⚙️  CONTROLS
├─ [Ctrl+C] Save & Exit
└─ Press 'q' to quit
```

## Monitor Display

When running `python monitor_generator.py` in a separate terminal:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        DATASET GENERATOR MONITOR                             ║
║                          Status: RUNNING                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 OVERALL PROGRESS
├─ Total Videos: 116
├─ Completed: 42 (36.2%)
├─ Current Index: 42
├─ Remaining: 74 videos
├─ Elapsed: 1 day, 8:32:15
├─ ETA: 2 days, 14:18:43
├─ Workers: 12
└─ Last Update: 2025-02-09 15:47:32

🎬 CURRENT VIDEO
├─ Index: 42
├─ Path: .../Planet Earth 2/S01E04.mkv
├─ Extractions: 1847 / 3000 (61.6%)
└─ Last Frame: 1847

                           📦 CATEGORY STATISTICS
┌─────────────┬──────────┬──────────┬─────────┬────────────────────┬──────────┐
│  Category   │  Videos  │  Images  │  Target │     Progress       │ Disk(GB) │
├─────────────┼──────────┼──────────┼─────────┼────────────────────┼──────────┤
│ GENERAL     │    28    │  18,456  │  80,000 │ ████░░░░░░░░░░ 23% │   28.20  │
│ SPACE       │    12    │   8,893  │  55,000 │ ███░░░░░░░░░░░ 16% │   38.80  │
│ TOON        │     5    │   2,102  │  30,000 │ █░░░░░░░░░░░░░  7% │    3.10  │
│ TOTAL       │          │  29,451  │ 165,000 │              17.8% │   70.10  │
└─────────────┴──────────┴──────────┴─────────┴────────────────────┴──────────┘

💾 DISK USAGE BREAKDOWN
├─ GENERAL: 28.20 GB
├─ SPACE: 38.80 GB
├─ TOON: 3.10 GB
└─ Total: 70.10 GB

🔖 CHECKPOINT SUMMARY
├─ Total Checkpoints: 43
├─ Completed: 42
├─ In Progress: 1
└─ Status File: .../.generator_status.json
```

## Progress States

### Initial Start
```
🚀 Initializing Dataset Generator v2.0...
✓ Config loaded: 116 videos
✓ Creating output directories...
✓ Checking for existing checkpoints...
→ No checkpoint found, starting fresh
```

### Resume from Checkpoint
```
🚀 Initializing Dataset Generator v2.0...
✓ Config loaded: 116 videos
✓ Creating output directories...
✓ Checking for existing checkpoints...
📍 Resuming from video 42 (Planet Earth 2 - S01E04)
→ Last frame: 1847/3000
```

### Graceful Shutdown
```
^C
Received shutdown signal. Saving progress...
✓ Progress saved to .generator_status.json
✓ Video 42: 1847/3000 frames completed
✓ Safe to resume later with: python make_dataset_multi.py
```

### Completion
```
✅ Dataset generation complete!

Final Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Videos Processed: 116
Total Images Created: 165,234
Total Disk Usage: 416.8 GB
Total Time: 6 days, 14:32:18

Category Breakdown:
  GENERAL: 80,123 images (146.2 GB)
  SPACE:   55,891 images (195.4 GB)  
  TOON:    29,220 images (35.2 GB)

Output: /mnt/data/training/dataset/
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Error Handling

### Video Not Found
```
⚠️  Warning: Video file not found
    Path: /mnt/data/video/FilmeUHD/Missing.mkv
    Video: Missing Video Name
    → Skipping and continuing...
```

### Low Success Rate Warning
```
⚠️  Warning: Low success rate on current video
    Video: Problem Video
    Success Rate: 45.2% (< 80% threshold)
    → Consider checking video file quality
    → Continuing with next video...
```

### Disk Space Warning
```
⚠️  Warning: Low disk space
    Available: 45.2 GB
    Estimated needed: 350 GB remaining
    → Consider freeing up disk space
```

## Features Highlighted

### Multi-Category Distribution
```
🎬 Processing: Avatar (2009)
   Categories: general(0.3) + space(0.4) + toon(0.3)
   
   ✓ Extracted 7 frames @ 00:45:32
   ✓ Scene validation passed (diff: 12.4 < 45)
   ✓ Saved to GENERAL: Patches/GT/, LR_5frames/, LR_7frames/
   ✓ Saved to SPACE: Patches_XLarge169/GT/, LR_5frames/, LR_7frames/
   ✓ Saved to TOON: Patches_Medium169/GT/, LR_5frames/, LR_7frames/
   → Different crops per category ✓
```

### Retry Logic
```
Attempt 1: Failed (only 5 frames extracted, need 7)
Attempt 2: Success (all 7 frames valid) ✓
```

### Checkpoint Saves
```
✓ Checkpoint saved (every 10 extractions)
  Video 42: frame 1850/3000
  GENERAL: 18,502 images
  SPACE: 8,917 images
  TOON: 2,115 images
```

## Performance Metrics

```
System Resource Usage:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CPU:    84.2% (12 workers)
Memory:  8.4 GB / 32 GB (26.3%)
Disk I/O: Read: 125 MB/s, Write: 45 MB/s
Temp:    2.1 GB

Performance:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Extraction Speed: 8.2 patches/min
Success Rate:     94.3%
Retry Rate:       12.1%
Average Attempts: 1.4
```

## Tips

- The GUI updates in real-time as extraction progresses
- Monitor shows live statistics refreshed every 2 seconds
- Both scripts can run simultaneously without interference
- Checkpoints are atomic - safe to interrupt at any time
- Color coding: Green = good, Yellow = warning, Red = error

