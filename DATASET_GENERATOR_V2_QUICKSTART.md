# Dataset Generator V2 - Quick Start Guide

## Quick Start (5 Minutes)

### 1. Setup Configuration

Edit `dataset_generator_v2/generator_config_v2.json`:

```json
{
  "dataset_name": "master",
  "root_path": "/mnt/data/training/datasetNeu",
  
  "source": {
    "categories": {
      "master": {
        "video_dir": "/path/to/your/UHD/videos",
        "extensions": [".mkv", ".mp4"]
      },
      "universal": {
        "video_dir": "/path/to/your/HD/videos",
        "extensions": [".mkv", ".mp4"]
      }
    },
    "category_weights": {
      "master": 0.25,    // 25% from UHD
      "universal": 0.75  // 75% from HD
    }
  },
  
  "processing": {
    "total_patches": 100000  // Total patches to generate
  }
}
```

### 2. Generate Dataset

```bash
cd dataset_generator_v2
python make_dataset_v2_clean.py
```

**First Run Output:**
```
✨ Created new generation: gen_20260208_233733
🔍 Scanning videos for metadata...
  master: Found 2 videos
  universal: Found 45 videos
✅ Total videos in metadata cache: 47
📊 Calculating patch distribution...
📹 Processing: Planet_Earth_S01E06.mkv
   Target: 5,692 patches
[████████████████████] 5692/5692 100%
```

### 3. Resume After Interruption

Press `Ctrl+C` to stop at any time. Progress is auto-saved every 100 patches.

Run again to resume:
```bash
python make_dataset_v2_clean.py
```

**Resume Output:**
```
✅ Resuming from existing state: gen_20260208_233733
Current Progress:
Progress: 3,450 / 100,000 patches (3.45%)
  master: 863 / 25,000
  universal: 2,587 / 75,000
```

### 4. Use in Training

```python
from core.dataset import VSRDataset

train_dataset = VSRDataset(
    root="/mnt/data/training/datasetNeu",
    dataset_name="master",
    size_key="540",  # or '720', '720_169'
    mode="train"
)
```

## Key Features

### ✅ UHD Quality
- Preserves full 4K resolution (3840×2160)
- Only tonemaps HDR→SDR (no downscaling!)
- Random crops from full UHD for maximum detail

### ✅ Smart Distribution
- Category-based: 25% UHD, 75% HD (configurable)
- Duration-based: Longer videos get more patches
- Fair and balanced

### ✅ Resume Capability
- Auto-saves every 100 patches
- Resume from exact position
- Safe interruption at any time

### ✅ 7-Frame Only
- Simplified implementation
- Horizontal stacking: 240×1680 for 7 frames

## Directory Structure

After generation:
```
/mnt/data/training/datasetNeu/master/
├── patches/
│   ├── 720/
│   │   ├── GT/       # 720×720 patches
│   │   └── LR/       # 240×1680 (7 frames)
│   ├── 540/
│   │   ├── GT/       # 540×540 patches
│   │   └── LR/       # 180×1260 (7 frames)
│   └── 720_169/
│       ├── GT/       # 720×405 patches (16:9)
│       └── LR/       # 240×945 (7 frames)
└── val/
    ├── 720/GT/       # Copy your validation GT here
    └── 720_169/GT/   # Copy your validation GT here
```

## Configuration Options

### Essential Settings

```json
{
  "dataset_name": "master",           // Dataset name
  "root_path": "/mnt/data/...",      // Output directory
  
  "source": {
    "categories": {
      "master": {...},                // UHD sources
      "universal": {...}              // HD sources
    },
    "category_weights": {
      "master": 0.25,                // 25% from UHD
      "universal": 0.75              // 75% from HD
    }
  },
  
  "processing": {
    "total_patches": 100000,         // Total patches
    "n_frames": 7,                   // Always 7
    "scene_threshold": 30.0,         // Scene detection
    "stride": 3                      // Frame skip
  }
}
```

### Advanced Options

```json
{
  "random_seed": 42,                 // For reproducible generation
  "ffmpeg_timeout": 120,             // FFmpeg timeout (seconds)
  "ffprobe_timeout": 60,             // ffprobe timeout (seconds)
  
  "output_patches": {
    "720": {
      "enabled": true,               // Enable/disable size
      "gt_size": [720, 720],
      "lr_size": [240, 240]
    }
  }
}
```

## Monitoring Progress

### Check State File
```bash
# View progress
jq '.progress' generation_state.json

# Output:
# {
#   "total_patches": 100000,
#   "completed_patches": 3450,
#   "percentage": 3.45
# }
```

### Check Per-Category Progress
```bash
jq '.category_distribution' generation_state.json
```

## Troubleshooting

### Video Not Found
- Check video paths in config
- Verify file extensions match
- Ensure read permissions

### FFmpeg Timeout
- Increase `ffmpeg_timeout` in config
- Default is 120s, try 180s or 240s

### Memory Issues
- Reduce workers in config
- Process one size at a time (disable others)

### Resume Not Working
- Don't delete `generation_state.json`
- Don't change config (creates new generation)

## Training Integration

### Size Distribution
Edit `vsr_plusplus_NEU/runtime_config.json`:
```json
{
  "size_distribution": {
    "720": 0.0,      // Validation only
    "540": 0.65,     // 65% training
    "720_169": 0.35  // 35% training
  }
}
```

### Validation Setup
1. Copy validation GT images to:
   - `/mnt/data/training/datasetNeu/master/val/720/GT/`
   - `/mnt/data/training/datasetNeu/master/val/720_169/GT/`

2. Generator will use matching LR from patches (automatic fallback)

## Performance Tips

### Faster Generation
- Use SSD/NVMe storage
- Increase workers (8-16 on powerful systems)
- Process local files (not network shares)

### Quality vs Speed
- Scene threshold: Lower = more patches, faster
- Blur threshold: Higher = more patches, faster
- Stride: Higher = fewer patches, much faster

## Comparison: Old vs New

| Feature | Old | New |
|---------|-----|-----|
| Quality | HD → crop | **UHD → crop** |
| Cropping | Center only | **Random (anywhere)** |
| Resume | ❌ No | **✅ Yes** |
| Distribution | Manual | **Automatic (weighted)** |
| Size Keys | `small_540` | **`540`** |
| State | None | **Complete caching** |

## Next Steps

1. ✅ Generate dataset
2. ✅ Copy validation GT to `val/` directories
3. ✅ Start training with new dataset
4. 📈 Monitor training progress
5. 🎉 Enjoy better results!

## Need Help?

See full documentation: `DATASET_GENERATOR_V2_REWRITE.md`

Run tests: `python test_dataset_generator_v2_rewrite.py`

Check examples in config file for all options.
