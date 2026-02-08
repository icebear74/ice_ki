# Multi-Category Dataset Generator v2.0

A powerful dataset generator for creating multi-category training patches from UHD videos. Generates training data for three specialized models: **GENERAL** (Universal/Mastermodell), **SPACE**, and **TOON**.

## 🎯 Overview

This generator processes 530+ UHD videos and creates training patches with:
- **Dual LR generation**: Both 5-frame and 7-frame versions for flexibility
- **Multi-category support**: Single videos can contribute to multiple categories
- **Beautiful live GUI**: Real-time progress monitoring with rich terminal UI
- **Resume capability**: Checkpoint system for long-running generation (5+ days)
- **HDR tonemap**: Proper HDR-to-SDR conversion for high-quality patches
- **Realistic training data**: Accepts all frames including scenes with cuts

## 📁 Structure

```
dataset_generator_v2/
├── README.md                      # This file
├── generator_config.json          # Video list with categorization
├── make_dataset_multi.py          # Main generator with GUI
├── monitor_generator.py           # Live monitoring tool
└── utils/
    ├── progress_tracker.py        # Checkpoint management
    └── format_definitions.py      # Format specifications
```

## 📦 Output Structure

All output goes to `/mnt/data/training/dataset/` with **VSR++ compatible** structure:

```
/mnt/data/training/dataset/
│
├── Universal/Mastermodell/Learn/     # GENERAL category
│   ├── Patches/                      # 540×540 (VSR++ Training)
│   │   ├── GT/                       # Ground truth frames
│   │   ├── LR/                       # 5-frame LR stack (180×900) ← VSR++ uses this!
│   │   └── LR_7frames/              # 7-frame LR stack (optional extended)
│   │
│   ├── Patches_Medium169/            # 720×405 (16:9)
│   │   ├── GT/
│   │   ├── LR/                       # 5-frame (240×675)
│   │   └── LR_7frames/              # 7-frame (240×945)
│   │
│   ├── Patches_Large/                # 720×720
│   ├── Patches_XLarge169/            # 1440×810 (16:9)
│   ├── Patches_FullHD/               # 1920×1080
│   │
│   └── Val/                          # Validation (manual selection)
│       ├── GT/                       # Ground truth only
│       └── LR/                       # Optional (falls back to Patches/LR)
│
├── Space/SpaceModel/Learn/           # SPACE category
│   ├── Patches/
│   │   ├── GT/
│   │   ├── LR/                       # ← VSR++ compatible
│   │   └── LR_7frames/
│   ├── Patches_XLarge169/
│   ├── Patches_FullHD/
│   └── Val/GT/
│
└── Toon/ToonModel/Learn/             # TOON category
    ├── Patches/
    │   ├── GT/
    │   ├── LR/                       # ← VSR++ compatible
    │   └── LR_7frames/
    ├── Patches_Medium169/
    └── Val/GT/
```

**VSR++ Compatibility:**
- Training expects: `Patches/GT/` and `Patches/LR/` (5-frame stack)
- LR format: 5 frames stacked vertically (frames -2, -1, 0, +1, +2)
- GT: Middle frame (frame 0) aligns with center of LR stack
- Extended 7-frame stacks available in `LR_7frames/` for future use

## 🎬 Video Categories

### Premium Multi-Category Videos
Videos that contribute to multiple categories with weighted distribution:

- **Ready Player One**: general(0.4) + space(0.3) + toon(0.3)
- **Avatar (2009)**: general(0.3) + space(0.4) + toon(0.3)
- **Avatar 2**: general(0.3) + space(0.4) + toon(0.3)
- **Dune (2021)**: general(0.5) + space(0.5)
- **Dune Part Two**: general(0.5) + space(0.5)
- **Interstellar**: general(0.3) + space(0.7)
- **Passengers**: general(0.3) + space(0.7)

### Category-Specific Videos

**GENERAL** (1.0 weight):
- Planet Earth 2 (all 6 episodes)
- Oppenheimer
- Top Gun Maverick
- Gladiator II
- The Dark Knight Trilogy
- Forrest Gump

**SPACE** (1.0 weight):
- Star Trek films and Strange New Worlds
- Star Wars films, Obi-Wan, and Ahsoka
- 2001: A Space Odyssey
- Arrival
- Event Horizon

**TOON** (1.0 weight):
- Shrek series
- How to Train Your Dragon
- Despicable Me
- The Super Mario Bros Movie
- Barbie (0.7 toon + 0.3 general)

## 🔧 Format Distribution

Each category uses different format distributions optimized for the content type:

**GENERAL** (diverse sizes):
- small_540 (540×540): 45%
- medium_169 (720×405): 35%
- large_720 (720×720): 20%

**SPACE** (focus on large formats):
- small_540: 30%
- xlarge_1440 (1440×810): 45%
- fullhd_1920 (1920×1080): 25%

**TOON** (smaller sufficient):
- small_540: 65%
- medium_169: 35%

## 🎯 VSR++ Training Integration

This generator is **fully compatible** with VSR++ model training:

### Directory Structure
VSR++ expects:
```
dataset_root/
├── Patches/
│   ├── GT/         # Ground truth frames
│   └── LR/         # 5-frame LR stacks (VSR++ uses this!)
└── Val/
    ├── GT/         # Validation ground truth
    └── LR/         # Optional (falls back to Patches/LR)
```

### LR Stack Format
- **5 frames** stacked vertically (e.g., 180×900 for 540×540 GT)
- Frame order: [-2, -1, 0, +1, +2] relative to GT
- Middle frame (0) temporally aligned with GT
- Each frame: 1/3 size of GT (180×180 for 540×540)

### Training Usage
```python
# VSR++ dataset.py automatically finds:
dataset = VSRDataset(
    dataset_root="/mnt/data/training/dataset/Universal/Mastermodell/Learn",
    mode='Patches'  # Uses Patches/GT and Patches/LR
)

# For validation:
val_dataset = VSRDataset(
    dataset_root="/mnt/data/training/dataset/Universal/Mastermodell/Learn",
    mode='Val'      # Uses Val/GT, falls back to Patches/LR
)
```

### Extended 7-Frame Support
- Optional `LR_7frames/` directories also created
- Contains 7-frame stacks for future extended models
- Not used by current VSR++ training (uses 5-frame)

## 🚀 Installation

1. Ensure Python 3.8+ is installed
2. Install dependencies:
```bash
cd dataset_generator_v2
pip install -r ../requirements.txt
```

Required packages:
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- rich >= 13.0.0
- tqdm >= 4.66.0

## 📋 Usage

### Basic Generation

Start the generator:
```bash
cd dataset_generator_v2
python make_dataset_multi.py
```

The generator will:
1. Load configuration from `generator_config.json`
2. Check for existing checkpoints (resume if found)
3. Create output directory structure
4. Process videos sequentially with beautiful live UI
5. Save checkpoints every 10 extractions
6. Handle Ctrl+C gracefully (saves progress)

### Monitoring Progress

In a separate terminal, run the monitor:
```bash
cd dataset_generator_v2
python monitor_generator.py
```

The monitor displays:
- Overall progress (videos completed, ETA)
- Current video status
- Category statistics (images created, disk usage)
- Checkpoint information
- Live updates every 2 seconds

### Resume After Interruption

Simply run the generator again:
```bash
python make_dataset_multi.py
```

It will automatically:
- Detect the checkpoint file
- Resume from the last completed video
- Continue from the last successful frame in partially-processed videos

## ⚙️ Configuration

Edit `generator_config.json` to customize:

### Base Settings
```json
{
  "base_settings": {
    "base_frame_limit": 3000,        // Frames per video (base)
    "max_workers": 12,                // Parallel extraction threads
    "val_percent": 0.0,               // No auto-validation
    "output_base_dir": "/mnt/data/training/dataset",
    "temp_dir": "/mnt/data/training/dataset/temp",
    "min_file_size": 10000,           // Minimum valid frame size
    "scene_diff_threshold": 45,       // (Unused - kept for compatibility)
    "max_retry_attempts": 10,         // Retries per extraction
    "retry_skip_seconds": 60          // Skip ahead on retry
  }
}
```

### Adding Videos
```json
{
  "videos": [
    {
      "name": "Video Name",
      "path": "/full/path/to/video.mkv",
      "categories": {
        "general": 0.5,               // 50% weight to general
        "space": 0.5                  // 50% weight to space
      }
    }
  ]
}
```

**Notes:**
- Category weights should sum to 1.0 (or any value - will be normalized)
- Each category gets DIFFERENT random crops from the same frames
- Total extractions = `base_frame_limit * sum(category_weights)`

## 🎨 Features

### Dual LR Generation

For each extraction, the generator creates:
1. **1× GT**: Middle frame (frame #3) cropped to target size
2. **1× LR_5frames**: 5 frames stacked vertically (frames 1-5)
3. **1× LR_7frames**: 7 frames stacked vertically (all frames)

Example for 540×540 format:
- GT: 540×540 PNG
- LR_5frames: 180×900 PNG (5 frames × 180 height)
- LR_7frames: 180×1260 PNG (7 frames × 180 height)

### HDR Tonemap

Uses the same HDR-to-SDR tonemap filter as the original generator:
```
zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,
tonemap=tonemap=mobius,zscale=t=bt709:m=bt709,format=yuv420p,
scale=1920:1080:flags=lanczos
```

### Frame Acceptance

Accepts all extracted frames including scenes with cuts:
- Extracts 7 frames centered at timestamp
- Validates frame count and file sizes
- Accepts all frames (realistic training data with scene changes)
- Model learns to handle cuts naturally during training

### Smart Retry Logic

For each extraction attempt:
1. Extract 7 frames at timestamp
2. Validate frame count and file sizes
3. Accept all valid frames (including scenes with cuts)
4. If validation fails, skip ahead +60s and retry
5. Maximum 10 attempts per extraction

### Checkpoint System

Status saved to `/mnt/data/training/dataset/.generator_status.json`:
```json
{
  "version": "2.0",
  "status": "running",
  "workers": 12,
  "progress": {
    "total_videos": 497,
    "completed_videos": 145,
    "current_video_index": 145
  },
  "category_stats": {
    "general": {
      "videos_processed": 42,
      "images_created": 28456,
      "target": 80000,
      "disk_usage_gb": 38.2
    }
  },
  "video_checkpoints": {
    "145": {
      "status": "in_progress",
      "last_frame_idx": 1847,
      "extractions_done": 1847
    }
  }
}
```

## 📊 Expected Output

**Total Images**: ~165,000
- GENERAL: ~80,000 images across all formats
- SPACE: ~55,000 images (focus on large formats)
- TOON: ~30,000 images (smaller formats)

**Total Disk**: ~416 GB
- GENERAL: ~146 GB
- SPACE: ~195 GB
- TOON: ~35 GB

**Generation Time**: 5-7 days (with 12 workers)

## ✅ Validation Data

**IMPORTANT**: The generator does NOT create validation images automatically!

The `Val/GT/` directories are created but left EMPTY. To create validation data:

1. Manually select high-quality GT images
2. Copy them to the respective `Val/GT/` directory
3. Training will generate LR versions on-the-fly

This ensures validation data is carefully curated.

## 🔍 Troubleshooting

### Generator crashes or hangs
- Check `/mnt/data/training/dataset/.generator_status.json` for last state
- Verify video files exist at specified paths
- Check disk space (need ~450 GB free)
- Reduce `max_workers` if system is overloaded

### Low success rate
- Increase `max_retry_attempts` (default: 10)
- Check video quality and encoding

### Resume not working
- Verify `.generator_status.json` exists and is valid JSON
- Check file permissions
- Delete status file to start fresh

### Monitor shows no data
- Verify generator is running
- Check status file path matches
- Ensure both scripts use same `status_file` path

### Disk space issues
- Generator uses `/mnt/data/training/dataset/` (not in backup)
- Temp files in `temp/` directory (cleaned after each extraction)
- Monitor disk usage via the monitor tool

## 🛠️ Advanced Usage

### Custom Video List

Edit `generator_config.json` and add your videos:
```json
{
  "name": "My Custom Video",
  "path": "/path/to/video.mkv",
  "categories": {
    "general": 1.0
  }
}
```

### Adjust Worker Count

Edit `max_workers` in `generator_config.json` before starting:
```json
"max_workers": 24  // Use more threads for faster generation
```

Or during runtime (requires code modification for live adjustment).

### Change Output Directory

Edit `output_base_dir` in `generator_config.json`:
```json
"output_base_dir": "/custom/path/dataset"
```

### Skip Specific Videos

Remove or comment out videos in the config file, or set all category weights to 0:
```json
{
  "name": "Skip This",
  "path": "/path/to/video.mkv",
  "categories": {}
}
```

## 📝 File Naming Convention

Generated files follow this pattern:
```
patch_{video_clean_name}_idx{frame_idx}{format_suffix}.png
```

Examples:
- `patch_planet_earth_s01e01_idx0042_fullhd.png`
- `patch_star_trek_into_darkness_idx1234_xl169.png`
- `patch_shrek_idx0567.png` (no suffix for small_540)

Where:
- `video_clean_name`: Alphanumeric only (special chars → underscore)
- `frame_idx`: Sequential extraction number
- `format_suffix`: `_med169`, `_large`, `_xl169`, `_fullhd`, or empty

## 🔒 Safety Features

- **Graceful shutdown**: Ctrl+C saves progress before exit
- **Atomic saves**: Checkpoints saved after each update
- **Validation checks**: File size and format verification
- **Temp cleanup**: Temporary files removed after each extraction
- **Resume capability**: Never lose progress on crashes

## 📚 Technical Details

### LR Stack Format

LR images are vertically stacked frames:
```
Frame 0 (or 1)  ─┐
Frame 1 (or 2)   │
Frame 2 (or 3)   ├─ Stacked vertically
Frame 3 (or 4)   │
Frame 4 (or 5)   │
[Frame 5]        │ (7-frame only)
[Frame 6]       ─┘ (7-frame only)
```

### Memory Usage

Per worker thread:
- 7 frames × 1920×1080×3 bytes ≈ 43 MB
- Temporary disk: ~50 MB per extraction
- Total: ~(workers × 50 MB) + overhead

With 12 workers: ~600 MB + Python overhead

### CPU Usage

- Video decoding: CPU-intensive (ffmpeg)
- Image processing: OpenCV (optimized)
- Expected: 80-95% CPU with 12 workers

## 🎓 Related Documentation

- Original generator: `/make_dataset.py`
- Training script: `/train.py`
- Model architecture: `/vsr_plus_plus/`

## 📄 License

Part of the ice_ki project.

## 🤝 Contributing

To add more videos:
1. Edit `generator_config.json`
2. Add video entry with path and categories
3. Run generator (will resume or process new videos)

## 💡 Tips

- Start with a small subset to test configuration
- Monitor disk I/O if generation is slow
- Use SSD for temp directory if available
- Run monitor in `tmux` or `screen` for persistent monitoring
- Back up `.generator_status.json` periodically

---

**Happy dataset generation! 🚀**
