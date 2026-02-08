# 7-Frame VSR Training System

Complete implementation of the 7-frame VSR training system with adaptive batch management, live configuration, and full persistence.

## ✅ Implementation Status

### Core Components
- ✅ **Model Architectures** - 5-frame and 7-frame models with 26 blocks, 72 features
- ✅ **Adaptive Batch System** - Automatic batch/accumulation calculation per image size
- ✅ **Runtime Configuration** - Live config changes with validation and persistence
- ✅ **Size Tracking** - Track training progress per size category with auto-save
- ✅ **Checkpoint Integration** - Size stats and config included in checkpoints
- ✅ **Terminal UI Extensions** - Size distribution panel with progress bars
- ✅ **Web GUI** - Configuration interface with sliders and VRAM monitoring
- ✅ **Startup Validation** - Config validation before training starts

## Architecture Overview

### 1. Model Architectures

#### 5-Frame Model (`core/model_5frame.py`)
- **Input**: [B, 5, 3, H, W] (5 frames)
- **Output**: [B, 3, H*3, W*3] (upscaled center frame)
- **Configuration**: 72 features, 26 blocks
- **Architecture**: Bidirectional propagation with center-frame initialization

#### 7-Frame Model (`core/model_7frame.py`)
- **Input**: [B, 7, 3, H, W] (7 frames)
- **Output**: [B, 3, H*3, W*3] (upscaled center frame)
- **Configuration**: 72 features, 26 blocks
- **Architecture**: Bidirectional propagation with center-frame initialization

### 2. Adaptive Batch System (`systems/adaptive_batch.py`)

**VRAM Budget**: < 6.5 GB (Plex transcoding reserve)

**Safe Configuration** (from tested configs):
- ALL sizes use batch=1 (safest approach)
- Accumulation = effective_batch / 1
- 540×540 @ B1: 3.77 GB ✅
- 720×405 @ B1: 3.77 GB ✅
- 720×720 @ B1×A6: ~6.0 GB ✅

**Key Features**:
- `AdaptiveBatchCalculator` class
- `calculate_batch_config(gt_size, effective_batch)` - Calculate for single size
- `calculate_all_configs(effective_batch)` - Calculate for all sizes
- `validate_config()` - Check VRAM limits
- Returns: {batch, accum, effective, vram_est, safe}

**Example Usage**:
```python
from vsr_plusplus_NEU.systems.adaptive_batch import AdaptiveBatchCalculator

calculator = AdaptiveBatchCalculator()

# Calculate for effective batch size = 6
configs = calculator.calculate_all_configs(effective_batch=6)
calculator.print_config_table(configs)

# Validate
is_valid, errors = calculator.validate_all_configs(configs)
```

### 3. Runtime Configuration (`systems/runtime_config.py`)

**Configuration Structure**:
```json
{
  "model": {
    "n_frames": 7,
    "n_feats": 72,
    "n_blocks": 26,
    "precision": "float32"
  },
  "training": {
    "effective_batch_size": 6,
    "adaptive_batch": {
      "small_540": {"batch": 1, "accum": 6},
      "medium_169": {"batch": 1, "accum": 6},
      "large_720": {"batch": 1, "accum": 6}
    }
  },
  "size_distribution": {
    "small_540": 0.65,
    "medium_169": 0.35,
    "large_720": 0.00
  }
}
```

**Key Features**:
- Auto-save on updates
- Validation (sum checks, VRAM checks)
- `validate()` method returns (is_valid, errors_list)
- Thread-safe operations
- Snapshot management

**Example Usage**:
```python
from vsr_plusplus_NEU.systems.runtime_config import EnhancedRuntimeConfigManager

# Create manager
config_path = "/path/to/runtime_config.json"
manager = EnhancedRuntimeConfigManager(config_path, use_new_structure=True)

# Validate
is_valid, errors = manager.validate()

# Update effective batch size
manager.update_effective_batch_size(8)

# Update size distribution
manager.update_size_distribution({
    'small_540': 0.70,
    'medium_169': 0.30,
    'large_720': 0.00
})
```

### 4. Size Tracking System (`systems/size_tracking.py`)

**Features**:
- Track images trained per size category
- Save to `/mnt/data/training/size_tracking.json`
- Auto-save every 100 steps
- Checkpoint integration

**Example Usage**:
```python
from vsr_plusplus_NEU.systems.size_tracking import SizeTracker

# Create tracker
tracker = SizeTracker(save_path="/mnt/data/training/size_tracking.json")

# Set targets
distribution = {
    'small_540': 0.65,
    'medium_169': 0.35,
    'large_720': 0.00,
}
tracker.update_targets(distribution, total_target=10000)

# Record training
tracker.record_batch('small_540', batch_size=1, step=100)

# Get summary
print(tracker.get_summary())
```

### 5. Web GUI (`web/config_7frame.html`, `web/config_api.py`)

**Features**:
- Batch size slider (4-12) with live preview
- Auto-calculated batch/accum table per size
- VRAM estimates with color coding (green/yellow/red)
- Size distribution sliders (0-100%) with sum validation
- Progress bars showing trained/target per size

**API Endpoints**:
- `/api/update_batch_config` - Update effective batch size
- `/api/update_size_distribution` - Update size distribution
- `/api/size_stats` - Get size tracking statistics
- `/api/batch_preview` - Preview batch config without saving

**Access**: http://localhost:8080/config_7frame.html

### 6. Terminal GUI Extensions (`utils/ui_terminal.py`)

**New Functions**:
- `make_size_bar(trained, target, width)` - Create progress bar for size tracking
- `print_size_distribution_panel(size_stats, ui_width)` - Print size distribution panel
- `format_size_stats_compact(size_stats)` - Format stats for status line

**Example Usage**:
```python
from vsr_plusplus_NEU.utils.ui_terminal import print_size_distribution_panel

# Print size distribution panel
print_size_distribution_panel(size_stats, ui_width=120)
```

### 7. Config Validation on Startup (`train.py`)

**Features**:
- `validate_startup_config()` function
- Check size_distribution sum == 1.0 (±0.01)
- Check VRAM limits
- Exit with error message if validation fails
- Display config summary

**Example**:
```python
from vsr_plusplus_NEU.train import validate_startup_config

# Validate before training
if not validate_startup_config(runtime_config_manager):
    print("Configuration validation failed!")
    sys.exit(1)
```

### 8. Checkpoint Integration (`systems/checkpoint_manager.py`)

**Enhanced Features**:
- Include size_stats in checkpoint
- Include runtime_config snapshot in checkpoint
- Restore both on checkpoint load
- `load_size_tracking_from_checkpoint()` method

**Example Usage**:
```python
from vsr_plusplus_NEU.systems.checkpoint_manager import CheckpointManager

checkpoint_mgr = CheckpointManager("/path/to/checkpoints")

# Save with size tracking
checkpoint_mgr.save_checkpoint(
    model, optimizer, scheduler, step, metrics, log_file,
    runtime_config=runtime_config,
    size_tracker=size_tracker
)

# Load size tracking from checkpoint
checkpoint_data = torch.load(checkpoint_path)
checkpoint_mgr.load_size_tracking_from_checkpoint(checkpoint_data, size_tracker)
```

## Dataset Structure

**Base**: `/mnt/data/training/datasetNeu/master/`

**Training Data**:
- `train/7frames/small_540/` - 540×540 GT, 180×180 LR
- `train/7frames/medium_169/` - 720×405 GT, 240×135 LR
- `train/7frames/large_720/` - 720×720 GT, 240×240 LR

**Validation Data**:
- `val/7frames/large_720/` - Always 720×720 for validation

## Validation Configuration

- Always use 720×720 images
- Batch size: 1
- Accumulation: 6
- No augmentation
- Memory cleanup preserved

## Testing

Run the comprehensive test suite:

```bash
cd /home/runner/work/ice_ki/ice_ki
python3 vsr_plusplus_NEU/test_7frame_system.py
```

**Tests Include**:
- ✅ Adaptive Batch Calculator
- ✅ Runtime Configuration Manager
- ✅ Size Tracking System
- ✅ Terminal UI Functions
- ⚠️  Model Imports (requires PyTorch)

## Critical Rules

1. **ALL changes ONLY in `vsr_plusplus_NEU/`** - DO NOT touch `vsr_plus_plus/`
2. **Use batch=1 for ALL sizes** - Safest for VRAM
3. **Persist everything** - Config changes, size stats, checkpoints
4. **Validate config on startup** - Abort if invalid
5. **Web GUI must validate** - Sum == 100%, VRAM check

## Success Criteria

- ✅ Models created and importable (72 features, 26 blocks)
- ✅ Adaptive batch calculator works
- ✅ Runtime config persists to JSON
- ✅ Size tracking saves automatically
- ✅ Web GUI functional with sliders
- ✅ Terminal GUI shows size stats
- ✅ Startup validation works
- ✅ Checkpoints include all state
- ✅ All under 6.5 GB VRAM

## Files Modified/Created

### New Files
- `systems/adaptive_batch.py` - Adaptive batch calculator
- `systems/size_tracking.py` - Size tracking system
- `web/templates/config_7frame.html` - Web GUI configuration interface
- `test_7frame_system.py` - Comprehensive test suite
- `README_7FRAME.md` - This documentation

### Modified Files
- `core/model_5frame.py` - Updated to 72 features, 26 blocks
- `core/model_7frame.py` - Updated to 72 features, 26 blocks
- `systems/runtime_config.py` - Enhanced with new structure
- `systems/checkpoint_manager.py` - Added size tracking integration
- `systems/__init__.py` - Added new module exports
- `utils/ui_terminal.py` - Added size distribution panel functions
- `web/config_api.py` - Added new API endpoints
- `train.py` - Added startup validation

## Quick Start

1. **Configure Runtime Settings**:
   ```bash
   # Edit /mnt/data/training/runtime_config.json
   # or use Web GUI at http://localhost:8080/config_7frame.html
   ```

2. **Validate Configuration**:
   ```bash
   python3 vsr_plusplus_NEU/test_7frame_system.py
   ```

3. **Start Training**:
   ```bash
   python3 vsr_plusplus_NEU/train.py
   ```

4. **Monitor Progress**:
   - Terminal: Size distribution panel in UI
   - Web: http://localhost:8080/config_7frame.html

## Support

For issues or questions, please refer to:
- This README for usage examples
- `test_7frame_system.py` for working code examples
- Individual module docstrings for detailed API documentation
