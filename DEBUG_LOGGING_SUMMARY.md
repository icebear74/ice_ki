# Debug Logging Implementation Summary

## Overview
Successfully implemented comprehensive debug logging for the dataset generator to diagnose why it stops after processing only 1 video (out of 467) without error messages.

## Changes Made

### 1. Configuration File (`generator_config.json`)
Added two new settings to `base_settings`:
```json
{
  "enable_debug_logging": true,
  "debug_log_path": "/mnt/data/training/dataset/generator_debug.log"
}
```

### 2. Source Code (`make_dataset_multi.py`)

#### Imports
- Added `import logging` to enable Python's logging system

#### Logger Initialization (`__init__`)
- Created `_setup_logger()` method that:
  - Reads config settings (`enable_debug_logging`, `debug_log_path`)
  - Creates log directory if it doesn't exist
  - Sets up file handler with DEBUG level
  - Uses format: `%(asctime)s - %(levelname)s - %(message)s`
  - Returns NullHandler if logging is disabled
- Initializes logger: `self.logger = self._setup_logger()`
- Makes logger globally available: `sys.logger = self.logger`
- Logs initialization: 
  - `INFO`: Total video count
  - `DEBUG`: First 5 video names

#### Main Loop Logging (`run()` method)
**Loop Start:**
- `INFO`: "=== STARTING GENERATOR ==="
- `INFO`: Resume index

**Loop Iterations:**
- `DEBUG`: Each loop iteration with current/total count
- `WARNING`: When stopped by `self.running=False`
- `INFO`: Video being processed (index + name)
- `INFO`: When video is already completed (skipping)
- `DEBUG`: Before calling `process_video()`
- `DEBUG`: Return value from `process_video()`
- `INFO`: Video completed successfully
- `DEBUG`: Moving to next video

**Exception Handling:**
- Inner try-catch for each video with `ERROR` logging + `exc_info=True`
- Continues to next video instead of crashing
- Outer try-catch for fatal errors with `CRITICAL` logging + `exc_info=True`
- Finally block with `INFO` logging for cleanup

**Loop End:**
- `INFO`: "=== MAIN LOOP ENDED ===" with count of processed videos
- `INFO`: "Setting status to 'finished'"

#### Video Processing Logging (`process_video()`)
- `DEBUG`: Function entry with video index and name
- `DEBUG`: Video file path
- `DEBUG`: Whether video file exists
- `ERROR`: When video file not found
- `DEBUG`: Total extractions and video duration
- `DEBUG`: Extraction progress every 100 frames
- `INFO`: Video completion with success count
- Exception handler logs errors with `exc_info=True`

#### Frame Extraction Logging
**`extract_full_resolution_frames()`:**
- `DEBUG`: Full FFmpeg command being executed

**`extract_with_retry()`:**
- `DEBUG`: Number of frames extracted on each attempt
- `DEBUG`: When 0 frames extracted (failed attempt)

#### Global Exception Handler
```python
def exception_handler(exc_type, exc_value, exc_traceback):
    """Log uncaught exceptions."""
    if hasattr(sys, 'logger'):
        sys.logger.critical("UNCAUGHT EXCEPTION", exc_info=(exc_type, exc_value, exc_traceback))
    else:
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = exception_handler
```

## Logging Levels Used

| Level | Usage |
|-------|-------|
| **DEBUG** | Detailed flow, FFmpeg commands, internal state, iteration details |
| **INFO** | Video start/complete, major milestones, loop start/end |
| **WARNING** | Generator stopped by flag, skipped videos |
| **ERROR** | Missing videos, exceptions in video processing |
| **CRITICAL** | Fatal exceptions, uncaught exceptions |

## Features

✅ **Toggleable**: Can be enabled/disabled via config without code changes
✅ **Configurable Path**: Log file location set in config
✅ **Full Stack Traces**: Uses `exc_info=True` for all exceptions
✅ **Non-Intrusive**: Uses NullHandler when disabled
✅ **Granular**: Different log levels for different scenarios
✅ **Comprehensive**: Logs every critical decision point
✅ **Global Handler**: Catches even uncaught exceptions

## Expected Log Output

When the generator runs, the log file will show:

1. **Initialization**
   ```
   2024-01-01 12:00:00 - INFO - Initializing generator with 467 videos
   2024-01-01 12:00:00 - DEBUG - First 5 videos: ['Video1', 'Video2', ...]
   ```

2. **Main Loop Start**
   ```
   2024-01-01 12:00:01 - INFO - === STARTING GENERATOR ===
   2024-01-01 12:00:01 - INFO - Resume from video index: 0
   ```

3. **Each Video**
   ```
   2024-01-01 12:00:02 - DEBUG - --- Loop iteration 0 / 467 ---
   2024-01-01 12:00:02 - INFO - Processing video 0: Movie Title
   2024-01-01 12:00:02 - DEBUG - Calling process_video() for video 0
   2024-01-01 12:00:02 - DEBUG - process_video(0): Movie Title
   2024-01-01 12:00:02 - DEBUG - Video path: /path/to/video.mkv
   2024-01-01 12:00:02 - DEBUG - Video exists: True
   2024-01-01 12:00:03 - DEBUG - FFmpeg command: nice -n 19 ffmpeg -y ...
   2024-01-01 12:00:03 - DEBUG - Extracted 7 frames on attempt 1
   2024-01-01 12:00:04 - DEBUG - Video 0: extraction 100/3000
   ...
   2024-01-01 12:05:00 - INFO - Video 0 COMPLETED: 2500/3000 successful
   2024-01-01 12:05:00 - DEBUG - Moving to next video (idx=1)
   ```

4. **If Generator Stops**
   ```
   2024-01-01 12:05:01 - WARNING - Generator stopped by self.running=False at video 1
   2024-01-01 12:05:01 - INFO - === MAIN LOOP ENDED === (processed 2 videos)
   ```

5. **If Exception Occurs**
   ```
   2024-01-01 12:05:01 - ERROR - EXCEPTION in video 1: ValueError: Invalid frame
   Traceback (most recent call last):
     ...
   ```

## Diagnostic Capabilities

The logging will reveal:

1. **Which video is being processed** when the loop exits
2. **Why the loop exits**:
   - Normal completion (all videos processed)
   - `self.running=False` (user stop or signal)
   - Exception in specific video
   - Fatal exception in main loop
3. **FFmpeg commands** being executed
4. **Frame extraction success/failure** for each attempt
5. **Full stack traces** for any errors
6. **Exact flow** between video transitions

## Testing

Created verification scripts:

1. **`verify_logging.py`** - Verifies code structure and configuration
   - Checks all logging statements are present
   - Verifies config settings
   - Provides implementation summary

2. **`test_debug_logging.py`** - Unit tests for logging system
   - Tests logger initialization
   - Tests logging can be disabled
   - Tests config settings

Run verification:
```bash
python3 verify_logging.py
```

## Usage

The logging is **enabled by default** and will automatically write to:
```
/mnt/data/training/dataset/generator_debug.log
```

To **disable logging**, edit `generator_config.json`:
```json
{
  "base_settings": {
    "enable_debug_logging": false
  }
}
```

To **change log location**, edit `generator_config.json`:
```json
{
  "base_settings": {
    "debug_log_path": "/your/custom/path/debug.log"
  }
}
```

## Next Steps

1. Run the dataset generator normally
2. Check the log file after it stops
3. The log will show exactly:
   - Which video it stopped on
   - Why it stopped (normal vs error vs interrupt)
   - Any exceptions that occurred
   - The complete execution flow

This will diagnose why the generator stops after 1 video instead of continuing to process all 467 videos.

## Files Modified

- `dataset_generator_v2/generator_config.json` - Added logging config
- `dataset_generator_v2/make_dataset_multi.py` - Added logging implementation

## Files Created

- `verify_logging.py` - Verification script
- `test_debug_logging.py` - Unit tests
- `DEBUG_LOGGING_SUMMARY.md` - This document
