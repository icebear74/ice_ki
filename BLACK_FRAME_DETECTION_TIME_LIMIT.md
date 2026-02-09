# Black Frame Detection Time Limit

## User Requirement

**German:**
> "Blackframe Detection : Mache die nur am anfang des films (ersten 10 sekunden) .. Danach nicht mehr .."

**English Translation:**
> "Black frame Detection: Only do this at the beginning of the film (first 10 seconds) .. After that no more .."

## Overview

Black frame detection is now **limited to the first 10 seconds** of each video. After 10 seconds, all frames are accepted without checking for black frames.

## Rationale

### Why Limit to First 10 Seconds?

1. **Where Black Frames Occur:**
   - Fade-ins at video start
   - Opening credits with black backgrounds
   - Pre-content darkness
   - Scene setup (rare after opening)

2. **Performance Benefits:**
   - Reduces processing time for long videos
   - No unnecessary file size checks after 10 seconds
   - No retry logic overhead for stable content

3. **Content Stability:**
   - After 10 seconds, video content is usually stable
   - Main content has started
   - Black frames are extremely rare in main content

## Implementation

### Parameters

```python
black_frame_detection_limit_seconds = 10.0  # Only check first 10 seconds
black_frame_threshold_kb = 15               # File size threshold
max_retries = 5                             # Max retry attempts
retry_jump_seconds = 1.0                    # Time jump per retry
```

### Logic Flow

```python
if saved:
    # Check if GT is a black frame (< 15 KB)
    # Only check during first 10 seconds of video
    if retry_time <= black_frame_detection_limit_seconds and \
       self._is_black_frame(gt_path, black_frame_threshold_kb):
        # DELETE and RETRY (black frame during first 10s)
        black_frames_detected += 1
        # ... delete files and retry ...
    else:
        # ACCEPT frame (either valid, or after 10s limit)
        if retry_time > black_frame_detection_limit_seconds:
            black_frames_skipped += 1
        # ... accept and continue ...
```

## Behavior

### Time-based Behavior

| Video Timestamp | Black Frame Detection | Action |
|----------------|----------------------|--------|
| 0.0 - 10.0 seconds | ✅ **ACTIVE** | Check file size, retry if < 15 KB |
| > 10.0 seconds | ❌ **SKIPPED** | Accept all frames without check |

### Example Scenarios

#### Scenario 1: Black Frame at 3 seconds
```
Time: 3.0s (within first 10s)
Detection: ACTIVE
File size: 4.2 KB
Result: DELETE and RETRY (< 15 KB threshold)
Action: Jump to 4.0s and try again
```

#### Scenario 2: Black Frame at 12 seconds
```
Time: 12.0s (after first 10s)
Detection: SKIPPED
File size: 4.2 KB (not checked)
Result: ACCEPT (detection disabled after 10s)
Action: Frame saved normally
```

#### Scenario 3: Valid Frame at 8 seconds
```
Time: 8.0s (within first 10s)
Detection: ACTIVE
File size: 856 KB
Result: ACCEPT (>= 15 KB threshold)
Action: Frame saved normally
```

#### Scenario 4: Valid Frame at 60 seconds
```
Time: 60.0s (after first 10s)
Detection: SKIPPED
File size: 1,200 KB (not checked)
Result: ACCEPT (detection disabled after 10s)
Action: Frame saved normally
```

## Logging

### Startup Log
```
INFO: Extracting 4000 patches for 2 categories
INFO: Black frame detection active for first 10.0 seconds only
```

### During Extraction (First 10s)
```
WARNING: Black frame detected at 2.30s (retry 0/5). Deleting and retrying...
WARNING: Black frame detected at 3.30s (retry 1/5). Deleting and retrying...
INFO: Valid frame found at 4.30s. Patch saved successfully.
```

### During Extraction (After 10s)
```
# No black frame warnings - detection is skipped
# Frames are accepted normally
```

### Final Statistics
```
INFO: Extraction complete for Video: 4000/4000 patches
INFO:   Black frames detected and handled: 12
INFO:   Frames saved without black frame check (after 10.0s): 3850
INFO:   master/large_720: 1000/1000 patches
INFO:   master/small_540: 500/500 patches
...
```

## Test Results

### Test Cases

```
✓ PASS  Time    0.0s: CHECK - Start of video
✓ PASS  Time    5.0s: CHECK - Within first 10 seconds
✓ PASS  Time    9.5s: CHECK - Just before 10 seconds
✓ PASS  Time   10.0s: CHECK - Exactly at 10 seconds
✓ PASS  Time   10.1s: SKIP  - Just after 10 seconds
✓ PASS  Time   15.0s: SKIP  - Well after 10 seconds
✓ PASS  Time   60.0s: SKIP  - 1 minute into video
✓ PASS  Time  300.0s: SKIP  - 5 minutes into video
```

### Condition Verification

```python
# Condition in code:
if retry_time <= 10.0 and is_black_frame(...):
    # Delete and retry
else:
    # Accept frame

# Examples:
Time  3.0s, black frame → DELETE and RETRY
Time  3.0s, valid frame → ACCEPT (not black)
Time 12.0s, black frame → ACCEPT (after 10.0s limit)
Time 12.0s, valid frame → ACCEPT (not black)
```

## Performance Impact

### Before (Unlimited Black Frame Detection)

```
Video duration: 1 hour (3600 seconds)
Stride: 3 seconds
Extractions: ~1200
Black frame checks: ~1200 (all frames)
File size checks: ~1200
```

### After (10-second Limit)

```
Video duration: 1 hour (3600 seconds)
Stride: 3 seconds
Extractions: ~1200
Black frame checks: ~4 (only first 10 seconds)
File size checks: ~4
Reduction: 99.7% fewer checks
```

### Time Savings

For a typical 1-hour video:
- **Before:** ~1200 file size checks + potential retries
- **After:** ~4 file size checks + potential retries
- **Savings:** 99%+ reduction in black frame processing overhead

## Edge Cases

### Very Short Videos (< 10 seconds)

```
Video duration: 8 seconds
Result: Black frame detection runs for entire video
Impact: No change (detection needed for full duration)
```

### Videos Starting Mid-content

```
Video type: TV show episode (no opening credits)
First 10s: Main content (no black frames)
Result: Detection runs but finds no black frames
Impact: Minimal overhead, quick validation
```

### Videos with Mid-content Black Frames

```
Scenario: Scene transition at 5 minutes with fade to black
Detection: SKIPPED (> 10 seconds)
Result: Black frame may be saved
Note: Rare occurrence, acceptable trade-off for performance
```

## Configuration

### Adjusting the Time Limit

To change the time limit, modify the parameter in `make_dataset_v2_uhd.py`:

```python
# Current default: 10 seconds
black_frame_detection_limit_seconds = 10.0

# Examples:
# No limit (always check):
black_frame_detection_limit_seconds = float('inf')

# Shorter limit (5 seconds):
black_frame_detection_limit_seconds = 5.0

# Longer limit (30 seconds):
black_frame_detection_limit_seconds = 30.0
```

### Disabling Time Limit

To disable the time limit and check all frames:

```python
black_frame_detection_limit_seconds = float('inf')
```

## Summary

### What Changed

- ✅ Black frame detection now limited to first 10 seconds
- ✅ After 10 seconds, all frames accepted without check
- ✅ Significant performance improvement for long videos
- ✅ Maintains quality control where needed (video start)

### Benefits

1. **Performance:** 99%+ reduction in file size checks
2. **Targeted:** Focuses on problem area (video start)
3. **Efficient:** No wasted processing on stable content
4. **Flexible:** Easy to adjust time limit if needed

### Files Modified

- `dataset_generator_v2/make_dataset_v2_uhd.py`
  - Added time limit parameter
  - Modified black frame check condition
  - Added tracking and logging

### Tests

- `test_black_frame_limit.py`
  - Verifies time-based logic
  - Tests 8 different timestamps
  - All tests passing (8/8)
