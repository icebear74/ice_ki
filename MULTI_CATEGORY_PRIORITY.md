# Multi-Category Priority Sorting and Enhanced Progress Tracking

## Overview

This implementation enhances video processing with:
1. **Multi-category priority sorting** - Videos in multiple categories processed first
2. **JSON order processing** - Dataset generator processes videos in exact JSON order
3. **Enhanced progress tracking** - Real-time category statistics after each video

## Problem Statement (German Requirements)

The user requested:
1. Dataset generator should extract in the exact order of the JSON
2. When saving JSON, videos in multiple categories should come first (with master on top)
3. Example: Video in "Master, Space" comes before video only in "Master"
4. Order priority: Videos with most categories first, then alphabetically by category, then by name
5. After each extracted video, show percentage and absolute numbers for all 4 categories

## Implementation

### 1. Video Sorting in video_manager.py

**New Sorting Logic:**
```python
def sort_key(video):
    cats = get_video_categories(video)
    # Primary: Number of categories (descending - negative for reverse)
    num_cats = -len(cats) if cats else 999
    # Secondary: First category alphabetically (master comes first)
    first_cat = cats[0] if cats else 'zzz_no_category'
    # Tertiary: Video name
    name = video.get('name', '').lower()
    return (num_cats, first_cat, name)
```

**Sorting Priority:**
1. **Primary**: Number of categories (descending)
   - Videos with 3 categories before videos with 2 categories
   - Videos with 2 categories before videos with 1 category
2. **Secondary**: First category name (alphabetically)
   - "master" comes before "space", "toon", "universal"
3. **Tertiary**: Video name (alphabetically)

**Example Order:**
```
1. "Video A" - [master, space, toon]     (3 categories)
2. "Video B" - [master, space]           (2 categories, master first)
3. "Video C" - [master, universal]       (2 categories, master first)
4. "Video D" - [space, toon]             (2 categories, space first)
5. "Video E" - [master]                  (1 category)
6. "Video F" - [space]                   (1 category)
7. "Video G" - [toon]                    (1 category)
8. "Video H" - [universal]               (1 category)
```

### 2. Dataset Generator Processing Order

**Before:**
```python
# Sort videos by priority (0 first, 255 last)
random.seed(42)  # Reproducible
for i, video in enumerate(self.videos):
    video['_sort_random'] = random.random()
self.videos.sort(key=lambda v: (v.get('priority', 255), v['_sort_random']))
```

**After:**
```python
# Videos are already sorted in JSON by multi-category priority
# Process them in exact JSON order (no additional sorting)
# This ensures videos with multiple categories are processed first
```

**Changes in make_dataset_v2_uhd.py:**
- Removed lines 99-105 (priority and random sorting)
- Videos now processed in exact JSON order
- Log message updated to reflect this

### 3. Category Progress Tracking

**New Method in progress_tracker.py:**
```python
def get_all_category_progress(self) -> str:
    """Get formatted progress string for all categories."""
    lines = []
    lines.append("📊 Category Progress:")
    
    categories = sorted(self.status["category_stats"].keys())
    
    for category in categories:
        stats = self.status["category_stats"][category]
        created = stats.get("images_created", 0)
        target = stats.get("target", 0)
        percent = (created / target * 100) if target > 0 else 0
        
        lines.append(f"   {category:12s}: {created:6d}/{target:6d} patches ({percent:5.1f}%)")
    
    return "\n".join(lines)
```

**Integration in make_dataset_v2_uhd.py:**
After each video is processed (line 2266-2268), the progress is logged:
```python
# Log category progress after each video
progress_info = self.tracker.get_all_category_progress()
self.logger.info(f"\n{progress_info}\n")
```

**Output Format:**
```
📊 Category Progress:
   master      :    250/150000 patches (  0.2%)
   space       :    100/ 60000 patches (  0.2%)
   toon        :     80/ 50000 patches (  0.2%)
   universal   :    120/ 50000 patches (  0.2%)
```

## Benefits

### Multi-Category Priority
✅ Important multi-category videos processed first  
✅ Better dataset quality early in the process  
✅ Predictable processing order  
✅ Master category prioritized (alphabetically first)  

### JSON Order Processing
✅ No random shuffling  
✅ Reproducible results  
✅ Respects manual JSON ordering  
✅ Simpler codebase (removed complexity)  

### Enhanced Progress Tracking
✅ Real-time visibility into all categories  
✅ Both absolute numbers and percentages  
✅ Easy to monitor progress  
✅ Helps identify when targets are met  

## Usage

### Using Video Manager
```bash
cd dataset_generator_v2
python3 video_manager.py

# Assign videos to categories
# Choose 's' to save

# Output:
# ✓ Saved to config.json (videos sorted by: multi-category priority, category, then title)
```

The saved JSON will have videos ordered with multi-category videos first.

### Running Dataset Generator
```bash
cd dataset_generator_v2
python3 make_dataset_v2_uhd.py

# Videos will be processed in the exact order they appear in JSON
# After each video, you'll see:
# [PROCESSING] Complete: Video Name - 150 patches created
# 
# 📊 Category Progress:
#    master      :   1500/150000 patches (  1.0%)
#    space       :    800/ 60000 patches (  1.3%)
#    toon        :    450/ 50000 patches (  0.9%)
#    universal   :    600/ 50000 patches (  1.2%)
```

## Testing

### Test: Multi-Category Sorting
```bash
python3 test_multi_category_sorting.py
```

Validates:
- Videos with 3 categories come first
- Videos with 2 categories come next
- Videos with 1 category come last
- Within same category count, sorted alphabetically by first category
- Within same category, sorted alphabetically by name

### Test: Progress Tracking
```bash
python3 test_progress_tracking.py
```

Validates:
- Progress display includes all categories
- Absolute numbers shown (created/target)
- Percentages calculated correctly
- Formatting is clean and aligned

### Test: Existing Tests Still Pass
```bash
python3 test_category_sorting.py
python3 test_category_assignment_improvements.py
```

All existing tests pass - no regressions introduced.

## Migration Notes

### For Existing Configurations
No migration needed! The new sorting is applied when you save in the video manager:
1. Open video manager
2. Press 's' to save
3. Videos will be resorted automatically

### For Existing Datasets
The dataset generator will now process videos in JSON order. If you want multi-category videos first:
1. Re-save your configuration using video manager
2. Restart dataset generation (or it will continue from where it left off)

## Technical Details

### Files Modified
- `video_manager.py` - Lines 66-91 (sorting logic)
- `make_dataset_v2_uhd.py` - Lines 96-104 (removed sorting), 2267-2270 (added progress logging)
- `utils/progress_tracker.py` - Lines 198-217 (new method)

### Files Added
- `test_multi_category_sorting.py` - Test for new sorting logic
- `test_progress_tracking.py` - Test for progress display

### Backward Compatibility
✅ Fully backward compatible  
✅ Existing JSON files work as-is  
✅ Old sorting was subset of new sorting (single-category videos still sorted correctly)  
✅ No breaking changes  

## Performance Impact

**None** - The changes are purely organizational:
- Sorting happens once at save time (same as before)
- Progress logging is negligible overhead
- No change to actual video processing logic

## Future Enhancements

Potential improvements:
- Add option to configure sorting preference in settings
- Allow manual video reordering in video manager UI
- Add ETA calculations based on category progress
- Export progress reports to CSV/JSON
