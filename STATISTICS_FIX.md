# Fix: Menu Option 10 (Statistics) Crash

## Problem

Menu choice '10' (Show statistics) was crashing with an AttributeError:

```
⚠️  Error processing menu choice '10': 'list' object has no attribute 'keys'
Traceback (most recent call last):
  File "/mnt/data/ice_ki/dataset_generator_v2/video_manager.py", line 620, in main
    manager.show_statistics()
  File "/mnt/data/ice_ki/dataset_generator_v2/video_manager.py", line 318, in show_statistics
    for cat in cats.keys():
AttributeError: 'list' object has no attribute 'keys'
```

## Root Cause

The `show_statistics()` method was written for the old dict-based category format (with weights), but categories have been migrated to a simple list format. The code was calling `.keys()` on what is now a list object.

## Solution

Updated the `show_statistics()` method to:

1. **Handle list-based categories** (current format)
2. **Maintain backwards compatibility** with dict-based categories (legacy)
3. **Use correct default value** (`[]` instead of `{}`)

### Code Changes

**Before (line 314-318):**
```python
for video in self.videos:
    cats = video.get('categories', {})  # Wrong default
    if not cats:
        unassigned += 1
    else:
        for cat in cats.keys():  # ERROR: lists don't have .keys()
            category_counts[cat] += 1
```

**After:**
```python
for video in self.videos:
    cats = video.get('categories', [])  # Correct default
    if not cats:
        unassigned += 1
    else:
        # Handle both list and dict formats
        if isinstance(cats, list):
            for cat in cats:  # Works with lists
                if cat in category_counts:
                    category_counts[cat] += 1
        else:
            # Legacy dict format
            for cat in cats.keys():  # Still works with legacy dicts
                if cat in category_counts:
                    category_counts[cat] += 1
```

## Testing

### Test Suite Created

`test_show_statistics.py` - Comprehensive test coverage:

✓ **List-based categories** (current format)
```python
{"categories": ["master", "space", "toon"]}
```

✓ **Dict-based categories** (legacy format)
```python
{"categories": {"master": 0.5, "space": 0.5}}
```

✓ **Mixed formats** (backwards compatibility)
✓ **Empty categories** (unassigned count)
✓ **Category counts accuracy**

### Test Results

```
✓ PASS: show_statistics() executed without error
✓ PASS: Correct count for 'master' (3 videos)
✓ PASS: Correct count for 'space' (1 video)
✓ PASS: Correct count for 'toon' (2 videos)
✓ PASS: Correct count for unassigned (1 video)
✓ PASS: show_statistics() handles mixed formats
```

### Manual Verification

Menu option 10 now works correctly:

```bash
$ python3 video_manager.py
Choice: 10

============================================================
STATISTICS
============================================================

Total videos: 466
Unassigned: 0

Category assignments:
  master         :  466 videos (target: 150000)
  space          :   84 videos (target: 60000)
  toon           :   34 videos (target: 50000)
  universal      :  358 videos (target: 50000)
```

## Benefits

✅ **Menu option 10 works** - No more crashes  
✅ **Accurate statistics** - Correct video counts per category  
✅ **Backwards compatible** - Handles both old and new formats  
✅ **Well tested** - Comprehensive test suite  
✅ **Consistent** - Matches other category format updates  

## Related Changes

This fix is part of the broader migration from dict-based categories (with weights) to list-based categories (simple presence/absence). Other methods previously fixed:

- `print_video_list()` 
- `interactive_select_videos()`
- `remove_from_category()`

Now `show_statistics()` is also updated to use the new format.
