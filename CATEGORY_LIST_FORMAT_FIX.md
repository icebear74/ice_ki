# AttributeError Fix: List-Format Categories in video_manager.py

## Problem Statement

User encountered an `AttributeError` when using menu choice '6' (Interactive multi-select):

```
⚠️  Error processing menu choice '6': 'list' object has no attribute 'items'
Traceback (most recent call last):
  File "/mnt/data/ice_ki/dataset_generator_v2/video_manager.py", line 554, in main
    selected_ids = manager.interactive_select_videos(filter_str if filter_str else None)
  File "/mnt/data/ice_ki/dataset_generator_v2/video_manager.py", line 219, in interactive_select_videos
    cat_str = ', '.join([f"{k}:{v:.1f}" for k, v in cats.items()])[:28] if cats else ""
AttributeError: 'list' object has no attribute 'items'
```

## Root Cause

The video manager system was updated to use **list-based categories** (no weights) as documented in `category_utils.py`:

```python
# New format (list):
{"name": "Video 1", "categories": ["master", "space", "toon"]}

# Old format (dict with weights):
{"name": "Video 1", "categories": {"master": 0.5, "space": 0.3, "toon": 0.2}}
```

However, three methods in `video_manager.py` still expected dictionary format with weights and tried to:
1. Call `.items()` on categories (which are now lists)
2. Format categories as `{category}:{weight}` pairs
3. Normalize weights when removing categories

## Solution

Updated three methods to use the existing `format_categories_display()` and `normalize_categories()` utilities from `category_utils.py`, which handle both list and dict formats.

### Fix 1: `print_video_list()` method (lines 138-140)

**Before:**
```python
cats = video.get('categories', {})
cat_str = ', '.join([f"{k}:{v:.2f}" for k, v in cats.items()])  # ❌ Fails if cats is a list
if not cat_str:
    cat_str = "⚠️  <WILL BE SKIPPED - no categories>"
```

**After:**
```python
cats = video.get('categories', [])  # ✓ Default to list
cat_str = format_categories_display(cats)  # ✓ Handles both formats
```

### Fix 2: `interactive_select_videos()` method (line 216-219)

**Before:**
```python
cats = video.get('categories', {})
cat_str = ', '.join([f"{k}:{v:.1f}" for k, v in cats.items()])[:28]  # ❌ Fails if cats is a list
```

**After:**
```python
cats = video.get('categories', [])  # ✓ Default to list
cat_str = format_categories_display(cats)[:28] if cats else ""  # ✓ Handles both formats
```

### Fix 3: `remove_from_category()` method (lines 284-292)

**Before:**
```python
cats = self.videos[idx].get('categories', {})
if category in cats:
    del cats[category]  # ❌ Fails if cats is a list
    count += 1
    # Renormalize remaining categories
    total = sum(cats.values())  # ❌ Lists don't have .values()
    if total > 0:
        cats = {k: v/total for k, v in cats.items()}  # ❌ Lists don't have .items()
        self.videos[idx]['categories'] = cats
```

**After:**
```python
cats = self.videos[idx].get('categories', [])  # ✓ Default to list
# Handle both dict and list formats
cats_list = normalize_categories(cats)  # ✓ Convert to list if needed

if category in cats_list:
    cats_list.remove(category)  # ✓ List operation
    self.videos[idx]['categories'] = cats_list
    count += 1
```

## Benefits

✅ **Uses existing utilities**: Leverages `format_categories_display()` and `normalize_categories()` from `category_utils.py`

✅ **Backward compatible**: Still works if any legacy dict-format data exists

✅ **Simplified code**: No manual formatting or weight normalization needed

✅ **Consistent**: Matches the design documented in `category_utils.py`

✅ **No weight logic**: Categories are now simple presence/absence (no fractional weights)

## Testing

### Manual Testing
All menu options tested and working:
- ✅ Menu choice 1 (List videos) - Categories display correctly
- ✅ Menu choice 6 (Interactive select) - **FIXED!** No more AttributeError
- ✅ Menu choice 8 (Remove from category) - Correctly removes from lists

### Automated Testing

Created comprehensive test suite: `test_category_list_format.py`

**Test 1: format_categories_display() handles both formats**
- ✅ List format: `['master', 'space']` → `"master, space"`
- ✅ Dict format: `{'master': 0.5, 'space': 0.5}` → `"master, space"`
- ✅ Empty: `[]` → `"⚠️ <WILL BE SKIPPED - no categories>"`

**Test 2: print_video_list() with list-format categories**
- ✅ No AttributeError when displaying videos
- ✅ Categories shown correctly in output

**Test 3: interactive_select_videos() with list-format categories**
- ✅ No AttributeError when displaying selection UI
- ✅ Categories shown correctly for each video

**Test 4: remove_from_category() with list-format categories**
- ✅ Categories removed correctly from lists
- ✅ No weight normalization attempted
- ✅ Empty list after removing last category

### Regression Testing
- ✅ All existing tests pass (`test_video_manager_improvements.py`)
- ✅ No security vulnerabilities (CodeQL scan)

## Code Review Feedback Addressed

**Issue**: Default value for `get('categories', {})` was a dict
**Fix**: Changed all occurrences to `get('categories', [])` to match list format

## Files Changed

1. **Modified**: `dataset_generator_v2/video_manager.py`
   - Lines 138-140: Use `format_categories_display()` in `print_video_list()`
   - Lines 216-219: Use `format_categories_display()` in `interactive_select_videos()`
   - Lines 284-292: Use `normalize_categories()` and list operations in `remove_from_category()`
   - Changed all `.get('categories', {})` to `.get('categories', [])`

2. **Added**: `dataset_generator_v2/test_category_list_format.py`
   - Comprehensive test suite for list-format category handling
   - Tests all three fixed methods
   - Tests both list and dict formats for backward compatibility

## Before and After

### Before Fix
```
Choice: 6
Optional filter: 
⚠️  Error processing menu choice '6': 'list' object has no attribute 'items'
Traceback (most recent call last):
  ...
AttributeError: 'list' object has no attribute 'items'

Continuing...
```

### After Fix
```
Choice: 6
Optional filter: 

================================================================================
INTERACTIVE VIDEO SELECTION
================================================================================
...
[ ]   0      Venom 2 - Let There Be Carnage     master, universal             
[ ]   1      Poltergeist                        master, universal             
[ ]   5      Shrek                              master, toon                  
...
```

## Conclusion

The fix successfully resolves the AttributeError by updating all category-handling code to use the list-based format. The changes are minimal, use existing utility functions, and maintain backward compatibility with any legacy dict-format data.

**Key Principle**: Categories are now simple lists of names. A video is either in a category (100%) or not (0%). No fractional weights.

This matches the documented design in `category_utils.py` and ensures consistency throughout the video manager system.
