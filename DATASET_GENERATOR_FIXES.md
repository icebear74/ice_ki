# Dataset Generator Fixes - Import Error and Category Sorting

## Issues Fixed

### 1. NameError: get_video_categories Not Defined

**Problem:**
```
2026-02-11 13:27:08,938 - ERROR - [PROCESSING] Error: House Of The Dragon - S01E04
NameError: name 'get_video_categories' is not defined
  File "make_dataset_v2_uhd.py", line 1068, in calculate_format_distribution_for_video
    video_cats = get_video_categories(video)
```

**Root Cause:**
- `get_video_categories()` function is defined in `category_utils.py`
- Used in `make_dataset_v2_uhd.py` at line 1069
- But NOT imported in `make_dataset_v2_uhd.py`

**Solution:**
Added missing import to `make_dataset_v2_uhd.py`:
```python
from category_utils import get_video_categories, normalize_categories
```

**Files Changed:**
- `dataset_generator_v2/make_dataset_v2_uhd.py`

**Result:**
✅ Dataset generator can now process videos without NameError crash  
✅ Function properly imported and available  
✅ All syntax checks pass  

---

### 2. JSON Save Order by Category

**User Request:**
"when i save the json in the manager .. save in order of categories .. so in the json the files that are in master comes first, so it generated first in the dataset generator"

**Previous Behavior:**
Videos were sorted only by name (alphabetically), regardless of category.

**New Behavior:**
Videos are sorted by:
1. **Primary**: Category name (alphabetically)
2. **Secondary**: Video name (alphabetically)

**Implementation:**
Modified `save()` method in `video_manager.py`:

```python
def sort_key(video):
    cats = get_video_categories(video)
    # Primary sort: first category alphabetically
    primary = cats[0] if cats else 'zzz_no_category'
    # Secondary sort: video name
    secondary = video.get('name', '').lower()
    return (primary, secondary)

self.videos.sort(key=sort_key)
```

**Example Output Order:**
```json
{
  "videos": [
    {"name": "Avatar", "categories": ["master"]},      # master first
    {"name": "Batman", "categories": ["master"]},      # master (sorted by name)
    {"name": "Star Wars", "categories": ["space"]},    # space next
    {"name": "Shrek", "categories": ["toon"]},         # toon next
    {"name": "Zulu", "categories": ["universal"]}      # universal last
  ]
}
```

**Files Changed:**
- `dataset_generator_v2/video_manager.py`

**Benefits:**
✅ Videos grouped by category in JSON  
✅ Master category processed first by dataset generator  
✅ Predictable, consistent processing order  
✅ Easier to review/edit JSON files  
✅ Within each category, videos alphabetically sorted  

---

## Testing

### Import Fix Testing
```bash
# Syntax verification
python3 -m py_compile make_dataset_v2_uhd.py  # ✓ Success

# Import test
python3 -c "import make_dataset_v2_uhd"  # ✓ Success
```

### Category Sorting Testing
Created comprehensive test: `test_category_sorting.py`

**Test Results:**
```
✓ Videos sorted by category first, then name
✓ Master videos appear first in JSON
✓ Categories in alphabetical order (master → space → toon → universal)
✓ Within categories, videos alphabetically sorted
```

**Existing Tests:**
```
✓ test_category_assignment_improvements.py - All pass
✓ test_video_manager_improvements.py - All pass
✓ No regressions introduced
```

---

## Impact

### Before
- ❌ Dataset generator crashed with NameError
- ❌ Videos in JSON in unpredictable order
- ❌ Processing order inconsistent

### After
- ✅ Dataset generator processes videos successfully
- ✅ Videos grouped by category in JSON
- ✅ Master category videos processed first
- ✅ Consistent, predictable processing order

---

## Usage

When using the video manager:

```bash
cd dataset_generator_v2
python3 video_manager.py

# Make changes to video categories
# Choose 's' to save

# Output message:
# ✓ Saved to config.json (videos sorted by category, then title)
```

The saved JSON will have videos ordered by category, ensuring the dataset generator processes them in the optimal order.
