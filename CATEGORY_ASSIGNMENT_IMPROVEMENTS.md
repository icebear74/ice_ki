# Category Assignment Improvements & Video Sorting

## New Features Implemented

### 1. Smart Category Assignment: Add vs Replace

**Problem:** Previously, when assigning categories to videos, ALL old categories were deleted and replaced with new ones. This was not ideal when you wanted to add a category to videos that already had categories assigned.

**Solution:** The system now asks you what to do when assigning categories to videos that already have categories:

```
⚠️  Some videos already have categories assigned.
Options:
  1. ADD to existing categories (keep old + add new)
  2. REPLACE all categories (remove old, set new)
Choose (1/2) [default: 1]:
```

#### Add Mode (Option 1)
- **Keeps** all existing categories
- **Adds** new categories to the list
- **Prevents** duplicates (if category already exists, it won't be added twice)

**Example:**
```
Before: Video has [master, space]
Assign: [toon]
Result: [master, space, toon]  ✓ Old categories preserved
```

#### Replace Mode (Option 2)
- **Removes** all existing categories
- **Sets** new categories as the only ones

**Example:**
```
Before: Video has [master, space]
Assign: [toon]
Result: [toon]  ✓ Old categories removed
```

### 2. Automatic Video Sorting by Title

**Problem:** Videos were displayed in random/original order, making it hard to find specific videos.

**Solution:** Videos are now automatically sorted alphabetically by title (case-insensitive):

- **On Load:** Videos are sorted when the configuration is loaded
- **On Save:** Videos are saved in sorted order to the JSON file
- **Display:** All video lists show videos in alphabetical order

**Example:**
```
Before (unsorted):
  - Zombieland
  - Avatar
  - Shrek
  - Batman

After (sorted):
  - Avatar
  - Batman
  - Shrek
  - Zombieland
```

The console will confirm: `✓ Loaded 466 videos (sorted by title)`

### 3. Categories Display in Brackets

**Problem:** Categories were shown but not clearly distinguished from the video name.

**Solution:** Categories are now shown in brackets `[...]` for better visibility:

```
ID     Name                     Categories              
-----------------------------------------------------------
0      Avatar                   [master, space, toon]
42     Batman Begins            [master, universal]
99     Shrek                    [master, toon]
150    Star Wars                [master, space]
200    Untitled                 [no categories]
```

Benefits:
- **Clear visual separation** between video name and categories
- **Easy to scan** and find videos by category
- **Consistent format** throughout the application

## Usage Examples

### Scenario 1: Add Categories to Existing Videos

You have videos with `[master]` category and want to add `[space]` to science fiction movies:

1. Choose menu option 5 or 6 to assign videos
2. Select the sci-fi videos
3. Select the `space` category
4. When prompted, choose **1** (ADD)
5. Result: Videos now have `[master, space]`

### Scenario 2: Replace All Categories

You mistakenly assigned wrong categories and want to fix them:

1. Choose menu option 5 or 6 to assign videos
2. Select the videos to fix
3. Select the correct categories
4. When prompted, choose **2** (REPLACE)
5. Result: Videos now have only the new categories

### Scenario 3: Finding Videos by Title

With automatic sorting, you can:
- Scroll through videos alphabetically
- Find videos quickly (all "Star Wars" movies are together)
- See the entire collection in a logical order

## Technical Details

### assign_videos() Method Signature

```python
def assign_videos(self, video_indices: List[int], 
                 categories: List[str],
                 mode: str = 'ask'):
    """
    Assign categories to videos.
    
    Args:
        video_indices: List of video indices to assign
        categories: List of category names to assign
        mode: 'ask' (prompt user), 'add' (append), 'replace' (replace all)
    """
```

### Modes Explained

- **'ask' mode (default):** 
  - If videos have existing categories → prompt user to choose
  - If no existing categories → directly assign (equivalent to replace)

- **'add' mode:**
  - Always append to existing categories
  - Prevents duplicates

- **'replace' mode:**
  - Always replace all categories
  - Old categories are removed

### Sorting Implementation

Videos are sorted using Python's built-in sort with a case-insensitive key:

```python
self.videos.sort(key=lambda v: v.get('name', '').lower())
```

This ensures:
- "Avatar" comes before "batman" (case-insensitive)
- Numbers and special characters are handled correctly
- Empty names are handled gracefully

## Benefits Summary

✅ **More Control:** Choose whether to add or replace categories  
✅ **No Accidents:** Won't lose categories by mistake  
✅ **Better Organization:** Videos sorted alphabetically  
✅ **Easier Navigation:** Find videos faster in sorted lists  
✅ **Clear Display:** Categories in brackets `[...]` stand out  
✅ **No Duplicates:** System prevents duplicate categories  
✅ **Persistent Order:** Sorted order saved to JSON file  

## Testing

Comprehensive test suite created: `test_category_assignment_improvements.py`

Tests verify:
- ✓ Videos sorted correctly by title
- ✓ Videos saved in sorted order
- ✓ Add mode preserves existing categories
- ✓ Replace mode removes old categories
- ✓ No duplicates when adding existing categories
- ✓ Categories displayed in brackets

Run tests:
```bash
cd dataset_generator_v2
python3 test_category_assignment_improvements.py
```

## Backward Compatibility

✅ **Fully compatible** with existing configurations  
✅ **No breaking changes** to JSON format  
✅ **Existing workflows** continue to work  
✅ **Progressive enhancement** - new features are optional  

Old code that calls `assign_videos(indices, categories)` will still work - it will prompt the user when appropriate.
