# ⚠️ LOADER UPDATE REQUIRED – Bucket-Subdir Migration

> **This file exists so the next AI task can find it immediately.**
> It documents exactly what needs to change in the training loader after the
> dataset generator was updated to write patches into bucket subdirectories.

---

## Why this is needed

The generator (`dataset_generator_v2`) previously wrote all patches into a
single flat directory:

```
master/patches/720/GT/          ← all 300 000+ PNG files here  ← OLD (flat)
master/patches/720/LR_7frames/
```

It now writes into 4-digit zero-padded bucket subdirectories
(`BUCKET_SIZE = 10 000` files per bucket):

```
master/patches/720/GT/0000/     ← first 10 000 patches
master/patches/720/GT/0001/     ← next 10 000 patches
master/patches/720/GT/0002/     ← …
master/patches/720/LR_7frames/0000/
master/patches/720/LR_7frames/0001/
…
```

GT and LR **always share the same bucket name**, so `GT/0001/foo.png` is
paired with `LR_7frames/0001/foo.png`.

**The loader (`core/dataset.py`) still uses `os.listdir(gt_dir)` on the base
directory and therefore sees only the bucket subdirectories, not the PNG
files.  It must be updated before training can run.**

---

## Files to change

### `core/dataset.py`   ← primary change

Every place that does `os.listdir(self.gt_dir)` and filters for `.png` must be
replaced with an `os.walk`-based scan that descends into bucket subdirs.

There are **5 affected spots** (search for `os.listdir(self.gt_dir)`):

| Location | Line (approx.) | What to change |
|---|---|---|
| Initial scan (slow path) | ~97 | `os.listdir` → walk over bucket subdirs |
| `reload_files()` | ~577 | same |
| `check_for_new_files()` | ~521 | same |
| Cache invalidation `_load_index` | ~418 | `getmtime(gt_dir)` alone is no longer reliable — also compare total PNG count |
| `_save_index` metadata | ~451 | store PNG count alongside mtime |

#### Helper to add

Add this private helper to `VSRDataset` (or as a module-level function):

```python
def _collect_png_files(base_dir: str) -> list[str]:
    """
    Return a sorted list of PNG basenames found under *base_dir*.

    Supports both the legacy flat layout (PNGs directly in base_dir) and the
    new bucket layout (PNGs in 4-digit subdirs 0000/, 0001/, …).

    GT and LR directories always use the same bucket names, so the returned
    list can be used for both sides — the LR path is reconstructed from the
    same filename inside the matching bucket subdir stored in lr_paths.
    """
    files = []
    if not os.path.isdir(base_dir):
        return files
    # Detect layout: bucket subdirs present?
    entries = os.listdir(base_dir)
    bucket_dirs = sorted(
        e for e in entries
        if len(e) == 4 and e.isdigit() and os.path.isdir(os.path.join(base_dir, e))
    )
    if bucket_dirs:
        # New bucket layout
        for bucket in bucket_dirs:
            bucket_path = os.path.join(base_dir, bucket)
            for f in sorted(os.listdir(bucket_path)):
                if f.lower().endswith('.png'):
                    files.append(os.path.join(bucket, f))  # relative: "0000/foo.png"
    else:
        # Legacy flat layout (backward-compatible)
        for f in sorted(entries):
            if f.lower().endswith('.png'):
                files.append(f)
    return files
```

#### How `lr_paths` dict changes

Currently `lr_paths` maps `filename → lr_directory`.

With buckets the key becomes the **relative path** (`"0000/foo.png"`), and the
value is still the LR *bucket* directory (`".../LR_7frames/0000"`):

```python
# Before (flat):
self.gt_files  = ["foo.png", "bar.png", …]
self.lr_paths  = {"foo.png": "/…/LR_7frames", …}

# After (bucket-aware):
self.gt_files  = ["0000/foo.png", "0001/bar.png", …]
self.lr_paths  = {"0000/foo.png": "/…/LR_7frames/0000", …}
```

#### `__getitem__` patch

The `__getitem__` method builds `gt_path` and `lr_path` from these values.
With the new relative key the paths assemble naturally:

```python
# current (line ~342):
gt_path = os.path.join(self.gt_dir, gt_file)   # gt_file = "foo.png"
lr_dir  = self.lr_paths[gt_file]
lr_path = os.path.join(lr_dir, gt_file)

# updated:
gt_path = os.path.join(self.gt_dir, gt_file)    # gt_file = "0000/foo.png"
lr_dir  = self.lr_paths[gt_file]                # = ".../LR_7frames/0000"
lr_path = os.path.join(lr_dir, os.path.basename(gt_file))
```

#### Cache invalidation (`_load_index` / `_save_index`)

`getmtime` on the base GT directory does **not** change when files are added
to a bucket subdir.  Add a `gt_file_count` field to the cache JSON and
invalidate when the count differs:

```python
# in _save_index – add to data dict:
'gt_file_count': len(gt_files),

# in _load_index – add after mtime checks:
if data.get('gt_file_count') != len(data.get('gt_files', [])):
    return None   # shouldn't happen, but guard against corruption
```

A more robust approach: also check the mtime of the most-recently-modified
bucket subdir, or simply always re-scan when a new bucket dir appears.

---

## Backward compatibility

`_collect_png_files` detects the layout automatically.  Existing flat datasets
(if any) continue to work without any migration.

---

## Related source files

| File | Role |
|---|---|
| `dataset_generator_v2/utils/format_definitions.py` | `BUCKET_SIZE`, `get_synced_bucket_dirs()` |
| `dataset_generator_v2/streaming_extractor.py` | `_output_dirs_cache` block, `save_patch_pair()` |
| `vsr_plusplus_NEU/core/dataset.py` | **Loader – needs update** |
| `vsr_plusplus_NEU/core/dataloader.py` | No change needed |
| `vsr_plusplus_NEU/dataset_strucure.txt` | Update path examples once loader is fixed |
