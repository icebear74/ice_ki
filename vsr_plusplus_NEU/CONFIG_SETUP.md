# Configuration Setup

## Quick Start

**Before running training, you MUST create `config.py`:**

```bash
cd vsr_plusplus_NEU
cp config.py.example config.py
```

That's it! The example config is already set up for the new dataset structure.

## Configuration File

- **config.py.example** - Template configuration (committed to git)
- **config.py** - Your local configuration (NOT in git, you must create it)

## Default Settings in config.py.example

The example config is already configured for:
- Dataset root: `/mnt/data/training/datasetNeu`
- Category: `master` (lowercase)
- New structure: `patches/{size_key}/` and `val/{size_key}/`
- 7-frame model optimized for Tesla P4

## If You Need Different Paths

After copying `config.py.example` to `config.py`, edit these lines:

```python
# Dataset root directory - base directory for all datasets
DATASET_ROOT = "/mnt/data/training/datasetNeu"  # ← Change if needed

# Default dataset name (category) - lowercase
DEFAULT_DATASET_NAME = "master"  # ← Change to 'universal', 'space', or 'toon' if needed
```

## Runtime Configuration

The `runtime_config.json` file controls multi-size training:
- Which size variants to use (540, 720, 720_169)
- Size distribution (e.g., 65% 540, 35% 720_169)
- Batch sizes per variant

This file is already set up and can be edited if needed.

## Verification

After creating `config.py`, run this to verify:

```bash
python3 config.py
```

You should see the configuration displayed with correct paths.
