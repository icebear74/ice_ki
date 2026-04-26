# Task 2 Handover Notes

> **Status: COMPLETED** – Task 2 has been implemented.
> This document is retained as reference for the design decisions made.

---

## Which Config Files Task 2 Must Load

Load **both** files via `utils/config_io.py`:

```python
from utils.config_io import (
    load_active_config,
    ensure_templates_file,
    validate_active_config,
    validate_templates,
)

templates = ensure_templates_file("templates.json")
config    = load_active_config("generator_config.json")
```

Never hardcode paths – derive them relative to `__file__`.

---

## New Config Structure Task 2 Must Understand

### Active config – categories

```json
"categories": {
  "<cat_name>": {
    "target_total": <int>,
    "formats": [
      {
        "template": "<format_template_name>",
        "weight": <int>,
        "source_mode": "resize" | "crop",
        "degradation_mix": {
          "<degradation_template_name>": <int_weight>,
          ...
        }
      },
      ...
    ]
  }
}
```

### Format selection algorithm (per patch)

```python
import random

def pick_format(formats: list) -> dict:
    weights = [f["weight"] for f in formats]
    return random.choices(formats, weights=weights, k=1)[0]
```

### Degradation selection algorithm (per patch)

```python
def pick_degradation(degradation_mix: dict) -> str:
    names   = list(degradation_mix.keys())
    weights = list(degradation_mix.values())
    return random.choices(names, weights=weights, k=1)[0]
```

### Format template → actual sizes

```python
fmt_spec = templates["format_templates"][format_entry["template"]]
gt_w, gt_h = fmt_spec["gt_size"]      # GT patch resolution
scale       = fmt_spec["scale"]        # LR = GT / scale
lr_w, lr_h  = gt_w // scale, gt_h // scale
```

### Degradation template → processing parameters

```python
deg_spec = templates["degradation_templates"][chosen_degradation_name]
# Keys present: blur, compression, noise, chroma, color
# Each sub-dict has numeric params (ranges, probs, strengths)
```

---

## Old Generator Assumptions That Are Now Wrong

| Old assumption | New reality |
|---|---|
| Hardcoded format names `small_540`, `medium_169`, `large_720` | Formats come from `templates["format_templates"]` keyed by arbitrary names |
| `CATEGORY_FORMAT_DISTRIBUTION` in `format_definitions.py` | Removed. Distribution is in `categories[name].formats[*].weight` |
| `output_patches` dict in config | Removed. Replaced by per-category `formats` list |
| `category_patches` dict (patch count per category) | Replaced by `categories[name].target_total` |
| Single global degradation pipeline | Per-format `degradation_mix` → weighted selection from `degradation_templates` |
| `source_mode` not in config (always resize) | Now explicit per format entry: `"resize"` or `"crop"` |

---

## Files to Rebuild in Task 2

| File | Action |
|---|---|
| `make_dataset_v2_uhd.py` | **Full rebuild** – load new config, weighted format+degradation selection |
| `streaming_extractor.py` | **Rebuild** – accept `source_mode`, apply degradation from templates |

### Files to fix (not full rebuild)

`utils/format_definitions.py` is already updated in Task 1:
- Legacy aliases `small_540`, `medium_169`, `large_720` removed
- `CATEGORY_FORMAT_DISTRIBUTION` removed
- `get_output_dirs_for_format()` uses `.get()` with safe fallback

---

## Old Paths / Logic No Longer Valid

- `generator_config_v2.json` – **deleted**. Use `generator_config.json`
- `create_default_config.py` – **deleted**. Use `config_io.create_default_active_config()`
- `create_full_config.py` – **deleted**
- `utils/ui_display.py`, `utils/ui_terminal.py` – **deleted**
- `config["output_patches"]` – **gone** (replaced by per-category formats)
- `config["category_patches"]` – **gone** (replaced by `target_total`)
- `config["dataset_name"]` – **gone** (dataset is now multi-category by default)

---

## Recommended Rebuild Order for Task 2

1. Write a `WeightedFormatPicker` / `WeightedDegradationPicker` helper (or inline the logic)
2. Implement `apply_degradation(frame, deg_spec)` that interprets a degradation template dict
3. Update `streaming_extractor.py`:
   - Accept `source_mode` parameter (`resize` or `crop`)
   - Accept `degradation_spec` dict and apply it
4. Rewrite `make_dataset_v2_uhd.py`:
   - Load config + templates via `config_io`
   - Iterate over `categories`, pick formats + degradations with weighted random
   - Call extractor with correct parameters
5. Validate with `config_io.validate_active_config()` at startup

---

## Key `config_io.py` Functions Task 2 Should Use

```python
from utils.config_io import (
    load_active_config,        # load generator_config.json
    load_templates,            # load templates.json
    ensure_templates_file,     # create default if missing, then load
    validate_templates,        # returns list[str] of errors
    validate_active_config,    # cross-validates config vs templates
    VALID_SOURCE_MODES,        # {"resize", "crop"}
)
```
