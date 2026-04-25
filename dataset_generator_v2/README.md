# dataset_generator_v2

Video super-resolution training dataset generator – v2 architecture.

---

## Architecture Overview

```
dataset_generator_v2/
├── templates.json                     # Reusable format & degradation templates
├── generator_config_v2.active.json   # Active project config (videos, categories, …)
├── video_manager.py                  # Central management UI (this is your main tool)
├── make_dataset_v2_uhd.py            # Generator (Task 2 – to be rebuilt)
├── streaming_extractor.py            # Frame extractor (Task 2 – to be rebuilt)
├── utils/
│   ├── config_io.py                  # Shared IO + validation layer (NEW in Task 1)
│   ├── format_definitions.py         # FORMATS dict + get_output_dirs_for_format()
│   └── …
└── category_utils.py                 # Video-to-category helpers
```

---

## Config System

### `templates.json`

Defines **reusable** building blocks referenced from the active config.

| Section | Purpose |
|---|---|
| `format_templates` | GT resolution, scale factor, aspect ratio (e.g. `960x540_169`) |
| `degradation_templates` | Blur / compression / noise / chroma / color degradation profiles |

Templates are **never modified by the generator** – only by `video_manager.py` (option 18).

### `generator_config_v2.active.json`

The live project config. Contains:

- `root_path` – base output directory for dataset patches
- `source_dirs` – list of directories scanned for video files
- `videos` – list of known videos with their category assignments
- `categories` – per-category config (see below)
- `processing` – frame extraction parameters
- `quality` – blur/sharpness thresholds
- `workers`, `batch_size`, `random_seed`, `ffmpeg_timeout`, `ffprobe_timeout`

#### Category config structure

```json
"categories": {
  "master": {
    "target_total": 350000,
    "formats": [
      {
        "template": "960x540_169",
        "weight": 60,
        "source_mode": "resize",
        "degradation_mix": {
          "dvd_film_balanced": 40,
          "classic_sitcom_sd": 20
        }
      }
    ]
  }
}
```

- `target_total` – total number of GT patches to generate for this category
- `formats` – list of format entries; the generator picks one per patch based on `weight`
- `template` – name of a key in `format_templates`
- `source_mode` – `"resize"` (downscale full frame) or `"crop"` (crop a region)
- `degradation_mix` – weighted map of degradation template names to apply

---

## How the Config System Works

1. **Format selection**: per patch, a format entry is selected randomly based on `weight` values (relative, not percentages).
2. **Degradation selection**: within a format entry, a degradation template is selected based on `degradation_mix` weights.
3. **source_mode**:
   - `resize` – the source frame is scaled down to LR size, then used as-is for GT
   - `crop` – a region matching `gt_size` is cropped from the source frame

---

## `video_manager.py` – Central Management UI

Run with:
```bash
cd dataset_generator_v2
python video_manager.py
```

### Menu Reference

| Option | Function |
|---|---|
| 1–4 | List / search videos |
| 5–7 | Assign videos to categories |
| 8–9 | Remove assignments / reset |
| 10 | Statistics (targets, format mix) |
| 11 | Manage categories (add/remove/edit target_total) |
| 12 | Manage category formats & degradation mix |
| 13–17 | Source directory management + rescan |
| 18 | Manage templates (format & degradation) |
| 19 | Config validation report |
| 20 | Create new active config file |
| s | Save config changes |
| t | Save template changes |
| q | Quit |

---

## First Start Behavior

If `generator_config_v2.active.json` is missing, `video_manager.py` creates a minimal
default. If `templates.json` is missing it is also created with default templates.

After first start:
1. Edit `root_path` in `generator_config_v2.active.json`
2. Add source directories (option 14)
3. Run a rescan (option 17) to populate the video list
4. Assign videos to categories (options 5/6/7)
5. Save (option `s`)

---

## What Task 2 Will Bring

Task 2 rebuilds the generator layer:

- `make_dataset_v2_uhd.py` – rewritten to load `templates.json` + active config via `config_io`
- `streaming_extractor.py` – updated to use new format/degradation pipeline
- The generator will read `categories[name].formats[*]` and apply the weighted selection logic
- Degradation templates drive actual image processing (blur, JPEG compression, noise, chroma, color)
