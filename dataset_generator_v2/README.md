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

## Running the Generator

After configuring via `video_manager.py`:

```bash
cd dataset_generator_v2
python make_dataset_v2_uhd.py
```

Optional: pass a custom config directory as the first argument:

```bash
python make_dataset_v2_uhd.py /path/to/config_dir
```

The generator loads `templates.json` and `generator_config_v2.active.json` from
the config directory, validates both at startup, then processes all videos in the
configured categories.

### Runtime Behaviour

1. **Phase 1** – scans all video durations using ffprobe (with caching).
2. **Phase 2** – distributes patch targets proportionally across videos based on duration.
3. **Phase 3** – processes videos sequentially in a single FFmpeg streaming pass per video:
   - for each frame window, picks a category and format entry by weight,
   - picks a degradation template from the format's `degradation_mix` by weight,
   - creates a (GT, LR) patch pair using the resolved template parameters,
   - saves GT and LR to `{root_path}/{category}/{template}/{n_frames}frames/gt|lr/`.
4. **Resume** – progress is persisted to `generation_status.json` and `extraction_plan.json`; interrupted runs continue from where they stopped.

### Source mode

| `source_mode` | GT | LR |
|---|---|---|
| `resize` | Full-frame Lanczos4 downscale to `gt_size` | Full-frame INTER\_AREA downscale to `lr_size` |
| `crop` | Centre-frame crop of `gt_size` (with 2× oversample when source allows) | Same crop region at `lr_size` via INTER\_AREA |

4:3 formats use `crop` mode by default and are treated identically to 16:9 crops –
no aspect-ratio name check is performed.

### Degradation templates

Each LR patch is degraded using the parameters sampled from the chosen template:

| Key | Effect |
|---|---|
| `blur` | Gaussian blur with `sigma_range` |
| `compression` | JPEG round-trip at `jpeg_quality_range` |
| `noise` | Gaussian noise at `sigma_range` (luma or colour) |
| `chroma` | HSV saturation scaling; optional chroma bleed |
| `color` | Contrast / brightness / gamma / black-lift |

Parameters are sampled **once per scene window** so all LR frames share consistent settings.

---

## Source of Truth

| Question | Answer |
|---|---|
| What formats exist? | `templates.json → format_templates` |
| What degradation profiles exist? | `templates.json → degradation_templates` |
| Which categories exist? | `generator_config_v2.active.json → categories` |
| How many patches per category? | `categories[name].target_total` |
| Which videos are assigned? | `generator_config_v2.active.json → videos` |
| Where is the output? | `generator_config_v2.active.json → root_path` |

Modify both files exclusively via `video_manager.py`.  Do **not** hand-edit the JSON
while the generator is running.

