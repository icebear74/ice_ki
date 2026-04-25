#!/usr/bin/env python3
"""
VIDEO CATEGORY MANAGER  (v2 – new config model)
Central management UI for dataset_generator_v2.
"""

import datetime
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from category_utils import (
    normalize_categories,
    get_video_categories,
    is_video_in_category,
    format_categories_display,
    convert_config_to_list_format,
)
from interactive_selector import select_items, select_categories
from utils.config_io import (
    load_templates,
    save_templates as _save_templates_io,
    load_active_config,
    save_active_config,
    create_default_active_config,
    ensure_templates_file,
    validate_templates,
    validate_active_config,
    VALID_SOURCE_MODES,
    ASPECT_RATIOS,
    BASE_X_PRESETS,
    build_format_template,
    compute_format_sizes,
)


def _short_path(full_path: str, depth: int = 2) -> str:
    parts = Path(full_path).parts
    return str(Path(*parts[-depth:])) if len(parts) >= depth else full_path


def _sorted_videos(videos: List[tuple]) -> List[tuple]:
    return sorted(
        videos,
        key=lambda iv: (
            _short_path(iv[1].get("path", ""), depth=3).lower(),
            iv[1].get("name", "").lower(),
        ),
    )


class VideoManager:
    """Central manager for videos, categories, formats and templates."""

    def __init__(self, config_path: str, templates_path: str = None):
        self.config_path = config_path
        script_dir = Path(config_path).parent
        self.templates_path = templates_path or str(script_dir / "templates.json")
        self.config: dict = {}
        self.templates: dict = {}
        self.videos: List[dict] = []
        self.categories: dict = {}
        self.modified = False
        self.templates_modified = False

    def load(self):
        self.templates = ensure_templates_file(self.templates_path)
        self.config = load_active_config(self.config_path)
        self.videos = self.config.get("videos", [])
        self.videos.sort(key=lambda v: v.get("name", "").lower())
        self.categories = self.config.get("categories", {})
        cat_names = ", ".join(sorted(self.categories.keys())) or "(none)"
        print(f"✓ Loaded {len(self.videos)} videos")
        print(f"✓ Categories: {cat_names}")

    def save(self, backup: bool = True):
        def _sort_key(video):
            cats = get_video_categories(video)
            return (-len(cats) if cats else 999, cats[0] if cats else "zzz", video.get("name", "").lower())
        self.videos.sort(key=_sort_key)
        self.config["videos"] = self.videos
        self.config["categories"] = self.categories
        save_active_config(self.config, self.config_path)
        print(f"✓ Saved to {Path(self.config_path).name}")
        self.modified = False

    def save_templates(self, backup: bool = True):
        if backup and os.path.exists(self.templates_path):
            with open(self.templates_path, "r", encoding="utf-8") as f:
                old = f.read()
            with open(self.templates_path + ".backup", "w", encoding="utf-8") as f:
                f.write(old)
        _save_templates_io(self.templates, self.templates_path)
        print(f"✓ Templates saved to {Path(self.templates_path).name}")
        self.templates_modified = False

    def list_videos(self, filter_pattern=None, category=None, show_unassigned=False, use_simple_search=False):
        import re
        filtered = []
        for i, video in enumerate(self.videos):
            if filter_pattern:
                if use_simple_search:
                    if filter_pattern.lower() not in video["name"].lower():
                        continue
                else:
                    try:
                        if not re.search(filter_pattern, video["name"], re.IGNORECASE):
                            continue
                    except re.error:
                        if filter_pattern.lower() not in video["name"].lower():
                            continue
            if category and category not in video.get("categories", {}):
                continue
            if show_unassigned and video.get("categories", {}):
                continue
            filtered.append((i, video))
        return filtered

    def print_video_list(self, videos, max_display=20):
        if not videos:
            print("No videos found.")
            return
        sorted_vids = _sorted_videos(videos)
        print(f"\n{'ID':<6} {'Path (depth-3)':<36} {'Name':<42} {'Categories'}")
        print("-" * 120)
        for idx, (i, video) in enumerate(sorted_vids):
            if max_display and idx >= max_display:
                print(f"... and {len(sorted_vids) - max_display} more (use -a to show all)")
                break
            path_short = _short_path(video.get("path", ""), depth=3)[:34]
            name = video["name"][:40]
            cats = video.get("categories", [])
            cat_str = "[" + ", ".join(cats) + "]" if cats else "[no categories]"
            forced = video.get("forced_frames", {})
            if forced:
                parts = [f"{cat}:{n:,}" for cat, n in sorted(forced.items()) if n > 0]
                if parts:
                    cat_str += "  ⚡ " + "  ".join(parts)
            print(f"{i:<6} {path_short:<36} {name:<42} {cat_str}")

    def assign_videos(self, video_indices, categories, mode="ask"):
        if not video_indices:
            return
        has_existing = any(
            bool(get_video_categories(self.videos[i]))
            for i in video_indices if 0 <= i < len(self.videos)
        )
        actual_mode = mode
        if mode == "ask" and has_existing:
            print("\n⚠️  Some videos already have categories assigned.")
            print("  1. ADD to existing  2. REPLACE all")
            choice = input("Choose (1/2) [default: 1]: ").strip()
            actual_mode = "replace" if choice == "2" else "add"
        elif mode == "ask":
            actual_mode = "replace"
        count = 0
        for idx in video_indices:
            if 0 <= idx < len(self.videos):
                if actual_mode == "add":
                    existing = get_video_categories(self.videos[idx])
                    combined = existing.copy()
                    for cat in categories:
                        if cat not in combined:
                            combined.append(cat)
                    self.videos[idx]["categories"] = combined
                else:
                    self.videos[idx]["categories"] = categories
                count += 1
        self.modified = True
        mode_text = "Added to" if actual_mode == "add" else "Replaced with"
        print(f"✓ {mode_text} {count} videos: {categories}")

    def remove_from_category(self, video_indices, category):
        count = 0
        for idx in video_indices:
            if 0 <= idx < len(self.videos):
                cats = normalize_categories(self.videos[idx].get("categories", []))
                if category in cats:
                    cats.remove(category)
                    self.videos[idx]["categories"] = cats
                    count += 1
        self.modified = True
        print(f"✓ Removed {count} videos from category '{category}'")

    def reset_all(self):
        confirm = input("⚠️  Reset ALL video assignments? This cannot be undone! (yes/no): ")
        if confirm.lower() != "yes":
            print("Cancelled.")
            return
        for video in self.videos:
            video["categories"] = []
        self.modified = True
        print(f"✓ Reset {len(self.videos)} videos")

    def interactive_select_videos(self, initial_filter=None):
        videos = (self.list_videos(filter_pattern=initial_filter, use_simple_search=True)
                  if initial_filter else self.list_videos())
        if not videos:
            print("No videos found.")
            return None
        try:
            sorted_vids = _sorted_videos(videos)
            selected = select_items(
                items=[v for _, v in sorted_vids],
                title=f"Select Videos – {len(sorted_vids)} available (Space toggle, Enter done, Esc cancel)",
                get_label=lambda v: _short_path(v.get("path", ""), depth=3),
                get_details=lambda v: format_categories_display(v.get("categories", [])),
            )
            if selected is None:
                return None
            return [sorted_vids[i][0] for i in selected]
        except Exception as e:
            print(f"⚠️  Curses UI failed: {e}")
            return None

    def set_forced_frames(self, video_indices):
        if isinstance(video_indices, int):
            video_indices = [video_indices]
        valid = [i for i in video_indices if 0 <= i < len(self.videos)]
        if not valid:
            print("❌ No valid video indices.")
            return
        all_cats: List[str] = []
        for idx in valid:
            for cat in get_video_categories(self.videos[idx]):
                if cat not in all_cats:
                    all_cats.append(cat)
        if not all_cats:
            print("❌ Selected videos have no categories assigned.")
            return
        names = [self.videos[i].get("name", "?") for i in valid]
        if len(names) == 1:
            print(f"\n📹 Forced frame overrides for: {names[0]}")
        else:
            print(f"\n📹 Forced frame overrides for {len(names)} videos:")
            for n in names:
                print(f"   • {n}")
        print(f"   Categories: {', '.join(all_cats)}")
        print("   blank = keep, 0 = auto, N = exact\n")
        new_values: Dict[str, Optional[int]] = {}
        for cat in all_cats:
            cur_values = [self.videos[i].get("forced_frames", {}).get(cat, 0) for i in valid]
            current_str = f"{cur_values[0]:,}" if len(set(cur_values)) == 1 else "mixed"
            raw = input(f"  {cat} (current: {current_str}  |  0 = auto): ").strip()
            if not raw:
                new_values[cat] = None
                continue
            try:
                value = int(raw)
            except ValueError:
                print(f"  ⚠️  Invalid number, keeping current")
                new_values[cat] = None
                continue
            new_values[cat] = None if value < 0 else value
        for idx in valid:
            video = self.videos[idx]
            forced = dict(video.get("forced_frames", {}))
            for cat, value in new_values.items():
                if value is None:
                    continue
                if value == 0:
                    forced.pop(cat, None)
                elif cat in get_video_categories(video):
                    forced[cat] = value
            if forced:
                video["forced_frames"] = forced
            else:
                video.pop("forced_frames", None)
        self.modified = True
        set_cats = {cat: v for cat, v in new_values.items() if v is not None}
        if set_cats:
            parts = "  ".join(f"{c}: {v:,}" if v else f"{c}: auto" for c, v in sorted(set_cats.items()))
            print(f"\n✓ Applied to {len(valid)} video(s): {parts}")
        else:
            print("\n✓ No changes made.")

    def show_statistics(self):
        print("\n" + "=" * 70)
        print("STATISTICS")
        print("=" * 70)
        cat_counts: Dict[str, int] = {cat: 0 for cat in self.categories}
        forced_by_cat: Dict[str, list] = {cat: [] for cat in self.categories}
        unassigned = 0
        for video in self.videos:
            cat_list = get_video_categories(video)
            if not cat_list:
                unassigned += 1
            else:
                for cat in cat_list:
                    if cat in cat_counts:
                        cat_counts[cat] += 1
            forced = video.get("forced_frames", {})
            for cat, n in forced.items():
                if cat in forced_by_cat and n > 0:
                    forced_by_cat[cat].append((video.get("name", "?"), _short_path(video.get("path", ""), depth=3), n))
        print(f"\nTotal videos : {len(self.videos)}")
        print(f"Unassigned   : {unassigned}\n")
        for cat in sorted(self.categories.keys()):
            cfg = self.categories[cat]
            target = cfg.get("target_total", "?")
            target_str = f"{target:,}" if isinstance(target, int) else str(target)
            print(f"  {cat:<18} {cat_counts.get(cat, 0):>4} videos   target: {target_str}")
            formats = cfg.get("formats", [])
            total_weight = sum(f.get("weight", 0) for f in formats) or 1
            for fi, fmt in enumerate(formats):
                tmpl = fmt.get("template", "?")
                w = fmt.get("weight", 0)
                mode = fmt.get("source_mode", "?")
                share = round(w / total_weight * 100, 1)
                deg_mix = fmt.get("degradation_mix", {})
                deg_str = ", ".join(f"{k}:{v}" for k, v in deg_mix.items())
                print(f"    [{fi}] {tmpl:<20} {mode:<8} {share:5.1f}%  mix: {deg_str}")
            if forced_by_cat.get(cat):
                forced_list = sorted(forced_by_cat[cat], key=lambda x: x[0].lower())
                forced_total = sum(n for _, _, n in forced_list)
                if isinstance(target, int):
                    remaining = max(0, target - forced_total)
                    print(f"    Scenes: total {target:>10,}  | forced {forced_total:>10,}  | remaining {remaining:>10,}")
                for name, sp, n in forced_list:
                    print(f"    ⚡ {name:<40} {sp:<36}  forced: {n:>8,}")
            print()

    def manage_categories(self):
        while True:
            cat_names = ", ".join(sorted(self.categories.keys())) or "(none)"
            print(f"\n── Category Management ───────────────────────────────────")
            print(f"  Categories: {cat_names}")
            print("  a) List categories  b) Add  c) Remove  d) Edit target  x) Back")
            sub = input("Choice: ").strip().lower()
            if sub == "x":
                break
            elif sub == "a":
                self._list_categories()
            elif sub == "b":
                self._add_category()
            elif sub == "c":
                self._remove_category()
            elif sub == "d":
                self._edit_category_target()
            else:
                print("Invalid choice")

    def _list_categories(self):
        if not self.categories:
            print("No categories configured.")
            return
        print(f"\n{'Category':<20} {'Target total':<14} {'Formats'}")
        print("-" * 50)
        for cat in sorted(self.categories.keys()):
            cfg = self.categories[cat]
            target = cfg.get("target_total", "?")
            target_str = f"{target:,}" if isinstance(target, int) else str(target)
            n_formats = len(cfg.get("formats", []))
            print(f"  {cat:<18} {target_str:<14} {n_formats} format(s)")

    def _add_category(self):
        name = input("New category name: ").strip().lower()
        if not name:
            print("❌ Name cannot be empty")
            return
        if name in self.categories:
            print(f"❌ Category '{name}' already exists")
            return
        target_str = input("target_total (default: 50000): ").strip()
        try:
            target = int(target_str) if target_str else 50000
        except ValueError:
            target = 50000
        self.categories[name] = {"target_total": target, "formats": []}
        self.config["categories"] = self.categories
        self.modified = True
        print(f"✓ Added category '{name}' with target {target:,}")

    def _remove_category(self):
        if not self.categories:
            print("No categories configured.")
            return
        self._list_categories()
        name = input("Category to remove: ").strip()
        if name not in self.categories:
            print(f"❌ Category '{name}' not found")
            return
        affected = sum(1 for v in self.videos if name in get_video_categories(v))
        suffix = f" and unassign {affected} video(s)" if affected else ""
        if input(f"⚠️  Remove '{name}'{suffix}? (yes/no): ").strip().lower() != "yes":
            print("Cancelled.")
            return
        for video in self.videos:
            cats = get_video_categories(video)
            if name in cats:
                cats.remove(name)
                video["categories"] = cats
        del self.categories[name]
        self.config["categories"] = self.categories
        self.modified = True
        print(f"✓ Removed category '{name}'" + (f", unassigned {affected} video(s)" if affected else ""))

    def _edit_category_target(self):
        if not self.categories:
            print("No categories configured.")
            return
        self._list_categories()
        name = input("Category to edit: ").strip()
        if name not in self.categories:
            print(f"❌ Category '{name}' not found")
            return
        current = self.categories[name].get("target_total", 0)
        val = input(f"New target_total (current: {current:,}): ").strip()
        if not val:
            print("No change.")
            return
        try:
            new_target = int(val)
        except ValueError:
            print("❌ Invalid number")
            return
        if new_target <= 0:
            print("❌ Must be positive")
            return
        self.categories[name]["target_total"] = new_target
        self.modified = True
        print(f"✓ '{name}' target: {current:,} → {new_target:,}")

    def manage_category_formats(self, category_name=None):
        if category_name is None:
            if not self.categories:
                print("No categories configured.")
                return
            self._list_categories()
            category_name = input("Category to manage formats for: ").strip()
        if category_name not in self.categories:
            print(f"❌ Category '{category_name}' not found")
            return
        while True:
            cat_cfg = self.categories[category_name]
            formats = cat_cfg.get("formats", [])
            print(f"\n── Formats for '{category_name}' ─────────────────────────────────")
            if not formats:
                print("  (no formats configured)")
            else:
                total_w = sum(f.get("weight", 0) for f in formats) or 1
                for i, fmt in enumerate(formats):
                    tmpl = fmt.get("template", "?")
                    w = fmt.get("weight", 0)
                    mode = fmt.get("source_mode", "?")
                    share = round(w / total_w * 100, 1)
                    deg_mix = fmt.get("degradation_mix", {})
                    print(f"  [{i}] {tmpl:<22} {mode:<8} weight={w} ({share:.1f}%)")
                    for dname, dw in deg_mix.items():
                        print(f"       degradation: {dname} = {dw}")
            print()
            print("  a) Add  r) Remove  e) Edit (weight/source_mode)  d) Degradation mix  x) Back")
            sub = input("Choice: ").strip().lower()
            if sub == "x":
                break
            elif sub == "a":
                self._add_format_entry(category_name)
            elif sub == "r":
                self._remove_format_entry(category_name)
            elif sub == "e":
                self._edit_format_entry(category_name)
            elif sub == "d":
                self._manage_degradation_mix(category_name)
            else:
                print("Invalid choice")

    def _add_format_entry(self, category_name):
        fmt_tmpls = self.templates.get("format_templates", {})
        if not fmt_tmpls:
            print("❌ No format_templates in templates.json")
            return
        print(f"Available: {', '.join(sorted(fmt_tmpls.keys()))}")
        tmpl = input("Template name: ").strip()
        if tmpl not in fmt_tmpls:
            print(f"❌ '{tmpl}' not found")
            return
        try:
            weight = int(input("Weight (e.g. 50): ").strip())
        except ValueError:
            print("❌ Invalid weight")
            return
        mode = input(f"source_mode ({'/'.join(sorted(VALID_SOURCE_MODES))}): ").strip()
        if mode not in VALID_SOURCE_MODES:
            print(f"❌ Invalid source_mode '{mode}'")
            return
        self.categories[category_name].setdefault("formats", []).append(
            {"template": tmpl, "weight": weight, "source_mode": mode, "degradation_mix": {}}
        )
        self.modified = True
        idx = len(self.categories[category_name]["formats"]) - 1
        print(f"✓ Added [{idx}]: {tmpl} / {mode} / weight={weight}  → set degradation_mix via option d")

    def _remove_format_entry(self, category_name):
        formats = self.categories[category_name].get("formats", [])
        if not formats:
            print("No format entries.")
            return
        try:
            idx = int(input(f"Index to remove (0–{len(formats)-1}): ").strip())
        except ValueError:
            print("❌ Invalid index")
            return
        if not (0 <= idx < len(formats)):
            print("❌ Out of range")
            return
        removed = formats.pop(idx)
        self.modified = True
        print(f"✓ Removed [{idx}]: {removed.get('template', '?')}")

    def _edit_format_entry(self, category_name):
        formats = self.categories[category_name].get("formats", [])
        if not formats:
            print("No format entries.")
            return
        try:
            idx = int(input(f"Index to edit (0–{len(formats)-1}): ").strip())
        except ValueError:
            print("❌ Invalid index")
            return
        if not (0 <= idx < len(formats)):
            print("❌ Out of range")
            return
        entry = formats[idx]
        w_str = input(f"  New weight (current: {entry.get('weight')}): ").strip()
        if w_str:
            try:
                entry["weight"] = int(w_str)
                self.modified = True
            except ValueError:
                print("  ⚠️  Invalid weight, keeping current")
        mode_str = input(f"  New source_mode (current: {entry.get('source_mode')}): ").strip()
        if mode_str:
            if mode_str in VALID_SOURCE_MODES:
                entry["source_mode"] = mode_str
                self.modified = True
            else:
                print(f"  ⚠️  Invalid source_mode, keeping current")
        print(f"✓ Format entry [{idx}] updated")

    def _manage_degradation_mix(self, category_name):
        formats = self.categories[category_name].get("formats", [])
        if not formats:
            print("No format entries.")
            return
        try:
            idx = int(input(f"Format index (0–{len(formats)-1}): ").strip())
        except ValueError:
            print("❌ Invalid index")
            return
        if not (0 <= idx < len(formats)):
            print("❌ Out of range")
            return
        deg_tmpls = self.templates.get("degradation_templates", {})
        entry = formats[idx]
        tmpl_name = entry.get("template", "?")
        while True:
            mix = entry.setdefault("degradation_mix", {})
            total_w = sum(mix.values()) or 1
            print(f"\n  degradation_mix for [{idx}] {tmpl_name}:")
            if not mix:
                print("    (empty)")
            else:
                for dname, dw in mix.items():
                    print(f"    {dname:<30} weight={dw}  ({round(dw/total_w*100,1):.1f}%)")
            print(f"  Available: {', '.join(sorted(deg_tmpls.keys()))}")
            print("    a) Add/update  r) Remove  x) Back")
            sub = input("  Choice: ").strip().lower()
            if sub == "x":
                break
            elif sub == "a":
                dname = input("    Template name: ").strip()
                if dname not in deg_tmpls:
                    print(f"    ❌ '{dname}' not found")
                    continue
                try:
                    dw = int(input(f"    Weight (current: {mix.get(dname, 0)}): ").strip())
                except ValueError:
                    print("    ❌ Invalid weight")
                    continue
                if dw <= 0:
                    print("    ❌ Weight must be positive")
                    continue
                mix[dname] = dw
                self.modified = True
                print(f"    ✓ Set {dname} = {dw}")
            elif sub == "r":
                dname = input("    Template name to remove: ").strip()
                if dname in mix:
                    del mix[dname]
                    self.modified = True
                    print(f"    ✓ Removed {dname}")
                else:
                    print(f"    ❌ Not in mix")
            else:
                print("    Invalid choice")

    def manage_templates(self):
        while True:
            fmt_tmpls = self.templates.get("format_templates", {})
            deg_tmpls = self.templates.get("degradation_templates", {})
            print(f"\n── Templates Manager ─────────────────────────────────────────")
            print(f"  format_templates      : {', '.join(sorted(fmt_tmpls.keys())) or '(none)'}")
            print(f"  degradation_templates : {', '.join(sorted(deg_tmpls.keys())) or '(none)'}")
            print("  fa) List format templates  fb) Add format  fc) Remove format")
            print("  da) List degradation  db) Add/edit degradation (JSON)  dc) Remove degradation")
            print("  x) Back")
            sub = input("Choice: ").strip().lower()
            if sub == "x":
                break
            elif sub == "fa":
                self._list_format_templates()
            elif sub == "fb":
                self._add_format_template()
            elif sub == "fc":
                self._remove_format_template()
            elif sub == "da":
                self._list_degradation_templates()
            elif sub == "db":
                self._add_edit_degradation_template()
            elif sub == "dc":
                self._remove_degradation_template()
            else:
                print("Invalid choice")

    def _list_format_templates(self):
        fmt_tmpls = self.templates.get("format_templates", {})
        if not fmt_tmpls:
            print("No format templates defined.")
            return
        print(f"\n{'Name':<16} {'base_x':<8} {'AR':<6} {'scale':<7} {'gt_size':<14} {'lr_size':<14} Description")
        print("-" * 105)
        for name, spec in sorted(fmt_tmpls.items()):
            base_x = spec.get("base_x", "–")
            ar     = spec.get("aspect_ratio", "?")
            scale  = spec.get("scale", "?")
            gt     = spec.get("gt_size", "?")
            lr     = spec.get("lr_size", "?")
            desc   = spec.get("description", "")
            print(f"  {name:<14} {str(base_x):<8} {ar:<6} {str(scale):<7} {str(gt):<14} {str(lr):<14} {desc}")

    def _add_format_template(self):
        """Add a new format template using declarative parameters (base_x, aspect_ratio, scale)."""
        fmt_tmpls = self.templates.setdefault("format_templates", {})

        # --- base_x ---
        presets_str = ", ".join(str(x) for x in BASE_X_PRESETS)
        print(f"  Common base_x values: {presets_str}  (or enter any custom value)")
        try:
            base_x = int(input("  base_x (GT width): ").strip())
        except ValueError:
            print("❌ Invalid number")
            return
        if base_x <= 0:
            print("❌ base_x must be positive")
            return

        # --- aspect_ratio ---
        ar_options = ", ".join(sorted(ASPECT_RATIOS))
        ar = input(f"  aspect_ratio ({ar_options}): ").strip()
        if ar not in ASPECT_RATIOS:
            print(f"❌ '{ar}' is not supported. Choose from: {ar_options}")
            return

        # --- scale ---
        try:
            scale = int(input("  scale (e.g. 3): ").strip())
        except ValueError:
            print("❌ Invalid number")
            return
        if scale <= 0:
            print("❌ scale must be positive")
            return

        # --- compute and validate sizes ---
        try:
            gt_size, lr_size = compute_format_sizes(base_x, ar, scale)
        except ValueError as exc:
            print(f"❌ {exc}")
            return

        # --- show preview ---
        ar_slug = ar.replace(":", "")
        auto_name = f"{base_x}_{ar_slug}"
        print(f"\n  Preview: gt_size={gt_size}  lr_size={lr_size}")
        name_input = input(f"  Template name [{auto_name}]: ").strip()
        name = name_input or auto_name
        if not name:
            print("❌ Name cannot be empty")
            return
        if name in fmt_tmpls:
            if input(f"  ⚠️  '{name}' already exists. Overwrite? (yes/no): ").strip().lower() != "yes":
                print("Cancelled.")
                return

        desc = input("  Description (optional): ").strip()

        try:
            fmt_tmpls[name] = build_format_template(base_x, ar, scale, desc)
        except ValueError as exc:
            print(f"❌ {exc}")
            return

        self.templates_modified = True
        print(f"✓ Added format template '{name}': gt_size={gt_size}, lr_size={lr_size}")

    def _remove_format_template(self):
        fmt_tmpls = self.templates.get("format_templates", {})
        if not fmt_tmpls:
            print("No format templates.")
            return
        print(f"Templates: {', '.join(sorted(fmt_tmpls.keys()))}")
        name = input("Name to remove: ").strip()
        if name not in fmt_tmpls:
            print(f"❌ '{name}' not found")
            return
        del fmt_tmpls[name]
        self.templates_modified = True
        print(f"✓ Removed format template '{name}'")

    def _list_degradation_templates(self):
        deg_tmpls = self.templates.get("degradation_templates", {})
        if not deg_tmpls:
            print("No degradation templates defined.")
            return
        print()
        for name, spec in sorted(deg_tmpls.items()):
            desc = spec.get("description", "")
            print(f"  {name}")
            if desc:
                print(f"    {desc}")
            for key in ("blur", "compression", "noise", "chroma", "color"):
                if key in spec:
                    print(f"    {key}: {spec[key]}")

    def _add_edit_degradation_template(self):
        deg_tmpls = self.templates.setdefault("degradation_templates", {})
        print(f"Existing: {', '.join(sorted(deg_tmpls.keys())) or '(none)'}")
        name = input("Template name (new or existing): ").strip()
        if not name:
            print("❌ Name cannot be empty")
            return
        print("Paste JSON definition (end with blank line):")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        try:
            spec = json.loads("\n".join(lines))
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse error: {e}")
            return
        deg_tmpls[name] = spec
        self.templates_modified = True
        print(f"✓ Saved degradation template '{name}'")

    def _remove_degradation_template(self):
        deg_tmpls = self.templates.get("degradation_templates", {})
        if not deg_tmpls:
            print("No degradation templates.")
            return
        print(f"Templates: {', '.join(sorted(deg_tmpls.keys()))}")
        name = input("Name to remove: ").strip()
        if name not in deg_tmpls:
            print(f"❌ '{name}' not found")
            return
        del deg_tmpls[name]
        self.templates_modified = True
        print(f"✓ Removed degradation template '{name}'")

    def _ensure_source_dirs(self):
        self.config.setdefault("source_dirs", [])
        return self.config["source_dirs"]

    def list_source_dirs(self):
        source_dirs = self._ensure_source_dirs()
        if not source_dirs:
            print("No source directories configured.")
            return
        print(f"\n{'#':<5} {'Path':<65} {'Extensions'}")
        print("-" * 100)
        for i, entry in enumerate(source_dirs):
            path = entry.get("path", "?")
            exts = ", ".join(entry.get("extensions", []))
            print(f"{i:<5} {path:<65} {exts}")

    def add_source_dir(self):
        source_dirs = self._ensure_source_dirs()
        path = input("Directory path: ").strip()
        if not path:
            print("❌ Path cannot be empty")
            return
        if any(d.get("path") == path for d in source_dirs):
            print(f"❌ Already configured: {path}")
            return
        exts_str = input("Extensions [default: .mkv,.mp4,.avi]: ").strip()
        exts = [e.strip() for e in exts_str.split(",")] if exts_str else [".mkv", ".mp4", ".avi"]
        source_dirs.append({"path": path, "extensions": exts})
        self.modified = True
        print(f"✓ Added: {path}")

    def edit_source_dir(self):
        source_dirs = self._ensure_source_dirs()
        if not source_dirs:
            print("No source directories configured.")
            return
        self.list_source_dirs()
        try:
            idx = int(input("Number to edit: ").strip())
        except ValueError:
            print("❌ Invalid input")
            return
        if not (0 <= idx < len(source_dirs)):
            print("❌ Invalid index")
            return
        entry = source_dirs[idx]
        new_path = input(f"New path (current: {entry['path']}): ").strip()
        if new_path:
            entry["path"] = new_path
        cur_exts = ", ".join(entry.get("extensions", []))
        new_exts = input(f"New extensions (current: {cur_exts}): ").strip()
        if new_exts:
            entry["extensions"] = [e.strip() for e in new_exts.split(",")]
        self.modified = True
        print(f"✓ Updated source directory #{idx}")

    def remove_source_dir(self):
        source_dirs = self._ensure_source_dirs()
        if not source_dirs:
            print("No source directories configured.")
            return
        self.list_source_dirs()
        try:
            idx = int(input("Number to remove: ").strip())
        except ValueError:
            print("❌ Invalid input")
            return
        if not (0 <= idx < len(source_dirs)):
            print("❌ Invalid index")
            return
        path = source_dirs[idx].get("path", "?")
        if input(f"⚠️  Remove '{path}'? (yes/no): ").strip().lower() != "yes":
            print("Cancelled.")
            return
        source_dirs.pop(idx)
        self.modified = True
        print(f"✓ Removed: {path}")

    def rescan_file_list(self):
        source_dirs = self._ensure_source_dirs()
        if not source_dirs:
            print("❌ No source directories configured. Add via option 14.")
            return
        if self.modified:
            if input("⚠️  Unsaved changes. Save before rescanning? (yes/no): ").strip().lower() == "yes":
                self.save(backup=False)
        existing_by_path = {v.get("path", ""): v for v in self.config.get("videos", [])}
        found_paths: List[str] = []
        seen_paths: set = set()
        for dir_cfg in source_dirs:
            video_dir = dir_cfg.get("path", "")
            extensions = dir_cfg.get("extensions", [".mkv", ".mp4", ".avi"])
            if not os.path.exists(video_dir):
                print(f"⚠️  Not found (skipped): {video_dir}")
                continue
            exts_lower = {e.lower() for e in extensions}
            for p in Path(video_dir).rglob("*"):
                if p.is_file() and p.suffix.lower() in exts_lower:
                    path_str = str(p)
                    if path_str not in seen_paths:
                        seen_paths.add(path_str)
                        found_paths.append(path_str)
        new_videos: List[dict] = []
        added = kept = 0
        for path in sorted(found_paths):
            if path in existing_by_path:
                new_videos.append(existing_by_path[path])
                kept += 1
            else:
                new_videos.append({"name": Path(path).stem, "path": path, "categories": []})
                added += 1
        self.config["videos"] = new_videos
        self.videos = new_videos
        self.modified = True
        print(f"✅ Rescan complete: {len(new_videos)} videos ({kept} kept, {added} newly added)")
        if added:
            print("   Use options 5/6/7 to assign categories.")

    def show_validation_report(self):
        print("\n── Validation Report ─────────────────────────────────────────")
        t_errors = validate_templates(self.templates)
        c_errors = validate_active_config(self.config, self.templates)
        all_errors = t_errors + c_errors
        if not all_errors:
            print("✅ No validation errors found.")
        else:
            print(f"❌ {len(all_errors)} error(s) found:\n")
            for e in all_errors:
                print(f"  • {e}")


def print_menu():
    print("\n" + "=" * 50)
    print("VIDEO CATEGORY MANAGER  (v2 – new config model)")
    print("=" * 50)
    print("── Videos ──────────────────────────────")
    print("1.  List all videos")
    print("2.  List videos by category")
    print("3.  List unassigned videos")
    print("4.  Search videos by name")
    print("5.  Assign video(s) to categories")
    print("6.  Interactive multi-select")
    print("7.  Multi-assign by pattern")
    print("8.  Remove from category")
    print("9.  Reset all assignments")
    print("── Categories & Formats ─────────────────")
    print("10. Show statistics")
    print("11. Manage categories (add / remove / edit targets)")
    print("12. Manage category formats & degradation mix")
    print("── Source Directories ───────────────────")
    print("13. List source directories")
    print("14. Add source directory")
    print("15. Edit source directory")
    print("16. Remove source directory")
    print("17. Rescan file list")
    print("── Config & Templates ───────────────────")
    print("18. Manage templates (format & degradation templates)")
    print("19. Show config validation report")
    print("20. Create new active config file")
    print("─────────────────────────────────────────")
    print("s.  Save config changes")
    print("t.  Save template changes")
    print("q.  Quit")


def get_categories_interactive(manager, current_categories=None):
    try:
        return select_categories(
            available_categories=list(manager.categories.keys()),
            current_categories=current_categories,
        )
    except Exception as e:
        print(f"⚠️  Curses UI unavailable: {e} – using simple input")
        return _get_categories_simple(manager, current_categories)


def _get_categories_simple(manager, current_categories=None):
    print(f"\nAvailable categories: {', '.join(manager.categories.keys())}")
    if current_categories:
        print(f"Current: {', '.join(current_categories)}")
    print("Enter comma-separated names, or 'none' to clear:")
    while True:
        val = input("Categories: ").strip()
        if not val:
            continue
        if val.lower() == "none":
            return []
        cats = [c.strip() for c in val.split(",")]
        invalid = [c for c in cats if c not in manager.categories]
        if invalid:
            print(f"❌ Invalid: {', '.join(invalid)}")
            continue
        return cats


def main():
    try:
        script_dir = Path(__file__).parent
        active_config_path = script_dir / "generator_config_v2.active.json"
        templates_path = script_dir / "templates.json"

        if not active_config_path.exists():
            print("⚠️  No active config found – creating default …")
            save_active_config(create_default_active_config(), str(active_config_path))
            print(f"✓ Created {active_config_path.name}")
            print("  → Adjust root_path and source_dirs, then restart.")

        ensure_templates_file(str(templates_path))

        print(f"📂 Config  : {active_config_path.name}")
        print(f"📋 Templates: {templates_path.name}")
        manager = VideoManager(str(active_config_path), str(templates_path))
        manager.load()

    except Exception as e:
        print(f"❌ Error initializing Video Manager: {e}")
        traceback.print_exc()
        sys.exit(1)

    while True:
        choice = ""
        try:
            print_menu()
            choice = input("\nChoice: ").strip().lower()

            if choice == "q":
                if manager.modified or manager.templates_modified:
                    s = input("Save changes before quitting? (y/n): ").strip().lower()
                    if s == "y":
                        if manager.modified:
                            manager.save()
                        if manager.templates_modified:
                            manager.save_templates()
                print("Goodbye!")
                break

            elif choice == "s":
                if manager.modified:
                    manager.save()
                else:
                    print("No config changes to save.")

            elif choice == "t":
                if manager.templates_modified:
                    manager.save_templates()
                else:
                    print("No template changes to save.")

            elif choice == "1":
                show_all = input("Show all videos? (y/n, default=first 20): ").strip().lower()
                max_d = None if show_all == "y" else 20
                manager.print_video_list(manager.list_videos(), max_d)

            elif choice == "2":
                print(f"Categories: {', '.join(manager.categories.keys())}")
                cat = input("Category: ").strip()
                if cat in manager.categories:
                    manager.print_video_list(manager.list_videos(category=cat))
                else:
                    print(f"❌ Unknown category: {cat}")

            elif choice == "3":
                manager.print_video_list(manager.list_videos(show_unassigned=True))

            elif choice == "4":
                pattern = input("Search pattern (regex): ").strip()
                if pattern:
                    manager.print_video_list(manager.list_videos(filter_pattern=pattern))

            elif choice == "5":
                print("\n🎯 Assign Videos to Categories")
                method = input("  a) Interactive (curses)  b) Enter IDs\nMethod (a/b): ").strip().lower()
                video_indices = []
                if method == "a":
                    filter_str = input("Optional filter: ").strip()
                    videos = manager.list_videos(filter_pattern=filter_str or None)
                    if not videos:
                        print("No videos found")
                        continue
                    try:
                        sorted_vids = _sorted_videos(videos)
                        selected = select_items(
                            items=[v for _, v in sorted_vids],
                            title="Select videos (Space toggle, Enter confirm)",
                            get_label=lambda v: _short_path(v.get("path", ""), depth=3),
                            get_details=lambda v: format_categories_display(v.get("categories", [])),
                        )
                        if selected is None:
                            print("❌ Cancelled")
                            continue
                        video_indices = [sorted_vids[i][0] for i in selected]
                        print(f"✓ Selected {len(video_indices)} videos")
                    except Exception as e:
                        print(f"⚠️  Curses UI failed: {e}. Use method b.")
                        continue
                elif method == "b":
                    ids_str = input("Video ID(s) (comma-separated): ").strip()
                    try:
                        video_indices = [int(x.strip()) for x in ids_str.split(",")]
                    except ValueError:
                        print("❌ Invalid IDs")
                        continue
                else:
                    print("❌ Invalid method")
                    continue
                if not video_indices:
                    print("No videos selected")
                    continue
                categories = get_categories_interactive(manager)
                if categories is None:
                    print("❌ Cancelled")
                    continue
                manager.assign_videos(video_indices, categories)

            elif choice == "6":
                filter_str = input("Optional filter (leave empty for all): ").strip()
                selected_ids = manager.interactive_select_videos(filter_str or None)
                if selected_ids:
                    print(f"\n✓ Selected {len(selected_ids)} videos")
                    categories = get_categories_interactive(manager)
                    if categories:
                        manager.assign_videos(selected_ids, categories)
                else:
                    print("Selection cancelled")

            elif choice == "7":
                pattern = input("Search pattern (text or regex): ").strip()
                if not pattern:
                    continue
                use_simple = "*" in pattern or not any(c in pattern for c in r"\.[](){}^$+?|")
                videos = manager.list_videos(filter_pattern=pattern, use_simple_search=use_simple)
                if not videos:
                    print(f"No videos match: {pattern}")
                    continue
                manager.print_video_list(videos)
                if input(f"\nAssign all {len(videos)} videos? (y/n): ").strip().lower() != "y":
                    continue
                categories = get_categories_interactive(manager)
                if categories:
                    manager.assign_videos([i for i, _ in videos], categories)

            elif choice == "8":
                print(f"Categories: {', '.join(manager.categories.keys())}")
                cat = input("Category to remove from: ").strip()
                if cat not in manager.categories:
                    print(f"❌ Unknown category: {cat}")
                    continue
                ids_str = input("Video ID(s) (comma-separated, or 'all'): ").strip()
                if ids_str.lower() == "all":
                    ids = list(range(len(manager.videos)))
                else:
                    try:
                        ids = [int(x.strip()) for x in ids_str.split(",")]
                    except ValueError:
                        print("❌ Invalid IDs")
                        continue
                manager.remove_from_category(ids, cat)

            elif choice == "9":
                manager.reset_all()

            elif choice == "10":
                manager.show_statistics()

            elif choice == "11":
                manager.manage_categories()

            elif choice == "12":
                manager.manage_category_formats()

            elif choice == "13":
                manager.list_source_dirs()

            elif choice == "14":
                manager.add_source_dir()

            elif choice == "15":
                manager.edit_source_dir()

            elif choice == "16":
                manager.remove_source_dir()

            elif choice == "17":
                manager.rescan_file_list()

            elif choice == "18":
                manager.manage_templates()

            elif choice == "19":
                manager.show_validation_report()

            elif choice == "20":
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                default_name = f"generator_config_new_{ts}.active.json"
                val = input(f"Output filename [{default_name}]: ").strip()
                out_path = str(script_dir / (val or default_name))
                if os.path.exists(out_path):
                    if input(f"⚠️  '{out_path}' exists. Overwrite? (yes/no): ").strip().lower() != "yes":
                        print("Cancelled.")
                        continue
                save_active_config(create_default_active_config(), out_path)
                print(f"✓ Created {out_path}")
                print("  → Adjust root_path and source_dirs, then reload.")

            else:
                print("Invalid choice")

        except EOFError:
            print("\n\n⚠️  End of input detected")
            break
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            if manager.modified or manager.templates_modified:
                try:
                    if input("\nSave changes? (y/n): ").strip().lower() == "y":
                        if manager.modified:
                            manager.save()
                        if manager.templates_modified:
                            manager.save_templates()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting without saving")
            break
        except Exception as e:
            print(f"\n⚠️  Error processing choice '{choice}': {e}")
            traceback.print_exc()
            print("\nContinuing…")
            continue


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
