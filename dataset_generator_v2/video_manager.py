#!/usr/bin/env python3
"""
VIDEO CATEGORY MANAGER  (v2 – new config model)
Central management UI for dataset_generator_v2.
"""

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ui import (
    console, Choice, Separator,
    ask_text, ask_int, ask_confirm, ask_select, ask_checkbox,
    print_success, print_error, print_warn, print_info,
    print_banner, print_rule, make_table,
)

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
        print_success(f"Loaded {len(self.videos)} videos")
        print_info(f"Categories: {cat_names}")

    def save(self, backup: bool = True):
        def _sort_key(video):
            cats = get_video_categories(video)
            return (-len(cats) if cats else 999, cats[0] if cats else "zzz", video.get("name", "").lower())
        self.videos.sort(key=_sort_key)
        self.config["videos"] = self.videos
        self.config["categories"] = self.categories
        save_active_config(self.config, self.config_path)
        print_success(f"Saved to {Path(self.config_path).name}")
        self.modified = False

    def save_templates(self, backup: bool = True):
        if backup and os.path.exists(self.templates_path):
            with open(self.templates_path, "r", encoding="utf-8") as f:
                old = f.read()
            with open(self.templates_path + ".backup", "w", encoding="utf-8") as f:
                f.write(old)
        _save_templates_io(self.templates, self.templates_path)
        print_success(f"Templates saved to {Path(self.templates_path).name}")
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
            print_warn("No videos found.")
            return
        sorted_vids = _sorted_videos(videos)
        display = sorted_vids if not max_display else sorted_vids[:max_display]
        t = make_table("ID", "Path", "Name", "Categories")
        for i, video in display:
            path_short = _short_path(video.get("path", ""), depth=3)[:34]
            cats = video.get("categories", [])
            cat_str = ", ".join(cats) if cats else "[dim]unassigned[/]"
            forced = video.get("forced_frames", {})
            if forced:
                parts = [f"{cat}:{n:,}" for cat, n in sorted(forced.items()) if n > 0]
                if parts:
                    cat_str += "  ⚡ " + "  ".join(parts)
            t.add_row(str(i), path_short, video["name"][:40], cat_str)
        console.print(t)
        if max_display and len(sorted_vids) > max_display:
            print_warn(f"… and {len(sorted_vids) - max_display} more (pass max_display=None to show all)")

    def assign_videos(self, video_indices, categories, mode="ask"):
        if not video_indices:
            return
        has_existing = any(
            bool(get_video_categories(self.videos[i]))
            for i in video_indices if 0 <= i < len(self.videos)
        )
        actual_mode = mode
        if mode == "ask" and has_existing:
            choice = ask_select(
                "Some videos already have categories – what to do?",
                [
                    Choice("Add to existing categories", value="add"),
                    Choice("Replace all categories",     value="replace"),
                ],
            )
            actual_mode = choice if choice else "add"
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
        print_success(f"{mode_text} {count} videos: {categories}")

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
        print_success(f"Removed {count} videos from category '{category}'")

    def reset_all(self):
        ok = ask_confirm("Reset ALL video assignments? This cannot be undone!", default=False)
        if not ok:
            print_info("Cancelled.")
            return
        for video in self.videos:
            video["categories"] = []
        self.modified = True
        print_success(f"Reset {len(self.videos)} videos")

    def interactive_select_videos(self, initial_filter=None):
        videos = (self.list_videos(filter_pattern=initial_filter, use_simple_search=True)
                  if initial_filter else self.list_videos())
        if not videos:
            print_warn("No videos found.")
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
            print_warn(f"Curses UI failed: {e}")
            return None

    def set_forced_frames(self, video_indices):
        if isinstance(video_indices, int):
            video_indices = [video_indices]
        valid = [i for i in video_indices if 0 <= i < len(self.videos)]
        if not valid:
            print_error("No valid video indices.")
            return
        all_cats: List[str] = []
        for idx in valid:
            for cat in get_video_categories(self.videos[idx]):
                if cat not in all_cats:
                    all_cats.append(cat)
        if not all_cats:
            print_error("Selected videos have no categories assigned.")
            return
        names = [self.videos[i].get("name", "?") for i in valid]
        if len(names) == 1:
            print_rule(f"Forced frames: {names[0]}")
        else:
            print_rule(f"Forced frames: {len(names)} videos")
            for n in names:
                console.print(f"  [dim]•[/] {n}")
        print_info(f"Categories: {', '.join(all_cats)}  |  blank=keep  0=auto  N=exact")
        new_values: Dict[str, Optional[int]] = {}
        for cat in all_cats:
            cur_values = [self.videos[i].get("forced_frames", {}).get(cat, 0) for i in valid]
            current_str = f"{cur_values[0]:,}" if len(set(cur_values)) == 1 else "mixed"
            raw = ask_text(f"{cat} (current: {current_str}):", default="")
            if not raw or not raw.strip():
                new_values[cat] = None
                continue
            try:
                value = int(raw.strip())
            except ValueError:
                print_warn("Invalid number, keeping current")
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
            print_success(f"Applied to {len(valid)} video(s): {parts}")
        else:
            print_info("No changes made.")

    def show_statistics(self):
        print_rule("Statistics")
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
                    forced_by_cat[cat].append(
                        (video.get("name", "?"), _short_path(video.get("path", ""), depth=3), n)
                    )
        t = make_table("Metric", "Value")
        t.add_row("Total videos",   str(len(self.videos)))
        t.add_row("Unassigned",     str(unassigned))
        console.print(t)
        console.print()
        for cat in sorted(self.categories.keys()):
            cfg = self.categories[cat]
            target = cfg.get("target_total", "?")
            target_str = f"{target:,}" if isinstance(target, int) else str(target)
            ft = make_table("Format", "Mode", "Weight", "Share", "Degradation mix")
            formats = cfg.get("formats", [])
            total_weight = sum(f.get("weight", 0) for f in formats) or 1
            for fmt in formats:
                tmpl  = fmt.get("template", "?")
                w     = fmt.get("weight", 0)
                mode  = fmt.get("source_mode", "?")
                share = f"{round(w / total_weight * 100, 1):.1f}%"
                deg   = ", ".join(f"{k}:{v}" for k, v in fmt.get("degradation_mix", {}).items())
                ft.add_row(tmpl, mode, str(w), share, deg)
            fl = forced_by_cat.get(cat, [])
            forced_total = sum(n for _, _, n in fl)
            header_extra = ""
            if fl and isinstance(target, int):
                remaining = max(0, target - forced_total)
                header_extra = (
                    f"  forced {forced_total:,} / remaining {remaining:,}"
                )
            console.print(
                f"[bold cyan]▸  {cat}[/]  [dim]target: {target_str}  videos: {cat_counts.get(cat, 0)}{header_extra}[/]"
            )
            console.print(ft)
            if fl:
                ft2 = make_table("Video", "Path", "Forced frames")
                for name, sp, n in sorted(fl, key=lambda x: x[0].lower()):
                    ft2.add_row(name[:40], sp[:36], f"{n:,}")
                console.print(ft2)

    def manage_categories(self):
        while True:
            action = ask_select(
                "Categories",
                [
                    Choice("List categories",       value="list"),
                    Choice("Add category",          value="add"),
                    Choice("Remove category",       value="remove"),
                    Choice("Edit target total",     value="edit"),
                    Separator(),
                    Choice("← Back",                value="back"),
                ],
            )
            if action is None or action == "back":
                break
            elif action == "list":
                self._list_categories()
            elif action == "add":
                self._add_category()
            elif action == "remove":
                self._remove_category()
            elif action == "edit":
                self._edit_category_target()

    def _list_categories(self):
        if not self.categories:
            print_warn("No categories configured.")
            return
        t = make_table("Category", "Target total", "Formats")
        for cat in sorted(self.categories.keys()):
            cfg = self.categories[cat]
            target = cfg.get("target_total", "?")
            target_str = f"{target:,}" if isinstance(target, int) else str(target)
            t.add_row(cat, target_str, str(len(cfg.get("formats", []))))
        console.print(t)

    def _add_category(self):
        name = ask_text("New category name:", validate=lambda v: True if v.strip() else "Name required")
        if name is None:
            return
        name = name.strip().lower()
        if not name:
            print_error("Name cannot be empty")
            return
        if name in self.categories:
            print_error(f"Category '{name}' already exists")
            return
        target = ask_int("Target total:", default=50000, min_val=1)
        if target is None:
            return
        self.categories[name] = {"target_total": target, "formats": []}
        self.config["categories"] = self.categories
        self.modified = True
        print_success(f"Added category '{name}' with target {target:,}")

    def _remove_category(self):
        if not self.categories:
            print_warn("No categories configured.")
            return
        name = ask_select(
            "Select category to remove:",
            [Choice(c, value=c) for c in sorted(self.categories.keys())] + [Separator(), Choice("← Cancel", value=None)],
        )
        if not name:
            return
        affected = sum(1 for v in self.videos if name in get_video_categories(v))
        suffix = f" and unassign {affected} video(s)" if affected else ""
        ok = ask_confirm(f"Remove '{name}'{suffix}?", default=False)
        if not ok:
            print_info("Cancelled.")
            return
        for video in self.videos:
            cats = get_video_categories(video)
            if name in cats:
                cats.remove(name)
                video["categories"] = cats
        del self.categories[name]
        self.config["categories"] = self.categories
        self.modified = True
        msg = f"Removed category '{name}'"
        if affected:
            msg += f", unassigned {affected} video(s)"
        print_success(msg)

    def _edit_category_target(self):
        if not self.categories:
            print_warn("No categories configured.")
            return
        name = ask_select(
            "Select category to edit:",
            [Choice(c, value=c) for c in sorted(self.categories.keys())] + [Separator(), Choice("← Cancel", value=None)],
        )
        if not name:
            return
        current = self.categories[name].get("target_total", 0)
        new_target = ask_int(f"New target_total (current: {current:,}):", default=current, min_val=1)
        if new_target is None:
            return
        self.categories[name]["target_total"] = new_target
        self.modified = True
        print_success(f"'{name}' target: {current:,} → {new_target:,}")

    def manage_category_formats(self, category_name=None):
        if category_name is None:
            if not self.categories:
                print_warn("No categories configured.")
                return
            category_name = ask_select(
                "Select category:",
                [Choice(c, value=c) for c in sorted(self.categories.keys())] + [Separator(), Choice("← Cancel", value=None)],
            )
            if not category_name:
                return
        if category_name not in self.categories:
            print_error(f"Category '{category_name}' not found")
            return
        while True:
            cat_cfg = self.categories[category_name]
            formats = cat_cfg.get("formats", [])
            # build a summary table before the menu
            console.print()
            print_rule(f"Formats for '{category_name}'")
            if formats:
                total_w = sum(f.get("weight", 0) for f in formats) or 1
                t = make_table("#", "Template", "Mode", "Weight", "Share", "Degradation mix")
                for i, fmt in enumerate(formats):
                    tmpl  = fmt.get("template", "?")
                    w     = fmt.get("weight", 0)
                    mode  = fmt.get("source_mode", "?")
                    share = f"{round(w / total_w * 100, 1):.1f}%"
                    deg   = ", ".join(f"{k}={v}" for k, v in fmt.get("degradation_mix", {}).items())
                    t.add_row(str(i), tmpl, mode, str(w), share, deg)
                console.print(t)
            else:
                print_warn("No format entries yet.")

            action = ask_select(
                "Format actions:",
                [
                    Choice("Add format entry",          value="add"),
                    Choice("Remove format entry",       value="remove"),
                    Choice("Edit weight / source_mode", value="edit"),
                    Choice("Manage degradation mix",    value="deg"),
                    Separator(),
                    Choice("← Back",                    value="back"),
                ],
            )
            if action is None or action == "back":
                break
            elif action == "add":
                self._add_format_entry(category_name)
            elif action == "remove":
                self._remove_format_entry(category_name)
            elif action == "edit":
                self._edit_format_entry(category_name)
            elif action == "deg":
                self._manage_degradation_mix(category_name)

    def _add_format_entry(self, category_name):
        fmt_tmpls = self.templates.get("format_templates", {})
        if not fmt_tmpls:
            print_error("No format_templates in templates.json")
            return
        tmpl = ask_select(
            "Select format template:",
            [Choice(n, value=n) for n in sorted(fmt_tmpls.keys())] + [Separator(), Choice("← Cancel", value=None)],
        )
        if not tmpl:
            return
        weight = ask_int("Weight (e.g. 50):", default=50, min_val=1)
        if weight is None:
            return
        mode = ask_select(
            "source_mode:",
            [Choice(m, value=m) for m in sorted(VALID_SOURCE_MODES)],
        )
        if mode is None:
            return
        self.categories[category_name].setdefault("formats", []).append(
            {"template": tmpl, "weight": weight, "source_mode": mode, "degradation_mix": {}}
        )
        self.modified = True
        idx = len(self.categories[category_name]["formats"]) - 1
        print_success(f"Added [{idx}]: {tmpl} / {mode} / weight={weight}  → add degradation mix next")

    def _remove_format_entry(self, category_name):
        formats = self.categories[category_name].get("formats", [])
        if not formats:
            print_warn("No format entries.")
            return
        choices = [
            Choice(f"[{i}] {f.get('template','?')} / {f.get('source_mode','?')} / w={f.get('weight',0)}", value=i)
            for i, f in enumerate(formats)
        ] + [Separator(), Choice("← Cancel", value=None)]
        idx = ask_select("Select entry to remove:", choices)
        if idx is None:
            return
        ok = ask_confirm(f"Remove [{idx}] '{formats[idx].get('template','?')}'?", default=False)
        if not ok:
            print_info("Cancelled.")
            return
        removed = formats.pop(idx)
        self.modified = True
        print_success(f"Removed [{idx}]: {removed.get('template', '?')}")

    def _edit_format_entry(self, category_name):
        formats = self.categories[category_name].get("formats", [])
        if not formats:
            print_warn("No format entries.")
            return
        choices = [
            Choice(f"[{i}] {f.get('template','?')} / {f.get('source_mode','?')} / w={f.get('weight',0)}", value=i)
            for i, f in enumerate(formats)
        ] + [Separator(), Choice("← Cancel", value=None)]
        idx = ask_select("Select entry to edit:", choices)
        if idx is None:
            return
        entry = formats[idx]
        new_w = ask_int(f"New weight (current: {entry.get('weight')}):", default=entry.get("weight", 1), min_val=1)
        if new_w is not None:
            entry["weight"] = new_w
            self.modified = True
        new_mode = ask_select(
            f"New source_mode (current: {entry.get('source_mode')}):",
            [Choice(m, value=m) for m in sorted(VALID_SOURCE_MODES)] + [Separator(), Choice("Keep current", value=None)],
        )
        if new_mode is not None:
            entry["source_mode"] = new_mode
            self.modified = True
        print_success(f"Format entry [{idx}] updated")

    def _manage_degradation_mix(self, category_name):
        formats = self.categories[category_name].get("formats", [])
        if not formats:
            print_warn("No format entries.")
            return
        choices = [
            Choice(f"[{i}] {f.get('template','?')} / {f.get('source_mode','?')}", value=i)
            for i, f in enumerate(formats)
        ] + [Separator(), Choice("← Cancel", value=None)]
        idx = ask_select("Select format entry:", choices)
        if idx is None:
            return
        deg_tmpls = self.templates.get("degradation_templates", {})
        entry = formats[idx]
        tmpl_name = entry.get("template", "?")
        while True:
            mix = entry.setdefault("degradation_mix", {})
            total_w = sum(mix.values()) or 1
            # show current mix table
            console.print()
            print_rule(f"Degradation mix for [{idx}] {tmpl_name}")
            if mix:
                t = make_table("Template", "Weight", "Share")
                for dname, dw in mix.items():
                    t.add_row(dname, str(dw), f"{round(dw/total_w*100,1):.1f}%")
                console.print(t)
            else:
                print_warn("Mix is empty.")

            action = ask_select(
                "Degradation mix actions:",
                [
                    Choice("Add / update entry",   value="add"),
                    Choice("Remove entry",          value="remove"),
                    Separator(),
                    Choice("← Back",                value="back"),
                ],
            )
            if action is None or action == "back":
                break
            elif action == "add":
                if not deg_tmpls:
                    print_error("No degradation templates defined.")
                    continue
                dname = ask_select(
                    "Select degradation template:",
                    [Choice(n, value=n) for n in sorted(deg_tmpls.keys())] + [Separator(), Choice("← Cancel", value=None)],
                )
                if not dname:
                    continue
                dw = ask_int(f"Weight (current: {mix.get(dname, 0)}):", default=mix.get(dname, 50), min_val=1)
                if dw is None:
                    continue
                mix[dname] = dw
                self.modified = True
                print_success(f"Set {dname} = {dw}")
            elif action == "remove":
                if not mix:
                    print_warn("Mix is already empty.")
                    continue
                dname = ask_select(
                    "Remove which entry:",
                    [Choice(n, value=n) for n in sorted(mix.keys())] + [Separator(), Choice("← Cancel", value=None)],
                )
                if dname and dname in mix:
                    del mix[dname]
                    self.modified = True
                    print_success(f"Removed {dname}")

    def manage_templates(self):
        while True:
            fmt_tmpls = self.templates.get("format_templates", {})
            deg_tmpls = self.templates.get("degradation_templates", {})
            action = ask_select(
                "Templates",
                [
                    Choice(f"List format templates  [{len(fmt_tmpls)}]",      value="fa"),
                    Choice("Add format template",                               value="fb"),
                    Choice("Remove format template",                            value="fc"),
                    Separator(),
                    Choice(f"List degradation templates  [{len(deg_tmpls)}]",  value="da"),
                    Choice("Add / edit degradation template (JSON)",            value="db"),
                    Choice("Remove degradation template",                       value="dc"),
                    Separator(),
                    Choice("← Back",                                            value="back"),
                ],
            )
            if action is None or action == "back":
                break
            elif action == "fa":
                self._list_format_templates()
            elif action == "fb":
                self._add_format_template()
            elif action == "fc":
                self._remove_format_template()
            elif action == "da":
                self._list_degradation_templates()
            elif action == "db":
                self._add_edit_degradation_template()
            elif action == "dc":
                self._remove_degradation_template()

    def _list_format_templates(self):
        fmt_tmpls = self.templates.get("format_templates", {})
        if not fmt_tmpls:
            print_warn("No format templates defined.")
            return
        t = make_table("Name", "base_x", "AR", "Scale", "GT size", "LR size", "Description")
        for name, spec in sorted(fmt_tmpls.items()):
            t.add_row(
                name,
                str(spec.get("base_x", "–")),
                spec.get("aspect_ratio", "?"),
                str(spec.get("scale", "?")),
                str(spec.get("gt_size", "?")),
                str(spec.get("lr_size", "?")),
                spec.get("description", ""),
            )
        console.print(t)

    def _add_format_template(self):
        """Add a new format template using declarative parameters (base_x, aspect_ratio, scale)."""
        fmt_tmpls = self.templates.setdefault("format_templates", {})

        # --- base_x: preset list + custom option ---
        preset_choices = [Choice(str(x), value=x) for x in BASE_X_PRESETS]
        preset_choices += [Separator(), Choice("Custom value…", value="custom")]
        base_x_sel = ask_select("base_x (GT width):", preset_choices)
        if base_x_sel is None:
            return
        if base_x_sel == "custom":
            base_x = ask_int("Custom base_x:", min_val=1)
            if base_x is None:
                return
        else:
            base_x = base_x_sel

        # --- aspect_ratio ---
        ar = ask_select(
            "aspect_ratio:",
            [Choice(r, value=r) for r in sorted(ASPECT_RATIOS.keys())],
        )
        if ar is None:
            return

        # --- scale ---
        scale = ask_int("scale (e.g. 3):", default=3, min_val=1)
        if scale is None:
            return

        # --- compute and validate sizes ---
        try:
            gt_size, lr_size = compute_format_sizes(base_x, ar, scale)
        except ValueError as exc:
            print_error(str(exc))
            return

        # --- show preview + name ---
        ar_slug = ar.replace(":", "")
        auto_name = f"{base_x}_{ar_slug}"
        print_info(f"Preview: gt_size={gt_size}  lr_size={lr_size}")
        name = ask_text(f"Template name:", default=auto_name,
                        validate=lambda v: True if v.strip() else "Name required")
        if name is None:
            return
        name = name.strip()
        if name in fmt_tmpls:
            ok = ask_confirm(f"'{name}' already exists. Overwrite?", default=False)
            if not ok:
                print_info("Cancelled.")
                return

        desc = ask_text("Description (optional):", default="")
        if desc is None:
            desc = ""

        try:
            fmt_tmpls[name] = build_format_template(base_x, ar, scale, desc)
        except ValueError as exc:
            print_error(str(exc))
            return

        self.templates_modified = True
        print_success(f"Added '{name}': gt_size={gt_size}, lr_size={lr_size}")

    def _remove_format_template(self):
        fmt_tmpls = self.templates.get("format_templates", {})
        if not fmt_tmpls:
            print_warn("No format templates.")
            return
        name = ask_select(
            "Select template to remove:",
            [Choice(n, value=n) for n in sorted(fmt_tmpls.keys())] + [Separator(), Choice("← Cancel", value=None)],
        )
        if not name:
            return
        ok = ask_confirm(f"Remove '{name}'?", default=False)
        if not ok:
            print_info("Cancelled.")
            return
        del fmt_tmpls[name]
        self.templates_modified = True
        print_success(f"Removed format template '{name}'")

    def _list_degradation_templates(self):
        deg_tmpls = self.templates.get("degradation_templates", {})
        if not deg_tmpls:
            print_warn("No degradation templates defined.")
            return
        for name, spec in sorted(deg_tmpls.items()):
            desc = spec.get("description", "")
            console.print(f"  [bold cyan]{name}[/]" + (f"  [dim]{desc}[/]" if desc else ""))
            for key in ("blur", "compression", "noise", "chroma", "color"):
                if key in spec:
                    console.print(f"    [dim]{key}:[/] {spec[key]}")

    def _add_edit_degradation_template(self):
        deg_tmpls = self.templates.setdefault("degradation_templates", {})
        existing = sorted(deg_tmpls.keys())
        choices = [Choice(n, value=n) for n in existing] + [Separator(), Choice("+ New template…", value="__new__")]
        sel = ask_select("Select template to edit (or create new):", choices)
        if sel is None:
            return
        if sel == "__new__":
            name = ask_text("New template name:", validate=lambda v: True if v.strip() else "Name required")
            if name is None:
                return
            name = name.strip()
        else:
            name = sel
        print_info("Paste / type JSON definition. End with a blank line:")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if not line:
                break
            lines.append(line)
        try:
            spec = json.loads("\n".join(lines))
        except json.JSONDecodeError as e:
            print_error(f"JSON parse error: {e}")
            return
        deg_tmpls[name] = spec
        self.templates_modified = True
        print_success(f"Saved degradation template '{name}'")

    def _remove_degradation_template(self):
        deg_tmpls = self.templates.get("degradation_templates", {})
        if not deg_tmpls:
            print_warn("No degradation templates.")
            return
        name = ask_select(
            "Select template to remove:",
            [Choice(n, value=n) for n in sorted(deg_tmpls.keys())] + [Separator(), Choice("← Cancel", value=None)],
        )
        if not name:
            return
        ok = ask_confirm(f"Remove '{name}'?", default=False)
        if not ok:
            print_info("Cancelled.")
            return
        del deg_tmpls[name]
        self.templates_modified = True
        print_success(f"Removed degradation template '{name}'")

    def _ensure_source_dirs(self):
        self.config.setdefault("source_dirs", [])
        return self.config["source_dirs"]

    def list_source_dirs(self):
        source_dirs = self._ensure_source_dirs()
        if not source_dirs:
            print_warn("No source directories configured.")
            return
        t = make_table("#", "Path", "Extensions")
        for i, entry in enumerate(source_dirs):
            t.add_row(str(i), entry.get("path", "?"), ", ".join(entry.get("extensions", [])))
        console.print(t)

    def add_source_dir(self):
        source_dirs = self._ensure_source_dirs()
        path = ask_text("Directory path:", validate=lambda v: True if v.strip() else "Path required")
        if path is None:
            return
        path = path.strip()
        if not path:
            print_error("Path cannot be empty")
            return
        if any(d.get("path") == path for d in source_dirs):
            print_error(f"Already configured: {path}")
            return
        exts_str = ask_text("Extensions (comma-separated):", default=".mkv,.mp4,.avi")
        if exts_str is None:
            return
        exts = [e.strip() for e in exts_str.split(",") if e.strip()]
        source_dirs.append({"path": path, "extensions": exts})
        self.modified = True
        print_success(f"Added: {path}")

    def edit_source_dir(self):
        source_dirs = self._ensure_source_dirs()
        if not source_dirs:
            print_warn("No source directories configured.")
            return
        choices = [
            Choice(f"[{i}] {e.get('path','?')}", value=i)
            for i, e in enumerate(source_dirs)
        ] + [Separator(), Choice("← Cancel", value=None)]
        idx = ask_select("Select source directory to edit:", choices)
        if idx is None:
            return
        entry = source_dirs[idx]
        new_path = ask_text("New path:", default=entry["path"])
        if new_path is not None and new_path.strip():
            entry["path"] = new_path.strip()
        cur_exts = ", ".join(entry.get("extensions", []))
        new_exts = ask_text("Extensions:", default=cur_exts)
        if new_exts is not None and new_exts.strip():
            entry["extensions"] = [e.strip() for e in new_exts.split(",") if e.strip()]
        self.modified = True
        print_success(f"Updated source directory #{idx}")

    def remove_source_dir(self):
        source_dirs = self._ensure_source_dirs()
        if not source_dirs:
            print_warn("No source directories configured.")
            return
        choices = [
            Choice(f"[{i}] {e.get('path','?')}", value=i)
            for i, e in enumerate(source_dirs)
        ] + [Separator(), Choice("← Cancel", value=None)]
        idx = ask_select("Select source directory to remove:", choices)
        if idx is None:
            return
        path = source_dirs[idx].get("path", "?")
        ok = ask_confirm(f"Remove '{path}'?", default=False)
        if not ok:
            print_info("Cancelled.")
            return
        source_dirs.pop(idx)
        self.modified = True
        print_success(f"Removed: {path}")

    def rescan_file_list(self):
        source_dirs = self._ensure_source_dirs()
        if not source_dirs:
            print_error("No source directories configured. Add one first.")
            return
        if self.modified:
            ok = ask_confirm("Unsaved changes. Save before rescanning?")
            if ok:
                self.save(backup=False)
        existing_by_path = {v.get("path", ""): v for v in self.config.get("videos", [])}
        found_paths: List[str] = []
        seen_paths: set = set()
        for dir_cfg in source_dirs:
            video_dir = dir_cfg.get("path", "")
            extensions = dir_cfg.get("extensions", [".mkv", ".mp4", ".avi"])
            if not os.path.exists(video_dir):
                print_warn(f"Not found (skipped): {video_dir}")
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
        print_success(f"Rescan complete: {len(new_videos)} videos ({kept} kept, {added} newly added)")
        if added:
            print_info("Use 'Assign to category' to categorise the new videos.")

    def show_validation_report(self):
        print_rule("Validation Report")
        t_errors = validate_templates(self.templates)
        c_errors = validate_active_config(self.config, self.templates)
        all_errors = t_errors + c_errors
        if not all_errors:
            print_success("No validation errors found.")
        else:
            print_error(f"{len(all_errors)} error(s) found:")
            for e in all_errors:
                console.print(f"  [red]•[/] {e}")




def get_categories_interactive(manager, current_categories=None):
    """Select categories via questionary checkbox, falling back to simple input."""
    cats = sorted(manager.categories.keys())
    if not cats:
        print_error("No categories configured.")
        return None
    try:
        from questionary import Choice as _C, Checkbox as _CB  # noqa: F401
        choices = [
            _C(c, checked=(current_categories and c in current_categories))
            for c in cats
        ]
        result = ask_checkbox("Select categories:", choices)
        return result  # None on cancel, list otherwise
    except Exception:
        return _get_categories_simple(manager, current_categories)


def _get_categories_simple(manager, current_categories=None):
    available = sorted(manager.categories.keys())
    print_info(f"Available: {', '.join(available)}")
    if current_categories:
        print_info(f"Current: {', '.join(current_categories)}")
    while True:
        raw = ask_text("Categories (comma-separated, or 'none' to clear):")
        if raw is None:
            return None
        if raw.lower() == "none":
            return []
        cats = [c.strip() for c in raw.split(",") if c.strip()]
        invalid = [c for c in cats if c not in manager.categories]
        if invalid:
            print_error(f"Unknown: {', '.join(invalid)}")
            continue
        return cats


# ── Main menu ─────────────────────────────────────────────────────────────────

def _build_main_menu(manager) -> list:
    m = manager
    n_vids = len(m.videos)
    n_unassigned = sum(1 for v in m.videos if not v.get("categories"))
    n_srcs = len(m.config.get("source_dirs", []))
    fmt_count = len(m.templates.get("format_templates", {}))
    deg_count  = len(m.templates.get("degradation_templates", {}))
    cfg_tag = "  [yellow bold]⚡ unsaved[/]" if m.modified else ""
    tpl_tag = "  [yellow bold]⚡ unsaved[/]" if m.templates_modified else ""

    return [
        Choice(f"📹  Videos  ({n_vids} total, {n_unassigned} unassigned)",  value="videos"),
        Choice(f"🗂️  Categories & Formats  ({len(m.categories)} categories)", value="categories"),
        Choice(f"📁  Source Directories  ({n_srcs} dirs)",                   value="sources"),
        Choice(f"🎨  Templates  ({fmt_count} format, {deg_count} degradation)", value="templates"),
        Choice("⚙️  Config & Validation",                                    value="config"),
        Separator(),
        Choice(f"💾  Save config{cfg_tag}",     value="save_config"),
        Choice(f"📋  Save templates{tpl_tag}",  value="save_templates"),
        Separator(),
        Choice("🚪  Quit",                       value="quit"),
    ]


def main():
    from pathlib import Path as _Path
    script_dir = _Path(__file__).parent
    active_config_path = script_dir / "generator_config_v2.active.json"
    templates_path = script_dir / "templates.json"

    if not active_config_path.exists():
        print_warn("No active config found – creating default…")
        save_active_config(create_default_active_config(), str(active_config_path))
        print_success(f"Created {active_config_path.name}")
        print_info("→ Adjust root_path and source_dirs, then restart.")

    ensure_templates_file(str(templates_path))

    console.print(f"[dim]config   : {active_config_path.name}[/]")
    console.print(f"[dim]templates: {templates_path.name}[/]")

    manager = VideoManager(str(active_config_path), str(templates_path))
    try:
        manager.load()
    except Exception as e:
        print_error(f"Error loading config: {e}")
        traceback.print_exc()
        sys.exit(1)

    while True:
        console.print()
        print_banner(
            videos=len(manager.videos),
            categories=len(manager.categories),
            unsaved_cfg=manager.modified,
            unsaved_tpl=manager.templates_modified,
        )
        try:
            action = ask_select("", _build_main_menu(manager))
        except KeyboardInterrupt:
            action = None

        if action is None or action == "quit":
            if manager.modified or manager.templates_modified:
                save = ask_confirm("Save changes before quitting?")
                if save:
                    if manager.modified:
                        manager.save()
                    if manager.templates_modified:
                        manager.save_templates()
            console.print("[dim]Goodbye![/]")
            break

        try:
            if action == "videos":
                _menu_videos(manager)

            elif action == "categories":
                manager.manage_categories()

            elif action == "sources":
                _menu_sources(manager)

            elif action == "templates":
                manager.manage_templates()

            elif action == "config":
                _menu_config(manager)

            elif action == "save_config":
                if manager.modified:
                    manager.save()
                else:
                    print_info("No config changes to save.")

            elif action == "save_templates":
                if manager.templates_modified:
                    manager.save_templates()
                else:
                    print_info("No template changes to save.")

        except KeyboardInterrupt:
            print_warn("Interrupted – back to main menu.")
        except Exception as e:
            print_error(f"Error: {e}")
            traceback.print_exc()
            console.print("[dim]Continuing…[/]")


# ── Section sub-menus ─────────────────────────────────────────────────────────

def _menu_videos(manager):
    while True:
        action = ask_select(
            "Videos",
            [
                Choice("List all videos",           value="list_all"),
                Choice("List by category",           value="list_by_cat"),
                Choice("List unassigned",            value="list_unassigned"),
                Choice("Search by name",             value="search"),
                Separator(),
                Choice("Assign to category",         value="assign"),
                Choice("Assign by pattern",          value="assign_pattern"),
                Choice("Interactive multi-select",   value="interactive"),
                Choice("Remove from category",       value="remove_from_cat"),
                Separator(),
                Choice("Set forced frames",          value="forced"),
                Choice("Reset all assignments",      value="reset"),
                Separator(),
                Choice("← Back",                    value="back"),
            ],
        )
        if action is None or action == "back":
            break

        elif action == "list_all":
            show_all = ask_confirm("Show all videos? (No = first 20)", default=False)
            manager.print_video_list(manager.list_videos(), max_display=None if show_all else 20)

        elif action == "list_by_cat":
            cat = ask_select(
                "Select category:",
                [Choice(c, value=c) for c in sorted(manager.categories.keys())] + [Separator(), Choice("← Cancel", value=None)],
            )
            if cat:
                manager.print_video_list(manager.list_videos(category=cat), max_display=None)

        elif action == "list_unassigned":
            manager.print_video_list(manager.list_videos(show_unassigned=True), max_display=None)

        elif action == "search":
            pattern = ask_text("Search pattern (text or regex):")
            if pattern and pattern.strip():
                manager.print_video_list(manager.list_videos(filter_pattern=pattern.strip()), max_display=None)

        elif action == "assign":
            filter_str = ask_text("Optional filter (leave empty for all):", default="")
            videos = manager.list_videos(filter_pattern=filter_str.strip() or None)
            if not videos:
                print_warn("No videos found.")
                continue
            sorted_vids = _sorted_videos(videos)
            try:
                selected = select_items(
                    items=[v for _, v in sorted_vids],
                    title="Select videos  Space=toggle  Enter=confirm  Esc=cancel",
                    get_label=lambda v: _short_path(v.get("path", ""), depth=3),
                    get_details=lambda v: format_categories_display(v.get("categories", [])),
                )
            except Exception as e:
                print_warn(f"Curses UI failed: {e}")
                continue
            if selected is None:
                print_info("Cancelled.")
                continue
            video_indices = [sorted_vids[i][0] for i in selected]
            print_success(f"Selected {len(video_indices)} videos")
            categories = get_categories_interactive(manager)
            if categories is None:
                print_info("Cancelled.")
                continue
            manager.assign_videos(video_indices, categories)

        elif action == "assign_pattern":
            pattern = ask_text("Pattern (text or regex):")
            if not pattern or not pattern.strip():
                continue
            pat = pattern.strip()
            use_simple = "*" in pat or not any(c in pat for c in r"\.[](){}^$+?|")
            videos = manager.list_videos(filter_pattern=pat, use_simple_search=use_simple)
            if not videos:
                print_warn(f"No matches for: {pat}")
                continue
            manager.print_video_list(videos)
            ok = ask_confirm(f"Assign all {len(videos)} matching videos?")
            if not ok:
                continue
            categories = get_categories_interactive(manager)
            if categories:
                manager.assign_videos([i for i, _ in videos], categories)

        elif action == "interactive":
            filter_str = ask_text("Optional filter:", default="")
            selected_ids = manager.interactive_select_videos(filter_str.strip() or None)
            if selected_ids:
                print_success(f"Selected {len(selected_ids)} videos")
                categories = get_categories_interactive(manager)
                if categories:
                    manager.assign_videos(selected_ids, categories)
            else:
                print_info("Selection cancelled.")

        elif action == "remove_from_cat":
            cat = ask_select(
                "Remove from category:",
                [Choice(c, value=c) for c in sorted(manager.categories.keys())] + [Separator(), Choice("← Cancel", value=None)],
            )
            if not cat:
                continue
            ids_sel = ask_select(
                "Which videos?",
                [Choice("All videos in this category", value="all"), Choice("Enter IDs manually", value="ids")],
            )
            if ids_sel == "all":
                ids = list(range(len(manager.videos)))
            elif ids_sel == "ids":
                raw = ask_text("Video IDs (comma-separated):")
                if raw is None:
                    continue
                try:
                    ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
                except ValueError:
                    print_error("Invalid IDs")
                    continue
            else:
                continue
            manager.remove_from_category(ids, cat)

        elif action == "forced":
            raw = ask_text("Video ID(s) (comma-separated):")
            if raw is None:
                continue
            try:
                ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
            except ValueError:
                print_error("Invalid IDs")
                continue
            manager.set_forced_frames(ids)

        elif action == "reset":
            manager.reset_all()


def _menu_sources(manager):
    while True:
        action = ask_select(
            "Source Directories",
            [
                Choice("List source dirs",   value="list"),
                Choice("Add source dir",     value="add"),
                Choice("Edit source dir",    value="edit"),
                Choice("Remove source dir",  value="remove"),
                Separator(),
                Choice("Rescan file list",   value="rescan"),
                Separator(),
                Choice("← Back",            value="back"),
            ],
        )
        if action is None or action == "back":
            break
        elif action == "list":
            manager.list_source_dirs()
        elif action == "add":
            manager.add_source_dir()
        elif action == "edit":
            manager.edit_source_dir()
        elif action == "remove":
            manager.remove_source_dir()
        elif action == "rescan":
            manager.rescan_file_list()


def _menu_config(manager):
    while True:
        action = ask_select(
            "Config & Validation",
            [
                Choice("Show statistics",           value="stats"),
                Choice("Validation report",         value="validate"),
                Separator(),
                Choice("Manage category formats",   value="cat_formats"),
                Separator(),
                Choice("Create new config file",    value="new_cfg"),
                Separator(),
                Choice("← Back",                   value="back"),
            ],
        )
        if action is None or action == "back":
            break
        elif action == "stats":
            manager.show_statistics()
        elif action == "validate":
            manager.show_validation_report()
        elif action == "cat_formats":
            manager.manage_category_formats()
        elif action == "new_cfg":
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"generator_config_new_{ts}.active.json"
            name = ask_text("Output filename:", default=default_name)
            if name is None:
                continue
            from pathlib import Path as _P
            out_path = str(_P(manager.config_path).parent / (name.strip() or default_name))
            if os.path.exists(out_path):
                ok = ask_confirm(f"'{out_path}' exists. Overwrite?", default=False)
                if not ok:
                    print_info("Cancelled.")
                    continue
            save_active_config(create_default_active_config(), out_path)
            print_success(f"Created {out_path}")
            print_info("→ Adjust root_path and source_dirs, then reload.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/]")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
