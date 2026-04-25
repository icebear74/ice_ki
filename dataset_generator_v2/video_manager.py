#!/usr/bin/env python3
"""
VIDEO CATEGORY MANAGER  (v2 – new config model)
Curses full-screen TUI: bordered menus, popup dialogs, category detail view.
"""

import curses
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import tui

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


# ── Utilities ──────────────────────────────────────────────────────────────────

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


# ── VideoManager (data + logic, no UI) ────────────────────────────────────────

class VideoManager:
    """Central data model for videos, categories, formats and templates."""

    def __init__(self, config_path: str, templates_path: str = None):
        self.config_path = config_path
        script_dir = Path(config_path).parent
        self.templates_path = templates_path or str(script_dir / "templates.json")
        self.config:    dict = {}
        self.templates: dict = {}
        self.videos:    List[dict] = []
        self.categories: dict = {}
        self.modified           = False
        self.templates_modified = False

    def load(self):
        self.templates = ensure_templates_file(self.templates_path)
        self.config    = load_active_config(self.config_path)
        self.videos    = self.config.get("videos", [])
        self.videos.sort(key=lambda v: v.get("name", "").lower())
        self.categories = self.config.get("categories", {})

    def save(self, backup: bool = True):
        def _key(v):
            cats = get_video_categories(v)
            return (-len(cats) if cats else 999, cats[0] if cats else "zzz", v.get("name", "").lower())
        self.videos.sort(key=_key)
        self.config["videos"]     = self.videos
        self.config["categories"] = self.categories
        save_active_config(self.config, self.config_path)
        self.modified = False

    def save_templates(self):
        if os.path.exists(self.templates_path):
            with open(self.templates_path, "r", encoding="utf-8") as f:
                old = f.read()
            with open(self.templates_path + ".backup", "w", encoding="utf-8") as f:
                f.write(old)
        _save_templates_io(self.templates, self.templates_path)
        self.templates_modified = False

    def list_videos(self, filter_pattern=None, category=None,
                    show_unassigned=False, use_simple_search=False):
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

    def assign_videos(self, video_indices, categories, mode="replace"):
        count = 0
        for idx in video_indices:
            if 0 <= idx < len(self.videos):
                if mode == "add":
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
        return count

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
        return count

    def reset_all_assignments(self):
        for video in self.videos:
            video["categories"] = []
        self.modified = True

    def set_forced_frames_for(self, video_idx: int, cat: str, value: Optional[int]):
        """value=None → keep, value=0 → remove override, value>0 → set."""
        video  = self.videos[video_idx]
        forced = dict(video.get("forced_frames", {}))
        if value is None:
            pass
        elif value == 0:
            forced.pop(cat, None)
        else:
            forced[cat] = value
        if forced:
            video["forced_frames"] = forced
        else:
            video.pop("forced_frames", None)
        self.modified = True

    def rescan(self):
        """Scan source_dirs and update videos list.  Returns (kept, added)."""
        source_dirs = self.config.get("source_dirs", [])
        existing_by_path = {v.get("path", ""): v for v in self.config.get("videos", [])}
        found_paths: List[str] = []
        seen: set = set()
        missing_dirs: List[str] = []
        for dir_cfg in source_dirs:
            video_dir  = dir_cfg.get("path", "")
            extensions = dir_cfg.get("extensions", [".mkv", ".mp4", ".avi"])
            if not os.path.exists(video_dir):
                missing_dirs.append(video_dir)
                continue
            exts_lower = {e.lower() for e in extensions}
            for p in Path(video_dir).rglob("*"):
                if p.is_file() and p.suffix.lower() in exts_lower:
                    ps = str(p)
                    if ps not in seen:
                        seen.add(ps)
                        found_paths.append(ps)
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
        return kept, added, missing_dirs

    def validation_report(self):
        """Return list of error strings (empty = OK)."""
        return validate_templates(self.templates) + validate_active_config(self.config, self.templates)


# ── Category detail computation ────────────────────────────────────────────────

def compute_category_detail(manager: VideoManager, cat_name: str) -> List[str]:
    """
    Build a detailed breakdown of what will be generated for a category.
    Returns a list of text lines suitable for tui.message_box().
    """
    cat_cfg    = manager.categories.get(cat_name, {})
    target     = cat_cfg.get("target_total", 0)
    fmt_tmpls  = manager.templates.get("format_templates", {})
    formats    = cat_cfg.get("formats", [])
    videos_in  = [v for v in manager.videos if cat_name in get_video_categories(v)]
    forced_map = {}
    for v in videos_in:
        for c, n in v.get("forced_frames", {}).items():
            if c == cat_name and n > 0:
                forced_map[v.get("name", "?")] = n

    total_weight = sum(f.get("weight", 0) for f in formats) or 1

    lines: List[str] = []
    lines.append(f"Category  : {cat_name}")
    lines.append(f"Target    : {target:,} scenes")
    lines.append(f"Videos    : {len(videos_in)}")
    if forced_map:
        forced_total = sum(forced_map.values())
        lines.append(f"Forced    : {forced_total:,} scenes across {len(forced_map)} video(s)")
    lines.append("")

    if not formats:
        lines.append("  (no format entries configured)")
        return lines

    lines.append("═" * 62)
    lines.append("  FORMAT BREAKDOWN")
    lines.append("═" * 62)

    for i, fmt in enumerate(formats):
        tmpl_name = fmt.get("template", "?")
        mode      = fmt.get("source_mode", "?")
        weight    = fmt.get("weight", 0)
        share     = weight / total_weight
        scenes    = round(target * share)

        spec    = fmt_tmpls.get(tmpl_name, {})
        gt_size = spec.get("gt_size", "?")
        lr_size = spec.get("lr_size", "?")
        gt_str  = f"{gt_size[0]}×{gt_size[1]}" if isinstance(gt_size, list) else str(gt_size)
        lr_str  = f"{lr_size[0]}×{lr_size[1]}" if isinstance(lr_size, list) else str(lr_size)

        lines.append("")
        lines.append(f"  [{i}] Template : {tmpl_name}")
        lines.append(f"      GT → LR  : {gt_str}  →  {lr_str}")
        lines.append(f"      Mode     : {mode}")
        lines.append(f"      Weight   : {weight}  /  share: {share*100:.1f}%")
        lines.append(f"      Scenes   : {scenes:,}")

        deg_mix = fmt.get("degradation_mix", {})
        if deg_mix:
            total_dw = sum(deg_mix.values()) or 1
            lines.append(f"      Degradation mix:")
            for dname, dw in sorted(deg_mix.items(), key=lambda x: -x[1]):
                dshare  = dw / total_dw
                dscenes = round(scenes * dshare)
                lines.append(
                    f"        {dname:<34} w={dw:>3}  {dshare*100:.1f}%  → {dscenes:,} scenes"
                )
        else:
            lines.append(f"      Degradation: (none configured)")

    lines.append("")
    lines.append("═" * 62)
    lines.append(f"  TOTAL : {target:,} scenes")
    lines.append(f"  GT/LR pairs per format are independent of degradation.")
    return lines


# ── Validation guidance ────────────────────────────────────────────────────────

def _guidance_for_error(err: str) -> str:
    """Return a short navigation hint for a validation error string."""
    e = err.lower()
    if "degradation_mix" in e and "non-empty" in e:
        return "→  Categories & Formats → Manage category formats → Manage degradation mix"
    if "degradation_mix" in e and "not found in degradation_templates" in e:
        return "→  Templates → Add / edit degradation template"
    if "degradation_mix" in e and "weight" in e:
        return "→  Categories & Formats → Manage degradation mix (set a positive weight)"
    if ".template:" in e and "not found" in e:
        return "→  Templates → Add format template  –or–  remove the format entry and re-add"
    if "target_total" in e:
        return "→  Categories & Formats → Edit target total (must be a positive integer)"
    if "source_mode" in e:
        return "→  Categories & Formats → Edit weight / source_mode"
    if "weight" in e:
        return "→  Categories & Formats → Edit weight / source_mode"
    return ""


# ── TUI Application ────────────────────────────────────────────────────────────

class _App:
    """Full-screen curses TUI for VideoManager."""

    def __init__(self, manager: VideoManager):
        self.m      = manager
        self.stdscr = None
        self._status     = ""
        self._status_err = False

    # ── helpers ────────────────────────────────────────────────────────────────

    def _bg(self) -> None:
        """Redraw the desktop background."""
        n  = len(self.m.videos)
        na = sum(1 for v in self.m.videos if not v.get("categories"))
        nc = len(self.m.categories)
        stats = f"videos:{n}  unassigned:{na}  categories:{nc}"
        if self.m.modified:
            stats += "  [config *]"
        if self.m.templates_modified:
            stats += "  [tpl *]"
        tui.draw_background(self.stdscr, stats=stats,
                            status=self._status, is_error=self._status_err)
        self._status     = ""
        self._status_err = False

    def _ok(self, msg: str) -> None:
        self._status     = "✓  " + msg
        self._status_err = False

    def _err(self, msg: str) -> None:
        tui.message_box(self.stdscr, [msg], "Error")
        self._status     = "✗  " + msg
        self._status_err = True

    def _warn(self, msg: str) -> None:
        tui.message_box(self.stdscr, [msg], "Warning")

    def _menu(self, title: str, items) -> Optional[str]:
        """items: list of (label, value) tuples."""
        return tui.menu_box(self.stdscr, title, items)

    def _confirm(self, msg: str, title: str = "Confirm", default: bool = True) -> bool:
        return tui.confirm_box(self.stdscr, msg, title, default)

    def _input(self, prompt: str, default: str = "", title: str = "Input") -> Optional[str]:
        return tui.input_box(self.stdscr, prompt, default, title)

    def _int(self, prompt: str, default: int = 0, min_val: int = 0,
             title: str = "Input") -> Optional[int]:
        return tui.int_box(self.stdscr, prompt, default, min_val, title)

    def _show(self, lines: List[str], title: str = "") -> None:
        tui.message_box(self.stdscr, lines, title)

    def _table(self, title: str, headers: List[str], rows: List[List[str]]) -> None:
        self._show(tui.text_table(headers, rows), title)

    def _checkbox(self, title: str, labels: List[str],
                  pre: Optional[List[int]] = None) -> Optional[List[int]]:
        return tui.checkbox_box(self.stdscr, title, labels, pre)

    # ── main run loop ──────────────────────────────────────────────────────────

    def run(self) -> None:
        curses.wrapper(self._main)

    def _main(self, stdscr) -> None:
        self.stdscr = stdscr
        tui.setup(stdscr)

        while True:
            self._bg()
            n_vids = len(self.m.videos)
            n_unas = sum(1 for v in self.m.videos if not v.get("categories"))
            n_srcs = len(self.m.config.get("source_dirs", []))
            fmt_c  = len(self.m.templates.get("format_templates", {}))
            deg_c  = len(self.m.templates.get("degradation_templates", {}))
            cfg_flag = " *" if self.m.modified else ""
            tpl_flag = " *" if self.m.templates_modified else ""

            items = [
                (f"Videos  ({n_vids} total, {n_unas} unassigned)",
                 "videos"),
                (f"Categories & Formats  ({len(self.m.categories)} categories)",
                 "categories"),
                (f"Source Directories  ({n_srcs} dirs)",
                 "sources"),
                (f"Templates  ({fmt_c} format, {deg_c} degradation)",
                 "templates"),
                ("Config & Validation",
                 "config"),
                ("───", None),
                (f"Save config{cfg_flag}",      "save_config"),
                (f"Save templates{tpl_flag}",   "save_templates"),
                ("───", None),
                ("Quit",                        "quit"),
            ]
            action = self._menu("ice_ki  Video Manager  v2", items)

            if action is None or action == "quit":
                do_quit = True
                if self.m.modified or self.m.templates_modified:
                    if self._confirm("Save changes before quitting?"):
                        if self.m.modified:
                            if self._pre_save_check():
                                self.m.save()
                            elif not self._confirm(
                                "Config not saved (errors found). Quit anyway?",
                                title="Quit",
                                default=False,
                            ):
                                do_quit = False
                        if do_quit and self.m.templates_modified:
                            self.m.save_templates()
                if do_quit:
                    break

            try:
                if action == "videos":
                    self._section_videos()
                elif action == "categories":
                    self._section_categories()
                elif action == "sources":
                    self._section_sources()
                elif action == "templates":
                    self._section_templates()
                elif action == "config":
                    self._section_config()
                elif action == "save_config":
                    if self.m.modified:
                        if self._pre_save_check():
                            self.m.save()
                            self._ok(f"Config saved → {Path(self.m.config_path).name}")
                    else:
                        self._ok("No config changes to save.")
                elif action == "save_templates":
                    if self.m.templates_modified:
                        self.m.save_templates()
                        self._ok(f"Templates saved → {Path(self.m.templates_path).name}")
                    else:
                        self._ok("No template changes to save.")
            except KeyboardInterrupt:
                self._status = "Interrupted"

    # ── VIDEOS section ─────────────────────────────────────────────────────────

    def _section_videos(self):
        while True:
            self._bg()
            action = self._menu("Videos", [
                ("List all videos",          "list_all"),
                ("List by category",          "list_by_cat"),
                ("List unassigned",           "list_unassigned"),
                ("Search by name",            "search"),
                ("───", None),
                ("Assign to category",        "assign"),
                ("Assign by pattern",         "assign_pattern"),
                ("Interactive multi-select",  "interactive"),
                ("Remove from category",      "remove_from_cat"),
                ("───", None),
                ("Set forced frames",         "forced"),
                ("Reset ALL assignments",     "reset"),
                ("───", None),
                ("← Back",                   "back"),
            ])
            if action is None or action == "back":
                break

            elif action == "list_all":
                vids = self.m.list_videos()
                if not vids:
                    self._warn("No videos found.")
                else:
                    rows = [[str(i),
                             _short_path(v.get("path", ""), depth=3)[:32],
                             v["name"][:36],
                             ", ".join(get_video_categories(v)) or "(unassigned)"]
                            for i, v in _sorted_videos(vids)]
                    self._table(f"All videos ({len(vids)})",
                                ["ID", "Path", "Name", "Categories"], rows)

            elif action == "list_by_cat":
                cats = sorted(self.m.categories.keys())
                if not cats:
                    self._warn("No categories configured.")
                    continue
                cat = self._menu("Select category",
                                 [(c, c) for c in cats] + [("───", None), ("← Cancel", None)])
                if not cat:
                    continue
                vids = self.m.list_videos(category=cat)
                rows = [[str(i), v["name"][:40], _short_path(v.get("path", ""), depth=3)[:34]]
                        for i, v in _sorted_videos(vids)]
                self._table(f"Category: {cat}  ({len(vids)} videos)",
                            ["ID", "Name", "Path"], rows)

            elif action == "list_unassigned":
                vids = self.m.list_videos(show_unassigned=True)
                rows = [[str(i), v["name"][:40], _short_path(v.get("path", ""), depth=3)[:34]]
                        for i, v in _sorted_videos(vids)]
                self._table(f"Unassigned videos ({len(vids)})",
                            ["ID", "Name", "Path"], rows)

            elif action == "search":
                pattern = self._input("Search (text or regex):", title="Search Videos")
                if not pattern or not pattern.strip():
                    continue
                vids = self.m.list_videos(filter_pattern=pattern.strip())
                if not vids:
                    self._warn(f"No matches for: {pattern.strip()}")
                    continue
                rows = [[str(i), v["name"][:40], ", ".join(get_video_categories(v)) or "(unassigned)"]
                        for i, v in _sorted_videos(vids)]
                self._table(f"Search: '{pattern.strip()}'  ({len(vids)} results)",
                            ["ID", "Name", "Categories"], rows)

            elif action == "assign":
                self._do_assign()

            elif action == "assign_pattern":
                self._do_assign_pattern()

            elif action == "interactive":
                self._do_interactive_select()

            elif action == "remove_from_cat":
                self._do_remove_from_cat()

            elif action == "forced":
                self._do_set_forced_frames()

            elif action == "reset":
                if self._confirm("Reset ALL video assignments?\nThis cannot be undone!", default=False):
                    self.m.reset_all_assignments()
                    self._ok(f"Reset {len(self.m.videos)} videos.")

    def _pick_categories(self, current: Optional[List[str]] = None) -> Optional[List[str]]:
        cats = sorted(self.m.categories.keys())
        if not cats:
            self._warn("No categories configured.")
            return None
        pre = [i for i, c in enumerate(cats) if current and c in current]
        sel = self._checkbox("Select categories", cats, pre)
        if sel is None:
            return None
        return [cats[i] for i in sel]

    def _do_assign(self):
        filter_str = self._input("Filter (leave blank for all):", title="Assign to Category")
        vids = self.m.list_videos(filter_pattern=filter_str.strip() if filter_str else None)
        if not vids:
            self._warn("No videos found.")
            return
        sorted_vids = _sorted_videos(vids)
        selected = select_items(
            items=[v for _, v in sorted_vids],
            title=f"Select videos  ({len(sorted_vids)} total)  Space=toggle  Enter=done  Esc=cancel",
            get_label=lambda v: _short_path(v.get("path", ""), depth=3),
            get_details=lambda v: format_categories_display(v.get("categories", [])),
            stdscr=self.stdscr,
        )
        if selected is None:
            return
        video_indices = [sorted_vids[i][0] for i in selected]
        cats = self._pick_categories()
        if cats is None:
            return
        has_existing = any(bool(get_video_categories(self.m.videos[i])) for i in video_indices)
        mode = "replace"
        if has_existing:
            m = self._menu("Some videos already have categories",
                           [("Add to existing categories", "add"),
                            ("Replace all categories",     "replace")])
            if m is None:
                return
            mode = m
        count = self.m.assign_videos(video_indices, cats, mode=mode)
        self._ok(f"{mode.title()} – {count} video(s) → {cats}")

    def _do_assign_pattern(self):
        pattern = self._input("Pattern (text or regex):", title="Assign by Pattern")
        if not pattern or not pattern.strip():
            return
        pat       = pattern.strip()
        simple    = "*" in pat or not any(c in pat for c in r"\.[](){}^$+?|")
        vids      = self.m.list_videos(filter_pattern=pat, use_simple_search=simple)
        if not vids:
            self._warn(f"No matches for: {pat}")
            return
        rows = [[str(i), v["name"][:40], ", ".join(get_video_categories(v)) or "(unassigned)"]
                for i, v in _sorted_videos(vids)]
        self._table(f"Pattern matches ({len(vids)})", ["ID", "Name", "Categories"], rows)
        if not self._confirm(f"Assign all {len(vids)} matching videos?"):
            return
        cats = self._pick_categories()
        if cats:
            count = self.m.assign_videos([i for i, _ in vids], cats)
            self._ok(f"Assigned {count} video(s) → {cats}")

    def _do_interactive_select(self):
        filter_str = self._input("Filter (leave blank for all):", title="Interactive Select")
        vids = (self.m.list_videos(filter_pattern=filter_str.strip(), use_simple_search=True)
                if filter_str and filter_str.strip() else self.m.list_videos())
        if not vids:
            self._warn("No videos found.")
            return
        sorted_vids = _sorted_videos(vids)
        selected = select_items(
            items=[v for _, v in sorted_vids],
            title=f"Interactive select  Space=toggle  Enter=done  Esc=cancel",
            get_label=lambda v: _short_path(v.get("path", ""), depth=3),
            get_details=lambda v: format_categories_display(v.get("categories", [])),
            stdscr=self.stdscr,
        )
        if selected is None:
            return
        video_indices = [sorted_vids[i][0] for i in selected]
        if not video_indices:
            return
        cats = self._pick_categories()
        if cats:
            count = self.m.assign_videos(video_indices, cats)
            self._ok(f"Assigned {count} video(s) → {cats}")

    def _do_remove_from_cat(self):
        cats = sorted(self.m.categories.keys())
        if not cats:
            self._warn("No categories configured.")
            return
        cat = self._menu("Remove from category",
                         [(c, c) for c in cats] + [("───", None), ("← Cancel", None)])
        if not cat:
            return
        which = self._menu("Which videos?",
                           [("All videos in this category", "all"),
                            ("Enter IDs manually",          "ids"),
                            ("← Cancel",                   None)])
        if which == "all":
            ids = list(range(len(self.m.videos)))
        elif which == "ids":
            raw = self._input("Video IDs (comma-separated):", title="Enter IDs")
            if not raw:
                return
            try:
                ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
            except ValueError:
                self._err("Invalid IDs")
                return
        else:
            return
        count = self.m.remove_from_category(ids, cat)
        self._ok(f"Removed {count} video(s) from '{cat}'")

    def _do_set_forced_frames(self):
        raw = self._input("Video ID(s) (comma-separated):", title="Set Forced Frames")
        if not raw:
            return
        try:
            ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            self._err("Invalid IDs")
            return
        valid = [i for i in ids if 0 <= i < len(self.m.videos)]
        if not valid:
            self._err("No valid video indices.")
            return
        all_cats: List[str] = []
        for idx in valid:
            for cat in get_video_categories(self.m.videos[idx]):
                if cat not in all_cats:
                    all_cats.append(cat)
        if not all_cats:
            self._err("Selected videos have no categories assigned.")
            return
        for cat in all_cats:
            cur_values = [self.m.videos[i].get("forced_frames", {}).get(cat, 0) for i in valid]
            current_str = f"{cur_values[0]:,}" if len(set(cur_values)) == 1 else "mixed"
            val = self._int(
                f"Forced frames for '{cat}' (current: {current_str})\n"
                f"  0 = auto/remove override",
                default=cur_values[0] if len(set(cur_values)) == 1 else 0,
                min_val=0,
                title="Forced Frames",
            )
            if val is not None:
                for idx in valid:
                    self.m.set_forced_frames_for(idx, cat, val if val > 0 else 0)
        self._ok(f"Forced frames updated for {len(valid)} video(s)")

    # ── CATEGORIES section ─────────────────────────────────────────────────────

    def _section_categories(self):
        while True:
            self._bg()
            action = self._menu("Categories & Formats", [
                ("List categories",            "list"),
                ("Add category",               "add"),
                ("Remove category",            "remove"),
                ("Edit target total",          "edit_target"),
                ("───", None),
                ("Manage category formats",    "formats"),
                ("View category detail",       "detail"),
                ("───", None),
                ("← Back",                    "back"),
            ])
            if action is None or action == "back":
                break
            elif action == "list":
                self._list_categories()
            elif action == "add":
                self._add_category()
            elif action == "remove":
                self._remove_category()
            elif action == "edit_target":
                self._edit_category_target()
            elif action == "formats":
                cat = self._pick_cat("Select category to manage formats")
                if cat:
                    self._manage_category_formats(cat)
            elif action == "detail":
                cat = self._pick_cat("Select category for detail view")
                if cat:
                    lines = compute_category_detail(self.m, cat)
                    self._show(lines, f"Detail: {cat}")

    def _pick_cat(self, prompt: str) -> Optional[str]:
        cats = sorted(self.m.categories.keys())
        if not cats:
            self._warn("No categories configured.")
            return None
        return self._menu(prompt,
                          [(c, c) for c in cats] + [("───", None), ("← Cancel", None)])

    def _list_categories(self):
        if not self.m.categories:
            self._warn("No categories configured.")
            return
        rows = []
        for cat in sorted(self.m.categories.keys()):
            cfg     = self.m.categories[cat]
            target  = cfg.get("target_total", "?")
            tstr    = f"{target:,}" if isinstance(target, int) else str(target)
            n_vids  = sum(1 for v in self.m.videos if cat in get_video_categories(v))
            n_fmts  = len(cfg.get("formats", []))
            rows.append([cat, tstr, str(n_vids), str(n_fmts)])
        self._table("Categories", ["Category", "Target", "Videos", "Formats"], rows)

    def _add_category(self):
        name = self._input("New category name:", title="Add Category")
        if not name or not name.strip():
            return
        name = name.strip().lower()
        if name in self.m.categories:
            self._err(f"Category '{name}' already exists")
            return
        target = self._int("Target total:", default=50000, min_val=1, title="Add Category")
        if target is None:
            return
        self.m.categories[name] = {"target_total": target, "formats": []}
        self.m.config["categories"] = self.m.categories
        self.m.modified = True
        self._ok(f"Added '{name}'  target: {target:,}")

    def _remove_category(self):
        cat = self._pick_cat("Select category to remove")
        if not cat:
            return
        affected = sum(1 for v in self.m.videos if cat in get_video_categories(v))
        suffix   = f"\n\nThis will unassign {affected} video(s)." if affected else ""
        if not self._confirm(f"Remove category '{cat}'?{suffix}", default=False):
            return
        for video in self.m.videos:
            cats = get_video_categories(video)
            if cat in cats:
                cats.remove(cat)
                video["categories"] = cats
        del self.m.categories[cat]
        self.m.config["categories"] = self.m.categories
        self.m.modified = True
        msg = f"Removed '{cat}'"
        if affected:
            msg += f", unassigned {affected} video(s)"
        self._ok(msg)

    def _edit_category_target(self):
        cat = self._pick_cat("Select category to edit")
        if not cat:
            return
        current    = self.m.categories[cat].get("target_total", 0)
        new_target = self._int(f"New target_total  (current: {current:,}):",
                               default=current, min_val=1, title=f"Edit: {cat}")
        if new_target is None:
            return
        self.m.categories[cat]["target_total"] = new_target
        self.m.modified = True
        self._ok(f"'{cat}' target: {current:,} → {new_target:,}")

    def _manage_category_formats(self, cat_name: str):
        while True:
            self._bg()
            cat_cfg = self.m.categories[cat_name]
            formats = cat_cfg.get("formats", [])
            total_w = sum(f.get("weight", 0) for f in formats) or 1
            fmt_rows = [
                [str(i),
                 f.get("template", "?"),
                 f.get("source_mode", "?"),
                 str(f.get("weight", 0)),
                 f"{f.get('weight',0)/total_w*100:.1f}%",
                 ", ".join(f"{k}={v}" for k, v in f.get("degradation_mix", {}).items())
                 or "⚠ empty"]
                for i, f in enumerate(formats)
            ]

            action = self._menu(f"Formats for '{cat_name}'", [
                ("Show format table",             "show"),
                ("Add format entry",              "add"),
                ("Remove format entry",           "remove"),
                ("Edit weight / source_mode",     "edit"),
                ("Manage degradation mix",        "deg"),
                ("───", None),
                ("← Back",                       "back"),
            ])
            if action is None or action == "back":
                break
            elif action == "show":
                if not formats:
                    self._warn("No format entries yet.")
                else:
                    self._table(f"Formats: {cat_name}",
                                ["#", "Template", "Mode", "Weight", "Share", "Deg mix"],
                                fmt_rows)
            elif action == "add":
                self._add_format_entry(cat_name)
            elif action == "remove":
                self._remove_format_entry(cat_name)
            elif action == "edit":
                self._edit_format_entry(cat_name)
            elif action == "deg":
                self._manage_degradation_mix(cat_name)

    def _add_format_entry(self, cat_name: str):
        fmt_tmpls = self.m.templates.get("format_templates", {})
        if not fmt_tmpls:
            self._err("No format_templates in templates.json")
            return
        tmpl = self._menu("Select format template",
                          [(n, n) for n in sorted(fmt_tmpls.keys())] +
                          [("───", None), ("← Cancel", None)])
        if not tmpl:
            return
        weight = self._int("Weight (e.g. 50):", default=50, min_val=1,
                           title="Add Format Entry")
        if weight is None:
            return
        mode = self._menu("source_mode",
                          [(m, m) for m in sorted(VALID_SOURCE_MODES)])
        if mode is None:
            return
        self.m.categories[cat_name].setdefault("formats", []).append(
            {"template": tmpl, "weight": weight, "source_mode": mode, "degradation_mix": {}}
        )
        self.m.modified = True
        idx = len(self.m.categories[cat_name]["formats"]) - 1
        self._ok(f"Added [{idx}]: {tmpl} / {mode} / weight={weight}")
        # Immediately open the degradation mix editor so the new entry is never left empty.
        self._show(
            [
                f"Format [{idx}] was added with an empty degradation mix.",
                "",
                "At least one degradation template must be assigned",
                "before the generator will accept this config.",
                "",
                "Opening the degradation mix editor now…",
            ],
            "⚠  Degradation Mix Required",
        )
        self._manage_degradation_mix(cat_name, preselect_idx=idx)

    def _remove_format_entry(self, cat_name: str):
        formats = self.m.categories[cat_name].get("formats", [])
        if not formats:
            self._warn("No format entries.")
            return
        items = [(f"[{i}] {f.get('template','?')} / {f.get('source_mode','?')} / w={f.get('weight',0)}", i)
                 for i, f in enumerate(formats)] + [("───", None), ("← Cancel", None)]
        idx = self._menu("Select entry to remove", items)
        if idx is None:
            return
        if not self._confirm(f"Remove [{idx}] '{formats[idx].get('template','?')}'?", default=False):
            return
        removed = formats.pop(idx)
        self.m.modified = True
        self._ok(f"Removed [{idx}]: {removed.get('template', '?')}")

    def _edit_format_entry(self, cat_name: str):
        formats = self.m.categories[cat_name].get("formats", [])
        if not formats:
            self._warn("No format entries.")
            return
        items = [(f"[{i}] {f.get('template','?')} / {f.get('source_mode','?')} / w={f.get('weight',0)}", i)
                 for i, f in enumerate(formats)] + [("───", None), ("← Cancel", None)]
        idx = self._menu("Select entry to edit", items)
        if idx is None:
            return
        entry  = formats[idx]
        new_w  = self._int(f"New weight (current: {entry.get('weight')}):",
                           default=entry.get("weight", 1), min_val=1,
                           title="Edit Format Entry")
        if new_w is not None:
            entry["weight"] = new_w
            self.m.modified = True
        new_mode = self._menu(f"New source_mode  (current: {entry.get('source_mode')})",
                              [(m, m) for m in sorted(VALID_SOURCE_MODES)] +
                              [("───", None), ("Keep current", None)])
        if new_mode:
            entry["source_mode"] = new_mode
            self.m.modified = True
        self._ok(f"Format entry [{idx}] updated")

    def _manage_degradation_mix(self, cat_name: str, preselect_idx: Optional[int] = None):
        formats = self.m.categories[cat_name].get("formats", [])
        if not formats:
            self._warn("No format entries.")
            return
        if preselect_idx is not None and 0 <= preselect_idx < len(formats):
            idx = preselect_idx
        else:
            items = [
                (f"[{i}] {f.get('template','?')} / {f.get('source_mode','?')}"
                 + ("  ⚠ empty mix" if not f.get("degradation_mix") else ""), i)
                for i, f in enumerate(formats)
            ] + [("───", None), ("← Cancel", None)]
            idx = self._menu("Select format entry to edit degradation mix", items)
            if idx is None:
                return
        deg_tmpls = self.m.templates.get("degradation_templates", {})
        entry     = formats[idx]
        tmpl_name = entry.get("template", "?")

        while True:
            self._bg()
            mix     = entry.setdefault("degradation_mix", {})
            total_w = sum(mix.values()) or 1

            action = self._menu(f"Degradation mix: [{idx}] {tmpl_name}", [
                ("Show current mix",   "show"),
                ("Add / update entry", "add"),
                ("Remove entry",       "remove"),
                ("───", None),
                ("← Back",            "back"),
            ])
            if action is None or action == "back":
                break
            elif action == "show":
                if not mix:
                    self._warn("Mix is empty.")
                else:
                    rows = [[dn, str(dw), f"{dw/total_w*100:.1f}%"]
                            for dn, dw in sorted(mix.items(), key=lambda x: -x[1])]
                    self._table(f"Deg mix: {tmpl_name}",
                                ["Template", "Weight", "Share"], rows)
            elif action == "add":
                if not deg_tmpls:
                    self._err("No degradation templates defined.")
                    continue
                dname = self._menu("Select degradation template",
                                   [(n, n) for n in sorted(deg_tmpls.keys())] +
                                   [("───", None), ("← Cancel", None)])
                if not dname:
                    continue
                dw = self._int(f"Weight (current: {mix.get(dname, 0)}):",
                               default=mix.get(dname, 50), min_val=1,
                               title="Degradation Weight")
                if dw is None:
                    continue
                mix[dname] = dw
                self.m.modified = True
                self._ok(f"Set {dname} = {dw}")
            elif action == "remove":
                if not mix:
                    self._warn("Mix is already empty.")
                    continue
                dname = self._menu("Remove which entry",
                                   [(n, n) for n in sorted(mix.keys())] +
                                   [("───", None), ("← Cancel", None)])
                if dname and dname in mix:
                    del mix[dname]
                    self.m.modified = True
                    self._ok(f"Removed {dname}")

    # ── SOURCES section ────────────────────────────────────────────────────────

    def _section_sources(self):
        while True:
            self._bg()
            action = self._menu("Source Directories", [
                ("List source dirs",  "list"),
                ("Add source dir",    "add"),
                ("Edit source dir",   "edit"),
                ("Remove source dir", "remove"),
                ("───", None),
                ("Rescan file list",  "rescan"),
                ("───", None),
                ("← Back",           "back"),
            ])
            if action is None or action == "back":
                break
            elif action == "list":
                self._list_source_dirs()
            elif action == "add":
                self._add_source_dir()
            elif action == "edit":
                self._edit_source_dir()
            elif action == "remove":
                self._remove_source_dir()
            elif action == "rescan":
                self._rescan()

    def _ensure_source_dirs(self):
        self.m.config.setdefault("source_dirs", [])
        return self.m.config["source_dirs"]

    def _list_source_dirs(self):
        sds = self._ensure_source_dirs()
        if not sds:
            self._warn("No source directories configured.")
            return
        rows = [[str(i), e.get("path", "?"), ", ".join(e.get("extensions", []))]
                for i, e in enumerate(sds)]
        self._table("Source Directories", ["#", "Path", "Extensions"], rows)

    def _add_source_dir(self):
        sds = self._ensure_source_dirs()
        path = self._input("Directory path:", title="Add Source Directory")
        if not path or not path.strip():
            return
        path = path.strip()
        if any(d.get("path") == path for d in sds):
            self._err(f"Already configured: {path}")
            return
        exts_raw = self._input("Extensions (comma-separated):",
                               default=".mkv,.mp4,.avi", title="Extensions")
        if exts_raw is None:
            return
        exts = [e.strip() for e in exts_raw.split(",") if e.strip()]
        sds.append({"path": path, "extensions": exts})
        self.m.modified = True
        self._ok(f"Added: {path}")

    def _edit_source_dir(self):
        sds = self._ensure_source_dirs()
        if not sds:
            self._warn("No source directories configured.")
            return
        items = [(f"[{i}] {e.get('path','?')}", i) for i, e in enumerate(sds)] + \
                [("───", None), ("← Cancel", None)]
        idx = self._menu("Select source directory to edit", items)
        if idx is None:
            return
        entry    = sds[idx]
        new_path = self._input("New path:", default=entry["path"], title="Edit Path")
        if new_path and new_path.strip():
            entry["path"] = new_path.strip()
        cur_exts = ", ".join(entry.get("extensions", []))
        new_exts = self._input("Extensions:", default=cur_exts, title="Edit Extensions")
        if new_exts and new_exts.strip():
            entry["extensions"] = [e.strip() for e in new_exts.split(",") if e.strip()]
        self.m.modified = True
        self._ok(f"Updated source directory #{idx}")

    def _remove_source_dir(self):
        sds = self._ensure_source_dirs()
        if not sds:
            self._warn("No source directories configured.")
            return
        items = [(f"[{i}] {e.get('path','?')}", i) for i, e in enumerate(sds)] + \
                [("───", None), ("← Cancel", None)]
        idx = self._menu("Select source directory to remove", items)
        if idx is None:
            return
        path = sds[idx].get("path", "?")
        if not self._confirm(f"Remove '{path}'?", default=False):
            return
        sds.pop(idx)
        self.m.modified = True
        self._ok(f"Removed: {path}")

    def _rescan(self):
        if self.m.modified:
            if self._confirm("Unsaved config changes. Save before rescanning?"):
                self.m.save()
        kept, added, missing = self.m.rescan()
        lines = [
            f"Rescan complete.",
            f"  Videos found : {kept + added}",
            f"  Kept (known) : {kept}",
            f"  New          : {added}",
        ]
        if missing:
            lines.append("")
            lines.append("  Missing directories (skipped):")
            for d in missing:
                lines.append(f"    {d}")
        self._show(lines, "Rescan Results")
        if added:
            self._ok(f"Added {added} new video(s). Use 'Assign to category' next.")
        else:
            self._ok(f"Rescan done: {kept + added} videos.")

    # ── TEMPLATES section ──────────────────────────────────────────────────────

    def _section_templates(self):
        while True:
            self._bg()
            fmt_tmpls = self.m.templates.get("format_templates", {})
            deg_tmpls = self.m.templates.get("degradation_templates", {})
            action = self._menu("Templates", [
                (f"List format templates  [{len(fmt_tmpls)}]",      "fa"),
                ("Add format template",                               "fb"),
                ("Remove format template",                            "fc"),
                ("───", None),
                (f"List degradation templates  [{len(deg_tmpls)}]",  "da"),
                ("Add / edit degradation template",                   "db"),
                ("Remove degradation template",                       "dc"),
                ("───", None),
                ("← Back",                                           "back"),
            ])
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
        fmt_tmpls = self.m.templates.get("format_templates", {})
        if not fmt_tmpls:
            self._warn("No format templates defined.")
            return
        rows = []
        for name, spec in sorted(fmt_tmpls.items()):
            gt = spec.get("gt_size", "?")
            lr = spec.get("lr_size", "?")
            gt_str = f"{gt[0]}×{gt[1]}" if isinstance(gt, list) else str(gt)
            lr_str = f"{lr[0]}×{lr[1]}" if isinstance(lr, list) else str(lr)
            rows.append([
                name,
                str(spec.get("base_x", "–")),
                spec.get("aspect_ratio", "?"),
                str(spec.get("scale", "?")),
                gt_str,
                lr_str,
                spec.get("description", ""),
            ])
        self._table("Format Templates",
                    ["Name", "base_x", "AR", "Scale", "GT size", "LR size", "Description"],
                    rows)

    def _add_format_template(self):
        fmt_tmpls = self.m.templates.setdefault("format_templates", {})
        preset_items = [(str(x), x) for x in BASE_X_PRESETS] + \
                       [("───", None), ("Custom value…", "custom")]
        base_x_sel = self._menu("base_x (GT width)", preset_items)
        if base_x_sel is None:
            return
        if base_x_sel == "custom":
            base_x = self._int("Custom base_x:", min_val=1, title="base_x")
            if base_x is None:
                return
        else:
            base_x = base_x_sel
        ar = self._menu("aspect_ratio",
                        [(r, r) for r in sorted(ASPECT_RATIOS.keys())])
        if ar is None:
            return
        scale = self._int("scale (e.g. 3):", default=3, min_val=1, title="Scale")
        if scale is None:
            return
        try:
            gt_size, lr_size = compute_format_sizes(base_x, ar, scale)
        except ValueError as exc:
            self._err(str(exc))
            return
        ar_slug   = ar.replace(":", "")
        auto_name = f"{base_x}_{ar_slug}"
        gt_str    = f"{gt_size[0]}×{gt_size[1]}" if isinstance(gt_size, list) else str(gt_size)
        lr_str    = f"{lr_size[0]}×{lr_size[1]}" if isinstance(lr_size, list) else str(lr_size)
        name = self._input(f"Template name (GT: {gt_str}  LR: {lr_str}):",
                           default=auto_name, title="Add Format Template")
        if name is None:
            return
        name = name.strip()
        if name in fmt_tmpls:
            if not self._confirm(f"'{name}' already exists. Overwrite?", default=False):
                return
        desc = self._input("Description (optional):", title="Add Format Template") or ""
        try:
            fmt_tmpls[name] = build_format_template(base_x, ar, scale, desc)
        except ValueError as exc:
            self._err(str(exc))
            return
        self.m.templates_modified = True
        self._ok(f"Added '{name}': GT={gt_str}  LR={lr_str}")

    def _remove_format_template(self):
        fmt_tmpls = self.m.templates.get("format_templates", {})
        if not fmt_tmpls:
            self._warn("No format templates.")
            return
        name = self._menu("Select template to remove",
                          [(n, n) for n in sorted(fmt_tmpls.keys())] +
                          [("───", None), ("← Cancel", None)])
        if not name:
            return
        if not self._confirm(f"Remove '{name}'?", default=False):
            return
        del fmt_tmpls[name]
        self.m.templates_modified = True
        self._ok(f"Removed format template '{name}'")

    def _list_degradation_templates(self):
        deg_tmpls = self.m.templates.get("degradation_templates", {})
        if not deg_tmpls:
            self._warn("No degradation templates defined.")
            return
        lines = []
        for name, spec in sorted(deg_tmpls.items()):
            desc = spec.get("description", "")
            lines.append(f"  {name}" + (f"  —  {desc}" if desc else ""))
            for key in ("blur", "compression", "noise", "chroma", "color"):
                if key in spec:
                    lines.append(f"    {key}: {spec[key]}")
            lines.append("")
        self._show(lines, "Degradation Templates")

    def _add_edit_degradation_template(self):
        deg_tmpls = self.m.templates.setdefault("degradation_templates", {})
        existing  = sorted(deg_tmpls.keys())
        choices   = [(n, n) for n in existing] + \
                    [("───", None), ("+ New template…", "__new__")]
        sel = self._menu("Select template to edit (or create new)", choices)
        if sel is None:
            return
        if sel == "__new__":
            name = self._input("New template name:", title="Add Degradation Template")
            if not name or not name.strip():
                return
            name = name.strip()
        else:
            name = sel
        lines = [
            "Paste / type JSON definition.",
            "End with a blank line.",
            "",
            "Example:",
            '  {"description": "...", "blur": {"sigma_range": [0.3, 0.9], "prob": 0.75}}',
        ]
        self._show(lines, "JSON Input — press any key, then type in the terminal")
        # Fall back to raw terminal for multi-line JSON input
        curses.def_prog_mode()
        curses.endwin()
        try:
            json_lines = []
            print(f"\n[{name}]  Paste JSON  (blank line to finish):\n")
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                if not line:
                    break
                json_lines.append(line)
            spec = json.loads("\n".join(json_lines))
        except json.JSONDecodeError as e:
            curses.reset_prog_mode()
            self.stdscr.refresh()
            self._err(f"JSON parse error: {e}")
            return
        except Exception as e:
            curses.reset_prog_mode()
            self.stdscr.refresh()
            self._err(str(e))
            return
        curses.reset_prog_mode()
        self.stdscr.refresh()
        deg_tmpls[name] = spec
        self.m.templates_modified = True
        self._ok(f"Saved degradation template '{name}'")

    def _remove_degradation_template(self):
        deg_tmpls = self.m.templates.get("degradation_templates", {})
        if not deg_tmpls:
            self._warn("No degradation templates.")
            return
        name = self._menu("Select template to remove",
                          [(n, n) for n in sorted(deg_tmpls.keys())] +
                          [("───", None), ("← Cancel", None)])
        if not name:
            return
        if not self._confirm(f"Remove '{name}'?", default=False):
            return
        del deg_tmpls[name]
        self.m.templates_modified = True
        self._ok(f"Removed '{name}'")

    # ── CONFIG section ─────────────────────────────────────────────────────────

    def _section_config(self):
        while True:
            self._bg()
            action = self._menu("Config & Validation", [
                ("Show statistics",        "stats"),
                ("Validation report",      "validate"),
                ("───", None),
                ("Create new config file", "new_cfg"),
                ("───", None),
                ("← Back",                "back"),
            ])
            if action is None or action == "back":
                break
            elif action == "stats":
                self._show_statistics()
            elif action == "validate":
                self._show_validation()
            elif action == "new_cfg":
                self._create_config()

    def _show_statistics(self):
        lines = []
        n_total    = len(self.m.videos)
        n_unassign = sum(1 for v in self.m.videos if not v.get("categories"))
        lines.append(f"  Total videos  : {n_total}")
        lines.append(f"  Unassigned    : {n_unassign}")
        lines.append("")

        for cat in sorted(self.m.categories.keys()):
            cfg     = self.m.categories[cat]
            target  = cfg.get("target_total", "?")
            tstr    = f"{target:,}" if isinstance(target, int) else str(target)
            n_vids  = sum(1 for v in self.m.videos if cat in get_video_categories(v))
            formats = cfg.get("formats", [])
            total_w = sum(f.get("weight", 0) for f in formats) or 1

            lines.append(f"▸  {cat}  (target: {tstr}  videos: {n_vids})")
            lines.append("─" * 60)
            for fmt in formats:
                tmpl  = fmt.get("template", "?")
                w     = fmt.get("weight", 0)
                mode  = fmt.get("source_mode", "?")
                share = f"{w/total_w*100:.1f}%"
                deg   = ", ".join(f"{k}:{v}" for k, v in fmt.get("degradation_mix", {}).items())
                lines.append(f"    {tmpl:<20} {mode:<8} w={w:>4}  share={share}  deg={deg or '(none)'}")
            lines.append("")

        self._show(lines, "Statistics")

    def _show_validation(self):
        errors = self.m.validation_report()
        if not errors:
            self._show(["✓  No validation errors found."], "Validation Report")
        else:
            lines = [f"  {len(errors)} error(s) found:", ""]
            for e in errors:
                lines.append(f"  • {e}")
            self._show(lines, "Validation Report — ERRORS")

    def _pre_save_check(self) -> bool:
        """Validate config before saving.

        Shows a detailed error report with navigation hints if there are errors,
        then asks for explicit confirmation.  Returns True if the save should proceed.
        """
        errors = self.m.validation_report()
        if not errors:
            return True

        lines: List[str] = [
            f"⚠  {len(errors)} validation error(s) in config:",
            "",
        ]
        for err in errors:
            lines.append(f"  • {err}")
            hint = _guidance_for_error(err)
            if hint:
                lines.append(f"      {hint}")
            lines.append("")
        lines += [
            "─" * 64,
            "",
            "  The generator will refuse to run with these errors.",
            "  Fix them before running make_dataset_v2_uhd.py.",
        ]
        self._show(lines, "⚠  Validation Errors — Pre-Save Check")
        return self._confirm(
            f"{len(errors)} error(s) found.  Save anyway?",
            title="Save with errors?",
            default=False,
        )

    def _create_config(self):
        ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"generator_config_{ts}.json"
        name         = self._input("Output filename:", default=default_name,
                                   title="Create Config File")
        if not name:
            return
        out_path = str(Path(self.m.config_path).parent / (name.strip() or default_name))
        if os.path.exists(out_path):
            if not self._confirm(f"File exists: {name.strip()}\nOverwrite?", default=False):
                return
        save_active_config(create_default_active_config(), out_path)
        self._show([
            f"✓  Created: {name.strip()}",
            "",
            "Next steps:",
            "  1. Edit the file to set root_path and source_dirs",
            "  2. Restart the manager and load the new config",
        ], "Config Created")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    script_dir         = Path(__file__).parent
    active_config_path = script_dir / "generator_config.json"
    templates_path     = script_dir / "templates.json"

    if not active_config_path.exists():
        save_active_config(create_default_active_config(), str(active_config_path))
        print(f"Created default config: {active_config_path.name}")
        print("Edit root_path and source_dirs, then restart.")

    ensure_templates_file(str(templates_path))

    manager = VideoManager(str(active_config_path), str(templates_path))
    try:
        manager.load()
    except Exception as e:
        print(f"ERROR loading config: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    app = _App(manager)
    app.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
