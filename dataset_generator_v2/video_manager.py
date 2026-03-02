#!/usr/bin/env python3
"""
Interactive Video Category Manager & Source Directory Config Tool
Manage video-to-category assignments and source directory configuration.

Features:
- List all videos with current assignments
- Reset all assignments
- Assign videos to categories with weights
- Multi-select by pattern (e.g., entire series)
- Edit category settings
- Preview and save changes
- Manage source directories (V2 config: add/edit/remove/rescan)
"""

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Union
import re

# Import utilities
from category_utils import (
    normalize_categories, 
    get_video_categories,
    is_video_in_category,
    format_categories_display,
    convert_config_to_list_format
)
from interactive_selector import select_items, select_categories


def _short_path(full_path: str, depth: int = 2) -> str:
    """Return the last *depth* path segments of *full_path* (depth-2 = dir/subdir/file)."""
    parts = Path(full_path).parts
    return str(Path(*parts[-depth:])) if len(parts) >= depth else full_path


class VideoManager:
    """Manager for video category assignments."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = {}
        self.videos = []
        self.categories = {}
        self.modified = False
        
    def load(self):
        """Load configuration from JSON."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # Migrate old lr_size → scale format in output_patches
        patches = self.config.get('output_patches', {})
        for val in patches.values():
            if 'lr_size' in val:
                gt_w = val['gt_size'][0]
                lr_w = val['lr_size'][0]
                val.setdefault('scale', gt_w // lr_w if lr_w else 3)
                del val['lr_size']
                self.modified = True
        
        self.videos = self.config.get('videos', [])
        
        # Sort videos by name (case-insensitive)
        self.videos.sort(key=lambda v: v.get('name', '').lower())
        
        # Extract unique categories from video assignments and config
        self.categories = set()
        for video in self.videos:
            cats = get_video_categories(video)
            self.categories.update(cats)

        self.categories.update(self.config.get('category_patches', {}).keys())
        self.categories = sorted(list(self.categories))
        
        print(f"✓ Loaded {len(self.videos)} videos (sorted by title)")
        print(f"✓ Categories: {', '.join(self.categories)}")
        
    def save(self, backup=True):
        """Save configuration to JSON."""
        # Sort videos with priority to multi-category videos
        # Order: (1) number of categories DESC, (2) first category ASC, (3) name ASC
        # This ensures videos in multiple categories come first, with master at the top
        def sort_key(video):
            cats = get_video_categories(video)
            # Primary sort: number of categories (descending - negative for reverse)
            # Videos in multiple categories (e.g., master+space) come before single category
            num_cats = -len(cats) if cats else 999
            # Secondary sort: first category alphabetically (master comes first)
            first_cat = cats[0] if cats else 'zzz_no_category'
            # Tertiary sort: video name
            name = video.get('name', '').lower()
            return (num_cats, first_cat, name)
        
        self.videos.sort(key=sort_key)
        self.config['videos'] = self.videos
        
        if backup:
            backup_path = self.config_path + '.backup'
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"✓ Backup saved to {backup_path}")
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved to {self.config_path} (videos sorted by: multi-category priority, category, then title)")
        self.modified = False
        
    def list_videos(self, filter_pattern: Optional[str] = None, 
                   category: Optional[str] = None, 
                   show_unassigned: bool = False,
                   use_simple_search: bool = False):
        """
        List videos with optional filtering.
        
        Args:
            filter_pattern: Pattern to match (regex or simple string)
            category: Filter by category
            show_unassigned: Show only unassigned videos
            use_simple_search: Use simple substring search instead of regex
        """
        
        filtered = []
        for i, video in enumerate(self.videos):
            # Filter by pattern
            if filter_pattern:
                if use_simple_search:
                    # Simple case-insensitive substring search
                    if filter_pattern.lower() not in video['name'].lower():
                        continue
                else:
                    # Regex search with error handling
                    try:
                        if not re.search(filter_pattern, video['name'], re.IGNORECASE):
                            continue
                    except re.error as e:
                        # Invalid regex - fall back to simple search
                        print(f"⚠️  Invalid regex pattern, using simple search instead: {e}")
                        if filter_pattern.lower() not in video['name'].lower():
                            continue
            
            # Filter by category
            if category:
                if category not in video.get('categories', {}):
                    continue
            
            # Filter unassigned
            if show_unassigned:
                if video.get('categories', {}):
                    continue
            
            filtered.append((i, video))
        
        return filtered
    
    def print_video_list(self, videos: List[tuple], max_display: int = 20):
        """Pretty print video list."""
        if not videos:
            print("No videos found.")
            return
        
        print(f"\n{'ID':<6} {'Name':<44} {'Path (depth-2)':<36} {'Categories'}")
        print("-" * 120)
        
        for idx, (i, video) in enumerate(videos):
            if max_display and idx >= max_display:
                print(f"... and {len(videos) - max_display} more (use -a to show all)")
                break
            
            name = video['name'][:42]
            path_short = _short_path(video.get('path', ''), depth=3)[:34]
            cats = video.get('categories', [])
            if cats:
                cat_str = "[" + ", ".join(cats) + "]"
            else:
                cat_str = "[no categories]"

            # Append ⚡ indicator when forced_frames are set for any category
            forced = video.get('forced_frames', {})
            if forced:
                forced_parts = [f"{cat}:{n:,}" for cat, n in sorted(forced.items()) if n > 0]
                if forced_parts:
                    cat_str += "  ⚡ " + "  ".join(forced_parts)
            
            print(f"{i:<6} {name:<44} {path_short:<36} {cat_str}")
    
    def set_forced_frames(self, video_indices) -> None:
        """
        Interactively set per-category forced frame overrides for one or more videos.

        When multiple indices are given the user enters frame counts once and the
        same values are applied to every selected video.  This makes bulk-setting
        easy for entire series or collections.

        For each category the user is prompted for an exact frame count:
          - blank   → keep the existing value for that video
          - 0       → remove override (back to auto proportional)
          - N > 0   → force exactly N frames for this category
        """
        # Accept both a single int and a list
        if isinstance(video_indices, int):
            video_indices = [video_indices]

        # Validate all indices first
        valid = []
        for idx in video_indices:
            if 0 <= idx < len(self.videos):
                valid.append(idx)
            else:
                print(f"  ⚠️  Skipping invalid index {idx}")
        if not valid:
            print("❌ No valid video indices provided.")
            return

        # Collect the union of all categories across selected videos
        all_cats: List[str] = []
        for idx in valid:
            for cat in get_video_categories(self.videos[idx]):
                if cat not in all_cats:
                    all_cats.append(cat)

        if not all_cats:
            print("❌ None of the selected videos have categories assigned — assign categories first.")
            return

        names = [self.videos[idx].get('name', '?') for idx in valid]
        if len(names) == 1:
            print(f"\n📹 Forced frame overrides for: {names[0]}")
        else:
            print(f"\n📹 Forced frame overrides for {len(names)} videos:")
            for n in names:
                print(f"   • {n}")
        print(f"   Categories: {', '.join(all_cats)}")
        print("   Enter the exact number of frames each video must contribute for each")
        print("   category (blank = keep existing, 0 = remove override / use auto).\n")

        # Collect new values from user (once, applied to all)
        new_values: Dict[str, Optional[int]] = {}   # cat → int or None (= keep unchanged)
        for cat in all_cats:
            # Show current values if all selected videos agree
            cur_values = [self.videos[idx].get('forced_frames', {}).get(cat, 0) for idx in valid]
            if len(set(cur_values)) == 1:
                current_str = f"{cur_values[0]:,}"
            else:
                current_str = "mixed"
            prompt = f"  {cat} (current: {current_str}  |  0 = auto): "
            raw = input(prompt).strip()

            if not raw:
                new_values[cat] = None   # keep unchanged
                continue

            try:
                value = int(raw)
            except ValueError:
                print(f"  ⚠️  Invalid number '{raw}' — keeping current values")
                new_values[cat] = None
                continue

            if value < 0:
                print(f"  ⚠️  Negative value ignored — keeping current values")
                new_values[cat] = None
            else:
                new_values[cat] = value   # 0 means "remove override"

        # Apply to every selected video
        for idx in valid:
            video = self.videos[idx]
            forced = dict(video.get('forced_frames', {}))
            for cat, value in new_values.items():
                if value is None:
                    continue   # keep unchanged
                if value == 0:
                    forced.pop(cat, None)
                else:
                    # Only apply if this video is actually in this category
                    if cat in get_video_categories(video):
                        forced[cat] = value
            if forced:
                video['forced_frames'] = forced
            else:
                video.pop('forced_frames', None)

        self.modified = True

        # Summary
        set_cats = {cat: v for cat, v in new_values.items() if v is not None}
        if set_cats:
            parts = "  ".join(
                f"{cat}: {v:,}" if v else f"{cat}: auto" for cat, v in sorted(set_cats.items())
            )
            print(f"\n✓ Applied to {len(valid)} video(s): {parts}")
        else:
            print(f"\n✓ No changes made (all entries were blank).")

    def reset_all(self):
        """Reset all video assignments."""
        confirm = input("⚠️  Reset ALL video assignments? This cannot be undone! (yes/no): ")
        if confirm.lower() != 'yes':
            print("Cancelled.")
            return
        
        for video in self.videos:
            video['categories'] = {}
        
        self.modified = True
        print(f"✓ Reset {len(self.videos)} videos")
    
    def assign_videos(self, video_indices: List[int], 
                     categories: List[str],
                     mode: str = 'ask'):
        """
        Assign categories to videos (NO WEIGHTS - simple list).
        
        Args:
            video_indices: List of video indices to assign
            categories: List of category names to assign
            mode: 'ask' (prompt user), 'add' (append to existing), 'replace' (replace all)
        """
        if not video_indices:
            return
        
        # Check if any videos already have categories
        has_existing_categories = False
        for idx in video_indices:
            if 0 <= idx < len(self.videos):
                existing = get_video_categories(self.videos[idx])
                if existing:
                    has_existing_categories = True
                    break
        
        # Determine the actual mode to use
        actual_mode = mode
        if mode == 'ask' and has_existing_categories:
            print("\n⚠️  Some videos already have categories assigned.")
            print("Options:")
            print("  1. ADD to existing categories (keep old + add new)")
            print("  2. REPLACE all categories (remove old, set new)")
            choice = input("Choose (1/2) [default: 1]: ").strip()
            
            if choice == '2':
                actual_mode = 'replace'
            else:
                actual_mode = 'add'
        elif mode == 'ask':
            # No existing categories, just assign
            actual_mode = 'replace'
        
        # Apply the assignment
        count = 0
        for idx in video_indices:
            if 0 <= idx < len(self.videos):
                if actual_mode == 'add':
                    # Get existing categories and add new ones
                    existing = get_video_categories(self.videos[idx])
                    # Merge and remove duplicates while preserving order
                    combined = existing.copy()
                    for cat in categories:
                        if cat not in combined:
                            combined.append(cat)
                    self.videos[idx]['categories'] = combined
                else:  # replace
                    self.videos[idx]['categories'] = categories
                count += 1
        
        self.modified = True
        mode_text = "Added to" if actual_mode == 'add' else "Replaced with"
        print(f"✓ {mode_text} {count} videos: {categories}")
    
    def interactive_select_videos(self, initial_filter: Optional[str] = None):
        """
        Interactive video selection with curses-based UI (arrow keys, space bar).
        
        Args:
            initial_filter: Optional filter to pre-filter videos
            
        Returns:
            List of selected video indices or None if cancelled
        """
        # Get initial list of videos
        if initial_filter:
            videos = self.list_videos(filter_pattern=initial_filter, use_simple_search=True)
        else:
            videos = self.list_videos()
        
        if not videos:
            print("No videos found.")
            return None
        
        # Use curses-based interactive selector
        try:
            # Extract just the video objects for display
            video_list = [v for _, v in videos]
            
            # Create display with video ID in details
            def get_video_label(video):
                return video['name']
            
            def get_video_details(video):
                # Find the video ID
                video_id = next((i for i, v in videos if v == video), None)
                cats = video.get('categories', [])
                cat_str = format_categories_display(cats)
                if video_id is not None:
                    return f"[{video_id}] {cat_str}"
                return cat_str
            
            selected_indices = select_items(
                items=video_list,
                title=f"Select Videos - {len(video_list)} available (↑↓ navigate, Space toggle, Enter done, Esc cancel)",
                get_label=get_video_label,
                get_details=get_video_details
            )
            
            if selected_indices is None:
                return None
            
            # Convert from list indices to video IDs
            video_ids = [videos[i][0] for i in selected_indices]
            return video_ids
            
        except Exception as e:
            print(f"⚠️  Curses UI failed: {e}")
            print("Please try using menu option 5 instead")
            return None
    
    def remove_from_category(self, video_indices: List[int], category: str):
        """Remove videos from a specific category."""
        count = 0
        for idx in video_indices:
            if 0 <= idx < len(self.videos):
                cats = self.videos[idx].get('categories', [])
                # Handle both dict and list formats
                cats_list = normalize_categories(cats)
                
                if category in cats_list:
                    cats_list.remove(category)
                    self.videos[idx]['categories'] = cats_list
                    count += 1
        
        self.modified = True
        print(f"✓ Removed {count} videos from category '{category}'")
    
    def show_statistics(self):
        """Show statistics about current assignments."""
        print("\n" + "="*60)
        print("STATISTICS")
        print("="*60)
        
        # Count by category; also collect forced-frames videos per category
        category_counts: Dict[str, int] = {cat: 0 for cat in self.categories}
        # cat → list of (name, short_path, {cat: forced_count})
        forced_by_cat: Dict[str, list] = {cat: [] for cat in self.categories}
        unassigned = 0
        
        for video in self.videos:
            cats = video.get('categories', [])
            cat_list = normalize_categories(cats)
            if not cat_list:
                unassigned += 1
            else:
                for cat in cat_list:
                    if cat in category_counts:
                        category_counts[cat] += 1
            forced = video.get('forced_frames', {})
            if forced:
                for cat, n in forced.items():
                    if cat in forced_by_cat and n > 0:
                        forced_by_cat[cat].append((
                            video.get('name', '?'),
                            _short_path(video.get('path', ''), depth=3),
                            n,
                        ))
        
        print(f"\nTotal videos: {len(self.videos)}")
        print(f"Unassigned: {unassigned}")
        print("\nCategory assignments:")
        for cat in sorted(category_counts.keys()):
            target = self.config.get('category_patches', {}).get(cat, '?')
            target_str = f"{target:,}" if isinstance(target, int) else str(target)
            print(f"  {cat:<15}: {category_counts[cat]:>4} videos (target: {target_str})")
            # List forced videos for this category
            if forced_by_cat.get(cat):
                forced_list = sorted(forced_by_cat[cat], key=lambda x: x[0].lower())
                for name, short_path, n in forced_list:
                    print(f"    ⚡ {name:<40} {short_path:<36}  forced: {n:>8,}")
    
    def manage_categories(self):
        """Category management submenu: list, add, remove, edit."""
        while True:
            print("\n── Category Management ───────────────────────────────────")
            print(f"  Configured categories: {', '.join(self.categories) or '(none)'}")
            print("  a) List categories")
            print("  b) Add category")
            print("  c) Remove category")
            print("  d) Edit category target")
            print("  x) Back")
            sub = input("Choice: ").strip().lower()

            if sub == 'x':
                break
            elif sub == 'a':
                self._list_categories()
            elif sub == 'b':
                self._add_category()
            elif sub == 'c':
                self._remove_category()
            elif sub == 'd':
                self._edit_category()
            else:
                print("Invalid choice")

    def _list_categories(self):
        """Print all configured categories with their patch targets."""
        if not self.categories:
            print("No categories configured.")
            return
        patches = self.config.get('category_patches', {})
        print(f"\n{'Category':<20} {'Patches':<12}")
        print("-" * 34)
        for cat in sorted(self.categories):
            p = patches.get(cat, '?')
            print(f"  {cat:<18} {str(p):<12}")

    def _add_category(self):
        """Add a new category with a patch target count."""
        name = input("New category name: ").strip().lower()
        if not name:
            print("❌ Name cannot be empty")
            return
        if name in self.categories:
            print(f"❌ Category '{name}' already exists")
            return

        target_str = input("Patch target count (default: 50000): ").strip()
        try:
            target = int(target_str) if target_str else 50000
        except ValueError:
            print("Invalid number, using 50000")
            target = 50000

        patches = self.config.setdefault('category_patches', {})
        patches[name] = target

        self.categories = sorted(self.categories + [name])
        self.modified = True
        print(f"✓ Added category '{name}' with target {target:,}")

    def _edit_category(self):
        """Edit the patch target count for an existing category."""
        if not self.categories:
            print("No categories configured.")
            return
        self._list_categories()
        name = input("Category to edit: ").strip()
        if name not in self.categories:
            print(f"❌ Category '{name}' not found")
            return

        patches = self.config.setdefault('category_patches', {})
        current = patches.get(name, 0)
        val_str = input(f"New patch target (current: {current:,}): ").strip()
        if not val_str:
            print("No change.")
            return
        try:
            new_target = int(val_str)
        except ValueError:
            print("❌ Invalid number")
            return
        if new_target <= 0:
            print("❌ Target must be a positive integer")
            return

        patches[name] = new_target
        self.modified = True
        print(f"✓ Category '{name}' target updated: {current:,} → {new_target:,}")

    def _remove_category(self):
        """Remove a category and unassign all videos from it."""
        if not self.categories:
            print("No categories configured.")
            return
        print(f"Categories: {', '.join(self.categories)}")
        name = input("Category to remove: ").strip()
        if name not in self.categories:
            print(f"❌ Category '{name}' not found")
            return

        affected = sum(1 for v in self.videos if name in get_video_categories(v))
        suffix = f" and unassign {affected} video(s)" if affected else ""
        confirm = input(f"⚠️  Remove '{name}'{suffix}? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("Cancelled.")
            return

        # Unassign videos
        for video in self.videos:
            cats = get_video_categories(video)
            if name in cats:
                cats.remove(name)
                video['categories'] = cats

        # Remove from config
        self.config.get('category_patches', {}).pop(name, None)

        self.categories = [c for c in self.categories if c != name]
        self.modified = True
        print(f"✓ Removed category '{name}'" + (f", unassigned {affected} video(s)" if affected else ""))

    # ── V2 Config: Output Patch Format Weights ────────────────────────────────

    def manage_patch_formats(self):
        """Output patch size weight management submenu."""
        while True:
            print("\n── Output Patch Size Weights ─────────────────────────────────")
            self._list_patch_formats()
            print("\n  a) Edit weight for a size")
            print("  x) Back")
            sub = input("Choice: ").strip().lower()

            if sub == 'x':
                break
            elif sub == 'a':
                self._edit_patch_weight()
            else:
                print("Invalid choice")

    def _list_patch_formats(self):
        """Print output patch sizes with their weights (in %) and resulting probabilities."""
        patches = self.config.get('output_patches', {})
        if not patches:
            print("No output patch sizes configured.")
            return

        enabled = {k: v for k, v in patches.items() if v.get('enabled', True)}
        total_weight = sum(v.get('weight', 1) for v in enabled.values())
        if total_weight <= 0:
            total_weight = max(len(enabled), 1)

        print(f"\n{'Size':<12} {'Enabled':<10} {'Weight':<10} {'Share %':<10}")
        print("-" * 44)
        for key in sorted(patches.keys()):
            val = patches[key]
            is_enabled = val.get('enabled', True)
            enabled_str = "✓" if is_enabled else "✗"
            w = val.get('weight', 1)
            if is_enabled and total_weight > 0:
                share = round(w / total_weight * 100, 1)
                share_str = f"{share}%"
            else:
                share_str = "–"
            print(f"  {key:<10} {enabled_str:<10} {w:<10} {share_str}")

    def _edit_patch_weight(self):
        """Edit the weight (in %) for an output patch size."""
        patches = self.config.get('output_patches', {})
        if not patches:
            print("No output patch sizes configured.")
            return

        print(f"Available sizes: {', '.join(sorted(patches.keys()))}")
        size = input("Size to edit (e.g. 720): ").strip()
        if size not in patches:
            print(f"❌ Unknown size '{size}'")
            return

        current = patches[size].get('weight', 1)
        val_str = input(f"New weight in % (current: {current}): ").strip()
        if not val_str:
            print("No change.")
            return
        try:
            weight = int(val_str)
        except ValueError:
            print("❌ Invalid number – must be a positive integer")
            return
        if weight <= 0:
            print("❌ Weight must be a positive integer")
            return

        patches[size]['weight'] = weight
        self.modified = True

        # Show updated distribution
        enabled = {k: v for k, v in patches.items() if v.get('enabled', True)}
        total_weight = sum(v.get('weight', 1) for v in enabled.values())
        share = round(weight / total_weight * 100, 1) if total_weight > 0 else 0.0
        print(f"✓ Weight for '{size}' set to {weight}  (→ {share}% of patches)")

    # ── V2 Config: Source Directory Management ────────────────────────────────

    def _ensure_source_dirs(self) -> list:
        """
        Return the source_dirs list, auto-initialising it in the config if absent.
        This allows source-directory management on any config format.
        """
        if 'source_dirs' not in self.config:
            self.config['source_dirs'] = []
            self.modified = True
        return self.config['source_dirs']

    def list_source_dirs(self):
        """List all configured source directories."""
        source_dirs = self._ensure_source_dirs()

        if not source_dirs:
            print("No source directories configured.")
            return

        print(f"\n{'#':<5} {'Path':<65} {'Extensions'}")
        print("-" * 100)
        for i, entry in enumerate(source_dirs):
            path = entry.get('path', '?')
            exts = ', '.join(entry.get('extensions', []))
            print(f"{i:<5} {path:<65} {exts}")

    def add_source_dir(self):
        """Add a new source directory. Not bound to any category."""
        source_dirs = self._ensure_source_dirs()

        path = input("Directory path: ").strip()
        if not path:
            print("❌ Path cannot be empty")
            return
        if any(d.get('path') == path for d in source_dirs):
            print(f"❌ Directory already configured: {path}")
            return

        exts_str = input("Extensions [default: .mkv,.mp4,.avi]: ").strip()
        exts = [e.strip() for e in exts_str.split(',')] if exts_str else ['.mkv', '.mp4', '.avi']

        source_dirs.append({'path': path, 'extensions': exts})
        self.modified = True
        print(f"✓ Added source directory: {path}")

    def edit_source_dir(self):
        """Edit an existing source directory by index."""
        source_dirs = self._ensure_source_dirs()
        if not source_dirs:
            print("No source directories configured.")
            return

        self.list_source_dirs()
        try:
            idx = int(input("Number of directory to edit: ").strip())
        except ValueError:
            print("❌ Invalid input")
            return
        if idx < 0 or idx >= len(source_dirs):
            print("❌ Invalid number")
            return

        entry = source_dirs[idx]
        current_exts = ', '.join(entry.get('extensions', []))

        new_path = input(f"New path (current: {entry['path']}): ").strip()
        if new_path:
            entry['path'] = new_path

        new_exts = input(f"New extensions (current: {current_exts}): ").strip()
        if new_exts:
            entry['extensions'] = [e.strip() for e in new_exts.split(',')]

        self.modified = True
        print(f"✓ Updated source directory #{idx}")

    def remove_source_dir(self):
        """Remove a source directory by index."""
        source_dirs = self._ensure_source_dirs()
        if not source_dirs:
            print("No source directories configured.")
            return

        self.list_source_dirs()
        try:
            idx = int(input("Number of directory to remove: ").strip())
        except ValueError:
            print("❌ Invalid input")
            return
        if idx < 0 or idx >= len(source_dirs):
            print("❌ Invalid number")
            return

        path = source_dirs[idx].get('path', '?')
        confirm = input(f"⚠️  Remove '{path}'? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("Cancelled.")
            return

        source_dirs.pop(idx)
        self.modified = True
        print(f"✓ Removed source directory: {path}")

    def rescan_file_list(self):
        """
        Scan all configured source directories and rebuild the video list.
        Existing category assignments are preserved for already-known paths.
        New files are added with empty categories; assign them via menu options 5/6/7.
        Works on any config format; source_dirs is auto-initialised if absent.
        """
        source_dirs = self._ensure_source_dirs()
        if not source_dirs:
            print("❌ No source directories configured. Add directories first (option 13).")
            return

        if self.modified:
            answer = input("⚠️  You have unsaved changes. Save before rescanning? (yes/no): ").strip().lower()
            if answer == 'yes':
                self.save(backup=False)

        # Build lookup: path → existing video entry (to preserve category assignments)
        existing_by_path = {v.get('path', ''): v for v in self.config.get('videos', [])}

        extensions_default = ['.mkv', '.mp4', '.avi']
        found_paths: List[str] = []
        seen_paths: set = set()

        for dir_config in source_dirs:
            video_dir = dir_config.get('path', '')
            extensions = dir_config.get('extensions', extensions_default)

            if not os.path.exists(video_dir):
                print(f"⚠️  Directory not found (skipped): {video_dir}")
                continue

            # Case-insensitive extension matching + deduplication
            exts_lower = {e.lower() for e in extensions}
            for p in Path(video_dir).rglob('*'):
                if p.is_file() and p.suffix.lower() in exts_lower:
                    path_str = str(p)
                    if path_str not in seen_paths:
                        seen_paths.add(path_str)
                        found_paths.append(path_str)

        # Merge: keep existing entries (with their categories), add new ones without category
        new_videos: List[dict] = []
        added = 0
        kept = 0
        for path in sorted(found_paths):
            if path in existing_by_path:
                new_videos.append(existing_by_path[path])
                kept += 1
            else:
                name = Path(path).stem
                new_videos.append({'name': name, 'path': path, 'categories': []})
                added += 1

        self.config['videos'] = new_videos
        self.videos = new_videos
        self.modified = True

        print(f"✅ Rescan complete: {len(new_videos)} videos total "
              f"({kept} kept with existing categories, {added} newly added)")
        if added:
            print("   Use menu options 5 / 6 / 7 to assign categories to new videos.")


def print_menu():
    """Print main menu."""
    print("\n" + "="*60)
    print("VIDEO CATEGORY MANAGER")
    print("="*60)
    print("1. List all videos")
    print("2. List videos by category")
    print("3. List unassigned videos")
    print("4. Search videos by name")
    print("5. Assign video(s) to categories")
    print("6. Interactive multi-select (NEW!)")
    print("7. Multi-assign by pattern (regex/search)")
    print("8. Remove from category")
    print("9. Reset all assignments")
    print("10. Show statistics")
    print("11. Manage categories (add / remove / edit targets & formats)")
    print("-" * 60)
    print("── Source Directory Config (V2) ──────────────────────")
    print("12. List source directories")
    print("13. Add source directory")
    print("14. Edit source directory")
    print("15. Remove source directory")
    print("16. Rescan file list (rebuild from source directories)")
    print("17. Create new default config file")
    print("18. Configure output patch size weights")
    print("19. Set forced frame overrides (per video / per category)")
    print("-" * 60)
    print("s. Save changes")
    print("q. Quit")
    print("="*60)


def get_categories_interactive(manager: VideoManager, current_categories: List[str] = None) -> Optional[List[str]]:
    """Interactive category selection using curses UI."""
    try:
        categories = select_categories(
            available_categories=manager.categories,
            current_categories=current_categories
        )
        return categories
    except Exception as e:
        print(f"⚠️  Curses UI not available: {e}")
        print("Falling back to simple input...")
        return get_categories_simple(manager, current_categories)


def get_categories_simple(manager: VideoManager, current_categories: List[str] = None) -> Optional[List[str]]:
    """Simple text-based category selection (fallback)."""
    print(f"\nAvailable categories: {', '.join(manager.categories)}")
    if current_categories:
        print(f"Current categories: {', '.join(current_categories)}")
    
    print("\nEnter category names (comma-separated) or 'none' to clear:")
    print("Example: master,space,toon")
    
    while True:
        val = input("Categories: ").strip()
        if not val:
            continue
        
        if val.lower() == 'none':
            return []
        
        # Split and clean
        cats = [c.strip() for c in val.split(',')]
        
        # Validate
        invalid = [c for c in cats if c not in manager.categories]
        if invalid:
            print(f"❌ Invalid categories: {', '.join(invalid)}")
            print(f"Available: {', '.join(manager.categories)}")
            continue
        
        return cats


def main():
    """Main entry point for video manager CLI."""
    try:
        script_dir = Path(__file__).parent
        v2_config  = script_dir / 'generator_config_v2.json'

        if v2_config.exists():
            config_path = v2_config
        else:
            print("⚠️  Keine Konfiguration gefunden – erstelle generator_config_v2.json ...")
            try:
                from create_default_config import create_default_config
                create_default_config(str(v2_config))
                config_path = v2_config
                print(f"✓ Konfiguration erstellt: {v2_config.name}")
                print("  → Bitte root_path und source_dirs anpassen, dann neu starten.")
            except Exception as ce:
                print(f"❌ Konnte keine Konfiguration erstellen: {ce}")
                sys.exit(1)

        print(f"📂 Using config: {config_path.name}")
        manager = VideoManager(str(config_path))
        manager.load()
    except Exception as e:
        print(f"❌ Error initializing Video Manager: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    while True:
        choice = ""  # Initialize to avoid NameError in exception handler
        try:
            print_menu()
            choice = input("\nChoice: ").strip().lower()
            
            if choice == 'q':
                if manager.modified:
                    save = input("Save changes before quitting? (y/n): ").strip().lower()
                    if save == 'y':
                        manager.save()
                print("Goodbye!")
                break
            
            elif choice == 's':
                if manager.modified:
                    manager.save()
                else:
                    print("No changes to save")
            
            elif choice == '1':
                # List all videos
                show_all = input("Show all videos? (y/n, default=first 20): ").strip().lower()
                max_display = None if show_all == 'y' else 20
                videos = manager.list_videos()
                manager.print_video_list(videos, max_display)
            
            elif choice == '2':
                # List by category
                print(f"Categories: {', '.join(manager.categories)}")
                cat = input("Category: ").strip()
                if cat in manager.categories:
                    videos = manager.list_videos(category=cat)
                    manager.print_video_list(videos)
                else:
                    print(f"❌ Unknown category: {cat}")
            
            elif choice == '3':
                # List unassigned
                videos = manager.list_videos(show_unassigned=True)
                manager.print_video_list(videos)
            
            elif choice == '4':
                # Search
                pattern = input("Search pattern (regex): ").strip()
                if pattern:
                    videos = manager.list_videos(filter_pattern=pattern)
                    manager.print_video_list(videos)
            
            elif choice == '5':
                # Assign categories to videos (with interactive selector)
                print("\n🎯 Assign Videos to Categories")
                print("Step 1: Select videos")
                print("  a) Select videos interactively (curses UI)")
                print("  b) Enter video IDs manually")
                
                method = input("Method (a/b): ").strip().lower()
                
                video_indices = []
                if method == 'a':
                    # Interactive video selection with curses
                    filter_str = input("Optional filter (leave empty for all): ").strip()
                    videos = manager.list_videos(filter_pattern=filter_str if filter_str else None)
                    
                    if not videos:
                        print("No videos found")
                        continue
                    
                    try:
                        selected = select_items(
                            items=[v for _, v in videos],
                            title="Select videos (Space to toggle, Enter to confirm)",
                            get_label=lambda v: v['name'],
                            get_details=lambda v: format_categories_display(v.get('categories', []))
                        )
                        
                        if selected is None:
                            print("❌ Cancelled")
                            continue
                        
                        video_indices = [videos[i][0] for i in selected]
                        print(f"✓ Selected {len(video_indices)} videos")
                        
                    except Exception as e:
                        print(f"⚠️  Curses UI failed: {e}")
                        print("Please use method 'b' for manual ID entry")
                        continue
                
                elif method == 'b':
                    # Manual ID entry
                    ids_str = input("Video ID(s) (comma-separated): ").strip()
                    try:
                        video_indices = [int(x.strip()) for x in ids_str.split(',')]
                    except ValueError:
                        print("❌ Invalid ID format")
                        continue
                else:
                    print("❌ Invalid method")
                    continue
                
                if not video_indices:
                    print("No videos selected")
                    continue
                
                # Step 2: Select categories interactively
                print("\nStep 2: Select categories")
                categories = get_categories_interactive(manager)
                
                if categories is None:
                    print("❌ Cancelled")
                    continue
                
                if not categories:
                    print("⚠️  No categories selected - videos will be SKIPPED")
                    confirm = input("Proceed anyway? (y/n): ").strip().lower()
                    if confirm != 'y':
                        continue
                
                # Assign
                manager.assign_videos(video_indices, categories)
            
            elif choice == '6':
                # Interactive multi-select (NEW!)
                filter_str = input("Optional filter (leave empty for all, or enter text to search): ").strip()
                selected_ids = manager.interactive_select_videos(filter_str if filter_str else None)
                
                if selected_ids:
                    print(f"\n✓ Selected {len(selected_ids)} videos")
                    # Select categories interactively
                    categories = get_categories_interactive(manager)
                    if categories:
                        manager.assign_videos(selected_ids, categories)
                else:
                    print("Selection cancelled")
            
            elif choice == '7':
                # Multi-assign by pattern (regex/search)
                pattern = input("Search pattern (text or regex, e.g., 'Star Trek' or 'Star Trek.*'): ").strip()
                if not pattern:
                    continue
                
                # Try simple search first
                use_simple = '*' in pattern or not any(c in pattern for c in r'\.[](){}^$+?|')
                
                videos = manager.list_videos(filter_pattern=pattern, use_simple_search=use_simple)
                if not videos:
                    print(f"No videos match pattern: {pattern}")
                    continue
                
                manager.print_video_list(videos)
                
                confirm = input(f"\nAssign all {len(videos)} videos? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue
                
                # Select categories interactively
                categories = get_categories_interactive(manager)
                if categories:
                    ids = [i for i, v in videos]
                    manager.assign_videos(ids, categories)
            
            elif choice == '8':
                # Remove from category
                print(f"Categories: {', '.join(manager.categories)}")
                cat = input("Category to remove: ").strip()
                if cat not in manager.categories:
                    print(f"❌ Unknown category: {cat}")
                    continue
                
                ids_str = input("Video ID(s) (comma-separated, or 'all'): ").strip()
                if ids_str.lower() == 'all':
                    ids = list(range(len(manager.videos)))
                else:
                    try:
                        ids = [int(x.strip()) for x in ids_str.split(',')]
                    except ValueError:
                        print("❌ Invalid ID format")
                        continue
                
                manager.remove_from_category(ids, cat)
            
            elif choice == '9':
                # Reset all
                manager.reset_all()
            
            elif choice == '10':
                # Statistics
                manager.show_statistics()
            
            elif choice == '11':
                # Manage categories
                manager.manage_categories()
            
            elif choice == '12':
                # List source directories
                manager.list_source_dirs()
            
            elif choice == '13':
                # Add source directory
                manager.add_source_dir()
            
            elif choice == '14':
                # Edit source directory
                manager.edit_source_dir()
            
            elif choice == '15':
                # Remove source directory
                manager.remove_source_dir()
            
            elif choice == '16':
                # Rescan file list
                manager.rescan_file_list()
            
            elif choice == '17':
                # Create new default config file
                try:
                    from create_default_config import create_default_config, build_default_config
                    script_dir = Path(__file__).parent
                    template   = str(script_dir / 'generator_config_v2.json')
                    ts         = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
                    default_name = f'generator_config_new_{ts}.json'
                    val = input(f"Output filename [{default_name}]: ").strip()
                    out_path = str(script_dir / (val or default_name))
                    if os.path.exists(out_path):
                        ow = input(f"⚠️  '{out_path}' exists. Overwrite? (yes/no): ").strip().lower()
                        if ow != 'yes':
                            print("Cancelled.")
                        else:
                            create_default_config(out_path, template_path=template)
                            print("   Open the file to adjust settings, then reload video_manager.py with it.")
                    else:
                        create_default_config(out_path, template_path=template)
                        print("   Open the file to adjust settings, then reload video_manager.py with it.")
                except Exception as e:
                    print(f"❌ Could not create config: {e}")
                    traceback.print_exc()

            elif choice == '18':
                # Configure output patch size weights
                manager.manage_patch_formats()

            elif choice == '19':
                # Set forced frame overrides (per video / per category)
                print("\n🎯 Set Forced Frame Overrides")
                print("  Select one or more videos, then enter per-category frame counts.")
                print("  Only videos with at least one category assigned are shown.")

                filter_str = input("Optional filter (leave empty for all assigned videos): ").strip()
                # Only show videos that have at least one category assigned
                assigned_videos = manager.list_videos(
                    filter_pattern=filter_str if filter_str else None,
                    use_simple_search=True,
                )
                assigned_videos = [(i, v) for i, v in assigned_videos if get_video_categories(v)]

                if not assigned_videos:
                    print("No assigned videos found.")
                    continue

                # Curses multi-select picker
                try:
                    selected = select_items(
                        items=[v for _, v in assigned_videos],
                        title=(
                            f"Select videos for forced frames — {len(assigned_videos)} available "
                            "(Space toggle, Enter confirm, Esc cancel)"
                        ),
                        get_label=lambda v: v['name'],
                        get_details=lambda v: (
                            _short_path(v.get('path', ''), depth=3)
                            + "  [" + ", ".join(get_video_categories(v)) + "]"
                            + ("  ⚡" if v.get('forced_frames') else "")
                        ),
                    )

                    if selected is None or len(selected) == 0:
                        print("❌ Cancelled or nothing selected")
                        continue

                    video_indices = [assigned_videos[s][0] for s in selected]

                except Exception as e:
                    print(f"⚠️  Curses UI failed ({e}), falling back to ID input")
                    manager.print_video_list(assigned_videos, max_display=20)
                    ids_str = input("Video ID(s) (comma-separated): ").strip()
                    try:
                        video_indices = [int(x.strip()) for x in ids_str.split(',')]
                    except ValueError:
                        print("❌ Invalid ID")
                        continue

                manager.set_forced_frames(video_indices)

            else:
                print("Invalid choice")
        
        except EOFError:
            print("\n\n⚠️  End of input detected")
            break
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            if manager.modified:
                try:
                    save = input("\nSave changes before quitting? (y/n): ").strip().lower()
                    if save == 'y':
                        manager.save()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting without saving")
            break
        except Exception as e:
            print(f"\n⚠️  Error processing menu choice '{choice}': {e}")
            traceback.print_exc()
            print("\nContinuing...")
            continue


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
