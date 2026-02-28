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


class VideoManager:
    """Manager for video category assignments."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = {}
        self.videos = []
        self.categories = {}
        self.category_targets = {}
        self.modified = False
        
    def load(self):
        """Load configuration from JSON."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.videos = self.config.get('videos', [])
        self.category_targets = self.config.get('category_targets', {})
        
        # Sort videos by name (case-insensitive)
        self.videos.sort(key=lambda v: v.get('name', '').lower())
        
        # Extract unique categories (handle both formats)
        self.categories = set()
        for video in self.videos:
            cats = get_video_categories(video)
            self.categories.update(cats)
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
        
        print(f"\n{'ID':<6} {'Name':<50} {'Categories':<40}")
        print("-" * 100)
        
        for idx, (i, video) in enumerate(videos):
            if max_display and idx >= max_display:
                print(f"... and {len(videos) - max_display} more (use -a to show all)")
                break
            
            name = video['name'][:48]
            cats = video.get('categories', [])
            # Format categories in brackets
            if cats:
                cat_str = "[" + ", ".join(cats) + "]"
            else:
                cat_str = "[no categories]"
            
            print(f"{i:<6} {name:<50} {cat_str:<40}")
    
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
        
        # Count by category
        category_counts = {cat: 0 for cat in self.categories}
        unassigned = 0
        
        for video in self.videos:
            cats = video.get('categories', [])
            if not cats:
                unassigned += 1
            else:
                # Handle both list and dict formats
                if isinstance(cats, list):
                    for cat in cats:
                        if cat in category_counts:
                            category_counts[cat] += 1
                else:
                    # Legacy dict format
                    for cat in cats.keys():
                        if cat in category_counts:
                            category_counts[cat] += 1
        
        print(f"\nTotal videos: {len(self.videos)}")
        print(f"Unassigned: {unassigned}")
        print("\nCategory assignments:")
        for cat in sorted(category_counts.keys()):
            target = self.category_targets.get(cat, '?')
            print(f"  {cat:<15}: {category_counts[cat]:>4} videos (target: {target})")
    
    def edit_category_targets(self):
        """Edit extraction targets for categories."""
        print("\n" + "="*60)
        print("EDIT CATEGORY TARGETS")
        print("="*60)
        
        print("\nCurrent targets:")
        for cat, target in self.category_targets.items():
            print(f"  {cat}: {target}")
        
        print("\nEnter new targets (or press Enter to keep current):")
        for cat in self.categories:
            current = self.category_targets.get(cat, 0)
            new_val = input(f"{cat} (current: {current}): ").strip()
            if new_val:
                try:
                    self.category_targets[cat] = int(new_val)
                    self.config['category_targets'] = self.category_targets
                    self.modified = True
                except ValueError:
                    print(f"  Invalid number, keeping {current}")

    # ── V2 Config: Source Directory Management ────────────────────────────────

    def _is_v2_config(self) -> bool:
        """Return True if the loaded config uses V2 format (source_dirs list)."""
        return 'source_dirs' in self.config

    def list_source_dirs(self):
        """List all configured source directories (V2 config)."""
        if not self._is_v2_config():
            print("❌ Source directory management requires V2 config format (source_dirs)")
            return

        source_dirs = self.config.get('source_dirs', [])

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
        """Add a new source directory (V2 config). Not bound to any category."""
        if not self._is_v2_config():
            print("❌ Source directory management requires V2 config format (source_dirs)")
            return

        source_dirs = self.config.setdefault('source_dirs', [])

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
        """Edit an existing source directory by index (V2 config)."""
        if not self._is_v2_config():
            print("❌ Source directory management requires V2 config format (source_dirs)")
            return

        source_dirs = self.config.get('source_dirs', [])
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
        """Remove a source directory by index (V2 config)."""
        if not self._is_v2_config():
            print("❌ Source directory management requires V2 config format (source_dirs)")
            return

        source_dirs = self.config.get('source_dirs', [])
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
        """
        if not self._is_v2_config():
            print("❌ Rescan requires V2 config format (source_dirs)")
            return

        source_dirs = self.config.get('source_dirs', [])
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

        for dir_config in source_dirs:
            video_dir = dir_config.get('path', '')
            extensions = dir_config.get('extensions', extensions_default)

            if not os.path.exists(video_dir):
                print(f"⚠️  Directory not found (skipped): {video_dir}")
                continue

            for ext in extensions:
                for p in Path(video_dir).rglob(f'*{ext}'):
                    found_paths.append(str(p))

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
    print("11. Edit category targets")
    print("-" * 60)
    print("── Source Directory Config (V2) ──────────────────────")
    print("12. List source directories")
    print("13. Add source directory")
    print("14. Edit source directory")
    print("15. Remove source directory")
    print("16. Rescan file list (rebuild from source directories)")
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
        # Prefer V2 config; fall back to classic config
        script_dir = Path(__file__).parent
        v2_config = script_dir / 'generator_config_v2.json'
        v1_config = script_dir / 'generator_config.json'

        if v2_config.exists():
            config_path = v2_config
        elif v1_config.exists():
            config_path = v1_config
        else:
            print(
                "❌ No config file found. "
                "Please run from dataset_generator_v2 directory "
                "(expected generator_config_v2.json or generator_config.json)."
            )
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
                # Edit category targets
                manager.edit_category_targets()
            
            elif choice == '12':
                # List source directories (V2)
                manager.list_source_dirs()
            
            elif choice == '13':
                # Add source directory (V2)
                manager.add_source_dir()
            
            elif choice == '14':
                # Edit source directory (V2)
                manager.edit_source_dir()
            
            elif choice == '15':
                # Remove source directory (V2)
                manager.remove_source_dir()
            
            elif choice == '16':
                # Rescan file list (V2)
                manager.rescan_file_list()
            
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
