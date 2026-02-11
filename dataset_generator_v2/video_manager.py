#!/usr/bin/env python3
"""
Interactive Video Category Manager
Manage video-to-category assignments easily without editing huge JSON files.

Features:
- List all videos with current assignments
- Reset all assignments
- Assign videos to categories with weights
- Multi-select by pattern (e.g., entire series)
- Edit category settings
- Preview and save changes
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
        
        # Extract unique categories (handle both formats)
        self.categories = set()
        for video in self.videos:
            cats = get_video_categories(video)
            self.categories.update(cats)
        self.categories = sorted(list(self.categories))
        
        print(f"✓ Loaded {len(self.videos)} videos")
        print(f"✓ Categories: {', '.join(self.categories)}")
        
    def save(self, backup=True):
        """Save configuration to JSON."""
        if backup:
            backup_path = self.config_path + '.backup'
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"✓ Backup saved to {backup_path}")
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved to {self.config_path}")
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
            cat_str = format_categories_display(cats)
            
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
                     categories: List[str]):
        """Assign categories to videos (NO WEIGHTS - simple list)."""
        count = 0
        for idx in video_indices:
            if 0 <= idx < len(self.videos):
                self.videos[idx]['categories'] = categories
                count += 1
        
        self.modified = True
        print(f"✓ Assigned {count} videos to categories: {categories}")
    
    def interactive_select_videos(self, initial_filter: Optional[str] = None):
        """
        Interactive video selection with simple terminal interface.
        Users can toggle videos with their ID and confirm with 'done'.
        
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
        
        selected = set()
        
        print("\n" + "="*80)
        print("INTERACTIVE VIDEO SELECTION")
        print("="*80)
        print("Commands:")
        print("  [ID]      - Toggle video selection (e.g., '5' or '5,7,9')")
        print("  all       - Select all videos")
        print("  none      - Deselect all videos")
        print("  show      - Show current selection")
        print("  done      - Confirm selection")
        print("  cancel    - Cancel and return")
        print("="*80)
        
        while True:
            # Show videos with selection status
            print(f"\n{len(videos)} videos available, {len(selected)} selected")
            print(f"\n{'Sel':<5} {'ID':<6} {'Name':<50} {'Categories':<30}")
            print("-" * 95)
            
            # Show first 20 and selected videos
            shown = 0
            for idx, (i, video) in enumerate(videos):
                if shown < 20 or i in selected:
                    sel_marker = "[X]" if i in selected else "[ ]"
                    name = video['name'][:48]
                    cats = video.get('categories', [])
                    cat_str = format_categories_display(cats)[:28] if cats else ""
                    print(f"{sel_marker:<5} {i:<6} {name:<50} {cat_str:<30}")
                    shown += 1
            
            if len(videos) > 20:
                remaining = len(videos) - 20
                print(f"... and {remaining} more (use filter or select by ID)")
            
            # Get command
            cmd = input(f"\nCommand (selected: {len(selected)}): ").strip().lower()
            
            if not cmd:
                continue
            
            if cmd == 'done':
                if not selected:
                    print("No videos selected.")
                    continue
                return list(selected)
            
            elif cmd == 'cancel':
                return None
            
            elif cmd == 'all':
                selected = {i for i, v in videos}
                print(f"✓ Selected all {len(selected)} videos")
            
            elif cmd == 'none':
                selected.clear()
                print("✓ Cleared selection")
            
            elif cmd == 'show':
                if not selected:
                    print("No videos selected.")
                else:
                    print(f"\nSelected {len(selected)} videos:")
                    for i, video in enumerate(self.videos):
                        if i in selected:
                            print(f"  {i:<6} {video['name']}")
            
            else:
                # Try to parse as ID(s)
                try:
                    # Support comma-separated IDs
                    ids = [int(x.strip()) for x in cmd.split(',')]
                    for vid_id in ids:
                        # Validate ID
                        if vid_id < 0 or vid_id >= len(self.videos):
                            print(f"❌ Invalid ID: {vid_id}")
                            continue
                        
                        # Toggle selection
                        if vid_id in selected:
                            selected.remove(vid_id)
                            print(f"  Deselected: {self.videos[vid_id]['name']}")
                        else:
                            selected.add(vid_id)
                            print(f"  Selected: {self.videos[vid_id]['name']}")
                
                except ValueError:
                    print(f"❌ Invalid command: {cmd}")
                    print("   Use ID numbers, 'all', 'none', 'show', 'done', or 'cancel'")
    
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
            cats = video.get('categories', {})
            if not cats:
                unassigned += 1
            else:
                for cat in cats.keys():
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
        # Find config file
        config_path = Path(__file__).parent / 'generator_config.json'
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            print("Please run from dataset_generator_v2 directory")
            sys.exit(1)
        
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
                    weights = get_category_weights(manager)
                    if weights:
                        manager.assign_videos(selected_ids, weights)
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
                
                weights = get_category_weights(manager)
                if weights:
                    ids = [i for i, v in videos]
                    manager.assign_videos(ids, weights)
            
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
