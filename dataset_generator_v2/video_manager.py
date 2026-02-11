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
from pathlib import Path
from typing import Dict, List, Optional
import re


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
        
        # Extract unique categories
        self.categories = set()
        for video in self.videos:
            for cat in video.get('categories', {}).keys():
                self.categories.add(cat)
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
                   show_unassigned: bool = False):
        """List videos with optional filtering."""
        
        filtered = []
        for i, video in enumerate(self.videos):
            # Filter by pattern
            if filter_pattern:
                if not re.search(filter_pattern, video['name'], re.IGNORECASE):
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
            cats = video.get('categories', {})
            cat_str = ', '.join([f"{k}:{v:.2f}" for k, v in cats.items()])
            if not cat_str:
                cat_str = "⚠️  <WILL BE SKIPPED - no categories>"
            
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
                     category_weights: Dict[str, float]):
        """Assign categories to videos."""
        # Normalize weights
        total = sum(category_weights.values())
        if total == 0:
            print("❌ Error: Total weight cannot be zero")
            return
        
        normalized = {k: v/total for k, v in category_weights.items()}
        
        count = 0
        for idx in video_indices:
            if 0 <= idx < len(self.videos):
                self.videos[idx]['categories'] = normalized
                count += 1
        
        self.modified = True
        print(f"✓ Assigned {count} videos to categories: {normalized}")
    
    def remove_from_category(self, video_indices: List[int], category: str):
        """Remove videos from a specific category."""
        count = 0
        for idx in video_indices:
            if 0 <= idx < len(self.videos):
                cats = self.videos[idx].get('categories', {})
                if category in cats:
                    del cats[category]
                    count += 1
                    # Renormalize remaining categories
                    total = sum(cats.values())
                    if total > 0:
                        cats = {k: v/total for k, v in cats.items()}
                        self.videos[idx]['categories'] = cats
        
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
    print("6. Multi-assign by pattern")
    print("7. Remove from category")
    print("8. Reset all assignments")
    print("9. Show statistics")
    print("10. Edit category targets")
    print("s. Save changes")
    print("q. Quit")
    print("="*60)


def get_category_weights(manager: VideoManager) -> Dict[str, float]:
    """Interactive input for category weights."""
    print(f"\nAvailable categories: {', '.join(manager.categories)}")
    print("Enter weights for each category (0 to skip):")
    
    weights = {}
    for cat in manager.categories:
        while True:
            val = input(f"  {cat}: ").strip()
            if not val:
                continue
            try:
                weight = float(val)
                if weight > 0:
                    weights[cat] = weight
                break
            except ValueError:
                print("    Invalid number, try again")
    
    if not weights:
        print("No categories assigned")
        return {}
    
    # Show preview
    total = sum(weights.values())
    print("\nNormalized weights:")
    for cat, weight in weights.items():
        print(f"  {cat}: {weight/total:.2f}")
    
    return weights


def main():
    # Find config file
    config_path = Path(__file__).parent / 'generator_config.json'
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        print("Please run from dataset_generator_v2 directory")
        sys.exit(1)
    
    manager = VideoManager(str(config_path))
    manager.load()
    
    while True:
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
            # Assign single/multiple videos
            ids_str = input("Video ID(s) (comma-separated): ").strip()
            try:
                ids = [int(x.strip()) for x in ids_str.split(',')]
                weights = get_category_weights(manager)
                if weights:
                    manager.assign_videos(ids, weights)
            except ValueError:
                print("❌ Invalid ID format")
        
        elif choice == '6':
            # Multi-assign by pattern
            pattern = input("Search pattern (regex, e.g., 'Star Trek.*'): ").strip()
            if not pattern:
                continue
            
            videos = manager.list_videos(filter_pattern=pattern)
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
        
        elif choice == '7':
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
        
        elif choice == '8':
            # Reset all
            manager.reset_all()
        
        elif choice == '9':
            # Statistics
            manager.show_statistics()
        
        elif choice == '10':
            # Edit category targets
            manager.edit_category_targets()
        
        else:
            print("Invalid choice")


if __name__ == '__main__':
    main()
