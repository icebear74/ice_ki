#!/usr/bin/env python3
"""
Demo script to show the curses-based interactive selector interface.

This demonstrates the new interactive selection that was implemented:
- Arrow keys (↑/↓) or j/k for navigation
- Space bar to toggle selection
- Enter to confirm
- Esc or 'q' to cancel
- 'a' to select all
- 'n' to select none
- 'g' to go to top
- 'G' to go to bottom
- Page Up/Down for fast scrolling

Run this in a real terminal to see it in action!
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interactive_selector import select_items

# Sample video data
videos = [
    {"name": "Venom 2 - Let There Be Carnage", "categories": ["master", "universal"]},
    {"name": "Poltergeist", "categories": ["master", "universal"]},
    {"name": "Zombieland", "categories": ["master", "universal"]},
    {"name": "Hellboy - Call Of Darkness", "categories": ["master", "universal"]},
    {"name": "Hellboy 2 - Die Goldene Armee", "categories": ["master", "universal"]},
    {"name": "Shrek", "categories": ["master", "toon"]},
    {"name": "Apollo 13", "categories": ["master", "universal"]},
    {"name": "Ich Einfach Unverbesserlich 1", "categories": ["master", "toon"]},
    {"name": "Illuminati", "categories": ["master", "universal"]},
    {"name": "Avatar", "categories": ["master", "universal", "space", "toon"]},
    {"name": "Spiderman - No Way Home", "categories": ["master", "universal"]},
    {"name": "Halloween III", "categories": ["master", "universal"]},
    {"name": "ES - Kapitel 2 (IT - Episode 2)", "categories": ["master", "universal"]},
    {"name": "Fast and Furious 6", "categories": ["master", "universal"]},
    {"name": "Der Super Mario Brothers Film", "categories": ["master", "toon"]},
    {"name": "Star Wars Episode V - Das Imperium Schlägt Zurück", "categories": ["master", "space"]},
]

print("=" * 80)
print("INTERACTIVE SELECTOR DEMO")
print("=" * 80)
print()
print("This will launch a curses-based interactive selector.")
print()
print("Controls:")
print("  ↑/↓ or j/k    - Navigate up/down")
print("  Space         - Toggle selection")
print("  Enter         - Confirm selection")
print("  Esc or 'q'    - Cancel")
print("  'a'           - Select all")
print("  'n'           - Select none")
print("  'g'           - Go to top")
print("  'G'           - Go to bottom")
print("  Page Up/Down  - Fast scrolling")
print()
print("Press Enter to continue...")
input()

try:
    selected_indices = select_items(
        items=videos,
        title=f"Select Videos - {len(videos)} available (↑↓ navigate, Space toggle, Enter done)",
        get_label=lambda v: v['name'],
        get_details=lambda v: f"[{videos.index(v)}] {', '.join(v['categories'])}"
    )
    
    print("\n" + "=" * 80)
    if selected_indices is not None:
        print(f"✓ Selected {len(selected_indices)} videos:")
        for idx in selected_indices:
            print(f"  [{idx}] {videos[idx]['name']}")
    else:
        print("❌ Selection cancelled")
    print("=" * 80)
    
except Exception as e:
    print(f"\n⚠️  Error: {e}")
    print("\nNote: This requires a proper terminal environment.")
    print("If running in a non-interactive environment, this won't work.")
    import traceback
    traceback.print_exc()
