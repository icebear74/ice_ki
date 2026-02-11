#!/usr/bin/env python3
"""
Interactive selector with curses - Space to toggle, Arrow keys to navigate
Similar to fzf or other modern CLI tools
"""

import curses
from typing import List, Dict, Any, Callable, Optional


class InteractiveSelector:
    """Curses-based interactive selector with checkboxes."""
    
    def __init__(self, items: List[Any], 
                 title: str = "Select items",
                 get_label: Callable[[Any], str] = str,
                 get_details: Callable[[Any], str] = lambda x: "",
                 preselected: Optional[List[int]] = None):
        """
        Args:
            items: List of items to select from
            title: Title shown at top
            get_label: Function to get display label from item
            get_details: Function to get additional details from item
            preselected: List of indices that should start selected
        """
        self.items = items
        self.title = title
        self.get_label = get_label
        self.get_details = get_details
        self.selected = set(preselected or [])
        self.current = 0
        self.scroll_offset = 0
        
    def run(self) -> Optional[List[int]]:
        """Run the interactive selector. Returns list of selected indices or None if cancelled."""
        if not self.items:
            return []
        
        try:
            result = curses.wrapper(self._curses_main)
            return result
        except KeyboardInterrupt:
            return None
    
    def _curses_main(self, stdscr):
        """Main curses loop."""
        curses.curs_set(0)  # Hide cursor
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Highlight
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Selected
        curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)   # Title
        
        while True:
            stdscr.clear()
            height, width = stdscr.getmaxyx()
            
            # Calculate visible area
            header_lines = 3
            footer_lines = 2
            visible_lines = height - header_lines - footer_lines
            
            # Adjust scroll offset
            if self.current < self.scroll_offset:
                self.scroll_offset = self.current
            elif self.current >= self.scroll_offset + visible_lines:
                self.scroll_offset = self.current - visible_lines + 1
            
            # Draw title
            title_text = f"  {self.title}  "
            stdscr.addstr(0, 0, title_text, curses.color_pair(3) | curses.A_BOLD)
            
            # Draw counter
            counter_text = f" {len(self.selected)} / {len(self.items)} selected "
            stdscr.addstr(1, 0, counter_text)
            
            # Draw items
            for i in range(visible_lines):
                idx = i + self.scroll_offset
                if idx >= len(self.items):
                    break
                
                line = header_lines + i
                if line >= height - footer_lines:
                    break
                
                # Checkbox
                checkbox = "[✓]" if idx in self.selected else "[ ]"
                
                # Label
                label = self.get_label(self.items[idx])
                
                # Details
                details = self.get_details(self.items[idx])
                
                # Combine
                text = f"{checkbox} {label}"
                if details:
                    # Truncate if needed
                    max_label_len = width - len(checkbox) - len(details) - 5
                    if len(label) > max_label_len:
                        label = label[:max_label_len-3] + "..."
                        text = f"{checkbox} {label}"
                    text = f"{text:<{width-len(details)-2}}{details}"
                else:
                    # Truncate to fit
                    if len(text) > width - 2:
                        text = text[:width-5] + "..."
                
                # Color
                attr = 0
                if idx == self.current:
                    attr = curses.color_pair(1)  # Highlight current
                if idx in self.selected:
                    attr |= curses.color_pair(2)  # Green for selected
                
                try:
                    stdscr.addstr(line, 1, text[:width-2], attr)
                except curses.error:
                    pass  # Ignore if can't write at edge
            
            # Draw footer/help
            help_line = height - 2
            help_text = "Space: toggle | ↑↓: navigate | a: all | n: none | Enter: done | Esc/q: cancel"
            try:
                stdscr.addstr(help_line, 0, help_text[:width-1])
            except curses.error:
                pass
            
            stdscr.refresh()
            
            # Handle input
            key = stdscr.getch()
            
            if key == ord(' '):  # Space - toggle
                if self.current in self.selected:
                    self.selected.remove(self.current)
                else:
                    self.selected.add(self.current)
            
            elif key == curses.KEY_UP or key == ord('k'):
                self.current = max(0, self.current - 1)
            
            elif key == curses.KEY_DOWN or key == ord('j'):
                self.current = min(len(self.items) - 1, self.current + 1)
            
            elif key == curses.KEY_PPAGE:  # Page up
                self.current = max(0, self.current - visible_lines)
            
            elif key == curses.KEY_NPAGE:  # Page down
                self.current = min(len(self.items) - 1, self.current + visible_lines)
            
            elif key == ord('g'):  # Go to top
                self.current = 0
            
            elif key == ord('G'):  # Go to bottom
                self.current = len(self.items) - 1
            
            elif key == ord('a'):  # Select all
                self.selected = set(range(len(self.items)))
            
            elif key == ord('n'):  # Select none
                self.selected.clear()
            
            elif key == 10 or key == 13:  # Enter - confirm
                return sorted(list(self.selected))
            
            elif key == 27 or key == ord('q'):  # Esc/q - cancel
                return None


def select_items(items: List[Any], 
                title: str = "Select items",
                get_label: Callable[[Any], str] = str,
                get_details: Callable[[Any], str] = lambda x: "",
                preselected: Optional[List[int]] = None) -> Optional[List[int]]:
    """
    Convenience function for interactive selection.
    
    Returns:
        List of selected indices, or None if cancelled
    """
    selector = InteractiveSelector(items, title, get_label, get_details, preselected)
    return selector.run()


def select_categories(available_categories: List[str], 
                     current_categories: Optional[List[str]] = None) -> Optional[List[str]]:
    """
    Select categories interactively.
    
    Args:
        available_categories: List of available category names
        current_categories: Currently selected categories (for pre-selection)
    
    Returns:
        List of selected category names, or None if cancelled
    """
    preselected = []
    if current_categories:
        for i, cat in enumerate(available_categories):
            if cat in current_categories:
                preselected.append(i)
    
    indices = select_items(
        items=available_categories,
        title="Select Categories (Space to toggle, Enter to confirm)",
        get_label=lambda x: x,
        preselected=preselected
    )
    
    if indices is None:
        return None
    
    return [available_categories[i] for i in indices]


if __name__ == "__main__":
    # Test
    test_items = [
        {"name": "Item 1", "value": "A"},
        {"name": "Item 2", "value": "B"},
        {"name": "Item 3", "value": "C"},
        {"name": "Item 4", "value": "D"},
        {"name": "Item 5", "value": "E"},
    ]
    
    selected = select_items(
        items=test_items,
        title="Test Selection",
        get_label=lambda x: x["name"],
        get_details=lambda x: f"({x['value']})",
        preselected=[0, 2]
    )
    
    if selected is not None:
        print(f"\nSelected indices: {selected}")
        print("Selected items:")
        for idx in selected:
            print(f"  - {test_items[idx]}")
    else:
        print("\nCancelled")
