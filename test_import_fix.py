#!/usr/bin/env python3
"""
Test that the imports in make_dataset_multi.py work correctly.
"""

import sys
import os

# Add dataset_generator_v2 to path (simulating how the script runs)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dataset_generator_v2'))

def test_ui_terminal_import():
    """Test that ui_terminal can be imported as absolute import"""
    try:
        from utils.ui_terminal import clear_and_home, hide_cursor, show_cursor
        print("✓ ui_terminal imports work (clear_and_home, hide_cursor, show_cursor)")
        return True
    except ImportError as e:
        print(f"✗ ui_terminal import failed: {e}")
        return False

def test_ui_display_import():
    """Test that ui_display can be imported as absolute import"""
    try:
        from utils.ui_display import draw_dataset_generator_ui
        print("✓ ui_display import works (draw_dataset_generator_ui)")
        return True
    except ImportError as e:
        print(f"✗ ui_display import failed: {e}")
        return False

def test_make_dataset_imports():
    """Test that make_dataset_multi.py can be imported"""
    try:
        # This will execute the module-level imports
        import make_dataset_multi
        print("✓ make_dataset_multi.py imports successfully")
        return True
    except ImportError as e:
        print(f"✗ make_dataset_multi.py import failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing import fixes for make_dataset_multi.py...\n")
    
    results = []
    results.append(test_ui_terminal_import())
    results.append(test_ui_display_import())
    results.append(test_make_dataset_imports())
    
    print(f"\n{'='*50}")
    if all(results):
        print("✅ All import tests passed!")
        print("The ImportError is fixed - absolute imports work correctly.")
    else:
        print("❌ Some import tests failed")
        sys.exit(1)
