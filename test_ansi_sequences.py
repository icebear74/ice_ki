#!/usr/bin/env python3
"""
Test ANSI escape sequences for display clearing and positioning.
"""

import sys
import time

def test_ansi_sequences():
    """Test different ANSI escape sequences."""
    print("\nTesting ANSI escape sequences for display control...")
    
    # Test 1: Home cursor only (old version - shows the problem)
    print("\n=== Test 1: Home cursor only (\\033[H) ===")
    print("Line 1: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("Line 2: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("Line 3: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    time.sleep(1)
    print('\033[H', end='', flush=True)  # Home only
    print("Short")
    print("Text")
    print("\n^ Notice old 'XXX' text remains visible!")
    time.sleep(2)
    
    # Test 2: Home + clear to end (new version)
    print("\n=== Test 2: Home + Clear to end (\\033[H\\033[J) ===")
    print("Line 1: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("Line 2: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("Line 3: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    time.sleep(1)
    print('\033[H\033[J', end='', flush=True)  # Home + clear to end
    print("Short")
    print("Text")
    print("\n^ Old 'XXX' text is gone!")
    time.sleep(2)
    
    # Test 3: Full clear (most flickering)
    print("\n=== Test 3: Full clear (\\033[2J\\033[H) ===")
    for i in range(3):
        print('\033[2J\033[H', end='', flush=True)
        print(f"Update {i+1}")
        print("Some content...")
        time.sleep(0.5)
    print("\n^ This flickers more!")
    
    print("\n✅ ANSI sequence tests complete!")
    print("\nRecommendation:")
    print("  - Use \\033[H\\033[J for updates (home + clear to end)")
    print("  - Use \\033[2J\\033[H only once at start")

def test_display_mode_detection():
    """Test which display mode would be used."""
    print("\n\nTesting display mode detection...")
    
    try:
        from rich.live import Live
        from rich.console import Console
        
        console = Console()
        print("✓ Rich library is available")
        
        # Try to create a Live display
        try:
            test_live = Live("Test", console=console)
            test_live.start()
            print("✓ Live display CAN be created")
            test_live.stop()
            print("  → Will use Live display mode (best, no flickering)")
        except Exception as e:
            print(f"✗ Live display FAILED: {e}")
            print("  → Will use fallback mode with \\033[H\\033[J")
    except ImportError:
        print("✗ Rich library not available")
        print("  → Will use simple text mode")

if __name__ == "__main__":
    test_ansi_sequences()
    test_display_mode_detection()
