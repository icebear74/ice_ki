#!/usr/bin/env python3
"""
Web UI Demo - Demonstrates the web monitoring interface
Runs a mock training loop with web UI updates
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsr_plus_plus.systems.web_ui import WebInterface


def demo_web_ui():
    """Demonstrate web UI with mock training data"""
    print("\n" + "="*70)
    print("VSR++ Web UI Demo")
    print("="*70)
    
    # Start web interface
    web_ui = WebInterface(port_number=5051)
    
    print(f"\n✓ Web UI started on http://localhost:5051")
    print(f"✓ Open your browser to view the dashboard")
    print(f"✓ Press Ctrl+C to stop\n")
    
    # Simulate training loop
    iteration = 0
    base_loss = 0.05
    
    try:
        while True:
            iteration += 1
            
            # Simulate decreasing loss
            total_loss = base_loss * (1.0 - min(iteration / 10000.0, 0.9))
            
            # Simulate varying learning rate
            learn_rate = 0.0001 * max(0.1, 1.0 - iteration / 20000.0)
            
            # Calculate mock ETA
            time_remaining = f"{int(100 - iteration/100):02d}:30:00"
            
            # Update web UI
            web_ui.update(
                iteration=iteration,
                total_loss=total_loss,
                learn_rate=learn_rate,
                time_remaining=time_remaining,
                iter_speed=0.42,  # ~2.4 it/s
                gpu_memory=7.8 + (iteration % 10) * 0.02,
                best_score=min(0.92, 0.5 + iteration / 10000.0),
                is_validating=(iteration % 500 == 0)
            )
            
            # Check for commands
            cmd = web_ui.check_commands()
            if cmd == 'validate':
                print(f"  [Step {iteration}] Web UI triggered validation!")
                # Simulate validation
                web_ui.update(
                    iteration=iteration,
                    total_loss=total_loss,
                    learn_rate=learn_rate,
                    time_remaining=time_remaining,
                    iter_speed=0.42,
                    gpu_memory=7.8,
                    best_score=min(0.92, 0.5 + iteration / 10000.0),
                    is_validating=True
                )
                time.sleep(2)  # Simulate validation time
            
            # Print progress every 100 iterations
            if iteration % 100 == 0:
                print(f"  Iteration {iteration:5d} | Loss: {total_loss:.6f} | LR: {learn_rate:.6f}")
            
            time.sleep(0.1)  # Simulate training time
            
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user")
        web_ui.shutdown()
        print("✓ Web UI shutdown complete")


if __name__ == '__main__':
    demo_web_ui()
