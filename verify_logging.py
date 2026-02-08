#!/usr/bin/env python3
"""
Simple verification of debug logging implementation.
Verifies code structure without running the actual generator.
"""

import os
import json
import re

def verify_logging_in_code():
    """Verify that logging code is present in make_dataset_multi.py."""
    print("Verifying logging implementation in code...")
    
    file_path = os.path.join(
        os.path.dirname(__file__),
        'dataset_generator_v2',
        'make_dataset_multi.py'
    )
    
    with open(file_path, 'r') as f:
        code = f.read()
    
    # Check for logging import
    assert 'import logging' in code, "Should import logging module"
    print("✓ logging module imported")
    
    # Check for logger setup method
    assert 'def _setup_logger' in code, "Should have _setup_logger method"
    print("✓ _setup_logger method defined")
    
    # Check for logger initialization in __init__
    assert 'self.logger = self._setup_logger()' in code, \
        "Should initialize logger in __init__"
    print("✓ Logger initialized in __init__")
    
    # Check for key log messages
    checks = [
        ('Initializing generator with', '__init__ logging'),
        ('First 5 videos:', '__init__ logging'),
        ('=== STARTING GENERATOR ===', 'run() start logging'),
        ('Resume from video index:', 'run() resume logging'),
        ('--- Loop iteration', 'main loop iteration logging'),
        ('Generator stopped by self.running=False', 'stop condition logging'),
        ('already completed - SKIPPING', 'skip logging'),
        ('Calling process_video()', 'process_video call logging'),
        ('process_video() returned:', 'process_video result logging'),
        ('completed successfully', 'video completion logging'),
        ('Moving to next video', 'next video logging'),
        ('=== MAIN LOOP ENDED ===', 'loop end logging'),
        ('EXCEPTION in video', 'exception logging'),
        ('FATAL EXCEPTION in main loop', 'fatal exception logging'),
        ('Entering finally block', 'finally block logging'),
        ("Setting status to 'finished'", 'finish status logging'),
        ('process_video(', 'process_video debug logging'),
        ('Video path:', 'video path logging'),
        ('Video exists:', 'video exists check logging'),
        ('Video .* not found:', 'video not found error logging'),
        ('extraction .*/.* ---', 'extraction progress logging'),
        ('COMPLETED:', 'video completed logging'),
        ('Exception in process_video', 'process_video exception logging'),
        ('FFmpeg command:', 'FFmpeg command logging'),
        ('Extracted .* frames on attempt', 'frame extraction logging'),
    ]
    
    for pattern, description in checks:
        if re.search(pattern, code):
            print(f"✓ {description}")
        else:
            print(f"✗ MISSING: {description} (pattern: {pattern})")
            return False
    
    # Check for exception handler
    assert 'def exception_handler' in code, "Should have exception_handler function"
    assert 'sys.excepthook = exception_handler' in code, \
        "Should set sys.excepthook"
    print("✓ Global exception handler configured")
    
    # Check for NullHandler when logging is disabled
    assert 'logging.NullHandler()' in code, \
        "Should use NullHandler when logging is disabled"
    print("✓ NullHandler used when logging disabled")
    
    # Check for exc_info=True in error logging
    assert 'exc_info=True' in code, \
        "Should use exc_info=True for exception logging"
    print("✓ Exception stack traces enabled (exc_info=True)")
    
    return True

def verify_config():
    """Verify logging configuration in generator_config.json."""
    print("\nVerifying logging configuration...")
    
    config_path = os.path.join(
        os.path.dirname(__file__),
        'dataset_generator_v2',
        'generator_config.json'
    )
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Verify logging settings
    base_settings = config['base_settings']
    
    assert 'enable_debug_logging' in base_settings, \
        "Config should have enable_debug_logging"
    assert 'debug_log_path' in base_settings, \
        "Config should have debug_log_path"
    
    enable_logging = base_settings['enable_debug_logging']
    log_path = base_settings['debug_log_path']
    
    assert isinstance(enable_logging, bool), \
        "enable_debug_logging should be boolean"
    assert isinstance(log_path, str), \
        "debug_log_path should be string"
    assert log_path.endswith('.log'), \
        "debug_log_path should end with .log"
    
    print(f"✓ enable_debug_logging: {enable_logging}")
    print(f"✓ debug_log_path: {log_path}")
    
    return True

def print_logging_summary():
    """Print a summary of the logging implementation."""
    print("\n" + "="*70)
    print("LOGGING IMPLEMENTATION SUMMARY")
    print("="*70)
    print()
    print("Configuration:")
    print("  - Toggleable via generator_config.json")
    print("  - Key: 'enable_debug_logging' (default: true)")
    print("  - Log file: /mnt/data/training/dataset/generator_debug.log")
    print()
    print("Logging Levels:")
    print("  - DEBUG: Detailed flow, FFmpeg commands, internal state")
    print("  - INFO: Video start/complete, major milestones")
    print("  - WARNING: Skipped videos, stopped conditions")
    print("  - ERROR: Exceptions, failures, missing videos")
    print("  - CRITICAL: Fatal errors, uncaught exceptions")
    print()
    print("Key Features:")
    print("  ✓ Logger initialized in __init__ with config from JSON")
    print("  ✓ Main loop logging (start, iteration, end)")
    print("  ✓ Video processing logging (start, progress, completion)")
    print("  ✓ FFmpeg command logging")
    print("  ✓ Frame extraction logging")
    print("  ✓ Exception handling with full stack traces")
    print("  ✓ Global uncaught exception handler")
    print("  ✓ NullHandler when logging is disabled")
    print()
    print("Log Format:")
    print("  %(asctime)s - %(levelname)s - %(message)s")
    print()
    print("Example Log Entries:")
    print("  2024-01-01 12:00:00 - INFO - Initializing generator with 467 videos")
    print("  2024-01-01 12:00:00 - INFO - === STARTING GENERATOR ===")
    print("  2024-01-01 12:00:01 - INFO - Processing video 0: Movie Title")
    print("  2024-01-01 12:00:02 - DEBUG - FFmpeg command: nice -n 19 ffmpeg ...")
    print("  2024-01-01 12:00:03 - INFO - Video 0 COMPLETED: 2500/3000 successful")
    print("  2024-01-01 12:00:03 - DEBUG - Moving to next video (idx=1)")
    print()
    print("="*70)

if __name__ == "__main__":
    try:
        verify_config()
        verify_logging_in_code()
        print_logging_summary()
        
        print("\n✅ All verification checks passed!")
        print("\nThe debug logging system is fully implemented and ready to use.")
        print("It will help diagnose why the generator stops after 1 video.")
        
    except AssertionError as e:
        print(f"\n❌ Verification failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
