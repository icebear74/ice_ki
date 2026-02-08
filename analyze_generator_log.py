#!/usr/bin/env python3
"""
Log analyzer for dataset generator debug logs.
Helps identify why the generator stops after first video.
"""

import sys
import re
from datetime import datetime

def analyze_log(log_file_path):
    """Analyze the debug log to find why generator stopped."""
    
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ Log file not found: {log_file_path}")
        print("\nGenerator has probably not run yet.")
        return
    except Exception as e:
        print(f"❌ Error reading log file: {e}")
        return
    
    if not lines:
        print("❌ Log file is empty!")
        return
    
    print("="*80)
    print("DATASET GENERATOR DEBUG LOG ANALYSIS")
    print("="*80)
    print(f"\nLog file: {log_file_path}")
    print(f"Total lines: {len(lines)}")
    
    # Find key events
    start_time = None
    videos_started = []
    videos_completed = []
    loop_iterations = []
    exceptions = []
    warnings = []
    errors = []
    last_line = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        last_line = line
        
        # Parse timestamp
        match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
        if match and start_time is None:
            start_time = match.group(1)
        
        # Find key events
        if "=== STARTING GENERATOR ===" in line:
            print(f"\n✅ Generator started: {line}")
        
        if "--- Loop iteration" in line:
            match = re.search(r'Loop iteration (\d+) / (\d+)', line)
            if match:
                idx = int(match.group(1))
                total = int(match.group(2))
                loop_iterations.append(idx)
        
        if "Processing video" in line and ":" in line:
            match = re.search(r'Processing video (\d+): (.+)$', line)
            if match:
                idx = int(match.group(1))
                name = match.group(2)
                videos_started.append((idx, name))
        
        if "COMPLETED:" in line:
            match = re.search(r'Video (\d+) COMPLETED', line)
            if match:
                idx = int(match.group(1))
                videos_completed.append(idx)
        
        if "EXCEPTION in video" in line or "Exception in process_video" in line:
            exceptions.append((i+1, line))
        
        if "WARNING" in line or "stopped by self.running=False" in line:
            warnings.append((i+1, line))
        
        if "ERROR" in line or "FATAL EXCEPTION" in line:
            errors.append((i+1, line))
    
    # Analysis
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    
    print(f"\n📊 Statistics:")
    print(f"   Loop iterations detected: {len(loop_iterations)}")
    print(f"   Videos started: {len(videos_started)}")
    print(f"   Videos completed: {len(videos_completed)}")
    print(f"   Exceptions: {len(exceptions)}")
    print(f"   Warnings: {len(warnings)}")
    print(f"   Errors: {len(errors)}")
    
    if videos_started:
        print(f"\n🎬 Videos Started:")
        for idx, name in videos_started[:10]:
            completed = "✅" if idx in videos_completed else "❌"
            print(f"   {completed} Video {idx}: {name[:60]}")
        if len(videos_started) > 10:
            print(f"   ... and {len(videos_started) - 10} more")
    
    if videos_completed:
        print(f"\n✅ Videos Completed:")
        for idx in videos_completed[:10]:
            print(f"   Video {idx}")
        if len(videos_completed) > 10:
            print(f"   ... and {len(videos_completed) - 10} more")
    
    if loop_iterations:
        print(f"\n🔄 Loop Iterations:")
        print(f"   Started at: {min(loop_iterations)}")
        print(f"   Last iteration: {max(loop_iterations)}")
        print(f"   Total iterations: {len(loop_iterations)}")
        
        # Check for gaps
        if len(loop_iterations) > 1:
            expected = list(range(min(loop_iterations), max(loop_iterations) + 1))
            missing = set(expected) - set(loop_iterations)
            if missing:
                print(f"   ⚠️  Missing iterations: {sorted(missing)}")
    
    # Diagnose the problem
    print(f"\n{'='*80}")
    print("DIAGNOSIS")
    print("="*80)
    
    if len(videos_completed) == 0:
        print("\n❌ PROBLEM: No videos completed!")
        print("   Likely crashed before finishing first video.")
    
    elif len(videos_completed) == 1 and len(videos_started) == 1:
        print("\n❌ PROBLEM: Generator stopped after first video!")
        print("   Video 0 completed successfully but loop didn't continue.")
        
        if loop_iterations and max(loop_iterations) == 0:
            print("\n   Possible cause: Loop only iterated once (idx=0)")
            print("   Check: Did loop condition fail? Was range() correct?")
        
        elif len(loop_iterations) >= 2:
            print(f"\n   Loop continued to iteration {max(loop_iterations)}")
            print("   But no second video was processed.")
            print("   Check: Were videos 1+ already marked as completed?")
    
    if exceptions:
        print(f"\n⚠️  EXCEPTIONS FOUND ({len(exceptions)}):")
        for line_num, exc in exceptions[:5]:
            print(f"   Line {line_num}: {exc[:100]}")
        if len(exceptions) > 5:
            print(f"   ... and {len(exceptions) - 5} more")
    
    if warnings:
        print(f"\n⚠️  WARNINGS FOUND ({len(warnings)}):")
        for line_num, warn in warnings[:5]:
            print(f"   Line {line_num}: {warn[:100]}")
        if len(warnings) > 5:
            print(f"   ... and {len(warnings) - 5} more")
    
    if errors:
        print(f"\n❌ ERRORS FOUND ({len(errors)}):")
        for line_num, err in errors[:5]:
            print(f"   Line {line_num}: {err[:100]}")
        if len(errors) > 5:
            print(f"   ... and {len(errors) - 5} more")
    
    print(f"\n{'='*80}")
    print("LAST LOG ENTRY")
    print("="*80)
    if last_line:
        print(f"\n{last_line}")
    
    print(f"\n{'='*80}")
    print("\nTo see full log: cat {log_file_path}")
    print("To see last 50 lines: tail -50 {log_file_path}")
    print("To see exceptions only: grep -i exception {log_file_path}")
    print("="*80)

if __name__ == "__main__":
    log_path = "/mnt/data/training/dataset/generator_debug.log"
    
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    
    analyze_log(log_path)
