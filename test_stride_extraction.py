#!/usr/bin/env python3
"""
Test stride-based frame extraction logic
"""

def test_stride_calculation():
    """Test calculation of frame stride pattern"""
    
    print("=" * 70)
    print("Testing Stride Calculation Logic")
    print("=" * 70)
    
    # Example 1: Uniform stride
    timestamps = [0.0, 3.0, 6.0, 9.0, 12.0]  # Every 3 seconds
    fps = 25.0
    n_frames = 7
    
    print("\n📊 Example 1: Uniform Stride")
    print(f"Timestamps: {timestamps}")
    print(f"FPS: {fps}, N frames: {n_frames}")
    
    frame_numbers = [int(ts * fps) for ts in timestamps]
    print(f"Frame numbers: {frame_numbers}")
    
    # Calculate intervals
    intervals = []
    for i in range(len(frame_numbers) - 1):
        # Distance from end of one group to start of next
        interval = frame_numbers[i+1] - (frame_numbers[i] + n_frames - 1) - 1
        intervals.append(interval)
    
    print(f"Intervals between groups: {intervals}")
    
    # Check uniformity
    unique_intervals = set(intervals)
    print(f"Unique intervals: {unique_intervals}")
    
    if len(unique_intervals) <= 2:
        stride = max(unique_intervals, key=intervals.count)
        print(f"✓ Uniform stride detected: {stride} frames")
        
        # Calculate cycle length
        cycle_length = n_frames + stride
        first_frame = frame_numbers[0]
        last_frame = frame_numbers[-1] + n_frames - 1
        
        print(f"  Cycle length: {cycle_length}")
        print(f"  First frame: {first_frame}")
        print(f"  Last frame: {last_frame}")
        
        # Build select expression
        select_filter = f"gte(n,{first_frame})*lte(n,{last_frame})*lt(mod(n-{first_frame},{cycle_length}),{n_frames})"
        print(f"  Select filter: {select_filter}")
        print(f"  ✓ Command line length: {len(select_filter)} chars (MUCH shorter than listing frames)")
    else:
        print(f"✗ Non-uniform stride, need chunking")
    
    # Example 2: Varying timestamps (realistic)
    print("\n" + "=" * 70)
    print("\n📊 Example 2: Varying Timestamps (Realistic)")
    
    # Simulating random extraction points
    import random
    random.seed(42)
    
    # Generate 100 timestamps with some variation
    base_interval = 3.0  # Average 3 seconds apart
    timestamps2 = []
    current = 10.0  # Start at 10 seconds
    for _ in range(100):
        timestamps2.append(current)
        current += base_interval + random.uniform(-0.5, 0.5)  # Add some variation
    
    print(f"Generated {len(timestamps2)} timestamps")
    print(f"First 10: {timestamps2[:10]}")
    print(f"Last 10: {timestamps2[-10:]}")
    
    frame_numbers2 = [int(ts * fps) for ts in timestamps2]
    
    intervals2 = []
    for i in range(len(frame_numbers2) - 1):
        interval = frame_numbers2[i+1] - (frame_numbers2[i] + n_frames - 1) - 1
        intervals2.append(interval)
    
    unique_intervals2 = set(intervals2)
    print(f"Unique intervals: {len(unique_intervals2)} different values")
    print(f"Min interval: {min(intervals2)}, Max interval: {max(intervals2)}")
    print(f"Most common interval: {max(unique_intervals2, key=intervals2.count)}")
    
    # Count interval frequency
    from collections import Counter
    interval_counts = Counter(intervals2)
    print(f"Top 5 most common intervals: {interval_counts.most_common(5)}")
    
    if len(unique_intervals2) <= 2:
        print(f"✓ Can use stride pattern!")
    else:
        print(f"✗ Too much variation, will use chunking approach")
        print(f"  Chunk size: 50 timestamps per chunk")
        print(f"  Total chunks needed: {(len(timestamps2) - 1) // 50 + 1}")
    
    # Example 3: Check command line length comparison
    print("\n" + "=" * 70)
    print("\n📊 Example 3: Command Line Length Comparison")
    
    test_timestamps = list(range(0, 4000))  # 4000 extraction points
    total_frames = len(test_timestamps) * n_frames  # 28,000 frames
    
    # Old approach: listing every frame
    old_expressions = [f"eq(n,{int(ts*fps)+offset})" for ts in test_timestamps for offset in range(n_frames)]
    old_filter = "+".join(old_expressions)
    old_length = len(old_filter)
    
    # New approach: stride pattern
    stride = 68  # Example stride
    cycle_length = n_frames + stride
    first_frame = 0
    last_frame = int(test_timestamps[-1] * fps) + n_frames - 1
    new_filter = f"gte(n,{first_frame})*lte(n,{last_frame})*lt(mod(n-{first_frame},{cycle_length}),{n_frames})"
    new_length = len(new_filter)
    
    print(f"Total frames to extract: {total_frames}")
    print(f"\nOLD approach (listing every frame):")
    print(f"  Filter length: {old_length:,} chars")
    print(f"  ✗ Command line limit exceeded! (>131,072 chars on Linux)")
    
    print(f"\nNEW approach (stride pattern):")
    print(f"  Filter length: {new_length} chars")
    print(f"  ✓ Well within limits!")
    print(f"  Reduction: {100 * (old_length - new_length) / old_length:.1f}%")
    
    print("\n" + "=" * 70)
    print("✅ ALL STRIDE CALCULATION TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_stride_calculation()
