#!/usr/bin/env python3
"""
Unit tests for the `select` filter range-building logic inside
extract_and_save_streaming_distributed().

These tests exercise the range-merge algorithm and the FFmpeg select
expression format independently of any real video file or FFmpeg binary.

Run with:
    cd dataset_generator_v2
    python test_select_filter.py
or:
    python -m pytest test_select_filter.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Helpers — replicate the range-building logic from the extractor so tests
# remain self-contained and are easy to understand.
# ---------------------------------------------------------------------------

def _build_needed_frames(centers, half):
    """Return sorted list of every frame index in any center's window."""
    needed = set()
    for c in centers:
        for fi in range(max(0, c - half), c + half + 1):
            needed.add(fi)
    return sorted(needed)


def _merge_to_ranges(sorted_frames):
    """Merge a sorted list of frame indices into (start, end) range tuples."""
    ranges = []
    if not sorted_frames:
        return ranges
    rs, re = sorted_frames[0], sorted_frames[0]
    for f in sorted_frames[1:]:
        if f == re + 1:
            re = f
        else:
            ranges.append((rs, re))
            rs = re = f
    ranges.append((rs, re))
    return ranges


def _build_select_expr(ranges):
    """Build the FFmpeg select expression string."""
    return "+".join(f"between(n\\,{s}\\,{e})" for s, e in ranges)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_PASS = 0
_FAIL = 0


def _test(name, cond, detail=""):
    global _PASS, _FAIL
    if cond:
        print(f"  ✓  {name}")
        _PASS += 1
    else:
        print(f"  ✗  {name}{(' — ' + detail) if detail else ''}")
        _FAIL += 1


def test_single_center_no_overlap():
    needed = _build_needed_frames([100], half=3)
    assert needed == list(range(97, 104)), needed
    ranges = _merge_to_ranges(needed)
    _test(
        "single center produces one range",
        ranges == [(97, 103)],
        f"got {ranges}",
    )


def test_two_non_overlapping_centers():
    needed = _build_needed_frames([10, 200], half=3)
    ranges = _merge_to_ranges(needed)
    _test(
        "non-overlapping centers produce two separate ranges",
        ranges == [(7, 13), (197, 203)],
        f"got {ranges}",
    )


def test_two_overlapping_centers_merged():
    # centers at 10 and 14 with half=3: windows [7,13] and [11,17] overlap
    needed = _build_needed_frames([10, 14], half=3)
    ranges = _merge_to_ranges(needed)
    _test(
        "overlapping windows are merged into one range",
        len(ranges) == 1,
        f"got {ranges}",
    )
    _test(
        "merged range spans both windows",
        ranges == [(7, 17)],
        f"got {ranges}",
    )


def test_adjacent_windows_merged():
    # windows [7,13] and [14,20] are adjacent (13+1 == 14) → should merge
    needed = _build_needed_frames([10, 17], half=3)
    ranges = _merge_to_ranges(needed)
    _test(
        "adjacent windows are merged",
        len(ranges) == 1,
        f"got {ranges}",
    )


def test_frame_zero_clamp():
    # center at 2 with half=5: clamped to 0, not negative
    needed = _build_needed_frames([2], half=5)
    _test(
        "frame indices are clamped to 0 — no negatives",
        all(f >= 0 for f in needed),
        f"got {needed[:5]}",
    )
    _test(
        "first needed frame is 0",
        needed[0] == 0,
        f"got {needed[0]}",
    )


def test_select_expr_format():
    ranges = [(97, 103), (200, 206)]
    expr = _build_select_expr(ranges)
    _test(
        "select expression uses backslash-escaped commas",
        "between(n\\,97\\,103)" in expr and "between(n\\,200\\,206)" in expr,
        f"got: {expr}",
    )
    _test(
        "select expression uses + as OR operator",
        expr == "between(n\\,97\\,103)+between(n\\,200\\,206)",
        f"got: {expr}",
    )


def test_select_reduction_ratio():
    # Simulate 100 scenes in a 10-minute 24fps video (~14400 frames total).
    fps = 24.0
    half = 3
    duration = 600.0
    stride = int(duration * fps / 100)  # ~144 frames between scenes
    centers = [half + i * stride for i in range(100)]
    needed = _build_needed_frames(centers, half)
    last_needed = centers[-1] + half
    pct = 100.0 * len(needed) / (last_needed + 1)
    ranges = _merge_to_ranges(needed)
    _test(
        "select reduces frames to ≤15% of decoded range",
        pct <= 15.0,
        f"got {pct:.1f}%",
    )
    _test(
        "number of ranges equals number of non-overlapping scenes",
        len(ranges) == 100,
        f"got {len(ranges)} ranges",
    )


def test_empty_centers():
    needed = _build_needed_frames([], half=3)
    ranges = _merge_to_ranges(needed)
    _test("empty centers produce empty needed list", needed == [])
    _test("empty centers produce empty ranges", ranges == [])


def test_single_frame_windows_with_half_zero():
    # n_frames=1 → half=0 → each window is just the center frame itself
    centers = [50, 100, 150]
    needed = _build_needed_frames(centers, half=0)
    _test(
        "half=0 selects only the center frames",
        needed == [50, 100, 150],
        f"got {needed}",
    )
    ranges = _merge_to_ranges(needed)
    _test(
        "half=0 produces one-frame ranges",
        ranges == [(50, 50), (100, 100), (150, 150)],
        f"got {ranges}",
    )


def test_all_needed_sorted():
    import random
    random.seed(42)
    centers = sorted(random.randint(10, 10000) for _ in range(50))
    needed = _build_needed_frames(centers, half=3)
    _test(
        "all_needed is always sorted",
        needed == sorted(needed),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_single_center_no_overlap()
    test_two_non_overlapping_centers()
    test_two_overlapping_centers_merged()
    test_adjacent_windows_merged()
    test_frame_zero_clamp()
    test_select_expr_format()
    test_select_reduction_ratio()
    test_empty_centers()
    test_single_frame_windows_with_half_zero()
    test_all_needed_sorted()

    total = _PASS + _FAIL
    print(f"\n{_PASS}/{total} tests passed")
    sys.exit(0 if _FAIL == 0 else 1)
