#!/usr/bin/env python3
"""
Unit tests for snap_assignments_to_centers().

Run with:
    cd dataset_generator_v2
    python -m pytest test_center_snapping.py -v
or simply:
    python test_center_snapping.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from streaming_extractor import snap_assignments_to_centers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _centers(assignments):
    """Return the sorted list of unique center frame indices."""
    return sorted({idx for idx, _, _ in assignments})


def _pairs(assignments):
    """Return the (category, format) pairs preserving order."""
    return [(cat, fmt) for _, cat, fmt in assignments]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_within_tolerance_snapped_to_first_center():
    """Assignments within tol_frames should share the first cluster center."""
    fps = 24.0
    # 24 frames apart = exactly 1 second → within tolerance of 24 frames
    asgn = [
        (100, "master", "fmt_a"),
        (112, "universal", "fmt_b"),  # 12 frames away — within ±24
        (124, "extra", "fmt_c"),      # 24 frames from 100 — within ±24
    ]
    result = snap_assignments_to_centers(asgn, fps=fps, tol_seconds=1.0)
    centers = _centers(result)
    assert len(centers) == 1, f"Expected 1 cluster, got {centers}"
    assert centers[0] == 100, f"Representative should be 100, got {centers[0]}"


def test_outside_tolerance_remain_separate():
    """Assignments further apart than tol_frames should stay as distinct centers."""
    fps = 24.0
    # 25 frames apart > 24 frames tolerance → different clusters
    asgn = [
        (100, "master", "fmt_a"),
        (125, "universal", "fmt_b"),  # 25 frames away > ±24
    ]
    result = snap_assignments_to_centers(asgn, fps=fps, tol_seconds=1.0)
    centers = _centers(result)
    assert len(centers) == 2, f"Expected 2 clusters, got {centers}"
    assert 100 in centers
    assert 125 in centers


def test_category_format_pairs_preserved():
    """(category, format_name) pairs must be preserved unchanged after snapping."""
    fps = 30.0
    asgn = [
        (50, "master", "x4_bicubic"),
        (60, "universal", "x2_lanczos"),   # 10 frames < 30 → snapped
        (200, "extra", "x8_nearest"),       # far away, separate cluster
    ]
    result = snap_assignments_to_centers(asgn, fps=fps, tol_seconds=1.0)
    result_pairs = set((cat, fmt) for _, cat, fmt in result)
    expected_pairs = {("master", "x4_bicubic"), ("universal", "x2_lanczos"), ("extra", "x8_nearest")}
    assert result_pairs == expected_pairs, f"Pairs mismatch: {result_pairs}"


def test_snapping_is_deterministic():
    """Calling snap_assignments_to_centers twice with the same input produces identical output."""
    fps = 25.0
    asgn = [
        (0, "cat_a", "fmt_1"),
        (10, "cat_b", "fmt_2"),
        (300, "cat_c", "fmt_3"),
        (305, "cat_d", "fmt_4"),
        (600, "cat_e", "fmt_5"),
    ]
    result1 = snap_assignments_to_centers(asgn, fps=fps, tol_seconds=1.0)
    result2 = snap_assignments_to_centers(asgn, fps=fps, tol_seconds=1.0)
    assert result1 == result2, "Snapping must be deterministic"


def test_zero_tolerance_disables_snapping():
    """tol_seconds=0 should return assignments unchanged (sorted)."""
    fps = 24.0
    asgn = [
        (100, "master", "fmt_a"),
        (101, "universal", "fmt_b"),
        (102, "extra", "fmt_c"),
    ]
    result = snap_assignments_to_centers(asgn, fps=fps, tol_seconds=0.0)
    centers = _centers(result)
    assert len(centers) == 3, "tol=0 should not snap any centers"
    assert sorted(centers) == [100, 101, 102]


def test_no_negative_frame_indices():
    """Snapped center indices must remain non-negative."""
    fps = 24.0
    asgn = [
        (0, "master", "fmt_a"),
        (10, "universal", "fmt_b"),
    ]
    result = snap_assignments_to_centers(asgn, fps=fps, tol_seconds=1.0)
    for idx, _, _ in result:
        assert idx >= 0, f"Negative frame index after snapping: {idx}"


def test_empty_input():
    """Empty assignment list should return an empty list."""
    result = snap_assignments_to_centers([], fps=24.0, tol_seconds=1.0)
    assert result == []


def test_single_assignment():
    """A single assignment should be returned unchanged."""
    asgn = [(500, "master", "fmt_a")]
    result = snap_assignments_to_centers(asgn, fps=24.0, tol_seconds=1.0)
    assert result == asgn


def test_multiple_clusters():
    """Multiple well-separated clusters are identified correctly."""
    fps = 24.0
    # clusters: [100, 110], [200, 205], [400]
    asgn = [
        (100, "cat_a", "fmt_1"),
        (110, "cat_b", "fmt_2"),   # within 24 of 100 → cluster 1
        (200, "cat_c", "fmt_3"),
        (205, "cat_d", "fmt_4"),   # within 24 of 200 → cluster 2
        (400, "cat_e", "fmt_5"),   # far away → cluster 3
    ]
    result = snap_assignments_to_centers(asgn, fps=fps, tol_seconds=1.0)
    centers = _centers(result)
    assert len(centers) == 3, f"Expected 3 clusters, got {centers}"
    assert 100 in centers
    assert 200 in centers
    assert 400 in centers


def test_result_sorted_by_center():
    """Output should be sorted by center_frame_idx."""
    fps = 24.0
    asgn = [
        (300, "cat_c", "fmt_3"),
        (100, "cat_a", "fmt_1"),
        (200, "cat_b", "fmt_2"),
    ]
    result = snap_assignments_to_centers(asgn, fps=fps, tol_seconds=1.0)
    indices = [idx for idx, _, _ in result]
    assert indices == sorted(indices), "Result must be sorted by center_frame_idx"


def test_fps_scaling():
    """tol_frames scales with fps: at 60fps 1s tolerance = 60 frames."""
    fps = 60.0
    # 50 frames apart < 60 → same cluster
    asgn = [
        (100, "master", "fmt_a"),
        (150, "universal", "fmt_b"),   # 50 < 60 → snapped
        (300, "extra", "fmt_c"),       # 150 > 60 → new cluster
    ]
    result = snap_assignments_to_centers(asgn, fps=fps, tol_seconds=1.0)
    centers = _centers(result)
    assert len(centers) == 2, f"Expected 2 clusters at 60fps, got {centers}"
    assert 100 in centers
    assert 300 in centers


# ---------------------------------------------------------------------------
# Main runner (for standalone execution without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_within_tolerance_snapped_to_first_center,
        test_outside_tolerance_remain_separate,
        test_category_format_pairs_preserved,
        test_snapping_is_deterministic,
        test_zero_tolerance_disables_snapping,
        test_no_negative_frame_indices,
        test_empty_input,
        test_single_assignment,
        test_multiple_clusters,
        test_result_sorted_by_center,
        test_fps_scaling,
    ]
    passed = 0
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ✓  {fn.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"  ✗  {fn.__name__}: {exc}")
            failed += 1
        except Exception as exc:
            print(f"  ✗  {fn.__name__}: unexpected error: {exc}")
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed")
    sys.exit(0 if failed == 0 else 1)
