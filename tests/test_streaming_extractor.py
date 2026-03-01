"""
Tests for dataset_generator_v2/streaming_extractor.py

Covers:
1. build_frame_assignments_distributed – slot assignment, interleaving,
   short-video scaling.
2. build_frame_ranges_from_assignments – range merging.
3. create_patch_pair – GT/LR dimensions, centre crop, n_frames validation.
4. save_patch_pair – correct paths, file creation, PNG parameters.
"""

import os
import shutil
import tempfile
import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dataset_generator_v2'))

from streaming_extractor import (
    build_frame_assignments_distributed,
    build_frame_ranges_from_assignments,
    create_patch_pair,
    save_patch_pair,
)


# ─── build_frame_assignments_distributed ─────────────────────────────────────

class TestBuildFrameAssignmentsDistributed(unittest.TestCase):
    """Tests for build_frame_assignments_distributed()."""

    def _make_timestamps(self, n, fps=25.0, stride=1.0):
        return [i * stride for i in range(n)]

    def test_empty_inputs_return_empty(self):
        result = build_frame_assignments_distributed([], {}, fps=25.0)
        self.assertEqual(result, [])

        result = build_frame_assignments_distributed([1.0], {}, fps=25.0)
        self.assertEqual(result, [])

    def test_single_category_single_format_all_assigned(self):
        timestamps = self._make_timestamps(10)
        dist = {'master': {'small_540': 10}}
        result = build_frame_assignments_distributed(timestamps, dist, fps=25.0)
        self.assertEqual(len(result), 10)
        self.assertTrue(all(cat == 'master' and fmt == 'small_540'
                            for _, cat, fmt in result))
        print("✓ single category/format: all scenes assigned")

    def test_result_sorted_by_frame_idx(self):
        timestamps = self._make_timestamps(20, stride=1.0)
        dist = {'master': {'small_540': 10, 'large_720': 10}}
        result = build_frame_assignments_distributed(timestamps, dist, fps=25.0)
        frame_indices = [r[0] for r in result]
        self.assertEqual(frame_indices, sorted(frame_indices))
        print("✓ result is sorted by frame index")

    def test_no_duplicate_frame_assignments(self):
        """Each frame index must appear at most once in the result."""
        timestamps = self._make_timestamps(100, stride=0.5)
        dist = {
            'master': {'small_540': 25, 'medium_169': 25, 'large_720': 50},
        }
        result = build_frame_assignments_distributed(timestamps, dist, fps=25.0)
        frame_indices = [r[0] for r in result]
        self.assertEqual(len(frame_indices), len(set(frame_indices)),
                         "Duplicate frame indices found in assignments")
        print("✓ no duplicate frame indices")

    def test_all_formats_appear_in_early_assignments(self):
        """With interleaved distribution, all formats must appear near the start."""
        timestamps = self._make_timestamps(3600, stride=1.0)
        dist = {'master': {'small_540': 1800, 'medium_169': 900, 'large_720': 900}}
        result = build_frame_assignments_distributed(timestamps, dist, fps=25.0)

        n_slots = 3  # three format slots
        early_formats = {r[2] for r in result[:n_slots]}
        expected = {'small_540', 'medium_169', 'large_720'}
        self.assertEqual(early_formats, expected,
                         f"Formats missing from early assignments: {expected - early_formats}")
        print("✓ all formats appear in the first N assignments (interleaved)")

    def test_short_video_scales_down_without_exceeding_scene_count(self):
        """When timestamps < total needed, counts are scaled and ≤ len(timestamps)."""
        timestamps = self._make_timestamps(50)  # only 50 scenes
        dist = {'master': {'small_540': 400, 'large_720': 600}}  # 1000 needed
        result = build_frame_assignments_distributed(timestamps, dist, fps=25.0)
        self.assertLessEqual(len(result), 50)
        print("✓ short video: result does not exceed available scenes")

    def test_center_frame_offset_by_half(self):
        """Center frame index should be int(ts * fps) + n_frames // 2."""
        fps = 25.0
        n_frames = 7
        half = n_frames // 2
        timestamps = [0.0, 1.0, 2.0]
        dist = {'master': {'small_540': 3}}
        result = build_frame_assignments_distributed(
            timestamps, dist, fps=fps, n_frames=n_frames
        )
        expected_centers = [int(ts * fps) + half for ts in timestamps]
        actual_centers = sorted(r[0] for r in result)
        self.assertEqual(actual_centers, expected_centers)
        print("✓ center frame = int(ts * fps) + half")

    def test_format_counts_match_distribution(self):
        """The number of assignments per format should match the target counts."""
        timestamps = self._make_timestamps(200)
        dist = {
            'master': {'small_540': 100, 'large_720': 100},
        }
        result = build_frame_assignments_distributed(timestamps, dist, fps=25.0)
        fmt_counter = Counter(r[2] for r in result)
        self.assertEqual(fmt_counter['small_540'], 100)
        self.assertEqual(fmt_counter['large_720'], 100)
        print("✓ format counts match distribution")

    def test_multi_category_distribution(self):
        """Assignments from multiple categories must all be present."""
        timestamps = self._make_timestamps(400)
        dist = {
            'master':    {'small_540': 100, 'large_720': 100},
            'universal': {'small_540': 100, 'large_720': 100},
        }
        result = build_frame_assignments_distributed(timestamps, dist, fps=25.0)
        cat_counter = Counter(r[1] for r in result)
        self.assertEqual(cat_counter['master'], 200)
        self.assertEqual(cat_counter['universal'], 200)
        print("✓ multi-category distribution")


# ─── build_frame_ranges_from_assignments ─────────────────────────────────────

class TestBuildFrameRangesFromAssignments(unittest.TestCase):

    def test_empty_returns_empty(self):
        self.assertEqual(build_frame_ranges_from_assignments([]), [])

    def test_single_assignment(self):
        asgn = [(50, 'master', 'small_540')]
        ranges = build_frame_ranges_from_assignments(asgn, n_frames=7)
        self.assertEqual(ranges, [(47, 53)])
        print("✓ single assignment → single range")

    def test_non_overlapping_assignments(self):
        asgn = [(10, 'a', 'b'), (100, 'a', 'b')]
        ranges = build_frame_ranges_from_assignments(asgn, n_frames=7)
        self.assertEqual(len(ranges), 2)
        self.assertEqual(ranges[0], (7, 13))
        self.assertEqual(ranges[1], (97, 103))
        print("✓ non-overlapping assignments → two ranges")

    def test_overlapping_assignments_merged(self):
        # Centers 10 and 14 with half=3 → [7,13] and [11,17] overlap
        asgn = [(10, 'a', 'b'), (14, 'a', 'b')]
        ranges = build_frame_ranges_from_assignments(asgn, n_frames=7)
        self.assertEqual(len(ranges), 1)
        self.assertEqual(ranges[0], (7, 17))
        print("✓ overlapping windows merged into one range")

    def test_adjacent_assignments_merged(self):
        # Centers 10 and 17 with half=3 → [7,13] and [14,20] are adjacent
        asgn = [(10, 'a', 'b'), (17, 'a', 'b')]
        ranges = build_frame_ranges_from_assignments(asgn, n_frames=7)
        self.assertEqual(len(ranges), 1)
        self.assertEqual(ranges[0], (7, 20))
        print("✓ adjacent windows merged into one range")

    def test_clamp_to_zero(self):
        # Center=2, half=3 → raw start=-1, clamped to 0
        asgn = [(2, 'a', 'b')]
        ranges = build_frame_ranges_from_assignments(asgn, n_frames=7)
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[0][1], 5)
        print("✓ frame range clamped at 0 for near-start assignments")

    def test_sorted_output(self):
        asgn = [(200, 'a', 'b'), (50, 'a', 'b'), (100, 'a', 'b')]
        ranges = build_frame_ranges_from_assignments(asgn, n_frames=7)
        starts = [r[0] for r in ranges]
        self.assertEqual(starts, sorted(starts))
        print("✓ ranges are sorted by start frame")


# ─── create_patch_pair ───────────────────────────────────────────────────────

class TestCreatePatchPair(unittest.TestCase):

    def _make_frames(self, n, h=1080, w=1920):
        """Create n random BGR frames."""
        return [np.random.randint(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]

    def _make_cfg(self, gt_w=540, gt_h=540, lr_w=180, lr_h=180):
        return {'gt_size': [gt_w, gt_h], 'lr_size': [lr_w, lr_h]}

    def test_invalid_frame_count_returns_none(self):
        cfg = self._make_cfg()
        for bad_n in (0, 1, 4, 6, 8):
            frames = self._make_frames(bad_n)
            gt, lr = create_patch_pair(frames, 'small_540', cfg)
            self.assertIsNone(gt)
            self.assertIsNone(lr)
        print("✓ invalid n_frames returns (None, None)")

    def test_valid_7_frames(self):
        frames = self._make_frames(7)
        cfg = self._make_cfg(540, 540, 180, 180)
        gt, lr = create_patch_pair(frames, 'small_540', cfg, force_center=True)
        self.assertIsNotNone(gt)
        self.assertIsNotNone(lr)
        self.assertEqual(gt.shape, (540, 540, 3))
        self.assertEqual(lr.shape, (7 * 180, 180, 3))
        print("✓ 7-frame patch: correct GT and LR shapes")

    def test_valid_5_frames(self):
        frames = self._make_frames(5)
        cfg = self._make_cfg(720, 405, 240, 135)
        gt, lr = create_patch_pair(frames, 'medium_169', cfg, force_center=True)
        self.assertIsNotNone(gt)
        self.assertIsNotNone(lr)
        self.assertEqual(gt.shape, (405, 720, 3))
        self.assertEqual(lr.shape, (5 * 135, 240, 3))
        print("✓ 5-frame patch: correct GT and LR shapes")

    def test_gt_is_center_frame_crop(self):
        """GT must come from the centre frame (index n//2)."""
        # Use a solid-colour frame for each position so we can identify them
        frames = []
        for i in range(7):
            f = np.full((1080, 1920, 3), i * 20, dtype=np.uint8)
            frames.append(f)
        cfg = self._make_cfg(540, 540, 180, 180)
        gt, _ = create_patch_pair(frames, 'small_540', cfg, force_center=True)
        self.assertIsNotNone(gt)
        # Center frame is index 3 → pixel value 60
        self.assertTrue(np.all(gt == 60))
        print("✓ GT comes from centre frame")

    def test_frame_too_small_returns_none(self):
        frames = self._make_frames(7, h=100, w=100)
        cfg = self._make_cfg(540, 540, 180, 180)
        gt, lr = create_patch_pair(frames, 'small_540', cfg)
        self.assertIsNone(gt)
        self.assertIsNone(lr)
        print("✓ frame smaller than gt_size returns (None, None)")

    def test_lr_stacked_vertically(self):
        """LR must be n_frames crops stacked along axis 0."""
        frames = self._make_frames(7)
        lr_w, lr_h = 180, 180
        cfg = self._make_cfg(540, 540, lr_w, lr_h)
        _, lr = create_patch_pair(frames, 'small_540', cfg, force_center=True)
        self.assertIsNotNone(lr)
        self.assertEqual(lr.shape[0], 7 * lr_h)
        self.assertEqual(lr.shape[1], lr_w)
        print("✓ LR frames stacked vertically (axis 0)")


# ─── save_patch_pair ─────────────────────────────────────────────────────────

class TestSavePatchPair(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _make_patch(self, h=540, w=540):
        return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    def test_save_creates_files(self):
        gt = self._make_patch(540, 540)
        lr = self._make_patch(7 * 180, 180)
        ok, gt_path, lr_path = save_patch_pair(
            gt, lr,
            video_path='/videos/myvideo.mkv',
            timestamp=10.0,
            category='master',
            format_name='small_540',
            n_frames=7,
            base_dir=self.tmp,
        )
        self.assertTrue(ok)
        self.assertTrue(os.path.exists(gt_path))
        self.assertTrue(os.path.exists(lr_path))
        print("✓ save_patch_pair creates both GT and LR files")

    def test_filename_uses_video_stem_and_timestamp(self):
        gt = self._make_patch()
        lr = self._make_patch(7 * 180, 180)
        ok, gt_path, _ = save_patch_pair(
            gt, lr,
            video_path='/some/path/myvideo.mkv',
            timestamp=5.123,
            category='master',
            format_name='small_540',
            n_frames=7,
            base_dir=self.tmp,
        )
        self.assertTrue(ok)
        fname = os.path.basename(gt_path)
        self.assertTrue(fname.startswith('myvideo_'))
        self.assertIn(f'{int(5.123 * 1000):08d}', fname)
        print("✓ filename contains video stem and timestamp")

    def test_returns_false_on_invalid_path(self):
        gt = self._make_patch()
        lr = self._make_patch(7 * 180, 180)
        ok, gt_path, lr_path = save_patch_pair(
            gt, lr,
            video_path='/bad.mkv',
            timestamp=1.0,
            category='nonexistent_category',
            format_name='nonexistent_format',
            n_frames=7,
            base_dir='/nonexistent_base_dir_xyz',
        )
        self.assertFalse(ok)
        self.assertIsNone(gt_path)
        self.assertIsNone(lr_path)
        print("✓ returns (False, None, None) when path is invalid")


if __name__ == '__main__':
    unittest.main(verbosity=2)
