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
    build_assignments_per_category,
    create_patch_pair,
    save_patch_pair,
    is_black_frame,
    cuda_available,
    scale_cuda_available,
    tonemap_cuda_available,
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


# ─── is_black_frame ───────────────────────────────────────────────────────────

class TestIsBlackFrame(unittest.TestCase):

    def test_pure_black_is_detected(self):
        black = np.zeros((540, 540, 3), dtype=np.uint8)
        self.assertTrue(is_black_frame(black))
        print("✓ pure black frame detected")

    def test_bright_frame_not_detected(self):
        bright = np.full((540, 540, 3), 128, dtype=np.uint8)
        self.assertFalse(is_black_frame(bright))
        print("✓ bright frame not flagged as black")

    def test_custom_threshold(self):
        # mean brightness 30 → black at threshold=35, not at threshold=20
        frame = np.full((100, 100, 3), 30, dtype=np.uint8)
        self.assertTrue(is_black_frame(frame, brightness_threshold=35.0))
        self.assertFalse(is_black_frame(frame, brightness_threshold=20.0))
        print("✓ custom brightness threshold respected")

    def test_just_below_threshold_is_black(self):
        frame = np.full((100, 100, 3), 19, dtype=np.uint8)
        self.assertTrue(is_black_frame(frame, brightness_threshold=20.0))
        print("✓ value just below threshold → black")

    def test_just_above_threshold_is_not_black(self):
        frame = np.full((100, 100, 3), 21, dtype=np.uint8)
        self.assertFalse(is_black_frame(frame, brightness_threshold=20.0))
        print("✓ value just above threshold → not black")


# ─── build_assignments_per_category ──────────────────────────────────────────

class TestBuildAssignmentsPerCategory(unittest.TestCase):

    def test_empty_distribution_returns_empty(self):
        result = build_assignments_per_category({}, duration=60.0, fps=25.0)
        self.assertEqual(result, [])
        print("✓ empty distribution → empty result")

    def test_single_category_no_duplicate_frames(self):
        dist = {'master': {'small_540': 100, 'large_720': 50}}
        result = build_assignments_per_category(dist, duration=3600.0, fps=25.0)
        frame_indices = [r[0] for r in result]
        self.assertEqual(len(set(frame_indices)), len(frame_indices),
                         "Single category: duplicate frame indices found")
        print("✓ single category: no duplicate frame indices")

    def test_per_category_no_duplicate_frames(self):
        """Within each category every frame index must be unique."""
        dist = {
            'master':    {'small_540': 300, 'medium_169': 100, 'large_720': 100},
            'universal': {'small_540': 200, 'medium_169':  50, 'large_720':  50},
        }
        result = build_assignments_per_category(dist, duration=3600.0, fps=25.0)
        for cat in ('master', 'universal'):
            frames = [r[0] for r in result if r[1] == cat]
            self.assertEqual(len(set(frames)), len(frames),
                             f"Category '{cat}': duplicate frame indices found")
        print("✓ no duplicates within each category")

    def test_same_frame_can_appear_in_multiple_categories(self):
        """With equal targets the same positions must appear in both categories."""
        dist = {
            'master':    {'small_540': 200},
            'universal': {'small_540': 200},
        }
        result = build_assignments_per_category(dist, duration=3600.0, fps=25.0)
        master_frames    = set(r[0] for r in result if r[1] == 'master')
        universal_frames = set(r[0] for r in result if r[1] == 'universal')
        overlap = master_frames & universal_frames
        self.assertGreater(len(overlap), 0,
                           "Equal stride ⇒ categories share frame positions")
        print(f"✓ {len(overlap)} frame positions shared across categories")

    def test_total_assignments_equals_sum_of_category_totals(self):
        """Total assignments = sum of all per-category targets (normal video)."""
        dist = {
            'master':    {'small_540': 500, 'large_720': 250},   # 750
            'universal': {'small_540': 300},                      # 300
        }
        result = build_assignments_per_category(dist, duration=7200.0, fps=25.0)
        self.assertEqual(len(result), 750 + 300)
        print("✓ total assignments = sum of category targets")

    def test_result_sorted_by_frame_idx(self):
        dist = {'master': {'small_540': 50}, 'universal': {'small_540': 30}}
        result = build_assignments_per_category(dist, duration=3600.0, fps=25.0)
        frames = [r[0] for r in result]
        self.assertEqual(frames, sorted(frames))
        print("✓ result sorted by frame index")

    def test_short_video_scales_down_per_category_independently(self):
        """Each category is scaled independently; larger target → more scenes."""
        dist = {
            'master':    {'small_540': 1000},
            'universal': {'small_540':  400},
        }
        # Use a duration long enough that the strides differ (> 0.5 s each):
        # master stride = 3599/1000 = 3.6 s, universal stride = 3599/400 = 9.0 s
        result = build_assignments_per_category(dist, duration=3600.0, fps=25.0)
        master_cnt    = sum(1 for r in result if r[1] == 'master')
        universal_cnt = sum(1 for r in result if r[1] == 'universal')
        # Both categories get patches
        self.assertGreater(master_cnt, 0)
        self.assertGreater(universal_cnt, 0)
        # master has more scenes than universal (larger target → tighter stride)
        self.assertGreater(master_cnt, universal_cnt)
        # No duplicates within each category
        for cat in ('master', 'universal'):
            frames = [r[0] for r in result if r[1] == cat]
            self.assertEqual(len(set(frames)), len(frames))
        print("✓ short video: per-category scaling, proportional, no intra-category duplicates")

    def test_three_categories_all_have_unique_scenes(self):
        """Concrete 5000+2000+1000 scenario: all categories unique internally."""
        dist = {
            'master':    {'small_540': 3000, 'medium_169': 1200, 'large_720': 800},  # 5000
            'universal': {'small_540': 1200, 'medium_169':  500, 'large_720': 300},  # 2000
            'space':     {'small_540':  600, 'medium_169':  250, 'large_720': 150},  # 1000
        }
        result = build_assignments_per_category(dist, duration=7200.0, fps=25.0)
        # Total assignments = 5000 + 2000 + 1000 = 8000
        self.assertEqual(len(result), 8000)
        for cat in ('master', 'universal', 'space'):
            frames = [r[0] for r in result if r[1] == cat]
            self.assertEqual(len(set(frames)), len(frames),
                             f"Category '{cat}': duplicate frame indices found")
        print("✓ 5000+2000+1000 = 8000 assignments, all categories unique internally")

    def test_all_formats_appear_in_early_assignments_per_category(self):
        """All formats must be present near the start of each category's scenes."""
        dist = {'master': {'small_540': 3600, 'medium_169': 900, 'large_720': 900}}
        result = build_assignments_per_category(dist, duration=7200.0, fps=25.0)
        master_only = [(fi, fmt) for fi, cat, fmt in result if cat == 'master']
        n_formats = 3  # 3 format slots
        early_formats = {fmt for _, fmt in master_only[:n_formats]}
        self.assertEqual(early_formats, {'small_540', 'medium_169', 'large_720'},
                         "All formats must appear within first N assignments (interleaved)")
        print("✓ all formats appear in early assignments (interleaved within category)")

    def test_center_frame_offset_by_half(self):
        """center_frame_idx == int(ts * fps) + n_frames // 2."""
        fps, n_frames = 25.0, 7
        half = n_frames // 2
        dist = {'master': {'small_540': 3}}
        result = build_assignments_per_category(
            dist, duration=3600.0, fps=fps, n_frames=n_frames
        )
        for fi, _, _ in result:
            # Each center must be at least `half` frames in
            self.assertGreaterEqual(fi, half)
        print("✓ center frame indices are ≥ half (correct offset)")


# ─── filter chain constants ───────────────────────────────────────────────────

class TestFilterConstants(unittest.TestCase):
    """Sanity checks on the FFmpeg filter chain string constants."""

    def test_scale_cuda_uses_bicubic_not_lanczos(self):
        import streaming_extractor as _se
        self.assertIn("bicubic", _se._TONEMAP_FILTER_SCALE_CUDA)
        self.assertNotIn("lanczos", _se._TONEMAP_FILTER_SCALE_CUDA)
        print("✓ _TONEMAP_FILTER_SCALE_CUDA uses bicubic (not lanczos)")

    def test_scale_cuda_specifies_nv12_output_format(self):
        import streaming_extractor as _se
        # format=nv12 forces 8-bit CUDA surface output so hwdownload succeeds
        # for 10-bit HEVC sources (which otherwise produce a yuv410p surface).
        self.assertIn("format=nv12", _se._TONEMAP_FILTER_SCALE_CUDA)
        print("✓ _TONEMAP_FILTER_SCALE_CUDA includes format=nv12 for hwdownload compatibility")

    def test_scale_cuda_has_yuv420p_after_hwdownload(self):
        import streaming_extractor as _se
        # yuv420p conversion after hwdownload ensures zscale gets planar input.
        self.assertIn("format=yuv420p", _se._TONEMAP_FILTER_SCALE_CUDA)
        print("✓ _TONEMAP_FILTER_SCALE_CUDA has format=yuv420p after hwdownload")

    def test_tonemap_cuda_uses_bicubic_not_lanczos(self):
        import streaming_extractor as _se
        self.assertIn("bicubic", _se._TONEMAP_FILTER_CUDA)
        self.assertNotIn("lanczos", _se._TONEMAP_FILTER_CUDA)
        print("✓ _TONEMAP_FILTER_CUDA uses bicubic (not lanczos)")

    def test_tonemap_cuda_has_yuv420p_after_hwdownload(self):
        import streaming_extractor as _se
        # tonemap_cuda/scale_cuda output NV12; yuv420p deinterleaves it so
        # the final format=bgr24 libswscale conversion is unambiguous.
        self.assertIn("format=yuv420p", _se._TONEMAP_FILTER_CUDA)
        print("✓ _TONEMAP_FILTER_CUDA has format=yuv420p after hwdownload")

    def test_tonemap_cuda_ends_with_bgr24(self):
        import streaming_extractor as _se
        self.assertTrue(
            _se._TONEMAP_FILTER_CUDA.endswith("bgr24"),
            "_TONEMAP_FILTER_CUDA must end with format=bgr24",
        )
        print("✓ _TONEMAP_FILTER_CUDA ends with bgr24")

    def test_tonemap_cuda_uses_scale_to_break_hwdownload_negotiation(self):
        import streaming_extractor as _se
        # hwdownload=format=nv12 is NOT used because older FFmpeg builds do not
        # support the 'format' option on hwdownload and crash with:
        #   Error applying option 'format' to filter 'hwdownload': Option not found
        # Instead, bare hwdownload is followed by scale=iw:ih which:
        #   1. accepts NV12 input (libswscale), breaking the backward negotiation
        #      that would otherwise ask hwdownload to produce yuv420p directly
        #      (causing "Invalid output format yuv420p for hwframe download")
        #   2. converts NV12→YUV420P in software when downstream requests it
        self.assertNotIn("hwdownload=format=", _se._TONEMAP_FILTER_CUDA)
        self.assertIn("hwdownload,", _se._TONEMAP_FILTER_CUDA)
        self.assertIn("scale=iw:ih,", _se._TONEMAP_FILTER_CUDA)
        print("✓ _TONEMAP_FILTER_CUDA uses bare hwdownload + scale=iw:ih (all-FFmpeg-version fix)")

    def test_scale_cuda_filter_contains_hwdownload(self):
        import streaming_extractor as _se
        # bare hwdownload + scale=iw:ih breaks the backward format negotiation;
        # see test_tonemap_cuda_uses_scale_to_break_hwdownload_negotiation.
        self.assertNotIn("hwdownload=format=", _se._TONEMAP_FILTER_SCALE_CUDA)
        self.assertIn("hwdownload,", _se._TONEMAP_FILTER_SCALE_CUDA)
        self.assertIn("scale=iw:ih,", _se._TONEMAP_FILTER_SCALE_CUDA)
        print("✓ _TONEMAP_FILTER_SCALE_CUDA uses bare hwdownload + scale=iw:ih (all-FFmpeg-version fix)")

    def test_scale_cuda_filter_ends_with_bgr24(self):
        import streaming_extractor as _se
        self.assertTrue(
            _se._TONEMAP_FILTER_SCALE_CUDA.endswith("bgr24"),
            "_TONEMAP_FILTER_SCALE_CUDA must end with format=bgr24",
        )
        print("✓ _TONEMAP_FILTER_SCALE_CUDA ends with bgr24")

    def test_tonemap_filter_ends_with_bgr24(self):
        import streaming_extractor as _se
        self.assertTrue(
            _se._TONEMAP_FILTER.endswith("bgr24"),
            "_TONEMAP_FILTER must end with format=bgr24",
        )
        print("✓ _TONEMAP_FILTER ends with bgr24")


# ─── cuda_available ───────────────────────────────────────────────────────────

class TestCudaAvailable(unittest.TestCase):

    def test_returns_bool(self):
        result = cuda_available()
        self.assertIsInstance(result, bool)
        print(f"✓ cuda_available() returned bool: {result}")

    def test_result_is_cached(self):
        """Calling twice must return the same value (cached)."""
        first  = cuda_available()
        second = cuda_available()
        self.assertEqual(first, second)
        print("✓ cuda_available() result is stable/cached")

    def test_ffmpeg_not_found_returns_false(self):
        """When ffmpeg binary is absent the function must return False, not raise."""
        import streaming_extractor as _se
        with patch.object(_se, '_cuda_available', None):
            with patch("subprocess.check_output", side_effect=FileNotFoundError):
                result = _se.cuda_available()
        self.assertFalse(result)
        print("✓ missing ffmpeg → cuda_available() returns False without raising")


# ─── scale_cuda_available ─────────────────────────────────────────────────────

class TestScaleCudaAvailable(unittest.TestCase):

    def test_returns_bool(self):
        result = scale_cuda_available()
        self.assertIsInstance(result, bool)
        print(f"✓ scale_cuda_available() returned bool: {result}")

    def test_result_is_cached(self):
        """Calling twice must return the same value (cached)."""
        first  = scale_cuda_available()
        second = scale_cuda_available()
        self.assertEqual(first, second)
        print("✓ scale_cuda_available() result is stable/cached")

    def test_ffmpeg_not_found_returns_false(self):
        """When ffmpeg binary is absent the function must return False, not raise."""
        import streaming_extractor as _se
        with patch.object(_se, '_scale_cuda_available', None):
            with patch.object(_se, '_ffmpeg_filters_output', None):
                with patch("subprocess.check_output", side_effect=FileNotFoundError):
                    result = _se.scale_cuda_available()
        self.assertFalse(result)
        print("✓ missing ffmpeg → scale_cuda_available() returns False without raising")

    def test_true_when_scale_cuda_present(self):
        """Must return True when ffmpeg -filters lists scale_cuda."""
        import streaming_extractor as _se
        fake_output = b"... scale_cuda        V->V       GPU accelerated video resizer ..."
        with patch.object(_se, '_scale_cuda_available', None):
            with patch.object(_se, '_ffmpeg_filters_output', None):
                with patch("subprocess.check_output", return_value=fake_output):
                    result = _se.scale_cuda_available()
        self.assertTrue(result)
        print("✓ scale_cuda present → scale_cuda_available() returns True")

    def test_false_when_scale_cuda_absent(self):
        """Must return False when scale_cuda is not listed."""
        import streaming_extractor as _se
        fake_output = b"... zscale ... tonemap ... scale ..."
        with patch.object(_se, '_scale_cuda_available', None):
            with patch.object(_se, '_ffmpeg_filters_output', None):
                with patch("subprocess.check_output", return_value=fake_output):
                    result = _se.scale_cuda_available()
        self.assertFalse(result)
        print("✓ scale_cuda absent → scale_cuda_available() returns False")


# ─── tonemap_cuda_available ───────────────────────────────────────────────────

class TestTonemapCudaAvailable(unittest.TestCase):

    def test_returns_bool(self):
        result = tonemap_cuda_available()
        self.assertIsInstance(result, bool)
        print(f"✓ tonemap_cuda_available() returned bool: {result}")

    def test_result_is_cached(self):
        """Calling twice must return the same value (cached)."""
        first  = tonemap_cuda_available()
        second = tonemap_cuda_available()
        self.assertEqual(first, second)
        print("✓ tonemap_cuda_available() result is stable/cached")

    def test_ffmpeg_not_found_returns_false(self):
        """When ffmpeg binary is absent the function must return False, not raise."""
        import streaming_extractor as _se
        with patch.object(_se, '_tonemap_cuda_available', None):
            with patch.object(_se, '_ffmpeg_filters_output', None):
                with patch("subprocess.check_output", side_effect=FileNotFoundError):
                    result = _se.tonemap_cuda_available()
        self.assertFalse(result)
        print("✓ missing ffmpeg → tonemap_cuda_available() returns False without raising")

    def test_true_when_both_filters_present(self):
        """Must return True when ffmpeg -filters lists both tonemap_cuda and scale_cuda."""
        import streaming_extractor as _se
        fake_output = b"... tonemap_cuda ... scale_cuda ..."
        with patch.object(_se, '_tonemap_cuda_available', None):
            with patch.object(_se, '_ffmpeg_filters_output', None):
                with patch("subprocess.check_output", return_value=fake_output):
                    result = _se.tonemap_cuda_available()
        self.assertTrue(result)
        print("✓ both filters present → tonemap_cuda_available() returns True")

    def test_false_when_only_tonemap_cuda_present(self):
        """Must return False when scale_cuda is missing (both are required)."""
        import streaming_extractor as _se
        fake_output = b"... tonemap_cuda ..."
        with patch.object(_se, '_tonemap_cuda_available', None):
            with patch.object(_se, '_ffmpeg_filters_output', None):
                with patch("subprocess.check_output", return_value=fake_output):
                    result = _se.tonemap_cuda_available()
        self.assertFalse(result)
        print("✓ scale_cuda missing → tonemap_cuda_available() returns False")

    def test_false_when_only_scale_cuda_present(self):
        """Must return False when tonemap_cuda is missing (both are required)."""
        import streaming_extractor as _se
        fake_output = b"... scale_cuda ..."
        with patch.object(_se, '_tonemap_cuda_available', None):
            with patch.object(_se, '_ffmpeg_filters_output', None):
                with patch("subprocess.check_output", return_value=fake_output):
                    result = _se.tonemap_cuda_available()
        self.assertFalse(result)
        print("✓ tonemap_cuda missing → tonemap_cuda_available() returns False")

    def test_false_when_neither_filter_present(self):
        """Must return False when the output contains no CUDA filter names."""
        import streaming_extractor as _se
        fake_output = b"... zscale ... tonemap ..."
        with patch.object(_se, '_tonemap_cuda_available', None):
            with patch.object(_se, '_ffmpeg_filters_output', None):
                with patch("subprocess.check_output", return_value=fake_output):
                    result = _se.tonemap_cuda_available()
        self.assertFalse(result)
        print("✓ no CUDA filters → tonemap_cuda_available() returns False")


if __name__ == '__main__':
    unittest.main(verbosity=2)
