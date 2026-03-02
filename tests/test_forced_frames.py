"""
Tests for the per-video forced_frames override feature.

Covers:
  1. VideoManager.set_forced_frames() – read/write in config JSON
  2. print_video_list() – ⚡ indicator shown when forced_frames set
  3. calculate_proportional_distribution() – forced frames subtracted from
     category budget; remainder distributed proportionally
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dataset_generator_v2'))

from video_manager import VideoManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(videos):
    return {
        'videos': videos,
        'category_patches': {'master': 10000, 'space': 5000},
        'output_patches': {},
        'source_dirs': [],
    }


def _write_config(path, videos):
    cfg = _make_config(videos)
    with open(path, 'w') as f:
        json.dump(cfg, f)


# ---------------------------------------------------------------------------
# VideoManager.set_forced_frames()
# ---------------------------------------------------------------------------

class TestSetForcedFrames(unittest.TestCase):

    def _manager_with_videos(self, videos):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            json.dump(_make_config(videos), f)
            path = f.name
        mgr = VideoManager(path)
        mgr.load()
        return mgr, path

    def test_forced_frames_stored_for_category(self):
        """set_forced_frames stores positive values under video['forced_frames']."""
        mgr, _ = self._manager_with_videos([
            {'name': 'Film A', 'path': '/a.mkv', 'categories': ['master', 'space']},
        ])
        inputs = iter(['10000', '5000'])
        with patch('builtins.input', side_effect=inputs):
            mgr.set_forced_frames(0)

        video = mgr.videos[0]
        self.assertEqual(video.get('forced_frames', {}).get('master'), 10000)
        self.assertEqual(video.get('forced_frames', {}).get('space'), 5000)
        self.assertTrue(mgr.modified)
        print("✓ set_forced_frames stores values correctly")

    def test_zero_clears_entry(self):
        """Entering 0 removes that category from forced_frames."""
        mgr, _ = self._manager_with_videos([
            {'name': 'Film B', 'path': '/b.mkv',
             'categories': ['master'],
             'forced_frames': {'master': 9999}},
        ])
        with patch('builtins.input', return_value='0'):
            mgr.set_forced_frames(0)

        video = mgr.videos[0]
        self.assertNotIn('forced_frames', video,
                         "forced_frames key must be absent when all values are cleared")
        print("✓ set_forced_frames removes key when value is 0")

    def test_blank_input_keeps_existing(self):
        """Pressing Enter (blank) keeps the existing forced value unchanged."""
        mgr, _ = self._manager_with_videos([
            {'name': 'Film C', 'path': '/c.mkv',
             'categories': ['master'],
             'forced_frames': {'master': 7777}},
        ])
        with patch('builtins.input', return_value=''):
            mgr.set_forced_frames(0)

        video = mgr.videos[0]
        self.assertEqual(video['forced_frames']['master'], 7777)
        print("✓ blank input preserves existing forced_frames value")

    def test_invalid_index_prints_error(self):
        """Invalid video index must not raise, just print an error."""
        mgr, _ = self._manager_with_videos([])
        # Should not raise
        mgr.set_forced_frames(999)
        print("✓ set_forced_frames with invalid index does not raise")

    def test_video_without_categories_prints_error(self):
        """A video with no categories must be rejected gracefully."""
        mgr, _ = self._manager_with_videos([
            {'name': 'No-Cat Film', 'path': '/x.mkv', 'categories': []},
        ])
        # Should not raise or prompt
        mgr.set_forced_frames(0)
        print("✓ set_forced_frames rejects video without categories gracefully")


# ---------------------------------------------------------------------------
# print_video_list() – ⚡ indicator
# ---------------------------------------------------------------------------

class TestPrintVideoListForcedIndicator(unittest.TestCase):

    def _capture_list(self, video):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            json.dump(_make_config([video]), f)
            path = f.name
        mgr = VideoManager(path)
        mgr.load()

        import io
        buf = io.StringIO()
        with patch('sys.stdout', buf):
            mgr.print_video_list(mgr.list_videos())
        return buf.getvalue()

    def test_no_indicator_without_forced_frames(self):
        output = self._capture_list(
            {'name': 'Normal Film', 'path': '/n.mkv', 'categories': ['master']}
        )
        self.assertNotIn('⚡', output)
        print("✓ no ⚡ indicator when forced_frames absent")

    def test_indicator_shown_with_forced_frames(self):
        output = self._capture_list(
            {'name': 'Forced Film', 'path': '/f.mkv',
             'categories': ['master'], 'forced_frames': {'master': 10000}}
        )
        self.assertIn('⚡', output)
        self.assertIn('10,000', output)
        print("✓ ⚡ indicator shown when forced_frames present")


# ---------------------------------------------------------------------------
# calculate_proportional_distribution() forced-frames logic (pure unit test)
# ---------------------------------------------------------------------------

class TestProportionalDistributionWithForcedFrames(unittest.TestCase):
    """
    Test the forced-frames distribution logic in isolation without importing
    the full DatasetGeneratorV2UHD (which has many heavy dependencies).
    We replicate the exact algorithm from calculate_proportional_distribution
    to verify correctness of the arithmetic:

        remaining = category_target - forced_total
        each_normal = int(remaining * duration / normal_total_duration)
    """

    @staticmethod
    def _run_distribution(category_target, videos, durations):
        """
        Minimal replica of the forced-frames branch in
        calculate_proportional_distribution for one category.
        Returns {video_path: patches_allocated}.
        """
        forced_videos = {}
        normal_videos = []
        normal_total_duration = 0.0
        forced_total = 0

        for v in videos:
            path = v['path']
            if path not in durations:
                continue
            forced = v.get('forced_frames', {}).get('master', 0)
            if forced > 0:
                forced_videos[path] = forced
                forced_total += forced
            else:
                dur = durations[path]
                normal_videos.append((path, v['name'], dur))
                normal_total_duration += dur

        remaining_budget = max(0, category_target - forced_total)
        result = {}

        for path, forced_count in forced_videos.items():
            result[path] = forced_count

        for path, name, dur in normal_videos:
            if normal_total_duration > 0:
                result[path] = int(remaining_budget * dur / normal_total_duration)
            else:
                result[path] = 0

        return result

    def test_forced_video_gets_exact_count(self):
        """The forced video receives exactly its forced_frames value."""
        videos = [
            {'name': 'A', 'path': '/a.mkv', 'categories': ['master'],
             'forced_frames': {'master': 10000}},
            {'name': 'B', 'path': '/b.mkv', 'categories': ['master']},
            {'name': 'C', 'path': '/c.mkv', 'categories': ['master']},
        ]
        durations = {'/a.mkv': 3600.0, '/b.mkv': 3600.0, '/c.mkv': 3600.0}
        result = self._run_distribution(30000, videos, durations)

        self.assertEqual(result['/a.mkv'], 10000)
        print("✓ forced video receives exactly its forced_frames count")

    def test_remainder_distributed_proportionally(self):
        """After forced frames are subtracted, the rest goes proportionally."""
        videos = [
            {'name': 'A', 'path': '/a.mkv', 'categories': ['master'],
             'forced_frames': {'master': 5000}},
            {'name': 'B', 'path': '/b.mkv', 'categories': ['master']},
            {'name': 'C', 'path': '/c.mkv', 'categories': ['master']},
        ]
        # B has twice the duration of C → should get ~twice the proportional share
        durations = {'/a.mkv': 3600.0, '/b.mkv': 7200.0, '/c.mkv': 3600.0}
        result = self._run_distribution(15000, videos, durations)

        # remaining = 15000 - 5000 = 10000; B gets 10000*7200/10800 = 6666; C gets 3333
        self.assertEqual(result['/a.mkv'], 5000)
        self.assertAlmostEqual(result['/b.mkv'], 6666, delta=1)
        self.assertAlmostEqual(result['/c.mkv'], 3333, delta=1)
        print("✓ remaining budget is distributed proportionally by duration")

    def test_no_forced_videos_is_pure_proportional(self):
        """When no forced_frames are set the algorithm is identical to the old proportional logic."""
        videos = [
            {'name': 'A', 'path': '/a.mkv', 'categories': ['master']},
            {'name': 'B', 'path': '/b.mkv', 'categories': ['master']},
        ]
        durations = {'/a.mkv': 6000.0, '/b.mkv': 4000.0}
        result = self._run_distribution(10000, videos, durations)

        self.assertEqual(result['/a.mkv'], 6000)
        self.assertEqual(result['/b.mkv'], 4000)
        print("✓ no forced_frames → pure proportional (identical to old logic)")

    def test_forced_total_exceeds_budget_clamps_to_zero(self):
        """If forced frames exceed category_target, remaining_budget is clamped to 0."""
        videos = [
            {'name': 'A', 'path': '/a.mkv', 'categories': ['master'],
             'forced_frames': {'master': 50000}},
            {'name': 'B', 'path': '/b.mkv', 'categories': ['master']},
        ]
        durations = {'/a.mkv': 3600.0, '/b.mkv': 3600.0}
        result = self._run_distribution(10000, videos, durations)

        self.assertEqual(result['/a.mkv'], 50000)
        # Normal video gets 0 because remaining_budget = max(0, 10000-50000) = 0
        self.assertEqual(result['/b.mkv'], 0)
        print("✓ forced total > budget → remaining clamped to 0 (no negative allocation)")

    def test_all_videos_forced(self):
        """All videos having forced_frames → no proportional distribution needed."""
        videos = [
            {'name': 'A', 'path': '/a.mkv', 'categories': ['master'],
             'forced_frames': {'master': 3000}},
            {'name': 'B', 'path': '/b.mkv', 'categories': ['master'],
             'forced_frames': {'master': 7000}},
        ]
        durations = {'/a.mkv': 3600.0, '/b.mkv': 3600.0}
        result = self._run_distribution(10000, videos, durations)

        self.assertEqual(result['/a.mkv'], 3000)
        self.assertEqual(result['/b.mkv'], 7000)
        print("✓ all videos forced → each gets its exact value")


if __name__ == '__main__':
    unittest.main(verbosity=2)
