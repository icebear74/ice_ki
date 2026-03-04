"""
Test suite for DataStrategyScheduler and SizeGroupedSampler distribution override.
"""

import os
import sys
import importlib.util
import unittest

# ---------------------------------------------------------------------------
# Import DataStrategyScheduler directly from its file, bypassing the package
# __init__.py which transitively imports torch (not available in this env).
# ---------------------------------------------------------------------------
_ROOT = os.path.join(os.path.dirname(__file__), '..')
_DS_PATH = os.path.join(_ROOT, 'vsr_plusplus_NEU', 'core', 'data_strategy.py')
_spec = importlib.util.spec_from_file_location('data_strategy', _DS_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
DataStrategyScheduler = _mod.DataStrategyScheduler


# ---------------------------------------------------------------------------
# Minimal fake dataset so we can construct a SizeGroupedSampler without I/O
# ---------------------------------------------------------------------------

class _FakeDataset:
    def __init__(self, size):
        self._size = size

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        return idx


# ---------------------------------------------------------------------------
# DataStrategyScheduler tests
# ---------------------------------------------------------------------------

class TestDataStrategySchedulerPhases(unittest.TestCase):
    """Phase identification and boundary conditions."""

    def setUp(self):
        self.sched = DataStrategyScheduler()

    def test_phase_warmup_start(self):
        self.assertEqual(self.sched.get_phase(0), DataStrategyScheduler.PHASE_WARMUP)

    def test_phase_warmup_before_boundary(self):
        self.assertEqual(self.sched.get_phase(999), DataStrategyScheduler.PHASE_WARMUP)

    def test_phase_crop_intro_at_boundary(self):
        self.assertEqual(self.sched.get_phase(1000), DataStrategyScheduler.PHASE_CROP_INTRO)

    def test_phase_crop_intro_mid(self):
        self.assertEqual(self.sched.get_phase(5500), DataStrategyScheduler.PHASE_CROP_INTRO)

    def test_phase_crop_intro_end_minus_1(self):
        self.assertEqual(self.sched.get_phase(9999), DataStrategyScheduler.PHASE_CROP_INTRO)

    def test_phase_stable_at_boundary(self):
        self.assertEqual(self.sched.get_phase(10000), DataStrategyScheduler.PHASE_STABLE)

    def test_phase_stable_far(self):
        self.assertEqual(self.sched.get_phase(50000), DataStrategyScheduler.PHASE_STABLE)


class TestDataStrategySchedulerDistribution(unittest.TestCase):
    """Distribution weights returned per phase."""

    ALL_SIZES = ['540', '720', '720_169']

    def setUp(self):
        self.sched = DataStrategyScheduler(all_size_keys=self.ALL_SIZES)

    def test_warmup_only_fullframe(self):
        dist = self.sched.get_distribution(0, self.ALL_SIZES)
        self.assertAlmostEqual(dist['720_169'], 1.0, places=6)
        self.assertAlmostEqual(dist['540'], 0.0, places=6)
        self.assertAlmostEqual(dist['720'], 0.0, places=6)

    def test_warmup_end_minus_1_still_fullframe(self):
        dist = self.sched.get_distribution(999, self.ALL_SIZES)
        self.assertAlmostEqual(dist['720_169'], 1.0, places=6)

    def test_crop_intro_start_equals_warmup(self):
        """At step 1000 (t=0) the distribution should equal the warmup distribution."""
        dist = self.sched.get_distribution(1000, self.ALL_SIZES)
        self.assertAlmostEqual(dist['720_169'], 1.0, places=6)
        self.assertAlmostEqual(dist['540'], 0.0, places=6)
        self.assertAlmostEqual(dist['720'], 0.0, places=6)

    def test_crop_intro_near_end_approaches_target(self):
        """At step 9999 (t≈1) the distribution should be very close to CROP_INTRO_END_DISTRIBUTION."""
        dist = self.sched.get_distribution(9999, self.ALL_SIZES)
        target = DataStrategyScheduler.CROP_INTRO_END_DISTRIBUTION
        for sk in self.ALL_SIZES:
            self.assertAlmostEqual(dist[sk], target.get(sk, 0.0), places=2,
                                   msg=f"Mismatch for size {sk}")

    def test_crop_intro_midpoint_monotonic(self):
        """720_169 weight decreases and crop weights increase during Phase 2."""
        dist_start = self.sched.get_distribution(1000, self.ALL_SIZES)
        dist_mid = self.sched.get_distribution(5500, self.ALL_SIZES)
        dist_near_end = self.sched.get_distribution(9999, self.ALL_SIZES)

        # 720_169 strictly decreasing
        self.assertGreater(dist_start['720_169'], dist_mid['720_169'])
        self.assertGreater(dist_mid['720_169'], dist_near_end['720_169'])

        # Crop sizes strictly increasing
        for sk in ('540', '720'):
            self.assertLess(dist_start[sk], dist_mid[sk])
            self.assertLess(dist_mid[sk], dist_near_end[sk])

    def test_stable_returns_none(self):
        """Phase 3 returns None so the sampler uses natural file-count proportional mode."""
        self.assertIsNone(self.sched.get_distribution(10000, self.ALL_SIZES))
        self.assertIsNone(self.sched.get_distribution(50000, self.ALL_SIZES))

    def test_stable_returns_none_ignores_available_sizes(self):
        """None is returned regardless of what available_sizes is passed in Phase 3."""
        self.assertIsNone(self.sched.get_distribution(10000, ['720_169']))
        self.assertIsNone(self.sched.get_distribution(10000, None))


class TestDataStrategySchedulerPerceptualWeight(unittest.TestCase):
    """Perceptual loss weight scheduling."""

    def setUp(self):
        self.sched = DataStrategyScheduler()

    def test_warmup_zero(self):
        self.assertAlmostEqual(self.sched.get_perceptual_weight(0), 0.0, places=6)
        self.assertAlmostEqual(self.sched.get_perceptual_weight(999), 0.0, places=6)

    def test_crop_intro_at_start_zero(self):
        self.assertAlmostEqual(self.sched.get_perceptual_weight(1000), 0.0, places=6)

    def test_crop_intro_at_end_target(self):
        w = self.sched.get_perceptual_weight(10000)
        self.assertAlmostEqual(w, DataStrategyScheduler.TARGET_PERCEPTUAL_WEIGHT, places=6)

    def test_crop_intro_midpoint_half_target(self):
        w = self.sched.get_perceptual_weight(5500)
        half_target = DataStrategyScheduler.TARGET_PERCEPTUAL_WEIGHT / 2.0
        self.assertAlmostEqual(w, half_target, places=6)

    def test_stable_equals_target(self):
        w = self.sched.get_perceptual_weight(50000)
        self.assertAlmostEqual(w, DataStrategyScheduler.TARGET_PERCEPTUAL_WEIGHT, places=6)

    def test_monotonically_increasing_through_crop_intro(self):
        steps = range(1000, 10001, 500)
        weights = [self.sched.get_perceptual_weight(s) for s in steps]
        for i in range(len(weights) - 1):
            self.assertLessEqual(weights[i], weights[i + 1])


class TestDataStrategySchedulerPhaseTransition(unittest.TestCase):
    """Phase transition detection and logging."""

    def test_first_call_records_phase(self):
        sched = DataStrategyScheduler()
        logs = []
        changed = sched.check_phase_transition(0, log_fn=logs.append)
        self.assertTrue(changed)
        self.assertEqual(len(logs), 0)  # No message on first transition (no old phase)

    def test_transition_logged(self):
        sched = DataStrategyScheduler()
        logs = []
        sched.check_phase_transition(0, log_fn=logs.append)   # warmup, no log
        sched.check_phase_transition(500, log_fn=logs.append)  # same phase, no log
        sched.check_phase_transition(1000, log_fn=logs.append)  # crop_introduction → logs
        self.assertEqual(len(logs), 1)
        self.assertIn('warmup', logs[0])
        self.assertIn('crop_introduction', logs[0])

    def test_no_duplicate_logs_within_phase(self):
        sched = DataStrategyScheduler()
        logs = []
        sched.check_phase_transition(0, log_fn=logs.append)
        for step in range(1, 1000):
            sched.check_phase_transition(step, log_fn=logs.append)
        # All steps within warmup — no transition message
        self.assertEqual(len(logs), 0)

    def test_transition_returns_false_when_no_change(self):
        sched = DataStrategyScheduler()
        sched.check_phase_transition(0)
        result = sched.check_phase_transition(500)
        self.assertFalse(result)

    def test_transition_returns_true_at_boundary(self):
        sched = DataStrategyScheduler()
        sched.check_phase_transition(0)
        result = sched.check_phase_transition(1000)
        self.assertTrue(result)


# ---------------------------------------------------------------------------
# Minimal SizeGroupedSampler re-implementation for testing
# (mirrors the real one without the torch.utils.data.Sampler base class)
# ---------------------------------------------------------------------------

class _MinimalSizeGroupedSampler:
    """
    A torch-free re-implementation of SizeGroupedSampler's core logic used
    only in unit tests.  Any change to the real sampler's _compute_batch_counts
    / __iter__ logic must be reflected here too.
    """

    def __init__(self, datasets_dict, batch_sizes, shuffle=False):
        self.datasets_dict = datasets_dict
        self.batch_sizes = batch_sizes
        self.shuffle = shuffle
        self.active_sizes = list(datasets_dict.keys())
        self._distribution_override = None
        self._compute_batch_counts()

    def set_distribution(self, distribution_dict):
        self._distribution_override = distribution_dict
        self._compute_batch_counts()

    def _compute_batch_counts(self):
        import random as _rand

        if self._distribution_override is None:
            self.num_batches_per_size = {
                sk: len(self.datasets_dict[sk]) // self.batch_sizes[sk]
                for sk in self.active_sizes
            }
            self.total_batches = sum(self.num_batches_per_size.values())
            return

        active = {
            sk: w
            for sk, w in self._distribution_override.items()
            if sk in self.active_sizes and w > 0
        }
        total_w = sum(active.values())

        if total_w == 0:
            self.num_batches_per_size = {
                sk: len(self.datasets_dict[sk]) // self.batch_sizes[sk]
                for sk in self.active_sizes
            }
            self.total_batches = sum(self.num_batches_per_size.values())
            return

        normalized = {sk: w / total_w for sk, w in active.items()}
        base_total = sum(
            len(self.datasets_dict[sk]) // self.batch_sizes[sk]
            for sk in self.active_sizes
        )

        self.num_batches_per_size = {}
        for sk in self.active_sizes:
            w = normalized.get(sk, 0.0)
            if w > 0:
                available = len(self.datasets_dict[sk]) // self.batch_sizes[sk]
                target = max(1, round(base_total * w))
                self.num_batches_per_size[sk] = min(target, available)
            else:
                self.num_batches_per_size[sk] = 0

        self.total_batches = sum(self.num_batches_per_size.values())

    def __iter__(self):
        import random as _rand
        indices_per_size = {}
        for sk in self.active_sizes:
            if self.num_batches_per_size.get(sk, 0) == 0:
                continue
            indices = list(range(len(self.datasets_dict[sk])))
            if self.shuffle:
                _rand.shuffle(indices)
            indices_per_size[sk] = indices

        batch_schedule = []
        for sk in self.active_sizes:
            n = self.num_batches_per_size.get(sk, 0)
            if n > 0:
                batch_schedule.extend([(sk, i) for i in range(n)])

        if self.shuffle:
            _rand.shuffle(batch_schedule)

        for sk, batch_idx in batch_schedule:
            bs = self.batch_sizes[sk]
            start = batch_idx * bs
            yield (sk, indices_per_size[sk][start:start + bs])

    def __len__(self):
        return self.total_batches


# ---------------------------------------------------------------------------
# SizeGroupedSampler distribution override tests
# ---------------------------------------------------------------------------

class TestSizeGroupedSamplerDistribution(unittest.TestCase):
    """SizeGroupedSampler.set_distribution() integrates with DataStrategyScheduler."""

    def _make_sampler(self):
        datasets = {
            '720_169': _FakeDataset(30),
            '540': _FakeDataset(20),
            '720': _FakeDataset(10),
        }
        batch_sizes = {'720_169': 1, '540': 1, '720': 1}
        return _MinimalSizeGroupedSampler(datasets, batch_sizes, shuffle=False)

    def test_default_total_batches(self):
        sampler = self._make_sampler()
        # Default: proportional to file counts  (30 + 20 + 10 = 60)
        self.assertEqual(len(sampler), 60)

    def test_set_distribution_only_fullframe(self):
        """Phase 1: only 720_169 active → only its batches are yielded."""
        sampler = self._make_sampler()
        sampler.set_distribution({'720_169': 1.0, '540': 0.0, '720': 0.0})
        size_keys = [sk for sk, _ in sampler]
        self.assertTrue(all(sk == '720_169' for sk in size_keys))
        self.assertGreater(len(size_keys), 0)

    def test_set_distribution_zero_sizes_excluded(self):
        """Sizes with weight 0 must not appear in the batch schedule."""
        sampler = self._make_sampler()
        sampler.set_distribution({'720_169': 1.0, '540': 0.0, '720': 0.0})
        for sk, _ in sampler:
            self.assertNotIn(sk, ('540', '720'))

    def test_set_distribution_all_active(self):
        """With all sizes at equal weight, all three size keys should appear."""
        sampler = self._make_sampler()
        sampler.set_distribution({'720_169': 0.33, '540': 0.33, '720': 0.33})
        seen = set(sk for sk, _ in sampler)
        self.assertEqual(seen, {'720_169', '540', '720'})

    def test_reset_to_file_count(self):
        """Calling set_distribution(None) restores file-count proportional behaviour."""
        sampler = self._make_sampler()
        sampler.set_distribution({'720_169': 1.0, '540': 0.0, '720': 0.0})
        sampler.set_distribution(None)
        self.assertEqual(len(sampler), 60)  # back to file-count total
        seen = set(sk for sk, _ in sampler)
        self.assertEqual(seen, {'720_169', '540', '720'})

    def test_len_reflects_updated_distribution(self):
        """__len__ must be consistent with what __iter__ yields."""
        sampler = self._make_sampler()
        sampler.set_distribution({'720_169': 1.0, '540': 0.0, '720': 0.0})
        actual_batches = sum(1 for _ in sampler)
        self.assertEqual(actual_batches, len(sampler))

    def test_scheduler_driven_integration(self):
        """DataStrategyScheduler distribution passed to sampler works end-to-end."""
        sched = DataStrategyScheduler(all_size_keys=['720_169', '540', '720'])
        sampler = self._make_sampler()

        # Phase 1: only full-frames
        dist = sched.get_distribution(0, sampler.active_sizes)
        sampler.set_distribution(dist)
        size_keys_p1 = [sk for sk, _ in sampler]
        self.assertTrue(all(sk == '720_169' for sk in size_keys_p1),
                        f"Phase 1 should yield only 720_169, got: {set(size_keys_p1)}")

        # Phase 3: get_distribution returns None → set_distribution(None) restores
        # natural file-count proportional sampling, so all sizes should appear.
        dist_p3 = sched.get_distribution(50000, sampler.active_sizes)
        self.assertIsNone(dist_p3, "Phase 3 must return None so sampler uses file counts")
        sampler.set_distribution(dist_p3)  # None → natural file-count mode
        size_keys_p3 = set(sk for sk, _ in sampler)
        self.assertEqual(size_keys_p3, {'720_169', '540', '720'})


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Running DataStrategyScheduler & SizeGroupedSampler tests")
    print("=" * 60 + "\n")
    unittest.main(verbosity=2)
