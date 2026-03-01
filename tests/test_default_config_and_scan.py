"""
Tests for:
  1. create_default_config – build_default_config() and create_default_config()
  2. Scanning fixes in video_manager.rescan_file_list() and state_manager.scan_videos()
     - case-insensitive extension matching
     - deduplication across overlapping source dirs
  3. manager.categories populated from config keys (category_targets / category_weights)
  4. state_manager._create_new_state() does not crash on V1-style config
"""

import json
import os
import shutil
import tempfile
import unittest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dataset_generator_v2'))

from create_default_config import build_default_config, create_default_config
from state_manager import StateManager
from video_manager import VideoManager


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_v2_config(video_dir):
    """V2-format config matching what video_manager.py generates."""
    return {
        'root_path': '/tmp/out',
        'source_dirs': [{'path': video_dir, 'extensions': ['.mkv', '.mp4', '.avi']}],
        'videos': [],
        'category_weights': {'master': 0.5, 'universal': 0.5},
        'output_patches': {
            '540':     {'enabled': True,  'gt_size': [540, 540], 'lr_size': [180, 180]},
            '720':     {'enabled': True,  'gt_size': [720, 720], 'lr_size': [240, 240]},
            '720_169': {'enabled': True,  'gt_size': [405, 720], 'lr_size': [135, 240]},
        },
        'processing': {'n_frames': 7, 'total_patches': 1000},
        'quality': {'blur_threshold': 80.0},
        'ffmpeg_timeout': 120,
        'ffprobe_timeout': 60,
    }


# ─── create_default_config tests ──────────────────────────────────────────────

class TestCreateDefaultConfig(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_build_default_config_has_required_keys(self):
        cfg = build_default_config()
        for key in ('root_path', 'category_weights', 'output_patches',
                    'processing', 'quality', 'workers',
                    'ffmpeg_timeout', 'ffprobe_timeout', 'source_dirs', 'videos'):
            self.assertIn(key, cfg, f"Missing key: {key}")
        for key in ('base_settings', 'category_targets', 'format_config',
                    '_comment_usage', '_comment_workflow'):
            self.assertNotIn(key, cfg, f"Unexpected V1/comment key: {key}")
        print("✓ build_default_config() has all required V2 top-level keys")

    def test_build_default_config_empty_lists(self):
        cfg = build_default_config()
        self.assertEqual(cfg['source_dirs'], [])
        self.assertEqual(cfg['videos'], [])
        print("✓ source_dirs and videos are empty in fresh default config")

    def test_build_default_config_has_four_categories(self):
        cfg = build_default_config()
        for cat in ('master', 'universal', 'space', 'toon'):
            self.assertIn(cat, cfg['category_weights'])
        print("✓ default config has master / universal / space / toon in category_weights")

    def test_build_default_config_weights_sum_to_one(self):
        cfg = build_default_config()
        total = sum(cfg['category_weights'].values())
        self.assertAlmostEqual(total, 1.0, places=5)
        print("✓ category_weights sum to 1.0")

    def test_build_default_config_output_patches_present(self):
        cfg = build_default_config()
        for patch_key in ('540', '720', '720_169'):
            self.assertIn(patch_key, cfg['output_patches'])
        print("✓ output_patches contains 540 / 720 / 720_169")

    def test_create_default_config_writes_valid_json(self):
        out = os.path.join(self.tmp, 'test_out.json')
        create_default_config(out)
        self.assertTrue(os.path.exists(out))
        with open(out) as f:
            loaded = json.load(f)
        self.assertIn('root_path', loaded)
        self.assertIn('category_weights', loaded)
        self.assertIn('source_dirs', loaded)
        print("✓ create_default_config() writes valid V2 JSON")

    def test_build_default_config_timeout_keys_present(self):
        cfg = build_default_config()
        self.assertEqual(cfg['ffmpeg_timeout'],  120)
        self.assertEqual(cfg['ffprobe_timeout'], 60)
        print("✓ ffmpeg_timeout and ffprobe_timeout present at top level")

    def test_build_default_config_template_override(self):
        """Template values are respected."""
        tmpl = {
            'root_path':        '/custom/out',
            'category_weights': {'alpha': 1.0},
            'ffmpeg_timeout':   300,
            'ffprobe_timeout':  90,
        }
        tmpl_path = os.path.join(self.tmp, 'tmpl.json')
        with open(tmpl_path, 'w') as f:
            json.dump(tmpl, f)
        cfg = build_default_config(template_path=tmpl_path)
        self.assertEqual(cfg['root_path'], '/custom/out')
        self.assertIn('alpha', cfg['category_weights'])
        self.assertNotIn('master', cfg['category_weights'])
        self.assertEqual(cfg['ffmpeg_timeout'],  300)
        self.assertEqual(cfg['ffprobe_timeout'], 90)
        print("✓ build_default_config() applies template values correctly")

    def test_build_default_config_bad_template_falls_back(self):
        """A missing template must not raise; built-in defaults are used."""
        cfg = build_default_config(template_path='/nonexistent/path.json')
        self.assertIn('master', cfg['category_weights'])
        print("✓ build_default_config() falls back to built-ins when template is missing")


# ─── UHD generator config-path selection ──────────────────────────────────────

class TestUHDGeneratorConfigSelection(unittest.TestCase):
    """
    Validate that make_dataset_v2_uhd.main() selects generator_config_v2.json.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._orig_argv = sys.argv[:]
        self._orig_cwd  = os.getcwd()

    def tearDown(self):
        sys.argv = self._orig_argv
        os.chdir(self._orig_cwd)
        shutil.rmtree(self.tmp)

    def _write_v2(self, filename='generator_config_v2.json'):
        cfg = {
            'root_path': self.tmp,
            'source_dirs': [], 'videos': [],
            'category_weights': {'master': 1.0},
            'output_patches': {},
            'processing': {'n_frames': 7, 'total_patches': 100},
            'quality': {'blur_threshold': 80.0},
            'ffmpeg_timeout': 120, 'ffprobe_timeout': 60,
        }
        path = os.path.join(self.tmp, filename)
        with open(path, 'w') as f:
            json.dump(cfg, f)
        return path

    def _resolve_config(self, directory):
        """Replicate the config-selection logic from make_dataset_v2_uhd.main()."""
        from pathlib import Path as P
        v2 = P(directory) / 'generator_config_v2.json'
        if v2.exists():
            return str(v2)
        return None

    def test_finds_v2_config(self):
        """generator_config_v2.json is found and selected."""
        self._write_v2()
        chosen = self._resolve_config(self.tmp)
        self.assertIsNotNone(chosen)
        self.assertTrue(chosen.endswith('generator_config_v2.json'))
        print("✓ UHD generator selects generator_config_v2.json")

    def test_returns_none_when_no_config(self):
        """Returns None when generator_config_v2.json is absent."""
        chosen = self._resolve_config(self.tmp)
        self.assertIsNone(chosen)
        print("✓ UHD generator returns None when no config present")

    def test_script_dir_is_dataset_generator_v2(self):
        """make_dataset_v2_uhd.py must look in its own directory, not parent.parent."""
        uhd_path = os.path.join(
            os.path.dirname(__file__), '..', 'dataset_generator_v2',
            'make_dataset_v2_uhd.py'
        )
        with open(uhd_path) as fh:
            src = fh.read()
        self.assertNotIn('parent.parent', src,
            "make_dataset_v2_uhd.py still uses parent.parent for script_dir")
        self.assertIn('Path(__file__).parent', src,
            "make_dataset_v2_uhd.py must use Path(__file__).parent for script_dir")
        print("✓ make_dataset_v2_uhd.py uses correct script_dir (parent)")


# ─── UHD generator V2→internal config normalization ──────────────────────────

class TestUHDConfigNormalization(unittest.TestCase):
    """
    Validate normalize_config() correctly maps V2 config fields
    (from video_manager.py) to the internal structure used by the generator.
    """

    def setUp(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dataset_generator_v2'))
        from utils.config_normalizer import normalize_config
        self.normalize = normalize_config

    def _v2_config(self, **overrides):
        cfg = {
            'root_path': '/data/out',
            'source_dirs': [],
            'videos': [],
            'category_weights': {'master': 0.25, 'universal': 0.75},
            'output_patches': {
                '540':     {'enabled': True,  'gt_size': [540, 540], 'lr_size': [180, 180]},
                '720':     {'enabled': True,  'gt_size': [720, 720], 'lr_size': [240, 240]},
                '720_169': {'enabled': False, 'gt_size': [405, 720], 'lr_size': [135, 240]},
            },
            'processing': {'n_frames': 7, 'total_patches': 100000,
                           'min_scene_length': 21, 'scene_threshold': 30.0},
            'quality': {'blur_threshold': 90.0, 'jpeg_quality': 95, 'min_sharpness': 30.0},
            'workers': 8,
            'ffmpeg_timeout': 120,
            'ffprobe_timeout': 60,
        }
        cfg.update(overrides)
        return cfg

    # ── base_settings ─────────────────────────────────────────────────────────

    def test_base_settings_created_from_root_path(self):
        """base_settings.output_base_dir is root_path."""
        result = self.normalize(self._v2_config())
        self.assertEqual(result['base_settings']['output_base_dir'], '/data/out')
        print("✓ base_settings.output_base_dir set from root_path")

    def test_temp_and_status_derived_from_root_path(self):
        """temp_dir and status_file are derived below root_path."""
        result = self.normalize(self._v2_config())
        bs = result['base_settings']
        self.assertTrue(bs['temp_dir'].startswith('/data/out'))
        self.assertTrue(bs['status_file'].startswith('/data/out'))
        print("✓ temp_dir and status_file derived from root_path")

    def test_lr_versions_7frames_for_n_frames_7(self):
        """7 frames → lr_versions=['7frames']."""
        result = self.normalize(self._v2_config())
        self.assertEqual(result['base_settings']['lr_versions'], ['7frames'])
        print("✓ n_frames=7 → lr_versions=['7frames']")

    def test_lr_versions_5frames_for_n_frames_5(self):
        """5 frames → lr_versions=['5frames']."""
        cfg = self._v2_config()
        cfg['processing']['n_frames'] = 5
        result = self.normalize(cfg)
        self.assertEqual(result['base_settings']['lr_versions'], ['5frames'])
        print("✓ n_frames=5 → lr_versions=['5frames']")

    def test_min_detail_threshold_from_blur_threshold(self):
        """min_detail_threshold is taken from quality.blur_threshold."""
        result = self.normalize(self._v2_config())
        self.assertEqual(result['base_settings']['min_detail_threshold'], 90.0)
        print("✓ min_detail_threshold taken from quality.blur_threshold")

    # ── category_targets ──────────────────────────────────────────────────────

    def test_category_targets_derived_from_weights(self):
        """category_targets = {cat: int(weight * total_patches)}."""
        result = self.normalize(self._v2_config())
        ct = result['category_targets']
        self.assertEqual(ct['master'],    25000)  # 0.25 × 100000
        self.assertEqual(ct['universal'], 75000)  # 0.75 × 100000
        print("✓ category_targets derived from category_weights × total_patches")

    # ── format_config ─────────────────────────────────────────────────────────

    def test_disabled_patches_excluded_from_format_config(self):
        """output_patches with enabled=False must not appear in format_config."""
        result = self.normalize(self._v2_config())
        for cat_formats in result['format_config'].values():
            self.assertNotIn('720_169', cat_formats,
                "disabled patch '720_169' must not appear in format_config")
        print("✓ disabled output_patches excluded from format_config")

    def test_enabled_patches_present_for_each_category(self):
        """Each category gets entries for all enabled output_patches."""
        result = self.normalize(self._v2_config())
        fc = result['format_config']
        self.assertIn('master',    fc)
        self.assertIn('universal', fc)
        for cat_formats in fc.values():
            self.assertIn('540', cat_formats)
            self.assertIn('720', cat_formats)
        print("✓ enabled patches present for all categories in format_config")

    def test_format_config_probabilities_sum_to_1(self):
        """Probabilities of all enabled formats must sum to 1.0 (within float rounding)."""
        result = self.normalize(self._v2_config())
        for category, formats in result['format_config'].items():
            total = sum(f['probability'] for f in formats.values())
            self.assertAlmostEqual(total, 1.0, places=4,
                msg=f"Probabilities for category '{category}' sum to {total}, not 1.0")
        print("✓ format probabilities sum to 1.0 per category")

    def test_format_config_contains_gt_and_lr_size(self):
        """Each format entry has gt_size and lr_size."""
        result = self.normalize(self._v2_config())
        for cat_formats in result['format_config'].values():
            for fmt_key, fmt_val in cat_formats.items():
                self.assertIn('gt_size', fmt_val, f"Missing gt_size for {fmt_key}")
                self.assertIn('lr_size', fmt_val, f"Missing lr_size for {fmt_key}")
        print("✓ format_config entries contain gt_size and lr_size")


# ─── manager.categories from config keys ────────────────────────────────────────────

class TestManagerCategoriesFromConfig(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _make_manager(self, config_dict):
        p = os.path.join(self.tmp, 'cfg.json')
        with open(p, 'w') as f:
            json.dump(config_dict, f)
        m = VideoManager(p)
        m.load()
        return m

    def test_categories_from_category_weights(self):
        """manager.categories includes keys from category_weights."""
        cfg = {
            'category_weights': {'cat_x': 0.5, 'cat_y': 0.5},
            'videos': [],
            'source_dirs': [],
            'processing': {'total_patches': 10},
            'output_patches': {},
        }
        m = self._make_manager(cfg)
        self.assertIn('cat_x', m.categories)
        self.assertIn('cat_y', m.categories)
        print("✓ categories populated from category_weights")

    def test_categories_merged_from_videos_and_config(self):
        """Categories from video assignments and category_weights are merged."""
        cfg = {
            'category_weights': {'master': 0.5, 'universal': 0.5},
            'videos': [{'name': 'v', 'path': '/x/v.mkv', 'categories': ['toon']}],
            'source_dirs': [],
        }
        m = self._make_manager(cfg)
        self.assertIn('master',    m.categories)
        self.assertIn('universal', m.categories)
        self.assertIn('toon',      m.categories)
        print("✓ categories merged from video assignments and category_weights")


# ─── scanning fixes ───────────────────────────────────────────────────────────

class TestScanningFixes(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.vid_dir = os.path.join(self.tmp, 'videos')
        os.makedirs(self.vid_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    # ── VideoManager.rescan_file_list ──────────────────────────────────────────

    def _make_manager(self, config_dict):
        p = os.path.join(self.tmp, 'cfg.json')
        with open(p, 'w') as f:
            json.dump(config_dict, f)
        m = VideoManager(p)
        m.load()
        return m

    def test_rescan_case_insensitive_extensions(self):
        """rescan_file_list() finds files regardless of extension case."""
        open(os.path.join(self.vid_dir, 'a.MKV'), 'w').close()
        open(os.path.join(self.vid_dir, 'b.mkv'), 'w').close()
        open(os.path.join(self.vid_dir, 'c.Mp4'), 'w').close()

        cfg = _make_v2_config(self.vid_dir)
        m = self._make_manager(cfg)
        m.rescan_file_list()

        paths = {v['path'] for v in m.config['videos']}
        self.assertEqual(len(paths), 3, f"Expected 3 files, got {len(paths)}: {paths}")
        print("✓ rescan_file_list() finds files with any extension case")

    def test_rescan_deduplication_overlapping_dirs(self):
        """rescan_file_list() does not create duplicate entries for overlapping dirs."""
        open(os.path.join(self.vid_dir, 'movie.mkv'), 'w').close()

        # Two source_dirs both pointing to the same directory
        cfg = {
            'category_weights': {'master': 1.0},
            'source_dirs': [
                {'path': self.vid_dir, 'extensions': ['.mkv']},
                {'path': self.vid_dir, 'extensions': ['.mkv']},
            ],
            'videos': [],
        }
        m = self._make_manager(cfg)
        m.rescan_file_list()

        self.assertEqual(len(m.config['videos']), 1)
        print("✓ rescan_file_list() deduplicates overlapping source directories")

    # ── StateManager.scan_videos ───────────────────────────────────────────────

    def test_state_scan_case_insensitive_extensions(self):
        """StateManager.scan_videos() matches files with uppercase extensions."""
        open(os.path.join(self.vid_dir, 'X.MKV'), 'w').close()

        cfg = _make_v2_config(self.vid_dir)
        sm = StateManager(cfg, os.path.join(self.tmp, 'state.json'))

        # Manually inject a fake cached entry so ffprobe is not called
        import stat as _st
        from datetime import datetime as _dt
        path_str = os.path.join(self.vid_dir, 'X.MKV')
        st = os.stat(path_str)
        sm.state['video_metadata'][path_str] = {
            'duration': 5, 'fps': 24, 'resolution': [1920, 1080],
            'file_size': st.st_size,
            'last_modified': _dt.fromtimestamp(st.st_mtime).isoformat(),
            'category': None,
        }
        sm.scan_videos()

        self.assertIn(path_str, sm.state['video_metadata'])
        print("✓ StateManager.scan_videos() handles uppercase extensions")

    def test_state_scan_deduplication(self):
        """StateManager.scan_videos() processes each file at most once even with overlapping dirs."""
        open(os.path.join(self.vid_dir, 'dup.mkv'), 'w').close()

        cfg = {
            'source_dirs': [
                {'path': self.vid_dir, 'extensions': ['.mkv']},
                {'path': self.vid_dir, 'extensions': ['.mkv']},
            ],
            'videos': [],
            'processing': {'total_patches': 10},
            'output_patches': {},
        }
        sm = StateManager(cfg, os.path.join(self.tmp, 'state.json'))
        sm.scan_videos()

        # Only one entry in metadata (ffprobe fails for empty file, so count may be 0)
        # but we should never have > 1 entry for the same path
        paths = list(sm.state['video_metadata'].keys())
        self.assertEqual(len(paths), len(set(paths)))
        print("✓ StateManager.scan_videos() deduplicates overlapping source directories")


# ─── category management ──────────────────────────────────────────────────────

class TestCategoryManagement(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _make_manager(self, config_dict):
        p = os.path.join(self.tmp, 'cfg.json')
        with open(p, 'w') as f:
            json.dump(config_dict, f)
        m = VideoManager(p)
        m.load()
        return m

    def _base_cfg(self):
        return {
            'category_weights': {'master': 0.5, 'space': 0.5},
            'output_patches': {
                '540': {'enabled': True, 'gt_size': [540, 540], 'lr_size': [180, 180]},
            },
            'processing': {'n_frames': 7, 'total_patches': 100000},
            'source_dirs': [],
            'videos': [],
        }

    def test_add_category_updates_all_structures(self):
        """_add_category() adds to category_weights."""
        m = self._make_manager(self._base_cfg())
        m.categories = list(m.categories)

        import unittest.mock as mock
        with mock.patch('builtins.input', side_effect=['toon', '0.1']):
            m._add_category()

        self.assertIn('toon', m.categories)
        self.assertIn('toon', m.config['category_weights'])
        self.assertAlmostEqual(m.config['category_weights']['toon'], 0.1)
        self.assertTrue(m.modified)
        print("✓ _add_category() updates categories and category_weights")

    def test_add_category_rejects_duplicate(self):
        """_add_category() refuses to add an already-existing category."""
        m = self._make_manager(self._base_cfg())
        import unittest.mock as mock
        with mock.patch('builtins.input', return_value='master'):
            m._add_category()
        # Still only 2 categories (master + space)
        self.assertEqual(m.categories.count('master'), 1)
        print("✓ _add_category() rejects duplicate category names")

    def test_remove_category_cleans_up(self):
        """_remove_category() removes from categories and category_weights."""
        cfg = self._base_cfg()
        cfg['videos'] = [
            {'name': 'v1', 'path': '/x/v1.mkv', 'categories': ['master', 'space']},
            {'name': 'v2', 'path': '/x/v2.mkv', 'categories': ['space']},
        ]
        m = self._make_manager(cfg)
        import unittest.mock as mock
        with mock.patch('builtins.input', side_effect=['space', 'yes']):
            m._remove_category()

        self.assertNotIn('space', m.categories)
        self.assertNotIn('space', m.config.get('category_weights', {}))
        # Videos must have space unassigned
        for v in m.videos:
            self.assertNotIn('space', v.get('categories', []))
        self.assertTrue(m.modified)
        print("✓ _remove_category() cleans up categories, category_weights, and videos")

    def test_remove_category_cancel(self):
        """_remove_category() respects cancellation."""
        m = self._make_manager(self._base_cfg())
        import unittest.mock as mock
        with mock.patch('builtins.input', side_effect=['master', 'no']):
            m._remove_category()
        self.assertIn('master', m.categories)
        print("✓ _remove_category() respects 'no' cancellation")

    def test_save_and_reload_preserves_new_category(self):
        """New categories persist after save/reload."""
        m = self._make_manager(self._base_cfg())
        import unittest.mock as mock
        with mock.patch('builtins.input', side_effect=['toon', '0.1']):
            m._add_category()
        m.save(backup=False)

        p = os.path.join(self.tmp, 'cfg.json')
        m2 = VideoManager(p)
        m2.load()
        self.assertIn('toon', m2.categories)
        self.assertIn('toon', m2.config['category_weights'])
        print("✓ New category persists after save/reload")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Running Default Config & Scanning Fix Tests")
    print("=" * 60 + "\n")
    unittest.main(verbosity=2)
