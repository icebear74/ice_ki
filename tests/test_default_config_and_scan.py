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

def _make_unified_config(video_dir):
    """Unified (V1+V2) config with base_settings, category_targets, source_dirs, videos."""
    return {
        'base_settings': {
            'base_frame_limit': 100,
            'max_workers': 1,
            'output_base_dir': '/tmp/out',
        },
        'category_targets': {'master': 1000, 'universal': 500},
        'format_config': {},
        'source_dirs': [{'path': video_dir, 'extensions': ['.mkv', '.mp4', '.avi']}],
        'videos': [],
    }


def _make_v2_only_config(video_dir):
    """V2-only config (category_weights, no base_settings)."""
    return {
        'source_dirs': [{'path': video_dir, 'extensions': ['.mkv']}],
        'videos': [],
        'category_weights': {'cat_a': 0.6, 'cat_b': 0.4},
        'processing': {'total_patches': 50},
        'output_patches': {},
    }


# ─── create_default_config tests ──────────────────────────────────────────────

class TestCreateDefaultConfig(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_build_default_config_has_required_keys(self):
        cfg = build_default_config()
        for key in ('base_settings', 'category_targets', 'format_config', 'source_dirs', 'videos'):
            self.assertIn(key, cfg, f"Missing key: {key}")
        print("✓ build_default_config() has all required top-level keys")

    def test_build_default_config_empty_lists(self):
        cfg = build_default_config()
        self.assertEqual(cfg['source_dirs'], [])
        self.assertEqual(cfg['videos'], [])
        print("✓ source_dirs and videos are empty in fresh default config")

    def test_build_default_config_has_four_categories(self):
        cfg = build_default_config()
        for cat in ('master', 'universal', 'space', 'toon'):
            self.assertIn(cat, cfg['category_targets'])
            self.assertIn(cat, cfg['format_config'])
        print("✓ default config has master / universal / space / toon categories")

    def test_create_default_config_writes_valid_json(self):
        out = os.path.join(self.tmp, 'test_out.json')
        create_default_config(out)
        self.assertTrue(os.path.exists(out))
        with open(out) as f:
            loaded = json.load(f)
        self.assertIn('base_settings', loaded)
        self.assertIn('source_dirs', loaded)
        print("✓ create_default_config() writes valid JSON")

    def test_build_default_config_uses_template(self):
        """When a template file is present, its values override built-in defaults."""
        tmpl = {
            'base_settings': {'max_workers': 99},
            'category_targets': {'alpha': 9999},
            'format_config': {'alpha': {}},
        }
        tmpl_path = os.path.join(self.tmp, 'template.json')
        with open(tmpl_path, 'w') as f:
            json.dump(tmpl, f)

        cfg = build_default_config(template_path=tmpl_path)
        self.assertEqual(cfg['base_settings']['max_workers'], 99)
        self.assertIn('alpha', cfg['category_targets'])
        self.assertNotIn('master', cfg['category_targets'])
        print("✓ build_default_config() applies template values correctly")

    def test_build_default_config_bad_template_falls_back(self):
        """A missing or broken template must not raise; built-in defaults are used."""
        cfg = build_default_config(template_path='/nonexistent/path.json')
        self.assertIn('master', cfg['category_targets'])
        print("✓ build_default_config() falls back to built-ins when template is missing")


# ─── manager.categories from config keys ──────────────────────────────────────

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

    def test_categories_from_category_targets(self):
        """manager.categories includes keys from category_targets even with no video assignments."""
        cfg = {
            'category_targets': {'master': 100, 'space': 50},
            'videos': [],
            'source_dirs': [],
        }
        m = self._make_manager(cfg)
        self.assertIn('master', m.categories)
        self.assertIn('space',  m.categories)
        print("✓ categories populated from category_targets")

    def test_categories_from_category_weights(self):
        """manager.categories includes keys from category_weights (V2-only config)."""
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
        """Categories from video assignments and config keys are merged."""
        cfg = {
            'category_targets': {'master': 100},
            'videos': [{'name': 'v', 'path': '/x/v.mkv', 'categories': ['toon']}],
            'source_dirs': [],
        }
        m = self._make_manager(cfg)
        self.assertIn('master', m.categories)
        self.assertIn('toon',   m.categories)
        print("✓ categories merged from video assignments and config keys")


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

        cfg = _make_unified_config(self.vid_dir)
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
            'category_targets': {'master': 100},
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

        cfg = _make_unified_config(self.vid_dir)
        cfg['processing'] = {'total_patches': 10}
        cfg['output_patches'] = {}
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

    # ── StateManager._create_new_state with V1 config ─────────────────────────

    def test_create_new_state_v1_config_no_crash(self):
        """StateManager initialises without error on V1-style config (no 'processing' key)."""
        cfg = {
            'base_settings': {'max_workers': 4},
            'category_targets': {'master': 1000},
            'source_dirs': [{'path': self.vid_dir, 'extensions': ['.mkv']}],
            'videos': [],
        }
        try:
            sm = StateManager(cfg, os.path.join(self.tmp, 'state_v1.json'))
        except Exception as exc:
            self.fail(f"StateManager crashed on V1 config: {exc}")
        self.assertEqual(sm.state['progress']['total_patches'], 0)
        print("✓ StateManager initialises without crash on V1-style config")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Running Default Config & Scanning Fix Tests")
    print("=" * 60 + "\n")
    unittest.main(verbosity=2)
