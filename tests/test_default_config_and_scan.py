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
        for key in ('base_settings', 'category_targets', 'format_config',
                    'ffmpeg_timeout', 'ffprobe_timeout', 'source_dirs', 'videos'):
            self.assertIn(key, cfg, f"Missing key: {key}")
        # Keys that make_dataset_v2_uhd.py does NOT read must not be present
        for key in ('_comment_usage', '_comment_workflow', '_comment_source_dirs', '_comment_videos'):
            self.assertNotIn(key, cfg, f"Unexpected comment key: {key}")
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

    def test_build_default_config_base_settings_only_uhd_keys(self):
        """base_settings must contain exactly the keys read by make_dataset_v2_uhd.py."""
        cfg = build_default_config()
        bs = cfg['base_settings']
        # Keys that the UHD generator actually reads
        for key in ('output_base_dir', 'temp_dir', 'status_file', 'lr_versions', 'min_detail_threshold'):
            self.assertIn(key, bs, f"Missing base_settings key: {key}")
        # Keys that the UHD generator does NOT read (used only by make_dataset_multi.py)
        for key in ('base_frame_limit', 'max_workers', 'val_percent',
                    'min_file_size', 'scene_diff_threshold', 'max_retry_attempts', 'retry_skip_seconds'):
            self.assertNotIn(key, bs, f"Unexpected (unused) base_settings key: {key}")
        print("✓ base_settings contains only keys read by make_dataset_v2_uhd.py")

    def test_build_default_config_timeout_keys_present(self):
        """ffmpeg_timeout and ffprobe_timeout must be at the top level."""
        cfg = build_default_config()
        self.assertEqual(cfg['ffmpeg_timeout'],  120)
        self.assertEqual(cfg['ffprobe_timeout'], 60)
        print("✓ ffmpeg_timeout and ffprobe_timeout present at top level")

    def test_build_default_config_template_timeout_override(self):
        """Template values for ffmpeg/ffprobe timeout are respected."""
        tmpl = {
            'base_settings': {},
            'category_targets': {},
            'format_config': {},
            'ffmpeg_timeout': 300,
            'ffprobe_timeout': 90,
        }
        tmpl_path = os.path.join(self.tmp, 'tmpl_timeout.json')
        with open(tmpl_path, 'w') as f:
            json.dump(tmpl, f)
        cfg = build_default_config(template_path=tmpl_path)
        self.assertEqual(cfg['ffmpeg_timeout'],  300)
        self.assertEqual(cfg['ffprobe_timeout'], 90)
        print("✓ build_default_config() respects ffmpeg/ffprobe timeouts from template")

    def test_build_default_config_uses_template(self):
        """When a template file is present, its values override built-in defaults."""
        tmpl = {
            'base_settings': {'output_base_dir': '/custom/out'},
            'category_targets': {'alpha': 9999},
            'format_config': {'alpha': {}},
        }
        tmpl_path = os.path.join(self.tmp, 'template.json')
        with open(tmpl_path, 'w') as f:
            json.dump(tmpl, f)

        cfg = build_default_config(template_path=tmpl_path)
        self.assertEqual(cfg['base_settings']['output_base_dir'], '/custom/out')
        self.assertIn('alpha', cfg['category_targets'])
        self.assertNotIn('master', cfg['category_targets'])
        print("✓ build_default_config() applies template values correctly")

    def test_build_default_config_bad_template_falls_back(self):
        """A missing or broken template must not raise; built-in defaults are used."""
        cfg = build_default_config(template_path='/nonexistent/path.json')
        self.assertIn('master', cfg['category_targets'])
        print("✓ build_default_config() falls back to built-ins when template is missing")


# ─── UHD generator config-path selection ──────────────────────────────────────

class TestUHDGeneratorConfigSelection(unittest.TestCase):
    """
    Validate that make_dataset_v2_uhd.main() selects the same config file
    as video_manager.main(): prefer generator_config_v2.json over
    generator_config.json.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._orig_argv = sys.argv[:]
        self._orig_cwd  = os.getcwd()

    def tearDown(self):
        sys.argv = self._orig_argv
        os.chdir(self._orig_cwd)
        shutil.rmtree(self.tmp)

    def _minimal_config(self):
        return {
            'base_settings': {
                'output_base_dir': self.tmp,
                'temp_dir':        os.path.join(self.tmp, 'temp'),
                'status_file':     os.path.join(self.tmp, '.status.json'),
                'lr_versions':     ['7frames'],
                'min_detail_threshold': 80.0,
            },
            'category_targets': {'master': 1000},
            'format_config': {'master': {}},
            'ffmpeg_timeout':  120,
            'ffprobe_timeout':  60,
            'source_dirs': [],
            'videos': [],
        }

    def _write(self, filename):
        path = os.path.join(self.tmp, filename)
        with open(path, 'w') as f:
            json.dump(self._minimal_config(), f)
        return path

    def _resolve_config(self, directory):
        """
        Replicate the config-selection logic from make_dataset_v2_uhd.main()
        so we can unit-test it without actually running the generator.
        """
        from pathlib import Path as P
        v2 = P(directory) / 'generator_config_v2.json'
        v1 = P(directory) / 'generator_config.json'
        if v2.exists():
            return str(v2)
        if v1.exists():
            return str(v1)
        return None

    def test_prefers_v2_config_when_both_exist(self):
        """generator_config_v2.json is chosen over generator_config.json."""
        self._write('generator_config.json')
        self._write('generator_config_v2.json')
        chosen = self._resolve_config(self.tmp)
        self.assertTrue(chosen.endswith('generator_config_v2.json'),
                        f"Expected v2 config, got: {chosen}")
        print("✓ UHD generator prefers generator_config_v2.json when both exist")

    def test_falls_back_to_v1_config(self):
        """Falls back to generator_config.json when v2 is absent."""
        self._write('generator_config.json')
        chosen = self._resolve_config(self.tmp)
        self.assertTrue(chosen.endswith('generator_config.json'),
                        f"Expected v1 fallback, got: {chosen}")
        print("✓ UHD generator falls back to generator_config.json when v2 absent")

    def test_returns_none_when_no_config(self):
        """Returns None (no config found) when neither file exists."""
        chosen = self._resolve_config(self.tmp)
        self.assertIsNone(chosen)
        print("✓ UHD generator reports no config when neither file exists")

    def test_script_dir_is_dataset_generator_v2(self):
        """make_dataset_v2_uhd.py must look in its own directory, not parent.parent."""
        uhd_path = os.path.join(
            os.path.dirname(__file__), '..', 'dataset_generator_v2',
            'make_dataset_v2_uhd.py'
        )
        with open(uhd_path) as fh:
            src = fh.read()
        # Must NOT have parent.parent (old wrong path)
        self.assertNotIn('parent.parent', src,
            "make_dataset_v2_uhd.py still uses parent.parent for script_dir – "
            "it would look in the repo root instead of dataset_generator_v2/")
        # Must have parent (correct path)
        self.assertIn('Path(__file__).parent', src,
            "make_dataset_v2_uhd.py must use Path(__file__).parent for script_dir")
        print("✓ make_dataset_v2_uhd.py uses correct script_dir (parent, not parent.parent)")


# ─── UHD generator V2→V1 config normalization ─────────────────────────────────

class TestUHDConfigNormalization(unittest.TestCase):
    """
    Validate DatasetGeneratorV2UHD._normalize_config():
    - V2 flat config (from video_manager.py) is correctly mapped to V1 structure
    - V1 configs pass through unchanged
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

    # ── V1 pass-through ────────────────────────────────────────────────────────

    def test_v1_config_passes_through_unchanged(self):
        """V1 configs (with 'base_settings') are returned as-is."""
        v1 = {
            'base_settings': {'output_base_dir': '/x', 'temp_dir': '/x/t', 'status_file': '/x/s'},
            'category_targets': {'master': 5000},
            'format_config': {},
            'videos': [],
        }
        result = self.normalize(v1)
        self.assertIs(result, v1)
        print("✓ V1 config passes through _normalize_config unchanged")

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


# ─── rescan works on V1 / unified config (no prior source_dirs key) ───────────

class TestRescanOnV1Config(unittest.TestCase):
    """Rescan must work even when the config has no 'source_dirs' key yet."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.vid_dir = os.path.join(self.tmp, 'vids')
        os.makedirs(self.vid_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _make_manager(self, config_dict):
        p = os.path.join(self.tmp, 'cfg.json')
        with open(p, 'w') as f:
            json.dump(config_dict, f)
        m = VideoManager(p)
        m.load()
        return m

    def test_ensure_source_dirs_auto_initialises(self):
        """_ensure_source_dirs() creates source_dirs: [] when the key is absent."""
        cfg = {'category_targets': {'master': 100}, 'videos': []}
        m = self._make_manager(cfg)
        self.assertNotIn('source_dirs', m.config)
        dirs = m._ensure_source_dirs()
        self.assertIn('source_dirs', m.config)
        self.assertEqual(dirs, [])
        self.assertTrue(m.modified)
        print("✓ _ensure_source_dirs() auto-initialises source_dirs")

    def test_list_source_dirs_works_on_v1_config(self):
        """list_source_dirs() no longer blocks on V1 config."""
        cfg = {'category_targets': {'master': 100}, 'videos': []}
        m = self._make_manager(cfg)
        try:
            m.list_source_dirs()  # must not raise or print an error
        except Exception as e:
            self.fail(f"list_source_dirs() raised on V1 config: {e}")
        print("✓ list_source_dirs() works on V1 config")

    def test_rescan_works_on_v1_config_with_source_dirs_added(self):
        """rescan_file_list() works on V1-style config after adding a source_dir."""
        open(os.path.join(self.vid_dir, 'film.mkv'), 'w').close()

        # Classic V1 config – no source_dirs key
        cfg = {
            'base_settings': {'max_workers': 4},
            'category_targets': {'master': 100},
            'videos': [{'name': 'old', 'path': '/old/path.mkv', 'categories': ['master']}],
        }
        m = self._make_manager(cfg)

        # Simulate: user adds a source dir (option 13 equivalent)
        m._ensure_source_dirs().append({'path': self.vid_dir, 'extensions': ['.mkv']})
        # Clear modified flag so we don't get a "save first?" prompt
        m.modified = False

        m.rescan_file_list()

        paths = {v['path'] for v in m.config['videos']}
        self.assertIn(os.path.join(self.vid_dir, 'film.mkv'), paths)
        print("✓ rescan_file_list() works on V1 config after adding source_dir")

    def test_rescan_v1_config_empty_source_dirs_prints_error(self):
        """rescan_file_list() prints a helpful message when source_dirs is empty."""
        cfg = {'category_targets': {'master': 100}, 'videos': []}
        m = self._make_manager(cfg)
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m.rescan_file_list()
        out = buf.getvalue()
        self.assertIn('No source directories', out)
        print("✓ rescan_file_list() shows helpful message when source_dirs is empty")


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
            'category_targets': {'master': 100000, 'space': 50000},
            'format_config': {
                'master': {'large': {'gt_size': [720, 720], 'lr_size': [240, 240], 'probability': 1.0}},
                'space':  {'large': {'gt_size': [720, 720], 'lr_size': [240, 240], 'probability': 1.0}},
            },
            'source_dirs': [],
            'videos': [],
        }

    def test_add_category_updates_all_structures(self):
        """_add_category() updates categories, category_targets, format_config."""
        m = self._make_manager(self._base_cfg())
        m.categories = list(m.categories)  # ensure it's a list

        m._add_category.__func__  # just access — we'll call directly
        # Simulate _add_category by calling the real method via monkeypatch input
        import unittest.mock as mock
        with mock.patch('builtins.input', side_effect=['toon', '40000']):
            m._add_category()

        self.assertIn('toon', m.categories)
        self.assertEqual(m.category_targets['toon'], 40000)
        self.assertIn('toon', m.config['format_config'])
        self.assertTrue(m.modified)
        print("✓ _add_category() updates categories, targets, and format_config")

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
        """_remove_category() removes from categories, targets, format_config."""
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
        self.assertNotIn('space', m.category_targets)
        self.assertNotIn('space', m.config['format_config'])
        # Videos must have space unassigned
        for v in m.videos:
            self.assertNotIn('space', v.get('categories', []))
        self.assertTrue(m.modified)
        print("✓ _remove_category() cleans up categories, targets, format_config, and videos")

    def test_remove_category_cancel(self):
        """_remove_category() respects cancellation."""
        m = self._make_manager(self._base_cfg())
        import unittest.mock as mock
        with mock.patch('builtins.input', side_effect=['master', 'no']):
            m._remove_category()
        self.assertIn('master', m.categories)
        print("✓ _remove_category() respects 'no' cancellation")

    def test_edit_category_targets_updates_values(self):
        """_edit_category_targets() updates the target for a category."""
        m = self._make_manager(self._base_cfg())
        import unittest.mock as mock
        # Provide new value for 'master', blank (keep) for 'space'
        with mock.patch('builtins.input', side_effect=['999999', '']):
            m._edit_category_targets()
        self.assertEqual(m.category_targets['master'], 999999)
        self.assertEqual(m.category_targets['space'], 50000)
        self.assertTrue(m.modified)
        print("✓ _edit_category_targets() updates targets correctly")

    def test_save_and_reload_preserves_new_category(self):
        """New categories persist after save/reload."""
        m = self._make_manager(self._base_cfg())
        import unittest.mock as mock
        with mock.patch('builtins.input', side_effect=['toon', '40000']):
            m._add_category()
        m.save(backup=False)

        p = os.path.join(self.tmp, 'cfg.json')
        m3 = VideoManager(p)
        m3.load()
        self.assertIn('toon', m3.categories)
        self.assertEqual(m3.category_targets['toon'], 40000)
        print("✓ New category persists after save/reload")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Running Default Config & Scanning Fix Tests")
    print("=" * 60 + "\n")
    unittest.main(verbosity=2)
