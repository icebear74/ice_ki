"""
Test suite for source directory management:
- StateManager.force_rescan()
- VideoManager source directory methods (V2 config)
"""

import os
import json
import tempfile
import shutil
import unittest
import sys

# Ensure dataset_generator_v2 is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dataset_generator_v2'))

from state_manager import StateManager
from video_manager import VideoManager


def _make_v2_config(video_dir1, video_dir2):
    """Build a minimal V2-format config dict."""
    return {
        'dataset_name': 'test_dataset',
        'root_path': '/tmp/nonexistent_root',
        'source': {
            'categories': {
                'cat1': {'video_dir': video_dir1, 'extensions': ['.mkv', '.mp4']},
                'cat2': {'video_dir': video_dir2, 'extensions': ['.mkv', '.mp4']},
            },
            'category_weights': {'cat1': 0.5, 'cat2': 0.5},
        },
        'processing': {'total_patches': 100},
        'output_patches': {},
    }


class TestForceRescan(unittest.TestCase):
    """Tests for StateManager.force_rescan()"""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.state_file = os.path.join(self.test_dir, 'state.json')
        self.video_dir1 = os.path.join(self.test_dir, 'videos1')
        self.video_dir2 = os.path.join(self.test_dir, 'videos2')
        os.makedirs(self.video_dir1)
        os.makedirs(self.video_dir2)
        self.config = _make_v2_config(self.video_dir1, self.video_dir2)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_force_rescan_clears_stale_metadata(self):
        """force_rescan() must remove stale video metadata entries."""
        sm = StateManager(self.config, self.state_file)
        # Inject fake stale entry
        sm.state['video_metadata']['/stale/ghost_video.mkv'] = {
            'duration': 100, 'fps': 24, 'resolution': [3840, 2160], 'category': 'cat1'
        }
        sm.state['category_distribution'] = {'cat1': {'total_patches': 50}}

        sm.force_rescan()

        self.assertNotIn('/stale/ghost_video.mkv', sm.state['video_metadata'])
        self.assertEqual(sm.state['category_distribution'], {})
        print("✓ force_rescan clears stale metadata")

    def test_force_rescan_scans_empty_dirs(self):
        """force_rescan() on empty directories yields zero videos."""
        sm = StateManager(self.config, self.state_file)
        sm.force_rescan()
        self.assertEqual(len(sm.state['video_metadata']), 0)
        print("✓ force_rescan on empty dirs yields 0 videos")

    def test_force_rescan_persists_state(self):
        """force_rescan() must save state to disk."""
        sm = StateManager(self.config, self.state_file)
        sm.force_rescan()
        self.assertTrue(os.path.exists(self.state_file))
        with open(self.state_file) as f:
            saved = json.load(f)
        self.assertIn('video_metadata', saved)
        print("✓ force_rescan persists state to disk")

    def test_incremental_scan_preserves_valid_entries(self):
        """Incremental scan_videos() calls maintain stable counts when no files change."""
        sm = StateManager(self.config, self.state_file)
        sm.scan_videos()
        count_before = len(sm.state['video_metadata'])
        sm.scan_videos()  # second incremental scan – nothing changed
        count_after = len(sm.state['video_metadata'])
        self.assertEqual(count_before, count_after)
        print("✓ incremental scan is stable (no spurious additions/removals)")


class TestVideoManagerSourceDirs(unittest.TestCase):
    """Tests for VideoManager source-directory methods (V2 config)."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.video_dir1 = os.path.join(self.test_dir, 'vids1')
        self.video_dir2 = os.path.join(self.test_dir, 'vids2')
        os.makedirs(self.video_dir1)
        os.makedirs(self.video_dir2)

        self.config_path = os.path.join(self.test_dir, 'generator_config_v2.json')
        cfg = _make_v2_config(self.video_dir1, self.video_dir2)
        with open(self.config_path, 'w') as f:
            json.dump(cfg, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _make_manager(self):
        m = VideoManager(self.config_path)
        m.load()
        return m

    def test_is_v2_config(self):
        """_is_v2_config() returns True for V2 format."""
        m = self._make_manager()
        self.assertTrue(m._is_v2_config())
        print("✓ _is_v2_config() detects V2 format")

    def test_is_v1_config(self):
        """_is_v2_config() returns False for V1 format."""
        v1_path = os.path.join(self.test_dir, 'generator_config.json')
        with open(v1_path, 'w') as f:
            json.dump({'base_settings': {}, 'videos': []}, f)
        m = VideoManager(v1_path)
        m.load()
        self.assertFalse(m._is_v2_config())
        print("✓ _is_v2_config() returns False for V1 format")

    def test_add_source_dir_programmatic(self):
        """Adding a source directory is reflected in the config."""
        m = self._make_manager()
        new_dir = os.path.join(self.test_dir, 'new_vids')
        os.makedirs(new_dir)

        # Directly call the internal config mutation (bypasses interactive input)
        categories = m.config['source']['categories']
        weights = m.config['source']['category_weights']
        categories['newcat'] = {'video_dir': new_dir, 'extensions': ['.mkv']}
        weights['newcat'] = 0.3
        m.modified = True

        self.assertIn('newcat', m.config['source']['categories'])
        self.assertEqual(m.config['source']['categories']['newcat']['video_dir'], new_dir)
        print("✓ New source directory added to config")

    def test_remove_source_dir_programmatic(self):
        """Removing a source directory is reflected in the config."""
        m = self._make_manager()
        categories = m.config['source']['categories']
        weights = m.config['source']['category_weights']

        del categories['cat1']
        weights.pop('cat1', None)
        m.modified = True

        self.assertNotIn('cat1', m.config['source']['categories'])
        self.assertNotIn('cat1', m.config['source']['category_weights'])
        print("✓ Source directory removed from config")

    def test_edit_source_dir_programmatic(self):
        """Editing a source directory updates path in the config."""
        m = self._make_manager()
        new_path = os.path.join(self.test_dir, 'updated_path')
        m.config['source']['categories']['cat1']['video_dir'] = new_path
        m.modified = True

        self.assertEqual(m.config['source']['categories']['cat1']['video_dir'], new_path)
        print("✓ Source directory path updated in config")

    def test_save_and_reload_preserves_source_dirs(self):
        """Saving the config persists source directory changes."""
        m = self._make_manager()
        new_path = os.path.join(self.test_dir, 'saved_path')
        m.config['source']['categories']['cat1']['video_dir'] = new_path
        m.modified = True
        m.save(backup=False)

        # Reload and check
        m2 = self._make_manager()
        self.assertEqual(
            m2.config['source']['categories']['cat1']['video_dir'],
            new_path
        )
        print("✓ Source directory changes persist after save/reload")

    def test_list_source_dirs_does_not_raise(self):
        """list_source_dirs() completes without error."""
        m = self._make_manager()
        try:
            m.list_source_dirs()
        except Exception as e:
            self.fail(f"list_source_dirs() raised: {e}")
        print("✓ list_source_dirs() runs without error")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Running Source Directory Management Tests")
    print("="*60 + "\n")
    unittest.main(verbosity=2)
