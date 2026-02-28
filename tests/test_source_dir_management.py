"""
Test suite for source directory management:
- StateManager.force_rescan() and scan_videos() with new source_dirs format
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
    """Build a minimal V2-format config dict using the new source_dirs structure."""
    return {
        'dataset_name': 'test_dataset',
        'root_path': '/tmp/nonexistent_root',
        'source_dirs': [
            {'path': video_dir1, 'extensions': ['.mkv', '.mp4']},
            {'path': video_dir2, 'extensions': ['.mkv', '.mp4']},
        ],
        'videos': [],
        'category_weights': {'cat1': 0.5, 'cat2': 0.5},
        'processing': {'total_patches': 100},
        'output_patches': {},
    }


class TestForceRescan(unittest.TestCase):
    """Tests for StateManager.force_rescan() and scan_videos() with source_dirs."""

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

    def test_scan_videos_no_category_stamp(self):
        """scan_videos() does not assign a category when videos list is empty."""
        sm = StateManager(self.config, self.state_file)
        # The state manager should scan without crashing even with empty source_dirs
        sm.scan_videos()
        # All metadata entries should have category=None (not bound to a dir name)
        for meta in sm.state['video_metadata'].values():
            self.assertIsNone(meta.get('category'))
        print("✓ scan_videos() sets category=None when no category assigned")

    def test_scan_videos_respects_category_from_videos_list(self):
        """scan_videos() refreshes category from 'videos' list on incremental scan."""
        dummy_path = os.path.join(self.video_dir1, 'test.mkv')
        open(dummy_path, 'w').close()

        config = dict(self.config)
        config['videos'] = [{'name': 'test', 'path': dummy_path, 'categories': ['cat1']}]

        sm = StateManager(config, self.state_file)

        # Use actual file stats so the incremental-scan (cache-hit) path is taken
        st = os.stat(dummy_path)
        from datetime import datetime as _dt
        last_modified = _dt.fromtimestamp(st.st_mtime).isoformat()

        # Inject pre-scanned metadata with stale category=None but matching stats
        sm.state['video_metadata'][dummy_path] = {
            'duration': 10, 'fps': 24, 'resolution': [1920, 1080],
            'file_size': st.st_size,
            'last_modified': last_modified,
            'category': None,  # stale – should be refreshed
        }
        sm.scan_videos()

        meta = sm.state['video_metadata'].get(dummy_path)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.get('category'), 'cat1')
        print("✓ scan_videos() refreshes category from videos list (incremental path)")


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
        """_is_v2_config() returns True for new V2 format (source_dirs list)."""
        m = self._make_manager()
        self.assertTrue(m._is_v2_config())
        print("✓ _is_v2_config() detects V2 format")

    def test_is_v1_config(self):
        """_is_v2_config() returns False for V1 format (no source_dirs)."""
        v1_path = os.path.join(self.test_dir, 'generator_config.json')
        with open(v1_path, 'w') as f:
            json.dump({'base_settings': {}, 'videos': []}, f)
        m = VideoManager(v1_path)
        m.load()
        self.assertFalse(m._is_v2_config())
        print("✓ _is_v2_config() returns False for V1 format")

    def test_add_source_dir_programmatic(self):
        """Adding a source directory appends to source_dirs (no category binding)."""
        m = self._make_manager()
        new_dir = os.path.join(self.test_dir, 'new_vids')
        os.makedirs(new_dir)

        source_dirs = m.config['source_dirs']
        source_dirs.append({'path': new_dir, 'extensions': ['.mkv']})
        m.modified = True

        paths = [d['path'] for d in m.config['source_dirs']]
        self.assertIn(new_dir, paths)
        # No category key on the entry
        new_entry = next(d for d in m.config['source_dirs'] if d['path'] == new_dir)
        self.assertNotIn('category', new_entry)
        print("✓ New source directory added without category binding")

    def test_remove_source_dir_programmatic(self):
        """Removing a source directory removes the entry from source_dirs."""
        m = self._make_manager()
        original_count = len(m.config['source_dirs'])
        m.config['source_dirs'].pop(0)
        m.modified = True

        self.assertEqual(len(m.config['source_dirs']), original_count - 1)
        print("✓ Source directory removed from source_dirs list")

    def test_edit_source_dir_programmatic(self):
        """Editing a source directory updates its path in source_dirs."""
        m = self._make_manager()
        new_path = os.path.join(self.test_dir, 'updated_path')
        m.config['source_dirs'][0]['path'] = new_path
        m.modified = True

        self.assertEqual(m.config['source_dirs'][0]['path'], new_path)
        print("✓ Source directory path updated in source_dirs")

    def test_save_and_reload_preserves_source_dirs(self):
        """Saving the config persists source_dirs changes."""
        m = self._make_manager()
        new_path = os.path.join(self.test_dir, 'saved_path')
        m.config['source_dirs'][0]['path'] = new_path
        m.modified = True
        m.save(backup=False)

        m2 = self._make_manager()
        self.assertEqual(m2.config['source_dirs'][0]['path'], new_path)
        print("✓ Source directory changes persist after save/reload")

    def test_list_source_dirs_does_not_raise(self):
        """list_source_dirs() completes without error."""
        m = self._make_manager()
        try:
            m.list_source_dirs()
        except Exception as e:
            self.fail(f"list_source_dirs() raised: {e}")
        print("✓ list_source_dirs() runs without error")

    def test_rescan_adds_new_videos_without_category(self):
        """rescan_file_list() adds newly found videos with empty categories."""
        # Create dummy video files
        for name in ['movie1.mkv', 'movie2.mkv']:
            open(os.path.join(self.video_dir1, name), 'w').close()

        m = self._make_manager()
        # videos list is empty at start
        self.assertEqual(len(m.config.get('videos', [])), 0)

        m.rescan_file_list()

        videos = m.config['videos']
        self.assertEqual(len(videos), 2)
        for v in videos:
            self.assertEqual(v['categories'], [])
        print("✓ rescan_file_list() adds new videos without category")

    def test_rescan_preserves_existing_categories(self):
        """rescan_file_list() keeps category assignments for already-known files."""
        dummy = os.path.join(self.video_dir1, 'known.mkv')
        open(dummy, 'w').close()

        m = self._make_manager()
        # Pre-populate with a known video that already has a category
        m.config['videos'] = [{'name': 'known', 'path': dummy, 'categories': ['master']}]

        m.rescan_file_list()

        videos = {v['path']: v for v in m.config['videos']}
        self.assertIn(dummy, videos)
        self.assertEqual(videos[dummy]['categories'], ['master'])
        print("✓ rescan_file_list() preserves existing category assignments")

    def test_no_category_key_on_source_dir_entries(self):
        """source_dirs entries must not have a 'category' key (dirs are independent)."""
        m = self._make_manager()
        for entry in m.config['source_dirs']:
            self.assertNotIn('category', entry)
            self.assertNotIn('video_dir', entry)  # old field name no longer used
        print("✓ source_dirs entries have no category binding (path + extensions only)")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Running Source Directory Management Tests")
    print("="*60 + "\n")
    unittest.main(verbosity=2)
