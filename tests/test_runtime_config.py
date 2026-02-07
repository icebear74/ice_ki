"""
Test suite for Runtime Configuration Manager
"""

import os
import json
import tempfile
import shutil
import unittest
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vsr_plus_plus.systems.runtime_config import RuntimeConfigManager


class TestRuntimeConfigManager(unittest.TestCase):
    """Test Runtime Configuration Manager"""
    
    def setUp(self):
        """Create temporary directory for tests"""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, 'runtime_config.json')
        
        self.base_config = {
            'plateau_safety_threshold': 800,
            'plateau_patience': 250,
            'max_lr': 1.5e-4,
            'min_lr': 1e-6,
            'l1_weight_target': 0.6,
            'ms_weight_target': 0.2,
            'grad_weight_target': 0.2,
            'perceptual_weight_target': 0.0,
            'n_feats': 64,
        }
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test initialization"""
        manager = RuntimeConfigManager(self.config_path, self.base_config)
        self.assertTrue(os.path.exists(self.config_path))
        self.assertEqual(manager.get('plateau_safety_threshold'), 800)
        print("✓ Initialization test passed")
    
    def test_get_and_set(self):
        """Test get and set"""
        manager = RuntimeConfigManager(self.config_path, self.base_config)
        value = manager.get('plateau_patience')
        self.assertEqual(value, 250)
        
        success = manager.set('plateau_patience', 300)
        self.assertTrue(success)
        self.assertEqual(manager.get('plateau_patience'), 300)
        print("✓ Get/Set test passed")
    
    def test_snapshot_creation(self):
        """Test snapshots"""
        manager = RuntimeConfigManager(self.config_path, self.base_config)
        snapshot_path = manager.save_snapshot(1000)
        self.assertTrue(os.path.exists(snapshot_path))
        print("✓ Snapshot test passed")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Running Runtime Config Tests")
    print("="*60 + "\n")
    
    unittest.main(verbosity=2)
