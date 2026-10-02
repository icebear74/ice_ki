"""Shared data path and restart persistence regression tests."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent


class DataDirectoryTests(unittest.TestCase):
    def run_python(self, code: str, data_dir: Path | None = None) -> None:
        env = os.environ.copy()
        env.pop("COMFYUI_WEBUI_DATA_DIR", None)
        if data_dir is not None:
            env["COMFYUI_WEBUI_DATA_DIR"] = str(data_dir)
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=APP_DIR, env=env,
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_default_data_directory_is_unchanged(self) -> None:
        self.run_python("""
from pathlib import Path
import auth, config, main, mapping_registry, template_registry
expected = Path.cwd() / 'data'
for module in (auth, config, main, mapping_registry, template_registry):
    assert module.DATA_DIR == expected, module.__name__
assert main._ALIASES_FILE == expected / 'model_aliases.json'
""")

    def test_all_components_persist_in_configured_directory_across_processes(self) -> None:
        with tempfile.TemporaryDirectory(dir=APP_DIR) as directory:
            data_dir = Path(directory) / "persistent"
            self.run_python("""
import json
import auth, config, main, mapping_registry, template_registry
for module in (auth, main, mapping_registry, template_registry):
    assert module.DATA_DIR == config.DATA_DIR, module.__name__
assert auth.bootstrap_admin()
mapping_registry.register_mapping('saved', 'Saved', template_name='saved')
template_registry.TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
path = template_registry.TEMPLATES_DIR / 'saved.json'
path.write_text(json.dumps(main.DEFAULT_WORKFLOW), encoding='utf-8')
template_registry.discover_local_templates()
main._save_aliases({'model.safetensors': 'Friendly model'})
main._setup_file_logging()
main._gen_logger.info('restart-persistence-marker')
main._gallery_dir('admin').joinpath('saved.json').write_text(
    json.dumps({'prompt_id': 'saved-prompt'}), encoding='utf-8')
for filename in ('users.json', 'bootstrap_credentials.txt', 'mappings.json',
                 'templates.json', 'model_aliases.json', 'generation.log',
                 'templates/saved.json', 'gallery/admin/saved.json'):
    assert (config.DATA_DIR / filename).is_file(), filename
""", data_dir)
            self.run_python("""
import auth, config, main, mapping_registry, template_registry
assert auth.bootstrap_admin() is None
assert auth.get_user('admin')['role'] == 'admin'
assert mapping_registry.get_mapping('saved')['template_name'] == 'saved'
assert template_registry.get_template('saved')['approved']
assert template_registry.analyze_template_file(
    template_registry.TEMPLATES_DIR / 'saved.json')['analysis']['is_usable']
assert main._load_aliases() == {'model.safetensors': 'Friendly model'}
assert main._list_gallery('admin')[0]['prompt_id'] == 'saved-prompt'
assert 'restart-persistence-marker' in (config.DATA_DIR / 'generation.log').read_text()
""", data_dir)


if __name__ == "__main__":
    unittest.main()
