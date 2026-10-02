"""Exercise the dependency-free metadata adapter with the existing Node runtime."""
import shutil
import subprocess
import threading
import unittest
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent.parent


@unittest.skipUnless(shutil.which("node"), "Node runtime not available")
class DownloaderMetadataTests(unittest.TestCase):
    def test_workflow_and_sidebar_metadata(self):
        source = APP_DIR / "docker/model_downloader/web/modelMetadata.js"
        result = subprocess.run(
            ["node", "--input-type=module", "-e", """
import fs from 'node:fs';
import assert from 'node:assert/strict';
const src = fs.readFileSync(process.argv[1], 'utf8');
const {collectModels, modelForRow} = await import('data:text/javascript;base64,' + Buffer.from(src).toString('base64'));
const model = {name:'z_image_turbo_bf16.safetensors', directory:'diffusion_models', url:'https://huggingface.co/test/resolve/main/z_image_turbo_bf16.safetensors'};
const graph = {models:[model], nodes:[{properties:{models:[model]}}], definitions:{subgraphs:[{models:[{...model,name:'other.safetensors'}]}]}};
const models = collectModels(graph);
assert.equal(models.length, 2);
const row = {textContent: model.name + ' diffusion_models · 11.46 GB', querySelectorAll: () => [{getAttribute:()=>model.name}]};
assert.deepEqual(modelForRow(row, models), {filename:model.name,save_path:model.directory,url:model.url});
assert.equal(modelForRow(row, []), null);
const ambiguous = [...models, {...models[0],save_path:'loras'}];
assert.deepEqual(modelForRow(row, ambiguous), models[0]);
const oldRow = {textContent:'vae / ae.safetensors', querySelectorAll:()=>[{textContent:'vae / ae.safetensors',getAttribute:()=> 'https://huggingface.co/test/resolve/main/ae.safetensors'}]};
assert.equal(modelForRow(oldRow, []).save_path, 'vae');
assert.equal(collectModels({nodes:[{properties:{models:[{name:'bad'}]}}]}).length, 0);
""", str(source)],
            capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("chromium"), "Existing Chromium not available")
    def test_sidebar_and_legacy_download_buttons_in_browser(self):
        handler = partial(SimpleHTTPRequestHandler, directory=str(APP_DIR.parent))
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/comfyui_webui/tests/fixtures/model_download.html?autorun=1"
            result = subprocess.run(
                ["chromium", "--headless", "--no-sandbox", "--disable-gpu",
                 "--disable-dev-shm-usage", "--virtual-time-budget=5000", "--dump-dom", url],
                capture_output=True, text=True, timeout=45,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('id="test-result">PASS:', result.stdout)
            self.assertNotIn('id="test-result">FAIL:', result.stdout)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
