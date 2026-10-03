"""Container build and rendered Service regression tests (no Docker required)."""
import os
import configparser
from pathlib import Path
import subprocess
import tempfile
import unittest


APP_DIR = Path(__file__).resolve().parent.parent


class ContainerDeploymentTests(unittest.TestCase):
    def test_ollama_bootstraps_small_model_on_persistent_cpu_storage(self):
        manifest = (APP_DIR / "k8s/deploy.yaml").read_text()
        deployment = next(
            doc for doc in manifest.split("---")
            if "\nkind: Deployment\n" in doc and "\n  name: ollama\n" in doc
        )
        self.assertIn("  labels:\n    app: ollama\n", deployment)
        self.assertEqual(manifest.count("  labels:\n    app: ollama\n"), 1)
        init, runtime = deployment.split("      containers:\n")
        self.assertIn("name: bootstrap-translation-model", init)
        self.assertIn("ollama serve &", init)
        self.assertIn('kill -0 "$server_pid"', init)
        self.assertIn('ollama show "$OLLAMA_DEFAULT_MODEL"', init)
        self.assertIn('ollama pull "$OLLAMA_DEFAULT_MODEL"', init)
        self.assertIn('trap \'kill "$server_pid"; wait "$server_pid" || true\' EXIT', init)
        self.assertIn("value: qwen2.5:1.5b", init)
        for section in (init, runtime):
            self.assertIn("name: OLLAMA_MODELS\n              value: /data/models", section)
            self.assertIn('name: CUDA_VISIBLE_DEVICES\n              value: "-1"', section)
            self.assertIn('cpu: "28"', section)
            self.assertNotIn("nvidia.com/gpu", section)
        self.assertIn('name: OLLAMA_NUM_PARALLEL\n              value: "1"', runtime)
        self.assertIn("claimName: ollama-data", runtime)
        webui = next(
            doc for doc in manifest.split("---")
            if "\nkind: Deployment\n" in doc and "\n  name: webui\n" in doc
        )
        self.assertIn("name: OLLAMA_DEFAULT_MODEL\n              value: qwen2.5:1.5b", webui)
        self.assertIn('name: OLLAMA_NUM_THREADS\n              value: "28"', webui)

    def test_pinned_downloader_loads_image_code_without_copying_it_to_pvc(self):
        dockerfile = (APP_DIR / "Dockerfile.comfyui").read_text()
        paths = (APP_DIR / "docker/extra_model_paths.yaml").read_text()
        entrypoint = (APP_DIR / "docker/comfyui-entrypoint.sh").read_text()
        self.assertIn("MODEL_DOWNLOADER_REF=419fd24ba57d20334351ddfff28ab93c84163a67", dockerfile)
        self.assertIn("FNGarvin/ComfyUI-AutoModelDownloader/archive/${MODEL_DOWNLOADER_REF}", dockerfile)
        self.assertIn("-o /opt/model-downloader.tar.gz", dockerfile)
        self.assertNotIn("rm /opt/model-downloader.tar.gz", dockerfile)
        self.assertIn("rm -rf /opt/model-custom-nodes/ComfyUI-AutoModelDownloader/web", dockerfile)
        self.assertIn("rm -f /opt/model-custom-nodes/ComfyUI-AutoModelDownloader/prestartup_script.py", dockerfile)
        self.assertLess(dockerfile.index("rm -rf /opt/model-custom-nodes"),
                        dockerfile.index("COPY docker/model_downloader/"))
        self.assertIn("COPY docker/model_downloader/ /opt/model-custom-nodes/ComfyUI-AutoModelDownloader/", dockerfile)
        self.assertIn('"--base-directory", "/data"', dockerfile)
        self.assertIn('"--extra-model-paths-config", "/opt/extra_model_paths.yaml"', dockerfile)
        self.assertIn("base_path: /opt/model-custom-nodes", paths)
        self.assertIn("custom_nodes: .", paths)
        self.assertNotIn("cp /opt/model-custom-nodes", entrypoint)
        self.assertNotIn("git clone", entrypoint)
        self.assertNotIn("curl ", entrypoint)

    def run_build(self, directory, **settings):
        output = directory / "deploy.yaml"
        log = directory / "commands.log"
        env = os.environ.copy()
        for key in ("IMAGE_TAG", "OLLAMA_IMAGE", "WEBUI_NODEPORT", "COMFYUI_NODEPORT"):
            env.pop(key, None)
        env.update(settings)
        env.update(SCRIPT=str(APP_DIR / "build-and-push.sh"),
                   OUTPUT=str(output), COMMAND_LOG=str(log))
        result = subprocess.run(
            ["bash", "-c", """
docker() { printf 'docker %s\\n' "$*" >> "$COMMAND_LOG"; }
curl() { printf 'curl %s\\n' "$*" >> "$COMMAND_LOG"; }
export -f docker curl
bash "$SCRIPT" registry.lan:5000 "$OUTPUT"
"""],
            env=env, capture_output=True, text=True, timeout=30,
        )
        return result, output, log

    def test_rendered_manifest_contains_both_nodeport_services(self):
        for settings, ports in (
            ({}, (30080, 30188)),
            ({"WEBUI_NODEPORT": "31080", "COMFYUI_NODEPORT": "31188"}, (31080, 31188)),
            ({"WEBUI_NODEPORT": "30188", "COMFYUI_NODEPORT": "30080"}, (30188, 30080)),
        ):
            with self.subTest(settings=settings), tempfile.TemporaryDirectory() as temp:
                result, output, log = self.run_build(Path(temp), IMAGE_TAG="p100-2", **settings)
                self.assertEqual(result.returncode, 0, result.stderr)
                manifest = output.read_text()
                services = [doc for doc in manifest.split("---") if "\nkind: Service\n" in doc]
                self.assertEqual(len(services), 3)
                for name, port, target in (("webui", ports[0], 8080), ("comfyui", ports[1], 8188)):
                    service = next(doc for doc in services if f"\n  name: {name}\n" in doc)
                    self.assertIn("\n  namespace: comfyui\n", service)
                    self.assertIn("type: NodePort", service)
                    self.assertIn(f"nodePort: {port}", service)
                    self.assertIn(f"port: {target}", service)
                    self.assertIn(f"app: {name}", service)
                    self.assertIn(f"http://<node-ip>:{port}", result.stdout)
                self.assertNotIn("registry.example.invalid", manifest)
                for image in ("comfyui-webui", "comfyui", "comfyui-ollama"):
                    self.assertIn(f"registry.lan:5000/{image}:p100-2", manifest)
                self.assertEqual(log.read_text().count("docker push "), 3)

    def test_invalid_or_duplicate_ports_fail_before_build(self):
        for settings in (
            {"COMFYUI_NODEPORT": "29999"},
            {"WEBUI_NODEPORT": "32768"},
            {"COMFYUI_NODEPORT": "abc"},
            {"COMFYUI_NODEPORT": "30080"},
        ):
            with self.subTest(settings=settings), tempfile.TemporaryDirectory() as temp:
                result, output, log = self.run_build(Path(temp), **settings)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(output.exists())
                self.assertFalse(log.exists())

    def test_cuda126_wheels_are_pinned_through_comfyui_install(self):
        dockerfile = (APP_DIR / "Dockerfile.comfyui").read_text()
        constraints = (APP_DIR / "docker/torch-constraints.txt").read_text()
        self.assertIn("torch==2.8.0+cu126", constraints)
        self.assertIn("torchvision==0.23.0+cu126", constraints)
        self.assertIn("--index-url https://download.pytorch.org/whl/cu126", dockerfile)
        self.assertNotIn("cu128", dockerfile)
        self.assertIn("-r /opt/torch-constraints.txt", dockerfile)
        self.assertIn("-r requirements.txt -c /opt/torch-constraints.txt", dockerfile)
        self.assertIn("torch._C._cuda_getArchFlags()", dockerfile)
        self.assertIn("'sm_60'", dockerfile)

    def test_manager_is_built_in_and_enabled_with_persistent_configuration(self):
        dockerfile = (APP_DIR / "Dockerfile.comfyui").read_text()
        entrypoint = (APP_DIR / "docker/comfyui-entrypoint.sh").read_text()
        self.assertIn("curl git ", dockerfile)
        self.assertIn("-r manager_requirements.txt -c /opt/torch-constraints.txt", dockerfile)
        self.assertIn("PIP_CONSTRAINT=/opt/torch-constraints.txt", dockerfile)
        self.assertIn('"--enable-manager"', dockerfile)
        self.assertIn('"--enable-manager-legacy-ui"', dockerfile)
        self.assertIn('"/data"', dockerfile)
        self.assertIn("COPY docker/manager-config.ini /opt/manager-config.ini", dockerfile)
        self.assertIn("if [ ! -e /data/user/__manager/config.ini ]; then", entrypoint)
        self.assertIn("cp /opt/manager-config.ini /data/user/__manager/config.ini", entrypoint)
        self.assertNotIn("pip install", entrypoint)
        self.assertNotIn("git clone", entrypoint)
        config = configparser.ConfigParser()
        config.read(APP_DIR / "docker/manager-config.ini")
        self.assertEqual(config["default"]["network_mode"], "personal_cloud")
        self.assertEqual(config["default"]["security_level"], "normal")
        self.assertFalse(config["default"].getboolean("allow_git_url_install"))
        self.assertFalse(config["default"].getboolean("allow_pip_install"))


if __name__ == "__main__":
    unittest.main()
