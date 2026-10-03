"""Selective Ollama deployment regressions without a live cluster."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parent.parent / "deploy-ollama.sh"


class OllamaDeployScriptTests(unittest.TestCase):
    def run_script(self, directory, arguments=(), **settings):
        env = os.environ.copy()
        env.update(
            IMAGE="127.0.0.1:5000/comfyui-ollama:working",
            GET_ERROR="0", APPLY_ERROR="0", ROLLOUT_ERROR="0",
            LOG=str(directory / "commands"), OUTPUT=str(directory / "manifest"),
        )
        env.update(settings)
        result = subprocess.run(
            ["bash", "-c", r"""
kubectl() {
  printf '%s\n' "$*" >> "$LOG"
  case "$*" in
    "-n comfyui get deployment ollama "*)
      [[ "$GET_ERROR" == 0 ]] || return 1
      printf '%s' "$IMAGE" ;;
    "apply -l app=ollama -f -")
      cat > "$OUTPUT"
      [[ "$APPLY_ERROR" == 0 ]] ;;
    "-n comfyui rollout status deployment/ollama --timeout=25m")
      [[ "$ROLLOUT_ERROR" == 0 ]] ;;
    "-n comfyui describe pods -l app=ollama")
      printf 'ImagePullBackOff: inspect registry\n' ;;
    "-n comfyui exec deployment/ollama -c ollama -- ollama list")
      printf 'qwen2.5:1.5b\n' ;;
    *) return 99 ;;
  esac
}
export -f kubectl
exec bash "$@"
""", "script-tests", str(SCRIPT), *arguments],
            env=env, capture_output=True, text=True, timeout=10,
        )
        log = (directory / "commands").read_text() if (directory / "commands").exists() else ""
        return result, log

    def test_existing_image_is_used_in_all_ollama_containers(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temp:
            directory = Path(temp)
            result, log = self.run_script(directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("apply -l app=ollama -f -", log)
            rendered = (directory / "manifest").read_text()
            self.assertEqual(rendered.count("127.0.0.1:5000/comfyui-ollama:working"), 3)
            self.assertIn("registry.example.invalid:5000/comfyui-webui:1", rendered)
            self.assertIn("registry.example.invalid:5000/comfyui:1", rendered)
            self.assertIn("qwen2.5:1.5b", result.stdout)

    def test_explicit_image_repairs_wrong_configured_address(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temp:
            directory = Path(temp)
            result, log = self.run_script(directory, arguments=("registry.lan:5000/comfyui-ollama:2",))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("get deployment", log)
            self.assertEqual((directory / "manifest").read_text().count("registry.lan:5000/comfyui-ollama:2"), 3)

    def test_invalid_images_and_lookup_failure_do_not_apply(self):
        for settings in (
            {"IMAGE": ""}, {"IMAGE": "registry.example.invalid:5000/comfyui-ollama:1"},
            {"IMAGE": "bad&replacement"}, {"GET_ERROR": "1"},
        ):
            with self.subTest(settings=settings), tempfile.TemporaryDirectory(dir="/tmp") as temp:
                result, log = self.run_script(Path(temp), **settings)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("apply ", log)

    def test_apply_error_stops_and_rollout_error_shows_events(self):
        for settings, describe in (({"APPLY_ERROR": "1"}, False), ({"ROLLOUT_ERROR": "1"}, True)):
            with self.subTest(settings=settings), tempfile.TemporaryDirectory(dir="/tmp") as temp:
                result, log = self.run_script(Path(temp), **settings)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual("describe pods" in log, describe)
                self.assertNotIn("exec deployment", log)
                if describe:
                    self.assertIn("ImagePullBackOff", result.stderr)


if __name__ == "__main__":
    unittest.main()
