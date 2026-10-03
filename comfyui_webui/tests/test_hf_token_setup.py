"""Interactive HF Secret provisioning with mocked Kubernetes commands."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parent.parent / "setup-hf-token.sh"


class HFTokenSetupTests(unittest.TestCase):
    def run_setup(self, directory, token, trace=False, arguments=(), **settings):
        env = os.environ.copy()
        env.update(LOG=str(directory / "commands"), FILE=str(directory / "token-path"),
                   MODE=str(directory / "mode"), TMPDIR=str(directory),
                   GET_FAIL="0", CREATE_FAIL="0", APPLY_FAIL="0", RESTART_FAIL="0")
        env.update(settings)
        result = subprocess.run(
            ["bash", "-c", r"""
kubectl() {
  printf '%s\n' "$*" >> "$LOG"
  case "$*" in
    "-n comfyui get deployment comfyui -o name") [[ "$GET_FAIL" == 0 ]] ;;
    "-n comfyui create secret generic comfyui-huggingface "*)
      local path="${7#--from-file=token=}"
      printf '%s' "$path" > "$FILE"
      stat -c '%a' "$path" > "$MODE"
      if [[ "$CREATE_FAIL" == 1 ]]; then cat "$path" >&2; return 1; fi
      cat "$path" ;;
    "-n comfyui apply --server-side --field-manager=hf-token-setup -f -")
      local payload
      payload=$(cat)
      if [[ "$APPLY_FAIL" == 1 ]]; then printf '%s' "$payload" >&2; return 1; fi ;;
    "-n comfyui rollout restart deployment/comfyui") [[ "$RESTART_FAIL" == 0 ]] ;;
    *) return 99 ;;
  esac
}
export -f kubectl
exec bash "$@"
""", "setup-tests", *(["-x"] if trace else []), str(SCRIPT), *arguments],
            input=token, env=env, text=True, capture_output=True, timeout=10,
        )
        self.assertNotIn("hf_" + "HarmlessFixture123", result.stdout + result.stderr)
        log = (directory / "commands").read_text() if (directory / "commands").exists() else ""
        self.assertNotIn("hf_" + "HarmlessFixture123", log)
        if (directory / "token-path").exists():
            path = Path((directory / "token-path").read_text())
            self.assertFalse(path.exists())
            self.assertFalse(path.parent.exists())
            self.assertEqual((directory / "mode").read_text().strip(), "600")
        return result, log

    def test_success_and_trace_do_not_expose_token(self):
        for trace in (False, True):
            with self.subTest(trace=trace), tempfile.TemporaryDirectory(dir="/tmp") as temp:
                result, log = self.run_setup(Path(temp), "hf_" + "HarmlessFixture123\n", trace=trace)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("rollout restart deployment/comfyui", log)
                self.assertNotIn("deployment/webui", log)
                self.assertNotIn("comfyui-model-api", log)
                self.assertIn("model API token unchanged", result.stdout)

    def test_invalid_input_or_eof_does_not_change_secret(self):
        for token in ("", "\n", "wrong\n", "hf_bad token\n", "hf_bad\tvalue\n"):
            with self.subTest(token=token), tempfile.TemporaryDirectory(dir="/tmp") as temp:
                result, log = self.run_setup(Path(temp), token)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("create secret", log)
                self.assertNotIn("rollout restart", log)

    def test_secret_failures_are_sanitized_and_do_not_restart(self):
        for setting in ("GET_FAIL", "CREATE_FAIL", "APPLY_FAIL"):
            with self.subTest(setting=setting), tempfile.TemporaryDirectory(dir="/tmp") as temp:
                result, log = self.run_setup(Path(temp), "hf_" + "HarmlessFixture123\n", **{setting: "1"})
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("rollout restart", log)

    def test_restart_failure_and_arguments_fail(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temp:
            result, _ = self.run_setup(Path(temp), "hf_" + "HarmlessFixture123\n", RESTART_FAIL="1")
            self.assertNotEqual(result.returncode, 0)
        with tempfile.TemporaryDirectory(dir="/tmp") as temp:
            result, log = self.run_setup(Path(temp), "", arguments=("unexpected",))
            self.assertEqual(result.returncode, 2)
            self.assertEqual(log, "")


if __name__ == "__main__":
    unittest.main()
