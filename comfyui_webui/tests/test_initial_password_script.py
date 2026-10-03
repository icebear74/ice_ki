"""Bootstrap credential display tests without Kubernetes access."""
import os
from pathlib import Path
import subprocess
import unittest


SCRIPT = Path(__file__).resolve().parent.parent / "show-initial-password.sh"


class InitialPasswordScriptTests(unittest.TestCase):
    def run_script(self, credentials="", status=0, arguments=(), trace=False):
        env = os.environ.copy()
        env.update(CREDENTIALS=credentials, EXEC_STATUS=str(status))
        return subprocess.run(
            ["bash", "-c", r"""
kubectl() {
    [[ "$*" == "-n comfyui exec deployment/webui -c webui -- cat /data/bootstrap_credentials.txt" ]] || return 99
    printf '%s' "$CREDENTIALS"
    if [[ "$EXEC_STATUS" != 0 ]]; then
        printf 'kubectl exec failed\n' >&2
    fi
    return "$EXEC_STATUS"
}
export -f kubectl
exec bash "$@"
""", "script-tests", *(["-x"] if trace else []), str(SCRIPT), *arguments],
            env=env, capture_output=True, text=True, timeout=10,
        )

    def test_displays_credentials_without_tracing_them(self):
        credentials = "username: admin\npassword: " + "-".join(("test", "fixture"))
        for trace in (False, True):
            with self.subTest(trace=trace):
                result = self.run_script(credentials, trace=trace)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, credentials + "\n")
                self.assertNotIn(credentials, result.stderr)
                self.assertNotIn("test-fixture", result.stderr)

    def test_exec_failure_discards_partial_output(self):
        result = self.run_script("partial credentials", status=1)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("bootstrap file may already have been deleted", result.stderr)
        self.assertNotIn("partial credentials", result.stderr)

    def test_empty_file_is_rejected(self):
        for credentials in ("", " \n\t"):
            with self.subTest(credentials=credentials):
                result = self.run_script(credentials)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn("empty", result.stderr)

    def test_arguments_are_rejected_before_cluster_access(self):
        result = self.run_script(arguments=("unexpected",))
        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage:", result.stderr)
        self.assertNotIn("kubectl", result.stderr)


if __name__ == "__main__":
    unittest.main()
