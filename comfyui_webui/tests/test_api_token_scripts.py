"""Exercise token scripts with shell doubles, never a cluster or image build."""
import base64
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


APP_DIR = Path(__file__).resolve().parent.parent
MOCKS = r"""
kubectl() {
    printf 'kubectl %s\n' "$*" >> "$COMMAND_LOG"
    if [[ "${1:-}" == "-n" ]]; then
        [[ "$2" == comfyui ]] || return 97
        shift 2
    fi
    case "$*" in
        "get secret comfyui-model-api "*)
            if [[ "$*" == *"jsonpath={.data.token}"* ]]; then
                if [[ "$GET_ERROR" == token ]]; then
                    printf 'Forbidden: cannot get secrets\n' >&2
                    return 23
                fi
                printf '%s' "$ENCODED"
            elif [[ "$*" == *"jsonpath={.metadata.resourceVersion}"* ]]; then
                printf '%s' "$RESOURCE_VERSION"
            elif [[ "$*" == *"-o name" ]]; then
                if [[ "$GET_ERROR" == name ]]; then
                    printf 'Forbidden: cannot get secrets\n' >&2
                    return 23
                fi
                printf '%s' "$SECRET_NAME"
            else
                return 97
            fi
            ;;
        "create namespace comfyui --dry-run=client -o yaml")
            printf 'kind: Namespace\n'
            ;;
        "apply -f -")
            cat > /dev/null
            ;;
        "create secret generic comfyui-model-api --from-file=token="*)
            local token_file="${5#--from-file=token=}"
            printf '%s\n' "$token_file" > "$TOKEN_PATH"
            stat -c '%a' "$token_file" > "$TOKEN_MODE"
            stat -c '%a' "$(dirname -- "$token_file")" > "$DIRECTORY_MODE"
            cp -- "$token_file" "$CREATED_PAYLOAD"
            if [[ "$CREATE_FAIL" == 1 ]]; then
                printf 'Secret creation rejected\n' >&2
                return 24
            fi
            ;;
        "apply -f "*)
            [[ "$3" == "$MANIFEST" ]] || return 97
            [[ -s "$3" ]] || return 97
            cp -- "$3" "$APPLIED_MANIFEST"
            if [[ "$APPLY_FAIL" == 1 ]]; then
                printf 'Manifest apply rejected\n' >&2
                return 25
            fi
            ;;
        "patch deployment comfyui webui --type=merge -p "*)
            local expected_patch="{\"spec\":{\"template\":{\"metadata\":{\"annotations\":{\"comfyui.ice-ki/model-api-secret-version\":\"${RESOURCE_VERSION}\"}}}}}"
            [[ $# == 7 && "$7" == "$expected_patch" ]] || return 97
            ;;
        *)
            printf 'Unexpected kubectl arguments: %s\n' "$*" >&2
            return 97
            ;;
    esac
}
openssl() {
    printf 'openssl %s\n' "$*" >> "$COMMAND_LOG"
    [[ "$*" == "rand -hex 32" ]] || return 97
    printf '%s\n' "$GENERATED"
}
docker() { printf 'docker %s\n' "$*" >> "$COMMAND_LOG"; }
curl() { printf 'curl %s\n' "$*" >> "$COMMAND_LOG"; }
export -f kubectl openssl docker curl
exec bash "$@"
"""


class ApiTokenScriptTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory(
            prefix="token-script-tests-", dir="/tmp"
        )
        self.addCleanup(self.workspace.cleanup)
        self.directory = Path(self.workspace.name)
        self.token_directory = self.directory / "token-files"
        self.token_directory.mkdir()
        self.manifest = self.directory / "rendered.yaml"
        self.manifest.write_text("kind: List\nitems: []\n")
        self.log = self.directory / "commands.log"
        self.generated = "-".join(("harmless", "generated", "fixture"))
        self.existing = "-".join(("harmless", "existing", "fixture"))
        self.env = os.environ.copy()
        for key in (
            "IMAGE_TAG", "OLLAMA_IMAGE", "WEBUI_NODEPORT", "COMFYUI_NODEPORT",
            "BASH_ENV", "ENV", "BASH_XTRACEFD",
        ):
            self.env.pop(key, None)
        self.env.update(
            TMPDIR=str(self.token_directory),
            COMMAND_LOG=str(self.log),
            MANIFEST=str(self.manifest),
            TOKEN_PATH=str(self.directory / "token-path"),
            TOKEN_MODE=str(self.directory / "token-mode"),
            DIRECTORY_MODE=str(self.directory / "directory-mode"),
            CREATED_PAYLOAD=str(self.directory / "created-payload"),
            APPLIED_MANIFEST=str(self.directory / "applied-manifest"),
            GENERATED=self.generated,
            ENCODED="",
            SECRET_NAME="",
            GET_ERROR="",
            CREATE_FAIL="0",
            APPLY_FAIL="0",
            RESOURCE_VERSION="12345",
        )

    def encode(self, token):
        return base64.b64encode(token.encode()).decode()

    def run_script(self, script, *arguments, trace=False, **settings):
        env = dict(self.env, **settings)
        invocation = ["-x"] if trace else []
        result = subprocess.run(
            ["bash", "-c", MOCKS, "script-tests", *invocation,
             str(APP_DIR / script), *map(str, arguments)],
            env=env, capture_output=True, text=True, timeout=30,
        )
        if script != "show-api-token.sh":
            for marker in (self.generated, self.existing, env["ENCODED"]):
                if marker:
                    self.assertNotIn(marker, result.stdout)
                    self.assertNotIn(marker, result.stderr)
        return result

    def commands(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def assert_no_mutations(self):
        commands = self.commands()
        self.assertFalse(any(" create " in command for command in commands))
        self.assertFalse(any(" apply " in command for command in commands))
        self.assertFalse(any("rollout restart" in command for command in commands))
        self.assertFalse(any(" patch " in command for command in commands))
        self.assertFalse(any(command.startswith("openssl ") for command in commands))
        self.assertEqual(list(self.token_directory.iterdir()), [])

    def assert_token_cleaned_up(self):
        token_file = Path((self.directory / "token-path").read_text().strip())
        self.assertEqual(token_file.parent.parent, self.token_directory)
        self.assertFalse(token_file.exists())
        self.assertFalse(token_file.parent.exists())
        self.assertEqual(list(self.token_directory.iterdir()), [])
        self.assertEqual((self.directory / "token-mode").read_text().strip(), "600")
        self.assertEqual((self.directory / "directory-mode").read_text().strip(), "700")

    def assert_version_apply_patch(self, version="12345"):
        commands = self.commands()
        version_get = (
            "kubectl -n comfyui get secret comfyui-model-api "
            "-o jsonpath={.metadata.resourceVersion}"
        )
        apply = f"kubectl apply -f {self.manifest}"
        payload = json.dumps({
            "spec": {"template": {"metadata": {"annotations": {
                "comfyui.ice-ki/model-api-secret-version": version,
            }}}},
        }, separators=(",", ":"))
        patch = (
            "kubectl -n comfyui patch deployment comfyui webui "
            f"--type=merge -p {payload}"
        )
        for command in (version_get, apply, patch):
            self.assertEqual(commands.count(command), 1)
        self.assertLess(commands.index(version_get), commands.index(apply))
        self.assertLess(commands.index(apply), commands.index(patch))
        self.assertFalse(any("rollout restart" in command for command in commands))

    def test_absent_secret_generates_token_and_patches_after_apply(self):
        for trace in (False, True):
            with self.subTest(trace=trace):
                result = self.run_script("deploy.sh", self.manifest, trace=trace)
                self.assertEqual(result.returncode, 0, result.stderr)
                commands = self.commands()
                self.assertIn("openssl rand -hex 32", commands)
                self.assertIn("kubectl create namespace comfyui --dry-run=client -o yaml",
                              commands)
                self.assertIn("kubectl apply -f -", commands)
                creation = next(command for command in commands
                                if "create secret generic" in command)
                apply = f"kubectl apply -f {self.manifest}"
                self.assertLess(commands.index(creation), commands.index(apply))
                self.assert_version_apply_patch()
                self.assertEqual((self.directory / "created-payload").read_text(),
                                 self.generated + "\n")
                self.assertEqual((self.directory / "applied-manifest").read_text(),
                                 self.manifest.read_text())
                self.assertIn("show-api-token.sh", result.stdout)
                self.assert_token_cleaned_up()
                self.log.unlink()

    def test_existing_token_is_preserved_and_version_patched_without_generation(self):
        for trace in (False, True):
            with self.subTest(trace=trace):
                result = self.run_script("deploy.sh", self.manifest, trace=trace,
                                         ENCODED=self.encode(self.existing),
                                         RESOURCE_VERSION="67890")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("preserved", result.stdout)
                self.assertEqual(len(self.commands()), 4)
                self.assert_version_apply_patch("67890")
                self.assertFalse(any(command.startswith("openssl ")
                                     for command in self.commands()))
                self.assertFalse(any(" create " in command
                                     for command in self.commands()))
                self.assertEqual(list(self.token_directory.iterdir()), [])
                self.assertFalse((self.directory / "created-payload").exists())
                self.log.unlink()

    def test_existing_secret_with_missing_key_is_not_overwritten(self):
        result = self.run_script("deploy.sh", self.manifest,
                                 SECRET_NAME="secret/comfyui-model-api")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no token key", result.stderr)
        self.assertEqual(len(self.commands()), 2)
        self.assert_no_mutations()

    def test_whitespace_token_is_not_overwritten(self):
        result = self.run_script("deploy.sh", self.manifest, trace=True,
                                 ENCODED=self.encode(" \t\n "))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("empty", result.stderr)
        self.assertEqual(len(self.commands()), 1)
        self.assert_no_mutations()

    def test_rbac_errors_abort_both_absence_checks(self):
        for stage in ("token", "name"):
            with self.subTest(stage=stage):
                result = self.run_script("deploy.sh", self.manifest, GET_ERROR=stage)
                self.assertEqual(result.returncode, 23)
                self.assertIn("Forbidden", result.stderr)
                self.assert_no_mutations()
                self.log.unlink()

    def test_apply_failure_never_patches_deployments(self):
        result = self.run_script("deploy.sh", self.manifest, trace=True, APPLY_FAIL="1")
        self.assertEqual(result.returncode, 25)
        self.assertIn("Manifest apply rejected", result.stderr)
        self.assertIn(f"kubectl apply -f {self.manifest}", self.commands())
        self.assertFalse(any("rollout restart" in command for command in self.commands()))
        self.assertFalse(any(" patch " in command for command in self.commands()))
        self.assert_token_cleaned_up()

    def test_retry_after_apply_failure_patches_existing_secret_version(self):
        failed = self.run_script("deploy.sh", self.manifest, APPLY_FAIL="1")
        self.assertEqual(failed.returncode, 25)
        self.assertFalse(any(" patch " in command for command in self.commands()))
        self.assert_token_cleaned_up()
        preserved_token = (self.directory / "created-payload").read_text()
        self.log.unlink()

        result = self.run_script("deploy.sh", self.manifest,
                                 ENCODED=self.encode(preserved_token))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("preserved", result.stdout)
        self.assert_version_apply_patch()
        self.assertFalse(any(command.startswith("openssl ")
                             or " create " in command for command in self.commands()))
        self.assertEqual((self.directory / "created-payload").read_text(),
                         preserved_token)
        self.assertEqual(list(self.token_directory.iterdir()), [])

    def test_invalid_secret_version_aborts_before_apply_or_patch(self):
        for version in ("", "not-numeric", "123\n456", '123"}'):
            with self.subTest(version=version):
                result = self.run_script("deploy.sh", self.manifest, trace=True,
                                         ENCODED=self.encode(self.existing),
                                         RESOURCE_VERSION=version)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Cannot determine model API Secret version", result.stderr)
                self.assertEqual(len(self.commands()), 2)
                self.assertIn(
                    "kubectl -n comfyui get secret comfyui-model-api "
                    "-o jsonpath={.metadata.resourceVersion}", self.commands())
                self.assert_no_mutations()
                self.log.unlink()

    def test_secret_creation_failure_cleans_up_and_does_not_apply_manifest(self):
        result = self.run_script("deploy.sh", self.manifest, trace=True, CREATE_FAIL="1")
        self.assertEqual(result.returncode, 24)
        self.assertIn("Secret creation rejected", result.stderr)
        self.assertNotIn(f"kubectl apply -f {self.manifest}", self.commands())
        self.assertFalse(any("rollout restart" in command for command in self.commands()))
        self.assertFalse(any(" patch " in command for command in self.commands()))
        self.assert_token_cleaned_up()

    def test_display_prints_decoded_token(self):
        result = self.run_script("show-api-token.sh", ENCODED=self.encode(self.existing))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, self.existing + "\n")
        self.assertEqual(result.stderr, "")
        self.assertEqual(self.commands(), [
            "kubectl -n comfyui get secret comfyui-model-api -o jsonpath={.data.token}"
        ])
        self.assert_no_mutations()

    def test_display_does_not_swallow_upstream_error(self):
        result = self.run_script("show-api-token.sh", GET_ERROR="token")
        self.assertEqual(result.returncode, 23)
        self.assertIn("Forbidden", result.stderr)
        self.assertNotIn("no token", result.stderr)
        self.assertEqual(result.stdout, "")
        self.assert_no_mutations()

    def test_display_rejects_missing_whitespace_and_invalid_encoding(self):
        for encoded, diagnostic in (
            ("", "no token"),
            (self.encode(" \t\n "), "empty token"),
            ("!not-base64!", "invalid input"),
        ):
            with self.subTest(encoded=encoded):
                result = self.run_script("show-api-token.sh", ENCODED=encoded)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assert_no_mutations()
                self.log.unlink()

    def test_build_deploy_provisions_token_and_applies_rendered_manifest(self):
        self.manifest.unlink()
        result = self.run_script("build-and-push.sh", "--deploy", "registry.lan:5000",
                                 self.manifest, IMAGE_TAG="fixture-build")
        self.assertEqual(result.returncode, 0, result.stderr)
        rendered = self.manifest.read_text()
        self.assertNotIn("registry.example.invalid", rendered)
        for image in ("comfyui-webui", "comfyui", "comfyui-ollama"):
            self.assertIn(f"registry.lan:5000/{image}:fixture-build", rendered)
        self.assertEqual((self.directory / "applied-manifest").read_text(), rendered)
        commands = self.commands()
        self.assertEqual(sum(command.startswith("docker build ") for command in commands), 2)
        self.assertEqual(sum(command.startswith("docker push ") for command in commands), 3)
        self.assertEqual(sum(command.startswith("curl ") for command in commands), 1)
        self.assertIn("openssl rand -hex 32", commands)
        creation = next(command for command in commands if "create secret generic" in command)
        self.assertLess(commands.index(creation),
                        commands.index(f"kubectl apply -f {self.manifest}"))
        self.assert_version_apply_patch()
        self.assertEqual((self.directory / "created-payload").read_text(),
                         self.generated + "\n")
        self.assert_token_cleaned_up()


if __name__ == "__main__":
    unittest.main()
