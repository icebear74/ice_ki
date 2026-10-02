"""Liveness and fail-closed workflow template regression tests."""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
import template_registry as registry


class HealthTests(unittest.TestCase):
    def test_healthz_needs_no_authentication_or_backends(self) -> None:
        client = TestClient(main.app)
        self.addCleanup(client.close)
        with patch.object(main.httpx, "AsyncClient", side_effect=AssertionError("backend called")) as backend:
            response = client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        backend.assert_not_called()
        self.assertEqual(client.get("/api/admin/templates").status_code, 401)

    def test_bootstrap_credential_stdout_is_configurable(self) -> None:
        for setting, should_print in ((None, True), ("false", False)):
            with self.subTest(setting=setting):
                env = {} if setting is None else {"COMFYUI_WEBUI_LOG_BOOTSTRAP_PASSWORD": setting}
                with (
                    patch.dict(main.os.environ, env, clear=True),
                    patch.object(main, "_setup_file_logging"),
                    patch.object(main._auth, "bootstrap_admin", return_value="example-generated-credential"),
                    patch.object(registry, "get_template", return_value={"name": "default"}),
                    patch.object(registry, "discover_local_templates", return_value=[]),
                    patch("builtins.print") as output,
                ):
                    asyncio.run(main._startup())
                self.assertEqual(output.called, should_print)


class TemplateTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.data_dir = Path(directory)
        self.templates_dir = self.data_dir / "templates"
        self.templates_dir.mkdir()
        self.enterContext(patch.object(registry, "DATA_DIR", self.data_dir))
        self.enterContext(patch.object(registry, "TEMPLATES_DIR", self.templates_dir))
        self.enterContext(patch.object(registry, "TEMPLATES_FILE", self.data_dir / "templates.json"))
        self.enterContext(patch.dict(main._sessions, {"test-admin": {"username": "admin", "role": "admin"}}))
        # Do not run lifespan/bootstrap: only the routes under test are needed.
        self.client = TestClient(main.app, cookies={"ki_session": "test-admin"})
        self.addCleanup(self.client.close)

    def write_workflow(self, filename: str, workflow: object) -> None:
        (self.templates_dir / filename).write_text(json.dumps(workflow), encoding="utf-8")

    def test_local_discovery_only_auto_approves_usable_workflows(self) -> None:
        self.write_workflow("valid.json", main.DEFAULT_WORKFLOW)
        self.write_workflow("empty.json", {})
        self.write_workflow("list.json", [])
        self.write_workflow("ui.json", {"nodes": [], "links": []})
        self.write_workflow("bad_node.json", {"1": {"class_type": "KSampler", "inputs": None}})
        (self.templates_dir / "broken.json").write_text("{broken", encoding="utf-8")
        found = {t["name"]: t for t in registry.discover_local_templates()}
        self.assertEqual(len(found), 6)
        self.assertTrue(found["valid"]["approved"])
        self.assertTrue(found["valid"]["enabled"])
        for name in ("empty", "list", "ui", "bad_node", "broken"):
            with self.subTest(name=name):
                self.assertFalse(found[name]["approved"])
                self.assertFalse(found[name]["enabled"])
                self.assertFalse(found[name]["analysis"]["is_usable"])

    def test_rediscovery_preserves_disabled_templates_and_revokes_broken_approval(self) -> None:
        self.write_workflow("valid.json", main.DEFAULT_WORKFLOW)
        registry.discover_local_templates()
        registry.update_template("valid", approved=False, enabled=False)
        registry.discover_local_templates()
        self.assertFalse(registry.get_template("valid")["approved"])
        self.assertFalse(registry.get_template("valid")["enabled"])
        registry.update_template("valid", approved=True)
        self.write_workflow("valid.json", {})
        registry.discover_local_templates()
        self.assertFalse(registry.get_template("valid")["approved"])

    def test_api_rejects_metadata_only_approval_on_create_and_update(self) -> None:
        response = self.client.post("/api/admin/templates", json={
            "name": "metadata", "display_name": "Metadata", "approved": True,
        })
        self.assertEqual(response.status_code, 400)
        self.assertIsNone(registry.get_template("metadata"))
        response = self.client.post("/api/admin/templates", json={
            "name": "metadata", "display_name": "Metadata", "source": "comfyui",
        })
        self.assertEqual(response.status_code, 201)
        response = self.client.patch("/api/admin/templates/metadata", json={"approved": True})
        self.assertEqual(response.status_code, 400)
        self.assertFalse(registry.get_template("metadata")["approved"])

    def test_api_approval_uses_current_file_not_stale_analysis(self) -> None:
        self.write_workflow("valid.json", main.DEFAULT_WORKFLOW)
        registry.discover_local_templates()
        registry.update_template("valid", approved=False)
        response = self.client.patch("/api/admin/templates/valid", json={"approved": True})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["approved"])
        for content in ("{}", "{broken"):
            (self.templates_dir / "valid.json").write_text(content, encoding="utf-8")
            registry.update_template("valid", approved=False)
            response = self.client.patch("/api/admin/templates/valid", json={"approved": True})
            self.assertEqual(response.status_code, 400)
            self.assertFalse(registry.get_template("valid")["approved"])
        (self.templates_dir / "valid.json").unlink()
        response = self.client.patch("/api/admin/templates/valid", json={"approved": True})
        self.assertEqual(response.status_code, 400)

    def test_builtin_default_can_still_be_approved(self) -> None:
        registry.register_template("default", "Default", source="local", approved=False)
        response = self.client.patch("/api/admin/templates/default", json={"approved": True})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["approved"])

    def test_upload_does_not_approve_unusable_workflow(self) -> None:
        response = self.client.post("/api/admin/templates/upload", files={
            "file": ("unusable.json", b"{}", "application/json"),
        })
        self.assertEqual(response.status_code, 201)
        self.assertFalse(response.json()["approved"])
        self.assertFalse(response.json()["analysis"]["is_usable"])

    def test_selected_missing_or_invalid_template_never_falls_back(self) -> None:
        cases = (
            ("unknown", None, None),
            ("metadata", None, None),
            ("missing", "missing.json", None),
            ("broken", "broken.json", "{broken"),
            ("list", "list.json", "[]"),
            ("unusable", "unusable.json", "{}"),
            ("bad_node", "bad_node.json", '{"1":{"class_type":"KSampler","inputs":null}}'),
        )
        for name, filename, contents in cases:
            with self.subTest(name=name):
                if name != "unknown":
                    registry.register_template(name, name, filename=filename)
                if contents is not None:
                    (self.templates_dir / filename).write_text(contents, encoding="utf-8")
                with patch.object(main, "_load_default_workflow") as default_loader:
                    response = self.client.post("/api/generate", json={
                        "prompt_de": "x", "ollama_model": "m",
                        "translated_prompt": "already translated", "workflow_template": name,
                    })
                    self.assertEqual(response.status_code, 400, response.text)
                    self.assertTrue(response.json()["detail"])
                    default_loader.assert_not_called()

    def test_filename_resolution_rejects_escape_and_symlink(self) -> None:
        outside = self.data_dir / "outside.json"
        outside.write_text(json.dumps(main.DEFAULT_WORKFLOW), encoding="utf-8")
        (self.templates_dir / "link.json").symlink_to(outside)
        for filename in ("../outside.json", str(outside), "link.json"):
            with self.subTest(filename=filename):
                with self.assertRaises(ValueError):
                    registry.resolve_template_path(filename)
                registry.register_template("escape", "Escape", filename=filename)
                response = self.client.patch("/api/admin/templates/escape", json={"approved": True})
                self.assertEqual(response.status_code, 400)
                payload = main.GenerateRequest(
                    prompt_de="x", ollama_model="m", workflow_template="escape",
                )
                with patch.object(main, "_load_default_workflow") as default_loader:
                    with self.assertRaises(ValueError):
                        main._build_workflow(payload, "p", "")
                    default_loader.assert_not_called()
        self.assertEqual(registry.discover_local_templates(), [])


if __name__ == "__main__":
    unittest.main()
