from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient

import main as webui_main


class OllamaConfigurationTests(unittest.TestCase):
    def test_environment_defaults_and_overrides(self) -> None:
        for settings, expected in (
            ({}, "qwen2.5:1.5b 28"),
            ({"OLLAMA_DEFAULT_MODEL": "custom:latest", "OLLAMA_NUM_THREADS": "4"}, "custom:latest 4"),
        ):
            with self.subTest(settings=settings):
                result = self.import_with_env(settings)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), expected)

    def test_thread_count_must_be_positive_integer(self) -> None:
        for value in ("0", "-1", "invalid", "1.5", ""):
            with self.subTest(value=value):
                result = self.import_with_env({"OLLAMA_NUM_THREADS": value})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("OLLAMA_NUM_THREADS must be a positive integer", result.stderr)

    def import_with_env(self, settings: dict[str, str]) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env.pop("OLLAMA_DEFAULT_MODEL", None)
        env.pop("OLLAMA_NUM_THREADS", None)
        env.update(settings)
        return subprocess.run(
            [sys.executable, "-c", "import main; print(main.OLLAMA_DEFAULT_MODEL, main.OLLAMA_NUM_THREADS)"],
            cwd=Path(webui_main.__file__).parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
        )


class OllamaRequestTests(unittest.IsolatedAsyncioTestCase):
    def mock_client(self) -> AsyncMock:
        client = AsyncMock()
        client.__aenter__.return_value = client
        self.enterContext(patch.object(webui_main.httpx, "AsyncClient", return_value=client))
        self.enterContext(patch.object(webui_main, "OLLAMA_NUM_THREADS", 6))
        return client

    def response(self, path: str, payload: dict, status: int = 200) -> httpx.Response:
        return httpx.Response(
            status,
            json=payload,
            request=httpx.Request("POST", f"{webui_main.OLLAMA_BASE_URL}{path}"),
        )

    async def test_chat_uses_cpu_and_configured_threads(self) -> None:
        client = self.mock_client()
        client.post.return_value = self.response("/api/chat", {"message": {"content": " translated "}})
        self.assertEqual(await webui_main._call_ollama_raw("instruction", "chosen:model"), "translated")
        client.post.assert_awaited_once_with(
            f"{webui_main.OLLAMA_BASE_URL}/api/chat",
            json={
                "model": "chosen:model",
                "messages": [{"role": "user", "content": "instruction"}],
                "stream": False,
                "keep_alive": -1,
                "options": {"temperature": 0.1, "num_gpu": 0, "num_thread": 6},
            },
        )

    async def test_generate_fallback_uses_cpu_and_configured_threads(self) -> None:
        client = self.mock_client()
        client.post.side_effect = [
            self.response("/api/chat", {}, 404),
            self.response("/api/generate", {"response": " translated "}),
        ]
        self.assertEqual(await webui_main._call_ollama_raw("instruction", "chosen:model"), "translated")
        self.assertEqual(client.post.await_count, 2)
        self.assertEqual(
            client.post.await_args_list[0].kwargs["json"]["options"],
            {"temperature": 0.1, "num_gpu": 0, "num_thread": 6},
        )
        self.assertEqual(client.post.await_args.args, (f"{webui_main.OLLAMA_BASE_URL}/api/generate",))
        self.assertEqual(client.post.await_args.kwargs["json"], {
            "model": "chosen:model",
            "prompt": "instruction",
            "stream": False,
            "keep_alive": -1,
            "options": {"temperature": 0.1, "num_gpu": 0, "num_thread": 6},
        })

    async def test_model_listing_exposes_configured_default(self) -> None:
        client = self.mock_client()
        client.get.return_value = self.response("/api/tags", {
            "models": [{"name": "chosen:model"}, {"name": ""}, {}],
        })
        with patch.object(webui_main, "OLLAMA_DEFAULT_MODEL", "chosen:model"):
            result = await webui_main.get_ollama_models({})
        self.assertEqual(result, {"models": ["chosen:model"], "default_model": "chosen:model"})
        client.get.assert_awaited_once_with(f"{webui_main.OLLAMA_BASE_URL}/api/tags")

    async def test_empty_model_listing_keeps_default(self) -> None:
        client = self.mock_client()
        client.get.return_value = self.response("/api/tags", {"models": []})
        result = await webui_main.get_ollama_models({})
        self.assertEqual(result, {"models": [], "default_model": webui_main.OLLAMA_DEFAULT_MODEL})

    async def test_model_listing_failure_is_not_hidden(self) -> None:
        client = self.mock_client()
        client.get.side_effect = httpx.ConnectError("offline")
        with self.assertRaises(HTTPException) as raised:
            await webui_main.get_ollama_models({})
        self.assertEqual(raised.exception.status_code, 502)
        self.assertIn("offline", raised.exception.detail)


class OllamaModelRouteTests(unittest.TestCase):
    def test_model_route_serializes_list_and_default_string(self) -> None:
        backend = AsyncMock()
        backend.__aenter__.return_value = backend
        backend.get.return_value = httpx.Response(
            200,
            json={"models": [{"name": "qwen2.5:1.5b"}]},
            request=httpx.Request("GET", f"{webui_main.OLLAMA_BASE_URL}/api/tags"),
        )
        client = TestClient(webui_main.app, cookies={"ki_session": "test-ollama"})
        self.addCleanup(client.close)
        with (
            patch.object(webui_main.httpx, "AsyncClient", return_value=backend),
            patch.dict(webui_main._sessions, {"test-ollama": {"username": "admin", "role": "admin"}}),
        ):
            response = client.get("/api/ollama/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "models": ["qwen2.5:1.5b"],
            "default_model": webui_main.OLLAMA_DEFAULT_MODEL,
        })


@unittest.skipUnless(shutil.which("node"), "Node runtime not available")
class OllamaMappingEditorTests(unittest.TestCase):
    def test_defaults_refresh_edit_choices_and_errors(self) -> None:
        result = subprocess.run(
            ["node", "-e", r"""
const fs = require("node:fs");
const vm = require("node:vm");
const assert = require("node:assert/strict");
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    tagName: "SELECT", value: "", children: [], disabled: false, textContent: "",
    classList: {
      classes: new Set(),
      add(value) { this.classes.add(value); },
      remove(value) { this.classes.delete(value); },
      contains(value) { return this.classes.has(value); },
      toggle(){},
    },
    set innerHTML(value) { this.children = []; },
    appendChild(option) { this.children.push(option); },
    insertBefore(option) { this.children.splice(1, 0, option); },
  });
  return elements.get(id);
}
const alerts = [];
const context = vm.createContext({
  document: {getElementById: element, createElement: () => ({})},
  alert: message => alerts.push(message),
});
const source = fs.readFileSync(process.argv[1], "utf8").split('$("loginBtn").addEventListener')[0];
vm.runInContext(source, context);
vm.runInContext(`
populateMappingFormSelects = () => {};
populateCheckpointSelect = () => {};
`, context);
let models = ["other:model"];
let chosen = "";
let failure = false;
let mappingFailure = false;
let modelRequests = 0;
let modelWait = null;
let mappingWait = null;
context.fetch = async path => {
  if (path === "/api/ollama/models") {
    modelRequests++;
    if (modelWait) await modelWait;
    return {
      ok: !failure, status: 502,
      json: async () => failure ? {detail:"offline"} : {models, default_model:"qwen2.5:1.5b"},
    };
  }
  assert.equal(path, "/api/admin/mappings");
  if (mappingWait) await mappingWait;
  return {ok:true, json:async () => ({
    mappings: mappingFailure ? [] : [{name:"existing", display_name:"Existing", ollama_model:chosen}],
  })};
};
async function open(name) {
  context.editorName = name;
  await vm.runInContext("openMappingForm(editorName)", context);
}
(async () => {
  await open(null);
  assert.equal(element("newMapOllamaModel").value, "");
  models = ["other:model", "qwen2.5:1.5b"];
  await open(null);
  assert.equal(modelRequests, 2);
  assert.equal(element("newMapOllamaModel").value, "qwen2.5:1.5b");
  let finishModels;
  let finishMapping;
  modelWait = new Promise(resolve => { finishModels = resolve; });
  mappingWait = new Promise(resolve => { finishMapping = resolve; });
  const pendingEdit = open("existing");
  assert(element("addMappingForm").classList.contains("hidden"));
  assert.equal(element("saveMappingBtn").disabled, true);
  finishModels();
  await new Promise(setImmediate);
  assert(element("addMappingForm").classList.contains("hidden"));
  finishMapping();
  await pendingEdit;
  assert(!element("addMappingForm").classList.contains("hidden"));
  assert.equal(element("saveMappingBtn").disabled, false);
  modelWait = null;
  mappingWait = null;
  chosen = "other:model";
  await open("existing");
  assert.equal(element("newMapOllamaModel").value, chosen);
  chosen = "";
  await open("existing");
  assert.equal(element("newMapOllamaModel").value, "");
  chosen = "missing:model";
  await open("existing");
  assert.equal(element("newMapOllamaModel").value, chosen);
  assert(element("newMapOllamaModel").children.some(option =>
    option.value === chosen && option.textContent.includes("nicht verfügbar")));
  models = [];
  await open(null);
  assert.equal(element("newMapOllamaModel").value, "");
  assert(alerts.at(-1).includes("ollama pull qwen2.5:1.5b"));
  assert(element("status").textContent.includes("Keine Ollama-Modelle"));
  failure = true;
  await open(null);
  assert(alerts.at(-1).includes("offline"));
  assert(alerts.at(-1).includes("Ollama-Dienst prüfen"));
  assert.equal(element("newMapOllamaModel").children.length, 1);
  failure = false;
  mappingFailure = true;
  await open("existing");
  assert(alerts.at(-1).includes("nicht gefunden"));
  assert.equal(element("saveMappingBtn").disabled, true);
})().catch(error => { console.error(error); process.exitCode = 1; });
""", str(Path(webui_main.__file__).parent / "static" / "app.js")],
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class TranslatePromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_translate_preserves_double_quoted_text(self) -> None:
        prompt_de = 'Erstelle ein Banner mit dem Text "Hallo Welt" in einem Wald'
        translated_raw = 'Create a banner in a forest displaying the exact text __ICEKI_LITERAL_0__'

        with patch.object(
            webui_main,
            "_call_ollama_raw",
            AsyncMock(return_value=translated_raw),
        ) as ollama_mock:
            translated = await webui_main._translate_german_to_english(prompt_de, "demo-model")

        self.assertEqual(
            translated,
            'Create a banner in a forest displaying the exact text "Hallo Welt"',
        )
        instruction = ollama_mock.await_args.args[0]
        self.assertIn("__ICEKI_LITERAL_0__", instruction)
        self.assertNotIn('"Hallo Welt"', instruction)

    async def test_refine_translation_restores_masked_context_and_changes(self) -> None:
        context_prompt = 'Create a banner in a forest displaying the exact text "Hallo Welt"'
        prompt_de = 'Ersetze den Text durch "Guten Morgen" und mache den Hintergrund dunkler'
        translated_raw = (
            "Create a banner in a darker forest displaying the exact text __ICEKI_LITERAL_0__"
        )

        with patch.object(
            webui_main,
            "_call_ollama_raw",
            AsyncMock(return_value=translated_raw),
        ) as ollama_mock:
            translated = await webui_main._translate_german_to_english(
                prompt_de,
                "demo-model",
                context_prompt=context_prompt,
            )

        self.assertEqual(
            translated,
            'Create a banner in a darker forest displaying the exact text "Guten Morgen"',
        )
        instruction = ollama_mock.await_args.args[0]
        self.assertIn("__ICEKI_LITERAL_0__", instruction)
        self.assertIn("__ICEKI_LITERAL_1__", instruction)
        self.assertNotIn('"Hallo Welt"', instruction)
        self.assertNotIn('"Guten Morgen"', instruction)
