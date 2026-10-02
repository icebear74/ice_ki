"""CPU-only security and persistence tests for the image-owned model API."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import signal
import select
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import urllib.error
import urllib.request
import uuid
import unittest
from unittest.mock import AsyncMock, patch

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


APP_DIR = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "model_downloader_backend", APP_DIR / "docker/model_downloader/backend.py")
backend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backend)
AUTH = {"Authorization": "Bearer " + "test-model-api"}
PAYLOAD = {"url": "https://huggingface.co/team/model/resolve/main/test.safetensors",
           "save_path": "checkpoints", "filename": "test.safetensors"}

# Match pinned ComfyUI's AppRunner and KeyboardInterrupt/finally shutdown flow.
RESTART_SERVER = """
import asyncio, importlib.util, os, pathlib, sys, types
from aiohttp import web
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
app = web.Application()
routes = web.RouteTableDef()
server = types.ModuleType('server')
server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(app=app, routes=routes))
sys.modules['server'] = server
directory = pathlib.Path(sys.argv[1]) / 'docker/model_downloader'
name = str(directory).replace('.', '_x_')
spec = importlib.util.spec_from_file_location(name, directory / '__init__.py')
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
app.add_routes(routes)
async def start():
    runner = web.AppRunner(app, handle_signals=False)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', int(os.getenv('COMFYUI_PID1_TEST_PORT', '0')))
    await site.start()
    print('PID=' + str(os.getpid()), flush=True)
    print('PORT=' + str(site._server.sockets[0].getsockname()[1]), flush=True)
    await asyncio.Future()
try:
    loop.run_until_complete(start())
except KeyboardInterrupt:
    print('STOPPED', flush=True)
finally:
    print('ASSET_MANAGER_SHUTDOWN', flush=True)
"""


def real_restart_request(port):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/server_download/restart", data=b"{}",
        headers={**AUTH, "Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, json.load(response)


class Content:
    def __init__(self, chunks):
        self.chunks = chunks

    async def iter_chunked(self, size):
        for chunk in self.chunks:
            await asyncio.sleep(0)
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


class Response:
    def __init__(self, chunks=(b"model-weights",), status=200, headers=None, length=None):
        self.status = status
        self.headers = headers or {"Content-Type": "application/octet-stream"}
        self.content_length = length
        self.content = Content(chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.urls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def get(self, url, **kwargs):
        assert kwargs["allow_redirects"] is False
        self.urls.append(url)
        return self.responses.pop(0)


class ModelAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=APP_DIR)
        self.root = Path(self.temp.name) / "models"
        self.service = backend.ModelDownloader(self.root, token="test-model-api", max_bytes=1024)
        self.service.extra_hosts = ()
        routes = web.RouteTableDef()
        self.service.register(routes)
        app = web.Application(client_max_size=2 * 1024 ** 2)
        app.add_routes(routes)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.service.shutdown()
        await self.client.close()
        self.temp.cleanup()

    async def request(self, method, path, **kwargs):
        return await self.client.request(method, "/server_download" + path,
                                         headers=kwargs.pop("headers", AUTH), **kwargs)

    async def upload(self, filename="test.safetensors", save_path="checkpoints",
                     content=b"model-weights", **kwargs):
        data = aiohttp.FormData()
        data.add_field("save_path", save_path)
        data.add_field("file", content, filename=filename, content_type="application/octet-stream")
        return await self.request("POST", "/upload", data=data, **kwargs)

    async def start(self, **changes):
        return await self.request("POST", "/start", json={**PAYLOAD, **changes})

    async def transfer(self, responses, **changes):
        session = Session(responses)
        with patch.object(self.service, "session", return_value=session):
            response = await self.start(**changes)
            self.assertEqual(response.status, 202, await response.text())
            download_id = (await response.json())["download_id"]
            await self.service.worker
        return self.service.downloads[download_id], session

    async def test_every_endpoint_requires_token_and_fails_closed_without_configuration(self):
        for configured in ("", "test-model-api"):
            self.service.token = configured
            for method, path in (("GET", "/directories"), ("GET", "/status"),
                                 ("GET", "/status/missing"), ("POST", "/start"),
                                 ("POST", "/upload"), ("POST", "/restart")):
                attempts = [{}, {"Authorization": "Bearer " + "incorrect"},
                            {"X-Model-Token": "test-model-api"}]
                if not configured:
                    attempts.append(AUTH)
                for headers in attempts:
                    with self.subTest(configured=bool(configured), path=path, headers=headers):
                        response = await self.request(method, path, headers=headers)
                        self.assertEqual(response.status, 403)
                        self.assertIn("error", await response.json())
            if not configured:
                response = await self.request("GET", "/directories")
                self.assertEqual(response.status, 403)
                self.assertIn("not configured", (await response.json())["error"])

    async def test_query_token_does_not_authenticate(self):
        response = await self.request("GET", "/status?token=test-model-api", headers={})
        self.assertEqual(response.status, 403)

    async def test_environment_token_whitespace_matches_browser_and_relay_normalization(self):
        for value, expected in ((" \ttest-model-api\n", 200), ("\n\t ", 403)):
            with patch.dict(os.environ, {"COMFYUI_MODEL_API_TOKEN": value}):
                configured = backend.ModelDownloader(self.root, max_bytes=1024)
            self.assertEqual(configured.token, value.strip())
            self.service.token = configured.token
            response = await self.request("GET", "/directories")
            self.assertEqual(response.status, expected, await response.text())
            if expected == 403:
                self.assertIn("not configured", (await response.json())["error"])

    async def test_exact_same_origin_and_fetch_metadata(self):
        origin = str(self.client.make_url("")).rstrip("/")
        for extra in ({"Origin": "https://evil.example"},
                      {"Origin": "null"},
                      {"Origin": origin + ".evil.example"},
                      {"Origin": origin + "/"},
                      {"Sec-Fetch-Site": "cross-site"},
                      {"Sec-Fetch-Site": "same-site"},
                      {"Origin": origin, "Sec-Fetch-Site": "cross-site"}):
            for method, path in (("GET", "/status"), ("POST", "/start"),
                                 ("POST", "/upload"), ("GET", "/directories"),
                                 ("GET", "/status/missing"), ("POST", "/restart")):
                response = await self.request(method, path, headers={**AUTH, **extra})
                self.assertEqual(response.status, 403, extra)
        for extra in ({}, {"Origin": origin}, {"Sec-Fetch-Site": "same-origin"},
                      {"Origin": origin, "Sec-Fetch-Site": "same-origin"},
                      {"Sec-Fetch-Site": "none"}):
            response = await self.request("GET", "/directories", headers={**AUTH, **extra})
            self.assertEqual(response.status, 200, await response.text())

    async def test_directory_allowlist_contains_model_types_and_never_code(self):
        response = await self.request("GET", "/directories")
        directories = (await response.json())["directories"]
        for name in ("checkpoints", "unet", "diffusion_models", "text_encoders", "vae",
                     "loras", "controlnet", "clip_vision", "embeddings", "upscale_models",
                     "style_models"):
            self.assertIn(name, directories)
        self.assertNotIn("custom_nodes", directories)

    async def test_upload_persists_atomic_success_and_canonical_alias(self):
        response = await self.upload(save_path="unet")
        self.assertEqual(response.status, 201, await response.text())
        self.assertEqual(await response.json(), {"filename": "test.safetensors",
                                                "save_path": "diffusion_models", "bytes": 13})
        target = self.root / "diffusion_models/test.safetensors"
        self.assertEqual(target.read_bytes(), b"model-weights")
        self.assertFalse(target.with_suffix(".safetensors.part").exists())
        # A fresh service sees the persisted file, not an in-memory success cache.
        restarted = backend.ModelDownloader(self.root, token="test-model-api")
        with self.assertRaises(backend.TransferError) as caught:
            restarted.destination("unet", "test.safetensors")
        self.assertEqual(caught.exception.status, 409)

    async def test_streaming_upload_is_not_buffered_as_a_complete_file(self):
        payload = b"weights" * 150
        self.service.max_bytes = 2000
        with patch.object(backend, "CHUNK_BYTES", 128):
            response = await self.upload(content=payload)
        self.assertEqual(response.status, 201, await response.text())
        self.assertEqual((self.root / "checkpoints/test.safetensors").read_bytes(), payload)

    async def test_chunked_multipart_relay_counts_only_file_bytes(self):
        boundary = "test-chunked-relay"
        payload = b"streamed-model-weights"
        filename = "space name.safetensors"

        async def chunks():
            yield (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"save_path\"\r\n\r\n"
                f"checkpoints\r\n--{boundary}\r\n"
                f"Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
                "Content-Type: application/octet-stream\r\n\r\n"
            ).encode("ascii")
            for offset in range(0, len(payload), 3):
                await asyncio.sleep(0)
                yield payload[offset:offset + 3]
            yield f"\r\n--{boundary}--\r\n".encode("ascii")

        response = await self.request("POST", "/upload", data=chunks(), headers={
            **AUTH, "Content-Type": f"multipart/form-data; boundary={boundary}",
        })
        self.assertEqual(response.status, 201, await response.text())
        self.assertEqual(response.request_info.headers.get("Transfer-Encoding"), "chunked")
        self.assertNotIn("Content-Length", response.request_info.headers)
        self.assertEqual(await response.json(), {
            "filename": filename, "save_path": "checkpoints", "bytes": len(payload),
        })
        self.assertEqual((self.root / "checkpoints" / filename).read_bytes(), payload)

    async def test_upload_requires_order_and_only_expected_fields(self):
        for order in ("file-first", "extra-field", "wrong-file"):
            data = aiohttp.FormData()
            if order == "file-first":
                data.add_field("file", b"weights", filename="test.gguf")
                data.add_field("save_path", "checkpoints")
            else:
                data.add_field("save_path", "checkpoints")
                data.add_field("file" if order == "extra-field" else "unexpected",
                               b"weights", filename="test.gguf")
                if order == "extra-field":
                    data.add_field("unexpected", "value")
            response = await self.request("POST", "/upload", data=data)
            self.assertEqual(response.status, 400, await response.text())
            self.assertFalse((self.root / "checkpoints/test.gguf").exists())
            self.assertFalse((self.root / "checkpoints/test.gguf.part").exists())

    async def test_upload_rejects_oversize_type_field(self):
        response = await self.upload(save_path="x" * 9000)
        self.assertEqual(response.status, 400)

    async def test_upload_limit_empty_html_and_cleanup(self):
        for payload, status in ((b"x" * 1025, 413), (b"", 400),
                                (b" <!DOCTYPE html><html>error</html>", 400)):
            response = await self.upload(content=payload)
            self.assertEqual(response.status, status, await response.text())
            self.assertFalse((self.root / "checkpoints/test.safetensors").exists())
            self.assertFalse((self.root / "checkpoints/test.safetensors.part").exists())

    async def test_traversal_unsafe_suffix_and_invalid_types(self):
        for changes in (
            {"filename": "../escape.safetensors"}, {"filename": "/escape.safetensors"},
            {"filename": "sub\\escape.safetensors"}, {"filename": "x..safetensors"},
            {"filename": ".hidden.safetensors"}, {"filename": "code.py"},
            {"filename": "code.so"}, {"filename": "weights.pkl"},
            {"filename": "weights.safetensors.py"}, {"filename": "x\n.safetensors"},
            {"save_path": "../custom_nodes"}, {"save_path": "/data/models/checkpoints"},
            {"save_path": "custom_nodes"}, {"save_path": ["checkpoints"]},
            {"filename": None},
        ):
            with self.subTest(changes=changes):
                response = await self.start(**changes)
                self.assertEqual(response.status, 400, await response.text())
        self.assertFalse(self.root.exists())

    async def test_all_safe_suffixes_supported(self):
        for suffix in backend.SUFFIXES:
            response = await self.upload(filename="model" + suffix)
            self.assertEqual(response.status, 201, await response.text())

    async def test_existing_models_cannot_be_overwritten_by_upload_or_download(self):
        response = await self.upload(content=b"original")
        self.assertEqual(response.status, 201)
        for response in (await self.upload(content=b"replacement"), await self.start()):
            self.assertEqual(response.status, 409)
        self.assertEqual((self.root / "checkpoints/test.safetensors").read_bytes(), b"original")

    async def test_symlink_root_directory_final_and_partial_are_rejected(self):
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        self.root.symlink_to(outside, target_is_directory=True)
        response = await self.upload()
        self.assertEqual(response.status, 400)
        self.root.unlink()
        self.root.mkdir()
        (self.root / "checkpoints").symlink_to(outside, target_is_directory=True)
        response = await self.upload()
        self.assertEqual(response.status, 400)
        (self.root / "checkpoints").unlink()
        (self.root / "checkpoints").mkdir()
        sentinel = outside / "sentinel.safetensors"
        sentinel.write_bytes(b"untouched")
        for filename in ("test.safetensors", "test.safetensors.part"):
            target = self.root / "checkpoints" / filename
            target.symlink_to(sentinel)
            response = await self.upload()
            self.assertEqual(response.status, 409, await response.text())
            self.assertTrue(target.is_symlink())
            self.assertEqual(sentinel.read_bytes(), b"untouched")
            target.unlink()
        self.assertEqual(list(outside.iterdir()), [sentinel])

    async def test_queued_download_reserves_destination_and_excludes_uploads(self):
        hold = asyncio.Event()

        async def download(url, destination, state):
            await hold.wait()
            destination.file.write(b"downloaded")
            state["downloaded"] = 10

        with patch.object(self.service, "download", side_effect=download):
            response = await self.start()
            self.assertEqual(response.status, 202)
            duplicate, upload = await self.start(), await self.upload()
            self.assertEqual(duplicate.status, 409)
            self.assertEqual(upload.status, 409)
            hold.set()
            await self.service.worker
        self.assertEqual((self.root / "checkpoints/test.safetensors").read_bytes(), b"downloaded")

    async def test_single_worker_and_fifo_queue(self):
        hold = asyncio.Event()
        entered = []

        async def download(url, destination, state):
            entered.append(destination.filename)
            await hold.wait()
            destination.file.write(b"weights")
            state["downloaded"] = 7

        with patch.object(self.service, "download", side_effect=download):
            first = await self.start(filename="first.gguf")
            worker = self.service.worker
            second = await self.start(filename="second.gguf")
            self.assertEqual(first.status, 202)
            self.assertEqual(second.status, 202)
            self.assertIs(self.service.worker, worker)
            self.assertEqual(entered, ["first.gguf"])
            hold.set()
            await worker
        self.assertEqual(entered, ["first.gguf", "second.gguf"])

    async def test_public_download_success_status_and_persistence(self):
        state, session = await self.transfer([Response(chunks=(b"model", b"-weights"), length=13)])
        self.assertEqual(state, {"status": "completed", "progress": 100, "downloaded": 13,
                                 "total": 13, "error": None, "filename": "test.safetensors",
                                 "save_path": "checkpoints"})
        self.assertEqual(session.urls, [PAYLOAD["url"]])
        self.assertEqual((self.root / "checkpoints/test.safetensors").read_bytes(), b"model-weights")
        response = await self.request("GET", "/status")
        all_states = (await response.json())["downloads"]
        self.assertEqual(all_states, {"checkpoints/test.safetensors": state})
        download_id = next(iter(self.service.downloads))
        response = await self.request("GET", "/status/" + download_id)
        self.assertEqual(await response.json(), state)
        response = await self.request("GET", "/status/unknown")
        self.assertEqual(response.status, 404)

    async def test_remote_url_allowlist_rejects_arbitrary_ssrf(self):
        for url in ("http://huggingface.co/file", "https://evil.example/model",
                    "https://127.0.0.1/model", "https://[::1]/model",
                    "https://192.168.1.1/model", "https://localhost/model",
                    "https://huggingface.co.evil.example/model",
                    "https://huggingface.co:8443/model",
                    "https://" + "user:password@" + "huggingface.co/model",
                    "https://huggingface.co./model", "https://huggingface.co\\evil/model",
                    "https://huggingface.co/model\n", "https://huggingface.co/model#fragment"):
            response = await self.start(url=url)
            self.assertEqual(response.status, 400, url)
        self.assertFalse(self.root.exists())

    async def test_trusted_redirects_checked_at_every_hop(self):
        state, session = await self.transfer([
            Response(status=302, headers={"Location": "https://cdn.xethub.hf.co/model"}),
            Response(chunks=(b"weights",)),
        ])
        self.assertEqual(state["status"], "completed")
        self.assertEqual(len(session.urls), 2)

    async def test_redirect_to_private_or_untrusted_host_is_never_requested(self):
        for location in ("https://127.0.0.1/model", "https://evil.example/model",
                         "http://huggingface.co/model", "https://[::1]/model"):
            state, session = await self.transfer([
                Response(status=302, headers={"Location": location})])
            self.assertEqual(state["status"], "error")
            self.assertEqual(session.urls, [PAYLOAD["url"]])
            self.assertFalse((self.root / "checkpoints/test.safetensors.part").exists())
            self.assertFalse((self.root / "checkpoints/test.safetensors").exists())

    async def test_redirect_bound_and_missing_location(self):
        for responses in ([Response(status=302, headers={"Other": "header"})],
                          [Response(status=302, headers={"Location": "/again"}) for _ in range(6)]):
            state, session = await self.transfer(responses)
            self.assertEqual(state["status"], "error")
            self.assertLessEqual(len(session.urls), 6)

    async def test_configured_cdn_allowlist(self):
        self.service.extra_hosts = ("cdn.example.org", "*.models.example.org")
        state, _ = await self.transfer([Response()], url="https://cdn.example.org/model")
        self.assertEqual(state["status"], "completed")
        for url in ("https://a.models.example.org/model", "https://huggingface.co/model",
                    "https://a.huggingface.co/model", "https://civitai.com/api/download/models/1"):
            self.assertEqual(backend.validate_url(url, self.service.extra_hosts), url)
        with self.assertRaises(backend.TransferError):
            backend.validate_url("https://cdn.example.org.evil.test/model", self.service.extra_hosts)

    async def test_response_size_bounds_mismatch_and_network_failure_cleanup(self):
        for response in (
            Response(length=1025), Response(chunks=(b"x" * 600, b"x" * 600)),
            Response(length=100, chunks=(b"short",)), Response(length=0),
            Response(chunks=(b"prefix", RuntimeError("secret-token-in-signed-url"))),
            Response(status=403), Response(headers={"Content-Type": "text/html"}),
            Response(headers={"Content-Type": "application/json"}),
            Response(headers={"Content-Encoding": "gzip"}),
            Response(chunks=(b"<!DOCTYPE html>error",)), Response(chunks=()),
        ):
            state, _ = await self.transfer([response])
            self.assertEqual(state["status"], "error")
            self.assertNotIn("secret-token", state["error"])
            self.assertFalse((self.root / "checkpoints/test.safetensors").exists())
            self.assertFalse((self.root / "checkpoints/test.safetensors.part").exists())

    async def test_shutdown_cleans_active_and_queued_partial_files(self):
        hold = asyncio.Event()

        async def download(*args):
            await hold.wait()

        with patch.object(self.service, "download", side_effect=download):
            await self.start(filename="active.gguf")
            await self.start(filename="queued.gguf")
            await self.service.shutdown()
        self.assertEqual(list((self.root / "checkpoints").iterdir()), [])
        self.assertTrue(all(state["status"] == "error" for state in self.service.downloads.values()))

    async def test_atomic_publish_does_not_overwrite_racing_existing_file(self):
        async def download(url, destination, state):
            destination.file.write(b"new")
            (self.root / "checkpoints/test.safetensors").write_bytes(b"existing")

        with patch.object(self.service, "download", side_effect=download):
            await self.start()
            await self.service.worker
        self.assertEqual((self.root / "checkpoints/test.safetensors").read_bytes(), b"existing")
        self.assertFalse((self.root / "checkpoints/test.safetensors.part").exists())
        self.assertEqual(next(iter(self.service.downloads.values()))["status"], "error")

    async def test_json_errors_are_bounded_and_do_not_create_targets(self):
        for data, expected in (("broken", 400), ("[]", 400), ("{}", 400),
                               ('{"url":"' + "a" * 17000 + '"}', 413)):
            response = await self.request("POST", "/start", data=data,
                                          headers={**AUTH, "Content-Type": "application/json"})
            self.assertEqual(response.status, expected, await response.text())
        self.assertFalse(self.root.exists())

    async def test_split_chunked_json_request_is_supported(self):
        async def chunks():
            encoded = json.dumps(PAYLOAD).encode()
            yield encoded[:20]
            await asyncio.sleep(0.01)
            yield encoded[20:]

        with patch.object(self.service, "session", return_value=Session([Response()])):
            response = await self.request("POST", "/start", data=chunks(),
                                          headers={**AUTH, "Content-Type": "application/json"})
            self.assertEqual(response.status, 202, await response.text())
            await self.service.worker

    async def test_restart_returns_accepted_then_signals_only_own_process(self):
        with patch.object(backend.os, "kill") as kill, patch.object(
            backend, "RESTART_DELAY_SECONDS", 0.01
        ):
            response = await self.request("POST", "/restart", json={})
            self.assertEqual(response.status, 202, await response.text())
            self.assertEqual(await response.json(), {"status": "restarting"})
            kill.assert_not_called()
            duplicate = await self.request("POST", "/restart", json={})
            self.assertEqual(duplicate.status, 409)
            start = await self.start()
            upload = await self.upload()
            self.assertEqual(start.status, 409)
            self.assertEqual(upload.status, 409)
            await self.service.restart_task
            kill.assert_called_once_with(os.getpid(), signal.SIGTERM)

    async def test_restart_is_rejected_during_active_or_queued_downloads(self):
        hold = asyncio.Event()

        async def download(*args):
            await hold.wait()

        with patch.object(self.service, "download", side_effect=download), patch.object(
            backend.os, "kill"
        ) as kill:
            await self.start(filename="active.gguf")
            await self.start(filename="queued.gguf")
            response = await self.request("POST", "/restart", json={})
            self.assertEqual(response.status, 409, await response.text())
            self.assertFalse(self.service.restarting)
            kill.assert_not_called()
            await self.service.shutdown()

    async def test_restart_is_rejected_during_upload_and_upload_counter_cleans_up(self):
        hold = asyncio.Event()
        started = asyncio.Event()

        async def stream(*args):
            started.set()
            await hold.wait()
            raise backend.TransferError("Simulated upload failure")

        with patch.object(self.service, "stream", side_effect=stream), patch.object(
            backend.os, "kill"
        ) as kill:
            upload_task = asyncio.create_task(self.upload())
            await started.wait()
            self.assertEqual(self.service.active_uploads, 1)
            response = await self.request("POST", "/restart", json={})
            self.assertEqual(response.status, 409, await response.text())
            kill.assert_not_called()
            hold.set()
            response = await upload_task
            self.assertEqual(response.status, 400)
            self.assertEqual(self.service.active_uploads, 0)
            self.assertFalse(self.service.restarting)
            self.assertFalse((self.root / "checkpoints/test.safetensors.part").exists())

    async def test_shutdown_cancels_pending_restart_without_signalling(self):
        with patch.object(backend.os, "kill") as kill:
            response = await self.request("POST", "/restart", json={})
            self.assertEqual(response.status, 202)
            await self.service.shutdown()
            self.assertTrue(self.service.restart_task.cancelled())
            kill.assert_not_called()


class ResolverTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_dns_answers_reject_private_local_multicast_and_mixed_results(self):
        resolver = backend.PublicResolver()
        try:
            for addresses in (["127.0.0.1"], ["10.0.0.1"], ["169.254.169.254"],
                              ["192.168.1.5"], ["::1"], ["fc00::1"], ["fe80::1"],
                              ["::ffff:127.0.0.1"], ["224.0.0.1"], ["0.0.0.0"],
                              ["93.184.216.34", "127.0.0.1"], []):
                resolver.delegate.resolve = AsyncMock(return_value=[{"host": value} for value in addresses])
                with self.assertRaises(OSError, msg=addresses):
                    await resolver.resolve("huggingface.co", 443)
            resolver.delegate.resolve = AsyncMock(return_value=[{"host": "93.184.216.34"}])
            self.assertEqual(await resolver.resolve("huggingface.co", 443),
                             [{"host": "93.184.216.34"}])
        finally:
            await resolver.close()

    async def test_client_disables_proxies_dns_cache_and_automatic_decompression(self):
        service = backend.ModelDownloader(token="test")
        async with service.session() as session:
            self.assertFalse(session.trust_env)
            self.assertFalse(session.auto_decompress)
            self.assertFalse(session.connector.use_dns_cache)
            self.assertTrue(session.connector.force_close)
            self.assertIsInstance(session.connector._resolver, backend.PublicResolver)

    async def test_image_package_registers_routes_with_pinned_comfyui_loader_metadata(self):
        directory = APP_DIR / "docker/model_downloader"
        module_name = str(directory).replace(".", "_x_")
        spec = importlib.util.spec_from_file_location(module_name, directory / "__init__.py")
        module = importlib.util.module_from_spec(spec)
        app = web.Application()
        instance = SimpleNamespace(routes=web.RouteTableDef(), app=app)
        server = SimpleNamespace(PromptServer=SimpleNamespace(instance=instance))
        with patch.dict(sys.modules, {"server": server, module_name: module}), patch.object(
            signal, "signal"
        ) as register_signal:
            spec.loader.exec_module(module)
            register_signal.assert_called_once_with(signal.SIGTERM, signal.default_int_handler)
            self.assertEqual(module.NODE_CLASS_MAPPINGS, {})
            self.assertEqual(module.NODE_DISPLAY_NAME_MAPPINGS, {})
            self.assertEqual(module.WEB_DIRECTORY, "./web")
            registered = {(route.method, route.path) for route in instance.routes}
            self.assertEqual(registered, {
                ("POST", "/server_download/start"), ("GET", "/server_download/status"),
                ("GET", "/server_download/status/{download_id}"),
                ("POST", "/server_download/upload"), ("GET", "/server_download/directories"),
                ("POST", "/server_download/restart"),
            })
            self.assertIn(module.cleanup_downloader, app.on_cleanup)
            await module.cleanup_downloader(app)


class RealRestartTests(unittest.TestCase):
    def test_real_sigterm_stops_server_through_comfyui_keyboardinterrupt_finally_flow(self):
        env = {**os.environ, "COMFYUI_MODEL_API_TOKEN": "test-model-api",
               "PYTHONDONTWRITEBYTECODE": "1"}
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", RESTART_SERVER, str(APP_DIR)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        )
        try:
            self.assertTrue(select.select([process.stdout], [], [], 10)[0], "Server did not start")
            pid_line = process.stdout.readline()
            port_line = process.stdout.readline()
            self.assertTrue(pid_line.startswith("PID="), pid_line)
            self.assertTrue(port_line.startswith("PORT="), port_line)
            self.assertEqual(real_restart_request(int(port_line[5:])),
                             (202, {"status": "restarting"}))
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertIn("STOPPED", stdout)
            self.assertIn("ASSET_MANAGER_SHUTDOWN", stdout)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)

    @unittest.skipUnless(os.getenv("COMFYUI_TEST_PID1_IMAGE") and shutil.which("docker"),
                         "Set COMFYUI_TEST_PID1_IMAGE to an existing Python 3.12 image")
    def test_restart_actually_exits_container_pid1(self):
        name = "iceki-restart-smoke-" + uuid.uuid4().hex[:12]
        dependency_roots = sorted({
            str(Path(path).resolve()) for path in sys.path
            if path.endswith(("site-packages", "dist-packages")) and Path(path).is_dir()
        })
        dependency_mounts = [f"/smoke-deps-{index}" for index in range(len(dependency_roots))]
        command = [
            "docker", "run", "--detach", "--name", name, "--read-only",
            "--publish", "127.0.0.1::8188", "--env", "COMFYUI_MODEL_API_TOKEN=test-model-api",
            "--env", "COMFYUI_PID1_TEST_PORT=8188", "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--env", "PYTHONPATH=" + ":".join(dependency_mounts),
            "--volume", f"{APP_DIR}:/smoke-app:ro",
        ]
        for source, target in zip(dependency_roots, dependency_mounts):
            command.extend(["--volume", f"{source}:{target}:ro"])
        command.extend([
            os.environ["COMFYUI_TEST_PID1_IMAGE"],
            "python", "-u", "-c", RESTART_SERVER, "/smoke-app",
        ])
        try:
            started = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(started.returncode, 0, started.stderr)
            published = subprocess.check_output(
                ["docker", "port", name, "8188/tcp"], text=True, timeout=10)
            port = int(published.strip().rsplit(":", 1)[1])
            for attempt in range(50):
                try:
                    accepted = real_restart_request(port)
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.1)
            else:
                logs = subprocess.check_output(["docker", "logs", name], text=True, timeout=10)
                self.fail("PID1 smoke server did not respond: " + logs)
            self.assertEqual(accepted, (202, {"status": "restarting"}))
            exit_code = subprocess.check_output(["docker", "wait", name], text=True, timeout=15)
            logs = subprocess.check_output(["docker", "logs", name], text=True, timeout=10)
            self.assertEqual(exit_code.strip(), "0", logs)
            self.assertIn("PID=1", logs)
            self.assertIn("STOPPED", logs)
            self.assertIn("ASSET_MANAGER_SHUTDOWN", logs)
        finally:
            subprocess.run(["docker", "rm", "--force", name], capture_output=True, timeout=15)


if __name__ == "__main__":
    unittest.main()
