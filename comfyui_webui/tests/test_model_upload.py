"""Admin-only model upload proxy and bounded multipart streaming contracts."""
from __future__ import annotations

import asyncio
import io
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

import main


class ModelUploadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.token = "private" + "-model-test-marker"
        self.enterContext(patch.dict(main.os.environ, {
            "COMFYUI_MODEL_API_TOKEN": self.token,
            "COMFYUI_MODEL_MAX_BYTES": str(40 * 1024**3),
        }))
        self.enterContext(patch.dict(main._sessions, {
            "model-admin": {"username": "admin", "role": "admin"},
            "model-user": {"username": "user", "role": "user"},
        }, clear=True))
        self.client = TestClient(main.app, cookies={"ki_session": "model-admin"})
        self.addCleanup(self.client.close)
        self.requests: list[httpx.Request] = []
        self.chunks: list[bytes] = []
        self.options: list[dict] = []
        self.directory_payload = {"directories": [
            "checkpoints", "diffusion_models", "text_encoders", "vae", "loras", "controlnet",
            "unet", "clip", "t2i_adapter",
        ]}
        self.upload_status = 201
        self.restart_status = 202
        self.upload_payload: object = None
        self.upload_raw: bytes | None = None
        self.fail_request = False
        original_client = httpx.AsyncClient

        async def backend(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            self.assertEqual(request.headers["authorization"], "Bearer " + self.token)
            if self.fail_request:
                raise httpx.ConnectError(self.token, request=request)
            if request.url.path == "/server_download/directories":
                return httpx.Response(200, json=self.directory_payload)
            if request.url.path == "/server_download/restart":
                self.assertEqual(request.method, "POST")
                self.assertEqual(json.loads(await request.aread()), {})
                return httpx.Response(self.restart_status, json={"debug": self.token})
            self.assertEqual(request.url.path, "/server_download/upload")
            self.assertIsInstance(request.stream, httpx.AsyncByteStream)
            self.chunks = [chunk async for chunk in request.stream]
            body = b"".join(self.chunks)
            first, *_, last = body.split(b"\r\n")
            self.assertTrue(first.startswith(b"--iceki-"))
            self.assertEqual(body.split(b"\r\n")[-2], first + b"--")
            self.assertEqual(last, b"")
            self.assertLess(body.index(b'name="save_path"'), body.index(b'name="file"'))
            if self.upload_raw is not None:
                return httpx.Response(self.upload_status, content=self.upload_raw)
            payload = self.upload_payload
            if payload is None:
                filename = body.split(b'filename="')[1].split(b'"')[0].decode()
                destination = body.split(b'name="save_path"\r\n\r\n')[1].split(b"\r\n")[0].decode()
                data = body.split(b"Content-Type: application/octet-stream\r\n\r\n")[1].rsplit(b"\r\n--", 1)[0]
                payload = {"filename": filename, "save_path": destination, "bytes": len(data)}
            return httpx.Response(self.upload_status, json=payload)

        class StreamingTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                return await backend(request)

        def make_client(**kwargs):
            self.options.append(kwargs)
            return original_client(transport=StreamingTransport(), **kwargs)

        self.backend = self.enterContext(patch.object(main.httpx, "AsyncClient", side_effect=make_client))

    def upload(self, filename="weights.safetensors", directory="checkpoints", contents=b"model"):
        return self.client.post("/api/admin/models/upload", data={"save_path": directory}, files={
            "file": (filename, contents, "application/octet-stream"),
        })

    def test_admin_authentication_required_for_both_endpoints(self) -> None:
        for cookie, expected in ((None, 401), ("model-user", 403)):
            with self.subTest(cookie=cookie):
                self.client.cookies.clear()
                if cookie:
                    self.client.cookies.set("ki_session", cookie)
                self.assertEqual(self.client.get("/api/admin/models/directories").status_code, expected)
                self.assertEqual(self.upload().status_code, expected)
                self.assertEqual(self.client.post("/api/admin/models/restart", json={}).status_code, expected)
        self.backend.assert_not_called()

    def test_missing_token_is_clear_503_without_backend_request(self) -> None:
        with patch.dict(main.os.environ, {"COMFYUI_MODEL_API_TOKEN": ""}):
            for response in (self.client.get("/api/admin/models/directories"), self.upload(),
                             self.client.post("/api/admin/models/restart", json={})):
                self.assertEqual(response.status_code, 503)
                self.assertIn("Token", response.json()["detail"])
        self.backend.assert_not_called()

    def test_directories_are_backend_allowlist_not_local_paths(self) -> None:
        response = self.client.get("/api/admin/models/directories")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), self.directory_payload)
        self.assertNotIn(self.token, response.text)

    def test_invalid_directory_responses_fail_closed(self) -> None:
        for payload in ({}, [], None, {"directories": []}, {"directories": ["../checkpoints"]},
                        {"directories": [1]}, {"directories": "checkpoints"}):
            with self.subTest(payload=payload):
                self.directory_payload = payload
                response = self.client.get("/api/admin/models/directories")
                self.assertEqual(response.status_code, 502)
                self.assertNotIn(self.token, response.text)

    def test_upload_streams_multipart_in_bounded_chunks_destination_first(self) -> None:
        contents = b"x" * (main._MODEL_UPLOAD_CHUNK_BYTES * 2 + 137)
        response = self.upload(contents=contents)
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json(), {
            "filename": "weights.safetensors", "save_path": "checkpoints", "bytes": len(contents),
        })
        self.assertEqual(self.chunks[1:-1], [
            contents[:main._MODEL_UPLOAD_CHUNK_BYTES],
            contents[main._MODEL_UPLOAD_CHUNK_BYTES:main._MODEL_UPLOAD_CHUNK_BYTES * 2],
            contents[main._MODEL_UPLOAD_CHUNK_BYTES * 2:],
        ])
        self.assertTrue(all(len(chunk) <= main._MODEL_UPLOAD_CHUNK_BYTES for chunk in self.chunks))
        self.assertFalse(self.options[0]["follow_redirects"])
        self.assertGreaterEqual(self.options[0]["timeout"].write, 3600)
        self.assertGreaterEqual(self.options[0]["timeout"].read, 3600)
        self.assertNotIn(self.token, response.text)

    def test_all_supported_extensions_and_available_destinations(self) -> None:
        for extension in main._MODEL_EXTENSIONS:
            for directory in self.directory_payload["directories"]:
                with self.subTest(extension=extension, directory=directory):
                    self.assertEqual(self.upload(filename="trusted" + extension, directory=directory).status_code, 201)
        self.assertEqual(self.upload(filename="x" * 237 + ".pt").status_code, 201)

    def test_directory_aliases_are_forwarded_and_returned_as_canonical_destinations(self) -> None:
        for alias, canonical in main._MODEL_DIRECTORY_ALIASES.items():
            with self.subTest(alias=alias):
                response = self.upload(directory=alias)
                self.assertEqual(response.status_code, 201, response.text)
                self.assertEqual(response.json()["save_path"], canonical)
                self.assertIn(f'name="save_path"\r\n\r\n{canonical}\r\n'.encode(), self.chunks[0])

    def test_rejects_unsafe_filename_and_extension(self) -> None:
        for name in ("../evil.pt", "/evil.pt", "nested/model.pt", "model\\evil.pt", "model.exe",
                     "model.safetensors.exe", "model..pt", ".hidden.pt", 'bad"name.pt', "x" * 238 + ".pt",
                     "evil\r\n.pt", "evil\n.pt"):
            with self.subTest(filename=name):
                self.assertEqual(self.upload(filename=name).status_code, 400)
        self.backend.assert_not_called()

    def test_destination_must_be_canonical_and_returned_by_backend(self) -> None:
        for directory in ("../vae", "/vae", "models/vae", "VAE", "vae/", "vae\\", "vae\r\n", "not_approved"):
            with self.subTest(directory=directory):
                response = self.upload(directory=directory)
                self.assertEqual(response.status_code, 400, response.text)
        self.assertTrue(all(request.url.path.endswith("/directories") for request in self.requests))

    def test_known_size_limit_and_empty_file_checked_before_backend(self) -> None:
        with patch.dict(main.os.environ, {"COMFYUI_MODEL_MAX_BYTES": "4"}):
            self.assertEqual(self.upload(contents=b"12345").status_code, 413)
            self.assertEqual(self.upload(contents=b"").status_code, 400)
        self.backend.assert_not_called()

    def test_limit_is_inclusive_and_bad_configuration_is_503(self) -> None:
        with patch.dict(main.os.environ, {"COMFYUI_MODEL_MAX_BYTES": "5"}):
            self.assertEqual(self.upload(contents=b"12345").status_code, 201)
        for value in ("0", "-1", "not-an-integer"):
            with patch.dict(main.os.environ, {"COMFYUI_MODEL_MAX_BYTES": value}):
                self.assertEqual(self.upload().status_code, 503)

    def test_streaming_limit_does_not_trust_spooled_size(self) -> None:
        with patch.dict(main.os.environ, {"COMFYUI_MODEL_MAX_BYTES": "4"}):
            for size in (None, 1):
                with self.subTest(size=size):
                    file = UploadFile(io.BytesIO(b"12345"), filename="weights.pt", size=size)
                    file.read = AsyncMock(wraps=file.read)
                    with self.assertRaises(HTTPException) as raised:
                        asyncio.run(main.admin_upload_model(save_path="checkpoints", file=file, _={}))
                    self.assertEqual(raised.exception.status_code, 413)
                    file.read.assert_awaited_with(main._MODEL_UPLOAD_CHUNK_BYTES)
                    self.assertTrue(file.file.closed)

    def test_meaningful_backend_statuses_are_sanitized(self) -> None:
        self.upload_payload = {"detail": self.token, "error": "<script>private backend path</script>"}
        for status in (400, 409, 413, 401, 403, 404, 500, 302):
            with self.subTest(status=status):
                self.upload_status = status
                response = self.upload()
                self.assertEqual(response.status_code, status if status in (400, 409, 413) else 502)
                self.assertNotIn(self.token, response.text)
                self.assertNotIn("script", response.text)
                self.assertNotIn("private backend", response.text)

    def test_backend_connection_error_is_sanitized(self) -> None:
        self.fail_request = True
        for response in (self.client.get("/api/admin/models/directories"), self.upload(),
                         self.client.post("/api/admin/models/restart", json={})):
            self.assertEqual(response.status_code, 502)
            self.assertNotIn(self.token, response.text)

    def test_restart_proxies_empty_json_and_sanitizes_response(self) -> None:
        response = self.client.post("/api/admin/models/restart", json={})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json(), {"status": "restarting"})
        self.assertNotIn(self.token, response.text)
        self.assertFalse(self.options[0]["follow_redirects"])

    def test_restart_reports_busy_and_unavailable_without_backend_details(self) -> None:
        for status in (409, 200, 201, 301, 400, 401, 403, 500, 503):
            with self.subTest(status=status):
                self.restart_status = status
                response = self.client.post("/api/admin/models/restart", json={})
                self.assertEqual(response.status_code, 409 if status == 409 else 502)
                if status == 409:
                    self.assertIn("beschäftigt", response.json()["detail"])
                self.assertNotIn(self.token, response.text)
                self.assertNotIn("debug", response.text)

    def test_malformed_upload_success_is_not_reflected(self) -> None:
        for payload in ({}, [], None, {"filename": self.token, "save_path": "checkpoints", "bytes": 5},
                        {"filename": "weights.safetensors", "save_path": "../outside", "bytes": 5},
                        {"filename": "weights.safetensors", "save_path": "checkpoints", "bytes": "5"},
                        {"filename": "weights.safetensors", "save_path": "checkpoints", "bytes": 6}):
            with self.subTest(payload=payload):
                # None normally triggers the valid default in the mock.
                self.upload_payload = payload if payload is not None else "invalid-json-shape"
                response = self.upload()
                self.assertEqual(response.status_code, 502, response.text)
                self.assertNotIn(self.token, response.text)
        self.upload_raw = ("not-json " + self.token).encode()
        response = self.upload()
        self.assertEqual(response.status_code, 502)
        self.assertNotIn(self.token, response.text)

    def test_response_discards_backend_extras_including_credentials(self) -> None:
        self.upload_payload = {
            "filename": "weights.safetensors", "save_path": "checkpoints", "bytes": 5,
            "debug": self.token,
        }
        response = self.upload()
        self.assertEqual(response.status_code, 201)
        self.assertEqual(set(response.json()), {"filename", "save_path", "bytes"})
        self.assertNotIn(self.token, response.text)

    def test_token_never_in_public_config_or_upload_response(self) -> None:
        response = self.client.get("/api/config")
        self.assertNotIn(self.token, response.text)
        self.assertNotIn("token", json.dumps(response.json()).lower())
        self.assertNotIn(self.token, self.upload().text)


if __name__ == "__main__":
    unittest.main()
