"""Bounded model transfers. Executable code never belongs on the model volume."""
import asyncio
import hmac
import ipaddress
import json
import os
import re
import signal
import socket
import stat
import uuid
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import aiohttp
from aiohttp import web


MODEL_TYPES = (
    "checkpoints", "diffusion_models", "text_encoders", "vae", "loras",
    "controlnet", "clip_vision", "embeddings", "upscale_models", "style_models",
    "vae_approx", "gligen", "latent_upscale_models", "hypernetworks",
    "photomaker", "model_patches", "audio_encoders", "background_removal",
    "frame_interpolation", "geometry_estimation", "optical_flow", "detection",
    "classifiers",
)
ALIASES = {"unet": "diffusion_models", "clip": "text_encoders",
           "t2i_adapter": "controlnet"}
SUFFIXES = {".safetensors", ".gguf", ".pt", ".pth", ".ckpt", ".bin"}
DEFAULT_MAX_BYTES = 40 * 1024 ** 3
CHUNK_BYTES = 256 * 1024
RESTART_DELAY_SECONDS = 0.5


class TransferError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def validate_url(url, extra_hosts=()):
    if not isinstance(url, str) or len(url) > 8192 or any(
        ord(char) < 33 or ord(char) == 127 for char in url
    ):
        raise TransferError("Invalid HTTPS model URL")
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError:
        raise TransferError("Invalid HTTPS model URL") from None
    if (parsed.scheme != "https" or not host or parsed.username is not None
            or parsed.password is not None or port not in (None, 443)
            or parsed.fragment or "\\" in url or host.endswith(".")):
        raise TransferError("Only trusted HTTPS model URLs are allowed")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise TransferError("IP address URLs are not allowed")
    patterns = ("huggingface.co", "*.huggingface.co", "hf.co", "*.hf.co",
                "civitai.com", *extra_hosts)
    if not any(host == pattern or (
        pattern.startswith("*.") and host.endswith(pattern[1:])
        and host != pattern[2:]
    ) for pattern in patterns):
        raise TransferError("Model URL host is not trusted")
    return url


def public_address(address):
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped:
        parsed = parsed.ipv4_mapped
    return parsed.is_global and not parsed.is_multicast


class PublicResolver(aiohttp.abc.AbstractResolver):
    """Validate the actual connector DNS answers, not a separate preflight lookup."""
    def __init__(self):
        self.delegate = aiohttp.resolver.DefaultResolver()

    async def resolve(self, host, port=0, family=socket.AF_INET):
        records = await self.delegate.resolve(host, port, family)
        if not records or any(not public_address(record["host"]) for record in records):
            raise OSError("Model host must resolve only to public addresses")
        return records

    async def close(self):
        await self.delegate.close()


class Destination:
    """Directory-relative no-follow creation and atomic, no-replace publication."""
    def __init__(self, root, model_type, filename):
        self.directory_fd = None
        self.file = None
        self.part = filename + ".part"
        self.filename = filename
        self.owned = False
        directory_fd = os.open(root.anchor, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for component in (*root.parts[1:], model_type):
                try:
                    os.mkdir(component, mode=0o750, dir_fd=directory_fd)
                except FileExistsError:
                    pass
                next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                  dir_fd=directory_fd)
                os.close(directory_fd)
                directory_fd = next_fd
            self.directory_fd = directory_fd
            try:
                os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise TransferError("Model file already exists", 409)
            fd = os.open(self.part, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o640, dir_fd=directory_fd)
            self.owned = True
            self.file = os.fdopen(fd, "wb")
        except Exception:
            os.close(directory_fd)
            self.directory_fd = None
            raise

    def publish(self):
        self.file.flush()
        os.fsync(self.file.fileno())
        opened = os.fstat(self.file.fileno())
        named = os.stat(self.part, dir_fd=self.directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(named.st_mode) or (opened.st_dev, opened.st_ino) != (
            named.st_dev, named.st_ino
        ):
            raise TransferError("Partial model target changed")
        os.link(self.part, self.filename, src_dir_fd=self.directory_fd,
                dst_dir_fd=self.directory_fd, follow_symlinks=False)
        os.unlink(self.part, dir_fd=self.directory_fd)
        self.owned = False
        os.fsync(self.directory_fd)

    def close(self):
        if self.file is not None:
            self.file.close()
        if self.directory_fd is not None:
            if self.owned:
                try:
                    os.unlink(self.part, dir_fd=self.directory_fd)
                except FileNotFoundError:
                    pass
            os.close(self.directory_fd)
            self.directory_fd = None


class ModelDownloader:
    def __init__(self, root=Path("/data/models"), token=None, max_bytes=None):
        self.root = Path(root).absolute()
        self.token = (os.environ.get("COMFYUI_MODEL_API_TOKEN", "")
                      if token is None else token).strip()
        self.hf_token = os.environ.get("HF_TOKEN", "").strip()
        if any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in self.hf_token):
            raise ValueError("Invalid HF_TOKEN configuration")
        limit = os.environ.get("COMFYUI_MODEL_MAX_BYTES", str(DEFAULT_MAX_BYTES))
        self.max_bytes = int(limit) if max_bytes is None else max_bytes
        if self.max_bytes <= 0:
            raise ValueError("COMFYUI_MODEL_MAX_BYTES must be positive")
        self.extra_hosts = tuple(host.strip().lower() for host in
                                 os.environ.get("COMFYUI_MODEL_ALLOWED_HOSTS", "").split(",")
                                 if host.strip())
        self.downloads = {}
        self.pending = []
        self.worker = None
        self.lock = asyncio.Lock()
        self.active_uploads = 0
        self.restarting = False
        self.restart_task = None

    def authorize(self, request):
        if not self.token:
            raise TransferError("Model API disabled: COMFYUI_MODEL_API_TOKEN is not configured", 403)
        credential = request.headers.get("Authorization", "")
        if not hmac.compare_digest(credential.encode(), ("Bearer " + self.token).encode()):
            raise TransferError("Model API authentication required", 403)
        origin = request.headers.get("Origin")
        if origin is not None and origin != f"{request.scheme}://{request.host}":
            raise TransferError("Cross-origin model API requests are forbidden", 403)
        fetch_site = request.headers.get("Sec-Fetch-Site")
        if fetch_site is not None and fetch_site not in ("same-origin", "none"):
            raise TransferError("Cross-origin model API requests are forbidden", 403)

    def destination(self, save_path, filename):
        if not isinstance(save_path, str):
            raise TransferError("Invalid model type")
        model_type = ALIASES.get(save_path, save_path)
        if model_type not in MODEL_TYPES:
            raise TransferError("Invalid model type")
        if (not isinstance(filename, str) or len(filename) > 240
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._ -]*", filename)
                or ".." in filename or Path(filename).suffix.lower() not in SUFFIXES):
            raise TransferError("Use a simple model filename with an approved model suffix")
        try:
            destination = Destination(self.root, model_type, filename)
        except FileExistsError:
            raise TransferError("Model file or active partial transfer already exists", 409) from None
        except OSError:
            raise TransferError("Model target is unsafe or unavailable", 400) from None
        return model_type, destination

    def register(self, routes):
        for method, path, handler in (
            ("POST", "/server_download/start", self.start),
            ("GET", "/server_download/status", self.status),
            ("GET", "/server_download/status/{download_id}", self.status_one),
            ("POST", "/server_download/upload", self.upload),
            ("GET", "/server_download/directories", self.directories),
            ("POST", "/server_download/restart", self.restart),
        ):
            async def endpoint(request, handler=handler):
                try:
                    self.authorize(request)
                    return await handler(request)
                except TransferError as exc:
                    return web.json_response({"error": str(exc)}, status=exc.status)
                except (ValueError, TypeError, KeyError, AssertionError):
                    return web.json_response({"error": "Invalid model API request"}, status=400)
                except OSError:
                    return web.json_response({"error": "Model storage unavailable"}, status=409)
                except Exception:
                    return web.json_response({"error": "Model API request failed"}, status=500)
            routes.route(method, path)(endpoint)

    async def directories(self, request):
        return web.json_response({"directories": [*MODEL_TYPES, *ALIASES]})

    async def status(self, request):
        return web.json_response({"downloads": {
            f"{state['save_path']}/{state['filename']}": state
            for state in self.downloads.values()
        }})

    async def status_one(self, request):
        download_id = request.match_info["download_id"]
        if download_id not in self.downloads:
            raise TransferError("Unknown download", 404)
        return web.json_response(self.downloads[download_id])

    async def start(self, request):
        if request.content_type != "application/json":
            raise TransferError("Expected application/json")
        raw = bytearray()
        async for chunk in request.content.iter_chunked(4096):
            raw.extend(chunk)
            if len(raw) > 16384:
                raise TransferError("Model start request is too large", 413)
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise TransferError("Expected a JSON object")
        url = validate_url(data.get("url"), self.extra_hosts)
        async with self.lock:
            if self.restarting:
                raise TransferError("ComfyUI restart is pending", 409)
            if len(self.pending) >= 128:
                raise TransferError("Model download queue is full", 429)
            model_type, destination = self.destination(data.get("save_path"), data.get("filename"))
            download_id = uuid.uuid4().hex
            for old_id in list(self.downloads):
                if len(self.downloads) < 256:
                    break
                if self.downloads[old_id]["status"] in ("completed", "error"):
                    del self.downloads[old_id]
            self.downloads[download_id] = {
                "status": "queued", "progress": 0, "downloaded": 0, "total": 0,
                "error": None, "filename": destination.filename, "save_path": model_type,
            }
            self.pending.append((download_id, url, destination))
            if self.worker is None or self.worker.done():
                self.worker = asyncio.create_task(self.run_queue())
        return web.json_response({"download_id": download_id}, status=202)

    async def restart(self, request):
        async with self.lock:
            if (self.restarting or self.active_uploads or self.pending
                    or (self.worker is not None and not self.worker.done())):
                raise TransferError("Model transfers are active or a restart is already pending", 409)
            self.restarting = True
            self.restart_task = asyncio.create_task(self.terminate_after_response())
        return web.json_response({"status": "restarting"}, status=202)

    async def terminate_after_response(self):
        await asyncio.sleep(RESTART_DELAY_SECONDS)
        os.kill(os.getpid(), signal.SIGTERM)

    async def run_queue(self):
        while self.pending:
            download_id, url, destination = self.pending.pop(0)
            state = self.downloads[download_id]
            state["status"] = "downloading"
            try:
                await self.download(url, destination, state)
                destination.publish()
                state.update(status="completed", progress=100)
            except asyncio.CancelledError:
                state.update(status="error", error="Model transfer stopped")
                raise
            except TransferError as exc:
                state.update(status="error", error=str(exc))
            except Exception:
                # Network exceptions can contain signed URLs or remote credentials.
                state.update(status="error", error="Model transfer failed")
            finally:
                destination.close()

    async def shutdown(self):
        if self.restart_task is not None and not self.restart_task.done():
            self.restart_task.cancel()
            await asyncio.gather(self.restart_task, return_exceptions=True)
        if self.worker is not None and not self.worker.done():
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
        for download_id, _, destination in self.pending:
            destination.close()
            self.downloads[download_id].update(status="error", error="Model transfer stopped")
        self.pending.clear()

    def session(self):
        connector = aiohttp.TCPConnector(resolver=PublicResolver(), use_dns_cache=False,
                                        force_close=True)
        return aiohttp.ClientSession(
            connector=connector, trust_env=False, auto_decompress=False,
            cookie_jar=aiohttp.DummyCookieJar(),
            timeout=aiohttp.ClientTimeout(total=24 * 3600, connect=30, sock_read=120),
        )

    async def download(self, url, destination, state):
        async with self.session() as session:
            for redirect in range(6):
                validate_url(url, self.extra_hosts)
                parsed = urlsplit(url)
                is_hf = (parsed.scheme == "https"
                         and parsed.hostname in ("huggingface.co", "hf.co")
                         and parsed.port in (None, 443))
                headers = {"Accept-Encoding": "identity"}
                if is_hf and self.hf_token:
                    headers["Authorization"] = "Bearer " + self.hf_token
                async with session.get(url, allow_redirects=False,
                                       headers=headers) as response:
                    if response.status in (301, 302, 303, 307, 308):
                        location = response.headers.get("Location")
                        if not location or redirect == 5:
                            raise TransferError("Invalid or excessive model download redirects")
                        url = urljoin(url, location)
                        continue
                    if is_hf and response.status in (401, 403):
                        raise TransferError(
                            "Hugging Face access denied. Configure HF_TOKEN with a read token "
                            "and ensure the account has access to the gated model."
                        )
                    if response.status != 200:
                        raise TransferError("Model server did not return a successful file response")
                    content_type = response.headers.get("Content-Type", "").lower()
                    if any(item in content_type for item in ("text/", "html", "json", "xml")):
                        raise TransferError("Model server returned a document instead of model data")
                    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                        raise TransferError("Encoded model responses are not supported")
                    length = response.content_length
                    if length is not None and (length <= 0 or length > self.max_bytes):
                        raise TransferError("Model exceeds the configured size limit", 413)
                    state["total"] = length or 0
                    await self.stream(response.content.iter_chunked(CHUNK_BYTES),
                                      destination, state)
                    if length is not None and length != state["downloaded"]:
                        raise TransferError("Incomplete model response")
                    return
            raise TransferError("Excessive model download redirects")

    async def stream(self, chunks, destination, state):
        prefix = bytearray()
        prefix_checked = False
        async for chunk in chunks:
            if not chunk:
                continue
            state["downloaded"] += len(chunk)
            if state["downloaded"] > self.max_bytes:
                raise TransferError("Model exceeds the configured size limit", 413)
            if not prefix_checked:
                prefix.extend(chunk)
                if len(prefix) < 512:
                    continue
                self.check_prefix(prefix)
                destination.file.write(prefix)
                prefix.clear()
                prefix_checked = True
            else:
                destination.file.write(chunk)
            if state["total"]:
                state["progress"] = min(99, state["downloaded"] * 100 / state["total"])
        if not state["downloaded"]:
            raise TransferError("Empty model file")
        if not prefix_checked:
            self.check_prefix(prefix)
            destination.file.write(prefix)

    @staticmethod
    def check_prefix(prefix):
        sample = bytes(prefix[:512]).lstrip().lower()
        if sample.startswith((b"<!doctype html", b"<html", b"<?xml", b"<head", b"<body")):
            raise TransferError("Received a document instead of model data")

    async def upload(self, request):
        if request.content_type != "multipart/form-data":
            raise TransferError("Expected multipart form data")
        reader = await request.multipart()
        field = await reader.next()
        if field is None or field.name != "save_path" or field.filename is not None:
            raise TransferError("The save_path field must precede the file field")
        save_path_bytes = await field.read_chunk(size=8192)
        if len(save_path_bytes) > 128 or not field.at_eof():
            raise TransferError("Invalid model type")
        save_path = save_path_bytes.decode("utf-8")
        field = await reader.next()
        if field is None or field.name != "file" or not field.filename:
            raise TransferError("Expected a file field after save_path")
        if field.headers.get("Content-Transfer-Encoding"):
            raise TransferError("Encoded model uploads are not supported")
        async with self.lock:
            if self.restarting:
                raise TransferError("ComfyUI restart is pending", 409)
            model_type, destination = self.destination(save_path, field.filename)
            self.active_uploads += 1
        state = {"downloaded": 0, "total": 0, "progress": 0}

        async def chunks():
            while not field.at_eof():
                yield await field.read_chunk(size=CHUNK_BYTES)

        try:
            await self.stream(chunks(), destination, state)
            if await reader.next() is not None:
                raise TransferError("Unexpected extra multipart fields")
            destination.publish()
        finally:
            try:
                destination.close()
            finally:
                self.active_uploads -= 1
        return web.json_response({"filename": destination.filename, "save_path": model_type,
                                  "bytes": state["downloaded"]}, status=201)
