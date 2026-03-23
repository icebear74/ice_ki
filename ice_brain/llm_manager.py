"""
LLM Manager – loads GGUF models via llama-cpp-python and provides
a thread-safe chat_completion() interface.

GPU assignment
--------------
Each model config contains a 'gpu' index (0-based CUDA device index).
We pass it directly as `main_gpu` to Llama together with `n_gpu_layers=-1`
so that all transformer layers are offloaded to that specific GPU.

`tensor_split` is intentionally NOT used here: that parameter splits a
single model across multiple GPUs, which is the opposite of what we want
(two separate models, each on its own GPU).

CUDA build requirement
----------------------
llama-cpp-python must be compiled with CUDA support, otherwise `n_gpu_layers`
is silently ignored and everything runs on CPU:

    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-binary llama-cpp-python --no-cache-dir

The manager detects CPU-only builds at startup and logs a prominent warning.

Thread-safety
-------------
llama-cpp-python is not thread-safe.  Each model gets its own
threading.Lock that must be held during inference.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
    _LLAMA_AVAILABLE = True
except ImportError:
    Llama = None  # type: ignore[assignment,misc]
    _LLAMA_AVAILABLE = False
    logger.warning("llama-cpp-python not installed – LLM features disabled.")


def _check_cuda_build() -> bool:
    """Return True if llama-cpp-python was compiled with CUDA support."""
    if not _LLAMA_AVAILABLE:
        return False
    try:
        import llama_cpp  # noqa: PLC0415
        # llama_cpp exposes the list of supported GGML backends since ~0.2.x
        support = getattr(llama_cpp, "llama_supports_gpu_offload", None)
        if support is not None:
            return bool(support())
        # Fallback: inspect the compiled-in backend list when available
        backend_list = getattr(llama_cpp, "_llama_backend_list", None)
        if backend_list is not None:
            return any("cuda" in str(b).lower() or "cublas" in str(b).lower() for b in backend_list)
        # Last resort: try to load a tiny dummy model on GPU and see if any
        # GPU memory is allocated – we skip this to avoid side effects and
        # instead rely on the log line llama.cpp always prints at load time.
        return False
    except Exception:  # noqa: BLE001
        return False


class _ModelHandle:
    """Container for a loaded model + its lock."""

    def __init__(self, name: str, llm: Any) -> None:
        self.name = name
        self.llm = llm
        self.lock = threading.Lock()


class LLMManager:
    """Manages multiple GGUF models and routes inference requests."""

    def __init__(self) -> None:
        self._models: Dict[str, _ModelHandle] = {}
        self._cuda_available = _check_cuda_build()
        if not self._cuda_available and _LLAMA_AVAILABLE:
            logger.warning(
                "=" * 70
            )
            logger.warning(
                "WARNING: llama-cpp-python has NO CUDA support – models run on CPU!"
            )
            logger.warning(
                "Reinstall with CUDA enabled:"
            )
            logger.warning(
                '  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python '
                "--upgrade --force-reinstall --no-binary llama-cpp-python --no-cache-dir"
            )
            logger.warning(
                "=" * 70
            )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_model(self, name: str, cfg: dict) -> bool:
        """Load a GGUF model onto the configured GPU.

        If the file at *path* does not exist and *hf_repo* + *hf_file* are
        configured, the model is downloaded automatically from HuggingFace
        before loading.

        Returns True on success, False on failure (server continues).
        """
        if not _LLAMA_AVAILABLE:
            logger.error("Cannot load '%s': llama-cpp-python is not installed.", name)
            return False

        path = cfg.get("path", "")
        gpu_index = cfg.get("gpu", 0)
        n_ctx = cfg.get("n_ctx", 4096)
        n_gpu_layers = cfg.get("n_gpu_layers", -1)

        # Auto-download from HuggingFace if the file is missing
        if path and not Path(path).exists():
            hf_repo = cfg.get("hf_repo", "")
            hf_file = cfg.get("hf_file", "")
            if hf_repo and hf_file:
                path = self._download_from_hf(name, path, hf_repo, hf_file)
                if path is None:
                    return False
            else:
                logger.error(
                    "Cannot load model '%s': file not found at '%s' and no "
                    "hf_repo/hf_file configured for auto-download.",
                    name, path,
                )
                return False

        if not self._cuda_available:
            logger.warning(
                "Loading model '%s' on CPU (CUDA build missing). "
                "n_gpu_layers ignored.",
                name,
            )
            effective_gpu_layers = 0
        else:
            effective_gpu_layers = n_gpu_layers

        logger.info(
            "Loading model '%s' from %s  →  GPU %d, n_gpu_layers=%s …",
            name, path, gpu_index,
            "all" if effective_gpu_layers == -1 else effective_gpu_layers,
        )
        try:
            llm = Llama(
                model_path=path,
                n_ctx=n_ctx,
                n_gpu_layers=effective_gpu_layers,
                # main_gpu selects which CUDA device receives the model.
                # This is the correct parameter for single-GPU assignment;
                # tensor_split is only for splitting ONE model across GPUs.
                main_gpu=gpu_index,
                verbose=False,
            )
            self._models[name] = _ModelHandle(name, llm)
            logger.info(
                "Model '%s' loaded.  GPU offload: %s",
                name,
                "yes (GPU %d)" % gpu_index if effective_gpu_layers != 0 else "no (CPU)",
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to load model '%s': %s\n"
                "  → Check that the file exists at '%s' and that GPU %d has enough VRAM.",
                name, exc, path, gpu_index,
            )
            return False

    def _download_from_hf(self, name: str, path: str, hf_repo: str, hf_file: str) -> str | None:
        """Download *hf_file* from *hf_repo* into the directory of *path*.

        Returns the local file path on success, None on failure.
        """
        try:
            from huggingface_hub import hf_hub_download  # noqa: PLC0415
        except ImportError:
            logger.error(
                "Cannot auto-download model '%s': huggingface-hub is not installed. "
                "Run: pip install huggingface-hub",
                name,
            )
            return None

        target_dir = str(Path(path).parent)
        os.makedirs(target_dir, exist_ok=True)
        logger.info(
            "Model '%s' not found at '%s' – downloading from HuggingFace: %s / %s …",
            name, path, hf_repo, hf_file,
        )
        hf_token = os.getenv("HF_TOKEN") or None
        if not hf_token:
            logger.warning(
                "HF_TOKEN is not set. Downloads may fail for gated or rate-limited "
                "repositories. Set HF_TOKEN in config.py or as an environment variable."
            )
        try:
            local_path = hf_hub_download(
                repo_id=hf_repo,
                filename=hf_file,
                local_dir=target_dir,
                token=hf_token,
            )
            logger.info("Model '%s' downloaded to '%s'.", name, local_path)
            return local_path
        except Exception as exc:  # noqa: BLE001
            exc_str = str(exc)
            if "404" in exc_str or "Entry Not Found" in exc_str:
                logger.error(
                    "Failed to download model '%s' from HuggingFace (%s/%s): %s\n"
                    "  → Verify the repo and filename are correct at "
                    "https://huggingface.co/%s\n"
                    "  → If the repo is gated, set HF_TOKEN in config.py after "
                    "accepting the model terms at https://huggingface.co/%s",
                    name, hf_repo, hf_file, exc, hf_repo, hf_repo,
                )
            else:
                logger.error(
                    "Failed to download model '%s' from HuggingFace (%s/%s): %s",
                    name, hf_repo, hf_file, exc,
                )
            return None

    def load_all(self, models_cfg: dict) -> None:
        """Load all models defined in the MODELS config dict."""
        for name, cfg in models_cfg.items():
            self.load_model(name, cfg)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def chat_completion(
        self,
        model_name: str,
        messages: List[Any],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """Generate a response and return the assistant content string.

        Raises RuntimeError if the model is not loaded.
        """
        if model_name not in self._models:
            raise RuntimeError(
                f"Model '{model_name}' is not loaded. "
                f"Available models: {list(self._models)}"
            )

        handle = self._models[model_name]
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]

        with handle.lock:
            result = handle.llm.create_chat_completion(
                messages=msg_dicts,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        return result["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return which models are loaded and basic metadata."""
        status: Dict[str, Any] = {}
        for name, handle in self._models.items():
            llm = handle.llm
            status[name] = {
                "loaded": True,
                "model_path": getattr(llm, "model_path", "unknown"),
                "gpu_offload": self._cuda_available,
                "n_gpu_layers": getattr(llm, "n_gpu_layers", "unknown"),
            }
        return status

    def is_ready(self, model_name: str) -> bool:
        return model_name in self._models
