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
import threading
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

        Returns True on success, False on failure (server continues).
        """
        if not _LLAMA_AVAILABLE:
            logger.error("Cannot load '%s': llama-cpp-python is not installed.", name)
            return False

        path = cfg.get("path", "")
        gpu_index = cfg.get("gpu", 0)
        n_ctx = cfg.get("n_ctx", 4096)
        n_gpu_layers = cfg.get("n_gpu_layers", -1)

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
