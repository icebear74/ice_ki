"""
LLM Manager – loads GGUF models via llama-cpp-python and provides
a thread-safe chat_completion() interface.

GPU assignment
--------------
Each model config contains a 'gpu' index.  Before loading a model
CUDA_VISIBLE_DEVICES is NOT modified globally (that would break the other
model).  Instead we pass the device index directly to Llama via the
tensor_split mechanism: a tensor_split list with 1.0 on the target GPU
and 0.0 on all others routes all layers to that GPU.

Thread-safety
-------------
llama-cpp-python is not thread-safe.  Each model gets its own
threading.Lock that must be held during inference.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
    _LLAMA_AVAILABLE = True
except ImportError:
    Llama = None  # type: ignore[assignment,misc]
    _LLAMA_AVAILABLE = False
    logger.warning("llama-cpp-python not installed – LLM features disabled.")


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

        logger.info("Loading model '%s' from %s on GPU %d …", name, path, gpu_index)
        try:
            # Build a tensor_split that routes everything to the target GPU.
            # We need to know the total GPU count; default to 2 (P100 + P4).
            # llama-cpp-python ignores extra entries, so passing [0,1,0,...] is safe.
            n_gpus = 2
            tensor_split = [0.0] * n_gpus
            tensor_split[gpu_index] = 1.0

            llm = Llama(
                model_path=path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                tensor_split=tensor_split,
                verbose=False,
            )
            self._models[name] = _ModelHandle(name, llm)
            logger.info("Model '%s' loaded successfully.", name)
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
            status[name] = {
                "loaded": True,
                "model_path": getattr(handle.llm, "model_path", "unknown"),
            }
        # List models that were never loaded (not in _models but in config).
        return status

    def is_ready(self, model_name: str) -> bool:
        return model_name in self._models
