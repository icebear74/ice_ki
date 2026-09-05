"""Thin client for the Oobabooga OpenAI-compatible chat completions API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from core import ExtractionError

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "character_extraction.md"


def render_prompt(story_text: str, character_name: str | None) -> str:
    """Render the extraction prompt.

    ``str.format`` is deliberately not used: the template embeds a JSON schema
    full of curly braces.
    """
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace(
        "{character_name}", character_name or "the main character of the text"
    ).replace("{story_text}", story_text)


class BackendError(RuntimeError):
    """The model backend was unreachable or answered with an error."""


class OobaboogaClient:
    """Minimal chat-completions client - no moderation layer, by design."""

    def __init__(self, base_url: str, *, model: str = "", timeout: float = 600.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    @property
    def base_url(self) -> str:
        return self._base_url

    def complete(self, prompt: str, *, max_tokens: int, temperature: float = 0.2) -> str:
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if self._model:
            payload["model"] = self._model

        try:
            response = httpx.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise BackendError(f"model backend request failed: {exc}") from exc
        except ValueError as exc:
            raise BackendError("model backend returned a non-JSON response") from exc

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ExtractionError("unexpected chat completion response shape") from exc
