"""
Intent Router – classifies user messages using the small router LLM (P4).

Phase 1 behaviour
-----------------
The router classifies the intent and returns a RouterResult, but the result
is only logged – the orchestrator always forwards to the main LLM regardless.

Supported intents
-----------------
general, weather, wiki, recipe, movie, memory_store, memory_recall
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from models import RouterResult

if TYPE_CHECKING:
    from llm_manager import LLMManager

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an intent classifier. Given a user message, output ONLY a JSON object – no prose, no markdown.

Available intents:
- general        : general conversation or questions
- weather        : weather queries (current or forecast)
- wiki           : factual / encyclopaedic questions
- recipe         : cooking / food recipes
- movie          : film recommendations or information
- memory_store   : user wants to save personal information
- memory_recall  : user asks about previously stored personal information

JSON schema:
{
  "intent": "<one of the intents above>",
  "confidence": <float 0.0-1.0>,
  "entities": {}
}
"""


class IntentRouter:
    """Wraps the router LLM and provides a classify() method."""

    def __init__(self, llm_manager: "LLMManager", model_name: str = "router") -> None:
        self._llm = llm_manager
        self._model_name = model_name

    def classify(self, user_message: str) -> RouterResult:
        """Classify *user_message* and return a RouterResult.

        Falls back to intent="general" if the model is unavailable or
        returns unparseable output.
        """
        if not self._llm.is_ready(self._model_name):
            logger.debug("Router model not loaded – defaulting to 'general'.")
            return RouterResult(intent="general", confidence=0.0)

        from models import ChatMessage  # noqa: PLC0415

        messages = [
            ChatMessage(role="system", content=_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_message),
        ]

        try:
            raw = self._llm.chat_completion(
                model_name=self._model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=128,
            )
            # Strip markdown code fences if present.
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            result = RouterResult(**data)
            logger.info("Intent classified: %s (%.2f)", result.intent, result.confidence)
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("Intent classification failed (%s) – defaulting to 'general'.", exc)
            return RouterResult(intent="general", confidence=0.0)
