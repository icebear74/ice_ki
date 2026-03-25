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
import re
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
- wiki           : factual / encyclopaedic questions about people, places, concepts, history, or products
- recipe         : cooking / food recipes
- movie          : film recommendations or information
- memory_store   : user wants to save personal information
- memory_recall  : user asks about previously stored personal information
- news           : current events, breaking news, today's headlines
- sports         : sports scores, results, standings, fixtures (e.g. Bundesliga, football)
- web_search     : questions requiring up-to-date web information that Wikipedia cannot answer
                   (prices, stocks, recent software releases, live events, etc.)

JSON schema:
{
  "intent": "<one of the intents above>",
  "confidence": <float 0.0-1.0>,
  "entities": {
    "topic": "<main topic or search term extracted from the message, or empty string if none>"
  }
}

The "topic" field should contain the core subject the user is asking about, stripped of question
words and filler phrases. Examples: "Albert Einstein", "Berlin", "Bayern München", "Bitcoin Kurs".
Leave it empty ("") for general chat, memory operations, or when no specific topic is identifiable.

Few-shot examples:
---
User: Was weißt du über Albert Einstein?
{"intent": "wiki", "confidence": 0.98, "entities": {"topic": "Albert Einstein"}}
---
User: Who was Marie Curie?
{"intent": "wiki", "confidence": 0.97, "entities": {"topic": "Marie Curie"}}
---
User: Wie ist das Wetter in Berlin?
{"intent": "weather", "confidence": 0.99, "entities": {"topic": "Berlin"}}
---
User: What's the weather like in London tomorrow?
{"intent": "weather", "confidence": 0.98, "entities": {"topic": "London"}}
---
User: Was gibt es heute Neues in den Nachrichten?
{"intent": "news", "confidence": 0.96, "entities": {"topic": ""}}
---
User: What's in the news today?
{"intent": "news", "confidence": 0.95, "entities": {"topic": ""}}
---
User: Wie hat Bayern München gespielt?
{"intent": "sports", "confidence": 0.97, "entities": {"topic": "Bayern München"}}
---
User: How did Real Madrid do last night?
{"intent": "sports", "confidence": 0.96, "entities": {"topic": "Real Madrid"}}
---
User: Was kostet Bitcoin aktuell?
{"intent": "web_search", "confidence": 0.94, "entities": {"topic": "Bitcoin Kurs"}}
---
User: What's the current price of Apple stock?
{"intent": "web_search", "confidence": 0.93, "entities": {"topic": "Apple stock price"}}
---
User: Kannst du mir ein Rezept für Pasta machen?
{"intent": "recipe", "confidence": 0.98, "entities": {"topic": "Pasta"}}
---
User: Give me a recipe for pizza.
{"intent": "recipe", "confidence": 0.97, "entities": {"topic": "Pizza"}}
---
User: Empfiehl mir einen Film.
{"intent": "movie", "confidence": 0.95, "entities": {"topic": ""}}
---
User: Recommend a good movie.
{"intent": "movie", "confidence": 0.94, "entities": {"topic": ""}}
---
User: Ich heiße Max Mustermann.
{"intent": "memory_store", "confidence": 0.97, "entities": {"topic": ""}}
---
User: My name is John Smith.
{"intent": "memory_store", "confidence": 0.96, "entities": {"topic": ""}}
---
User: Wie heiße ich?
{"intent": "memory_recall", "confidence": 0.98, "entities": {"topic": ""}}
---
User: What's my name?
{"intent": "memory_recall", "confidence": 0.97, "entities": {"topic": ""}}
---
User: Hallo, wie geht es dir?
{"intent": "general", "confidence": 0.99, "entities": {"topic": ""}}
---
User: Hello, how are you?
{"intent": "general", "confidence": 0.99, "entities": {"topic": ""}}
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
            # /no_think suppresses Qwen3's extended <think>…</think> block so the
            # model outputs the JSON intent object directly without burning tokens
            # on chain-of-thought reasoning.
            ChatMessage(role="user", content=f"{user_message}\n/no_think"),
        ]

        try:
            raw = self._llm.chat_completion(
                model_name=self._model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
            )
            logger.debug("Router input: %r", user_message)
            logger.debug("Router raw LLM output: %r", raw)
            # Strip markdown code fences if present.
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            # Strip <think>…</think> reasoning blocks emitted by some models.
            # Also handles unclosed blocks (output truncated before </think>).
            raw = re.sub(r"<think>.*?(?:</think>|$)", "", raw, flags=re.DOTALL)
            raw = raw.strip()
            if not raw:
                logger.warning("Router model returned empty response – defaulting to 'general'.")
                return RouterResult(intent="general", confidence=0.0)
            logger.debug("Router raw response: %r", raw)
            data = json.loads(raw)
            result = RouterResult(**data)
            logger.info("Intent classified: %s (%.2f)", result.intent, result.confidence)
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Intent classification failed (%s) – raw response: %r – defaulting to 'general'.",
                exc, raw if "raw" in dir() else "<not captured>",
            )
            return RouterResult(intent="general", confidence=0.0)
