from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

import main as webui_main


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
