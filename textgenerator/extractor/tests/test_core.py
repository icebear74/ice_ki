from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core  # noqa: E402


SAMPLE = {
    "name": "Elena Vasquez",
    "aliases": ["Lena"],
    "age": "29",
    "gender": "female",
    "species": None,
    "occupation": "bartender",
    "appearance": {
        "height": "tall",
        "build": None,
        "skin": "olive",
        "hair_color": "black",
        "hair_style": "braided",
        "eye_color": "green",
        "distinguishing_features": ["scar on the left cheek", "scar on the left cheek"],
    },
    "clothing": ["red dress"],
    "personality": {"summary": "guarded but warm", "traits": ["stubborn", "loyal"]},
    "speech_style": "dry humour",
    "background": "grew up in the harbour district",
    "relationships": [{"name": "Marco", "relation": "brother", "notes": None}, "unknown"],
    "scenario": None,
    "first_message": None,
    "example_dialogue": None,
    "tags": ["noir"],
}


class SafeFilenameTests(unittest.TestCase):
    def test_strips_path_traversal_and_separators(self) -> None:
        self.assertEqual(core.safe_filename("../../etc/passwd"), "etc_passwd")
        self.assertEqual(core.safe_filename("a/b\\c"), "a_b_c")

    def test_transliterates_and_truncates(self) -> None:
        self.assertEqual(core.safe_filename("Jürgen Groß"), "Jurgen_Gro")
        self.assertLessEqual(len(core.safe_filename("x" * 500)), core.MAX_FILENAME_LENGTH)

    def test_suffix_is_appended(self) -> None:
        self.assertEqual(core.safe_filename("Elena", suffix=".json"), "Elena.json")

    def test_empty_result_is_rejected(self) -> None:
        with self.assertRaises(core.ExtractionError):
            core.safe_filename("///")


class ParseModelJsonTests(unittest.TestCase):
    def test_plain_json(self) -> None:
        self.assertEqual(core.parse_model_json('{"name": "A"}'), {"name": "A"})

    def test_code_fence_is_removed(self) -> None:
        raw = '```json\n{"name": "A"}\n```'
        self.assertEqual(core.parse_model_json(raw), {"name": "A"})

    def test_surrounding_prose_is_ignored(self) -> None:
        raw = 'Sure! Here you go:\n{"name": "A", "note": "}"}\nHope that helps.'
        self.assertEqual(core.parse_model_json(raw), {"name": "A", "note": "}"})

    def test_empty_answer(self) -> None:
        with self.assertRaises(core.ExtractionError):
            core.parse_model_json("   ")

    def test_non_object_answer(self) -> None:
        with self.assertRaises(core.ExtractionError):
            core.parse_model_json("[1, 2, 3]")

    def test_unterminated_object(self) -> None:
        with self.assertRaises(core.ExtractionError):
            core.parse_model_json('{"name": "A"')


class NormalizeProfileTests(unittest.TestCase):
    def test_keeps_facts_and_drops_placeholders(self) -> None:
        profile = core.normalize_person_profile(SAMPLE)
        self.assertEqual(profile["name"], "Elena Vasquez")
        self.assertEqual(profile["aliases"], ["Lena"])
        self.assertIsNone(profile["species"])
        self.assertEqual(
            profile["appearance"]["distinguishing_features"], ["scar on the left cheek"]
        )
        self.assertEqual(profile["relationships"], [{"name": "Marco", "relation": "brother", "notes": None}])

    def test_unknown_keys_are_dropped_and_missing_keys_stay_empty(self) -> None:
        profile = core.normalize_person_profile({"name": "A", "evil": "x"})
        self.assertNotIn("evil", profile)
        self.assertEqual(profile["tags"], [])
        self.assertIsNone(profile["background"])
        self.assertEqual(sorted(profile), sorted(core.PERSON_PROFILE_TEMPLATE))

    def test_unknown_placeholder_values_become_null(self) -> None:
        profile = core.normalize_person_profile({"name": "A", "age": "unknown", "gender": " "})
        self.assertIsNone(profile["age"])
        self.assertIsNone(profile["gender"])

    def test_personality_string_is_accepted(self) -> None:
        profile = core.normalize_person_profile({"name": "A", "personality": "shy"})
        self.assertEqual(profile["personality"]["summary"], "shy")

    def test_missing_name_is_rejected(self) -> None:
        with self.assertRaises(core.ExtractionError):
            core.normalize_person_profile({"age": "20"})

    def test_template_is_not_mutated(self) -> None:
        core.normalize_person_profile(SAMPLE)
        self.assertEqual(core.PERSON_PROFILE_TEMPLATE["aliases"], [])
        self.assertEqual(core.PERSON_PROFILE_TEMPLATE["appearance"]["hair_color"], None)


class DerivedArtefactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = core.normalize_person_profile(SAMPLE)

    def test_confidence_between_zero_and_one(self) -> None:
        empty = core.normalize_person_profile({"name": "A"})
        self.assertGreater(core.profile_confidence(self.profile), core.profile_confidence(empty))
        self.assertLessEqual(core.profile_confidence(self.profile), 1.0)

    def test_visual_prompt_uses_extracted_facts_only(self) -> None:
        prompt = core.build_visual_prompt(self.profile)
        self.assertEqual(prompt["safety_mode"], "off")
        self.assertIn("green eyes", prompt["positive"])
        self.assertIn("red dress", prompt["positive"])
        self.assertNotIn("nsfw", prompt["negative"])
        self.assertTrue(prompt["has_appearance_data"])

    def test_visual_prompt_sfw_mode_adds_wording(self) -> None:
        prompt = core.build_visual_prompt(self.profile, safety_mode="sfw")
        self.assertIn("safe for work", prompt["positive"])
        self.assertIn("nsfw", prompt["negative"])

    def test_visual_prompt_without_appearance(self) -> None:
        prompt = core.build_visual_prompt(core.normalize_person_profile({"name": "A"}))
        self.assertFalse(prompt["has_appearance_data"])
        self.assertEqual(prompt["positive"], "A")

    def test_character_card_is_v2(self) -> None:
        card = core.build_character_card(self.profile)
        self.assertEqual(card["spec"], "chara_card_v2")
        self.assertEqual(card["spec_version"], "2.0")
        self.assertEqual(card["data"]["name"], "Elena Vasquez")
        self.assertIn("Relationship: Marco - brother", card["data"]["description"])
        self.assertIn("guarded but warm", card["data"]["personality"])
        self.assertEqual(card["data"]["scenario"], "")
        self.assertEqual(card["name"], card["data"]["name"])

    def test_metadata_contains_confidence_and_timestamp(self) -> None:
        meta = core.build_source_metadata(
            source_name="story.txt", model="mymodel", profile=self.profile
        )
        self.assertEqual(meta["source"], "story.txt")
        self.assertEqual(meta["model"], "mymodel")
        self.assertTrue(meta["extracted_at"].endswith("Z"))
        self.assertEqual(meta["confidence"], core.profile_confidence(self.profile))


class WriteJsonTests(unittest.TestCase):
    def test_deterministic_encoding(self) -> None:
        self.assertEqual(core.dump_json({"b": 1, "a": "ä"}), '{\n  "a": "ä",\n  "b": 1\n}\n')

    def test_no_overwrite_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "card.json"
            core.write_json_file(path, {"a": 1})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"a": 1})
            with self.assertRaises(FileExistsError):
                core.write_json_file(path, {"a": 2})
            core.write_json_file(path, {"a": 2}, allow_overwrite=True)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"a": 2})


if __name__ == "__main__":
    unittest.main()
