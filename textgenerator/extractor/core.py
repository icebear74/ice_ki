"""Pure, dependency free core logic of the character extractor.

Everything in this module works on plain Python data structures so it can be
unit tested without a running model backend, without FastAPI and without a
Kubernetes cluster.

Guiding rule for all normalisation code below: **never invent data**. Fields
the model did not provide stay ``null`` (scalars) or empty (lists/objects).
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_FILENAME_LENGTH = 80

_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

#: Structured person profile skeleton. Keys are stable; values describe the
#: expected "empty" value of the field.
PERSON_PROFILE_TEMPLATE: dict[str, Any] = {
    "name": None,
    "aliases": [],
    "age": None,
    "gender": None,
    "species": None,
    "occupation": None,
    "appearance": {
        "height": None,
        "build": None,
        "skin": None,
        "hair_color": None,
        "hair_style": None,
        "eye_color": None,
        "distinguishing_features": [],
    },
    "clothing": [],
    "personality": {
        "summary": None,
        "traits": [],
    },
    "speech_style": None,
    "background": None,
    "relationships": [],
    "scenario": None,
    "first_message": None,
    "example_dialogue": None,
    "tags": [],
}

_RELATIONSHIP_KEYS = ("name", "relation", "notes")


class ExtractionError(ValueError):
    """Raised when the model answer cannot be turned into a valid profile."""


# ---------------------------------------------------------------------------
# Filenames
# ---------------------------------------------------------------------------
def safe_filename(name: str, *, suffix: str = "") -> str:
    """Return a safe, deterministic file name for ``name``.

    Path separators, traversal sequences and control characters are removed;
    the result never escapes its target directory.
    """
    normalised = unicodedata.normalize("NFKD", str(name))
    ascii_only = normalised.encode("ascii", "ignore").decode("ascii")
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", ascii_only).strip("._-")
    cleaned = cleaned[:MAX_FILENAME_LENGTH]
    if not cleaned:
        raise ExtractionError("character name does not contain usable characters")
    return f"{cleaned}{suffix}"


# ---------------------------------------------------------------------------
# Model answer parsing
# ---------------------------------------------------------------------------
def parse_model_json(raw_answer: str) -> dict[str, Any]:
    """Extract a JSON object from a raw model answer.

    Handles markdown code fences and leading/trailing prose, which LLMs emit
    regularly even when asked not to.
    """
    if not isinstance(raw_answer, str) or not raw_answer.strip():
        raise ExtractionError("model returned an empty answer")

    candidate = _CODE_FENCE.sub("", raw_answer.strip())
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        parsed = _load_first_json_object(candidate)

    if not isinstance(parsed, dict):
        raise ExtractionError("model answer is not a JSON object")
    return parsed


def _load_first_json_object(text: str) -> Any:
    start = text.find("{")
    if start == -1:
        raise ExtractionError("model answer does not contain a JSON object")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : index + 1])
                except json.JSONDecodeError as exc:
                    raise ExtractionError(f"invalid JSON in model answer: {exc}") from exc
    raise ExtractionError("model answer contains an unterminated JSON object")


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def _clean_text(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.lower() not in {"null", "none", "unknown", "n/a", "-"}:
            return stripped
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _clean_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        cleaned = _clean_text(item)
        if cleaned is not None and cleaned not in result:
            result.append(cleaned)
    return result


def _clean_relationships(value: Any) -> list[dict[str, str | None]]:
    if not isinstance(value, list):
        return []
    relationships: list[dict[str, str | None]] = []
    for item in value:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            continue
        entry = {key: _clean_text(item.get(key)) for key in _RELATIONSHIP_KEYS}
        if any(entry.values()):
            relationships.append(entry)
    return relationships


def normalize_person_profile(raw: dict[str, Any]) -> dict[str, Any]:
    """Map a raw model answer onto :data:`PERSON_PROFILE_TEMPLATE`.

    Unknown keys from the model are dropped, missing keys stay empty.
    """
    if not isinstance(raw, dict):
        raise ExtractionError("profile data is not an object")

    profile = json.loads(json.dumps(PERSON_PROFILE_TEMPLATE))  # deep copy

    for key in ("name", "age", "gender", "species", "occupation", "speech_style",
                "background", "scenario", "first_message", "example_dialogue"):
        profile[key] = _clean_text(raw.get(key))

    profile["aliases"] = _clean_list(raw.get("aliases"))
    profile["clothing"] = _clean_list(raw.get("clothing"))
    profile["tags"] = _clean_list(raw.get("tags"))
    profile["relationships"] = _clean_relationships(raw.get("relationships"))

    appearance_raw = raw.get("appearance")
    appearance_raw = appearance_raw if isinstance(appearance_raw, dict) else {}
    for key in ("height", "build", "skin", "hair_color", "hair_style", "eye_color"):
        profile["appearance"][key] = _clean_text(appearance_raw.get(key))
    profile["appearance"]["distinguishing_features"] = _clean_list(
        appearance_raw.get("distinguishing_features")
    )

    personality_raw = raw.get("personality")
    if isinstance(personality_raw, str):
        personality_raw = {"summary": personality_raw}
    personality_raw = personality_raw if isinstance(personality_raw, dict) else {}
    profile["personality"]["summary"] = _clean_text(personality_raw.get("summary"))
    profile["personality"]["traits"] = _clean_list(personality_raw.get("traits"))

    if profile["name"] is None:
        raise ExtractionError("the extracted profile has no character name")
    return profile


def profile_confidence(profile: dict[str, Any]) -> float:
    """Fraction of profile fields that actually carry information (0.0-1.0)."""
    filled = 0
    total = 0
    for key, empty_value in PERSON_PROFILE_TEMPLATE.items():
        value = profile.get(key)
        if isinstance(empty_value, dict):
            for sub_key in empty_value:
                total += 1
                if value.get(sub_key) if isinstance(value, dict) else None:
                    filled += 1
            continue
        total += 1
        if value:
            filled += 1
    return round(filled / total, 3) if total else 0.0


def build_source_metadata(
    *,
    source_name: str | None,
    model: str | None,
    profile: dict[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Confidence and provenance metadata stored next to the profile."""
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return {
        "source": source_name,
        "model": model or None,
        "extracted_at": timestamp.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "confidence": profile_confidence(profile),
        "note": (
            "Fields are null/empty when the source text did not contain the "
            "information. Nothing is inferred or invented."
        ),
    }


# ---------------------------------------------------------------------------
# Derived artefacts
# ---------------------------------------------------------------------------
def _joined(values: list[str]) -> str | None:
    return ", ".join(values) if values else None


def describe_appearance(profile: dict[str, Any]) -> str | None:
    """Human readable appearance summary built from extracted facts only."""
    appearance = profile.get("appearance", {})
    parts = [
        appearance.get("height"),
        appearance.get("build"),
        f"{appearance['skin']} skin" if appearance.get("skin") else None,
        " ".join(
            value
            for value in (appearance.get("hair_style"), appearance.get("hair_color"))
            if value
        )
        + " hair"
        if (appearance.get("hair_color") or appearance.get("hair_style"))
        else None,
        f"{appearance['eye_color']} eyes" if appearance.get("eye_color") else None,
        _joined(appearance.get("distinguishing_features", [])),
        _joined(profile.get("clothing", [])),
    ]
    filtered = [part for part in parts if part]
    return ", ".join(filtered) if filtered else None


def build_visual_prompt(profile: dict[str, Any], *, safety_mode: str = "off") -> dict[str, Any]:
    """Build an image generation prompt from extracted facts only.

    ``safety_mode`` is the single, documented content policy switch:

    * ``"off"``   - default; no extra wording is added (uncensored).
    * ``"sfw"``   - appends safe-for-work wording to positive/negative prompt.
    """
    subject_parts = [
        profile.get("name"),
        profile.get("age"),
        profile.get("gender"),
        profile.get("species"),
        profile.get("occupation"),
    ]
    positive_parts = [part for part in subject_parts if part]
    appearance = describe_appearance(profile)
    if appearance:
        positive_parts.append(appearance)

    negative_parts = ["lowres", "deformed", "extra limbs", "watermark", "text"]
    if safety_mode == "sfw":
        positive_parts.append("safe for work, fully clothed")
        negative_parts.extend(["nsfw", "nudity"])

    return {
        "safety_mode": safety_mode,
        "positive": ", ".join(positive_parts),
        "negative": ", ".join(negative_parts),
        "has_appearance_data": appearance is not None,
    }


def build_character_card(profile: dict[str, Any]) -> dict[str, Any]:
    """Build a SillyTavern V2 compatible character card from the profile."""
    name = profile["name"]
    description_parts = []
    appearance = describe_appearance(profile)
    for label, value in (
        ("Age", profile.get("age")),
        ("Gender", profile.get("gender")),
        ("Species", profile.get("species")),
        ("Occupation", profile.get("occupation")),
        ("Aliases", _joined(profile.get("aliases", []))),
        ("Appearance", appearance),
        ("Speech", profile.get("speech_style")),
    ):
        if value:
            description_parts.append(f"{label}: {value}")
    for relationship in profile.get("relationships", []):
        rendered = " - ".join(
            value for value in (relationship.get("name"), relationship.get("relation"),
                                relationship.get("notes")) if value
        )
        if rendered:
            description_parts.append(f"Relationship: {rendered}")

    personality = profile["personality"]
    personality_text = " ".join(
        value for value in (personality.get("summary"), _joined(personality.get("traits", [])))
        if value
    )

    card_data = {
        "name": name,
        "description": "\n".join(description_parts),
        "personality": personality_text,
        "scenario": profile.get("scenario") or "",
        "first_mes": profile.get("first_message") or "",
        "mes_example": profile.get("example_dialogue") or "",
        "creator_notes": profile.get("background") or "",
        "system_prompt": "",
        "post_history_instructions": "",
        "alternate_greetings": [],
        "character_book": None,
        "tags": profile.get("tags", []),
        "creator": "ice_ki character-extractor",
        "character_version": "1.0",
        "extensions": {},
    }

    return {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": card_data,
        # V1 fields kept at the top level for older SillyTavern readers.
        "name": card_data["name"],
        "description": card_data["description"],
        "personality": card_data["personality"],
        "scenario": card_data["scenario"],
        "first_mes": card_data["first_mes"],
        "mes_example": card_data["mes_example"],
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def dump_json(payload: Any) -> str:
    """Deterministic UTF-8 JSON encoding used for every written file."""
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_json_file(path: Path, payload: Any, *, allow_overwrite: bool = False) -> Path:
    """Write ``payload`` as JSON, refusing to replace existing files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"{path} already exists")
    path.write_text(dump_json(payload), encoding="utf-8")
    return path
