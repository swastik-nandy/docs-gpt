from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from rag.generation.memory.schema import MemoryFactType


_ALLOWED_FACT_TYPES: set[str] = {
    "preference",
    "project_context",
    "technical_context",
    "user_profile",
    "conversation_fact",
}


@dataclass(frozen=True, slots=True)
class ExtractedMemoryFact:
    text: str
    fact_type: MemoryFactType
    confidence: float = 0.8
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryExtractionParseError(ValueError):
    pass


def parse_memory_extraction_response(raw_text: str) -> list[ExtractedMemoryFact]:
    payload = _load_json_payload(raw_text)

    raw_facts: Any
    if isinstance(payload, dict):
        raw_facts = payload.get("facts", [])
    elif isinstance(payload, list):
        raw_facts = payload
    else:
        raise MemoryExtractionParseError("Memory extraction response must be a JSON object or list.")

    if not isinstance(raw_facts, list):
        raise MemoryExtractionParseError("'facts' must be a list.")

    facts: list[ExtractedMemoryFact] = []

    for index, raw_fact in enumerate(raw_facts):
        if not isinstance(raw_fact, dict):
            continue

        text = str(raw_fact.get("text") or raw_fact.get("fact") or "").strip()
        if not text:
            continue

        fact_type = str(raw_fact.get("fact_type") or "conversation_fact").strip()
        if fact_type not in _ALLOWED_FACT_TYPES:
            fact_type = "conversation_fact"

        confidence = _coerce_confidence(raw_fact.get("confidence", 0.8))
        metadata = raw_fact.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        facts.append(
            ExtractedMemoryFact(
                text=text,
                fact_type=fact_type,  # type: ignore[arg-type]
                confidence=confidence,
                metadata={
                    **metadata,
                    "extraction_index": index,
                },
            )
        )

    return facts


def _load_json_payload(raw_text: str) -> Any:
    cleaned = raw_text.strip()
    cleaned = _strip_markdown_json_fence(cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        json_blob = _extract_first_json_blob(cleaned)
        if json_blob is None:
            raise MemoryExtractionParseError("Could not parse memory extraction JSON.")
        return json.loads(json_blob)


def _strip_markdown_json_fence(text: str) -> str:
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()

    return text


def _extract_first_json_blob(text: str) -> str | None:
    start_candidates = [position for position in (text.find("{"), text.find("[")) if position != -1]
    if not start_candidates:
        return None

    start = min(start_candidates)
    opener = text[start]
    closer = "}" if opener == "{" else "]"

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]

        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.8

    return max(0.0, min(1.0, confidence))
