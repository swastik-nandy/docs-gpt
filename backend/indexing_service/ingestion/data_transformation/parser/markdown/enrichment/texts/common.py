# rag/ingestion/data_transformation/parser/adapters/markdown/enrichment/texts/common.py

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, Mapping


#------------------------ PRESERVED METADATA -------------------------

PRESERVED_META_KEYS = (
    "links",
    "images",
    "inline_code",
    "html_inline",
    "html_block",
    "strong_text",
    "label",
    "kind",
    "code_blocks",
    "heading_anchor",
    "cell_meta",
    "raw_items",
    "source",
    "structural_role",
    "local_scope_id",
    "local_scope_title",
    "selector_metadata",
)


#------------------------ TEXT NORMALIZATION -------------------------

def clean_text(text: Any) -> str:
    """
    Normalize text without destroying parser-level meaning.

    Paragraph boundaries and line breaks remain available for later semantic
    packing and sentence-safe splitting.
    """

    if text is None:
        return ""

    value = str(text)
    value = value.replace("\r\n", "\n").replace("\r", "\n")

    value = re.sub(r"\*\*(.*?)\*\*", r"\1", value)
    value = re.sub(r"__(.*?)__", r"\1", value)

    value = re.sub(
        r"(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)",
        r"\1",
        value,
    )

    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)

    return value.strip()


#------------------------ VALUE CHECKS -------------------------

def is_empty_value(value: Any) -> bool:
    """
    Return True when a value should be removed from compact dictionaries.

    Boolean false values and numeric zero values are retained.
    """

    if value is None:
        return True

    if isinstance(value, str) and not value.strip():
        return True

    if isinstance(value, (list, dict, tuple, set)) and not value:
        return True

    return False


#------------------------ DICTIONARY HELPERS -------------------------

def compact_dict(data: Mapping[str, Any] | None) -> Dict[str, Any]:
    """
    Return a shallow dictionary without empty values.
    """

    compacted: Dict[str, Any] = {}

    for key, value in dict(data or {}).items():
        if is_empty_value(value):
            continue

        compacted[key] = value

    return compacted


def copy_preserved_meta(source: Mapping[str, Any] | None) -> Dict[str, Any]:
    """
    Copy parser metadata into an enriched content block.

    Deep copies prevent later enrichment stages from accidentally mutating the
    original AST metadata.
    """

    if not source:
        return {}

    copied: Dict[str, Any] = {}

    for key in PRESERVED_META_KEYS:
        value = source.get(key)

        if is_empty_value(value):
            continue

        copied[key] = deepcopy(value)

    return copied


def compact_content_block(
    block: Mapping[str, Any] | None,
) -> Dict[str, Any] | None:
    """
    Compact one enriched content block.

    Components may be meaningful without visible text. Other block types must
    retain text to remain in the semantic pipeline.
    """

    compacted = compact_dict(block)
    block_type = str(compacted.get("type", "") or "").strip().lower()

    if block_type != "component" and not compacted.get("text"):
        return None

    return compacted


#------------------------LIST NORMALIZATION-------------------------

def clean_list_items(items: list[Any]) -> list[str]:
    """
    Normalize list-item text while preserving source order.
    """

    cleaned: list[str] = []

    for item in items or []:
        text = clean_text(item)

        if text:
            cleaned.append(text)

    return cleaned