from __future__ import annotations

from typing import Any, Dict

from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.sections.callouts.normalize import (
    clean_text,
    is_known_callout_label,
)


# --------------- BLOCK PREDICATES --------------------------------

def is_paragraph(block: Dict[str, Any]) -> bool:
    return bool(block and block.get("type") == "paragraph")


def has_heavy_inline_structure(block: Dict[str, Any]) -> bool:
    """
    Avoid treating rich/code/image-heavy blocks as standalone callout labels.
    """

    return bool(
        block.get("inline_code")
        or block.get("images")
        or block.get("code_blocks")
    )


def is_plain_callout_label(block: Dict[str, Any]) -> bool:
    """
    Detect standalone label paragraphs.

    Examples:
        Note
        Warning:
        Important

    Label vocabulary comes from heuristics.toml.
    """

    if not is_paragraph(block):
        return False

    if has_heavy_inline_structure(block):
        return False

    text = clean_text(block.get("text", ""))

    if not text:
        return False

    if len(text.split()) > 4:
        return False

    return is_known_callout_label(text)


def is_mergeable_body_paragraph(block: Dict[str, Any]) -> bool:
    """
    Body paragraph that can be merged under a standalone callout label.
    """

    if not is_paragraph(block):
        return False

    if is_plain_callout_label(block):
        return False

    return bool(clean_text(block.get("text", "")))
