# rag/ingestion/data_transformation/parser/adapters/markdown/enrichment/texts/content_blocks.py

from __future__ import annotations

from typing import Any, Dict, Mapping

from ingestion.data_transformation.parser.markdown.enrichment.texts.common import (
    clean_list_items,
    clean_text,
    compact_content_block,
    copy_preserved_meta,
)


def enrich_paragraph(value: Any) -> Dict[str, Any] | None:
    """
    Convert paragraph AST into a normalized content block.

    Accepts:
    - raw string
    - AST dict
    """

    if isinstance(value, Mapping):
        text = clean_text(value.get("text", ""))
        source = value
    else:
        text = clean_text(value)
        source = {}

    if not text or len(text) < 3:
        return None

    return compact_content_block({
        "type": "paragraph",
        "text": text,
        **copy_preserved_meta(source),
    })


def enrich_blockquote(value: Mapping[str, Any]) -> Dict[str, Any] | None:
    """
    Convert blockquote/callout AST into a normalized content block.

    Generic Markdown docs examples:
    - Note
    - Tip
    - Warning
    - Important
    - Good to know
    """

    if not value:
        return None

    text = clean_text(value.get("text", ""))

    if not text or len(text) < 2:
        return None

    label = clean_text(value.get("label", ""))
    kind = clean_text(value.get("kind", "")) or "quote"

    return compact_content_block({
        "type": "blockquote",
        "text": text,
        "label": label,
        "kind": kind,
        **copy_preserved_meta(value),
    })


def enrich_code(value: Any, language: str = "") -> Dict[str, Any] | None:
    """
    Convert code AST into a normalized content block.

    Accepts:
    - raw code text + language
    - AST dict
    """

    if isinstance(value, Mapping):
        code = str(value.get("text", "") or "").strip()
        language = str(value.get("language", language) or "").strip()
        source = value
    else:
        code = str(value or "").strip()
        source = {}

    if not code:
        return None

    return compact_content_block({
        "type": "code",
        "language": language,
        "text": code,
        **copy_preserved_meta(source),
    })


def enrich_list(block: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize list AST/content blocks safely.

    Keeps:
    - items
    - ordered flag
    - raw_items metadata when present
    """

    if not block:
        return block

    if block.get("type") != "list":
        return block

    enriched = dict(block)
    enriched["items"] = clean_list_items(enriched.get("items", []))
    enriched["ordered"] = bool(enriched.get("ordered", False))

    return enriched