# rag/ingestion/data_transformation/parser/adapters/markdown/enrichment/lists.py

from __future__ import annotations

from typing import Any, Dict, List

from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.texts.common import clean_list_items


def _detect_ordered(items: List[str]) -> bool:
    """
    Generic ordered list detection.

    Handles:
    - 1. item
    - 1) item
    """

    if not items:
        return False

    hits = 0

    for item in items:
        value = str(item or "").lstrip()

        if len(value) > 2 and value[0].isdigit() and value[1] in {".", ")"}:
            hits += 1

    return hits >= max(1, len(items) // 2)


def _normalize_list_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize any list-like content block to canonical type=list.
    """

    raw_items = item.get("items", []) or []
    items = clean_list_items(raw_items, min_len=1)

    ordered = bool(item.get("ordered", False))

    if not ordered:
        ordered = _detect_ordered(items)

    normalized = {
        **item,
        "type": "list",
        "items": items,
        "ordered": ordered,
    }

    if item.get("raw_items"):
        normalized["raw_items"] = item.get("raw_items", [])

    return normalized


def enrich_lists(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize legacy list block shapes.

    Supported:
    - list
    - list_block
    - list_raw

    This module is intentionally conservative. The main new path is:
        texts.content_blocks.enrich_list()
    """

    new_content: List[Dict[str, Any]] = []

    for item in content or []:
        if not item:
            continue

        item_type = item.get("type")

        if item_type in {"list", "list_block", "list_raw"}:
            new_content.append(_normalize_list_item(item))
            continue

        new_content.append(item)

    return new_content