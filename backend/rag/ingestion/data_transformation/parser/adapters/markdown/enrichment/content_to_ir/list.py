from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List

from rag.ingestion.data_transformation.parser.adapters.markdown.ir.block_builder import build_block
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.component_context import merge_into_meta
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.content_to_ir.common import (
    base_meta,
    compact_meta,
)


# --------------- CONSTANTS --------------------------------

ITEM_META_KEYS = (
    "links",
    "images",
    "inline_code",
    "html_inline",
    "html_block",
    "strong_text",
)

FENCED_CODE_RE = re.compile(
    r"""
    (?:
        ```[\s\S]*?```
        |
        ~~~[\s\S]*?~~~
    )
    """,
    re.VERBOSE,
)


# --------------- NORMALIZATION --------------------------------

def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normalize_raw_items(raw_items: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_items, list):
        return []

    return [
        item
        for item in raw_items
        if isinstance(item, dict)
    ]


def _raw_item_for_index(
    raw_items: List[Dict[str, Any]],
    index: int,
) -> Dict[str, Any]:
    if index < 0 or index >= len(raw_items):
        return {}

    return raw_items[index] or {}


def _strip_fenced_code_from_text(text: str) -> str:
    cleaned = FENCED_CODE_RE.sub("", text or "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()

    return cleaned


def _resolve_item_text(
    item_text: Any,
    raw_item: Dict[str, Any],
) -> str:
    raw_text = str(raw_item.get("text") or item_text or "").strip()

    if not raw_text:
        return ""

    stripped = _strip_fenced_code_from_text(raw_text)

    return stripped or raw_text


def _normalize_code_blocks(raw_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    code_blocks = raw_item.get("code_blocks", []) or []

    if not isinstance(code_blocks, list):
        return []

    return [
        block
        for block in code_blocks
        if isinstance(block, dict) and str(block.get("text") or "").strip()
    ]


def _copy_item_metadata(raw_item: Dict[str, Any]) -> Dict[str, Any]:
    copied: Dict[str, Any] = {}

    for key in ITEM_META_KEYS:
        value = raw_item.get(key)

        if value:
            copied[key] = deepcopy(value)

    children = raw_item.get("children")

    if children:
        copied["children"] = deepcopy(children)
        copied["has_nested_children"] = True

    code_blocks = _normalize_code_blocks(raw_item)

    if code_blocks:
        copied["code_blocks"] = deepcopy(code_blocks)
        copied["has_nested_code_blocks"] = True

    return copied


# --------------- META BUILDERS --------------------------------

def _base_list_meta(
    *,
    item: Dict[str, Any],
    section: Dict[str, Any],
    item_index: int,
    ordered: bool,
    item_count: int,
    raw_items: List[Dict[str, Any]],
    item_stack: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return merge_into_meta(
        compact_meta({
            **base_meta(item_index, "list", section, item=item),
            "ordered": ordered,
            "raw_items": raw_items,
            "item_count": item_count,
        }),
        item_stack,
    )


def _list_item_meta(
    *,
    base: Dict[str, Any],
    raw_item: Dict[str, Any],
    item_index: int,
    ordered: bool,
) -> Dict[str, Any]:
    return compact_meta({
        **base,
        **_copy_item_metadata(raw_item),
        "ordered": ordered,
        "item_index": item_index,
        "source_index": item_index,
        "source_type": "list",
    })


def _nested_code_meta(
    *,
    base: Dict[str, Any],
    raw_item: Dict[str, Any],
    code_block: Dict[str, Any],
    item_index: int,
    code_index: int,
    ordered: bool,
    parent_text: str,
) -> Dict[str, Any]:
    return compact_meta({
        **base,
        "source_type": "list_code",
        "source_index": item_index,
        "ordered": ordered,
        "item_index": item_index,
        "code_index": code_index,
        "language": str(code_block.get("language") or "").strip(),
        "parent_instruction": parent_text,
        "parent_list_item_text": parent_text,
        "parent_list_item_index": item_index,
        "links": raw_item.get("links", []),
        "images": raw_item.get("images", []),
        "inline_code": raw_item.get("inline_code", []),
        "html_inline": raw_item.get("html_inline", []),
        "html_block": raw_item.get("html_block", []),
        "strong_text": raw_item.get("strong_text", []),
    })


# --------------- BLOCK BUILDERS --------------------------------

def _build_list_item_block(
    *,
    text: str,
    meta: Dict[str, Any],
    heading: str,
    context: str,
    local_context: str,
) -> Dict[str, Any] | None:
    return build_block(
        base_type="list",
        form="item",
        text=text,
        heading=heading,
        context=context,
        local_context=local_context,
        meta=meta,
    )


def _build_nested_code_block(
    *,
    code_block: Dict[str, Any],
    meta: Dict[str, Any],
    heading: str,
    context: str,
    local_context: str,
) -> Dict[str, Any] | None:
    code_text = str(code_block.get("text") or "").strip()

    if not code_text:
        return None

    return build_block(
        base_type="code",
        form="block",
        text=code_text,
        heading=heading,
        context=context,
        local_context=local_context,
        meta=meta,
    )


# --------------- PUBLIC API --------------------------------

def build_list_ir(
    item: Dict[str, Any],
    section: Dict[str, Any],
    item_index: int,
    heading: str,
    context: str,
    local_context: str,
    item_stack: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build retrieval-ready IR from a list content block.

    This emits:
    - one list/item block per list item
    - one code/block IR block for each nested code block inside a list item

    This preserves list evidence for chunking:
    - links
    - inline code
    - nested children
    - nested code blocks
    - parent instruction/context for code
    """

    items = item.get("items", []) or []

    if not items:
        return []

    ordered = bool(item.get("ordered", False))
    raw_items = _normalize_raw_items(item.get("raw_items", []))
    item_count = len(items)

    base = _base_list_meta(
        item=item,
        section=section,
        item_index=item_index,
        ordered=ordered,
        item_count=item_count,
        raw_items=raw_items,
        item_stack=item_stack,
    )

    blocks: List[Dict[str, Any]] = []

    for list_item_index, list_item_text in enumerate(items):
        raw_item = _raw_item_for_index(raw_items, list_item_index)
        parent_text = _resolve_item_text(list_item_text, raw_item)

        if not parent_text:
            continue

        list_item_block = _build_list_item_block(
            text=parent_text,
            meta=_list_item_meta(
                base=base,
                raw_item=raw_item,
                item_index=list_item_index,
                ordered=ordered,
            ),
            heading=heading,
            context=context,
            local_context=local_context,
        )

        if list_item_block:
            blocks.append(list_item_block)

        for code_index, code_block in enumerate(_normalize_code_blocks(raw_item)):
            nested_code_block = _build_nested_code_block(
                code_block=code_block,
                meta=_nested_code_meta(
                    base=base,
                    raw_item=raw_item,
                    code_block=code_block,
                    item_index=list_item_index,
                    code_index=code_index,
                    ordered=ordered,
                    parent_text=parent_text,
                ),
                heading=heading,
                context=context,
                local_context=local_context,
            )

            if nested_code_block:
                blocks.append(nested_code_block)

    return blocks
