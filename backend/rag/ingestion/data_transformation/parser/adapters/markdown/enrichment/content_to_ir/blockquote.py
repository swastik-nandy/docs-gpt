# rag/ingestion/data_transformation/parser/adapters/markdown/enrichment/ir_builder/blockquote.py

from __future__ import annotations

from typing import Any, Dict, List

from rag.ingestion.data_transformation.parser.adapters.markdown.ir.block_builder import build_block
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.component_context import merge_into_meta
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.content_to_ir.common import base_meta, compact_meta


def build_blockquote_ir(
    item: Dict[str, Any],
    section: Dict[str, Any],
    item_index: int,
    heading: str,
    context: str,
    local_context: str,
    item_stack: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    """
    Build one IR block from a blockquote/callout content block.
    """

    text = (item.get("text") or "").strip()

    if not text:
        return None

    label = (item.get("label") or "").strip()
    kind = (item.get("kind") or "").strip()

    display_text = f"{label}: {text}" if label else text
    form = "callout" if label or kind else "blockquote"

    meta = merge_into_meta(
        compact_meta({
            **base_meta(item_index, "blockquote", section, item=item),
            "kind": kind,
            "label": label,
            "links": item.get("links", []),
            "images": item.get("images", []),
            "inline_code": item.get("inline_code", []),
            "html_inline": item.get("html_inline", []),
            "html_block": item.get("html_block", []),
            "strong_text": item.get("strong_text", []),
            "code_blocks": item.get("code_blocks", []),
        }),
        item_stack,
    )

    return build_block(
        base_type="text",
        form=form,
        text=display_text,
        heading=heading,
        context=context,
        local_context=local_context,
        meta=meta,
    )