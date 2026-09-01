# rag/ingestion/data_transformation/parser/adapters/markdown/enrichment/content_to_ir/fallback.py

from __future__ import annotations

from typing import Any, Dict, List

from rag.ingestion.data_transformation.parser.adapters.markdown.ir.block_builder import build_block
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.component_context import merge_into_meta
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.content_to_ir.common import base_meta, compact_meta


def build_fallback_ir(
    item: Dict[str, Any],
    section: Dict[str, Any],
    item_index: int,
    heading: str,
    context: str,
    local_context: str,
    item_stack: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    """
    Build a safe fallback IR block for unknown-but-textual content blocks.

    This keeps the pipeline loss-aware:
    if a content block has useful text but no dedicated builder yet,
    we still preserve it instead of silently dropping it.
    """

    text = (item.get("text") or "").strip()

    if not text:
        return None

    item_type = item.get("type") or "unknown"

    meta = merge_into_meta(
        compact_meta({
            **base_meta(item_index, item_type, section, item=item),
            "fallback": True,
            "original_type": item_type,
        }),
        item_stack,
    )

    return build_block(
        base_type="text",
        form="block",
        text=text,
        heading=heading,
        context=context,
        local_context=local_context,
        meta=meta,
    )