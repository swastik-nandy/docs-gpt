# rag/ingestion/data_transformation/parser/adapters/markdown/enrichment/content_to_ir/paragraph.py

from __future__ import annotations

from typing import Any, Dict, List

from ingestion.data_transformation.parser.markdown.ir.block_builder import build_block
from ingestion.data_transformation.parser.markdown.enrichment.content_to_ir.common import base_meta, compact_meta


def build_paragraph_ir(
    item: Dict[str, Any],
    section: Dict[str, Any],
    item_index: int,
    heading: str,
    context: str,
    local_context: str,
) -> Dict[str, Any] | None:
    """
    Build one IR block from a paragraph content block.
    """

    text = (item.get("text") or "").strip()

    if not text:
        return None

    meta = compact_meta({
            **base_meta(item_index, "paragraph", section, item=item),
            "links": item.get("links", []),
            "images": item.get("images", []),
            "inline_code": item.get("inline_code", []),
            "html_inline": item.get("html_inline", []),
            "strong_text": item.get("strong_text", []),
        })

    return build_block(
        base_type="text",
        form="paragraph",
        text=text,
        heading=heading,
        context=context,
        local_context=local_context,
        meta=meta,
    )