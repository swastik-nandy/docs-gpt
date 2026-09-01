# rag/ingestion/data_transformation/parser/adapters/markdown/enrichment/content_to_ir/component.py

from __future__ import annotations

from typing import Any, Dict, List

from rag.ingestion.data_transformation.parser.adapters.markdown.ir.block_builder import build_block
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.component_context import merge_into_meta
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.content_to_ir.common import base_meta, compact_meta


def build_component_ir(
    item: Dict[str, Any],
    section: Dict[str, Any],
    item_index: int,
    heading: str,
    context: str,
    local_context: str,
    item_stack: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    """
    Build one IR block from a self-closing MDX component content block.

    Example content block:
        {
            "type": "component",
            "name": "Card",
            "props": {"title": "API Reference"},
            "text": "Component: Card (title=API Reference)"
        }

    No component name is hardcoded.
    """

    text = (item.get("text") or "").strip()

    if not text:
        return None

    meta = merge_into_meta(
        compact_meta({
            **base_meta(item_index, "component", section, item=item),
            "component_name": item.get("name", ""),
            "component_props": item.get("props", {}),
        }),
        item_stack,
    )

    return build_block(
        base_type="component",
        form="block",
        text=text,
        heading=heading,
        context=context,
        local_context=local_context,
        meta=meta,
    )