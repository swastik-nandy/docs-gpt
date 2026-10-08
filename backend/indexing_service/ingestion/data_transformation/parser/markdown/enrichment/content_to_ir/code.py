# rag/ingestion/data_transformation/parser/adapters/markdown/enrichment/content_to_ir/code.py

from __future__ import annotations

from typing import Any, Dict, List

from ingestion.data_transformation.parser.markdown.ir.block_builder import build_block
from ingestion.data_transformation.parser.markdown.enrichment.content_to_ir.common import base_meta, compact_meta


def build_code_ir(
    item: Dict[str, Any],
    section: Dict[str, Any],
    item_index: int,
    heading: str,
    context: str,
    local_context: str,
) -> Dict[str, Any] | None:
    """
    Build one IR block from a code content block.

    Preserves:
    - language
    - source position metadata
    - component context
    """

    text = (item.get("text") or "").strip()

    if not text:
        return None

    language = (item.get("language") or "").strip()

    meta = compact_meta({
            **base_meta(item_index, "code", section, item=item),
            "language": language,
        })

    return build_block(
        base_type="code",
        form="block",
        text=text,
        heading=heading,
        context=context,
        local_context=local_context,
        meta=meta,
    )