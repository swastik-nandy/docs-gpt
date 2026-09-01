# rag/ingestion/data_transformation/parser/adapters/markdown/enrichment/content_to_ir/table.py

from __future__ import annotations

from typing import Any, Dict, List

from rag.ingestion.data_transformation.parser.adapters.markdown.ir.block_builder import build_blocks_from_table_rows
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.component_context import merge_into_meta
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.content_to_ir.common import base_meta, compact_meta


def build_table_ir(
    item: Dict[str, Any],
    section: Dict[str, Any],
    item_index: int,
    heading: str,
    context: str,
    local_context: str,
    item_stack: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build row-level IR blocks from a generic_table content block.

    Preserves:
    - normalized headers
    - raw headers
    - table profile
    - shape/anomaly metadata
    - parser-level cell metadata
    """

    row_texts = item.get("row_text", []) or []

    if not row_texts:
        return []

    meta = merge_into_meta(
        compact_meta({
            **base_meta(item_index, "table", section, item=item),
            "headers": item.get("headers", []),
            "header_raw": item.get("header_raw", []),
            "table_profile": item.get("table_profile", {}),
            "shape": item.get("shape", {}),
            "anomalies": item.get("anomalies", []),
            "cell_meta": item.get("cell_meta", []),
            "raw_rows": item.get("raw_rows", []),
        }),
        item_stack,
    )

    return build_blocks_from_table_rows(
        row_texts=row_texts,
        heading=heading,
        context=context,
        local_context=local_context,
        meta=meta,
    )