# rag/ingestion/data_transformation/parser/adapters/markdown/enrichment/enricher.py

from __future__ import annotations

from typing import Any, Dict, List

from ingestion.data_transformation.parser.markdown.enrichment.sections import (
    build_sections_from_elements,
    finalize_sections,
)


def enrich(
    elements: List[Dict[str, Any]],
    doc_id: str = "doc",
) -> List[Dict[str, Any]]:
    """
    Markdown enrichment controller.

    Responsibilities:
    - AST events → sections
    - sections → final enriched sections with IR blocks

    All heavy logic lives in focused modules.
    """

    sections = build_sections_from_elements(elements)
    return finalize_sections(sections, doc_id=doc_id)