from __future__ import annotations

from typing import Any

from rag.ingestion.data_transformation.parser.adapters.markdown.adapters import normalize_mdx
from rag.ingestion.data_transformation.parser.adapters.markdown.ast import parse_md_ast
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.enricher import enrich
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.timing import timed
from rag.ingestion.data_transformation.parser.adapters.markdown.preprocessing import clean_markdown


# --------------- TEXT TO SECTIONS --------------------------------

def parse_markdown_text_to_sections(
    text: str,
    *,
    variant: str,
    doc_id: str,
) -> list[dict[str, Any]]:
    """
    Parse already-loaded Markdown/MDX text into enriched sections.
    """

    with timed(
        "text_pipeline.total",
        variant=variant,
        doc_id=doc_id,
        input_chars=len(text or ""),
    ):
        if not text or not text.strip():
            return []

        raw_text = text

        if variant == "mdx":
            with timed("text_pipeline.mdx_normalize", doc_id=doc_id):
                raw_text = normalize_mdx(raw_text)

        with timed("text_pipeline.clean", doc_id=doc_id):
            cleaned_text = clean_markdown(raw_text)

        if not cleaned_text.strip():
            return []

        with timed("text_pipeline.ast", doc_id=doc_id):
            elements = parse_md_ast(cleaned_text)

        if not elements:
            return []

        with timed("text_pipeline.enrich", doc_id=doc_id):
            sections = enrich(
                elements,
                doc_id=doc_id,
            )

    return sections


# --------------- BLOCK COLLECTION --------------------------------

def collect_ir_blocks_from_sections(
    sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []

    for section in sections or []:
        if not isinstance(section, dict):
            continue

        section_blocks = section.get("blocks", []) or []

        for block in section_blocks:
            if isinstance(block, dict):
                blocks.append(block)

    return blocks


# --------------- CONTENT COLLECTION --------------------------------

def collect_content_blocks_from_sections(
    sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    content_blocks: list[dict[str, Any]] = []

    for section in sections or []:
        if not isinstance(section, dict):
            continue

        content = section.get("content", []) or []

        for block in content:
            if isinstance(block, dict):
                content_blocks.append(block)

    return content_blocks
