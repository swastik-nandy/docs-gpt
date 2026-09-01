from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.ingestion.data_transformation.parser.adapters.markdown.execution.schema import (
    MarkdownParsedUnit,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.text_pipeline import (
    collect_ir_blocks_from_sections,
    parse_markdown_text_to_sections,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.io import load_markdown
from rag.ingestion.data_transformation.parser.adapters.markdown.ir.block_builder import refresh_retrieval_text
from rag.ingestion.data_transformation.parser.adapters.markdown.preprocessing import (
    extract_document_prolog,
)


FULL_DOCUMENT_UNIT_ID = "full"


# --------------- DOCUMENT META --------------------------------

def _attach_document_meta_to_sections(
    sections: list[dict[str, Any]],
    document_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    if not document_meta:
        return sections

    for section in sections or []:
        if not isinstance(section, dict):
            continue

        section["document_meta"] = document_meta

        for block in section.get("blocks", []) or []:
            if not isinstance(block, dict):
                continue

            meta = block.setdefault("meta", {})
            meta["document_meta"] = document_meta

            for key, value in document_meta.items():
                meta.setdefault(key, value)

            refresh_retrieval_text(block)

    return sections


def _build_full_document_meta(
    *,
    file_path: str,
    doc_id: str,
    variant: str,
    parse_strategy: str,
    sections: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    document_meta: dict[str, Any],
    raw_prolog_comments: list[str],
) -> dict[str, Any]:
    resolved_path = str(Path(file_path).resolve())

    return {
        "unit_kind": "full_document",
        "source_path": resolved_path,
        "parser_variant": variant,
        "parse_strategy": parse_strategy,
        "doc_id": doc_id,
        "source_doc_id": doc_id,
        "section_count": len(sections),
        "block_count": len(blocks),
        "document_meta": document_meta,
        "raw_prolog_comments": raw_prolog_comments,
    }


# --------------- FULL DOCUMENT PARSE --------------------------------

def parse_full_markdown_document(
    *,
    file_path: str,
    doc_id: str,
    variant: str,
    parse_strategy: str,
) -> MarkdownParsedUnit:
    resolved_path = str(Path(file_path).resolve())

    raw_text = load_markdown(resolved_path)
    prolog = extract_document_prolog(raw_text)

    sections = parse_markdown_text_to_sections(
        prolog.text,
        variant=variant,
        doc_id=doc_id,
    )

    if not sections:
        raise ValueError(
            f"Full-document Markdown parse produced no sections: {resolved_path}"
        )

    sections = _attach_document_meta_to_sections(
        sections=sections,
        document_meta=prolog.metadata,
    )

    blocks = collect_ir_blocks_from_sections(sections)

    if not blocks:
        raise ValueError(
            f"Full-document Markdown parse produced no IR blocks: {resolved_path}"
        )

    return MarkdownParsedUnit(
        unit_id=FULL_DOCUMENT_UNIT_ID,
        doc_id=doc_id,
        source_doc_id=doc_id,
        parse_strategy=parse_strategy,
        sections=sections,
        blocks=blocks,
        meta=_build_full_document_meta(
            file_path=resolved_path,
            doc_id=doc_id,
            variant=variant,
            parse_strategy=parse_strategy,
            sections=sections,
            blocks=blocks,
            document_meta=prolog.metadata,
            raw_prolog_comments=prolog.raw_comments,
        ),
        warnings=[],
    )