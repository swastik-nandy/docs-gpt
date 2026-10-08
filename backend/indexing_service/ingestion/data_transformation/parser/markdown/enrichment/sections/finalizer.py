from __future__ import annotations

from typing import Any

from ingestion.data_transformation.parser.markdown.enrichment.content_to_ir import build_ir_blocks_for_section
from ingestion.data_transformation.parser.markdown.enrichment.content_to_ir import build_section_id
from ingestion.data_transformation.parser.markdown.enrichment.metadata import enrich_metadata
from ingestion.data_transformation.parser.markdown.enrichment.sections.callout_merger import merge_plain_callouts
from ingestion.data_transformation.parser.markdown.enrichment.sections.semantic_packing import pack_semantic_paragraphs
from ingestion.data_transformation.parser.markdown.enrichment.sections.structural_artifacts import attach_anchors_and_remove_noise_blocks
from ingestion.data_transformation.parser.markdown.enrichment.tables import enrich_tables
from ingestion.data_transformation.parser.markdown.enrichment.texts import build_section_text


#--------------------CONTENT PREPARATION--------------------

def _prepare_content(
    content: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not content:
        return []

    prepared = merge_plain_callouts(
        content
    )

    prepared = enrich_tables(
        prepared
    )

    prepared = pack_semantic_paragraphs(
        prepared
    )

    return prepared


#--------------------SECTION FINALIZATION--------------------

def _finalize_one_section(
    section: dict[str, Any],
    section_index: int,
    doc_id: str,
) -> dict[str, Any] | None:
    section = attach_anchors_and_remove_noise_blocks(
        section
    )

    content = section.get(
        "content",
        [],
    ) or []

    if not content:
        return None

    content = _prepare_content(
        content
    )

    if not content:
        return None

    section[
        "content"
    ] = content

    section_text = build_section_text(
        content
    )

    if not section_text:
        return None

    section[
        "text"
    ] = section_text

    section = enrich_metadata(
        section
    )

    section[
        "section_id"
    ] = build_section_id(
        section,
        section_index,
    )

    section[
        "blocks"
    ] = build_ir_blocks_for_section(
        section=section,
        section_index=section_index,
        doc_id=doc_id,
    )

    section[
        "block_count"
    ] = len(
        section.get(
            "blocks",
            [],
        )
    )

    return section


#--------------------PUBLIC API--------------------

def finalize_sections(
    sections: list[dict[str, Any]],
    doc_id: str = "doc",
) -> list[dict[str, Any]]:
    output: list[
        dict[str, Any]
    ] = []

    for section_index, section in enumerate(
        sections
        or []
    ):
        if not isinstance(
            section,
            dict,
        ):
            continue

        finalized = _finalize_one_section(
            section=section,
            section_index=section_index,
            doc_id=doc_id,
        )

        if finalized is not None:
            output.append(
                finalized
            )

    return output
