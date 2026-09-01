from __future__ import annotations

from typing import Any

from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.component_context import effective_local_context
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.component_context import stack_from_content_or_section
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.content_to_ir.blockquote import build_blockquote_ir
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.content_to_ir.code import build_code_ir
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.content_to_ir.common import build_section_id
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.content_to_ir.common import extend_blocks
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.content_to_ir.common import resolve_base_local_context
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.content_to_ir.component import build_component_ir
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.content_to_ir.fallback import build_fallback_ir
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.content_to_ir.list import build_list_ir
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.content_to_ir.paragraph import build_paragraph_ir
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.content_to_ir.table import build_table_ir
from rag.ingestion.data_transformation.parser.adapters.markdown.ir.block_builder import assign_block_identity
from rag.ingestion.data_transformation.parser.adapters.markdown.ir.relationships import enrich_block_relationships


#--------------------HELPERS--------------------

def _single(
    block: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if block is None:
        return []

    return [
        block
    ]


def _resolve_item_context(
    item: dict[str, Any],
    section: dict[str, Any],
    base_local_context: str,
) -> tuple[
    list[dict[str, Any]],
    str,
]:
    item_stack = stack_from_content_or_section(
        item,
        section,
    )

    local_scope_title = str(
        item.get(
            "local_scope_title"
        )
        or ""
    ).strip()

    structural_context = (
        local_scope_title
        or base_local_context
    )

    local_context = effective_local_context(
        base_local_context=structural_context,
        stack=item_stack,
    )

    return (
        item_stack,
        local_context,
    )


#--------------------ITEM ROUTING--------------------

def _build_item_ir(
    item: dict[str, Any],
    section: dict[str, Any],
    item_index: int,
    heading: str,
    context: str,
    base_local_context: str,
) -> list[dict[str, Any]]:
    if not item:
        return []

    item_type = str(
        item.get(
            "type"
        )
        or ""
    ).strip().lower()

    item_stack, local_context = (
        _resolve_item_context(
            item=item,
            section=section,
            base_local_context=base_local_context,
        )
    )

    common_kwargs = {
        "item": item,
        "section": section,
        "item_index": item_index,
        "heading": heading,
        "context": context,
        "local_context": local_context,
        "item_stack": item_stack,
    }

    if item_type == "paragraph":
        return _single(
            build_paragraph_ir(
                **common_kwargs
            )
        )

    if item_type == "blockquote":
        return _single(
            build_blockquote_ir(
                **common_kwargs
            )
        )

    if item_type == "list":
        return build_list_ir(
            **common_kwargs
        )

    if item_type in {
        "generic_table",
        "table",
    }:
        return build_table_ir(
            **common_kwargs
        )

    if item_type == "code":
        return _single(
            build_code_ir(
                **common_kwargs
            )
        )

    if item_type == "component":
        return _single(
            build_component_ir(
                **common_kwargs
            )
        )

    return _single(
        build_fallback_ir(
            **common_kwargs
        )
    )


#--------------------IDENTITY--------------------

def _assign_block_ids(
    ir_blocks: list[dict[str, Any]],
    section: dict[str, Any],
    section_index: int,
    doc_id: str,
) -> None:
    section_id = build_section_id(
        section,
        section_index,
    )

    for block_index, block in enumerate(
        ir_blocks
    ):
        assign_block_identity(
            block=block,
            doc_id=doc_id,
            section_id=section_id,
            block_index=block_index,
        )


#--------------------PUBLIC API--------------------

def build_ir_blocks_for_section(
    section: dict[str, Any],
    section_index: int,
    doc_id: str = "doc",
) -> list[dict[str, Any]]:
    """
    Convert one finalized section into canonical IR.

    IR is built exactly once.

    Content-level semantic packing has already happened before this function.
    This layer performs only:
    content routing -> identity -> relationships.
    """
    content = section.get(
        "content",
        [],
    ) or []

    if not content:
        return []

    heading = str(
        section.get(
            "heading"
        )
        or ""
    ).strip()

    context = str(
        section.get(
            "context"
        )
        or ""
    ).strip()

    base_local_context = (
        resolve_base_local_context(
            section
        )
    )

    ir_blocks: list[
        dict[str, Any]
    ] = []

    for item_index, item in enumerate(
        content
    ):
        if not isinstance(
            item,
            dict,
        ):
            continue

        blocks = _build_item_ir(
            item=item,
            section=section,
            item_index=item_index,
            heading=heading,
            context=context,
            base_local_context=base_local_context,
        )

        extend_blocks(
            ir_blocks,
            blocks,
        )

    _assign_block_ids(
        ir_blocks=ir_blocks,
        section=section,
        section_index=section_index,
        doc_id=doc_id,
    )

    return enrich_block_relationships(
        ir_blocks
    )
