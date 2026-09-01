# rag/ingestion/data_transformation/parser/adapters/markdown/enrichment/sections/content.py

from __future__ import annotations

from typing import Any

from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.component_context import attach_to_content_block
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.component_context import build_self_closing_component_block
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.tables import parse_table
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.texts import enrich_blockquote
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.texts import enrich_code
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.texts import enrich_list
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.texts import enrich_paragraph
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.texts.common import copy_preserved_meta


#------------------------component context-------------------------

def _attach(
    block: dict[str, Any] | None,
    component_stack: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not block:
        return None

    return attach_to_content_block(block, component_stack)


#------------------------table blocks-------------------------

def build_table_content_block(
    element: dict[str, Any],
    component_stack: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Convert one table AST element into a structured content block.

    Parser metadata such as source provenance, links, images and cell metadata
    remains attached to the structured table.
    """

    structured = parse_table(element.get("rows", []))

    if not structured:
        return None

    structured.update(copy_preserved_meta(element))

    return _attach(structured, component_stack)


#------------------------list blocks-------------------------

def build_list_content_block(
    element: dict[str, Any],
    component_stack: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Convert one list AST element into a normalized content block.
    """

    block_input: dict[str, Any] = {
        "type": "list",
        "items": element.get("items", []),
        "ordered": element.get("ordered", False),
        **copy_preserved_meta(element),
    }

    block = enrich_list(block_input)

    return _attach(block, component_stack)


#------------------------code blocks-------------------------

def build_code_content_block(
    element: dict[str, Any],
    component_stack: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Convert one code AST element into a normalized content block.
    """

    block = enrich_code(element)

    return _attach(block, component_stack)


#------------------------paragraph blocks-------------------------

def build_paragraph_content_block(
    element: dict[str, Any],
    component_stack: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Convert one paragraph AST element into a normalized content block.
    """

    block = enrich_paragraph(element)

    return _attach(block, component_stack)


#------------------------blockquote blocks-------------------------

def build_blockquote_content_block(
    element: dict[str, Any],
    component_stack: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Convert one blockquote or callout AST element into a content block.
    """

    block = enrich_blockquote(element)

    return _attach(block, component_stack)


#------------------------component blocks-------------------------

def build_self_component_content_block(
    element: dict[str, Any],
    component_stack: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Convert one self-closing MDX component into a content block.
    """

    block = build_self_closing_component_block(
        name=element.get("name", ""),
        props=element.get("props", {}) or {},
        stack=component_stack,
    )

    if not block:
        return None

    block.update(copy_preserved_meta(element))

    return block


#------------------------fallback blocks-------------------------

def build_fallback_content_block(
    element: dict[str, Any],
    component_stack: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Preserve an unknown textual AST element as a paragraph block.
    """

    text = str(element.get("text", "") or "").strip()

    if not text:
        return None

    paragraph_input: dict[str, Any] = {
        "type": "paragraph",
        "text": text,
        **copy_preserved_meta(element),
    }

    block = enrich_paragraph(paragraph_input)

    return _attach(block, component_stack)


#------------------------content routing-------------------------

def build_content_block(
    element: dict[str, Any],
    component_stack: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Route one AST element to its content-block builder.
    """

    if not element:
        return None

    element_type = str(element.get("type", "") or "").strip().lower()

    if element_type == "table":
        return build_table_content_block(element, component_stack)

    if element_type == "list":
        return build_list_content_block(element, component_stack)

    if element_type == "code":
        return build_code_content_block(element, component_stack)

    if element_type == "blockquote":
        return build_blockquote_content_block(element, component_stack)

    if element_type == "paragraph":
        return build_paragraph_content_block(element, component_stack)

    if element_type == "component_self":
        return build_self_component_content_block(
            element,
            component_stack,
        )

    return build_fallback_content_block(element, component_stack)