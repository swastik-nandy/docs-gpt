# rag/ingestion/data_transformation/parser/adapters/markdown/enrichment/sections/builder.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.sections.heading_stack import (
    extract_path,
    update_heading_stack,
)

from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.component_context import (
    current_component_meta,
    pop_component,
    push_component,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.sections.content import build_content_block
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.sections.factory import (
    add_content_block,
    attach_section_component_meta,
    new_overview_section,
    new_section,
)


def build_sections_from_elements(
    elements: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert AST events into heading-based sections.

    Responsibilities:
    - maintain heading hierarchy
    - maintain active MDX component stack
    - preserve content before first heading as Overview
    - route AST content elements into content blocks
    """

    sections: List[Dict[str, Any]] = []
    current_section: Optional[Dict[str, Any]] = None
    heading_stack: List[Any] = []
    component_stack: List[Dict[str, Any]] = []

    def flush_section() -> None:
        nonlocal current_section

        if current_section and current_section.get("content"):
            sections.append(current_section)

        current_section = None

    def ensure_section() -> Dict[str, Any]:
        nonlocal current_section

        if current_section is None:
            current_section = new_overview_section(component_stack)

        attach_section_component_meta(current_section, component_stack)

        return current_section

    for element in elements or []:
        if not element:
            continue

        element_type = element.get("type")

        # ---------------- MDX COMPONENT OPEN ---------------- #

        if element_type == "component_open":
            push_component(
                component_stack,
                name=element.get("name", ""),
                props=element.get("props", {}) or {},
            )

            if current_section and not current_section.get("component_stack"):
                current_section.update(current_component_meta(component_stack))

            continue

        # ---------------- MDX COMPONENT CLOSE ---------------- #

        if element_type == "component_close":
            pop_component(
                component_stack,
                name=element.get("name", ""),
            )
            continue

        # ---------------- HEADING ---------------- #

        if element_type == "heading":
            flush_section()

            heading_stack = update_heading_stack(
                heading_stack,
                element.get("level"),
                element.get("text"),
            )

            current_section = new_section(
                heading=element.get("text", ""),
                path=extract_path(heading_stack[:-1]),
                component_stack=component_stack,
            )
            continue

        # ---------------- CONTENT ELEMENT ---------------- #

        block = build_content_block(
            element=element,
            component_stack=component_stack,
        )

        add_content_block(ensure_section(), block)

    flush_section()

    return sections