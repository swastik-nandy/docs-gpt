# rag/ingestion/data_transformation/parser/adapters/markdown/enrichment/sections/factory.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.component_context import current_component_meta


OVERVIEW_HEADING = "Overview"


def new_section(
    heading: str,
    path: List[str],
    component_stack: List[Dict[str, Any]],
    synthetic: bool = False,
    synthetic_reason: str = "",
) -> Dict[str, Any]:
    """
    Create a section shell.
    """

    section: Dict[str, Any] = {
        "heading": heading or "",
        "path": path or [],
        "content": [],
        "blocks": [],
    }

    if synthetic:
        section["synthetic"] = True
        section["synthetic_reason"] = synthetic_reason

    if component_stack:
        section.update(current_component_meta(component_stack))

    return section


def new_overview_section(
    component_stack: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Create synthetic Overview section for content before first heading.
    """

    return new_section(
        heading=OVERVIEW_HEADING,
        path=[],
        component_stack=component_stack,
        synthetic=True,
        synthetic_reason="content_before_first_heading",
    )


def add_content_block(
    section: Dict[str, Any],
    block: Optional[Dict[str, Any]],
) -> None:
    """
    Add a valid content block to a section.
    """

    if not block:
        return

    section.setdefault("content", []).append(block)


def attach_section_component_meta(
    section: Dict[str, Any],
    component_stack: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Attach component metadata to a section if not already attached.
    """

    if not section:
        return section

    if component_stack and not section.get("component_stack"):
        section.update(current_component_meta(component_stack))

    return section