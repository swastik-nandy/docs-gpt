from __future__ import annotations

from typing import Any

from ingestion.data_transformation.parser.markdown.enrichment.sections.heading_stack import (
    extract_path,
    update_heading_stack,
)
from ingestion.data_transformation.parser.markdown.enrichment.sections.content import (
    build_content_block,
)
from ingestion.data_transformation.parser.markdown.enrichment.sections.factory import (
    add_content_block,
    new_overview_section,
    new_section,
)


def build_sections_from_elements(
    elements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []

    current_section: dict[str, Any] | None = None
    heading_stack: list[Any] = []

    def flush_section() -> None:
        nonlocal current_section

        if (
            current_section
            and current_section.get("content")
        ):
            sections.append(current_section)

        current_section = None

    def ensure_section() -> dict[str, Any]:
        nonlocal current_section

        if current_section is None:
            current_section = new_overview_section()

        return current_section

    for element in elements or []:
        if not element:
            continue

        element_type = str(
            element.get("type", "")
            or ""
        ).strip().lower()

        if element_type == "heading":
            flush_section()

            heading_stack = update_heading_stack(
                heading_stack,
                element.get("level"),
                element.get("text"),
            )

            current_section = new_section(
                heading=element.get(
                    "text",
                    "",
                ),
                path=extract_path(
                    heading_stack[:-1]
                ),
            )

            continue

        block = build_content_block(
            element
        )

        add_content_block(
            ensure_section(),
            block,
        )

    flush_section()

    return sections
