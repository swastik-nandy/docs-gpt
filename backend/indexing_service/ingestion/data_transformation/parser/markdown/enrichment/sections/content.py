from __future__ import annotations

from typing import Any

from ingestion.data_transformation.parser.markdown.enrichment.tables import (
    parse_table,
)
from ingestion.data_transformation.parser.markdown.enrichment.texts import (
    enrich_blockquote,
    enrich_code,
    enrich_list,
    enrich_paragraph,
)
from ingestion.data_transformation.parser.markdown.enrichment.texts.common import (
    copy_preserved_meta,
)


def build_content_block(
    element: dict[str, Any],
) -> dict[str, Any] | None:
    if not element:
        return None

    element_type = str(
        element.get(
            "type",
            "",
        )
        or ""
    ).strip().lower()

    if element_type == "table":
        block = parse_table(
            element.get(
                "rows",
                [],
            )
        )

        if block:
            block.update(
                copy_preserved_meta(
                    element
                )
            )

        return block

    if element_type == "list":
        return enrich_list(
            {
                "type": "list",
                "items": element.get(
                    "items",
                    [],
                ),
                "ordered": element.get(
                    "ordered",
                    False,
                ),
                **copy_preserved_meta(
                    element
                ),
            }
        )

    if element_type == "code":
        return enrich_code(
            element
        )

    if element_type == "blockquote":
        return enrich_blockquote(
            element
        )

    if element_type == "paragraph":
        return enrich_paragraph(
            element
        )

    text = str(
        element.get(
            "text",
            "",
        )
        or ""
    ).strip()

    if not text:
        return None

    return enrich_paragraph(
        {
            "type": "paragraph",
            "text": text,
            **copy_preserved_meta(
                element
            ),
        }
    )
