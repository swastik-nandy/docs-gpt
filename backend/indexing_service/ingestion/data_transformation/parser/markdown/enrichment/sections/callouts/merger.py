from __future__ import annotations

from typing import Any, Dict, List

from ingestion.data_transformation.parser.markdown.enrichment.sections.callouts.builders import (
    build_label_plus_body_callout,
)
from ingestion.data_transformation.parser.markdown.enrichment.sections.callouts.config import (
    MAX_CALLOUT_PARAGRAPHS,
)
from ingestion.data_transformation.parser.markdown.enrichment.sections.callouts.predicates import (
    is_mergeable_body_paragraph,
    is_plain_callout_label,
)
from ingestion.data_transformation.parser.markdown.enrichment.sections.callouts.prefix_label import (
    split_prefixed_label_callout,
)
from ingestion.data_transformation.parser.markdown.enrichment.sections.callouts.strong_label import (
    split_strong_label_callout,
)


# --------------- PUBLIC API --------------------------------

def merge_plain_callouts(
    content: List[Dict[str, Any]],
    max_body_paragraphs: int = MAX_CALLOUT_PARAGRAPHS,
) -> List[Dict[str, Any]]:
    if not content:
        return []

    merged: List[Dict[str, Any]] = []
    index = 0

    while index < len(content):
        block = content[index]

        strong_label_callout = split_strong_label_callout(block)

        if strong_label_callout:
            merged.append(strong_label_callout)
            index += 1
            continue

        prefixed_label_callout = split_prefixed_label_callout(block)

        if prefixed_label_callout:
            merged.append(prefixed_label_callout)
            index += 1
            continue

        if not is_plain_callout_label(block):
            merged.append(block)
            index += 1
            continue

        body_blocks: List[Dict[str, Any]] = []
        cursor = index + 1

        while cursor < len(content) and len(body_blocks) < max_body_paragraphs:
            candidate = content[cursor]

            if not is_mergeable_body_paragraph(candidate):
                break

            body_blocks.append(candidate)
            cursor += 1

        if not body_blocks:
            merged.append(block)
            index += 1
            continue

        merged.append(
            build_label_plus_body_callout(
                label_block=block,
                body_blocks=body_blocks,
            )
        )

        index = cursor

    return merged
