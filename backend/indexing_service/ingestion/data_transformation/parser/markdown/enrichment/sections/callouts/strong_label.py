from __future__ import annotations

from typing import Any, Dict

from ingestion.data_transformation.parser.markdown.enrichment.sections.callouts.builders import (
    build_callout_block,
)
from ingestion.data_transformation.parser.markdown.enrichment.sections.callouts.normalize import (
    clean_text,
    display_label,
    known_callout_kind,
    remove_label_prefix,
)
from ingestion.data_transformation.parser.markdown.enrichment.sections.callouts.predicates import (
    is_paragraph,
)


# --------------- STRONG LABEL DETECTION --------------------------------

def first_strong_text(block: Dict[str, Any]) -> str:
    values = block.get("strong_text", []) or []

    if not isinstance(values, list):
        return ""

    for value in values:
        text = clean_text(value)

        if text:
            return text

    return ""


def split_strong_label_callout(
    block: Dict[str, Any],
) -> Dict[str, Any] | None:
    if not is_paragraph(block):
        return None

    label = first_strong_text(block)

    if not label:
        return None

    kind = known_callout_kind(label)

    if not kind:
        return None

    full_text = clean_text(block.get("text", ""))
    body = remove_label_prefix(full_text, label)

    if not body:
        return None

    if len(body.split()) < 3:
        return None

    return build_callout_block(
        label=display_label(label),
        kind=kind,
        body_text=body,
        merged_from="strong_label_paragraph",
        label_block=block,
        body_blocks=[],
    )
