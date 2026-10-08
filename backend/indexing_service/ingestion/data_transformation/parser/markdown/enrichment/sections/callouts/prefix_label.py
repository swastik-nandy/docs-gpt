from __future__ import annotations

from typing import Any, Dict, List

from ingestion.data_transformation.parser.markdown.enrichment.sections.callouts.builders import (
    build_callout_block,
)
from ingestion.data_transformation.parser.markdown.enrichment.sections.callouts.config import (
    CALLOUT_KIND_MAP,
)
from ingestion.data_transformation.parser.markdown.enrichment.sections.callouts.normalize import (
    clean_text,
    display_label,
    known_callout_kind,
    remove_label_prefix,
)
from ingestion.data_transformation.parser.markdown.enrichment.sections.callouts.predicates import (
    has_heavy_inline_structure,
    is_paragraph,
)


# --------------- PREFIX LABEL DETECTION --------------------------------

def candidate_prefix_labels() -> List[str]:
    return sorted(
        CALLOUT_KIND_MAP.keys(),
        key=len,
        reverse=True,
    )


def has_safe_label_boundary(
    text: str,
    label: str,
) -> bool:
    value = clean_text(text)
    label_value = clean_text(label)

    if not value or not label_value:
        return False

    if not value.lower().startswith(label_value.lower()):
        return False

    body = value[len(label_value):]

    if not body:
        return False

    if body[0] in {":", "-", "–", "—"}:
        return True

    if body[0].isspace():
        body_text = body.strip()

        if not body_text:
            return False

        first_body_word = body_text.split()[0].lower().strip(".,:;!?")

        if label_value.lower() == "note" and first_body_word in {"that", "when", "if"}:
            return False

        return True

    return False


def split_prefixed_label_callout(
    block: Dict[str, Any],
) -> Dict[str, Any] | None:
    if not is_paragraph(block):
        return None

    if has_heavy_inline_structure(block):
        return None

    text = clean_text(block.get("text", ""))

    if not text:
        return None

    for label in candidate_prefix_labels():
        if not has_safe_label_boundary(text, label):
            continue

        kind = known_callout_kind(label)

        if not kind:
            continue

        body = remove_label_prefix(text, label)

        if not body:
            continue

        if len(body.split()) < 3:
            continue

        return build_callout_block(
            label=display_label(label),
            kind=kind,
            body_text=body,
            merged_from="prefixed_label_paragraph",
            label_block=block,
            body_blocks=[],
        )

    return None
