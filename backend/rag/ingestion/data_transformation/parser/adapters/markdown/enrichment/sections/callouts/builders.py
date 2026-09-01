from __future__ import annotations

from typing import Any, Dict, List

from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.sections.callouts.metadata import (
    copy_context_fields,
    copy_inline_metadata,
    merge_body_metadata,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.sections.callouts.normalize import (
    clean_text,
    display_label,
    known_callout_kind,
)


# --------------- CALLOUT BUILDERS --------------------------------

def build_callout_block(
    *,
    label: str,
    kind: str,
    body_text: str,
    merged_from: str,
    label_block: Dict[str, Any] | None = None,
    body_blocks: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    callout: Dict[str, Any] = {
        "type": "blockquote",
        "text": body_text,
        "label": label,
        "kind": kind,
        "plain_callout": True,
        "merged_from": merged_from,
    }

    if label_block:
        copy_context_fields(callout, label_block)
        copy_inline_metadata(callout, label_block)

    if body_blocks:
        merge_body_metadata(callout, body_blocks)

    return callout


def build_label_plus_body_callout(
    label_block: Dict[str, Any],
    body_blocks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    label = display_label(label_block.get("text", ""))
    kind = known_callout_kind(label) or "note"

    body_text = "\n\n".join(
        clean_text(block.get("text", ""))
        for block in body_blocks
        if clean_text(block.get("text", ""))
    ).strip()

    return build_callout_block(
        label=label,
        kind=kind,
        body_text=body_text,
        merged_from="paragraph_label",
        label_block=label_block,
        body_blocks=body_blocks,
    )
