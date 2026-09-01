# rag/ingestion/data_transformation/parser/adapters/markdown/enrichment/component_context/blocks.py

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.component_context.format import (
    current_component_meta,
    format_props,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.component_context.normalize import (
    clean_component_name,
    clean_props,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.component_context.types import (
    ComponentStack,
)


def attach_to_content_block(
    block: Dict[str, Any] | None,
    stack: ComponentStack | None,
) -> Dict[str, Any] | None:
    """
    Attach active component metadata to a parser content block.

    Content blocks include:
    - paragraph
    - blockquote
    - list
    - code
    - table
    - self-closing component
    """

    if not block:
        return block

    if not stack:
        return block

    meta = current_component_meta(stack)

    if not meta:
        return block

    block.setdefault("meta", {})
    block["meta"].update(meta)

    # Expose direct fields for easier debugging and downstream access.
    block["component_stack"] = meta.get("component_stack", [])
    block["component_context"] = meta.get("component_context", "")
    block["component_name"] = meta.get("component_name", "")
    block["component_props"] = meta.get("component_props", {})

    return block


def merge_into_meta(
    base_meta: Optional[Dict[str, Any]],
    stack: ComponentStack | None,
) -> Dict[str, Any]:
    """
    Merge component metadata into IR block metadata.
    """

    result = dict(base_meta or {})

    if stack:
        result.update(current_component_meta(stack))

    return result


def build_self_closing_component_block(
    name: str,
    props: Optional[Mapping[str, Any]],
    stack: ComponentStack | None,
) -> Dict[str, Any]:
    """
    Convert a self-closing MDX component into a lightweight content block.

    Example:
        <Card title="API Reference" href="/reference/api" />

    Becomes:
        {
            "type": "component",
            "name": "Card",
            "props": {...},
            "text": "Component: Card (href=/reference/api; title=API Reference)"
        }

    No component name is hardcoded.
    """

    clean_name = clean_component_name(name)
    clean_component_props = clean_props(props)

    prop_text = format_props(clean_component_props)

    text = f"Component: {clean_name}"

    if prop_text:
        text += f" ({prop_text})"

    block: Dict[str, Any] = {
        "type": "component",
        "name": clean_name,
        "props": clean_component_props,
        "text": text,
    }

    attached = attach_to_content_block(block, stack)

    return attached or block