# rag/ingestion/data_transformation/parser/adapters/markdown/enrichment/component_context/format.py

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Optional

from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.component_context.normalize import (
    clean_component_name,
    clean_props,
    clean_text,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.component_context.types import (
    ComponentStack,
)


def format_props(props: Optional[Mapping[str, Any]]) -> str:
    """
    Convert component props into deterministic readable text.

    Example:
        {"label": "v1.42", "description": "2026-04-13"}

    Becomes:
        description=2026-04-13; label=v1.42
    """

    cleaned = clean_props(props)

    if not cleaned:
        return ""

    parts: list[str] = []

    for key in sorted(cleaned.keys()):
        value = cleaned.get(key)

        if value is None:
            continue

        if isinstance(value, (list, dict, tuple, set)):
            value_text = str(value)
        else:
            value_text = clean_text(value)

        if not value_text:
            continue

        parts.append(f"{key}={value_text}")

    return "; ".join(parts)


def format_component(frame: Mapping[str, Any]) -> str:
    """
    Convert one component frame into readable context.

    Example:
        Update(description=2026-04-13; label=v1.42)
    """

    name = clean_component_name(frame.get("name"))
    props = clean_props(frame.get("props") or {})

    prop_text = format_props(props)

    if prop_text:
        return f"{name}({prop_text})"

    return name


def format_stack(stack: ComponentStack | None) -> str:
    """
    Convert a nested component stack into readable context.

    Example:
        Update(label=v1.42) > Tabs > Tab(title=npm)
    """

    frames = stack or []

    if not frames:
        return ""

    parts = [
        format_component(frame)
        for frame in frames
        if frame
    ]

    return " > ".join(part for part in parts if part)


def current_component_meta(stack: ComponentStack | None) -> Dict[str, Any]:
    """
    Build generic component metadata from the current stack.

    This metadata can be attached to:
    - sections
    - content blocks
    - IR blocks
    """

    if not stack:
        return {}

    stack_copy = deepcopy(stack or [])
    nearest = stack_copy[-1] if stack_copy else {}

    return {
        "component_stack": stack_copy,
        "component_context": format_stack(stack_copy),
        "component_name": nearest.get("name", ""),
        "component_props": nearest.get("props", {}),
    }


def effective_local_context(
    base_local_context: str,
    stack: ComponentStack | None,
) -> str:
    """
    Combine component context with heading/local context.

    This makes retrieval text benefit from MDX nesting without making
    lower-level IR builders know about MDX syntax.
    """

    base = clean_text(base_local_context)
    component_context = format_stack(stack)

    if component_context and base:
        return f"{component_context} > {base}"

    return component_context or base