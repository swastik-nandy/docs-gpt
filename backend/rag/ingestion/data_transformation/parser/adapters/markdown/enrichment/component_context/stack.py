# rag/ingestion/data_transformation/parser/adapters/markdown/enrichment/component_context/stack.py

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Optional

from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.component_context.normalize import (
    clean_component_name,
    clean_props,
    clean_text,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.component_context.types import (
    ComponentStack,
)


def copy_stack(stack: ComponentStack | None) -> ComponentStack:
    """
    Return a safe copy of the active component stack.
    """

    return deepcopy(stack or [])


def push_component(
    stack: ComponentStack,
    name: str,
    props: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Push a component frame onto the active stack.
    """

    stack.append({
        "name": clean_component_name(name),
        "props": clean_props(props),
    })


def pop_component(
    stack: ComponentStack,
    name: str = "",
) -> None:
    """
    Pop a component frame safely.

    Handles:
    - normal nesting
    - missing close names
    - imperfect close order
    """

    if not stack:
        return

    clean_name = clean_text(name)

    if not clean_name:
        stack.pop()
        return

    if stack[-1].get("name") == clean_name:
        stack.pop()
        return

    # Recover from imperfect close order by removing the nearest matching
    # frame from the top side.
    for index in range(len(stack) - 1, -1, -1):
        if stack[index].get("name") == clean_name:
            del stack[index:]
            return


def stack_from_content_or_section(
    item: Mapping[str, Any],
    section: Mapping[str, Any],
) -> ComponentStack:
    """
    Resolve component stack for a content block.

    Priority:
    1. direct item component_stack
    2. item meta component_stack
    3. section component_stack
    """

    item_meta = item.get("meta", {}) or {}

    return copy_stack(
        item.get("component_stack")
        or item_meta.get("component_stack")
        or section.get("component_stack")
        or []
    )