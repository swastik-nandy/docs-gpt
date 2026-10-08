from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


# --------------- KEYS --------------------------------

INLINE_META_KEYS = (
    "links",
    "images",
    "inline_code",
    "html_inline",
    "html_block",
    "strong_text",
)

CONTEXT_META_KEYS = (
    "meta",
)


# --------------- COPY HELPERS --------------------------------

def copy_context_fields(
    target: Dict[str, Any],
    source: Dict[str, Any],
) -> None:
    for key in CONTEXT_META_KEYS:
        if source.get(key):
            target[key] = deepcopy(source[key])


def copy_inline_metadata(
    target: Dict[str, Any],
    source: Dict[str, Any],
) -> None:
    for key in INLINE_META_KEYS:
        if source.get(key):
            target[key] = deepcopy(source[key])


def extend_meta_list(
    target: Dict[str, Any],
    key: str,
    values: Any,
) -> None:
    if not values:
        return

    if not isinstance(values, list):
        values = [values]

    if not values:
        return

    target.setdefault(key, [])
    target[key].extend(deepcopy(values))


def merge_body_metadata(
    target: Dict[str, Any],
    body_blocks: List[Dict[str, Any]],
) -> None:
    for block in body_blocks:
        for key in INLINE_META_KEYS:
            extend_meta_list(target, key, block.get(key))

        meta = block.get("meta")

        if isinstance(meta, dict) and meta:
            target.setdefault("merged_meta", [])
            target["merged_meta"].append(deepcopy(meta))
