# rag/ingestion/data_transformation/parser/adapters/markdown/ast/utils.py

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, MutableMapping


INLINE_META_KEYS = (
    "links",
    "images",
    "inline_code",
    "html_inline",
    "html_block",
    "strong_text",
)


def clean_text(text: Any) -> str:
    """
    Normalize visible text without destroying Markdown meaning.

    This is intentionally conservative:
    - collapse horizontal whitespace
    - keep paragraph boundaries
    - trim edges
    """

    if text is None:
        return ""

    value = str(text)

    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n[ \t]+", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)

    return value.strip()


def clean_single_line(text: Any) -> str:
    """
    Convert text to a compact single-line value.

    Useful for headings, table cells, labels, and link text.
    """

    value = clean_text(text)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def get_attr(token: Any, name: str, default: str = "") -> str:
    """
    Safely read an attribute from a markdown-it token.

    Works with token.attrGet() and falls back to token.attrs.
    """

    if token is None:
        return default

    try:
        value = token.attrGet(name)
        if value is not None:
            return str(value)
    except Exception:
        pass

    attrs = getattr(token, "attrs", None)

    if isinstance(attrs, dict):
        value = attrs.get(name)
        return str(value) if value is not None else default

    if isinstance(attrs, list):
        for item in attrs:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue

            key, value = item

            if key == name:
                return str(value) if value is not None else default

    return default


def find_matching_close(
    tokens: List[Any],
    start_index: int,
    close_type: str,
) -> int:
    """
    Find the matching close token for a block.

    Uses token level when available. This keeps nested structures from
    accidentally closing too early.
    """

    if start_index < 0 or start_index >= len(tokens):
        return len(tokens) - 1

    start_level = getattr(tokens[start_index], "level", None)

    for index in range(start_index + 1, len(tokens)):
        token = tokens[index]

        if getattr(token, "type", None) != close_type:
            continue

        token_level = getattr(token, "level", None)

        if start_level is None or token_level == start_level:
            return index

    return len(tokens) - 1


def ensure_list(value: Any) -> List[Any]:
    """
    Normalize a maybe-list value into a list.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    return [value]


def append_meta(
    target: MutableMapping[str, Any],
    key: str,
    value: Any,
) -> None:
    """
    Append one metadata value to target[key], ignoring empty values.
    """

    if value is None:
        return

    if isinstance(value, str) and not value.strip():
        return

    target.setdefault(key, [])
    target[key].append(value)


def extend_meta(
    target: MutableMapping[str, Any],
    key: str,
    values: Iterable[Any],
) -> None:
    """
    Extend one metadata list, ignoring empty values.
    """

    for value in values or []:
        append_meta(target, key, value)


def merge_inline_meta(
    target: MutableMapping[str, Any],
    source: MutableMapping[str, Any] | Dict[str, Any] | None,
) -> None:
    """
    Merge inline parser metadata into a block-level AST element.

    Supported keys:
    - links
    - images
    - inline_code
    - html_inline
    - html_block
    """

    if not source:
        return

    for key in INLINE_META_KEYS:
        values = source.get(key) or []

        if values:
            extend_meta(target, key, values)


def compact_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove empty values from shallow dictionaries.

    Keeps:
    - False
    - 0
    - non-empty strings
    - non-empty lists/dicts
    """

    compacted: Dict[str, Any] = {}

    for key, value in data.items():
        if value is None:
            continue

        if isinstance(value, str) and not value.strip():
            continue

        if isinstance(value, (list, dict, tuple, set)) and not value:
            continue

        compacted[key] = value

    return compacted


def normalize_language(language: Any) -> str:
    """
    Normalize fenced-code language identifiers.

    Examples:
        "bash title=file.sh" -> "bash"
        "language-python" -> "python"
    """

    value = clean_single_line(language)

    if not value:
        return ""

    value = value.split()[0].strip()
    value = value.removeprefix("language-")

    return value.lower()


def is_blank_text(text: Any) -> bool:
    """
    True when text has no meaningful visible content.
    """

    return not clean_text(text)