from __future__ import annotations

from typing import Any


def node_id(node: Any) -> str:
    return str(getattr(node, "node_id", "") or "").strip()


def node_level(node: Any) -> int:
    return int(getattr(node, "level", 0) or 0)


def node_path(node: Any) -> list[str]:
    path = getattr(node, "path", []) or []

    if not isinstance(path, list):
        return []

    cleaned: list[str] = []

    for item in path:
        value = str(item or "").strip()

        if value:
            cleaned.append(value)

    return cleaned


def node_byte_start(node: Any) -> int:
    return int(getattr(node, "byte_start", 0) or 0)


def node_byte_end(node: Any) -> int:
    return int(getattr(node, "byte_end", 0) or 0)


def node_line_start(node: Any) -> int:
    return int(getattr(node, "line_start", 0) or 0)


def node_line_end(node: Any) -> int:
    return int(getattr(node, "line_end", 0) or 0)


def node_estimated_tokens(node: Any) -> int:
    return int(getattr(node, "estimated_tokens", 0) or 0)