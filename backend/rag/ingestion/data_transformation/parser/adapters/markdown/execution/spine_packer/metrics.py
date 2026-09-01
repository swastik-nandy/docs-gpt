from __future__ import annotations

from typing import Any, Sequence

from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.node_access import (
    node_byte_end,
    node_byte_start,
    node_estimated_tokens,
    node_level,
    node_line_end,
    node_line_start,
    node_path,
)


def common_prefix(paths: Sequence[Sequence[str]]) -> tuple[str, ...]:
    if not paths:
        return ()

    first = list(paths[0])
    prefix: list[str] = []

    for index, value in enumerate(first):
        if all(index < len(path) and path[index] == value for path in paths):
            prefix.append(value)
            continue

        break

    return tuple(prefix)


def heading_levels(nodes: Sequence[Any]) -> list[int]:
    return [
        node_level(node)
        for node in nodes
        if node_level(node) > 0
    ]


def estimated_tokens(nodes: Sequence[Any]) -> int:
    return sum(node_estimated_tokens(node) for node in nodes)


def byte_span(nodes: Sequence[Any]) -> tuple[int, int]:
    if not nodes:
        return (0, 0)

    return (
        min(node_byte_start(node) for node in nodes),
        max(node_byte_end(node) for node in nodes),
    )


def line_span(nodes: Sequence[Any]) -> tuple[int, int]:
    if not nodes:
        return (0, 0)

    return (
        min(node_line_start(node) for node in nodes),
        max(node_line_end(node) for node in nodes),
    )


def byte_size(nodes: Sequence[Any]) -> int:
    byte_start, byte_end = byte_span(nodes)

    return max(0, byte_end - byte_start)


def heading_count(nodes: Sequence[Any]) -> int:
    return len(heading_levels(nodes))


def depth_variance(nodes: Sequence[Any]) -> int:
    levels = heading_levels(nodes)

    if not levels:
        return 0

    return max(levels) - min(levels)


def paths_for_nodes(nodes: Sequence[Any]) -> list[list[str]]:
    return [node_path(node) for node in nodes]