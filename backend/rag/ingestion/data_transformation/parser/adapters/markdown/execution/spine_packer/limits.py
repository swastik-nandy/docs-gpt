from __future__ import annotations

from typing import Any, Sequence

from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.config import (
    PackedSpineConfig,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.metrics import (
    byte_size,
    depth_variance,
    estimated_tokens,
    heading_count,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.node_access import (
    node_level,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.schema import (
    PackedSpineFragment,
)


def hard_limit_close_reason(
    current_nodes: Sequence[Any],
    next_node: Any,
    config: PackedSpineConfig,
) -> str:
    if not current_nodes:
        return ""

    if config.never_cross_h1 and node_level(next_node) == 1:
        return "h1_boundary"

    candidate = [*current_nodes, next_node]

    if byte_size(candidate) > config.max_fragment_bytes:
        return "max_fragment_bytes"

    if estimated_tokens(candidate) > config.max_estimated_tokens:
        return "max_estimated_tokens"

    if len(candidate) > config.max_spine_nodes:
        return "max_spine_nodes"

    if heading_count(candidate) > config.max_heading_count:
        return "max_heading_count"

    if depth_variance(candidate) > config.max_heading_depth_variance:
        return "max_heading_depth_variance"

    return ""


def soft_limit_close_reason(
    current_nodes: Sequence[Any],
    config: PackedSpineConfig,
) -> str:
    if not current_nodes:
        return ""

    if byte_size(current_nodes) >= config.target_fragment_bytes:
        return "target_fragment_bytes"

    return ""


def close_reason_before_next(
    current_nodes: Sequence[Any],
    next_node: Any,
    config: PackedSpineConfig,
) -> str:
    hard_reason = hard_limit_close_reason(
        current_nodes=current_nodes,
        next_node=next_node,
        config=config,
    )

    if hard_reason:
        return hard_reason

    return soft_limit_close_reason(
        current_nodes=current_nodes,
        config=config,
    )


def oversized_reasons(
    fragment: PackedSpineFragment,
    config: PackedSpineConfig,
) -> tuple[str, ...]:
    reasons: list[str] = []

    if fragment.byte_size > config.max_fragment_bytes:
        reasons.append("max_fragment_bytes")

    if fragment.estimated_tokens > config.max_estimated_tokens:
        reasons.append("max_estimated_tokens")

    if fragment.spine_node_count > config.max_spine_nodes:
        reasons.append("max_spine_nodes")

    if fragment.heading_count > config.max_heading_count:
        reasons.append("max_heading_count")

    if fragment.heading_depth_variance > config.max_heading_depth_variance:
        reasons.append("max_heading_depth_variance")

    return tuple(reasons)