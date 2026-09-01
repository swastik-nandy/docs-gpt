from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.boundary_safety import (
    BoundarySafetyIndex,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.config import (
    PackedSpineConfig,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.limits import (
    oversized_reasons,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.schema import (
    PackedSpineDiagnostics,
    PackedSpineFragment,
)


def avg(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def config_to_meta(config: PackedSpineConfig) -> dict[str, Any]:
    return {
        "target_fragment_bytes": config.target_fragment_bytes,
        "max_fragment_bytes": config.max_fragment_bytes,
        "max_estimated_tokens": config.max_estimated_tokens,
        "max_spine_nodes": config.max_spine_nodes,
        "max_heading_count": config.max_heading_count,
        "max_heading_depth_variance": config.max_heading_depth_variance,
        "never_cross_h1": config.never_cross_h1,
        "enforce_safe_boundaries": config.enforce_safe_boundaries,
        "protect_code_fences": config.protect_code_fences,
        "protect_html_blocks": config.protect_html_blocks,
        "protect_mdx_components": config.protect_mdx_components,
        "protect_tables": config.protect_tables,
    }


def build_oversized_reason_counts(
    packed_fragments: Sequence[PackedSpineFragment],
    config: PackedSpineConfig,
) -> Counter[str]:
    counts: Counter[str] = Counter()

    for fragment in packed_fragments:
        counts.update(oversized_reasons(fragment, config))

    return counts


def build_diagnostics(
    *,
    source_doc_id: str,
    spine_node_count: int,
    packed_fragments: list[PackedSpineFragment],
    unsafe_boundary_defer_counts: Counter[str],
    boundary_index: BoundarySafetyIndex | None,
    config: PackedSpineConfig,
) -> PackedSpineDiagnostics:
    node_counts = [
        fragment.spine_node_count
        for fragment in packed_fragments
    ]

    byte_sizes = [
        fragment.byte_size
        for fragment in packed_fragments
    ]

    line_counts = [
        fragment.line_count
        for fragment in packed_fragments
    ]

    depth_variances = [
        fragment.heading_depth_variance
        for fragment in packed_fragments
    ]

    close_reason_counts = Counter(
        fragment.close_reason
        for fragment in packed_fragments
    )

    oversized_reason_counts = build_oversized_reason_counts(
        packed_fragments=packed_fragments,
        config=config,
    )

    oversized_fragment_count = sum(
        1
        for fragment in packed_fragments
        if oversized_reasons(fragment, config)
    )

    strict_boundary_defer_count = sum(unsafe_boundary_defer_counts.values())

    return PackedSpineDiagnostics(
        source_doc_id=source_doc_id,
        spine_node_count=spine_node_count,
        packed_fragment_count=len(packed_fragments),
        close_reason_counts=dict(close_reason_counts),
        unsafe_boundary_defer_counts=dict(unsafe_boundary_defer_counts),
        unsafe_boundary_range_count=len(boundary_index.unsafe_ranges) if boundary_index else 0,
        boundary_safety_warnings=list(boundary_index.warnings) if boundary_index else [],
        strict_boundary_enforced=bool(config.enforce_safe_boundaries),
        strict_boundary_defer_count=strict_boundary_defer_count,
        oversized_fragment_count=oversized_fragment_count,
        oversized_fragment_reason_counts=dict(oversized_reason_counts),
        avg_spine_nodes_per_fragment=avg([float(value) for value in node_counts]),
        max_spine_nodes_per_fragment=max(node_counts) if node_counts else 0,
        avg_fragment_bytes=avg([float(value) for value in byte_sizes]),
        max_fragment_bytes=max(byte_sizes) if byte_sizes else 0,
        avg_fragment_lines=avg([float(value) for value in line_counts]),
        max_fragment_lines=max(line_counts) if line_counts else 0,
        avg_heading_depth_variance=avg([float(value) for value in depth_variances]),
        max_heading_depth_variance=max(depth_variances) if depth_variances else 0,
        config=config_to_meta(config),
    )