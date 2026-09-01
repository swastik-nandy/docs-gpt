from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PackedSpineFragment:
    packed_fragment_id: str
    source_doc_id: str

    fragment_index: int
    close_reason: str

    byte_start: int
    byte_end: int

    line_start: int
    line_end: int

    spine_nodes: tuple[Any, ...] = field(default_factory=tuple)
    spine_node_ids: tuple[str, ...] = field(default_factory=tuple)

    heading_path_prefix: tuple[str, ...] = field(default_factory=tuple)

    min_heading_level: int = 0
    max_heading_level: int = 0
    heading_count: int = 0
    estimated_tokens: int = 0

    @property
    def byte_size(self) -> int:
        return max(0, self.byte_end - self.byte_start)

    @property
    def line_count(self) -> int:
        return max(0, self.line_end - self.line_start + 1)

    @property
    def spine_node_count(self) -> int:
        return len(self.spine_nodes)

    @property
    def heading_depth_variance(self) -> int:
        if not self.min_heading_level or not self.max_heading_level:
            return 0

        return max(0, self.max_heading_level - self.min_heading_level)

    def to_meta(self) -> dict[str, Any]:
        return {
            "packed_fragment_id": self.packed_fragment_id,
            "fragment_index": self.fragment_index,
            "close_reason": self.close_reason,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "byte_size": self.byte_size,
            "line_count": self.line_count,
            "spine_node_count": self.spine_node_count,
            "spine_node_ids": list(self.spine_node_ids),
            "heading_path_prefix": list(self.heading_path_prefix),
            "min_heading_level": self.min_heading_level,
            "max_heading_level": self.max_heading_level,
            "heading_depth_variance": self.heading_depth_variance,
            "heading_count": self.heading_count,
            "estimated_tokens": self.estimated_tokens,
        }


@dataclass(frozen=True)
class PackedSpineDiagnostics:
    source_doc_id: str
    spine_node_count: int
    packed_fragment_count: int

    close_reason_counts: dict[str, int]

    unsafe_boundary_defer_counts: dict[str, int]
    unsafe_boundary_range_count: int
    boundary_safety_warnings: list[str]

    strict_boundary_enforced: bool
    strict_boundary_defer_count: int

    oversized_fragment_count: int
    oversized_fragment_reason_counts: dict[str, int]

    avg_spine_nodes_per_fragment: float
    max_spine_nodes_per_fragment: int

    avg_fragment_bytes: float
    max_fragment_bytes: int

    avg_fragment_lines: float
    max_fragment_lines: int

    avg_heading_depth_variance: float
    max_heading_depth_variance: int

    config: dict[str, Any]

    def to_meta(self) -> dict[str, Any]:
        return {
            "source_doc_id": self.source_doc_id,
            "spine_node_count": self.spine_node_count,
            "packed_fragment_count": self.packed_fragment_count,
            "close_reason_counts": self.close_reason_counts,
            "unsafe_boundary_defer_counts": self.unsafe_boundary_defer_counts,
            "unsafe_boundary_range_count": self.unsafe_boundary_range_count,
            "boundary_safety_warnings": self.boundary_safety_warnings,
            "strict_boundary_enforced": self.strict_boundary_enforced,
            "strict_boundary_defer_count": self.strict_boundary_defer_count,
            "oversized_fragment_count": self.oversized_fragment_count,
            "oversized_fragment_reason_counts": self.oversized_fragment_reason_counts,
            "avg_spine_nodes_per_fragment": round(self.avg_spine_nodes_per_fragment, 3),
            "max_spine_nodes_per_fragment": self.max_spine_nodes_per_fragment,
            "avg_fragment_bytes": round(self.avg_fragment_bytes, 3),
            "max_fragment_bytes": self.max_fragment_bytes,
            "avg_fragment_lines": round(self.avg_fragment_lines, 3),
            "max_fragment_lines": self.max_fragment_lines,
            "avg_heading_depth_variance": round(self.avg_heading_depth_variance, 3),
            "max_heading_depth_variance": self.max_heading_depth_variance,
            "config": self.config,
        }