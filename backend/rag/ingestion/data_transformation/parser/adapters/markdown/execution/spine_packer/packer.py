from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.boundary_safety import (
    BoundarySafetyIndex,
    build_boundary_safety_index,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.config import (
    PackedSpineConfig,
    load_packed_spine_config,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.diagnostics import (
    build_diagnostics,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.limits import (
    close_reason_before_next,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.metrics import (
    byte_span,
    common_prefix,
    estimated_tokens,
    heading_levels,
    line_span,
    paths_for_nodes,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.node_access import (
    node_byte_start,
    node_id,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.schema import (
    PackedSpineDiagnostics,
    PackedSpineFragment,
)


def _unsafe_boundary_kinds(
    *,
    boundary_index: BoundarySafetyIndex | None,
    next_node: Any,
) -> tuple[str, ...]:
    if boundary_index is None:
        return ()

    return boundary_index.unsafe_kinds_at(node_byte_start(next_node))


def _make_defer_key(
    *,
    close_reason: str,
    unsafe_kinds: tuple[str, ...],
) -> str:
    return f"{close_reason}|{'/'.join(unsafe_kinds)}"


def _build_boundary_index(
    *,
    source_path: str | None,
    config: PackedSpineConfig,
) -> BoundarySafetyIndex | None:
    if not source_path:
        return None

    if not config.enforce_safe_boundaries:
        return None

    return build_boundary_safety_index(
        source_path,
        config=config,
    )


def _make_fragment(
    *,
    source_doc_id: str,
    fragment_index: int,
    close_reason: str,
    nodes: Sequence[Any],
) -> PackedSpineFragment:
    byte_start, byte_end = byte_span(nodes)
    line_start, line_end = line_span(nodes)

    levels = heading_levels(nodes)
    min_level = min(levels) if levels else 0
    max_level = max(levels) if levels else 0

    return PackedSpineFragment(
        packed_fragment_id=f"pf{fragment_index:04d}",
        source_doc_id=source_doc_id,
        fragment_index=fragment_index,
        close_reason=close_reason,
        byte_start=byte_start,
        byte_end=byte_end,
        line_start=line_start,
        line_end=line_end,
        spine_nodes=tuple(nodes),
        spine_node_ids=tuple(node_id(node) for node in nodes),
        heading_path_prefix=common_prefix(paths_for_nodes(nodes)),
        min_heading_level=min_level,
        max_heading_level=max_level,
        heading_count=len(levels),
        estimated_tokens=estimated_tokens(nodes),
    )


def _append_fragment(
    packed: list[PackedSpineFragment],
    *,
    source_doc_id: str,
    close_reason: str,
    nodes: Sequence[Any],
) -> None:
    if not nodes:
        return

    packed.append(
        _make_fragment(
            source_doc_id=source_doc_id,
            fragment_index=len(packed),
            close_reason=close_reason,
            nodes=nodes,
        )
    )


def pack_spine_nodes_with_diagnostics(
    spine_nodes: Iterable[Any],
    *,
    source_doc_id: str,
    config: PackedSpineConfig | None = None,
    source_path: str | None = None,
) -> tuple[list[PackedSpineFragment], PackedSpineDiagnostics]:
    active_config = config or load_packed_spine_config()
    nodes = list(spine_nodes or [])

    boundary_index = _build_boundary_index(
        source_path=source_path,
        config=active_config,
    )

    packed: list[PackedSpineFragment] = []
    current_nodes: list[Any] = []
    unsafe_boundary_defer_counts: Counter[str] = Counter()

    for node in nodes:
        close_reason = close_reason_before_next(
            current_nodes=current_nodes,
            next_node=node,
            config=active_config,
        )

        if close_reason:
            unsafe_kinds = _unsafe_boundary_kinds(
                boundary_index=boundary_index,
                next_node=node,
            )

            if unsafe_kinds:
                unsafe_boundary_defer_counts.update([
                    _make_defer_key(
                        close_reason=close_reason,
                        unsafe_kinds=unsafe_kinds,
                    )
                ])
                current_nodes.append(node)
                continue

            _append_fragment(
                packed,
                source_doc_id=source_doc_id,
                close_reason=close_reason,
                nodes=current_nodes,
            )

            current_nodes = [node]
            continue

        current_nodes.append(node)

    _append_fragment(
        packed,
        source_doc_id=source_doc_id,
        close_reason="final",
        nodes=current_nodes,
    )

    diagnostics = build_diagnostics(
        source_doc_id=source_doc_id,
        spine_node_count=len(nodes),
        packed_fragments=packed,
        unsafe_boundary_defer_counts=unsafe_boundary_defer_counts,
        boundary_index=boundary_index,
        config=active_config,
    )

    return packed, diagnostics


def pack_spine_nodes(
    spine_nodes: Iterable[Any],
    *,
    source_doc_id: str,
    config: PackedSpineConfig | None = None,
    source_path: str | None = None,
) -> list[PackedSpineFragment]:
    packed, _diagnostics = pack_spine_nodes_with_diagnostics(
        spine_nodes,
        source_doc_id=source_doc_id,
        config=config,
        source_path=source_path,
    )

    return packed