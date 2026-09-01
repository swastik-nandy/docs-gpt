from __future__ import annotations

from pathlib import Path

from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.boundary_safety.code_fences import (
    scan_code_fence_ranges,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.boundary_safety.html_blocks import (
    scan_html_block_ranges,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.boundary_safety.line_spans import (
    read_line_spans,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.boundary_safety.mdx_components import (
    scan_mdx_component_ranges,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.boundary_safety.ranges import (
    dedupe_ranges,
    lines_outside_ranges,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.boundary_safety.schema import (
    BoundarySafetyIndex,
    UnsafeBoundaryRange,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.boundary_safety.tables import (
    scan_table_ranges,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.config import (
    PackedSpineConfig,
)


def build_boundary_safety_index(
    source_path: str,
    *,
    config: PackedSpineConfig,
) -> BoundarySafetyIndex:
    path = Path(source_path).resolve()

    raw, lines = read_line_spans(path)

    ranges: list[UnsafeBoundaryRange] = []
    warnings: list[str] = []

    fence_ranges: list[UnsafeBoundaryRange] = []

    if config.protect_code_fences:
        fence_ranges, fence_warnings = scan_code_fence_ranges(lines)
        ranges.extend(fence_ranges)
        warnings.extend(fence_warnings)

    non_code_lines = lines_outside_ranges(
        lines=lines,
        ranges=fence_ranges,
    )

    html_ranges: list[UnsafeBoundaryRange] = []

    if config.protect_html_blocks:
        html_ranges, html_warnings = scan_html_block_ranges(non_code_lines)
        ranges.extend(html_ranges)
        warnings.extend(html_warnings)

    mdx_ranges: list[UnsafeBoundaryRange] = []

    if config.protect_mdx_components:
        mdx_ranges, mdx_warnings = scan_mdx_component_ranges(non_code_lines)
        ranges.extend(mdx_ranges)
        warnings.extend(mdx_warnings)

    if config.protect_tables:
        table_exclusion_ranges = [
            *fence_ranges,
            *html_ranges,
            *mdx_ranges,
        ]

        table_lines = lines_outside_ranges(
            lines=lines,
            ranges=table_exclusion_ranges,
        )

        ranges.extend(scan_table_ranges(table_lines))

    return BoundarySafetyIndex(
        source_path=str(path),
        byte_size=len(raw),
        unsafe_ranges=tuple(dedupe_ranges(ranges)),
        warnings=tuple(warnings),
    )