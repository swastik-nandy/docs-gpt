from __future__ import annotations

import re

from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.boundary_safety.ranges import (
    range_from_lines,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.boundary_safety.schema import (
    LineSpan,
    UnsafeBoundaryRange,
)


TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$"
)


def _line_looks_table_like(line: str) -> bool:
    stripped = line.strip()

    if not stripped:
        return False

    return "|" in stripped


def scan_table_ranges(lines: list[LineSpan]) -> list[UnsafeBoundaryRange]:
    ranges: list[UnsafeBoundaryRange] = []
    segment: list[LineSpan] = []

    def flush() -> None:
        nonlocal segment

        if len(segment) < 2:
            segment = []
            return

        has_separator = any(
            TABLE_SEPARATOR_RE.match(item.text.strip())
            for item in segment
        )

        if not has_separator:
            segment = []
            return

        unsafe_range = range_from_lines(
            kind="table",
            lines=segment,
            detail="markdown_table",
        )

        if unsafe_range:
            ranges.append(unsafe_range)

        segment = []

    for line in lines:
        if _line_looks_table_like(line.text):
            segment.append(line)
            continue

        flush()

    flush()

    return ranges