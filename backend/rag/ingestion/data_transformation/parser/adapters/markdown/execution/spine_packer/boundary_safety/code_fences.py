from __future__ import annotations

import re

from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.boundary_safety.schema import (
    LineSpan,
    UnsafeBoundaryRange,
)


FENCE_RE = re.compile(r"^\s*(?P<fence>`{3,}|~{3,})")


def scan_code_fence_ranges(
    lines: list[LineSpan],
) -> tuple[list[UnsafeBoundaryRange], list[str]]:
    ranges: list[UnsafeBoundaryRange] = []
    warnings: list[str] = []

    open_line: LineSpan | None = None
    fence_char = ""
    fence_len = 0

    for line in lines:
        match = FENCE_RE.match(line.text)

        if not match:
            continue

        fence = match.group("fence")
        marker_char = fence[0]
        marker_len = len(fence)

        if open_line is None:
            open_line = line
            fence_char = marker_char
            fence_len = marker_len
            continue

        if marker_char != fence_char:
            continue

        if marker_len < fence_len:
            continue

        ranges.append(
            UnsafeBoundaryRange(
                kind="code_fence",
                start_byte=open_line.start_byte,
                end_byte=line.end_byte,
                start_line=open_line.line_number,
                end_line=line.line_number,
                detail=f"{fence_char * fence_len}",
            )
        )

        open_line = None
        fence_char = ""
        fence_len = 0

    if open_line is not None and lines:
        warnings.append(f"unclosed_code_fence_at_line_{open_line.line_number}")
        ranges.append(
            UnsafeBoundaryRange(
                kind="code_fence",
                start_byte=open_line.start_byte,
                end_byte=lines[-1].end_byte,
                start_line=open_line.line_number,
                end_line=lines[-1].line_number,
                detail="unclosed",
            )
        )

    return ranges, warnings