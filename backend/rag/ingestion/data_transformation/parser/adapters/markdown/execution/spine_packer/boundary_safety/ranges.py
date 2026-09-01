from __future__ import annotations

from typing import Iterable

from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.boundary_safety.schema import (
    LineSpan,
    UnsafeBoundaryRange,
)


def range_from_lines(
    *,
    kind: str,
    lines: list[LineSpan],
    detail: str = "",
) -> UnsafeBoundaryRange | None:
    if not lines:
        return None

    return UnsafeBoundaryRange(
        kind=kind,
        start_byte=lines[0].start_byte,
        end_byte=lines[-1].end_byte,
        start_line=lines[0].line_number,
        end_line=lines[-1].line_number,
        detail=detail,
    )


def is_self_closing_tag(raw_tag: str) -> bool:
    return raw_tag.rstrip().endswith("/>")


def line_overlaps_any_range(
    line: LineSpan,
    ranges: Iterable[UnsafeBoundaryRange],
) -> bool:
    return any(unsafe_range.overlaps_line(line) for unsafe_range in ranges)


def lines_outside_ranges(
    lines: list[LineSpan],
    ranges: Iterable[UnsafeBoundaryRange],
) -> list[LineSpan]:
    protected = tuple(ranges)

    if not protected:
        return list(lines)

    return [
        line
        for line in lines
        if not line_overlaps_any_range(line, protected)
    ]


def sorted_ranges(
    ranges: Iterable[UnsafeBoundaryRange],
) -> tuple[UnsafeBoundaryRange, ...]:
    return tuple(
        sorted(
            ranges,
            key=lambda item: (
                item.start_byte,
                item.end_byte,
                item.kind,
                item.detail,
            ),
        )
    )


def dedupe_ranges(
    ranges: Iterable[UnsafeBoundaryRange],
) -> list[UnsafeBoundaryRange]:
    seen: set[tuple[str, int, int, int, int, str]] = set()
    deduped: list[UnsafeBoundaryRange] = []

    for unsafe_range in sorted_ranges(ranges):
        key = (
            unsafe_range.kind,
            unsafe_range.start_byte,
            unsafe_range.end_byte,
            unsafe_range.start_line,
            unsafe_range.end_line,
            unsafe_range.detail,
        )

        if key in seen:
            continue

        seen.add(key)
        deduped.append(unsafe_range)

    return deduped