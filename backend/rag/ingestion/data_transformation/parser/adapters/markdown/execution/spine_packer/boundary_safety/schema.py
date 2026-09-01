from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LineSpan:
    line_number: int
    start_byte: int
    end_byte: int
    text: str


@dataclass(frozen=True)
class UnsafeBoundaryRange:
    kind: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    detail: str = ""

    @property
    def byte_size(self) -> int:
        return max(0, self.end_byte - self.start_byte)

    @property
    def line_count(self) -> int:
        return max(0, self.end_line - self.start_line + 1)

    def contains_boundary(self, byte_offset: int) -> bool:
        return self.start_byte < byte_offset < self.end_byte

    def overlaps_line(self, line: LineSpan) -> bool:
        return self.start_byte < line.end_byte and line.start_byte < self.end_byte


@dataclass(frozen=True)
class BoundarySafetyIndex:
    source_path: str
    byte_size: int
    unsafe_ranges: tuple[UnsafeBoundaryRange, ...]
    warnings: tuple[str, ...]

    def unsafe_ranges_at(self, byte_offset: int) -> tuple[UnsafeBoundaryRange, ...]:
        return tuple(
            unsafe_range
            for unsafe_range in self.unsafe_ranges
            if unsafe_range.contains_boundary(byte_offset)
        )

    def unsafe_kinds_at(self, byte_offset: int) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    unsafe_range.kind
                    for unsafe_range in self.unsafe_ranges_at(byte_offset)
                }
            )
        )

    def is_safe_boundary(self, byte_offset: int) -> bool:
        return not self.unsafe_kinds_at(byte_offset)

    def to_meta(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "byte_size": self.byte_size,
            "unsafe_range_count": len(self.unsafe_ranges),
            "warnings": list(self.warnings),
            "unsafe_ranges": [
                {
                    "kind": unsafe_range.kind,
                    "start_byte": unsafe_range.start_byte,
                    "end_byte": unsafe_range.end_byte,
                    "start_line": unsafe_range.start_line,
                    "end_line": unsafe_range.end_line,
                    "byte_size": unsafe_range.byte_size,
                    "line_count": unsafe_range.line_count,
                    "detail": unsafe_range.detail,
                }
                for unsafe_range in self.unsafe_ranges
            ],
        }