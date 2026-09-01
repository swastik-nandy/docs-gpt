from __future__ import annotations

from pathlib import Path

from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.boundary_safety.schema import (
    LineSpan,
)


def read_line_spans(path: Path) -> tuple[bytes, list[LineSpan]]:
    raw = path.read_bytes()
    chunks = raw.splitlines(keepends=True)

    spans: list[LineSpan] = []
    cursor = 0

    for index, chunk in enumerate(chunks, start=1):
        start = cursor
        end = start + len(chunk)
        cursor = end

        spans.append(
            LineSpan(
                line_number=index,
                start_byte=start,
                end_byte=end,
                text=chunk.decode("utf-8", errors="replace"),
            )
        )

    return raw, spans