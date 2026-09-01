from __future__ import annotations

import re

from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.boundary_safety.ranges import (
    is_self_closing_tag,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer.boundary_safety.schema import (
    LineSpan,
    UnsafeBoundaryRange,
)


MDX_COMPONENT_TAG_RE = re.compile(
    r"<(?P<closing>/)?(?P<tag>[A-Z][A-Za-z0-9_]*(?:\.[A-Z][A-Za-z0-9_]*)?)(?P<rest>[^>]*)>"
)


def scan_mdx_component_ranges(
    lines: list[LineSpan],
) -> tuple[list[UnsafeBoundaryRange], list[str]]:
    ranges: list[UnsafeBoundaryRange] = []
    warnings: list[str] = []

    stack: list[tuple[str, LineSpan]] = []

    for line in lines:
        for match in MDX_COMPONENT_TAG_RE.finditer(line.text):
            raw_tag = match.group(0)
            closing = bool(match.group("closing"))
            tag = match.group("tag")

            if is_self_closing_tag(raw_tag):
                continue

            if closing:
                for index in range(len(stack) - 1, -1, -1):
                    open_tag, open_line = stack[index]

                    if open_tag != tag:
                        continue

                    del stack[index:]

                    ranges.append(
                        UnsafeBoundaryRange(
                            kind="mdx_component",
                            start_byte=open_line.start_byte,
                            end_byte=line.end_byte,
                            start_line=open_line.line_number,
                            end_line=line.line_number,
                            detail=tag,
                        )
                    )
                    break

                continue

            stack.append((tag, line))

    if stack and lines:
        for tag, open_line in stack:
            warnings.append(f"unclosed_mdx_component_{tag}_at_line_{open_line.line_number}")
            ranges.append(
                UnsafeBoundaryRange(
                    kind="mdx_component",
                    start_byte=open_line.start_byte,
                    end_byte=lines[-1].end_byte,
                    start_line=open_line.line_number,
                    end_line=lines[-1].line_number,
                    detail=f"unclosed:{tag}",
                )
            )

    return ranges, warnings