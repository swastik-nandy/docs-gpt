from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List, Optional

from rag.ingestion.data_transformation.parser.inspection.schema import (
    SourceIdentity,
    SpineNodeSummary,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.spine.schema import (
    MarkdownSpineScan,
    MarkdownVariant,
)


HEADING_RE = re.compile(
    r"^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*#*\s*$"
)

FENCE_RE = re.compile(
    r"^\s*(?P<fence>`{3,}|~{3,})"
)

TABLE_LIKE_RE = re.compile(r"\|")

MDX_COMPONENT_HINT_RE = re.compile(
    r"</?[A-Z][A-Za-z0-9_.]*(\s|>|/>)"
)


def _hash_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute SHA-256 without loading the whole file into memory.
    """

    sha = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            sha.update(chunk)

    return sha.hexdigest()


def _estimate_tokens_from_text(text: str) -> int:
    """
    Cheap token estimate.

    Avoid tokenizer dependency in inspection path.
    Good enough for routing decisions.
    """

    text = text or ""

    if not text.strip():
        return 0

    return max(1, len(text) // 4)


def _clean_heading_title(title: str) -> str:
    return " ".join(str(title or "").strip().split())


def _make_node_id(path: List[str], line_no: int) -> str:
    """
    Create readable spine node ID.

    Later we can replace/augment this with stable hashing.
    """

    raw = " > ".join(path) or f"line-{line_no}"
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", raw.lower()).strip("_") or "section"
    return f"{slug}_{line_no}"


class _OpenNode:
    """
    Mutable internal node while streaming.

    Converted to frozen SpineNodeSummary at close time.
    """

    def __init__(
        self,
        node_id: str,
        level: int,
        title: str,
        path: List[str],
        byte_start: int,
        line_start: int,
        parent_id: str,
    ) -> None:
        self.node_id = node_id
        self.level = level
        self.title = title
        self.path = path
        self.byte_start = byte_start
        self.byte_end = byte_start
        self.line_start = line_start
        self.line_end = line_start
        self.estimated_tokens = 0
        self.parent_id = parent_id

    def to_summary(self) -> SpineNodeSummary:
        return SpineNodeSummary(
            node_id=self.node_id,
            level=self.level,
            title=self.title,
            path=self.path,
            byte_start=self.byte_start,
            byte_end=self.byte_end,
            line_start=self.line_start,
            line_end=self.line_end,
            estimated_tokens=self.estimated_tokens,
            parent_id=self.parent_id,
        )


def scan_markdown_spine(
    file_path: str,
    variant: MarkdownVariant,
) -> MarkdownSpineScan:
    """
    Stream through a Markdown/MDX file and build a lightweight structural spine.

    This does NOT:
    - run markdown-it
    - build AST
    - normalize full MDX
    - build IR
    - enrich sections

    It only discovers safe structural anchors and estimates complexity.
    """

    path = Path(file_path)

    if not path.exists():
        raise ValueError(f"Markdown spine scan failed. File not found: {file_path}")

    byte_size = path.stat().st_size
    sha256 = _hash_file(file_path)

    source = SourceIdentity(
        path=str(path),
        file_name=path.name,
        extension=path.suffix.lower(),
        byte_size=byte_size,
        sha256=sha256,
    )

    warnings: List[str] = []

    line_count = 0
    estimated_tokens = 0
    heading_count = 0
    max_heading_level = 0
    table_like_line_count = 0
    fenced_code_block_count = 0
    mdx_component_hint_count = 0

    in_fence = False
    fence_marker: Optional[str] = None

    finalized_nodes: List[SpineNodeSummary] = []
    stack: List[_OpenNode] = []

    byte_offset = 0

    def close_nodes_until(
        level: int,
        close_byte: int,
        close_line: int,
    ) -> None:
        while stack and stack[-1].level >= level:
            node = stack.pop()
            node.byte_end = max(node.byte_start, close_byte)
            node.line_end = max(node.line_start, close_line)
            finalized_nodes.append(node.to_summary())

    with open(file_path, "rb") as file:
        for raw_line in file:
            line_start_byte = byte_offset
            byte_offset += len(raw_line)
            line_count += 1

            # Inspector should not die on bad bytes.
            line = raw_line.decode("utf-8", errors="replace")
            stripped = line.strip("\n\r")

            line_tokens = _estimate_tokens_from_text(stripped)
            estimated_tokens += line_tokens

            # Every open parent section owns this line semantically.
            for node in stack:
                node.estimated_tokens += line_tokens

            fence_match = FENCE_RE.match(stripped)

            if fence_match:
                marker = fence_match.group("fence")[0]

                if not in_fence:
                    in_fence = True
                    fence_marker = marker
                    fenced_code_block_count += 1
                elif fence_marker == marker:
                    in_fence = False
                    fence_marker = None

                continue

            if in_fence:
                continue

            if TABLE_LIKE_RE.search(stripped):
                table_like_line_count += 1

            if variant == "mdx" and MDX_COMPONENT_HINT_RE.search(stripped):
                mdx_component_hint_count += 1

            heading_match = HEADING_RE.match(stripped)

            if not heading_match:
                continue

            level = len(heading_match.group("marks"))
            title = _clean_heading_title(heading_match.group("title"))

            if not title:
                continue

            heading_count += 1
            max_heading_level = max(max_heading_level, level)

            close_nodes_until(
                level=level,
                close_byte=line_start_byte,
                close_line=line_count - 1,
            )

            parent_path = stack[-1].path if stack else []
            path_parts = parent_path + [title]
            parent_id = stack[-1].node_id if stack else ""
            node_id = _make_node_id(path_parts, line_count)

            stack.append(
                _OpenNode(
                    node_id=node_id,
                    level=level,
                    title=title,
                    path=path_parts,
                    byte_start=line_start_byte,
                    line_start=line_count,
                    parent_id=parent_id,
                )
            )

    close_nodes_until(
        level=0,
        close_byte=byte_size,
        close_line=line_count,
    )

    if in_fence:
        warnings.append("unclosed_code_fence_detected")

    if not finalized_nodes:
        finalized_nodes.append(
            SpineNodeSummary(
                node_id="root_1",
                level=1,
                title="root",
                path=[],
                byte_start=0,
                byte_end=byte_size,
                line_start=1,
                line_end=line_count,
                estimated_tokens=estimated_tokens,
                parent_id="",
            )
        )
        warnings.append("no_heading_spine_detected")

    largest_section_tokens = max(
        (node.estimated_tokens for node in finalized_nodes),
        default=0,
    )

    return MarkdownSpineScan(
        source=source,
        variant=variant,
        estimated_tokens=estimated_tokens,
        line_count=line_count,
        heading_count=heading_count,
        max_heading_level=max_heading_level,
        table_like_line_count=table_like_line_count,
        fenced_code_block_count=fenced_code_block_count,
        unclosed_fence=in_fence,
        mdx_component_hint_count=mdx_component_hint_count,
        spine_nodes=finalized_nodes,
        warnings=warnings,
        metrics={
            "largest_section_estimated_tokens": largest_section_tokens,
            "spine_node_count": len(finalized_nodes),
            "avg_tokens_per_spine_node": round(
                estimated_tokens / max(1, len(finalized_nodes)),
                2,
            ),
        },
    )