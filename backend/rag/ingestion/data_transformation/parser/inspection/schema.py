from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal


FileFamily = Literal[
    "markdown",
    "html",
    "unknown",
]


MarkdownVariant = Literal[
    "markdown",
    "mdx",
    "unknown",
]


SizeClass = Literal[
    "small",
    "normal",
    "large",
    "huge",
    "pathological",
]


RiskLevel = Literal[
    "low",
    "medium",
    "high",
    "reject",
]


ParseStrategy = Literal[
    "single_document",
    "full_parse_then_chunk_fanout",
    "spine_fragmented",
    "quarantine",
]


@dataclass(frozen=True)
class SourceIdentity:
    """
    Stable identity for the raw uploaded artifact.

    Important:
    - Queue messages should carry IDs/pointers.
    - S3/object storage should carry the actual bytes.
    """

    path: str
    file_name: str
    extension: str
    byte_size: int
    sha256: str = ""


@dataclass(frozen=True)
class SpineNodeSummary:
    """
    Lightweight structural node discovered before full parsing.

    This is NOT a Markdown AST node.
    This is a planning/partitioning anchor.
    """

    node_id: str
    level: int
    title: str
    path: List[str]

    byte_start: int
    byte_end: int

    line_start: int
    line_end: int

    estimated_tokens: int
    parent_id: str = ""


@dataclass(frozen=True)
class MarkdownInspection:
    """
    Result of Markdown/MDX inspection.

    This tells the ingestion system:
    - what the file probably is
    - how risky it is
    - how large/complex it is
    - which parse strategy should be used
    - which queue should receive the next job
    """

    source: SourceIdentity

    family: FileFamily
    variant: MarkdownVariant

    size_class: SizeClass
    risk_level: RiskLevel
    parse_strategy: ParseStrategy
    target_queue: str

    estimated_tokens: int
    line_count: int
    heading_count: int
    max_heading_level: int
    table_like_line_count: int
    fenced_code_block_count: int
    unclosed_fence: bool
    mdx_component_hint_count: int

    spine_nodes: List[SpineNodeSummary] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)