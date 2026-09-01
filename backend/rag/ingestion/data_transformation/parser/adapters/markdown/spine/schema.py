from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal

from rag.ingestion.data_transformation.parser.inspection.schema import (
    SourceIdentity,
    SpineNodeSummary,
)


MarkdownVariant = Literal[
    "markdown",
    "mdx",
]


@dataclass(frozen=True)
class MarkdownSpineScan:
    """
    Result of a cheap streaming scan over a Markdown/MDX file.

    This is NOT:
    - final AST
    - final IR
    - enriched sections
    - chunks

    This is only a planning artifact used to decide:
    - single parse
    - full parse then chunk fanout
    - spine-fragmented parse
    - quarantine
    """

    source: SourceIdentity
    variant: MarkdownVariant

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

    metrics: Dict[str, int | float | str | bool] = field(default_factory=dict)