from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal


UserTier = Literal[
    "free",
    "standard",
    "premium",
    "internal",
]


DemandLevel = Literal[
    "low",
    "normal",
    "high",
    "critical",
]


ResourceClass = Literal[
    "tiny",
    "small",
    "medium",
    "large",
    "xlarge",
]


ExecutionMode = Literal[
    "sequential",
    "process_pool",
]


ParseStrategy = Literal[
    "single_document",
    "full_parse_then_chunk_fanout",
    "spine_fragmented",
    "quarantine",
]


@dataclass(frozen=True)
class MarkdownExecutionPolicyContext:
    """
    Runtime policy context for Markdown parsing.

    This is not document inspection.
    This represents the current execution environment and product policy.

    Examples:
    - user tier
    - platform demand level
    - local CPU availability
    - whether process pools are allowed
    - which resource classes are available
    """

    user_tier: UserTier
    demand_level: DemandLevel

    local_cpu_count: int
    max_local_workers: int

    allow_process_pool: bool
    available_resource_classes: tuple[ResourceClass, ...]

    prefer_fast_execution: bool = False

    def __post_init__(self) -> None:
        if self.local_cpu_count < 1:
            raise ValueError("local_cpu_count must be >= 1")

        if self.max_local_workers < 1:
            raise ValueError("max_local_workers must be >= 1")

        if not self.available_resource_classes:
            raise ValueError("available_resource_classes cannot be empty")


@dataclass(frozen=True)
class MarkdownExecutionPlan:
    """
    Concrete execution plan derived from:

    - MarkdownInspection
    - MarkdownExecutionPolicyContext
    - policy.toml

    This object does not parse anything.
    It only tells executor.py how parsing should run.
    """

    parse_strategy: ParseStrategy
    execution_mode: ExecutionMode
    resource_class: ResourceClass
    max_workers: int

    should_parse: bool = True

    reasons: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")

        if not self.should_parse and self.max_workers != 1:
            raise ValueError("non-parsing execution plans should use max_workers=1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MarkdownParsedUnit:
    """
    One independently parsed Markdown unit.

    For small/normal documents:
        unit_id = "full"
        doc_id = source document id

    For large/spine-fragmented documents:
        unit_id = fragment id
        doc_id = fragment-specific id
        source_doc_id = original source document id

    This object is storage-free. It does not know whether the caller will:
    - print quality output
    - write artifacts
    - send blocks to chunking
    - discard the result
    """

    unit_id: str
    doc_id: str
    source_doc_id: str
    parse_strategy: str

    sections: list[dict[str, Any]] = field(default_factory=list)
    blocks: list[dict[str, Any]] = field(default_factory=list)

    meta: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def section_count(self) -> int:
        return len(self.sections)

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def has_sections(self) -> bool:
        return self.section_count > 0

    @property
    def has_blocks(self) -> bool:
        return self.block_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "doc_id": self.doc_id,
            "source_doc_id": self.source_doc_id,
            "parse_strategy": self.parse_strategy,
            "section_count": self.section_count,
            "block_count": self.block_count,
            "sections": self.sections,
            "blocks": self.blocks,
            "meta": self.meta,
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class MarkdownParseExecutionResult:
    """
    Strategy-aware Markdown parse result.

    This is the unified parser output for both:
    - full-document parsing
    - spine-fragmented parsing

    It writes nothing.
    It only returns IR-ready parsed units.

    The runner/orchestrator decides what to do next:
    - quality runner prints diagnostics
    - artifact runner writes .ir files
    - future queue worker persists intermediate stages
    """

    doc_id: str
    source_path: str
    parser_family: str
    parser_variant: str
    parse_strategy: str

    inspection: Any
    execution_plan: MarkdownExecutionPlan

    units: list[MarkdownParsedUnit] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def iter_sections(self) -> Iterable[dict[str, Any]]:
        for unit in self.units:
            yield from unit.sections

    def iter_blocks(self) -> Iterable[dict[str, Any]]:
        for unit in self.units:
            yield from unit.blocks

    @property
    def unit_count(self) -> int:
        return len(self.units)

    @property
    def section_count(self) -> int:
        return sum(unit.section_count for unit in self.units)

    @property
    def block_count(self) -> int:
        return sum(unit.block_count for unit in self.units)

    @property
    def has_units(self) -> bool:
        return bool(self.units)

    @property
    def has_blocks(self) -> bool:
        return self.block_count > 0

    @property
    def should_parse(self) -> bool:
        return bool(self.execution_plan.should_parse)

    def to_dict(self) -> dict[str, Any]:
        inspection_value = self.inspection

        if hasattr(inspection_value, "to_dict"):
            inspection_value = inspection_value.to_dict()

        return {
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "parser_family": self.parser_family,
            "parser_variant": self.parser_variant,
            "parse_strategy": self.parse_strategy,
            "inspection": inspection_value,
            "execution_plan": self.execution_plan.to_dict(),
            "units": [unit.to_dict() for unit in self.units],
            "warnings": self.warnings,
            "metrics": {
                **self.metrics,
                "unit_count": self.unit_count,
                "section_count": self.section_count,
                "block_count": self.block_count,
            },
        }