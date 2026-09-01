from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, Sequence

from rag.retrieval.contracts.enums import QueryDefect, QueryIntent, RetrievalMode
from rag.retrieval.contracts.primitives import (
    Payload,
    validate_bool,
    validate_enum_instance,
    validate_finite_number,
    validate_mapping_size,
    validate_non_empty_str,
    validate_optional_clean_str,
    validate_optional_sequence_limit,
)
from rag.retrieval.contracts.query import ExtractedFilter, NormalizedQuery, QueryVariant
from rag.retrieval.contracts.request import RetrievalRequest


class RetrievalToolHint(str, Enum):
    DENSE_SEARCH = "dense_search"
    SPARSE_SEARCH = "sparse_search"
    HYBRID_SEARCH = "hybrid_search"
    EXACT_PAYLOAD_SEARCH = "exact_payload_search"
    SOURCE_NAVIGATION = "source_navigation"
    FILE_READ = "file_read"
    CHUNK_READ = "chunk_read"
    SECTION_READ = "section_read"
    LINKED_EXPANSION = "linked_expansion"
    MEMORY_LOOKUP = "memory_lookup"
    VERIFY_EVIDENCE = "verify_evidence"
    AGENTIC_LOOP = "agentic_loop"


@dataclass(frozen=True)
class QueryProcessingConfig:
    processor: str = "adaptive"
    allow_llamaindex: bool = False
    fallback_to_heuristic: bool = True
    max_semantic_queries: int = 6
    max_exact_terms: int = 32
    max_source_hints: int = 24
    max_tool_hints: int = 16
    max_warnings: int = 16
    max_metadata_keys: int = 64

    def validate(self) -> None:
        validate_non_empty_str("query_processing.processor", self.processor)
        validate_bool("query_processing.allow_llamaindex", self.allow_llamaindex)
        validate_bool("query_processing.fallback_to_heuristic", self.fallback_to_heuristic)

        for name, value in (
            ("query_processing.max_semantic_queries", self.max_semantic_queries),
            ("query_processing.max_exact_terms", self.max_exact_terms),
            ("query_processing.max_source_hints", self.max_source_hints),
            ("query_processing.max_tool_hints", self.max_tool_hints),
            ("query_processing.max_warnings", self.max_warnings),
            ("query_processing.max_metadata_keys", self.max_metadata_keys),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero.")


@dataclass(frozen=True)
class ProcessedQueryPlan:
    normalized_query: NormalizedQuery
    retrieval_mode: RetrievalMode
    tool_hints: tuple[RetrievalToolHint, ...] = field(default_factory=tuple)
    semantic_queries: tuple[str, ...] = field(default_factory=tuple)
    exact_terms: tuple[str, ...] = field(default_factory=tuple)
    source_hints: tuple[str, ...] = field(default_factory=tuple)
    extracted_filters: tuple[ExtractedFilter, ...] = field(default_factory=tuple)
    needs_agentic_loop: bool = False
    needs_exact_search: bool = False
    needs_source_navigation: bool = False
    requires_memory: bool = False
    confidence: float = 0.5
    planner: str = "unknown"
    reasoning: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    metadata: Payload = field(default_factory=dict)

    @property
    def query(self) -> str:
        return self.normalized_query.normalized_query

    @property
    def intent(self) -> QueryIntent:
        return self.normalized_query.intent

    @property
    def defects(self) -> tuple[QueryDefect, ...]:
        return self.normalized_query.defects

    @property
    def wants_memory_tool(self) -> bool:
        return self.requires_memory or RetrievalToolHint.MEMORY_LOOKUP in self.tool_hints

    def validate(self, *, config: QueryProcessingConfig | None = None) -> None:
        resolved_config = config or QueryProcessingConfig()
        resolved_config.validate()

        if not isinstance(self.normalized_query, NormalizedQuery):
            raise ValueError("processed_query_plan.normalized_query must be NormalizedQuery.")

        self.normalized_query.validate()
        validate_enum_instance(
            "processed_query_plan.retrieval_mode",
            self.retrieval_mode,
            RetrievalMode,
        )

        validate_optional_sequence_limit(
            "processed_query_plan.tool_hints",
            self.tool_hints,
            resolved_config.max_tool_hints,
        )
        for index, hint in enumerate(self.tool_hints):
            validate_enum_instance(
                f"processed_query_plan.tool_hints[{index}]",
                hint,
                RetrievalToolHint,
            )

        _validate_clean_str_sequence(
            "processed_query_plan.semantic_queries",
            self.semantic_queries,
            resolved_config.max_semantic_queries,
        )
        _validate_clean_str_sequence(
            "processed_query_plan.exact_terms",
            self.exact_terms,
            resolved_config.max_exact_terms,
        )
        _validate_clean_str_sequence(
            "processed_query_plan.source_hints",
            self.source_hints,
            resolved_config.max_source_hints,
        )

        for index, extracted_filter in enumerate(self.extracted_filters):
            if not isinstance(extracted_filter, ExtractedFilter):
                raise ValueError(
                    f"processed_query_plan.extracted_filters[{index}] "
                    "must be ExtractedFilter."
                )
            extracted_filter.validate()

        validate_bool("processed_query_plan.needs_agentic_loop", self.needs_agentic_loop)
        validate_bool("processed_query_plan.needs_exact_search", self.needs_exact_search)
        validate_bool(
            "processed_query_plan.needs_source_navigation",
            self.needs_source_navigation,
        )
        validate_bool("processed_query_plan.requires_memory", self.requires_memory)
        validate_finite_number("processed_query_plan.confidence", self.confidence)

        if not 0 <= self.confidence <= 1:
            raise ValueError("processed_query_plan.confidence must be between 0 and 1.")

        validate_non_empty_str("processed_query_plan.planner", self.planner)
        validate_optional_clean_str("processed_query_plan.reasoning", self.reasoning)

        _validate_clean_str_sequence(
            "processed_query_plan.warnings",
            self.warnings,
            resolved_config.max_warnings,
        )
        validate_mapping_size(
            "processed_query_plan.metadata",
            self.metadata,
            resolved_config.max_metadata_keys,
        )


class QueryProcessor(Protocol):
    async def process(self, request: RetrievalRequest) -> ProcessedQueryPlan:
        raise NotImplementedError


def build_normalized_query_from_plan(
    *,
    request: RetrievalRequest,
    normalized_query: str,
    intent: QueryIntent,
    defects: Sequence[QueryDefect] = (),
    semantic_queries: Sequence[str] = (),
    extracted_terms: Sequence[str] = (),
    extracted_filters: Sequence[ExtractedFilter] = (),
    requires_memory: bool = False,
    metadata: Payload | None = None,
) -> NormalizedQuery:
    cleaned_query = _clean_required(normalized_query)

    rewritten_queries = tuple(
        QueryVariant(
            query=query,
            reason="query_processing_semantic_variant",
            metadata={"source": "query_processing"},
        )
        for query in _clean_unique(semantic_queries)
        if query != cleaned_query
    )

    normalized = NormalizedQuery(
        original_query=request.query,
        normalized_query=cleaned_query,
        intent=intent,
        defects=tuple(dict.fromkeys(defects)),
        rewritten_queries=rewritten_queries,
        extracted_terms=_clean_unique(extracted_terms),
        extracted_filters=tuple(extracted_filters),
        requires_memory=requires_memory,
        metadata=metadata or {},
    )
    normalized.validate()
    return normalized


def merge_tool_hints(*groups: Sequence[RetrievalToolHint]) -> tuple[RetrievalToolHint, ...]:
    merged: list[RetrievalToolHint] = []
    seen: set[RetrievalToolHint] = set()

    for group in groups:
        for item in group:
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)

    return tuple(merged)


def clean_unique_strings(values: Sequence[Any], *, limit: int | None = None) -> tuple[str, ...]:
    cleaned = _clean_unique(values)
    if limit is None:
        return cleaned
    return cleaned[:limit]


def _validate_clean_str_sequence(
    name: str,
    values: Sequence[str],
    max_count: int | None,
) -> None:
    validate_optional_sequence_limit(name, values, max_count)

    for index, value in enumerate(values):
        validate_non_empty_str(f"{name}[{index}]", value)


def _clean_required(value: Any) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError("normalized query cannot be empty.")
    return " ".join(cleaned.split())


def _clean_unique(values: Sequence[Any]) -> tuple[str, ...]:
    cleaned: list[str] = []
    seen: set[str] = set()

    for value in values:
        item = " ".join(str(value).strip().split())
        if not item or item in seen:
            continue
        seen.add(item)
        cleaned.append(item)

    return tuple(cleaned)