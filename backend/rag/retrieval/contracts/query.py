from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

from rag.retrieval.contracts.config import (
    RetrievalContractConfig,
    get_retrieval_contract_config,
)
from rag.retrieval.contracts.enums import QueryDefect, QueryIntent
from rag.retrieval.contracts.primitives import (
    Payload,
    clean_tuple,
    validate_enum_instance,
    validate_mapping_size,
    validate_non_empty_str,
    validate_optional_clean_str,
    validate_optional_sequence_limit,
    validate_positive_int,
)


@dataclass(frozen=True)
class QueryVariant:
    query: str
    reason: str | None = None
    weight: float = 1.0
    metadata: Payload = field(default_factory=dict)

    def normalized(self) -> Self:
        return type(self)(
            query=self.query.strip(),
            reason=_clean_optional(self.reason),
            weight=float(self.weight),
            metadata=dict(self.metadata),
        )

    def validate(
        self,
        *,
        max_metadata_keys: int | None,
    ) -> None:
        validate_non_empty_str("query_variant.query", self.query)
        validate_optional_clean_str("query_variant.reason", self.reason)

        if self.weight <= 0:
            raise ValueError("query_variant.weight must be greater than zero.")

        validate_mapping_size(
            "query_variant.metadata",
            self.metadata,
            max_metadata_keys,
        )


@dataclass(frozen=True)
class ExtractedFilter:
    key: str
    value: str | int | float | bool
    operator: str = "eq"
    confidence: float = 1.0

    def normalized(self) -> Self:
        return type(self)(
            key=self.key.strip(),
            value=self.value,
            operator=self.operator.strip().lower(),
            confidence=float(self.confidence),
        )

    def validate(self) -> None:
        validate_non_empty_str("extracted_filter.key", self.key)
        validate_non_empty_str("extracted_filter.operator", self.operator)

        if self.operator not in {"eq", "neq", "gt", "gte", "lt", "lte", "contains"}:
            raise ValueError(
                "extracted_filter.operator must be one of: "
                "eq, neq, gt, gte, lt, lte, contains."
            )

        if not 0 <= self.confidence <= 1:
            raise ValueError("extracted_filter.confidence must be between 0 and 1.")


@dataclass(frozen=True)
class NormalizedQuery:
    original_query: str
    normalized_query: str
    intent: QueryIntent = QueryIntent.UNKNOWN
    defects: tuple[QueryDefect, ...] = field(default_factory=tuple)
    rewritten_queries: tuple[QueryVariant, ...] = field(default_factory=tuple)
    extracted_terms: tuple[str, ...] = field(default_factory=tuple)
    extracted_filters: tuple[ExtractedFilter, ...] = field(default_factory=tuple)
    hypothetical_answer: str | None = None
    requires_memory: bool = False
    metadata: Payload = field(default_factory=dict)

    @property
    def search_queries(self) -> tuple[str, ...]:
        queries = [self.normalized_query]
        queries.extend(variant.query for variant in self.rewritten_queries)
        return tuple(dict.fromkeys(query for query in queries if query.strip()))

    @property
    def has_defects(self) -> bool:
        return bool(self.defects)

    @property
    def is_rewritten(self) -> bool:
        return bool(self.rewritten_queries)

    @property
    def is_chat_dependent(self) -> bool:
        return QueryDefect.CHAT_DEPENDENT in self.defects or self.requires_memory

    def normalized(self) -> Self:
        return type(self)(
            original_query=self.original_query.strip(),
            normalized_query=self.normalized_query.strip(),
            intent=self.intent,
            defects=tuple(self.defects),
            rewritten_queries=tuple(
                variant.normalized() for variant in self.rewritten_queries
            ),
            extracted_terms=clean_tuple(self.extracted_terms),
            extracted_filters=tuple(
                extracted_filter.normalized()
                for extracted_filter in self.extracted_filters
            ),
            hypothetical_answer=_clean_optional(self.hypothetical_answer),
            requires_memory=bool(self.requires_memory),
            metadata=dict(self.metadata),
        )

    def validate(
        self,
        *,
        config: RetrievalContractConfig | None = None,
    ) -> None:
        resolved_config = config or get_retrieval_contract_config()

        validate_non_empty_str("normalized_query.original_query", self.original_query)
        validate_non_empty_str(
            "normalized_query.normalized_query",
            self.normalized_query,
        )
        validate_enum_instance("normalized_query.intent", self.intent, QueryIntent)

        self._validate_query_length(resolved_config)
        self._validate_defects()
        self._validate_rewritten_queries(resolved_config)
        self._validate_extracted_terms(resolved_config)
        self._validate_extracted_filters()
        validate_optional_clean_str(
            "normalized_query.hypothetical_answer",
            self.hypothetical_answer,
        )
        validate_mapping_size(
            "normalized_query.metadata",
            self.metadata,
            resolved_config.request_limits.max_metadata_keys,
        )

    def _validate_query_length(self, config: RetrievalContractConfig) -> None:
        length = len(self.normalized_query.strip())
        limits = config.query_text_limits

        if length < limits.min_chars:
            raise ValueError(
                "normalized_query.normalized_query must contain at least "
                f"{limits.min_chars} characters."
            )

        if length > limits.max_chars:
            raise ValueError(
                "normalized_query.normalized_query cannot exceed "
                f"{limits.max_chars} characters."
            )

    def _validate_defects(self) -> None:
        for index, defect in enumerate(self.defects):
            validate_enum_instance(
                f"normalized_query.defects[{index}]",
                defect,
                QueryDefect,
            )

    def _validate_rewritten_queries(
        self,
        config: RetrievalContractConfig,
    ) -> None:
        validate_optional_sequence_limit(
            "normalized_query.rewritten_queries",
            self.rewritten_queries,
            config.query_processing_limits.max_rewritten_queries,
        )

        for index, variant in enumerate(self.rewritten_queries):
            if not isinstance(variant, QueryVariant):
                raise ValueError(
                    f"normalized_query.rewritten_queries[{index}] "
                    "must be QueryVariant."
                )

            variant.validate(
                max_metadata_keys=config.request_limits.max_metadata_keys,
            )

        validate_optional_sequence_limit(
            "normalized_query.search_queries",
            self.search_queries,
            config.query_processing_limits.max_search_queries,
        )

    def _validate_extracted_terms(
        self,
        config: RetrievalContractConfig,
    ) -> None:
        validate_optional_sequence_limit(
            "normalized_query.extracted_terms",
            self.extracted_terms,
            config.query_processing_limits.max_extracted_terms,
        )

        for index, term in enumerate(self.extracted_terms):
            validate_non_empty_str(
                f"normalized_query.extracted_terms[{index}]",
                term,
            )

    def _validate_extracted_filters(self) -> None:
        for index, extracted_filter in enumerate(self.extracted_filters):
            if not isinstance(extracted_filter, ExtractedFilter):
                raise ValueError(
                    f"normalized_query.extracted_filters[{index}] "
                    "must be ExtractedFilter."
                )

            extracted_filter.validate()


def build_initial_normalized_query(query: str) -> NormalizedQuery:
    normalized_query = NormalizedQuery(
        original_query=query,
        normalized_query=query.strip(),
    )
    normalized_query.validate()
    return normalized_query


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None

    cleaned = value.strip()
    return cleaned or None