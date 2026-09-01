from __future__ import annotations

from dataclasses import dataclass

from rag.retrieval.contracts.enums import QueryProcessingMode
from rag.retrieval.contracts.config.models.shared import (
    validate_bool,
    validate_non_negative_int,
    validate_positive_int,
)


@dataclass(frozen=True)
class QueryProcessingDefaultsConfig:
    mode: QueryProcessingMode
    allow_query_rewrite: bool
    allow_filter_extraction: bool
    allow_intent_classification: bool
    allow_hypothetical_answer: bool
    max_rewritten_queries: int
    max_extracted_terms: int

    def validate(self) -> None:
        if not isinstance(self.mode, QueryProcessingMode):
            raise ValueError(
                "query_processing.defaults.mode must be QueryProcessingMode."
            )

        validate_bool(
            "query_processing.defaults.allow_query_rewrite",
            self.allow_query_rewrite,
        )
        validate_bool(
            "query_processing.defaults.allow_filter_extraction",
            self.allow_filter_extraction,
        )
        validate_bool(
            "query_processing.defaults.allow_intent_classification",
            self.allow_intent_classification,
        )
        validate_bool(
            "query_processing.defaults.allow_hypothetical_answer",
            self.allow_hypothetical_answer,
        )

        validate_non_negative_int(
            "query_processing.defaults.max_rewritten_queries",
            self.max_rewritten_queries,
        )
        validate_positive_int(
            "query_processing.defaults.max_extracted_terms",
            self.max_extracted_terms,
        )

        if not self.allow_query_rewrite and self.max_rewritten_queries > 0:
            raise ValueError(
                "query_processing.defaults.max_rewritten_queries must be zero "
                "when query rewrite is disabled."
            )


@dataclass(frozen=True)
class QueryProcessingLimitsConfig:
    max_rewritten_queries: int
    max_extracted_terms: int
    max_search_queries: int

    def validate(self) -> None:
        validate_non_negative_int(
            "query_processing.limits.max_rewritten_queries",
            self.max_rewritten_queries,
        )
        validate_positive_int(
            "query_processing.limits.max_extracted_terms",
            self.max_extracted_terms,
        )
        validate_positive_int(
            "query_processing.limits.max_search_queries",
            self.max_search_queries,
        )


@dataclass(frozen=True)
class QueryTextLimitsConfig:
    min_chars: int
    max_chars: int
    too_short_max_chars: int
    too_long_min_chars: int
    low_information_max_terms: int

    def validate(self) -> None:
        validate_positive_int("query_text.limits.min_chars", self.min_chars)
        validate_positive_int("query_text.limits.max_chars", self.max_chars)
        validate_positive_int(
            "query_text.limits.too_short_max_chars",
            self.too_short_max_chars,
        )
        validate_positive_int(
            "query_text.limits.too_long_min_chars",
            self.too_long_min_chars,
        )
        validate_non_negative_int(
            "query_text.limits.low_information_max_terms",
            self.low_information_max_terms,
        )

        if self.max_chars < self.min_chars:
            raise ValueError(
                "query_text.limits.max_chars must be >= query_text.limits.min_chars."
            )

        if self.too_short_max_chars < self.min_chars:
            raise ValueError(
                "query_text.limits.too_short_max_chars must be >= "
                "query_text.limits.min_chars."
            )

        if self.too_long_min_chars > self.max_chars:
            raise ValueError(
                "query_text.limits.too_long_min_chars must be <= "
                "query_text.limits.max_chars."
            )