from __future__ import annotations

from dataclasses import dataclass

from rag.retrieval.contracts.config.models.shared import (
    validate_bool,
    validate_non_empty,
    validate_non_negative_int,
    validate_positive_int,
)
from rag.retrieval.contracts.enums import RetrievalMode


@dataclass(frozen=True)
class ContractSchemaConfig:
    version: str

    def validate(self) -> None:
        validate_non_empty("schema.version", self.version)


@dataclass(frozen=True)
class RequestDefaultsConfig:
    mode: RetrievalMode
    top_k: int

    def validate(self) -> None:
        if not isinstance(self.mode, RetrievalMode):
            raise ValueError("request.defaults.mode must be RetrievalMode.")

        validate_positive_int("request.defaults.top_k", self.top_k)


@dataclass(frozen=True)
class RequestLimitsConfig:
    max_top_k: int
    max_chat_history_turns: int
    max_metadata_keys: int

    def validate(self) -> None:
        validate_positive_int("request.limits.max_top_k", self.max_top_k)
        validate_non_negative_int(
            "request.limits.max_chat_history_turns",
            self.max_chat_history_turns,
        )
        validate_non_negative_int(
            "request.limits.max_metadata_keys",
            self.max_metadata_keys,
        )


@dataclass(frozen=True)
class SourceConstraintDefaultsConfig:
    allow_global_search: bool
    allow_payload_filters: bool
    max_heading_path_depth: int
    max_payload_filter_keys: int

    def validate(self) -> None:
        validate_bool(
            "source_constraints.defaults.allow_global_search",
            self.allow_global_search,
        )
        validate_bool(
            "source_constraints.defaults.allow_payload_filters",
            self.allow_payload_filters,
        )
        validate_non_negative_int(
            "source_constraints.defaults.max_heading_path_depth",
            self.max_heading_path_depth,
        )
        validate_non_negative_int(
            "source_constraints.defaults.max_payload_filter_keys",
            self.max_payload_filter_keys,
        )