from __future__ import annotations

from dataclasses import dataclass

from rag.retrieval.contracts.config.models.context import (
    ContextDefaultsConfig,
    ContextLimitsConfig,
)
from rag.retrieval.contracts.config.models.memory import (
    MemoryDefaultsConfig,
    MemoryLimitsConfig,
)
from rag.retrieval.contracts.config.models.query import (
    QueryProcessingDefaultsConfig,
    QueryProcessingLimitsConfig,
    QueryTextLimitsConfig,
)
from rag.retrieval.contracts.config.models.request import (
    ContractSchemaConfig,
    RequestDefaultsConfig,
    RequestLimitsConfig,
    SourceConstraintDefaultsConfig,
)
from rag.retrieval.contracts.config.models.validation import ValidationConfig


@dataclass(frozen=True)
class RetrievalContractConfig:
    schema: ContractSchemaConfig
    request_defaults: RequestDefaultsConfig
    request_limits: RequestLimitsConfig
    source_constraints: SourceConstraintDefaultsConfig
    memory_defaults: MemoryDefaultsConfig
    memory_limits: MemoryLimitsConfig
    query_processing_defaults: QueryProcessingDefaultsConfig
    query_processing_limits: QueryProcessingLimitsConfig
    query_text_limits: QueryTextLimitsConfig
    context_defaults: ContextDefaultsConfig
    context_limits: ContextLimitsConfig
    validation: ValidationConfig

    @property
    def schema_version(self) -> str:
        return self.schema.version

    def validate(self) -> None:
        self.schema.validate()
        self.request_defaults.validate()
        self.request_limits.validate()
        self.source_constraints.validate()
        self.memory_defaults.validate()
        self.memory_limits.validate()
        self.query_processing_defaults.validate()
        self.query_processing_limits.validate()
        self.query_text_limits.validate()
        self.context_defaults.validate()
        self.context_limits.validate()
        self.validation.validate()

        self._validate_request_defaults_within_limits()
        self._validate_memory_defaults_within_limits()
        self._validate_query_processing_defaults_within_limits()
        self._validate_context_defaults_within_limits()

    def _validate_request_defaults_within_limits(self) -> None:
        if self.request_defaults.top_k > self.request_limits.max_top_k:
            raise ValueError(
                "request.defaults.top_k must be <= request.limits.max_top_k."
            )

    def _validate_memory_defaults_within_limits(self) -> None:
        if self.memory_defaults.max_chat_turns > self.memory_limits.max_chat_turns:
            raise ValueError(
                "memory.defaults.max_chat_turns must be <= "
                "memory.limits.max_chat_turns."
            )

        if self.memory_defaults.max_user_turns > self.memory_limits.max_user_turns:
            raise ValueError(
                "memory.defaults.max_user_turns must be <= "
                "memory.limits.max_user_turns."
            )

        if (
            self.memory_defaults.max_assistant_turns
            > self.memory_limits.max_assistant_turns
        ):
            raise ValueError(
                "memory.defaults.max_assistant_turns must be <= "
                "memory.limits.max_assistant_turns."
            )

        if (
            self.memory_defaults.preserve_recent_turns
            > self.memory_limits.max_preserve_recent_turns
        ):
            raise ValueError(
                "memory.defaults.preserve_recent_turns must be <= "
                "memory.limits.max_preserve_recent_turns."
            )

        if self.memory_defaults.memory_top_k > self.memory_limits.max_memory_top_k:
            raise ValueError(
                "memory.defaults.memory_top_k must be <= "
                "memory.limits.max_memory_top_k."
            )

        if self.memory_defaults.token_budget > self.memory_limits.max_token_budget:
            raise ValueError(
                "memory.defaults.token_budget must be <= "
                "memory.limits.max_token_budget."
            )

    def _validate_query_processing_defaults_within_limits(self) -> None:
        if (
            self.query_processing_defaults.max_rewritten_queries
            > self.query_processing_limits.max_rewritten_queries
        ):
            raise ValueError(
                "query_processing.defaults.max_rewritten_queries must be <= "
                "query_processing.limits.max_rewritten_queries."
            )

        if (
            self.query_processing_defaults.max_extracted_terms
            > self.query_processing_limits.max_extracted_terms
        ):
            raise ValueError(
                "query_processing.defaults.max_extracted_terms must be <= "
                "query_processing.limits.max_extracted_terms."
            )

    def _validate_context_defaults_within_limits(self) -> None:
        if (
            self.context_defaults.max_context_blocks
            > self.context_limits.max_context_blocks
        ):
            raise ValueError(
                "context.defaults.max_context_blocks must be <= "
                "context.limits.max_context_blocks."
            )

        if self.context_defaults.max_total_tokens > self.context_limits.max_total_tokens:
            raise ValueError(
                "context.defaults.max_total_tokens must be <= "
                "context.limits.max_total_tokens."
            )

        if self.context_defaults.max_block_tokens > self.context_limits.max_block_tokens:
            raise ValueError(
                "context.defaults.max_block_tokens must be <= "
                "context.limits.max_block_tokens."
            )

        if (
            self.context_defaults.reserve_tokens_for_answer
            > self.context_limits.max_reserve_tokens_for_answer
        ):
            raise ValueError(
                "context.defaults.reserve_tokens_for_answer must be <= "
                "context.limits.max_reserve_tokens_for_answer."
            )