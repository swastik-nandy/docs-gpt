from __future__ import annotations

from dataclasses import dataclass

from rag.retrieval.contracts.config.models.shared import (
    validate_bool,
    validate_non_negative_int,
    validate_positive_int,
)


@dataclass(frozen=True)
class ContextDefaultsConfig:
    max_context_blocks: int
    max_total_tokens: int
    max_block_tokens: int
    reserve_tokens_for_answer: int
    deduplicate_by_chunk_id: bool
    deduplicate_by_text_hash: bool

    def validate(self) -> None:
        validate_positive_int(
            "context.defaults.max_context_blocks",
            self.max_context_blocks,
        )
        validate_positive_int(
            "context.defaults.max_total_tokens",
            self.max_total_tokens,
        )
        validate_positive_int(
            "context.defaults.max_block_tokens",
            self.max_block_tokens,
        )
        validate_non_negative_int(
            "context.defaults.reserve_tokens_for_answer",
            self.reserve_tokens_for_answer,
        )
        validate_bool(
            "context.defaults.deduplicate_by_chunk_id",
            self.deduplicate_by_chunk_id,
        )
        validate_bool(
            "context.defaults.deduplicate_by_text_hash",
            self.deduplicate_by_text_hash,
        )

        if self.max_block_tokens > self.max_total_tokens:
            raise ValueError(
                "context.defaults.max_block_tokens must be <= "
                "context.defaults.max_total_tokens."
            )

        if self.reserve_tokens_for_answer >= self.max_total_tokens:
            raise ValueError(
                "context.defaults.reserve_tokens_for_answer must be less than "
                "context.defaults.max_total_tokens."
            )


@dataclass(frozen=True)
class ContextLimitsConfig:
    max_context_blocks: int
    max_total_tokens: int
    max_block_tokens: int
    max_reserve_tokens_for_answer: int

    def validate(self) -> None:
        validate_positive_int(
            "context.limits.max_context_blocks",
            self.max_context_blocks,
        )
        validate_positive_int(
            "context.limits.max_total_tokens",
            self.max_total_tokens,
        )
        validate_positive_int(
            "context.limits.max_block_tokens",
            self.max_block_tokens,
        )
        validate_non_negative_int(
            "context.limits.max_reserve_tokens_for_answer",
            self.max_reserve_tokens_for_answer,
        )

        if self.max_block_tokens > self.max_total_tokens:
            raise ValueError(
                "context.limits.max_block_tokens must be <= "
                "context.limits.max_total_tokens."
            )

        if self.max_reserve_tokens_for_answer >= self.max_total_tokens:
            raise ValueError(
                "context.limits.max_reserve_tokens_for_answer must be less than "
                "context.limits.max_total_tokens."
            )