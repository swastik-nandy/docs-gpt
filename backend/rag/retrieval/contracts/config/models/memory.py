from __future__ import annotations

from dataclasses import dataclass

from rag.retrieval.contracts.enums import MemoryScope
from rag.retrieval.contracts.config.models.shared import (
    validate_bool,
    validate_non_negative_int,
    validate_positive_int,
)


@dataclass(frozen=True)
class MemoryDefaultsConfig:
    enabled: bool
    scope: MemoryScope
    include_short_term: bool
    include_long_term: bool
    max_chat_turns: int
    max_user_turns: int
    max_assistant_turns: int
    preserve_recent_turns: int
    memory_top_k: int
    token_budget: int

    def validate(self) -> None:
        validate_bool("memory.defaults.enabled", self.enabled)
        validate_bool("memory.defaults.include_short_term", self.include_short_term)
        validate_bool("memory.defaults.include_long_term", self.include_long_term)

        if not isinstance(self.scope, MemoryScope):
            raise ValueError("memory.defaults.scope must be MemoryScope.")

        validate_non_negative_int(
            "memory.defaults.max_chat_turns",
            self.max_chat_turns,
        )
        validate_non_negative_int(
            "memory.defaults.max_user_turns",
            self.max_user_turns,
        )
        validate_non_negative_int(
            "memory.defaults.max_assistant_turns",
            self.max_assistant_turns,
        )
        validate_non_negative_int(
            "memory.defaults.preserve_recent_turns",
            self.preserve_recent_turns,
        )
        validate_non_negative_int(
            "memory.defaults.memory_top_k",
            self.memory_top_k,
        )
        validate_positive_int("memory.defaults.token_budget", self.token_budget)

        if self.enabled and self.scope == MemoryScope.NONE:
            raise ValueError(
                "memory.defaults.scope cannot be none when memory is enabled."
            )

        if self.preserve_recent_turns > self.max_chat_turns:
            raise ValueError(
                "memory.defaults.preserve_recent_turns must be <= "
                "memory.defaults.max_chat_turns."
            )

        if self.enabled and self.include_short_term and self.max_chat_turns == 0:
            raise ValueError(
                "memory.defaults.max_chat_turns must be greater than zero when "
                "short-term memory is enabled."
            )

        if self.enabled and self.include_long_term and self.memory_top_k == 0:
            raise ValueError(
                "memory.defaults.memory_top_k must be greater than zero when "
                "long-term memory is enabled."
            )


@dataclass(frozen=True)
class MemoryLimitsConfig:
    max_chat_turns: int
    max_user_turns: int
    max_assistant_turns: int
    max_preserve_recent_turns: int
    max_memory_top_k: int
    max_token_budget: int

    def validate(self) -> None:
        validate_non_negative_int(
            "memory.limits.max_chat_turns",
            self.max_chat_turns,
        )
        validate_non_negative_int(
            "memory.limits.max_user_turns",
            self.max_user_turns,
        )
        validate_non_negative_int(
            "memory.limits.max_assistant_turns",
            self.max_assistant_turns,
        )
        validate_non_negative_int(
            "memory.limits.max_preserve_recent_turns",
            self.max_preserve_recent_turns,
        )
        validate_non_negative_int(
            "memory.limits.max_memory_top_k",
            self.max_memory_top_k,
        )
        validate_positive_int(
            "memory.limits.max_token_budget",
            self.max_token_budget,
        )

        if self.max_preserve_recent_turns > self.max_chat_turns:
            raise ValueError(
                "memory.limits.max_preserve_recent_turns must be <= "
                "memory.limits.max_chat_turns."
            )