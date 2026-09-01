from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

from rag.retrieval.contracts.config import (
    RetrievalContractConfig,
    get_retrieval_contract_config,
)
from rag.retrieval.contracts.enums import (
    ChatRole,
    MemoryScope,
    QueryProcessingMode,
    RetrievalMode,
)
from rag.retrieval.contracts.primitives import (
    Payload,
    clean_tuple,
    validate_bool,
    validate_enum_instance,
    validate_mapping_size,
    validate_non_empty_str,
    validate_non_negative_int,
    validate_optional_clean_str,
    validate_optional_positive_int,
    validate_optional_sequence_limit,
    validate_positive_int,
)


@dataclass(frozen=True)
class TenantContext:
    tenant_id: str
    user_id: str | None = None
    session_id: str | None = None

    def validate(self) -> None:
        validate_non_empty_str("tenant.tenant_id", self.tenant_id)
        validate_optional_clean_str("tenant.user_id", self.user_id)
        validate_optional_clean_str("tenant.session_id", self.session_id)


@dataclass(frozen=True)
class SourceConstraints:
    source_id: str | None = None
    job_id: str | None = None
    parser_doc_id: str | None = None
    section_id: str | None = None
    ir_id: str | None = None
    heading_path: tuple[str, ...] = field(default_factory=tuple)
    payload_filters: Payload = field(default_factory=dict)

    @property
    def is_global(self) -> bool:
        return not any(
            (
                self.source_id,
                self.job_id,
                self.parser_doc_id,
                self.section_id,
                self.ir_id,
                self.heading_path,
                self.payload_filters,
            )
        )

    def normalized(self) -> Self:
        return type(self)(
            source_id=_clean_optional(self.source_id),
            job_id=_clean_optional(self.job_id),
            parser_doc_id=_clean_optional(self.parser_doc_id),
            section_id=_clean_optional(self.section_id),
            ir_id=_clean_optional(self.ir_id),
            heading_path=clean_tuple(self.heading_path),
            payload_filters=dict(self.payload_filters),
        )

    def validate(
        self,
        *,
        max_heading_path_depth: int | None,
        max_payload_filter_keys: int | None,
    ) -> None:
        validate_optional_clean_str("source_constraints.source_id", self.source_id)
        validate_optional_clean_str("source_constraints.job_id", self.job_id)
        validate_optional_clean_str(
            "source_constraints.parser_doc_id",
            self.parser_doc_id,
        )
        validate_optional_clean_str("source_constraints.section_id", self.section_id)
        validate_optional_clean_str("source_constraints.ir_id", self.ir_id)

        validate_optional_sequence_limit(
            "source_constraints.heading_path",
            self.heading_path,
            max_heading_path_depth,
        )
        validate_mapping_size(
            "source_constraints.payload_filters",
            self.payload_filters,
            max_payload_filter_keys,
        )

        for index, heading in enumerate(self.heading_path):
            validate_non_empty_str(
                f"source_constraints.heading_path[{index}]",
                heading,
            )


@dataclass(frozen=True)
class ChatTurn:
    role: str
    content: str
    turn_id: str | None = None
    created_at: str | None = None
    metadata: Payload = field(default_factory=dict)

    @property
    def role_enum(self) -> ChatRole:
        return ChatRole(self.role)

    def normalized(self) -> Self:
        return type(self)(
            role=str(self.role).strip(),
            content=self.content.strip(),
            turn_id=_clean_optional(self.turn_id),
            created_at=_clean_optional(self.created_at),
            metadata=dict(self.metadata),
        )

    def validate(
        self,
        *,
        max_metadata_keys: int | None,
    ) -> None:
        try:
            ChatRole(self.role)
        except ValueError as exc:
            allowed = ", ".join(role.value for role in ChatRole)
            raise ValueError(
                f"Unsupported chat role: {self.role!r}. Allowed values: {allowed}."
            ) from exc

        validate_non_empty_str("chat_history.content", self.content)
        validate_optional_clean_str("chat_history.turn_id", self.turn_id)
        validate_optional_clean_str("chat_history.created_at", self.created_at)
        validate_mapping_size(
            "chat_history.metadata",
            self.metadata,
            max_metadata_keys,
        )


@dataclass(frozen=True)
class MemoryPolicy:
    enabled: bool
    scope: MemoryScope
    include_short_term: bool
    include_long_term: bool
    max_chat_turns: int
    max_user_turns: int
    max_assistant_turns: int
    preserve_recent_turns: int
    memory_top_k: int
    token_budget: int | None = None

    @classmethod
    def from_config(cls, config: RetrievalContractConfig) -> Self:
        defaults = config.memory_defaults
        return cls(
            enabled=defaults.enabled,
            scope=defaults.scope,
            include_short_term=defaults.include_short_term,
            include_long_term=defaults.include_long_term,
            max_chat_turns=defaults.max_chat_turns,
            max_user_turns=defaults.max_user_turns,
            max_assistant_turns=defaults.max_assistant_turns,
            preserve_recent_turns=defaults.preserve_recent_turns,
            memory_top_k=defaults.memory_top_k,
            token_budget=defaults.token_budget,
        )

    def validate(self) -> None:
        validate_bool("memory_policy.enabled", self.enabled)
        validate_bool("memory_policy.include_short_term", self.include_short_term)
        validate_bool("memory_policy.include_long_term", self.include_long_term)
        validate_enum_instance("memory_policy.scope", self.scope, MemoryScope)

        if self.enabled and self.scope == MemoryScope.NONE:
            raise ValueError(
                "memory_policy.scope cannot be none when memory is enabled."
            )

        validate_non_negative_int(
            "memory_policy.max_chat_turns",
            self.max_chat_turns,
        )
        validate_non_negative_int(
            "memory_policy.max_user_turns",
            self.max_user_turns,
        )
        validate_non_negative_int(
            "memory_policy.max_assistant_turns",
            self.max_assistant_turns,
        )
        validate_non_negative_int(
            "memory_policy.preserve_recent_turns",
            self.preserve_recent_turns,
        )
        validate_non_negative_int("memory_policy.memory_top_k", self.memory_top_k)
        validate_optional_positive_int("memory_policy.token_budget", self.token_budget)

        if self.preserve_recent_turns > self.max_chat_turns:
            raise ValueError(
                "memory_policy.preserve_recent_turns must be <= "
                "memory_policy.max_chat_turns."
            )

        if self.enabled and self.include_short_term and self.max_chat_turns == 0:
            raise ValueError(
                "memory_policy.max_chat_turns must be greater than zero when "
                "short-term memory is enabled."
            )

        if self.enabled and self.include_long_term and self.memory_top_k == 0:
            raise ValueError(
                "memory_policy.memory_top_k must be greater than zero when "
                "long-term memory is enabled."
            )


@dataclass(frozen=True)
class QueryProcessingProfile:
    mode: QueryProcessingMode
    allow_query_rewrite: bool
    allow_filter_extraction: bool
    allow_intent_classification: bool
    allow_hypothetical_answer: bool
    max_rewritten_queries: int
    max_extracted_terms: int

    @classmethod
    def from_config(cls, config: RetrievalContractConfig) -> Self:
        defaults = config.query_processing_defaults
        return cls(
            mode=defaults.mode,
            allow_query_rewrite=defaults.allow_query_rewrite,
            allow_filter_extraction=defaults.allow_filter_extraction,
            allow_intent_classification=defaults.allow_intent_classification,
            allow_hypothetical_answer=defaults.allow_hypothetical_answer,
            max_rewritten_queries=defaults.max_rewritten_queries,
            max_extracted_terms=defaults.max_extracted_terms,
        )

    def validate(self) -> None:
        validate_enum_instance(
            "query_processing.mode",
            self.mode,
            QueryProcessingMode,
        )
        validate_bool(
            "query_processing.allow_query_rewrite",
            self.allow_query_rewrite,
        )
        validate_bool(
            "query_processing.allow_filter_extraction",
            self.allow_filter_extraction,
        )
        validate_bool(
            "query_processing.allow_intent_classification",
            self.allow_intent_classification,
        )
        validate_bool(
            "query_processing.allow_hypothetical_answer",
            self.allow_hypothetical_answer,
        )

        validate_non_negative_int(
            "query_processing.max_rewritten_queries",
            self.max_rewritten_queries,
        )
        validate_positive_int(
            "query_processing.max_extracted_terms",
            self.max_extracted_terms,
        )

        if not self.allow_query_rewrite and self.max_rewritten_queries > 0:
            raise ValueError(
                "query_processing.max_rewritten_queries must be zero when "
                "query rewrite is disabled."
            )


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    tenant: TenantContext
    mode: RetrievalMode
    top_k: int
    memory_policy: MemoryPolicy
    query_processing: QueryProcessingProfile
    request_id: str | None = None
    conversation_id: str | None = None
    source_constraints: SourceConstraints = field(default_factory=SourceConstraints)
    chat_history: tuple[ChatTurn, ...] = field(default_factory=tuple)
    metadata: Payload = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        query: str,
        tenant: TenantContext,
        mode: RetrievalMode | None = None,
        top_k: int | None = None,
        memory_policy: MemoryPolicy | None = None,
        query_processing: QueryProcessingProfile | None = None,
        request_id: str | None = None,
        conversation_id: str | None = None,
        source_constraints: SourceConstraints | None = None,
        chat_history: tuple[ChatTurn, ...] = (),
        metadata: Payload | None = None,
        config: RetrievalContractConfig | None = None,
    ) -> Self:
        resolved_config = config or get_retrieval_contract_config()

        request = cls(
            query=query,
            tenant=tenant,
            mode=mode or resolved_config.request_defaults.mode,
            top_k=top_k or resolved_config.request_defaults.top_k,
            memory_policy=memory_policy or MemoryPolicy.from_config(resolved_config),
            query_processing=(
                query_processing
                or QueryProcessingProfile.from_config(resolved_config)
            ),
            request_id=request_id,
            conversation_id=conversation_id,
            source_constraints=source_constraints or SourceConstraints(),
            chat_history=chat_history,
            metadata=metadata or {},
        )
        request.validate(config=resolved_config)
        return request

    @property
    def wants_memory(self) -> bool:
        return self.memory_policy.enabled and self.memory_policy.scope != MemoryScope.NONE

    @property
    def wants_short_term_memory(self) -> bool:
        return self.wants_memory and self.memory_policy.include_short_term

    @property
    def wants_long_term_memory(self) -> bool:
        return self.wants_memory and self.memory_policy.include_long_term

    def normalized(self) -> Self:
        return type(self)(
            query=self.query.strip(),
            tenant=self.tenant,
            mode=self.mode,
            top_k=self.top_k,
            memory_policy=self.memory_policy,
            query_processing=self.query_processing,
            request_id=_clean_optional(self.request_id),
            conversation_id=_clean_optional(self.conversation_id),
            source_constraints=self.source_constraints.normalized(),
            chat_history=tuple(turn.normalized() for turn in self.chat_history),
            metadata=dict(self.metadata),
        )

    def validate(
        self,
        *,
        config: RetrievalContractConfig | None = None,
    ) -> None:
        resolved_config = config or get_retrieval_contract_config()

        validate_non_empty_str("query", self.query)

        if not isinstance(self.tenant, TenantContext):
            raise ValueError("tenant must be TenantContext.")

        if not isinstance(self.memory_policy, MemoryPolicy):
            raise ValueError("memory_policy must be MemoryPolicy.")

        if not isinstance(self.query_processing, QueryProcessingProfile):
            raise ValueError("query_processing must be QueryProcessingProfile.")

        if not isinstance(self.source_constraints, SourceConstraints):
            raise ValueError("source_constraints must be SourceConstraints.")

        validate_enum_instance("mode", self.mode, RetrievalMode)
        validate_positive_int("top_k", self.top_k)

        if self.top_k > resolved_config.request_limits.max_top_k:
            raise ValueError(
                "top_k cannot be greater than "
                f"{resolved_config.request_limits.max_top_k}."
            )

        validate_optional_clean_str("request_id", self.request_id)
        validate_optional_clean_str("conversation_id", self.conversation_id)
        validate_mapping_size(
            "request.metadata",
            self.metadata,
            resolved_config.request_limits.max_metadata_keys,
        )

        self.tenant.validate()
        self.memory_policy.validate()
        self.query_processing.validate()
        self.source_constraints.validate(
            max_heading_path_depth=(
                resolved_config.source_constraints.max_heading_path_depth
            ),
            max_payload_filter_keys=(
                resolved_config.source_constraints.max_payload_filter_keys
            ),
        )

        validate_optional_sequence_limit(
            "chat_history",
            self.chat_history,
            resolved_config.request_limits.max_chat_history_turns,
        )

        for index, turn in enumerate(self.chat_history):
            if not isinstance(turn, ChatTurn):
                raise ValueError(f"chat_history[{index}] must be ChatTurn.")

            turn.validate(
                max_metadata_keys=resolved_config.request_limits.max_metadata_keys,
            )


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None

    cleaned = value.strip()
    return cleaned or None