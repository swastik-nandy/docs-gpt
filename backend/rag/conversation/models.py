from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Payload = dict[str, Any]
TurnRole = Literal["user", "assistant", "system", "tool"]


@dataclass(frozen=True)
class ConversationTurn:
    turn_id: str
    conversation_id: str
    tenant_id: str
    user_id: str | None
    turn_index: int
    role: TurnRole
    raw_text: str
    resolved_text: str | None = None
    entities_extracted: list[Payload] = field(default_factory=list)
    citations: list[Payload] = field(default_factory=list)
    model_used: str | None = None
    latency_ms: int | None = None
    hallucination_check_result: Payload = field(default_factory=dict)
    retrieval_trace_id: str | None = None
    metadata: Payload = field(default_factory=dict)

    @property
    def text_for_context(self) -> str:
        return self.resolved_text or self.raw_text

    def validate(self) -> None:
        _require_clean("turn_id", self.turn_id)
        _require_clean("conversation_id", self.conversation_id)
        _require_clean("tenant_id", self.tenant_id)
        _require_clean("raw_text", self.raw_text)

        if self.turn_index <= 0:
            raise ValueError("turn_index must be greater than zero.")

        if self.role not in {"user", "assistant", "system", "tool"}:
            raise ValueError(f"Unsupported turn role: {self.role!r}")

        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative.")


@dataclass(frozen=True)
class ConversationSummary:
    summary_id: str
    conversation_id: str
    tenant_id: str
    user_id: str | None
    covers_up_to_turn_index: int
    summary_text: str
    model_used: str | None = None
    metadata: Payload = field(default_factory=dict)

    def validate(self) -> None:
        _require_clean("summary_id", self.summary_id)
        _require_clean("conversation_id", self.conversation_id)
        _require_clean("tenant_id", self.tenant_id)
        _require_clean("summary_text", self.summary_text)

        if self.covers_up_to_turn_index <= 0:
            raise ValueError("covers_up_to_turn_index must be greater than zero.")


@dataclass(frozen=True)
class EntityMention:
    entity_text: str
    entity_type: str = "unknown"
    salience_rank: int = 1
    turn_index: int | None = None
    metadata: Payload = field(default_factory=dict)

    def normalized(self) -> EntityMention:
        return type(self)(
            entity_text=" ".join(self.entity_text.strip().split()),
            entity_type=(self.entity_type or "unknown").strip() or "unknown",
            salience_rank=int(self.salience_rank),
            turn_index=self.turn_index,
            metadata=dict(self.metadata),
        )

    def validate(self) -> None:
        _require_clean("entity_text", self.entity_text)
        _require_clean("entity_type", self.entity_type)

        if self.salience_rank <= 0:
            raise ValueError("salience_rank must be greater than zero.")

        if self.turn_index is not None and self.turn_index <= 0:
            raise ValueError("turn_index must be greater than zero when provided.")


@dataclass(frozen=True)
class WorkingMemoryItem:
    key: str
    value: Payload
    metadata: Payload = field(default_factory=dict)

    def validate(self) -> None:
        _require_clean("key", self.key)


@dataclass(frozen=True)
class ConversationContext:
    conversation_id: str
    tenant_id: str
    user_id: str | None
    recent_turns: tuple[ConversationTurn, ...] = field(default_factory=tuple)
    latest_summary: ConversationSummary | None = None
    recent_entities: tuple[EntityMention, ...] = field(default_factory=tuple)
    working_memory: tuple[WorkingMemoryItem, ...] = field(default_factory=tuple)

    @property
    def has_recent_turns(self) -> bool:
        return bool(self.recent_turns)

    @property
    def has_summary(self) -> bool:
        return self.latest_summary is not None

    @property
    def has_recent_entities(self) -> bool:
        return bool(self.recent_entities)

    def validate(self) -> None:
        _require_clean("conversation_id", self.conversation_id)
        _require_clean("tenant_id", self.tenant_id)

        for turn in self.recent_turns:
            turn.validate()

        if self.latest_summary is not None:
            self.latest_summary.validate()

        for entity in self.recent_entities:
            entity.validate()

        for item in self.working_memory:
            item.validate()


@dataclass(frozen=True)
class ReferenceResolutionResult:
    raw_query: str
    resolved_query: str
    needed_resolution: bool
    resolved_entity: str | None = None
    strategy: str = "none"
    confidence: float = 1.0
    metadata: Payload = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.raw_query.strip() != self.resolved_query.strip()

    def validate(self) -> None:
        _require_clean("raw_query", self.raw_query)
        _require_clean("resolved_query", self.resolved_query)

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")


def _require_clean(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty.")
