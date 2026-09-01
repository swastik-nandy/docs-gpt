from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Mapping


MemoryFactType = Literal[
    "preference",
    "project_context",
    "technical_context",
    "user_profile",
    "conversation_fact",
]


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    tenant_id: str
    user_id: str
    text: str
    vector: list[float]
    fact_type: MemoryFactType
    conversation_id: str | None = None
    source_turn_id: str | None = None
    confidence: float = 0.8
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id is required.")
        if not self.user_id.strip():
            raise ValueError("user_id is required.")
        if not self.text.strip():
            raise ValueError("text is required.")
        if not self.vector:
            raise ValueError("vector is required.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class RetrievedMemory:
    memory_id: str
    tenant_id: str
    user_id: str
    text: str
    fact_type: str
    score: float
    confidence: float
    conversation_id: str | None
    source_turn_id: str | None
    created_at: datetime | None
    last_confirmed_at: datetime | None
    metadata: Mapping[str, Any]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
