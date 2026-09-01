from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Literal
from typing import Mapping
from typing import Sequence


#--------------------ROLES--------------------

ChatRole = Literal[
    "system",
    "user",
    "assistant",
    "tool",
]

ModelRole = Literal[
    "generation_primary",
    "generation_fallback",
    "query_rewrite",
    "turn_planning",
    "memory_extraction",
    "groundedness_check",
]

ModelStreamEventType = Literal[
    "started",
    "delta",
    "completed",
]


#--------------------MESSAGES--------------------

@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: ChatRole
    content: str


#--------------------REQUEST--------------------

@dataclass(frozen=True, slots=True)
class ModelRequest:
    role: ModelRole
    messages: Sequence[ChatMessage]
    system: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    response_mime_type: str | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


#--------------------USAGE--------------------

@dataclass(frozen=True, slots=True)
class ModelUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


#--------------------FULL RESPONSE--------------------

@dataclass(frozen=True, slots=True)
class ModelResponse:
    text: str
    provider: str
    model: str
    role: ModelRole

    usage: ModelUsage = field(
        default_factory=ModelUsage
    )

    raw_finish_reason: str | None = None
    raw: Any | None = None


#--------------------PROVIDER STREAM--------------------

@dataclass(frozen=True, slots=True)
class ProviderStreamChunk:
    text: str = ""

    usage: ModelUsage | None = None

    raw_finish_reason: str | None = None

    raw: Any | None = None


#--------------------ROUTED STREAM--------------------

@dataclass(frozen=True, slots=True)
class ModelStreamEvent:
    event_type: ModelStreamEventType

    provider: str
    model: str
    role: ModelRole
    slot_name: str

    fallback_used: bool

    text: str = ""

    usage: ModelUsage = field(
        default_factory=ModelUsage
    )

    raw_finish_reason: str | None = None


#--------------------ERRORS--------------------

class ModelRouterError(
    RuntimeError
):
    pass


class ModelProviderError(
    RuntimeError
):
    pass


class ModelStreamInterruptedError(
    ModelRouterError
):
    pass


__all__ = [
    "ChatMessage",
    "ChatRole",
    "ModelProviderError",
    "ModelRequest",
    "ModelResponse",
    "ModelRole",
    "ModelRouterError",
    "ModelStreamEvent",
    "ModelStreamEventType",
    "ModelStreamInterruptedError",
    "ModelUsage",
    "ProviderStreamChunk",
]
