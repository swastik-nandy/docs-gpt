from __future__ import annotations

import json

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from enum import Enum
from time import time
from typing import Any
from typing import Callable
from typing import Iterator
from typing import Mapping
from uuid import uuid4


#--------------------EVENT TYPES--------------------

class GenerationStreamEventType(
    str,
    Enum,
):
    STREAM_STARTED = "stream.started"
    MODEL_STARTED = "model.started"
    ANSWER_DELTA = "answer.delta"
    MODEL_COMPLETED = "model.completed"
    ANSWER_REPLACE = "answer.replace"
    STREAM_COMPLETED = "stream.completed"
    STREAM_ERROR = "stream.error"


#--------------------EVENT--------------------

@dataclass(frozen=True, slots=True)
class GenerationStreamEvent:
    event_type: GenerationStreamEventType

    sequence: int = 0

    event_id: str = field(
        default_factory=lambda: str(
            uuid4()
        )
    )

    timestamp: float = field(
        default_factory=time
    )

    conversation_id: str | None = None
    turn_id: str | None = None

    text: str = ""

    payload: dict[str, Any] = field(
        default_factory=dict
    )

    final_state: Mapping[str, Any] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def with_sequence(
        self,
        sequence: int,
    ) -> "GenerationStreamEvent":
        return replace(
            self,
            sequence=sequence,
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "event_id": self.event_id,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "text": self.text,
            "payload": dict(
                self.payload
            ),
        }

    def to_sse(
        self,
    ) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
            default=str,
        )

        return (
            f"id: {self.event_id}\n"
            f"event: {self.event_type.value}\n"
            f"data: {payload}\n\n"
        )


#--------------------STREAM CONTEXT--------------------

StreamSink = Callable[
    [
        GenerationStreamEvent
    ],
    None,
]


_STREAM_SINK: ContextVar[
    StreamSink | None
] = ContextVar(
    "generation_stream_sink",
    default=None,
)


@contextmanager
def bind_stream_sink(
    sink: StreamSink,
) -> Iterator[None]:
    token = _STREAM_SINK.set(
        sink
    )

    try:
        yield

    finally:
        _STREAM_SINK.reset(
            token
        )


def emit_stream_event(
    event: GenerationStreamEvent,
) -> None:
    sink = _STREAM_SINK.get()

    if sink is None:
        return

    sink(
        event
    )


__all__ = [
    "GenerationStreamEvent",
    "GenerationStreamEventType",
    "StreamSink",
    "bind_stream_sink",
    "emit_stream_event",
]
