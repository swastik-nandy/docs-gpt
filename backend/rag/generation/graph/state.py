from __future__ import annotations

from typing import Any
from typing import NotRequired
from typing import TypedDict

from rag.generation.llm.schema import ChatMessage
from rag.generation.memory.schema import RetrievedMemory


class RetrievalState(TypedDict, total=False):
    query: str
    rewritten_query: str
    candidates: list[dict[str, Any]]
    selected_context: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    warnings: list[str]


class ContextBundle(TypedDict, total=False):
    retrieved_documents: list[dict[str, Any]]
    retrieved_memory: list[dict[str, Any]]
    working_memory: list[dict[str, Any]]
    conversation_summary: str | None
    provenance_map: dict[str, dict[str, Any]]


class GroundingResult(TypedDict, total=False):
    passed: bool
    score: float
    reasons: list[str]


class ConversationState(TypedDict):
    tenant_id: str
    user_id: str
    conversation_id: str
    turn_id: str
    current_user_message: str

    raw_user_message: NotRequired[str]
    retrieval_query: NotRequired[str]
    turn_plan: NotRequired[dict[str, Any]]

    recent_turns: NotRequired[
        list[ChatMessage]
    ]

    conversation_summary: NotRequired[
        str | None
    ]

    retrieved_memory: NotRequired[
        list[RetrievedMemory]
    ]

    retrieval_state: NotRequired[
        RetrievalState
    ]

    context_bundle: NotRequired[
        ContextBundle
    ]

    draft_answer: NotRequired[str]
    final_answer: NotRequired[str]

    generation_metadata: NotRequired[
        dict[str, Any]
    ]

    assistant_turn_id: NotRequired[str]

    grounding_result: NotRequired[
        GroundingResult
    ]

    written_memory_ids: NotRequired[
        list[str]
    ]

    skipped_memory_ids: NotRequired[
        list[str]
    ]

    memory_write_warnings: NotRequired[
        list[str]
    ]

    model_role_overrides: NotRequired[
        dict[str, str]
    ]

    disable_long_term_memory: NotRequired[
        bool
    ]

    disable_memory_write: NotRequired[
        bool
    ]

    _stream_sink: NotRequired[Any]
