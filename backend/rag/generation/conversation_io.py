from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from rag.conversation.models import ConversationTurn
from rag.conversation.repository import PostgresConversationRepository
from rag.generation.graph.state import ConversationState
from rag.generation.llm.schema import ChatMessage


Payload = dict[str, Any]


async def build_generation_state_from_turn(
    *,
    repository: PostgresConversationRepository,
    conversation_id: str,
    tenant_id: str,
    user_id: str,
    user_turn_id: str,
    current_user_message: str,
    turn_window_limit: int = 8,
) -> ConversationState:
    context = await repository.fetch_context(
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        user_id=user_id,
        turn_limit=turn_window_limit,
        entity_limit=8,
    )

    recent_turns = [
        _chat_message_from_turn(turn)
        for turn in context.recent_turns
        if turn.turn_id != user_turn_id
        and turn.role in {"user", "assistant", "system", "tool"}
    ]

    state: ConversationState = {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "conversation_id": conversation_id,
        "turn_id": user_turn_id,
        "current_user_message": current_user_message,
        "raw_user_message": current_user_message,
        "recent_turns": recent_turns,
        "conversation_summary": (
            context.latest_summary.summary_text
            if context.latest_summary is not None
            else None
        ),
    }

    return state


async def persist_assistant_turn_from_generation_state(
    *,
    repository: PostgresConversationRepository,
    state: ConversationState,
    metadata: Payload | None = None,
) -> ConversationTurn:
    answer = (
        state.get("final_answer")
        or state.get("draft_answer")
        or ""
    ).strip()

    if not answer:
        raise ValueError("generation state does not contain final_answer or draft_answer.")

    generation_metadata = dict(state.get("generation_metadata") or {})
    grounding_result = dict(state.get("grounding_result") or {})
    retrieval_state = dict(state.get("retrieval_state") or {})

    citations = _safe_sequence_of_mappings(
        retrieval_state.get("citations") or []
    )

    assistant_retrieval_trace_id = _optional_text(
        retrieval_state.get("retrieval_trace_id")
        or (metadata or {}).get("retrieval_trace_id")
    )

    model_used = _model_used(generation_metadata)

    turn = await repository.insert_turn(
        conversation_id=state["conversation_id"],
        tenant_id=state["tenant_id"],
        user_id=state["user_id"],
        role="assistant",
        raw_text=answer,
        citations=citations,
        model_used=model_used,
        retrieval_trace_id=assistant_retrieval_trace_id,
        hallucination_check_result=grounding_result,
        metadata={
            "source": "generation_conversation_io",
            "generation_metadata": generation_metadata,
            **(metadata or {}),
        },
    )

    return turn


def build_generation_state_with_retrieval_output(
    *,
    state: ConversationState,
    retrieval_output: Mapping[str, Any],
) -> ConversationState:
    metadata = dict(retrieval_output.get("metadata") or {})
    stateful = dict(metadata.get("stateful_retrieval") or {})

    context_blocks = list(retrieval_output.get("context_blocks") or [])
    reranked_candidates = list(retrieval_output.get("reranked_candidates") or [])
    warnings = list(retrieval_output.get("warnings") or [])

    selected_context = [
        _context_block_payload(block)
        for block in context_blocks
    ]

    citations = [
        _context_block_citation(block)
        for block in context_blocks
    ]

    updated: ConversationState = dict(state)  # type: ignore[assignment]
    raw_user_message = str(
        state.get("raw_user_message")
        or stateful.get("raw_query")
        or state.get("current_user_message")
        or ""
    )

    updated["raw_user_message"] = raw_user_message
    updated["current_user_message"] = str(
        stateful.get("resolved_query")
        or state.get("current_user_message")
        or ""
    )
    updated["retrieval_state"] = {
        "query": str(stateful.get("raw_query") or ""),
        "rewritten_query": str(stateful.get("resolved_query") or ""),
        "retrieval_trace_id": metadata.get("retrieval_trace_id"),
        "stateful_retrieval": stateful,
        "retrieval_metadata": metadata,
        "candidates": [
            _candidate_payload(candidate)
            for candidate in reranked_candidates
        ],
        "selected_context": selected_context,
        "citations": citations,
        "warnings": [str(item) for item in warnings],
    }

    return updated


def _chat_message_from_turn(turn: ConversationTurn) -> ChatMessage:
    role = turn.role
    if role not in {"system", "user", "assistant", "tool"}:
        role = "user"

    return ChatMessage(
        role=role,  # type: ignore[arg-type]
        content=turn.text_for_context,
    )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None

    cleaned = str(value).strip()
    return cleaned or None


def _model_used(generation_metadata: Mapping[str, Any]) -> str | None:
    provider = str(generation_metadata.get("provider") or "").strip()
    model = str(generation_metadata.get("model") or "").strip()

    if provider and model:
        return f"{provider}:{model}"

    return model or provider or None


def _safe_sequence_of_mappings(value: Any) -> list[Payload]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []

    result: list[Payload] = []
    for item in value:
        if isinstance(item, Mapping):
            result.append(dict(item))

    return result


def _context_block_payload(block: Any) -> Payload:
    source_name = _context_block_source_name(
        block
    )

    return {
        "context_id": source_name,
        "source_name": source_name,
        "citation_label": source_name,
        "source_title": getattr(
            block,
            "source_title",
            None,
        ),
        "block_id": getattr(
            block,
            "block_id",
            None,
        ),
        "source_id": getattr(
            block,
            "source_id",
            None,
        ),
        "chunk_id": getattr(
            block,
            "chunk_id",
            None,
        ),
        "heading": getattr(
            block,
            "heading",
            None,
        ),
        "text": getattr(
            block,
            "text",
            "",
        ),
        "score": getattr(
            block,
            "score",
            None,
        ),
        "payload": _safe_mapping(
            getattr(
                block,
                "payload",
                None,
            )
        ),
        "metadata": _safe_mapping(
            getattr(
                block,
                "metadata",
                None,
            )
        ),
    }


def _context_block_citation(block: Any) -> Payload:
    source_name = _context_block_source_name(
        block
    )

    return {
        "source_name": source_name,
        "citation_label": source_name,
        "source_title": getattr(
            block,
            "source_title",
            None,
        ),
        "block_id": getattr(
            block,
            "block_id",
            None,
        ),
        "source_id": getattr(
            block,
            "source_id",
            None,
        ),
        "chunk_id": getattr(
            block,
            "chunk_id",
            None,
        ),
        "heading": getattr(
            block,
            "heading",
            None,
        ),
    }


def _context_block_source_name(
    block: Any,
) -> str:
    payload = _safe_mapping(
        getattr(
            block,
            "payload",
            None,
        )
    )

    for key in (
        "file_name",
        "filename",
        "original_file_name",
        "original_filename",
        "source_name",
    ):
        value = _clean_text(
            payload.get(
                key
            )
        )

        if value:
            name = Path(
                value
            ).name

            return name or value

    for key in (
        "source",
        "source_path",
        "file_path",
    ):
        value = _clean_text(
            payload.get(
                key
            )
        )

        if not value:
            continue

        # Only treat this as a file source when it actually
        # resembles a path or filename. A plain source slug
        # should remain an internal source identifier.
        if (
            "/"
            not in value
            and "\\"
            not in value
            and not Path(
                value
            ).suffix
        ):
            continue

        name = Path(
            value
        ).name

        if name:
            return name

    source_title = _clean_text(
        getattr(
            block,
            "source_title",
            None,
        )
    )

    if source_title:
        return source_title

    source_id = _clean_text(
        getattr(
            block,
            "source_id",
            None,
        )
    )

    if source_id:
        return source_id

    heading = _clean_text(
        getattr(
            block,
            "heading",
            None,
        )
    )

    if heading:
        return heading

    return str(
        getattr(
            block,
            "block_id",
            "unknown-source",
        )
    )


def _clean_text(
    value: Any,
) -> str | None:
    if value is None:
        return None

    cleaned = str(
        value
    ).strip()

    return cleaned or None


def _candidate_payload(candidate: Any) -> Payload:
    return {
        "point_id": getattr(candidate, "point_id", None),
        "source_id": getattr(candidate, "source_id", None),
        "chunk_id": getattr(candidate, "chunk_id", None),
        "score": getattr(candidate, "score", None),
        "rerank_score": getattr(candidate, "rerank_score", None),
        "rank": getattr(candidate, "rank", None),
    }


def _safe_mapping(value: Any) -> Payload:
    if isinstance(value, Mapping):
        return dict(value)

    return {}
