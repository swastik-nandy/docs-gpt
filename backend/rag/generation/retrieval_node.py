from __future__ import annotations

from typing import Any, Mapping

from rag.generation.conversation_io import build_generation_state_with_retrieval_output
from rag.generation.graph.state import ConversationState
from rag.retrieval.contracts.request import RetrievalRequest, SourceConstraints, TenantContext
from rag.retrieval.service import RetrievalService, RetrievalServiceConfig

Payload = dict[str, Any]


def build_generation_retrieval_node(
    *,
    service: RetrievalService | None = None,
    top_k: int = 5,
):
    """
    Generation-to-retrieval adapter.

    This node does not choose retrieval backends, rerankers, expansion strategy,
    or heavy tools. RetrievalService owns the adaptive LangGraph runtime.
    """
    resolved_service = service or RetrievalService(
        config=RetrievalServiceConfig(
            include_debug_metadata=True,
            enable_stateful_resolution=True,
            persist_retrieval_trace=False,
        )
    )

    async def retrieval_node(state: ConversationState) -> dict[str, Any]:
        tenant_id = _required_state_str(state, "tenant_id")
        user_id = _required_state_str(state, "user_id")
        conversation_id = _required_state_str(state, "conversation_id")

        raw_query = _required_state_str(state, "current_user_message")
        planned_query = _clean_optional(state.get("retrieval_query"))
        retrieval_query = planned_query or raw_query

        requested_top_k = int(state.get("top_k") or top_k)
        effective_top_k = max(1, requested_top_k)

        request = RetrievalRequest.create(
            query=retrieval_query,
            tenant=TenantContext(
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=_optional_state_str(state, "session_id"),
            ),
            conversation_id=conversation_id,
            source_constraints=_source_constraints_from_state(state),
            top_k=effective_top_k,
            metadata={
                "source": "generation_retrieval_node",
                "turn_id": state.get("turn_id"),
                "raw_user_message": state.get("raw_user_message") or raw_query,
                "current_user_message": raw_query,
                "planned_retrieval_query": planned_query,
                "retrieval_query": retrieval_query,
                "effective_top_k": effective_top_k,
                "turn_plan": state.get("turn_plan"),
            },
        )

        retrieval_output = await resolved_service.retrieve(
            request,
            metadata={
                "source": "generation_retrieval_node",
                "turn_id": state.get("turn_id"),
                "raw_user_message": state.get("raw_user_message") or raw_query,
                "current_user_message": raw_query,
                "planned_retrieval_query": planned_query,
                "retrieval_query": retrieval_query,
                "effective_top_k": effective_top_k,
            },
        )

        return build_generation_state_with_retrieval_output(
            state=state,
            retrieval_output=retrieval_output,
        )

    return retrieval_node


def _source_constraints_from_state(state: Mapping[str, Any]) -> SourceConstraints:
    raw = state.get("retrieval_source_constraints") or {}
    if not isinstance(raw, Mapping):
        raw = {}

    return SourceConstraints(
        job_id=_clean_optional(raw.get("job_id")),
        source_id=_clean_optional(raw.get("source_id")),
        parser_doc_id=_clean_optional(raw.get("parser_doc_id")),
        section_id=_clean_optional(raw.get("section_id")),
        ir_id=_clean_optional(raw.get("ir_id")),
        payload_filters=(
            dict(raw.get("payload_filters"))
            if isinstance(raw.get("payload_filters"), Mapping)
            else {}
        ),
    )


def _required_state_str(state: Mapping[str, Any], key: str) -> str:
    value = state.get(key)
    if value is None:
        raise ValueError(f"generation state is missing required key: {key}")

    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"generation state key cannot be empty: {key}")

    return cleaned


def _optional_state_str(state: Mapping[str, Any], key: str) -> str | None:
    return _clean_optional(state.get(key))


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None

    cleaned = str(value).strip()
    return cleaned or None
