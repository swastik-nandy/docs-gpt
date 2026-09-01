from __future__ import annotations

import time
from typing import Any

from rag.retrieval.contracts.enums import ConfidenceBand, QueryIntent
from rag.retrieval.contracts.query import NormalizedQuery
from rag.retrieval.contracts.request import RetrievalRequest
from rag.retrieval.contracts.response import RetrievalDecision
from rag.retrieval.graph.state import (
    RetrievalGraphError,
    RetrievalGraphOutput,
    RetrievalGraphStage,
    RetrievalGraphState,
    to_graph_output,
)


Payload = dict[str, Any]


def finalize_graph_output(
    state: RetrievalGraphState,
    *,
    started: float,
    include_debug_metadata: bool,
) -> RetrievalGraphOutput:
    """
    Convert graph state into a stable external output.

    Successful graph paths should already contain normalized_query and decision.
    Early failures may not. This function patches safe fallback values so callers
    always receive a valid RetrievalGraphOutput.
    """

    completed_state = dict(state)

    metadata = dict(completed_state.get("metadata", {}))
    metadata["retrieval_graph_seconds"] = round(time.perf_counter() - started, 6)

    if include_debug_metadata:
        metadata["retrieval_graph_debug"] = _debug_metadata(completed_state)

    completed_state["metadata"] = metadata

    if "normalized_query" not in completed_state:
        completed_state["normalized_query"] = _fallback_normalized_query(
            RetrievalGraphState(completed_state)
        )

    if "decision" not in completed_state:
        completed_state["decision"] = _fallback_decision(
            RetrievalGraphState(completed_state)
        )

    return to_graph_output(RetrievalGraphState(completed_state))


def create_runtime_error_state(
    *,
    base_state: RetrievalGraphState,
    exc: Exception,
) -> RetrievalGraphState:
    """
    Create a failed graph state when LangGraph itself raises.

    Node-level failures should already be represented through graph node errors.
    """

    next_state = dict(base_state)
    existing_errors = list(next_state.get("errors", []))

    next_state["current_stage"] = RetrievalGraphStage.FAILED
    next_state["errors"] = existing_errors + [
        RetrievalGraphError(
            stage=RetrievalGraphStage.FAILED,
            message=str(exc),
            error_type=type(exc).__name__,
            metadata={
                "runtime": "langgraph",
            },
        )
    ]

    return RetrievalGraphState(next_state)


def _fallback_normalized_query(state: RetrievalGraphState) -> NormalizedQuery:
    request = _require_request_for_output(state)
    query = request.query.strip()

    normalized_query = NormalizedQuery(
        original_query=query,
        normalized_query=query,
        intent=QueryIntent.UNKNOWN,
        metadata={
            "fallback": True,
            "source": "graph_output",
        },
    )
    normalized_query.validate()
    return normalized_query


def _fallback_decision(state: RetrievalGraphState) -> RetrievalDecision:
    errors = list(state.get("errors", ()))

    reason = "Retrieval failed before a decision could be produced."

    if errors:
        last_error = errors[-1]
        reason = f"{last_error.error_type}: {last_error.message}"

    decision = RetrievalDecision(
        is_sufficient=False,
        confidence=ConfidenceBand.VERY_LOW,
        confidence_score=0.0,
        reason=reason,
        should_rewrite=False,
        metadata={
            "fallback": True,
            "source": "graph_output",
            "warnings": _warning_messages(state),
        },
    )
    decision.validate()
    return decision


def _require_request_for_output(state: RetrievalGraphState) -> RetrievalRequest:
    request = state.get("request")

    if request is None:
        raise ValueError("Cannot finalize retrieval graph output without request.")

    return request


def _warning_messages(state: RetrievalGraphState) -> list[str]:
    return [warning.message for warning in state.get("warnings", [])]


def _debug_metadata(state: RetrievalGraphState) -> Payload:
    return {
        "current_stage": _stage_value(state.get("current_stage")),
        "retry_count": int(state.get("retry_count", 0)),
        "rewrite_count": int(state.get("rewrite_count", 0)),
        "dense_count": len(tuple(state.get("dense_candidates", ()))),
        "sparse_count": len(tuple(state.get("sparse_candidates", ()))),
        "corrective_count": len(tuple(state.get("corrective_candidates", ()))),
        "fused_count": len(tuple(state.get("fused_candidates", ()))),
        "expanded_count": len(tuple(state.get("expanded_candidates", ()))),
        "reranked_count": len(tuple(state.get("reranked_candidates", ()))),
        "context_count": len(tuple(state.get("context_blocks", ()))),
        "error_count": len(list(state.get("errors", []))),
        "warning_count": len(list(state.get("warnings", []))),
    }


def _stage_value(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, RetrievalGraphStage):
        return value.value

    return str(value)