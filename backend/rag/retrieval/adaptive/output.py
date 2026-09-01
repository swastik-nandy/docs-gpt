from __future__ import annotations

from typing import Any

from rag.retrieval.adaptive.actions import RetrievalAction
from rag.retrieval.contracts.enums import ConfidenceBand
from rag.retrieval.contracts.response import RetrievalDecision
from rag.retrieval.graph.state import (
    RetrievalGraphError,
    RetrievalGraphOutput,
    RetrievalGraphStage,
    RetrievalTraceEvent,
    RetrievalGraphWarning,
)


Payload = dict[str, Any]


def adaptive_result_to_graph_output(
    result: Any,
    *,
    metadata: Payload | None = None,
) -> RetrievalGraphOutput:
    state = result.state

    decision = _build_decision(result)

    output_metadata: Payload = {
        "backend": "adaptive",
        "action_count": len(state.action_records),
        "snapshot_count": len(state.snapshots),
        "confidence": state.confidence.value,
        "confidence_reason": state.confidence.reason,
        "query_signals": _signals_to_metadata(state.query_signals),
        "actions": [_action_to_metadata(record) for record in state.action_records],
    }

    if metadata:
        output_metadata.update(metadata)

    return RetrievalGraphOutput(
        request=state.request,
        normalized_query=state.normalized_query,
        decision=decision,
        context_blocks=tuple(state.context_blocks),
        reranked_candidates=tuple(state.reranked_candidates),
        trace=_build_trace(state),
        warnings=_build_warnings(state),
        errors=_build_errors(state),
        metadata=output_metadata,
    )


def _build_decision(result: Any) -> RetrievalDecision:
    state = result.state
    confidence_value = float(state.confidence.value)

    has_context = bool(state.context_blocks)
    has_errors = bool(state.errors)

    final_decision = getattr(result, "final_decision", None)
    reason = (
        getattr(final_decision, "reason", None)
        or state.confidence.reason
        or "adaptive retrieval completed"
    )

    return RetrievalDecision(
        is_sufficient=has_context and not has_errors,
        confidence=_confidence_band(confidence_value),
        confidence_score=confidence_value,
        reason=reason,
        should_rewrite=False,
        metadata={
            "backend": "adaptive",
            "final_action": (
                getattr(getattr(final_decision, "action", None), "value", None)
                or str(getattr(final_decision, "action", ""))
                or None
            ),
            "exact_coverage": state.confidence.exact_coverage,
            "source_coverage": state.confidence.source_coverage,
            "score_gap": state.confidence.score_gap,
        },
    )


def _confidence_band(value: float) -> ConfidenceBand:
    if value >= 0.80:
        return ConfidenceBand.HIGH

    if value >= 0.55:
        return ConfidenceBand.MEDIUM

    if value >= 0.25:
        return ConfidenceBand.LOW

    return ConfidenceBand.VERY_LOW


def _build_trace(state: Any) -> list[RetrievalTraceEvent]:
    trace: list[RetrievalTraceEvent] = []

    for record in state.action_records:
        action = record.action
        trace.append(
            RetrievalTraceEvent(
                stage=_stage_for_action(action),
                message=f"Adaptive action {action.value}: {record.status.value}",
                metadata=_action_to_metadata(record),
            )
        )

    return trace


def _build_warnings(state: Any) -> list[RetrievalGraphWarning]:
    warnings: list[RetrievalGraphWarning] = []

    for warning in getattr(state, "warnings", ()) or ():
        warnings.append(
            RetrievalGraphWarning(
                stage=RetrievalGraphStage.COMPLETED,
                message=str(warning),
                metadata={"backend": "adaptive"},
            )
        )

    return warnings


def _build_errors(state: Any) -> list[RetrievalGraphError]:
    errors: list[RetrievalGraphError] = []

    for error in getattr(state, "errors", ()) or ():
        errors.append(
            RetrievalGraphError(
                stage=RetrievalGraphStage.FAILED,
                message=str(error),
                error_type="AdaptiveRetrievalError",
                metadata={"backend": "adaptive"},
            )
        )

    return errors


def _stage_for_action(action: RetrievalAction) -> RetrievalGraphStage:
    mapping = {
        RetrievalAction.MEMORY_LOOKUP: RetrievalGraphStage.MEMORY_PLANNED,
        RetrievalAction.SPARSE_SEARCH: RetrievalGraphStage.SPARSE_RETRIEVED,
        RetrievalAction.DENSE_SEARCH: RetrievalGraphStage.DENSE_RETRIEVED,
        RetrievalAction.FUSE: RetrievalGraphStage.HYBRID_FUSED,
        RetrievalAction.SCORE_RERANK: RetrievalGraphStage.RERANKED,
        RetrievalAction.CE_RERANK: RetrievalGraphStage.RERANKED,
        RetrievalAction.LINKED_EXPAND: RetrievalGraphStage.STRUCTURE_EXPANDED,
        RetrievalAction.ESCALATE_K: RetrievalGraphStage.CORRECTIVE_RETRIEVED,
        RetrievalAction.ESCALATE_DENSE: RetrievalGraphStage.DENSE_RETRIEVED,
        RetrievalAction.VERIFY_EVIDENCE: RetrievalGraphStage.GRADED,
        RetrievalAction.PACK_CONTEXT: RetrievalGraphStage.CONTEXT_PACKED,
        RetrievalAction.STOP: RetrievalGraphStage.COMPLETED,
        RetrievalAction.FALLBACK_NO_ANSWER: RetrievalGraphStage.FAILED,
    }

    return mapping.get(action, RetrievalGraphStage.COMPLETED)


def _action_to_metadata(record: Any) -> Payload:
    return {
        "action": record.action.value,
        "status": record.status.value,
        "hop": record.hop,
        "reason": record.reason,
        "latency_ms": record.latency_ms,
        "input_count": record.input_count,
        "output_count": record.output_count,
        "materially_changed": record.materially_changed,
        "error": record.error,
        **dict(record.metadata or {}),
    }


def _signals_to_metadata(signals: Any) -> Payload:
    return {
        "normalized_query": signals.normalized_query,
        "token_count": signals.token_count,
        "lexical_specificity": signals.lexical_specificity,
        "is_exactish": signals.is_exactish,
        "is_structural_or_multi_hop": signals.is_structural_or_multi_hop,
        "is_chat_dependent": signals.is_chat_dependent,
        "likely_needs_memory": signals.likely_needs_memory,
        "has_chat_history": signals.has_chat_history,
        "has_session_context": signals.has_session_context,
    }
