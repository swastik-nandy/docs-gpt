from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any, Self
from uuid import uuid4

from rag.retrieval.contracts.candidates import (
    FusedCandidate,
    RerankedCandidate,
    RetrievalCandidate,
)
from rag.retrieval.contracts.context import ContextBlock
from rag.retrieval.contracts.embeddings import QueryEmbedding
from rag.retrieval.contracts.query import NormalizedQuery
from rag.retrieval.contracts.request import RetrievalRequest
from rag.retrieval.contracts.response import RetrievalDecision, RetrievalResponse


Payload = dict[str, Any]


class RetrievalEventType(str, Enum):
    REQUEST_CREATED = "request_created"
    QUERY_NORMALIZED = "query_normalized"
    MEMORY_PLANNED = "memory_planned"
    QUERY_EMBEDDED = "query_embedded"
    DENSE_RETRIEVED = "dense_retrieved"
    SPARSE_RETRIEVED = "sparse_retrieved"
    CANDIDATES_FUSED = "candidates_fused"
    STRUCTURE_EXPANDED = "structure_expanded"
    CANDIDATES_RERANKED = "candidates_reranked"
    CONTEXT_GRADED = "context_graded"
    QUERY_REWRITTEN = "query_rewritten"
    CONTEXT_PACKED = "context_packed"
    RESPONSE_CREATED = "response_created"
    GRAPH_STARTED = "graph_started"
    GRAPH_COMPLETED = "graph_completed"
    GRAPH_FAILED = "graph_failed"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class RetrievalEvent:
    event_type: RetrievalEventType
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = field(default_factory=time)
    request_id: str | None = None
    conversation_id: str | None = None
    tenant_id: str | None = None
    payload: Payload = field(default_factory=dict)
    metadata: Payload = field(default_factory=dict)

    def with_metadata(self, values: Payload) -> Self:
        merged = dict(self.metadata)
        merged.update(values)

        return type(self)(
            event_type=self.event_type,
            event_id=self.event_id,
            timestamp=self.timestamp,
            request_id=self.request_id,
            conversation_id=self.conversation_id,
            tenant_id=self.tenant_id,
            payload=dict(self.payload),
            metadata=merged,
        )


@dataclass(frozen=True)
class RetrievalEventBatch:
    events: tuple[RetrievalEvent, ...]

    @property
    def size(self) -> int:
        return len(self.events)

    def append(self, event: RetrievalEvent) -> Self:
        return type(self)(events=self.events + (event,))

    def extend(self, events: tuple[RetrievalEvent, ...]) -> Self:
        return type(self)(events=self.events + events)


def request_created_event(request: RetrievalRequest) -> RetrievalEvent:
    return RetrievalEvent(
        event_type=RetrievalEventType.REQUEST_CREATED,
        request_id=request.request_id,
        conversation_id=request.conversation_id,
        tenant_id=request.tenant.tenant_id,
        payload={
            "query": request.query,
            "mode": request.mode.value,
            "top_k": request.top_k,
            "wants_memory": request.wants_memory,
            "has_chat_history": bool(request.chat_history),
            "has_source_constraints": not request.source_constraints.is_global,
        },
    )


def query_normalized_event(
    *,
    request: RetrievalRequest,
    normalized_query: NormalizedQuery,
) -> RetrievalEvent:
    return RetrievalEvent(
        event_type=RetrievalEventType.QUERY_NORMALIZED,
        request_id=request.request_id,
        conversation_id=request.conversation_id,
        tenant_id=request.tenant.tenant_id,
        payload={
            "original_query": normalized_query.original_query,
            "normalized_query": normalized_query.normalized_query,
            "intent": normalized_query.intent.value,
            "defects": [defect.value for defect in normalized_query.defects],
            "requires_memory": normalized_query.requires_memory,
            "search_queries": list(normalized_query.search_queries),
        },
    )


def memory_planned_event(
    *,
    request: RetrievalRequest,
    memory_plan: Payload,
) -> RetrievalEvent:
    return RetrievalEvent(
        event_type=RetrievalEventType.MEMORY_PLANNED,
        request_id=request.request_id,
        conversation_id=request.conversation_id,
        tenant_id=request.tenant.tenant_id,
        payload={
            "memory_plan": memory_plan,
            "wants_short_term_memory": request.wants_short_term_memory,
            "wants_long_term_memory": request.wants_long_term_memory,
        },
    )


def query_embedded_event(
    *,
    request: RetrievalRequest,
    query_embedding: QueryEmbedding,
) -> RetrievalEvent:
    return RetrievalEvent(
        event_type=RetrievalEventType.QUERY_EMBEDDED,
        request_id=request.request_id,
        conversation_id=request.conversation_id,
        tenant_id=request.tenant.tenant_id,
        payload={
            "query": query_embedding.query,
            "model": query_embedding.model,
            "dense_dimension": query_embedding.dense_dimension,
            "has_dense": query_embedding.has_dense,
            "has_sparse": query_embedding.has_sparse,
            "is_hybrid": query_embedding.is_hybrid,
        },
    )


def candidates_retrieved_event(
    *,
    event_type: RetrievalEventType,
    request: RetrievalRequest,
    candidates: tuple[RetrievalCandidate, ...],
) -> RetrievalEvent:
    if event_type not in {
        RetrievalEventType.DENSE_RETRIEVED,
        RetrievalEventType.SPARSE_RETRIEVED,
        RetrievalEventType.STRUCTURE_EXPANDED,
    }:
        raise ValueError("event_type must be a retrieval candidate event type.")

    return RetrievalEvent(
        event_type=event_type,
        request_id=request.request_id,
        conversation_id=request.conversation_id,
        tenant_id=request.tenant.tenant_id,
        payload={
            "candidate_count": len(candidates),
            "point_ids": [candidate.point_id for candidate in candidates],
            "source_ids": sorted(
                {
                    source_id
                    for candidate in candidates
                    if (source_id := candidate.source_id)
                }
            ),
        },
    )


def candidates_fused_event(
    *,
    request: RetrievalRequest,
    candidates: tuple[FusedCandidate, ...],
) -> RetrievalEvent:
    return RetrievalEvent(
        event_type=RetrievalEventType.CANDIDATES_FUSED,
        request_id=request.request_id,
        conversation_id=request.conversation_id,
        tenant_id=request.tenant.tenant_id,
        payload={
            "candidate_count": len(candidates),
            "point_ids": [candidate.point_id for candidate in candidates],
        },
    )


def candidates_reranked_event(
    *,
    request: RetrievalRequest,
    candidates: tuple[RerankedCandidate, ...],
) -> RetrievalEvent:
    return RetrievalEvent(
        event_type=RetrievalEventType.CANDIDATES_RERANKED,
        request_id=request.request_id,
        conversation_id=request.conversation_id,
        tenant_id=request.tenant.tenant_id,
        payload={
            "candidate_count": len(candidates),
            "point_ids": [candidate.point_id for candidate in candidates],
            "top_score": candidates[0].rerank_score if candidates else None,
        },
    )


def context_graded_event(
    *,
    request: RetrievalRequest,
    decision: RetrievalDecision,
) -> RetrievalEvent:
    return RetrievalEvent(
        event_type=RetrievalEventType.CONTEXT_GRADED,
        request_id=request.request_id,
        conversation_id=request.conversation_id,
        tenant_id=request.tenant.tenant_id,
        payload={
            "is_sufficient": decision.is_sufficient,
            "confidence": decision.confidence.value,
            "confidence_score": decision.confidence_score,
            "should_rewrite": decision.should_rewrite,
            "reason": decision.reason,
        },
    )


def query_rewritten_event(
    *,
    request: RetrievalRequest,
    rewritten_query: str,
    retry_count: int,
) -> RetrievalEvent:
    return RetrievalEvent(
        event_type=RetrievalEventType.QUERY_REWRITTEN,
        request_id=request.request_id,
        conversation_id=request.conversation_id,
        tenant_id=request.tenant.tenant_id,
        payload={
            "rewritten_query": rewritten_query,
            "retry_count": retry_count,
        },
    )


def context_packed_event(
    *,
    request: RetrievalRequest,
    context_blocks: tuple[ContextBlock, ...],
) -> RetrievalEvent:
    return RetrievalEvent(
        event_type=RetrievalEventType.CONTEXT_PACKED,
        request_id=request.request_id,
        conversation_id=request.conversation_id,
        tenant_id=request.tenant.tenant_id,
        payload={
            "context_count": len(context_blocks),
            "block_ids": [block.block_id for block in context_blocks],
            "source_ids": sorted(
                {
                    source_id
                    for block in context_blocks
                    if (source_id := block.source_id)
                }
            ),
        },
    )


def response_created_event(response: RetrievalResponse) -> RetrievalEvent:
    return RetrievalEvent(
        event_type=RetrievalEventType.RESPONSE_CREATED,
        request_id=response.request_id,
        conversation_id=response.conversation_id,
        payload={
            "query": response.query,
            "is_successful": response.is_successful,
            "has_context": response.has_context,
            "context_count": len(response.context_blocks),
            "candidate_count": len(response.candidates),
            "warning_count": len(response.warnings),
            "error_count": len(response.errors),
        },
    )


def warning_event(
    *,
    request: RetrievalRequest | None,
    message: str,
    metadata: Payload | None = None,
) -> RetrievalEvent:
    return RetrievalEvent(
        event_type=RetrievalEventType.WARNING,
        request_id=request.request_id if request else None,
        conversation_id=request.conversation_id if request else None,
        tenant_id=request.tenant.tenant_id if request else None,
        payload={
            "message": message,
        },
        metadata=metadata or {},
    )


def error_event(
    *,
    request: RetrievalRequest | None,
    message: str,
    error_type: str,
    metadata: Payload | None = None,
) -> RetrievalEvent:
    return RetrievalEvent(
        event_type=RetrievalEventType.ERROR,
        request_id=request.request_id if request else None,
        conversation_id=request.conversation_id if request else None,
        tenant_id=request.tenant.tenant_id if request else None,
        payload={
            "message": message,
            "error_type": error_type,
        },
        metadata=metadata or {},
    )