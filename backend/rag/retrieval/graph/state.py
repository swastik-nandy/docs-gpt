from __future__ import annotations

import operator
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, TypedDict

from rag.retrieval.contracts.candidates import (
    FusedCandidate,
    RerankedCandidate,
    RetrievalCandidate,
)
from rag.retrieval.contracts.context import ContextBlock
from rag.retrieval.contracts.embeddings import QueryEmbedding
from rag.retrieval.contracts.query import NormalizedQuery
from rag.retrieval.contracts.request import RetrievalRequest
from rag.retrieval.contracts.response import RetrievalDecision


Payload = dict[str, Any]


class RetrievalGraphStage(str, Enum):
    CREATED = "created"
    NORMALIZED = "normalized"
    MEMORY_PLANNED = "memory_planned"
    QUERY_EMBEDDED = "query_embedded"
    DENSE_RETRIEVED = "dense_retrieved"
    SPARSE_RETRIEVED = "sparse_retrieved"
    HYBRID_FUSED = "hybrid_fused"
    STRUCTURE_EXPANDED = "structure_expanded"
    RERANKED = "reranked"
    GRADED = "graded"
    QUERY_REWRITTEN = "query_rewritten"
    CORRECTIVE_RETRIEVED = "corrective_retrieved"
    CONTEXT_PACKED = "context_packed"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class RetrievalTraceEvent:
    stage: RetrievalGraphStage
    message: str
    metadata: Payload = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalGraphError:
    stage: RetrievalGraphStage
    message: str
    error_type: str = "RetrievalGraphError"
    metadata: Payload = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalGraphWarning:
    stage: RetrievalGraphStage
    message: str
    metadata: Payload = field(default_factory=dict)


class RetrievalGraphState(TypedDict, total=False):
    request: RetrievalRequest

    normalized_query: NormalizedQuery
    rewritten_query: str
    rewrite_count: int

    query_embedding: QueryEmbedding

    dense_candidates: tuple[RetrievalCandidate, ...]
    sparse_candidates: tuple[RetrievalCandidate, ...]
    corrective_candidates: tuple[RetrievalCandidate, ...]

    fused_candidates: tuple[FusedCandidate, ...]
    expanded_candidates: tuple[RetrievalCandidate, ...]
    reranked_candidates: tuple[RerankedCandidate, ...]

    context_blocks: tuple[ContextBlock, ...]
    decision: RetrievalDecision

    current_stage: RetrievalGraphStage
    retry_count: int

    trace: Annotated[list[RetrievalTraceEvent], operator.add]
    warnings: Annotated[list[RetrievalGraphWarning], operator.add]
    errors: Annotated[list[RetrievalGraphError], operator.add]

    metadata: Payload


class RetrievalGraphInput(TypedDict):
    request: RetrievalRequest


class RetrievalGraphOutput(TypedDict):
    request: RetrievalRequest
    normalized_query: NormalizedQuery
    decision: RetrievalDecision
    context_blocks: tuple[ContextBlock, ...]
    reranked_candidates: tuple[RerankedCandidate, ...]
    trace: list[RetrievalTraceEvent]
    warnings: list[RetrievalGraphWarning]
    errors: list[RetrievalGraphError]
    metadata: Payload


def create_initial_state(request: RetrievalRequest) -> RetrievalGraphState:
    request.validate()

    return RetrievalGraphState(
        request=request,
        current_stage=RetrievalGraphStage.CREATED,
        retry_count=0,
        rewrite_count=0,
        trace=[
            RetrievalTraceEvent(
                stage=RetrievalGraphStage.CREATED,
                message="Retrieval graph state created.",
                metadata={
                    "tenant_id": request.tenant.tenant_id,
                    "mode": request.mode.value,
                    "top_k": request.top_k,
                    "has_chat_history": bool(request.chat_history),
                    "has_source_constraints": not request.source_constraints.is_global,
                    "wants_memory": request.wants_memory,
                    "wants_short_term_memory": request.wants_short_term_memory,
                    "wants_long_term_memory": request.wants_long_term_memory,
                },
            )
        ],
        warnings=[],
        errors=[],
        metadata={},
    )


def require_request(state: RetrievalGraphState) -> RetrievalRequest:
    request = state.get("request")
    if request is None:
        raise ValueError("RetrievalGraphState is missing request.")

    return request


def require_normalized_query(state: RetrievalGraphState) -> NormalizedQuery:
    normalized_query = state.get("normalized_query")
    if normalized_query is None:
        raise ValueError("RetrievalGraphState is missing normalized_query.")

    return normalized_query


def require_query_embedding(state: RetrievalGraphState) -> QueryEmbedding:
    query_embedding = state.get("query_embedding")
    if query_embedding is None:
        raise ValueError("RetrievalGraphState is missing query_embedding.")

    return query_embedding


def require_decision(state: RetrievalGraphState) -> RetrievalDecision:
    decision = state.get("decision")
    if decision is None:
        raise ValueError("RetrievalGraphState is missing decision.")

    return decision


def mark_stage(
    stage: RetrievalGraphStage,
    message: str,
    *,
    metadata: Payload | None = None,
) -> RetrievalGraphState:
    return RetrievalGraphState(
        current_stage=stage,
        trace=[
            RetrievalTraceEvent(
                stage=stage,
                message=message,
                metadata=metadata or {},
            )
        ],
    )


def add_warning(
    stage: RetrievalGraphStage,
    message: str,
    *,
    metadata: Payload | None = None,
) -> RetrievalGraphState:
    cleaned_message = message.strip()
    if not cleaned_message:
        return RetrievalGraphState()

    return RetrievalGraphState(
        warnings=[
            RetrievalGraphWarning(
                stage=stage,
                message=cleaned_message,
                metadata=metadata or {},
            )
        ],
    )


def add_error(
    stage: RetrievalGraphStage,
    message: str,
    *,
    error_type: str = "RetrievalGraphError",
    metadata: Payload | None = None,
) -> RetrievalGraphState:
    cleaned_message = message.strip()
    if not cleaned_message:
        return RetrievalGraphState()

    return RetrievalGraphState(
        current_stage=RetrievalGraphStage.FAILED,
        errors=[
            RetrievalGraphError(
                stage=stage,
                message=cleaned_message,
                error_type=error_type,
                metadata=metadata or {},
            )
        ],
    )


def increment_retry(state: RetrievalGraphState) -> RetrievalGraphState:
    return RetrievalGraphState(
        retry_count=int(state.get("retry_count", 0)) + 1,
    )


def increment_rewrite(state: RetrievalGraphState) -> RetrievalGraphState:
    return RetrievalGraphState(
        rewrite_count=int(state.get("rewrite_count", 0)) + 1,
    )


def merge_metadata(
    state: RetrievalGraphState,
    metadata: Payload,
) -> RetrievalGraphState:
    current = dict(state.get("metadata", {}))
    current.update(metadata)

    return RetrievalGraphState(metadata=current)


def to_graph_output(state: RetrievalGraphState) -> RetrievalGraphOutput:
    request = require_request(state)
    normalized_query = require_normalized_query(state)
    decision = require_decision(state)

    return RetrievalGraphOutput(
        request=request,
        normalized_query=normalized_query,
        decision=decision,
        context_blocks=tuple(state.get("context_blocks", ())),
        reranked_candidates=tuple(state.get("reranked_candidates", ())),
        trace=list(state.get("trace", [])),
        warnings=list(state.get("warnings", [])),
        errors=list(state.get("errors", [])),
        metadata=dict(state.get("metadata", {})),
    )