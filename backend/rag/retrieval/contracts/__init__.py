from __future__ import annotations

from rag.retrieval.contracts.candidates import (
    CandidateScore,
    CandidateSourceRef,
    FusedCandidate,
    RerankedCandidate,
    RetrievalCandidate,
)
from rag.retrieval.contracts.context import (
    ContextBlock,
    PackedContext,
    build_empty_packed_context,
)
from rag.retrieval.contracts.embeddings import (
    QueryEmbedding,
    QueryEmbeddingBatch,
    SparseQueryVector,
)
from rag.retrieval.contracts.enums import (
    CandidateOrigin,
    ChatRole,
    ConfidenceBand,
    MemoryScope,
    QueryDefect,
    QueryIntent,
    QueryProcessingMode,
    RetrievalMode,
)
from rag.retrieval.contracts.primitives import Payload, Vector
from rag.retrieval.contracts.query import (
    ExtractedFilter,
    NormalizedQuery,
    QueryVariant,
    build_initial_normalized_query,
)
from rag.retrieval.contracts.request import (
    ChatTurn,
    MemoryPolicy,
    QueryProcessingProfile,
    RetrievalRequest,
    SourceConstraints,
    TenantContext,
)
from rag.retrieval.contracts.response import (
    RetrievalDecision,
    RetrievalResponse,
    build_empty_retrieval_response,
)

__all__ = [
    "CandidateOrigin",
    "CandidateScore",
    "CandidateSourceRef",
    "ChatRole",
    "ChatTurn",
    "ConfidenceBand",
    "ContextBlock",
    "ExtractedFilter",
    "FusedCandidate",
    "MemoryPolicy",
    "MemoryScope",
    "NormalizedQuery",
    "PackedContext",
    "Payload",
    "QueryDefect",
    "QueryEmbedding",
    "QueryEmbeddingBatch",
    "QueryIntent",
    "QueryProcessingMode",
    "QueryProcessingProfile",
    "QueryVariant",
    "RerankedCandidate",
    "RetrievalCandidate",
    "RetrievalDecision",
    "RetrievalMode",
    "RetrievalRequest",
    "RetrievalResponse",
    "SourceConstraints",
    "SparseQueryVector",
    "TenantContext",
    "Vector",
    "build_empty_packed_context",
    "build_empty_retrieval_response",
    "build_initial_normalized_query",
]