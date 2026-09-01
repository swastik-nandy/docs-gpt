from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


Payload = dict[str, Any]


class RetrievalErrorCode(str, Enum):
    UNKNOWN = "unknown"

    INVALID_REQUEST = "invalid_request"
    INVALID_TENANT = "invalid_tenant"
    INVALID_SOURCE_CONSTRAINTS = "invalid_source_constraints"

    QUERY_NORMALIZATION_FAILED = "query_normalization_failed"
    QUERY_TOO_SHORT = "query_too_short"
    QUERY_TOO_LONG = "query_too_long"
    QUERY_EMPTY_AFTER_NORMALIZATION = "query_empty_after_normalization"

    MEMORY_PLANNING_FAILED = "memory_planning_failed"
    MEMORY_STORE_UNAVAILABLE = "memory_store_unavailable"
    MEMORY_SEARCH_FAILED = "memory_search_failed"

    QUERY_EMBEDDING_FAILED = "query_embedding_failed"
    QUERY_EMBEDDING_EMPTY = "query_embedding_empty"
    QUERY_EMBEDDING_DIMENSION_MISMATCH = "query_embedding_dimension_mismatch"
    QUERY_EMBEDDING_MODEL_UNAVAILABLE = "query_embedding_model_unavailable"

    VECTOR_STORE_UNAVAILABLE = "vector_store_unavailable"
    VECTOR_STORE_QUERY_FAILED = "vector_store_query_failed"
    VECTOR_STORE_FILTER_FAILED = "vector_store_filter_failed"

    DENSE_RETRIEVAL_FAILED = "dense_retrieval_failed"
    SPARSE_RETRIEVAL_FAILED = "sparse_retrieval_failed"
    HYBRID_RETRIEVAL_FAILED = "hybrid_retrieval_failed"
    CORRECTIVE_RETRIEVAL_FAILED = "corrective_retrieval_failed"

    FUSION_FAILED = "fusion_failed"
    STRUCTURE_EXPANSION_FAILED = "structure_expansion_failed"
    RERANKING_FAILED = "reranking_failed"
    CONTEXT_GRADING_FAILED = "context_grading_failed"
    QUERY_REWRITE_FAILED = "query_rewrite_failed"
    CONTEXT_PACKING_FAILED = "context_packing_failed"

    GRAPH_STATE_INVALID = "graph_state_invalid"
    GRAPH_NODE_FAILED = "graph_node_failed"
    GRAPH_WORKFLOW_FAILED = "graph_workflow_failed"
    GRAPH_RECURSION_LIMIT_EXCEEDED = "graph_recursion_limit_exceeded"

    CONFIGURATION_ERROR = "configuration_error"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class RetrievalErrorDetail:
    code: RetrievalErrorCode
    message: str
    stage: str | None = None
    cause_type: str | None = None
    metadata: Payload = field(default_factory=dict)

    def to_dict(self) -> Payload:
        return {
            "code": self.code.value,
            "message": self.message,
            "stage": self.stage,
            "cause_type": self.cause_type,
            "metadata": dict(self.metadata),
        }


class RetrievalError(Exception):
    """
    Base exception for the retrieval subsystem.

    Every module under rag.retrieval should raise RetrievalError or a subclass
    when a failure crosses module boundaries.
    """

    def __init__(
        self,
        message: str,
        *,
        code: RetrievalErrorCode = RetrievalErrorCode.UNKNOWN,
        stage: str | None = None,
        cause: Exception | None = None,
        metadata: Payload | None = None,
    ) -> None:
        super().__init__(message)

        self.message = message
        self.code = code
        self.stage = stage
        self.cause = cause
        self.metadata = metadata or {}

    @property
    def detail(self) -> RetrievalErrorDetail:
        return RetrievalErrorDetail(
            code=self.code,
            message=self.message,
            stage=self.stage,
            cause_type=type(self.cause).__name__ if self.cause else None,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> Payload:
        return self.detail.to_dict()


class RetrievalConfigurationError(RetrievalError):
    def __init__(
        self,
        message: str,
        *,
        cause: Exception | None = None,
        metadata: Payload | None = None,
    ) -> None:
        super().__init__(
            message,
            code=RetrievalErrorCode.CONFIGURATION_ERROR,
            cause=cause,
            metadata=metadata,
        )


class RetrievalDependencyError(RetrievalError):
    def __init__(
        self,
        message: str,
        *,
        code: RetrievalErrorCode = RetrievalErrorCode.DEPENDENCY_UNAVAILABLE,
        stage: str | None = None,
        cause: Exception | None = None,
        metadata: Payload | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            stage=stage,
            cause=cause,
            metadata=metadata,
        )


class RetrievalRequestError(RetrievalError):
    def __init__(
        self,
        message: str,
        *,
        code: RetrievalErrorCode = RetrievalErrorCode.INVALID_REQUEST,
        cause: Exception | None = None,
        metadata: Payload | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            cause=cause,
            metadata=metadata,
        )


class QueryNormalizationError(RetrievalError):
    def __init__(
        self,
        message: str,
        *,
        code: RetrievalErrorCode = RetrievalErrorCode.QUERY_NORMALIZATION_FAILED,
        cause: Exception | None = None,
        metadata: Payload | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            stage="query_normalization",
            cause=cause,
            metadata=metadata,
        )


class QueryMemoryError(RetrievalError):
    def __init__(
        self,
        message: str,
        *,
        code: RetrievalErrorCode = RetrievalErrorCode.MEMORY_PLANNING_FAILED,
        cause: Exception | None = None,
        metadata: Payload | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            stage="query_memory",
            cause=cause,
            metadata=metadata,
        )


class QueryEmbeddingError(RetrievalError):
    def __init__(
        self,
        message: str,
        *,
        code: RetrievalErrorCode = RetrievalErrorCode.QUERY_EMBEDDING_FAILED,
        cause: Exception | None = None,
        metadata: Payload | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            stage="query_embedding",
            cause=cause,
            metadata=metadata,
        )


class VectorStoreError(RetrievalError):
    def __init__(
        self,
        message: str,
        *,
        code: RetrievalErrorCode = RetrievalErrorCode.VECTOR_STORE_QUERY_FAILED,
        cause: Exception | None = None,
        metadata: Payload | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            stage="vector_store",
            cause=cause,
            metadata=metadata,
        )


class RetrievalExecutionError(RetrievalError):
    def __init__(
        self,
        message: str,
        *,
        code: RetrievalErrorCode,
        stage: str,
        cause: Exception | None = None,
        metadata: Payload | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            stage=stage,
            cause=cause,
            metadata=metadata,
        )


class FusionError(RetrievalExecutionError):
    def __init__(
        self,
        message: str,
        *,
        cause: Exception | None = None,
        metadata: Payload | None = None,
    ) -> None:
        super().__init__(
            message,
            code=RetrievalErrorCode.FUSION_FAILED,
            stage="fusion",
            cause=cause,
            metadata=metadata,
        )


class StructureExpansionError(RetrievalExecutionError):
    def __init__(
        self,
        message: str,
        *,
        cause: Exception | None = None,
        metadata: Payload | None = None,
    ) -> None:
        super().__init__(
            message,
            code=RetrievalErrorCode.STRUCTURE_EXPANSION_FAILED,
            stage="structure_expansion",
            cause=cause,
            metadata=metadata,
        )


class RerankingError(RetrievalExecutionError):
    def __init__(
        self,
        message: str,
        *,
        cause: Exception | None = None,
        metadata: Payload | None = None,
    ) -> None:
        super().__init__(
            message,
            code=RetrievalErrorCode.RERANKING_FAILED,
            stage="reranking",
            cause=cause,
            metadata=metadata,
        )


class ContextPackingError(RetrievalExecutionError):
    def __init__(
        self,
        message: str,
        *,
        cause: Exception | None = None,
        metadata: Payload | None = None,
    ) -> None:
        super().__init__(
            message,
            code=RetrievalErrorCode.CONTEXT_PACKING_FAILED,
            stage="context_packing",
            cause=cause,
            metadata=metadata,
        )


class RetrievalGraphErrorBase(RetrievalError):
    def __init__(
        self,
        message: str,
        *,
        code: RetrievalErrorCode = RetrievalErrorCode.GRAPH_WORKFLOW_FAILED,
        stage: str | None = "graph",
        cause: Exception | None = None,
        metadata: Payload | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            stage=stage,
            cause=cause,
            metadata=metadata,
        )


def wrap_retrieval_error(
    exc: Exception,
    *,
    message: str,
    code: RetrievalErrorCode,
    stage: str,
    metadata: Payload | None = None,
) -> RetrievalError:
    if isinstance(exc, RetrievalError):
        return exc

    return RetrievalError(
        message,
        code=code,
        stage=stage,
        cause=exc,
        metadata=metadata,
    )