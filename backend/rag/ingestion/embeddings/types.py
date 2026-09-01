from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


Payload = Dict[str, Any]
Vector = List[float]


@dataclass(frozen=True)
class SparseVectorData:
    indices: List[int]
    values: List[float]

    @property
    def count(self) -> int:
        return len(self.indices)

    @property
    def is_empty(self) -> bool:
        return not self.indices or not self.values


@dataclass(frozen=True)
class EmbeddingBatch:
    """
    Output from BGEM3FlagModel.

    Order is preserved:
      chunks[i] -> dense_vectors[i]
      chunks[i] -> sparse_vectors[i]
    """

    dense_vectors: List[Vector]
    sparse_vectors: List[SparseVectorData]
    model_name: str
    dense_model_name: str
    sparse_model_name: str
    vector_size: int
    normalized: bool

    @property
    def vectors(self) -> List[Vector]:
        """
        Backward-compatible dense alias.
        """
        return self.dense_vectors

    @property
    def count(self) -> int:
        return len(self.dense_vectors)

    @property
    def sparse_count(self) -> int:
        return len(self.sparse_vectors)


@dataclass(frozen=True)
class EmbeddingValidationIssue:
    index: int
    reason: str
    chunk_id: Optional[str] = None


@dataclass(frozen=True)
class EmbeddingValidationReport:
    total_chunks: int
    total_embeddings: int
    valid_count: int
    invalid_count: int
    expected_dim: int
    issues: Sequence[EmbeddingValidationIssue] = field(default_factory=tuple)

    @property
    def is_valid(self) -> bool:
        return self.invalid_count == 0 and self.valid_count > 0


@dataclass(frozen=True)
class ValidEmbeddingPairs:
    chunks: List[Any]
    dense_vectors: List[Vector]
    sparse_vectors: List[SparseVectorData]
    report: EmbeddingValidationReport

    @property
    def vectors(self) -> List[Vector]:
        """
        Backward-compatible dense alias.
        """
        return self.dense_vectors

    @property
    def count(self) -> int:
        return len(self.chunks)


@dataclass(frozen=True)
class VectorPoint:
    id: str
    dense_vector: Vector
    sparse_vector: Optional[SparseVectorData]
    payload: Payload

    @property
    def vector(self) -> Vector:
        """
        Backward-compatible dense alias.
        """
        return self.dense_vector


@dataclass(frozen=True)
class StoreUpsertResult:
    collection_name: str
    attempted_count: int
    upserted_count: int
    skipped_count: int = 0
    status: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.upserted_count > 0 and self.skipped_count == 0


@dataclass(frozen=True)
class SearchResult:
    id: str
    score: float
    text: str
    payload: Payload


@dataclass(frozen=True)
class SearchResults:
    query: str
    results: List[SearchResult]

    @property
    def count(self) -> int:
        return len(self.results)


@dataclass(frozen=True)
class IngestionResult:
    source_path: str
    collection_name: str

    parsed: bool
    chunk_count: int
    embedded_count: int
    stored_count: int

    embedding_model: str
    vector_size: int

    validation_report: Optional[EmbeddingValidationReport] = None
    warnings: Sequence[str] = field(default_factory=tuple)

    @property
    def success(self) -> bool:
        return self.parsed and self.stored_count > 0