from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

from rag.retrieval.contracts.primitives import (
    Payload,
    Vector,
    validate_finite_number,
    validate_mapping_size,
    validate_non_empty_str,
    validate_non_negative_int,
    validate_optional_clean_str,
    validate_optional_positive_int,
    validate_optional_sequence_limit,
    validate_positive_int,
)


@dataclass(frozen=True)
class SparseQueryVector:
    indices: tuple[int, ...]
    values: tuple[float, ...]

    @property
    def nnz(self) -> int:
        return len(self.indices)

    @property
    def is_empty(self) -> bool:
        return self.nnz == 0

    def normalized(self) -> Self:
        pairs = sorted(
            (int(index), float(value))
            for index, value in zip(self.indices, self.values, strict=True)
            if float(value) != 0.0
        )
        return type(self)(
            indices=tuple(index for index, _ in pairs),
            values=tuple(value for _, value in pairs),
        )

    def validate(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError(
                "sparse_query_vector.indices and sparse_query_vector.values "
                "must have the same length."
            )

        previous_index: int | None = None

        for position, index in enumerate(self.indices):
            validate_non_negative_int(
                f"sparse_query_vector.indices[{position}]",
                index,
            )

            if previous_index is not None and index <= previous_index:
                raise ValueError(
                    "sparse_query_vector.indices must be strictly increasing."
                )

            previous_index = index

        for position, value in enumerate(self.values):
            validate_finite_number(
                f"sparse_query_vector.values[{position}]",
                value,
            )

            if float(value) == 0.0:
                raise ValueError(
                    "sparse_query_vector.values cannot contain zero weights."
                )


@dataclass(frozen=True)
class QueryEmbedding:
    query: str
    dense_vector: Vector | None = None
    sparse_vector: SparseQueryVector | None = None
    embedding_model: str | None = None
    dense_model: str | None = None
    sparse_model: str | None = None
    embedding_dimension: int | None = None
    metadata: Payload = field(default_factory=dict)

    @property
    def has_dense(self) -> bool:
        return bool(self.dense_vector)

    @property
    def has_sparse(self) -> bool:
        return self.sparse_vector is not None and not self.sparse_vector.is_empty

    @property
    def is_hybrid(self) -> bool:
        return self.has_dense and self.has_sparse

    @property
    def dense_dimension(self) -> int | None:
        if self.dense_vector is None:
            return None

        return len(self.dense_vector)

    @property
    def vector(self) -> Vector:
        """
        Backward-compatible dense vector alias.

        Older retrieval graph code may still call query_embedding.vector.
        Backward-compatible alias for older retrieval code.
        """
        if self.dense_vector is None:
            raise ValueError("QueryEmbedding does not contain a dense vector.")

        return self.dense_vector

    @property
    def model(self) -> str | None:
        """
        Backward-compatible model alias.

        Prefer dense_model / sparse_model in new code.
        """
        return self.embedding_model or self.dense_model or self.sparse_model

    def normalized(self) -> Self:
        return type(self)(
            query=self.query.strip(),
            dense_vector=(
                [float(value) for value in self.dense_vector]
                if self.dense_vector is not None
                else None
            ),
            sparse_vector=(
                self.sparse_vector.normalized()
                if self.sparse_vector is not None
                else None
            ),
            embedding_model=_clean_optional(self.embedding_model),
            dense_model=_clean_optional(self.dense_model),
            sparse_model=_clean_optional(self.sparse_model),
            embedding_dimension=self.embedding_dimension,
            metadata=dict(self.metadata),
        )

    def validate(
        self,
        *,
        expected_dense_dimension: int | None = None,
        max_metadata_keys: int | None = None,
    ) -> None:
        validate_non_empty_str("query_embedding.query", self.query)
        validate_optional_clean_str(
            "query_embedding.embedding_model",
            self.embedding_model,
        )
        validate_optional_clean_str("query_embedding.dense_model", self.dense_model)
        validate_optional_clean_str("query_embedding.sparse_model", self.sparse_model)
        validate_optional_positive_int(
            "query_embedding.embedding_dimension",
            self.embedding_dimension,
        )
        validate_mapping_size(
            "query_embedding.metadata",
            self.metadata,
            max_metadata_keys,
        )

        if self.dense_vector is None and self.sparse_vector is None:
            raise ValueError(
                "QueryEmbedding must contain at least one of dense_vector "
                "or sparse_vector."
            )

        self._validate_dense_vector(expected_dense_dimension)
        self._validate_sparse_vector()

    def _validate_dense_vector(
        self,
        expected_dense_dimension: int | None,
    ) -> None:
        if self.dense_vector is None:
            return

        validate_optional_sequence_limit(
            "query_embedding.dense_vector",
            self.dense_vector,
            expected_dense_dimension,
        )

        if len(self.dense_vector) == 0:
            raise ValueError("query_embedding.dense_vector cannot be empty.")

        if expected_dense_dimension is not None:
            validate_positive_int(
                "query_embedding.expected_dense_dimension",
                expected_dense_dimension,
            )

            if len(self.dense_vector) != expected_dense_dimension:
                raise ValueError(
                    "query_embedding.dense_vector dimension mismatch. "
                    f"Expected {expected_dense_dimension}, got "
                    f"{len(self.dense_vector)}."
                )

        if self.embedding_dimension is not None and (
            len(self.dense_vector) != self.embedding_dimension
        ):
            raise ValueError(
                "query_embedding.embedding_dimension must match dense vector length."
            )

        for index, value in enumerate(self.dense_vector):
            validate_finite_number(f"query_embedding.dense_vector[{index}]", value)

    def _validate_sparse_vector(self) -> None:
        if self.sparse_vector is None:
            return

        if not isinstance(self.sparse_vector, SparseQueryVector):
            raise ValueError(
                "query_embedding.sparse_vector must be SparseQueryVector."
            )

        self.sparse_vector.validate()


@dataclass(frozen=True)
class QueryEmbeddingBatch:
    embeddings: tuple[QueryEmbedding, ...]
    metadata: Payload = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.embeddings)

    @property
    def dense_embeddings(self) -> tuple[QueryEmbedding, ...]:
        return tuple(embedding for embedding in self.embeddings if embedding.has_dense)

    @property
    def sparse_embeddings(self) -> tuple[QueryEmbedding, ...]:
        return tuple(embedding for embedding in self.embeddings if embedding.has_sparse)

    @property
    def hybrid_embeddings(self) -> tuple[QueryEmbedding, ...]:
        return tuple(embedding for embedding in self.embeddings if embedding.is_hybrid)

    def validate(
        self,
        *,
        expected_dense_dimension: int | None = None,
        max_metadata_keys: int | None = None,
    ) -> None:
        if not self.embeddings:
            raise ValueError("query_embedding_batch.embeddings cannot be empty.")

        validate_mapping_size(
            "query_embedding_batch.metadata",
            self.metadata,
            max_metadata_keys,
        )

        for index, embedding in enumerate(self.embeddings):
            if not isinstance(embedding, QueryEmbedding):
                raise ValueError(
                    f"query_embedding_batch.embeddings[{index}] "
                    "must be QueryEmbedding."
                )

            embedding.validate(
                expected_dense_dimension=expected_dense_dimension,
                max_metadata_keys=max_metadata_keys,
            )


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None

    cleaned = value.strip()
    return cleaned or None