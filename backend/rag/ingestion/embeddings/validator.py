from __future__ import annotations

import math
from typing import Any, Sequence

from rag.ingestion.embeddings.types import (
    EmbeddingValidationIssue,
    EmbeddingValidationReport,
    SparseVectorData,
    ValidEmbeddingPairs,
    Vector,
)


def filter_valid_pairs(
    chunks: Sequence[Any],
    dense_vectors: Sequence[Vector],
    *,
    expected_dim: int,
    sparse_vectors: Sequence[SparseVectorData] | None = None,
) -> ValidEmbeddingPairs:
    valid_chunks: list[Any] = []
    valid_dense_vectors: list[Vector] = []
    valid_sparse_vectors: list[SparseVectorData] = []
    issues: list[EmbeddingValidationIssue] = []

    total = max(len(chunks), len(dense_vectors))
    sparse_enabled = sparse_vectors is not None

    for index in range(total):
        chunk = chunks[index] if index < len(chunks) else None
        dense = dense_vectors[index] if index < len(dense_vectors) else None
        sparse = sparse_vectors[index] if sparse_enabled and index < len(sparse_vectors or []) else None

        chunk_id = _chunk_id(chunk)

        reason = _invalid_reason(
            chunk=chunk,
            dense=dense,
            sparse=sparse,
            expected_dim=expected_dim,
            sparse_required=sparse_enabled,
        )

        if reason:
            issues.append(
                EmbeddingValidationIssue(
                    index=index,
                    reason=reason,
                    chunk_id=chunk_id,
                )
            )
            continue

        valid_chunks.append(chunk)
        valid_dense_vectors.append([float(value) for value in dense or []])
        valid_sparse_vectors.append(sparse or SparseVectorData(indices=[], values=[]))

    return ValidEmbeddingPairs(
        chunks=valid_chunks,
        dense_vectors=valid_dense_vectors,
        sparse_vectors=valid_sparse_vectors,
        report=EmbeddingValidationReport(
            total_chunks=len(chunks),
            total_embeddings=len(dense_vectors),
            valid_count=len(valid_chunks),
            invalid_count=len(issues),
            expected_dim=expected_dim,
            issues=tuple(issues),
        ),
    )


def _invalid_reason(
    *,
    chunk: Any,
    dense: Sequence[float] | None,
    sparse: SparseVectorData | None,
    expected_dim: int,
    sparse_required: bool,
) -> str:
    if chunk is None:
        return "missing_chunk"

    if dense is None:
        return "missing_dense_vector"

    if len(dense) != expected_dim:
        return f"dense_dimension_mismatch:{len(dense)}!={expected_dim}"

    for value in dense:
        if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
            return "invalid_dense_value"

    if sparse_required:
        if sparse is None:
            return "missing_sparse_vector"

        if len(sparse.indices) != len(sparse.values):
            return "sparse_indices_values_mismatch"

        if not sparse.indices:
            return "empty_sparse_vector"

        previous = -1
        for index, value in zip(sparse.indices, sparse.values):
            if not isinstance(index, int):
                return "invalid_sparse_index"

            if index < 0:
                return "negative_sparse_index"

            if index <= previous:
                return "sparse_indices_not_strictly_increasing"

            if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
                return "invalid_sparse_value"

            previous = index

    return ""


def _chunk_id(chunk: Any) -> str | None:
    if chunk is None:
        return None

    direct = getattr(chunk, "chunk_id", None) or getattr(chunk, "id", None)
    if direct:
        return str(direct)

    metadata = getattr(chunk, "metadata", None)
    if isinstance(metadata, dict):
        value = metadata.get("chunk_id") or metadata.get("id")
        if value:
            return str(value)

    return None