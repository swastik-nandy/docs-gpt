from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Mapping, Sequence

from rag.ingestion.data_transformation.chunking.models import Chunk
from rag.ingestion.embeddings.artifact_reader import ChunkArtifactFile
from rag.ingestion.embeddings.types import Payload, SparseVectorData, Vector, VectorPoint


def build_vector_points_from_chunks(
    *,
    artifact_file: ChunkArtifactFile,
    chunks: Sequence[Chunk],
    dense_vectors: Sequence[Vector],
    sparse_vectors: Sequence[SparseVectorData],
    dense_embedding_model: str,
    sparse_embedding_model: str,
    schema_version: str,
) -> list[VectorPoint]:
    if len(chunks) != len(dense_vectors):
        raise ValueError(
            f"Cannot build vector points: chunks={len(chunks)} dense_vectors={len(dense_vectors)}."
        )

    if len(chunks) != len(sparse_vectors):
        raise ValueError(
            f"Cannot build vector points: chunks={len(chunks)} sparse_vectors={len(sparse_vectors)}."
        )

    points: list[VectorPoint] = []

    for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors):
        payload = chunk_to_payload(
            artifact_file=artifact_file,
            chunk=chunk,
            dense_embedding_model=dense_embedding_model,
            sparse_embedding_model=sparse_embedding_model,
            schema_version=schema_version,
            has_sparse_vector=not sparse.is_empty,
        )

        points.append(
            VectorPoint(
                id=_generate_point_id(payload),
                dense_vector=list(dense),
                sparse_vector=sparse,
                payload=payload,
            )
        )

    return points


def chunk_to_payload(
    *,
    artifact_file: ChunkArtifactFile,
    chunk: Chunk,
    dense_embedding_model: str,
    sparse_embedding_model: str,
    schema_version: str,
    has_sparse_vector: bool,
) -> Payload:
    metadata = _safe_mapping(getattr(chunk, "metadata", None))

    text = _as_text(getattr(chunk, "text", None))
    heading = _as_text(getattr(chunk, "heading", None))
    context = _as_text(getattr(chunk, "context", None))

    chunk_id = (
        _as_text(getattr(chunk, "chunk_id", None))
        or _as_text(metadata.get("chunk_id"))
        or _content_hash(text)
    )

    payload: Payload = {
        "tenant_id": artifact_file.tenant_id,
        "job_id": artifact_file.job_id,
        "source_id": artifact_file.source_id,

        "text": text,
        "heading": heading,
        "context": context,

        "chunk_id": chunk_id,
        "chunk_type": _as_text(getattr(chunk, "chunk_type", None)),
        "subtype": _as_text(getattr(chunk, "subtype", None)),
        "tokens": _as_int_or_none(getattr(chunk, "tokens", None)),
        "order": _as_int_or_none(getattr(chunk, "order", None)),

        "source": _as_text(getattr(chunk, "source", None)),
        "chunk_artifact_file": artifact_file.path.as_posix(),
        "chunk_artifact_name": artifact_file.file_name,

        "text_sha256": _content_hash(text),
        "schema_version": schema_version,
        "point_identity_version": "chunk_identity_v2",

        "embedding_model": dense_embedding_model,
        "dense_embedding_model": dense_embedding_model,
        "sparse_embedding_model": sparse_embedding_model,
        "has_dense_vector": True,
        "has_sparse_vector": has_sparse_vector,
        "vector_backend": "qdrant_named_dense_sparse",

        "metadata": _json_safe(metadata),
    }

    heading_path = metadata.get("heading_path") or metadata.get("path")
    rendered_heading_path = _as_string_list(heading_path)
    if rendered_heading_path:
        payload["heading_path"] = rendered_heading_path

    return _drop_empty(payload)

def _generate_point_id(payload: Mapping[str, Any]) -> str:
    identity = {
        "tenant_id": payload.get("tenant_id"),
        "job_id": payload.get("job_id"),
        "source_id": payload.get("source_id"),
        "chunk_id": payload.get("chunk_id"),
        "text_sha256": payload.get("text_sha256"),
        "point_identity_version": payload.get("point_identity_version"),
    }

    identity_text = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity_text))


def _content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _drop_empty(payload: Payload) -> Payload:
    result: Payload = {}

    for key, value in payload.items():
        if value is None:
            continue

        if value == "":
            continue

        if value == []:
            continue

        if value == {}:
            continue

        result[key] = value

    return result


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False, default=str)
        return value
    except Exception:
        return str(value)


def _safe_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_text(value: Any) -> str:
    if value is None:
        return ""

    return " ".join(str(value).split())


def _as_int_or_none(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        cleaned = _as_text(value)
        return [cleaned] if cleaned else []

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result: list[str] = []
        for item in value:
            cleaned = _as_text(item)
            if cleaned:
                result.append(cleaned)
        return result

    cleaned = _as_text(value)
    return [cleaned] if cleaned else []