from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import PurePath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from rag.ingestion.data_transformation.chunking.models import Chunk
from rag.ingestion.embeddings.types import Payload, Vector, VectorPoint


DEFAULT_SCHEMA_VERSION = "embedding_payload_v1"


# ---------------- SINGLE CONVERSION ---------------- #


def chunk_to_payload(
    chunk: Chunk,
    *,
    embedding_model: Optional[str] = None,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
) -> Payload:
    """
    Convert a Chunk into a Qdrant payload.

    Payload design:
    - keep generation text directly available
    - preserve source traceability
    - expose generic filter fields
    - keep parser/chunker metadata without making retrieval domain-specific
    """

    metadata = _safe_mapping(chunk.metadata)
    source = _as_str(chunk.source)
    file_name = _extract_file_name(source)

    text = _as_str(chunk.text)
    heading = _as_optional_str(chunk.heading)
    context = _as_optional_str(chunk.context)

    chunk_type = _first_present(
        getattr(chunk, "type", None),
        getattr(chunk, "chunk_type", None),
        metadata.get("type"),
        metadata.get("chunk_type"),
    )

    subtype = _first_present(
        getattr(chunk, "subtype", None),
        metadata.get("subtype"),
    )

    heading_path = _as_list(
        _first_present(
            metadata.get("heading_path"),
            metadata.get("path"),
        )
    )

    path = _as_list(metadata.get("path"))

    chunk_index = _first_present(
        getattr(chunk, "index", None),
        getattr(chunk, "chunk_index", None),
        metadata.get("chunk_index"),
        metadata.get("order"),
    )

    split_part = _first_present(
        getattr(chunk, "split_part", None),
        metadata.get("split_part"),
    )

    doc_id = _generate_doc_id(source=source, file_name=file_name)
    content_hash = _content_hash(text)

    payload: Payload = {
        # -------- CORE CONTENT -------- #
        "text": text,
        "heading": heading,
        "context": context,

        # -------- SOURCE TRACEABILITY -------- #
        "source": source,
        "file_name": file_name,
        "doc_id": doc_id,

        # -------- CHUNK IDENTITY -------- #
        "chunk_id": _chunk_identity(chunk),
        "content_hash": content_hash,
        "schema_version": schema_version,

        # -------- RETRIEVAL / FILTER FIELDS -------- #
        "chunk_type": _as_optional_str(chunk_type),
        "subtype": _as_optional_str(subtype),
        "tokens": _as_int_or_none(getattr(chunk, "tokens", None)),
        "heading_path": heading_path,
        "path": path,
        "chunk_index": _as_int_or_none(chunk_index),
        "split_part": _as_int_or_none(split_part),

        # -------- EMBEDDING TRACEABILITY -------- #
        "embedding_model": embedding_model,

        # -------- RAW GENERIC METADATA -------- #
        "metadata": _json_safe(metadata),
    }

    return _drop_empty(payload)


def chunk_to_point(
    chunk: Chunk,
    *,
    embedding_model: Optional[str] = None,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
) -> Dict[str, Any]:
    """
    Convert Chunk -> Qdrant point shell without vector.

    Kept for compatibility with the older store layer.
    Newer store code should prefer build_vector_points(...).
    """

    payload = chunk_to_payload(
        chunk,
        embedding_model=embedding_model,
        schema_version=schema_version,
    )

    return {
        "id": _generate_point_id(chunk, payload),
        "payload": payload,
    }


# ---------------- BULK CONVERSION ---------------- #


def chunks_to_points(
    chunks: Sequence[Chunk],
    *,
    embedding_model: Optional[str] = None,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
) -> Dict[str, List[Any]]:
    """
    Convert chunks into ids and payloads.

    This keeps backward compatibility with old Qdrant upsert code.
    """

    ids: List[str] = []
    payloads: List[Payload] = []

    for chunk in chunks:
        point = chunk_to_point(
            chunk,
            embedding_model=embedding_model,
            schema_version=schema_version,
        )
        ids.append(point["id"])
        payloads.append(point["payload"])

    return {
        "ids": ids,
        "payloads": payloads,
    }


def build_vector_points(
    chunks: Sequence[Chunk],
    vectors: Sequence[Vector],
    *,
    embedding_model: Optional[str] = None,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
) -> List[VectorPoint]:
    """
    Convert validated chunk/vector pairs into store-ready VectorPoint objects.

    The caller must validate vector dimensions before calling this function.
    """

    if len(chunks) != len(vectors):
        raise ValueError(
            f"Cannot build vector points: chunks={len(chunks)} "
            f"vectors={len(vectors)}."
        )

    points: List[VectorPoint] = []

    for chunk, vector in zip(chunks, vectors):
        payload = chunk_to_payload(
            chunk,
            embedding_model=embedding_model,
            schema_version=schema_version,
        )

        points.append(
            VectorPoint(
                id=_generate_point_id(chunk, payload),
                vector=list(vector),
                payload=payload,
            )
        )

    return points


# ---------------- ID GENERATION ---------------- #


def _generate_point_id(chunk: Chunk, payload: Payload) -> str:
    """
    Generate a deterministic UUID for Qdrant.

    Same source + same chunk identity + same text content should overwrite the
    same point during re-ingestion instead of creating duplicates.
    """

    stable_parts = {
        "source": payload.get("source"),
        "doc_id": payload.get("doc_id"),
        "chunk_id": payload.get("chunk_id"),
        "heading": payload.get("heading"),
        "context": payload.get("context"),
        "chunk_type": payload.get("chunk_type"),
        "subtype": payload.get("subtype"),
        "chunk_index": payload.get("chunk_index"),
        "split_part": payload.get("split_part"),
        "content_hash": payload.get("content_hash"),
    }

    raw = _stable_json(stable_parts)
    digest = hashlib.sha256(raw.encode("utf-8")).digest()[:16]
    return str(uuid.UUID(bytes=digest))


def _chunk_identity(chunk: Chunk) -> str:
    """
    Return an existing chunk id if your chunking layer already has one.
    Otherwise derive a compact stable identity from generic chunk fields.
    """

    existing = _first_present(
        getattr(chunk, "id", None),
        getattr(chunk, "chunk_id", None),
        _safe_mapping(chunk.metadata).get("chunk_id"),
    )

    if existing:
        return str(existing)

    metadata = _safe_mapping(chunk.metadata)

    identity_parts = {
        "source": _as_str(chunk.source),
        "heading": _as_optional_str(chunk.heading),
        "context": _as_optional_str(chunk.context),
        "chunk_type": _first_present(
            getattr(chunk, "type", None),
            getattr(chunk, "chunk_type", None),
            metadata.get("type"),
            metadata.get("chunk_type"),
        ),
        "subtype": _first_present(
            getattr(chunk, "subtype", None),
            metadata.get("subtype"),
        ),
        "chunk_index": _first_present(
            getattr(chunk, "index", None),
            getattr(chunk, "chunk_index", None),
            metadata.get("chunk_index"),
            metadata.get("order"),
        ),
        "split_part": _first_present(
            getattr(chunk, "split_part", None),
            metadata.get("split_part"),
        ),
        "text_hash": _content_hash(_as_str(chunk.text)),
    }

    return hashlib.sha1(_stable_json(identity_parts).encode("utf-8")).hexdigest()


def _generate_doc_id(*, source: str, file_name: Optional[str]) -> str:
    base = source or file_name or "unknown_source"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def _content_hash(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ---------------- NORMALIZATION HELPERS ---------------- #


def _safe_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)

    return {}


def _extract_file_name(source: str) -> Optional[str]:
    if not source:
        return None

    normalized = source.replace("\\", "/")
    return PurePath(normalized).name or None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            if isinstance(value, str) and not value.strip():
                continue
            return value

    return None


def _as_str(value: Any) -> str:
    if value is None:
        return ""

    return str(value)


def _as_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def _as_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]

    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]

    return [_json_safe(value)]


def _json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(inner_value)
            for key, inner_value in value.items()
            if inner_value is not None
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]

    return str(value)


def _stable_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _drop_empty(payload: Payload) -> Payload:
    """
    Keep payload lean while preserving important zero values.

    Removes:
      - None
      - empty strings
      - empty lists
      - empty dicts
    """

    cleaned: Payload = {}

    for key, value in payload.items():
        if value is None:
            continue

        if value == "":
            continue

        if value == []:
            continue

        if value == {}:
            continue

        cleaned[key] = value

    return cleaned