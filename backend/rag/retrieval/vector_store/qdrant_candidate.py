from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Mapping, Sequence

from rag.retrieval.contracts.candidates import (
    CandidateOrigin,
    CandidateSourceRef,
    RetrievalCandidate,
)
from rag.retrieval.vector_store.qdrant_filter import QdrantPayloadKeys


Payload = dict[str, Any]


@dataclass(frozen=True)
class QdrantCandidateMapperConfig:
    payload_keys: QdrantPayloadKeys = field(default_factory=QdrantPayloadKeys)
    text_payload_keys: tuple[str, ...] = (
        "text",
        "chunk_text",
        "content",
        "page_content",
        "markdown",
    )
    preserve_payload: bool = True

    def validate(self) -> None:
        if not self.text_payload_keys:
            raise ValueError("text_payload_keys cannot be empty.")


class QdrantCandidateMapper:
    """
    Converts Qdrant scored points into retrieval candidates.

    This mapper keeps file/source metadata attached to candidates so both retrieval
    and generation can understand where evidence came from.
    """

    def __init__(
        self,
        config: QdrantCandidateMapperConfig | None = None,
    ) -> None:
        self.config = config or QdrantCandidateMapperConfig()
        self.config.validate()

    def from_scored_point(
        self,
        point: Any,
        *,
        origin: CandidateOrigin,
        rank: int,
    ) -> RetrievalCandidate:
        payload = _payload_dict(getattr(point, "payload", None))
        point_id = str(getattr(point, "id", ""))
        score = float(getattr(point, "score", 0.0))

        text, missing_text = self._extract_text(payload, point_id=point_id)

        ref = self._build_ref(
            payload=payload,
            point_id=point_id,
        )

        metadata: Payload = {
            "rank": rank,
            "qdrant_point_id": point_id,
            "qdrant_score": score,
            "missing_text_payload": missing_text,
            "origin": origin.value,
        }

        if self.config.preserve_payload:
            metadata["payload"] = payload

        return _build_retrieval_candidate(
            ref=ref,
            text=text,
            score=score,
            origin=origin,
            metadata=metadata,
            payload=payload,
            rank=rank,
        )

    def _build_ref(
        self,
        *,
        payload: Mapping[str, Any],
        point_id: str,
    ) -> CandidateSourceRef:
        keys = self.config.payload_keys

        values: Payload = {
            "tenant_id": _payload_str(payload, keys.tenant_id) or "",
            "point_id": point_id,
            "source_id": _payload_str(payload, keys.source_id),
            "job_id": _payload_str(payload, "job_id"),
            "parser_doc_id": _payload_str(payload, "parser_doc_id")
            or _payload_str(payload, "doc_id")
            or _payload_str(payload, keys.source_doc_id),
            "section_id": _payload_str(payload, keys.section_id),
            "chunk_id": _payload_str(payload, keys.chunk_id) or point_id,
            "text_sha256": _payload_str(payload, "text_sha256"),
            "source_doc_id": _payload_str(payload, keys.source_doc_id),
            "doc_id": _payload_str(payload, keys.doc_id),
            "file_id": _payload_str(payload, keys.file_id),
            "file_name": _payload_str(payload, keys.file_name),
            "file_path": _payload_str(payload, keys.file_path),
        }

        return _construct_dataclass_with_supported_fields(
            CandidateSourceRef,
            values,
        )

    def _extract_text(
        self,
        payload: Mapping[str, Any],
        *,
        point_id: str,
    ) -> tuple[str, bool]:
        for key in self.config.text_payload_keys:
            value = _payload_str(payload, key)
            if value:
                return value, False

        return f"[missing text payload for qdrant point {point_id}]", True


def _build_retrieval_candidate(
    *,
    ref: CandidateSourceRef,
    text: str,
    score: float,
    origin: CandidateOrigin,
    metadata: Payload,
    payload: Payload,
    rank: int,
) -> RetrievalCandidate:
    values: Payload = {
        "ref": ref,
        "text": text,
        "score": score,
        "origin": origin,
        "metadata": metadata,
        "payload": payload,
        "rank": rank,
    }

    candidate = _construct_dataclass_with_supported_fields(
        RetrievalCandidate,
        values,
    )
    candidate.validate()

    return candidate


def _construct_dataclass_with_supported_fields(
    cls: type[Any],
    values: Mapping[str, Any],
) -> Any:
    supported_fields = _dataclass_field_names(cls)

    if not supported_fields:
        return cls(**dict(values))

    kwargs = {
        key: value
        for key, value in values.items()
        if key in supported_fields
    }

    return cls(**kwargs)


def _dataclass_field_names(cls: type[Any]) -> set[str]:
    if not is_dataclass(cls):
        return set()

    return {field.name for field in fields(cls)}


def _payload_dict(value: Any) -> Payload:
    if value is None:
        return {}

    if isinstance(value, Mapping):
        return dict(value)

    return {}


def _payload_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)

    if value is None:
        return None

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return None

        value = value[0]

    cleaned = str(value).strip()
    return cleaned or None
