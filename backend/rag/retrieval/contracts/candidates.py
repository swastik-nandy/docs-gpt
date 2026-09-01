from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

from rag.retrieval.contracts.config import (
    RetrievalContractConfig,
    get_retrieval_contract_config,
)
from rag.retrieval.contracts.enums import CandidateOrigin
from rag.retrieval.contracts.primitives import (
    Payload,
    optional_payload_str,
    validate_enum_instance,
    validate_finite_number,
    validate_mapping_size,
    validate_non_empty_str,
    validate_optional_clean_str,
    validate_optional_sequence_limit,
    validate_positive_int,
)


@dataclass(frozen=True)
class CandidateScore:
    value: float
    source: str
    rank: int | None = None
    metadata: Payload = field(default_factory=dict)

    def normalized(self) -> Self:
        return type(self)(
            value=float(self.value),
            source=self.source.strip(),
            rank=self.rank,
            metadata=dict(self.metadata),
        )

    def validate(
        self,
        *,
        max_metadata_keys: int | None,
    ) -> None:
        validate_finite_number("candidate_score.value", self.value)
        validate_non_empty_str("candidate_score.source", self.source)

        if self.rank is not None:
            validate_positive_int("candidate_score.rank", self.rank)

        validate_mapping_size(
            "candidate_score.metadata",
            self.metadata,
            max_metadata_keys,
        )


@dataclass(frozen=True)
class CandidateSourceRef:
    tenant_id: str
    point_id: str
    source_id: str | None = None
    job_id: str | None = None
    parser_doc_id: str | None = None
    section_id: str | None = None
    chunk_id: str | None = None
    text_sha256: str | None = None

    @classmethod
    def from_payload(
        cls,
        *,
        point_id: str,
        payload: Payload,
    ) -> Self:
        return cls(
            tenant_id=str(payload.get("tenant_id", "")).strip(),
            point_id=point_id,
            source_id=optional_payload_str(payload, "source_id"),
            job_id=optional_payload_str(payload, "job_id"),
            parser_doc_id=optional_payload_str(payload, "parser_doc_id"),
            section_id=optional_payload_str(payload, "section_id"),
            chunk_id=optional_payload_str(payload, "chunk_id"),
            text_sha256=optional_payload_str(payload, "text_sha256"),
        )

    def normalized(self) -> Self:
        return type(self)(
            tenant_id=self.tenant_id.strip(),
            point_id=self.point_id.strip(),
            source_id=_clean_optional(self.source_id),
            job_id=_clean_optional(self.job_id),
            parser_doc_id=_clean_optional(self.parser_doc_id),
            section_id=_clean_optional(self.section_id),
            chunk_id=_clean_optional(self.chunk_id),
            text_sha256=_clean_optional(self.text_sha256),
        )

    def validate(self) -> None:
        validate_non_empty_str("candidate_source_ref.tenant_id", self.tenant_id)
        validate_non_empty_str("candidate_source_ref.point_id", self.point_id)
        validate_optional_clean_str("candidate_source_ref.source_id", self.source_id)
        validate_optional_clean_str("candidate_source_ref.job_id", self.job_id)
        validate_optional_clean_str(
            "candidate_source_ref.parser_doc_id",
            self.parser_doc_id,
        )
        validate_optional_clean_str("candidate_source_ref.section_id", self.section_id)
        validate_optional_clean_str("candidate_source_ref.chunk_id", self.chunk_id)
        validate_optional_clean_str(
            "candidate_source_ref.text_sha256",
            self.text_sha256,
        )


@dataclass(frozen=True)
class RetrievalCandidate:
    ref: CandidateSourceRef
    text: str
    score: float
    origin: CandidateOrigin
    payload: Payload = field(default_factory=dict)
    rank: int | None = None
    heading: str | None = None
    heading_path: tuple[str, ...] = field(default_factory=tuple)
    chunk_type: str | None = None
    metadata: Payload = field(default_factory=dict)

    @classmethod
    def from_qdrant_payload(
        cls,
        *,
        point_id: str,
        score: float,
        origin: CandidateOrigin,
        payload: Payload,
        rank: int | None = None,
        metadata: Payload | None = None,
    ) -> Self:
        return cls(
            ref=CandidateSourceRef.from_payload(point_id=point_id, payload=payload),
            text=str(payload.get("text", "")).strip(),
            score=score,
            origin=origin,
            payload=dict(payload),
            rank=rank,
            heading=optional_payload_str(payload, "heading"),
            heading_path=_payload_heading_path(payload),
            chunk_type=optional_payload_str(payload, "chunk_type"),
            metadata=metadata or {},
        )

    @property
    def point_id(self) -> str:
        return self.ref.point_id

    @property
    def chunk_id(self) -> str | None:
        return self.ref.chunk_id

    @property
    def source_id(self) -> str | None:
        return self.ref.source_id

    def normalized(self) -> Self:
        return type(self)(
            ref=self.ref.normalized(),
            text=self.text.strip(),
            score=float(self.score),
            origin=self.origin,
            payload=dict(self.payload),
            rank=self.rank,
            heading=_clean_optional(self.heading),
            heading_path=tuple(
                item.strip() for item in self.heading_path if item.strip()
            ),
            chunk_type=_clean_optional(self.chunk_type),
            metadata=dict(self.metadata),
        )

    def validate(
        self,
        *,
        config: RetrievalContractConfig | None = None,
    ) -> None:
        resolved_config = config or get_retrieval_contract_config()
        max_metadata_keys = resolved_config.request_limits.max_metadata_keys

        if not isinstance(self.ref, CandidateSourceRef):
            raise ValueError("retrieval_candidate.ref must be CandidateSourceRef.")

        self.ref.validate()
        validate_non_empty_str("retrieval_candidate.text", self.text)
        validate_finite_number("retrieval_candidate.score", self.score)
        validate_enum_instance(
            "retrieval_candidate.origin",
            self.origin,
            CandidateOrigin,
        )

        if self.rank is not None:
            validate_positive_int("retrieval_candidate.rank", self.rank)

        validate_optional_clean_str("retrieval_candidate.heading", self.heading)
        validate_optional_clean_str("retrieval_candidate.chunk_type", self.chunk_type)

        validate_optional_sequence_limit(
            "retrieval_candidate.heading_path",
            self.heading_path,
            resolved_config.source_constraints.max_heading_path_depth,
        )

        for index, heading in enumerate(self.heading_path):
            validate_non_empty_str(
                f"retrieval_candidate.heading_path[{index}]",
                heading,
            )

        validate_mapping_size("retrieval_candidate.payload", self.payload, None)
        validate_mapping_size(
            "retrieval_candidate.metadata",
            self.metadata,
            max_metadata_keys,
        )


@dataclass(frozen=True)
class FusedCandidate:
    candidate: RetrievalCandidate
    fused_score: float
    origins: tuple[CandidateOrigin, ...]
    component_scores: tuple[CandidateScore, ...] = field(default_factory=tuple)
    rank: int | None = None
    metadata: Payload = field(default_factory=dict)

    @property
    def point_id(self) -> str:
        return self.candidate.point_id

    def normalized(self) -> Self:
        return type(self)(
            candidate=self.candidate.normalized(),
            fused_score=float(self.fused_score),
            origins=tuple(self.origins),
            component_scores=tuple(
                score.normalized() for score in self.component_scores
            ),
            rank=self.rank,
            metadata=dict(self.metadata),
        )

    def validate(
        self,
        *,
        config: RetrievalContractConfig | None = None,
    ) -> None:
        resolved_config = config or get_retrieval_contract_config()
        max_metadata_keys = resolved_config.request_limits.max_metadata_keys

        if not isinstance(self.candidate, RetrievalCandidate):
            raise ValueError("fused_candidate.candidate must be RetrievalCandidate.")

        self.candidate.validate(config=resolved_config)
        validate_finite_number("fused_candidate.fused_score", self.fused_score)

        if not self.origins:
            raise ValueError("fused_candidate.origins cannot be empty.")

        for index, origin in enumerate(self.origins):
            validate_enum_instance(
                f"fused_candidate.origins[{index}]",
                origin,
                CandidateOrigin,
            )

        for index, score in enumerate(self.component_scores):
            if not isinstance(score, CandidateScore):
                raise ValueError(
                    f"fused_candidate.component_scores[{index}] "
                    "must be CandidateScore."
                )

            score.validate(max_metadata_keys=max_metadata_keys)

        if self.rank is not None:
            validate_positive_int("fused_candidate.rank", self.rank)

        validate_mapping_size(
            "fused_candidate.metadata",
            self.metadata,
            max_metadata_keys,
        )


@dataclass(frozen=True)
class RerankedCandidate:
    candidate: RetrievalCandidate
    rerank_score: float
    rank: int
    reranker: str
    original_score: float | None = None
    explanation: str | None = None
    metadata: Payload = field(default_factory=dict)

    @property
    def point_id(self) -> str:
        return self.candidate.point_id

    @property
    def text(self) -> str:
        return self.candidate.text

    def normalized(self) -> Self:
        return type(self)(
            candidate=self.candidate.normalized(),
            rerank_score=float(self.rerank_score),
            rank=self.rank,
            reranker=self.reranker.strip(),
            original_score=(
                float(self.original_score)
                if self.original_score is not None
                else None
            ),
            explanation=_clean_optional(self.explanation),
            metadata=dict(self.metadata),
        )

    def validate(
        self,
        *,
        config: RetrievalContractConfig | None = None,
    ) -> None:
        resolved_config = config or get_retrieval_contract_config()
        max_metadata_keys = resolved_config.request_limits.max_metadata_keys

        if not isinstance(self.candidate, RetrievalCandidate):
            raise ValueError(
                "reranked_candidate.candidate must be RetrievalCandidate."
            )

        self.candidate.validate(config=resolved_config)
        validate_finite_number("reranked_candidate.rerank_score", self.rerank_score)
        validate_positive_int("reranked_candidate.rank", self.rank)
        validate_non_empty_str("reranked_candidate.reranker", self.reranker)
        validate_optional_clean_str(
            "reranked_candidate.explanation",
            self.explanation,
        )

        if self.original_score is not None:
            validate_finite_number(
                "reranked_candidate.original_score",
                self.original_score,
            )

        validate_mapping_size(
            "reranked_candidate.metadata",
            self.metadata,
            max_metadata_keys,
        )


def _payload_heading_path(payload: Payload) -> tuple[str, ...]:
    value = payload.get("heading_path")

    if value is None:
        return ()

    if isinstance(value, str):
        cleaned = value.strip()
        return (cleaned,) if cleaned else ()

    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())

    return ()


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None

    cleaned = value.strip()
    return cleaned or None