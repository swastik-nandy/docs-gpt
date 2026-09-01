from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Mapping, Sequence

from rag.retrieval.config.runtime import (
    RetrievalRuntimeConfig,
    load_retrieval_runtime_config,
)
from rag.retrieval.contracts.candidates import (
    CandidateOrigin,
    FusedCandidate,
    RetrievalCandidate,
)
from rag.retrieval.contracts.query import NormalizedQuery
from rag.retrieval.contracts.request import RetrievalRequest


Payload = dict[str, Any]


@dataclass(frozen=True)
class RrfFusionConfig:
    k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    limit_multiplier: int = 1

    @classmethod
    def from_runtime_config(
        cls,
        runtime_config: RetrievalRuntimeConfig,
    ) -> RrfFusionConfig:
        config = cls(
            k=runtime_config.rrf.k,
            dense_weight=runtime_config.rrf.dense_weight,
            sparse_weight=runtime_config.rrf.sparse_weight,
            limit_multiplier=runtime_config.rrf.limit_multiplier,
        )
        config.validate()

        return config

    def validate(self) -> None:
        if self.k <= 0:
            raise ValueError("k must be greater than zero.")

        if self.dense_weight < 0:
            raise ValueError("dense_weight cannot be negative.")

        if self.sparse_weight < 0:
            raise ValueError("sparse_weight cannot be negative.")

        if self.limit_multiplier <= 0:
            raise ValueError("limit_multiplier must be greater than zero.")


@dataclass(frozen=True)
class _FusionAccumulator:
    candidate: RetrievalCandidate
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float = 0.0

    @property
    def origins(self) -> tuple[CandidateOrigin, ...]:
        values: list[CandidateOrigin] = []

        if self.dense_rank is not None:
            values.append(CandidateOrigin.DENSE)

        if self.sparse_rank is not None:
            values.append(CandidateOrigin.SPARSE)

        return tuple(values)

    @property
    def source_count(self) -> int:
        return len(self.origins)


class RrfCandidateFusion:
    """
    Reciprocal Rank Fusion for dense + sparse retrieval candidates.

    Dense and sparse scores can have different scales, so RRF trusts rank order
    more than raw score magnitude.
    """

    def __init__(
        self,
        config: RrfFusionConfig | None = None,
        *,
        runtime_config: RetrievalRuntimeConfig | None = None,
    ) -> None:
        resolved_runtime = runtime_config or load_retrieval_runtime_config()
        self.config = config or RrfFusionConfig.from_runtime_config(resolved_runtime)
        self.config.validate()

    async def fuse(
        self,
        *,
        dense_candidates: Sequence[RetrievalCandidate],
        sparse_candidates: Sequence[RetrievalCandidate],
        request: RetrievalRequest,
        normalized_query: NormalizedQuery,
    ) -> Sequence[FusedCandidate]:
        request.validate()
        normalized_query.validate()

        for candidate in dense_candidates:
            candidate.validate()

        for candidate in sparse_candidates:
            candidate.validate()

        accumulators: dict[str, _FusionAccumulator] = {}

        self._add_candidates(
            accumulators=accumulators,
            candidates=dense_candidates,
            origin=CandidateOrigin.DENSE,
        )
        self._add_candidates(
            accumulators=accumulators,
            candidates=sparse_candidates,
            origin=CandidateOrigin.SPARSE,
        )

        ranked = sorted(
            accumulators.values(),
            key=lambda item: (
                -item.fused_score,
                -item.source_count,
                -float(getattr(item.candidate, "score", 0.0)),
                _candidate_identity(item.candidate),
            ),
        )

        limit = max(1, request.top_k * self.config.limit_multiplier)

        fused_candidates = tuple(
            _build_fused_candidate(
                accumulator=accumulator,
                rank=index + 1,
            )
            for index, accumulator in enumerate(ranked[:limit])
        )

        for candidate in fused_candidates:
            candidate.validate()

        return fused_candidates

    def _add_candidates(
        self,
        *,
        accumulators: dict[str, _FusionAccumulator],
        candidates: Sequence[RetrievalCandidate],
        origin: CandidateOrigin,
    ) -> None:
        for index, candidate in enumerate(candidates, start=1):
            key = _candidate_identity(candidate)
            existing = accumulators.get(key)

            if existing is None:
                existing = _FusionAccumulator(candidate=candidate)

            contribution = self._rrf_contribution(
                rank=index,
                origin=origin,
            )

            if origin == CandidateOrigin.DENSE:
                updated = _FusionAccumulator(
                    candidate=_prefer_candidate(existing.candidate, candidate),
                    dense_rank=index,
                    sparse_rank=existing.sparse_rank,
                    dense_score=candidate.score,
                    sparse_score=existing.sparse_score,
                    fused_score=existing.fused_score + contribution,
                )
            elif origin == CandidateOrigin.SPARSE:
                updated = _FusionAccumulator(
                    candidate=_prefer_candidate(existing.candidate, candidate),
                    dense_rank=existing.dense_rank,
                    sparse_rank=index,
                    dense_score=existing.dense_score,
                    sparse_score=candidate.score,
                    fused_score=existing.fused_score + contribution,
                )
            else:
                updated = existing

            accumulators[key] = updated

    def _rrf_contribution(
        self,
        *,
        rank: int,
        origin: CandidateOrigin,
    ) -> float:
        if origin == CandidateOrigin.DENSE:
            weight = self.config.dense_weight
        elif origin == CandidateOrigin.SPARSE:
            weight = self.config.sparse_weight
        else:
            weight = 1.0

        return weight * (1.0 / (self.config.k + rank))


def _build_fused_candidate(
    *,
    accumulator: _FusionAccumulator,
    rank: int,
) -> FusedCandidate:
    metadata: Payload = {
        "rank": rank,
        "dense_rank": accumulator.dense_rank,
        "sparse_rank": accumulator.sparse_rank,
        "dense_score": accumulator.dense_score,
        "sparse_score": accumulator.sparse_score,
        "fused_score": accumulator.fused_score,
        "origins": [origin.value for origin in accumulator.origins],
        "fusion": "rrf",
    }

    values: Payload = {
        "candidate": accumulator.candidate,
        "fused_score": accumulator.fused_score,
        "score": accumulator.fused_score,
        "rank": rank,
        "origin": CandidateOrigin.RRF,
        "origins": accumulator.origins,
        "dense_rank": accumulator.dense_rank,
        "sparse_rank": accumulator.sparse_rank,
        "dense_score": accumulator.dense_score,
        "sparse_score": accumulator.sparse_score,
        "metadata": metadata,
    }

    return _construct_dataclass_with_supported_fields(FusedCandidate, values)


def _candidate_identity(candidate: RetrievalCandidate) -> str:
    for attr_name in (
        "chunk_id",
        "point_id",
        "source_doc_id",
        "doc_id",
        "source_id",
    ):
        value = _clean_optional(getattr(candidate, attr_name, None))
        if value:
            return value

    ref = getattr(candidate, "ref", None)
    if ref is not None:
        for attr_name in (
            "chunk_id",
            "point_id",
            "source_doc_id",
            "doc_id",
            "source_id",
        ):
            value = _clean_optional(getattr(ref, attr_name, None))
            if value:
                return value

    metadata = getattr(candidate, "metadata", {}) or {}
    if isinstance(metadata, Mapping):
        payload = metadata.get("payload", {})
        if isinstance(payload, Mapping):
            for key in (
                "chunk_id",
                "point_id",
                "source_doc_id",
                "doc_id",
                "source_id",
                "file_path",
                "file_name",
            ):
                value = _clean_optional(payload.get(key))
                if value:
                    return value

    text = _clean_optional(getattr(candidate, "text", None))
    if text:
        return text[:200]

    return str(id(candidate))


def _prefer_candidate(
    current: RetrievalCandidate,
    incoming: RetrievalCandidate,
) -> RetrievalCandidate:
    if incoming.score > current.score:
        return incoming

    current_metadata = getattr(current, "metadata", {}) or {}
    if isinstance(current_metadata, Mapping):
        if current_metadata.get("missing_text_payload") and len(incoming.text) > len(
            current.text
        ):
            return incoming

    return current


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


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None

    cleaned = str(value).strip()
    return cleaned or None
