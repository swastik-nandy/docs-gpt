from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rag.retrieval.config.runtime import (
    RetrievalRuntimeConfig,
    load_retrieval_runtime_config,
)
from rag.retrieval.contracts.candidates import (
    FusedCandidate,
    RerankedCandidate,
    RetrievalCandidate,
)
from rag.retrieval.contracts.query import NormalizedQuery
from rag.retrieval.contracts.request import RetrievalRequest


Payload = dict[str, Any]
CandidateLike = RetrievalCandidate | FusedCandidate


@dataclass(frozen=True)
class ScoreRerankerConfig:
    enabled: bool = True
    max_candidates: int = 50
    normalize_scores: bool = False

    @classmethod
    def from_runtime_config(
        cls,
        runtime_config: RetrievalRuntimeConfig,
    ) -> ScoreRerankerConfig:
        config = cls(
            enabled=runtime_config.score_reranker.enabled,
            max_candidates=runtime_config.score_reranker.max_candidates,
            normalize_scores=runtime_config.score_reranker.normalize_scores,
        )
        config.validate()

        return config

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be boolean.")

        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be greater than zero.")

        if not isinstance(self.normalize_scores, bool):
            raise ValueError("normalize_scores must be boolean.")


class ScoreReranker:
    """
    Lightweight fallback reranker.

    Accepts RetrievalCandidate or FusedCandidate. When given a FusedCandidate,
    it reranks using fused_score and stores the original RetrievalCandidate in
    the RerankedCandidate contract.
    """

    def __init__(
        self,
        config: ScoreRerankerConfig | None = None,
        *,
        runtime_config: RetrievalRuntimeConfig | None = None,
    ) -> None:
        resolved_runtime = runtime_config or load_retrieval_runtime_config()
        self.config = config or ScoreRerankerConfig.from_runtime_config(
            resolved_runtime,
        )
        self.config.validate()

    async def rerank(
        self,
        *,
        candidates: Sequence[CandidateLike],
        request: RetrievalRequest,
        normalized_query: NormalizedQuery,
    ) -> Sequence[RerankedCandidate]:
        request.validate()
        normalized_query.validate()

        for candidate in candidates:
            _validate_candidate_like(candidate)

        if not self.config.enabled:
            return ()

        sorted_candidates = sorted(
            candidates,
            key=lambda item: (
                -_candidate_score(item),
                _candidate_identity(item),
            ),
        )[: self.config.max_candidates]

        scores = tuple(_candidate_score(candidate) for candidate in sorted_candidates)

        reranked = tuple(
            _build_reranked_candidate(
                candidate_like=candidate,
                rank=index + 1,
                rerank_score=_rerank_score(
                    score=_candidate_score(candidate),
                    scores=scores,
                    normalize=self.config.normalize_scores,
                ),
                metadata={
                    "reranker": "score",
                    "input_type": candidate.__class__.__name__,
                    "original_score": _candidate_score(candidate),
                    "normalized": self.config.normalize_scores,
                },
            )
            for index, candidate in enumerate(sorted_candidates)
        )

        for candidate in reranked:
            candidate.validate()

        return reranked


def _build_reranked_candidate(
    *,
    candidate_like: CandidateLike,
    rank: int,
    rerank_score: float,
    metadata: Payload,
) -> RerankedCandidate:
    return RerankedCandidate(
        candidate=_base_candidate(candidate_like),
        rerank_score=rerank_score,
        rank=rank,
        reranker="score",
        original_score=_candidate_score(candidate_like),
        metadata=metadata,
    )


def _rerank_score(
    *,
    score: float,
    scores: Sequence[float],
    normalize: bool,
) -> float:
    if not normalize:
        return score

    if not scores:
        return score

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return 1.0

    return (score - min_score) / (max_score - min_score)


def _base_candidate(candidate_like: CandidateLike) -> RetrievalCandidate:
    nested = getattr(candidate_like, "candidate", None)

    if nested is not None and hasattr(nested, "text"):
        return nested

    return candidate_like  # type: ignore[return-value]


def _candidate_score(candidate_like: CandidateLike) -> float:
    fused_score = getattr(candidate_like, "fused_score", None)

    if fused_score is not None:
        return float(fused_score)

    return float(getattr(candidate_like, "score", 0.0))


def _validate_candidate_like(candidate_like: CandidateLike) -> None:
    validate = getattr(candidate_like, "validate", None)
    if callable(validate):
        validate()

    base = _base_candidate(candidate_like)
    base.validate()


def _candidate_identity(candidate_like: CandidateLike) -> str:
    candidate = _base_candidate(candidate_like)

    for attr_name in (
        "chunk_id",
        "point_id",
        "source_id",
        "parser_doc_id",
        "section_id",
    ):
        value = _clean_optional(getattr(candidate, attr_name, None))
        if value:
            return value

    ref = getattr(candidate, "ref", None)
    if ref is not None:
        for attr_name in (
            "chunk_id",
            "point_id",
            "source_id",
            "parser_doc_id",
            "section_id",
        ):
            value = _clean_optional(getattr(ref, attr_name, None))
            if value:
                return value

    for item in (candidate_like, candidate):
        metadata = getattr(item, "metadata", {}) or {}
        if isinstance(metadata, Mapping):
            payload = metadata.get("payload", {})
            if isinstance(payload, Mapping):
                for key in (
                    "chunk_id",
                    "point_id",
                    "source_id",
                    "parser_doc_id",
                    "section_id",
                    "file_path",
                    "file_name",
                ):
                    value = _clean_optional(payload.get(key))
                    if value:
                        return value

    return _clean_optional(getattr(candidate, "text", None)) or str(id(candidate))


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None

    cleaned = str(value).strip()
    return cleaned or None
