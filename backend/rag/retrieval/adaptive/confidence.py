from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from rag.retrieval.adaptive.gain import candidate_score_entropy, candidate_score_gap
from rag.retrieval.config.runtime import RetrievalRuntimeConfig
from rag.retrieval.adaptive.state import (
    AdaptiveRetrievalState,
    RetrievalConfidence,
    candidate_identity,
)
from rag.retrieval.contracts.primitives import (
    Payload,
    validate_finite_number,
    validate_mapping_size,
)


class ConfidenceEstimator(Protocol):
    def estimate(self, state: AdaptiveRetrievalState) -> RetrievalConfidence:
        raise NotImplementedError


@dataclass(frozen=True)
class StaticConfidenceConfig:
    score_gap_weight: float = 0.34
    exact_coverage_weight: float = 0.26
    source_coverage_weight: float = 0.18
    candidate_presence_weight: float = 0.12
    entropy_weight: float = 0.10

    high_entropy_penalty: float = 0.18
    low_corpus_affinity_penalty: float = 0.18
    no_candidate_penalty: float = 0.45

    @classmethod
    def from_runtime_config(
        cls,
        runtime_config: RetrievalRuntimeConfig,
    ) -> StaticConfidenceConfig:
        config = runtime_config.adaptive_confidence
        return cls(
            score_gap_weight=config.score_gap_weight,
            exact_coverage_weight=config.exact_coverage_weight,
            source_coverage_weight=config.source_coverage_weight,
            candidate_presence_weight=config.candidate_presence_weight,
            entropy_weight=config.entropy_weight,
            high_entropy_penalty=config.high_entropy_penalty,
            low_corpus_affinity_penalty=config.low_corpus_affinity_penalty,
            no_candidate_penalty=config.no_candidate_penalty,
        )

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            validate_finite_number(f"static_confidence_config.{name}", value)
            if value < 0:
                raise ValueError(f"static_confidence_config.{name} cannot be negative.")


@dataclass(frozen=True)
class StaticConfidenceEstimator:
    config: StaticConfidenceConfig = StaticConfidenceConfig()

    @classmethod
    def from_runtime_config(
        cls,
        runtime_config: RetrievalRuntimeConfig,
    ) -> StaticConfidenceEstimator:
        runtime_config.validate()
        return cls(
            config=StaticConfidenceConfig.from_runtime_config(runtime_config),
        )

    def estimate(self, state: AdaptiveRetrievalState) -> RetrievalConfidence:
        state.validate()
        self.config.validate()

        active_candidates = state.active_candidates()
        has_candidates = bool(active_candidates)

        score_gap = candidate_score_gap(state)
        score_entropy = candidate_score_entropy(state)

        exact_coverage = estimate_exact_coverage(state)
        source_coverage = estimate_source_coverage(active_candidates)
        candidate_presence = 1.0 if has_candidates else 0.0

        normalized_gap = score_gap if score_gap is not None else 0.0
        normalized_entropy_confidence = (
            1.0 - score_entropy
            if score_entropy is not None
            else 0.0
        )

        value = (
            self.config.score_gap_weight * normalized_gap
            + self.config.exact_coverage_weight * exact_coverage
            + self.config.source_coverage_weight * source_coverage
            + self.config.candidate_presence_weight * candidate_presence
            + self.config.entropy_weight * normalized_entropy_confidence
        )

        penalties: list[str] = []

        if not has_candidates:
            value -= self.config.no_candidate_penalty
            penalties.append("no_candidates")

        if score_entropy is not None and score_entropy >= 0.82:
            value -= self.config.high_entropy_penalty
            penalties.append("high_score_entropy")

        if state.query_signals.corpus_affinity == 0.0:
            value -= self.config.low_corpus_affinity_penalty
            penalties.append("zero_corpus_affinity")

        value = _clamp_probability(value)

        confidence = RetrievalConfidence(
            value=value,
            reason=_confidence_reason(
                value=value,
                has_candidates=has_candidates,
                score_gap=score_gap,
                exact_coverage=exact_coverage,
                source_coverage=source_coverage,
                penalties=penalties,
            ),
            score_gap=score_gap,
            exact_coverage=exact_coverage,
            source_coverage=source_coverage,
            metadata={
                "estimator": "static_confidence",
                "score_entropy": score_entropy,
                "candidate_count": len(active_candidates),
                "penalties": penalties,
            },
        )
        confidence.validate()
        return confidence


def apply_confidence_estimate(
    state: AdaptiveRetrievalState,
    estimator: ConfidenceEstimator | None = None,
) -> RetrievalConfidence:
    estimator = estimator or StaticConfidenceEstimator()
    confidence = estimator.estimate(state)

    state.set_confidence(
        confidence.value,
        reason=confidence.reason,
        score_gap=confidence.score_gap,
        exact_coverage=confidence.exact_coverage,
        source_coverage=confidence.source_coverage,
        metadata=confidence.metadata,
    )
    return confidence


def estimate_exact_coverage(state: AdaptiveRetrievalState, *, max_candidates: int = 10) -> float:
    exact_terms = _exact_terms_from_state(state)
    if not exact_terms:
        return 0.5 if state.has_candidates else 0.0

    haystack = " ".join(
        _candidate_text(candidate)
        for candidate in state.active_candidates()[:max_candidates]
    ).lower()

    if not haystack.strip():
        return 0.0

    hits = sum(1 for term in exact_terms if term.lower() in haystack)
    return _clamp_probability(hits / max(1, len(exact_terms)))


def estimate_source_coverage(candidates: Sequence[Any], *, max_candidates: int = 10) -> float:
    if not candidates:
        return 0.0

    identities = {
        candidate_identity(candidate)
        for candidate in candidates[:max_candidates]
    }

    unique_count = len(identities)
    if unique_count <= 1:
        return 0.35

    if unique_count <= 3:
        return 0.68

    return 1.0


def _exact_terms_from_state(state: AdaptiveRetrievalState) -> tuple[str, ...]:
    metadata = state.query_signals.metadata or {}
    values: list[str] = []

    for key in ("exact_terms", "quoted_terms", "code_like_terms", "file_hints"):
        raw = metadata.get(key)

        if isinstance(raw, str):
            values.append(raw)

        elif isinstance(raw, Sequence):
            values.extend(str(item) for item in raw if str(item).strip())

    cleaned: list[str] = []
    seen: set[str] = set()

    for value in values:
        item = " ".join(str(value).strip().split())
        if not item or item in seen:
            continue

        seen.add(item)
        cleaned.append(item)

    return tuple(cleaned)


def _candidate_text(candidate: Any) -> str:
    base = getattr(candidate, "candidate", candidate)
    text = getattr(base, "text", None)

    if isinstance(text, str) and text.strip():
        return text

    payload = getattr(base, "payload", {}) or {}
    if isinstance(payload, Mapping):
        for key in ("text", "chunk_text", "content", "page_content", "markdown"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value

    return ""


def _confidence_reason(
    *,
    value: float,
    has_candidates: bool,
    score_gap: float | None,
    exact_coverage: float,
    source_coverage: float,
    penalties: Sequence[str],
) -> str:
    if not has_candidates:
        return "low confidence because no active candidates exist"

    parts = [
        f"value={value:.3f}",
        f"score_gap={score_gap if score_gap is not None else 'none'}",
        f"exact_coverage={exact_coverage:.3f}",
        f"source_coverage={source_coverage:.3f}",
    ]

    if penalties:
        parts.append(f"penalties={list(penalties)}")

    return "; ".join(parts)


def _clamp_probability(value: float) -> float:
    validate_finite_number("probability", value)
    return max(0.0, min(1.0, float(value)))
