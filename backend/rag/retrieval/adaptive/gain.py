from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from rag.retrieval.adaptive.actions import RetrievalAction
from rag.retrieval.adaptive.state import AdaptiveRetrievalState
from rag.retrieval.config.runtime import RetrievalRuntimeConfig
from rag.retrieval.contracts.candidates import (
    FusedCandidate,
    RerankedCandidate,
    RetrievalCandidate,
)
from rag.retrieval.contracts.primitives import (
    Payload,
    validate_finite_number,
    validate_mapping_size,
    validate_non_empty_str,
)


@dataclass(frozen=True)
class ActionGainEstimate:
    action: RetrievalAction
    estimated_gain: float
    reason: str
    metadata: Payload = field(default_factory=dict)

    def validate(self, *, max_metadata_keys: int | None = None) -> None:
        if not isinstance(self.action, RetrievalAction):
            raise ValueError("action_gain_estimate.action must be RetrievalAction.")

        validate_finite_number("action_gain_estimate.estimated_gain", self.estimated_gain)
        if not 0 <= self.estimated_gain <= 1:
            raise ValueError("action_gain_estimate.estimated_gain must be between 0 and 1.")

        validate_non_empty_str("action_gain_estimate.reason", self.reason)
        validate_mapping_size(
            "action_gain_estimate.metadata",
            self.metadata,
            max_metadata_keys,
        )


class ActionGainModel(Protocol):
    def estimate(
        self,
        action: RetrievalAction,
        state: AdaptiveRetrievalState,
    ) -> ActionGainEstimate:
        raise NotImplementedError


@dataclass(frozen=True)
class StaticActionGainConfig:
    sparse_base_gain: float = 0.72
    dense_base_gain: float = 0.68
    memory_base_gain: float = 0.70

    fuse_base_gain: float = 0.42
    score_rerank_base_gain: float = 0.18
    ce_rerank_base_gain: float = 0.52
    linked_expand_base_gain: float = 0.35

    verify_evidence_base_gain: float = 0.28
    pack_context_base_gain: float = 0.95

    high_confidence_stop_threshold: float = 0.78
    medium_confidence_threshold: float = 0.55

    strong_score_gap_threshold: float = 0.22
    flat_score_gap_threshold: float = 0.08
    entropy_high_threshold: float = 0.82

    min_gain: float = 0.0
    max_gain: float = 1.0

    @classmethod
    def from_runtime_config(
        cls,
        runtime_config: RetrievalRuntimeConfig,
    ) -> StaticActionGainConfig:
        config = runtime_config.adaptive_gain
        return cls(
            sparse_base_gain=config.sparse_base_gain,
            dense_base_gain=config.dense_base_gain,
            memory_base_gain=config.memory_base_gain,
            fuse_base_gain=config.fuse_base_gain,
            score_rerank_base_gain=config.score_rerank_base_gain,
            ce_rerank_base_gain=config.ce_rerank_base_gain,
            linked_expand_base_gain=config.linked_expand_base_gain,
            verify_evidence_base_gain=config.verify_evidence_base_gain,
            pack_context_base_gain=config.pack_context_base_gain,
            high_confidence_stop_threshold=config.high_confidence_stop_threshold,
            medium_confidence_threshold=config.medium_confidence_threshold,
            strong_score_gap_threshold=config.strong_score_gap_threshold,
            flat_score_gap_threshold=config.flat_score_gap_threshold,
            entropy_high_threshold=config.entropy_high_threshold,
            min_gain=config.min_gain,
            max_gain=config.max_gain,
        )

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            _validate_probability(f"static_action_gain_config.{name}", value)

        if self.min_gain > self.max_gain:
            raise ValueError("static_action_gain_config.min_gain must be <= max_gain.")


@dataclass(frozen=True)
class StaticActionGainModel:
    config: StaticActionGainConfig = StaticActionGainConfig()

    def estimate(
        self,
        action: RetrievalAction,
        state: AdaptiveRetrievalState,
    ) -> ActionGainEstimate:
        state.validate()
        self.config.validate()

        signals = state.query_signals
        score_gap = candidate_score_gap(state)
        entropy = candidate_score_entropy(state)
        has_candidates = state.has_candidates
        has_links = active_candidates_have_link_hints(state)

        if action == RetrievalAction.MEMORY_LOOKUP:
            gain = self.config.memory_base_gain if signals.likely_needs_memory else 0.08
            return _estimate(
                action,
                gain,
                "memory gain from referential markers, chat history, or memory policy",
                likely_needs_memory=signals.likely_needs_memory,
            )

        if action == RetrievalAction.SPARSE_SEARCH:
            gain = self.config.sparse_base_gain
            gain += 0.22 * signals.lexical_specificity
            gain += 0.12 if signals.is_exactish else 0.0
            gain -= 0.16 if signals.corpus_affinity == 0.0 else 0.0
            return _estimate(
                action,
                gain,
                "sparse gain rises with lexical specificity and exact-shaped terms",
                lexical_specificity=signals.lexical_specificity,
                is_exactish=signals.is_exactish,
                corpus_affinity=signals.corpus_affinity,
            )

        if action == RetrievalAction.DENSE_SEARCH:
            gain = self.config.dense_base_gain
            gain += 0.18 * (1.0 - signals.lexical_specificity)
            gain += 0.14 if signals.is_structural_or_multi_hop else 0.0
            gain -= 0.2 if signals.corpus_affinity == 0.0 else 0.0
            return _estimate(
                action,
                gain,
                "dense gain rises for conceptual, structural, or low-lexical queries",
                lexical_specificity=signals.lexical_specificity,
                structural_or_multi_hop=signals.is_structural_or_multi_hop,
                corpus_affinity=signals.corpus_affinity,
            )

        if action == RetrievalAction.FUSE:
            has_multiple_sources = sum(
                1
                for count in (
                    len(state.memory_candidates),
                    len(state.sparse_candidates),
                    len(state.dense_candidates),
                )
                if count > 0
            ) >= 2
            gain = self.config.fuse_base_gain if has_multiple_sources else 0.14
            return _estimate(
                action,
                gain,
                "fusion gain depends on whether multiple candidate channels exist",
                has_multiple_sources=has_multiple_sources,
            )

        if action == RetrievalAction.SCORE_RERANK:
            gain = self.config.score_rerank_base_gain
            gain += 0.1 if has_candidates else 0.0
            gain -= 0.08 if score_gap and score_gap > self.config.strong_score_gap_threshold else 0.0
            return _estimate(
                action,
                gain,
                "score rerank is cheap but usually low-gain",
                has_candidates=has_candidates,
                score_gap=score_gap,
            )

        if action == RetrievalAction.CE_RERANK:
            gain = self.config.ce_rerank_base_gain
            gain += 0.22 if score_gap is not None and score_gap <= self.config.flat_score_gap_threshold else 0.0
            gain += 0.16 if entropy is not None and entropy >= self.config.entropy_high_threshold else 0.0
            gain += 0.12 if signals.is_structural_or_multi_hop else 0.0
            gain -= 0.22 if score_gap is not None and score_gap >= self.config.strong_score_gap_threshold else 0.0
            gain -= 0.3 if not has_candidates else 0.0
            return _estimate(
                action,
                gain,
                "CE gain rises when candidate scores are flat or ambiguous",
                score_gap=score_gap,
                score_entropy=entropy,
                structural_or_multi_hop=signals.is_structural_or_multi_hop,
                has_candidates=has_candidates,
            )

        if action == RetrievalAction.LINKED_EXPAND:
            gain = self.config.linked_expand_base_gain
            gain += 0.28 if has_links else -0.3
            gain += 0.1 if state.confidence.is_medium else 0.0
            gain -= 0.22 if state.confidence.is_low else 0.0
            return _estimate(
                action,
                gain,
                "linked expansion gain depends on link hints and seed confidence",
                has_links=has_links,
                confidence=state.confidence.value,
            )

        if action == RetrievalAction.ESCALATE_K:
            gain = 0.38 if state.confidence.is_low and has_candidates else 0.08
            return _estimate(
                action,
                gain,
                "candidate widening is useful only after low-confidence retrieval",
                confidence=state.confidence.value,
                has_candidates=has_candidates,
            )

        if action == RetrievalAction.ESCALATE_DENSE:
            gain = 0.58 if state.confidence.is_low and not state.dense_candidates else 0.05
            gain += 0.15 if signals.lexical_specificity < 0.45 else 0.0
            return _estimate(
                action,
                gain,
                "dense escalation is useful after sparse-only low-confidence retrieval",
                confidence=state.confidence.value,
                lexical_specificity=signals.lexical_specificity,
                has_dense=bool(state.dense_candidates),
            )

        if action == RetrievalAction.VERIFY_EVIDENCE:
            gain = self.config.verify_evidence_base_gain
            gain += 0.22 if signals.is_exactish else 0.0
            gain += 0.12 if signals.is_structural_or_multi_hop else 0.0
            gain -= 0.25 if not has_candidates else 0.0
            return _estimate(
                action,
                gain,
                "verification gain rises for exact, structural, or risky queries",
                is_exactish=signals.is_exactish,
                structural_or_multi_hop=signals.is_structural_or_multi_hop,
                has_candidates=has_candidates,
            )

        if action == RetrievalAction.PACK_CONTEXT:
            gain = self.config.pack_context_base_gain if has_candidates else 0.0
            gain += 0.02 if state.confidence.is_high else 0.0
            return _estimate(
                action,
                gain,
                "packing is high gain once usable candidates exist",
                has_candidates=has_candidates,
                confidence=state.confidence.value,
            )

        if action == RetrievalAction.STOP:
            gain = 1.0 if state.context_blocks or state.confidence.is_high else 0.0
            return _estimate(
                action,
                gain,
                "stop gain is high after context exists or confidence is high",
                has_context=bool(state.context_blocks),
                confidence=state.confidence.value,
            )

        if action == RetrievalAction.FALLBACK_NO_ANSWER:
            gain = 0.82 if not has_candidates and state.hop > 0 else 0.0
            gain += 0.15 if signals.corpus_affinity == 0.0 else 0.0
            return _estimate(
                action,
                gain,
                "fallback gain rises when no candidates exist after attempted retrieval",
                has_candidates=has_candidates,
                hop=state.hop,
                corpus_affinity=signals.corpus_affinity,
            )

        raise ValueError(f"Unsupported retrieval action for gain estimation: {action}")


def candidate_score_gap(state: AdaptiveRetrievalState) -> float | None:
    scores = candidate_scores(state.active_candidates())
    if len(scores) < 2:
        return None

    sorted_scores = sorted(scores, reverse=True)
    top = sorted_scores[0]
    second = sorted_scores[1]

    if top <= 0:
        return 0.0

    return _clamp_probability((top - second) / max(abs(top), 1e-9))


def candidate_score_entropy(state: AdaptiveRetrievalState) -> float | None:
    scores = candidate_scores(state.active_candidates())
    if len(scores) < 2:
        return None

    positive_scores = [max(0.0, score) for score in scores]
    total = sum(positive_scores)

    if total <= 0:
        return 1.0

    probabilities = [score / total for score in positive_scores if score > 0]
    if len(probabilities) <= 1:
        return 0.0

    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    max_entropy = math.log(len(probabilities))
    return _clamp_probability(entropy / max_entropy)


def candidate_scores(candidates: tuple[object, ...]) -> tuple[float, ...]:
    scores: list[float] = []

    for candidate in candidates:
        score = _candidate_score(candidate)
        if score is not None and math.isfinite(score):
            scores.append(float(score))

    return tuple(scores)


def active_candidates_have_link_hints(state: AdaptiveRetrievalState) -> bool:
    return any(_candidate_has_link_hint(candidate) for candidate in state.active_candidates()[:10])


def _candidate_score(candidate: object) -> float | None:
    if isinstance(candidate, RerankedCandidate):
        return float(candidate.rerank_score)

    if isinstance(candidate, FusedCandidate):
        return float(candidate.fused_score)

    if isinstance(candidate, RetrievalCandidate):
        return float(candidate.score)

    score = getattr(candidate, "score", None)
    if score is None:
        return None

    return float(score)


def _candidate_has_link_hint(candidate: object) -> bool:
    base = getattr(candidate, "candidate", candidate)
    payload = getattr(base, "payload", {}) or {}

    if not isinstance(payload, dict):
        return False

    stack: list[object] = [payload]

    while stack:
        value = stack.pop()

        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in {"href", "url", "link", "links"}:
                    if isinstance(child, str) and child.strip():
                        return True
                    if isinstance(child, (list, tuple)) and any(str(item).strip() for item in child):
                        return True
                stack.append(child)

        elif isinstance(value, (list, tuple)):
            stack.extend(value)

    return False


def _estimate(
    action: RetrievalAction,
    gain: float,
    reason: str,
    **metadata: int | float | str | bool | None,
) -> ActionGainEstimate:
    estimate = ActionGainEstimate(
        action=action,
        estimated_gain=_clamp_probability(gain),
        reason=reason,
        metadata={key: value for key, value in metadata.items() if value is not None},
    )
    estimate.validate()
    return estimate


def _validate_probability(name: str, value: float) -> None:
    validate_finite_number(name, value)

    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")


def _clamp_probability(value: float) -> float:
    validate_finite_number("probability", value)
    return max(0.0, min(1.0, float(value)))
