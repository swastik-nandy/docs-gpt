from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rag.retrieval.adaptive.actions import RetrievalAction
from rag.retrieval.adaptive.state import AdaptiveRetrievalState
from rag.retrieval.config.runtime import RetrievalRuntimeConfig
from rag.retrieval.contracts.primitives import (
    Payload,
    validate_finite_number,
    validate_mapping_size,
    validate_non_empty_str,
    validate_non_negative_int,
    validate_positive_int,
)


@dataclass(frozen=True)
class ActionCostEstimate:
    action: RetrievalAction
    estimated_ms: float
    reason: str
    metadata: Payload = field(default_factory=dict)

    def validate(self, *, max_metadata_keys: int | None = None) -> None:
        if not isinstance(self.action, RetrievalAction):
            raise ValueError("action_cost_estimate.action must be RetrievalAction.")

        validate_finite_number("action_cost_estimate.estimated_ms", self.estimated_ms)
        if self.estimated_ms < 0:
            raise ValueError("action_cost_estimate.estimated_ms cannot be negative.")

        validate_non_empty_str("action_cost_estimate.reason", self.reason)
        validate_mapping_size(
            "action_cost_estimate.metadata",
            self.metadata,
            max_metadata_keys,
        )


class ActionCostModel(Protocol):
    def estimate(self, action: RetrievalAction, state: AdaptiveRetrievalState) -> ActionCostEstimate:
        raise NotImplementedError


@dataclass(frozen=True)
class StaticActionCostConfig:
    memory_lookup_ms: float = 12.0
    sparse_search_ms: float = 45.0
    dense_search_ms: float = 95.0
    fuse_ms: float = 4.0

    score_rerank_base_ms: float = 3.0
    score_rerank_per_candidate_ms: float = 0.08

    ce_rerank_base_ms: float = 90.0
    ce_rerank_per_candidate_ms: float = 8.0

    linked_expand_base_ms: float = 35.0
    linked_expand_per_seed_ms: float = 20.0
    linked_expand_per_link_hint_ms: float = 8.0

    escalate_k_ms: float = 30.0
    escalate_dense_ms: float = 95.0

    verify_evidence_base_ms: float = 4.0
    verify_evidence_per_candidate_ms: float = 0.2

    pack_context_base_ms: float = 5.0
    pack_context_per_candidate_ms: float = 0.1

    stop_ms: float = 0.0
    fallback_no_answer_ms: float = 0.0

    min_action_cost_ms: float = 1.0

    @classmethod
    def from_runtime_config(
        cls,
        runtime_config: RetrievalRuntimeConfig,
    ) -> StaticActionCostConfig:
        config = runtime_config.adaptive_cost
        return cls(
            memory_lookup_ms=config.memory_lookup_ms,
            sparse_search_ms=config.sparse_search_ms,
            dense_search_ms=config.dense_search_ms,
            fuse_ms=config.fuse_ms,
            score_rerank_base_ms=config.score_rerank_base_ms,
            score_rerank_per_candidate_ms=config.score_rerank_per_candidate_ms,
            ce_rerank_base_ms=config.ce_rerank_base_ms,
            ce_rerank_per_candidate_ms=config.ce_rerank_per_candidate_ms,
            linked_expand_base_ms=config.linked_expand_base_ms,
            linked_expand_per_seed_ms=config.linked_expand_per_seed_ms,
            linked_expand_per_link_hint_ms=config.linked_expand_per_link_hint_ms,
            escalate_k_ms=config.escalate_k_ms,
            escalate_dense_ms=config.escalate_dense_ms,
            verify_evidence_base_ms=config.verify_evidence_base_ms,
            verify_evidence_per_candidate_ms=config.verify_evidence_per_candidate_ms,
            pack_context_base_ms=config.pack_context_base_ms,
            pack_context_per_candidate_ms=config.pack_context_per_candidate_ms,
            stop_ms=config.stop_ms,
            fallback_no_answer_ms=config.fallback_no_answer_ms,
            min_action_cost_ms=config.min_action_cost_ms,
        )

    def validate(self) -> None:
        for name, value in (
            ("memory_lookup_ms", self.memory_lookup_ms),
            ("sparse_search_ms", self.sparse_search_ms),
            ("dense_search_ms", self.dense_search_ms),
            ("fuse_ms", self.fuse_ms),
            ("score_rerank_base_ms", self.score_rerank_base_ms),
            ("score_rerank_per_candidate_ms", self.score_rerank_per_candidate_ms),
            ("ce_rerank_base_ms", self.ce_rerank_base_ms),
            ("ce_rerank_per_candidate_ms", self.ce_rerank_per_candidate_ms),
            ("linked_expand_base_ms", self.linked_expand_base_ms),
            ("linked_expand_per_seed_ms", self.linked_expand_per_seed_ms),
            ("linked_expand_per_link_hint_ms", self.linked_expand_per_link_hint_ms),
            ("escalate_k_ms", self.escalate_k_ms),
            ("escalate_dense_ms", self.escalate_dense_ms),
            ("verify_evidence_base_ms", self.verify_evidence_base_ms),
            ("verify_evidence_per_candidate_ms", self.verify_evidence_per_candidate_ms),
            ("pack_context_base_ms", self.pack_context_base_ms),
            ("pack_context_per_candidate_ms", self.pack_context_per_candidate_ms),
            ("stop_ms", self.stop_ms),
            ("fallback_no_answer_ms", self.fallback_no_answer_ms),
            ("min_action_cost_ms", self.min_action_cost_ms),
        ):
            validate_finite_number(f"static_action_cost_config.{name}", value)
            if value < 0:
                raise ValueError(f"static_action_cost_config.{name} cannot be negative.")


@dataclass(frozen=True)
class StaticActionCostModel:
    config: StaticActionCostConfig = StaticActionCostConfig()

    def estimate(
        self,
        action: RetrievalAction,
        state: AdaptiveRetrievalState,
    ) -> ActionCostEstimate:
        state.validate()
        self.config.validate()

        active_count = len(state.active_candidates())
        rerank_pool_size = _rerank_pool_size(state)
        link_seed_count = _estimated_link_seed_count(state)

        if action == RetrievalAction.MEMORY_LOOKUP:
            return _estimate(action, self.config.memory_lookup_ms, "memory lookup static cost")

        if action == RetrievalAction.SPARSE_SEARCH:
            return _estimate(action, self.config.sparse_search_ms, "sparse search static cost")

        if action == RetrievalAction.DENSE_SEARCH:
            return _estimate(action, self.config.dense_search_ms, "dense search static cost")

        if action == RetrievalAction.FUSE:
            return _estimate(
                action,
                self.config.fuse_ms,
                "RRF fusion static cost",
                dense_count=len(state.dense_candidates),
                sparse_count=len(state.sparse_candidates),
                memory_count=len(state.memory_candidates),
            )

        if action == RetrievalAction.SCORE_RERANK:
            return _estimate(
                action,
                self.config.score_rerank_base_ms
                + self.config.score_rerank_per_candidate_ms * rerank_pool_size,
                "score rerank cost scales with candidate pool",
                rerank_pool_size=rerank_pool_size,
            )

        if action == RetrievalAction.CE_RERANK:
            return _estimate(
                action,
                self.config.ce_rerank_base_ms
                + self.config.ce_rerank_per_candidate_ms * rerank_pool_size,
                "cross-encoder cost scales with candidate pool",
                rerank_pool_size=rerank_pool_size,
            )

        if action == RetrievalAction.LINKED_EXPAND:
            link_hint_count = _estimated_link_hint_count(state)
            return _estimate(
                action,
                self.config.linked_expand_base_ms
                + self.config.linked_expand_per_seed_ms * link_seed_count
                + self.config.linked_expand_per_link_hint_ms * link_hint_count,
                "linked expansion cost scales with seeds and href hints",
                link_seed_count=link_seed_count,
                link_hint_count=link_hint_count,
            )

        if action == RetrievalAction.ESCALATE_K:
            return _estimate(action, self.config.escalate_k_ms, "candidate widening static cost")

        if action == RetrievalAction.ESCALATE_DENSE:
            return _estimate(action, self.config.escalate_dense_ms, "dense escalation static cost")

        if action == RetrievalAction.VERIFY_EVIDENCE:
            return _estimate(
                action,
                self.config.verify_evidence_base_ms
                + self.config.verify_evidence_per_candidate_ms * active_count,
                "evidence verification cost scales with active candidate count",
                active_count=active_count,
            )

        if action == RetrievalAction.PACK_CONTEXT:
            return _estimate(
                action,
                self.config.pack_context_base_ms
                + self.config.pack_context_per_candidate_ms * active_count,
                "context packing cost scales with active candidate count",
                active_count=active_count,
            )

        if action == RetrievalAction.STOP:
            return _estimate(action, self.config.stop_ms, "terminal stop has no cost")

        if action == RetrievalAction.FALLBACK_NO_ANSWER:
            return _estimate(
                action,
                self.config.fallback_no_answer_ms,
                "terminal fallback has no cost",
            )

        raise ValueError(f"Unsupported retrieval action for cost estimation: {action}")


def _estimate(
    action: RetrievalAction,
    estimated_ms: float,
    reason: str,
    **metadata: int | float | str | bool | None,
) -> ActionCostEstimate:
    estimate = ActionCostEstimate(
        action=action,
        estimated_ms=max(0.0, float(estimated_ms)),
        reason=reason,
        metadata={key: value for key, value in metadata.items() if value is not None},
    )
    estimate.validate()
    return estimate


def _rerank_pool_size(state: AdaptiveRetrievalState) -> int:
    if state.expanded_candidates and state.fused_candidates:
        return len(state.expanded_candidates) + len(state.fused_candidates)

    if state.fused_candidates:
        return len(state.fused_candidates)

    return (
        len(state.memory_candidates)
        + len(state.sparse_candidates)
        + len(state.dense_candidates)
        + len(state.expanded_candidates)
    )


def _estimated_link_seed_count(state: AdaptiveRetrievalState, *, max_seeds: int = 10) -> int:
    validate_positive_int("max_seeds", max_seeds)

    candidates = state.active_candidates()
    if not candidates:
        return 0

    candidates_with_links = sum(
        1
        for candidate in candidates[:max_seeds]
        if _candidate_has_link_hint(candidate)
    )

    return candidates_with_links


def _estimated_link_hint_count(state: AdaptiveRetrievalState, *, max_seeds: int = 10) -> int:
    validate_positive_int("max_seeds", max_seeds)

    total = 0
    for candidate in state.active_candidates()[:max_seeds]:
        total += _candidate_link_hint_count(candidate)

    validate_non_negative_int("estimated_link_hint_count", total)
    return total


def _candidate_has_link_hint(candidate: object) -> bool:
    return _candidate_link_hint_count(candidate) > 0


def _candidate_link_hint_count(candidate: object) -> int:
    base = getattr(candidate, "candidate", candidate)
    payload = getattr(base, "payload", {}) or {}

    if not isinstance(payload, dict):
        return 0

    total = 0
    stack: list[object] = [payload]

    while stack:
        value = stack.pop()

        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in {"href", "url", "link", "links"}:
                    if isinstance(child, str) and child.strip():
                        total += 1
                    elif isinstance(child, list | tuple):
                        total += sum(1 for item in child if str(item).strip())
                stack.append(child)

        elif isinstance(value, list | tuple):
            stack.extend(value)

    return total
