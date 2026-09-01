from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from rag.retrieval.adaptive.actions import (
    RetrievalAction,
    RetrievalActionDecision,
    RetrievalActionEvaluation,
    action_preconditions,
    non_terminal_actions,
)
from rag.retrieval.adaptive.cost import (
    ActionCostModel,
    StaticActionCostConfig,
    StaticActionCostModel,
)
from rag.retrieval.adaptive.gain import (
    ActionGainModel,
    StaticActionGainConfig,
    StaticActionGainModel,
    candidate_score_entropy,
    candidate_score_gap,
)
from rag.retrieval.adaptive.state import AdaptiveRetrievalState
from rag.retrieval.config.runtime import RetrievalRuntimeConfig
from rag.retrieval.contracts.primitives import (
    Payload,
    validate_bool,
    validate_finite_number,
    validate_mapping_size,
    validate_positive_int,
)


@dataclass(frozen=True)
class AdaptivePolicyConfig:
    max_evaluations: int = 64
    max_hops: int = 6
    min_action_score: float = 0.001
    high_confidence_stop_threshold: float = 0.78
    min_confidence_to_pack: float = 0.55
    allow_repeat_actions: bool = False
    allow_pack_without_verification: bool = True
    prefer_stop_after_context_packed: bool = True
    metadata: Payload = field(default_factory=dict)

    @classmethod
    def from_runtime_config(
        cls,
        runtime_config: RetrievalRuntimeConfig,
    ) -> AdaptivePolicyConfig:
        config = runtime_config.adaptive_policy
        return cls(
            max_evaluations=config.max_evaluations,
            max_hops=config.max_hops,
            min_action_score=config.min_action_score,
            high_confidence_stop_threshold=config.high_confidence_stop_threshold,
            min_confidence_to_pack=config.min_confidence_to_pack,
            allow_repeat_actions=config.allow_repeat_actions,
            allow_pack_without_verification=config.allow_pack_without_verification,
            prefer_stop_after_context_packed=config.prefer_stop_after_context_packed,
            metadata={"source": "retrieval_runtime_config"},
        )

    def validate(self) -> None:
        validate_positive_int("adaptive_policy.max_evaluations", self.max_evaluations)
        validate_positive_int("adaptive_policy.max_hops", self.max_hops)

        validate_finite_number("adaptive_policy.min_action_score", self.min_action_score)
        if self.min_action_score < 0:
            raise ValueError("adaptive_policy.min_action_score cannot be negative.")

        validate_finite_number(
            "adaptive_policy.high_confidence_stop_threshold",
            self.high_confidence_stop_threshold,
        )
        if not 0 <= self.high_confidence_stop_threshold <= 1:
            raise ValueError(
                "adaptive_policy.high_confidence_stop_threshold must be between 0 and 1."
            )

        validate_finite_number(
            "adaptive_policy.min_confidence_to_pack",
            self.min_confidence_to_pack,
        )
        if not 0 <= self.min_confidence_to_pack <= 1:
            raise ValueError(
                "adaptive_policy.min_confidence_to_pack must be between 0 and 1."
            )

        validate_finite_number(
            "adaptive_policy.min_confidence_to_pack",
            self.min_confidence_to_pack,
        )
        if not 0 <= self.min_confidence_to_pack <= 1:
            raise ValueError(
                "adaptive_policy.min_confidence_to_pack must be between 0 and 1."
            )

        validate_finite_number(
            "adaptive_policy.min_confidence_to_pack",
            self.min_confidence_to_pack,
        )
        if not 0 <= self.min_confidence_to_pack <= 1:
            raise ValueError(
                "adaptive_policy.min_confidence_to_pack must be between 0 and 1."
            )

        validate_bool("adaptive_policy.allow_repeat_actions", self.allow_repeat_actions)
        validate_bool(
            "adaptive_policy.allow_pack_without_verification",
            self.allow_pack_without_verification,
        )
        validate_bool(
            "adaptive_policy.prefer_stop_after_context_packed",
            self.prefer_stop_after_context_packed,
        )
        validate_mapping_size("adaptive_policy.metadata", self.metadata, None)


@dataclass(frozen=True)
class GreedyAdaptivePolicy:
    cost_model: ActionCostModel = field(default_factory=StaticActionCostModel)
    gain_model: ActionGainModel = field(default_factory=StaticActionGainModel)
    config: AdaptivePolicyConfig = field(default_factory=AdaptivePolicyConfig)

    @classmethod
    def from_runtime_config(
        cls,
        runtime_config: RetrievalRuntimeConfig,
    ) -> GreedyAdaptivePolicy:
        runtime_config.validate()

        return cls(
            cost_model=StaticActionCostModel(
                StaticActionCostConfig.from_runtime_config(runtime_config),
            ),
            gain_model=StaticActionGainModel(
                StaticActionGainConfig.from_runtime_config(runtime_config),
            ),
            config=AdaptivePolicyConfig.from_runtime_config(runtime_config),
        )

    def choose_next_action(self, state: AdaptiveRetrievalState) -> RetrievalActionDecision:
        state.validate()
        self.config.validate()

        terminal_decision = self._terminal_decision_if_ready(state)
        if terminal_decision is not None:
            terminal_decision.validate(max_evaluations=self.config.max_evaluations)
            return terminal_decision

        forced_ce_decision = self._forced_ce_rerank_decision(state)
        if forced_ce_decision is not None:
            forced_ce_decision.validate(max_evaluations=self.config.max_evaluations)
            return forced_ce_decision

        evaluations = self.evaluate_actions(state)
        available = tuple(
            evaluation for evaluation in evaluations if evaluation.preconditions_met
        )

        if not available:
            decision = self._fallback_decision(
                state,
                reason="no available actions with satisfied preconditions",
                evaluations=evaluations,
            )
            decision.validate(max_evaluations=self.config.max_evaluations)
            return decision

        best = max(
            available,
            key=lambda evaluation: (
                evaluation.score,
                evaluation.estimated_gain,
                -evaluation.estimated_cost_ms,
                evaluation.action.value,
            ),
        )

        if best.score < self.config.min_action_score:
            decision = self._fallback_decision(
                state,
                reason=(
                    "best available action score below threshold: "
                    f"{best.score:.6f} < {self.config.min_action_score:.6f}"
                ),
                evaluations=evaluations,
            )
            decision.validate(max_evaluations=self.config.max_evaluations)
            return decision

        decision = RetrievalActionDecision(
            action=best.action,
            reason=(
                f"selected highest gain/cost action: {best.action.value}; "
                f"score={best.score:.6f}"
            ),
            score=best.score,
            estimated_gain=best.estimated_gain,
            estimated_cost_ms=best.estimated_cost_ms,
            evaluations=evaluations,
            metadata={
                "policy": "greedy_gain_cost",
                "completed_actions": sorted(
                    action.value for action in state.completed_actions
                ),
                "hop": state.hop,
                "confidence": state.confidence.value,
            },
        )
        decision.validate(max_evaluations=self.config.max_evaluations)
        return decision

    def _forced_ce_rerank_decision(
        self,
        state: AdaptiveRetrievalState,
    ) -> RetrievalActionDecision | None:
        reason = self._ce_rerank_required_reason(state)
        if reason is None:
            return None

        evaluations = self.evaluate_actions(state)
        ce_evaluation = next(
            (
                evaluation
                for evaluation in evaluations
                if evaluation.action == RetrievalAction.CE_RERANK
            ),
            None,
        )

        if ce_evaluation is None or not ce_evaluation.preconditions_met:
            return None

        decision = RetrievalActionDecision(
            action=RetrievalAction.CE_RERANK,
            reason=f"forced CE rerank before context packing: {reason}",
            score=max(ce_evaluation.score, self.config.min_action_score),
            estimated_gain=ce_evaluation.estimated_gain,
            estimated_cost_ms=ce_evaluation.estimated_cost_ms,
            evaluations=evaluations,
            metadata={
                "policy": "greedy_gain_cost_with_ce_gate",
                "forced": True,
                "trigger": reason,
                "completed_actions": sorted(
                    action.value for action in state.completed_actions
                ),
                "hop": state.hop,
                "confidence": state.confidence.value,
                "score_gap": candidate_score_gap(state),
                "score_entropy": candidate_score_entropy(state),
                "reranked_count": len(state.reranked_candidates),
            },
        )
        decision.validate(max_evaluations=self.config.max_evaluations)
        return decision

    def _ce_rerank_required_reason(
        self,
        state: AdaptiveRetrievalState,
    ) -> str | None:
        # CE is an expensive second-stage judge. Do not run it before the cheap
        # score reranker has produced a candidate ordering.
        if RetrievalAction.CE_RERANK in state.completed_actions:
            return None

        if RetrievalAction.SCORE_RERANK not in state.completed_actions:
            return None

        if state.context_blocks:
            return None

        if not state.reranked_candidates:
            return None

        if len(state.reranked_candidates) < 4:
            return None

        confidence = state.confidence.value
        exact_coverage = state.confidence.exact_coverage or 0.0
        source_coverage = state.confidence.source_coverage or 0.0

        # If evidence is already very strong, do not pay the CE cost.
        if (
            confidence >= self.config.high_confidence_stop_threshold
            and (exact_coverage >= 0.85 or source_coverage >= 0.90)
        ):
            return None

        gap = candidate_score_gap(state)
        entropy = candidate_score_entropy(state)
        signals = state.query_signals

        if gap is not None and gap <= 0.15:
            return f"flat top candidate scores; score_gap={gap:.4f}"

        if entropy is not None and entropy >= 0.82:
            return f"high candidate score entropy; entropy={entropy:.4f}"

        if signals.is_chat_dependent and confidence < 0.82:
            return "chat-dependent query with less-than-strong confidence"

        if signals.is_structural_or_multi_hop and confidence < 0.82:
            return "structural or multi-hop query with less-than-strong confidence"

        if confidence < self.config.min_confidence_to_pack:
            return (
                "confidence below context packing threshold; "
                f"{confidence:.4f} < {self.config.min_confidence_to_pack:.4f}"
            )

        return None

    def evaluate_actions(
        self,
        state: AdaptiveRetrievalState,
        *,
        actions: Sequence[RetrievalAction] | None = None,
    ) -> tuple[RetrievalActionEvaluation, ...]:
        state.validate()

        selected_actions = tuple(actions or non_terminal_actions())
        evaluations: list[RetrievalActionEvaluation] = []

        for action in selected_actions:
            if action in {RetrievalAction.STOP, RetrievalAction.FALLBACK_NO_ANSWER}:
                continue

            preconditions = action_preconditions(action)
            preconditions.validate()

            blocked_reason = preconditions.blocked_reason(state.completed_actions)
            preconditions_met = blocked_reason is None

            if not self.config.allow_repeat_actions and action in state.completed_actions:
                preconditions_met = False
                blocked_reason = "action_already_completed"

            if (
                action == RetrievalAction.PACK_CONTEXT
                and not self.config.allow_pack_without_verification
                and RetrievalAction.VERIFY_EVIDENCE not in state.completed_actions
            ):
                preconditions_met = False
                blocked_reason = "verification_required_before_pack_context"

            dynamic_block_reason = self._dynamic_block_reason(
                action=action,
                state=state,
            )
            if dynamic_block_reason is not None:
                preconditions_met = False
                blocked_reason = dynamic_block_reason

            gain = self.gain_model.estimate(action, state)
            cost = self.cost_model.estimate(action, state)

            evaluation = RetrievalActionEvaluation.from_gain_cost(
                action=action,
                estimated_gain=gain.estimated_gain if preconditions_met else 0.0,
                estimated_cost_ms=cost.estimated_ms,
                preconditions_met=preconditions_met,
                blocked_reason=blocked_reason,
                metadata={
                    "gain_reason": gain.reason,
                    "cost_reason": cost.reason,
                    "gain_metadata": gain.metadata,
                    "cost_metadata": cost.metadata,
                },
            )
            evaluation.validate()
            evaluations.append(evaluation)

            if len(evaluations) >= self.config.max_evaluations:
                break

        return tuple(evaluations)

    def _dynamic_block_reason(
        self,
        *,
        action: RetrievalAction,
        state: AdaptiveRetrievalState,
    ) -> str | None:
        if action == RetrievalAction.MEMORY_LOOKUP and not state.query_signals.likely_needs_memory:
            return "memory_not_needed_for_query"

        if action == RetrievalAction.FUSE:
            channel_count = sum(
                1
                for count in (
                    len(state.memory_candidates),
                    len(state.sparse_candidates),
                    len(state.dense_candidates),
                )
                if count > 0
            )
            if channel_count < 2:
                return "fusion_requires_candidates_from_at_least_two_channels"

        if action in {
            RetrievalAction.SCORE_RERANK,
            RetrievalAction.CE_RERANK,
            RetrievalAction.VERIFY_EVIDENCE,
            RetrievalAction.PACK_CONTEXT,
        } and not state.has_candidates:
            return "action_requires_active_candidates"

        if action == RetrievalAction.PACK_CONTEXT and not state.reranked_candidates:
            return "pack_context_requires_reranked_candidates"

        if action == RetrievalAction.PACK_CONTEXT:
            strong_evidence = (
                (state.confidence.exact_coverage or 0.0) >= 0.80
                or (
                    (state.confidence.source_coverage or 0.0) >= 0.80
                    and (
                        bool(state.reranked_candidates)
                        or bool(state.fused_candidates)
                        or bool(state.sparse_candidates)
                        or bool(state.dense_candidates)
                    )
                )
            )

            if (
                state.confidence.value < self.config.min_confidence_to_pack
                and not strong_evidence
            ):
                return (
                    "pack_context_requires_min_confidence="
                    f"{self.config.min_confidence_to_pack:.2f}"
                )

        if action == RetrievalAction.LINKED_EXPAND and not state.has_candidates:
            return "linked_expand_requires_seed_candidates"

        if action == RetrievalAction.ESCALATE_K and not state.has_candidates:
            return "escalate_k_requires_existing_candidates"

        if (
            action == RetrievalAction.ESCALATE_K
            and state.reranked_candidates
            and (
                (state.confidence.exact_coverage or 0.0) >= 0.80
                or (state.confidence.source_coverage or 0.0) >= 0.80
            )
        ):
            return "escalate_k_blocked_after_good_reranked_evidence"

        return None

    def _terminal_decision_if_ready(
        self,
        state: AdaptiveRetrievalState,
    ) -> RetrievalActionDecision | None:
        if self.config.prefer_stop_after_context_packed and state.context_blocks:
            return RetrievalActionDecision(
                action=RetrievalAction.STOP,
                reason="context already packed",
                score=1.0,
                estimated_gain=1.0,
                estimated_cost_ms=0.0,
                metadata={
                    "policy": "greedy_gain_cost",
                    "context_count": len(state.context_blocks),
                },
            )

        if not state.can_continue:
            action = (
                RetrievalAction.STOP
                if state.has_candidates
                else RetrievalAction.FALLBACK_NO_ANSWER
            )
            return RetrievalActionDecision(
                action=action,
                reason="state cannot continue because max hops reached or errors exist",
                score=1.0,
                estimated_gain=1.0,
                estimated_cost_ms=0.0,
                metadata={
                    "policy": "greedy_gain_cost",
                    "hop": state.hop,
                    "max_hops": state.max_hops,
                    "errors": list(state.errors),
                },
            )

        if (
            state.confidence.value >= self.config.high_confidence_stop_threshold
            and state.has_candidates
            and RetrievalAction.PACK_CONTEXT in state.completed_actions
        ):
            return RetrievalActionDecision(
                action=RetrievalAction.STOP,
                reason="high confidence and context has been packed",
                score=1.0,
                estimated_gain=1.0,
                estimated_cost_ms=0.0,
                metadata={
                    "policy": "greedy_gain_cost",
                    "confidence": state.confidence.value,
                },
            )

        if (
            state.query_signals.corpus_affinity == 0.0
            and state.hop == 0
            and not state.query_signals.is_exactish
            and not state.query_signals.likely_needs_memory
        ):
            return RetrievalActionDecision(
                action=RetrievalAction.FALLBACK_NO_ANSWER,
                reason="zero corpus affinity before retrieval and no exact or memory signal",
                score=1.0,
                estimated_gain=1.0,
                estimated_cost_ms=0.0,
                metadata={
                    "policy": "greedy_gain_cost",
                    "corpus_affinity": state.query_signals.corpus_affinity,
                },
            )

        return None

    def _fallback_decision(
        self,
        state: AdaptiveRetrievalState,
        *,
        reason: str,
        evaluations: tuple[RetrievalActionEvaluation, ...],
    ) -> RetrievalActionDecision:
        action = (
            RetrievalAction.STOP
            if state.has_candidates
            else RetrievalAction.FALLBACK_NO_ANSWER
        )

        return RetrievalActionDecision(
            action=action,
            reason=reason,
            score=0.0,
            estimated_gain=0.0,
            estimated_cost_ms=0.0,
            evaluations=evaluations,
            metadata={
                "policy": "greedy_gain_cost",
                "has_candidates": state.has_candidates,
                "hop": state.hop,
            },
        )


def explain_decision(decision: RetrievalActionDecision) -> str:
    decision.validate()

    return (
        f"action={decision.action.value}; "
        f"score={decision.score:.6f}; "
        f"gain={decision.estimated_gain:.4f}; "
        f"cost_ms={decision.estimated_cost_ms:.2f}; "
        f"reason={decision.reason}"
    )


def validate_policy_sample(decision: RetrievalActionDecision) -> None:
    decision.validate()

    if not decision.reason.strip():
        raise ValueError("policy decision must include a reason.")
