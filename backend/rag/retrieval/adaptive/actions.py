from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any, Mapping, Sequence

from rag.retrieval.contracts.primitives import (
    Payload,
    validate_bool,
    validate_enum_instance,
    validate_finite_number,
    validate_mapping_size,
    validate_non_empty_str,
    validate_non_negative_int,
    validate_optional_clean_str,
    validate_optional_sequence_limit,
)


class RetrievalAction(str, Enum):
    MEMORY_LOOKUP = "memory_lookup"

    SPARSE_SEARCH = "sparse_search"
    DENSE_SEARCH = "dense_search"
    FUSE = "fuse"

    SCORE_RERANK = "score_rerank"
    CE_RERANK = "ce_rerank"

    LINKED_EXPAND = "linked_expand"

    ESCALATE_K = "escalate_k"
    ESCALATE_DENSE = "escalate_dense"

    VERIFY_EVIDENCE = "verify_evidence"
    PACK_CONTEXT = "pack_context"

    STOP = "stop"
    FALLBACK_NO_ANSWER = "fallback_no_answer"


class RetrievalActionStatus(str, Enum):
    PLANNED = "planned"
    STARTED = "started"
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class RetrievalActionPreconditions:
    requires_all: tuple[RetrievalAction, ...] = field(default_factory=tuple)
    requires_any: tuple[RetrievalAction, ...] = field(default_factory=tuple)
    forbidden_after: tuple[RetrievalAction, ...] = field(default_factory=tuple)

    def is_satisfied_by(self, completed_actions: set[RetrievalAction]) -> bool:
        if any(action in completed_actions for action in self.forbidden_after):
            return False

        if any(action not in completed_actions for action in self.requires_all):
            return False

        if self.requires_any and not any(
            action in completed_actions for action in self.requires_any
        ):
            return False

        return True

    def blocked_reason(self, completed_actions: set[RetrievalAction]) -> str | None:
        forbidden = [
            action.value
            for action in self.forbidden_after
            if action in completed_actions
        ]
        if forbidden:
            return f"forbidden_after={forbidden}"

        missing_all = [
            action.value
            for action in self.requires_all
            if action not in completed_actions
        ]
        if missing_all:
            return f"missing_required_actions={missing_all}"

        if self.requires_any and not any(
            action in completed_actions for action in self.requires_any
        ):
            return f"requires_one_of={[action.value for action in self.requires_any]}"

        return None

    def validate(self) -> None:
        for field_name, values in (
            ("requires_all", self.requires_all),
            ("requires_any", self.requires_any),
            ("forbidden_after", self.forbidden_after),
        ):
            for index, action in enumerate(values):
                validate_enum_instance(
                    f"retrieval_action_preconditions.{field_name}[{index}]",
                    action,
                    RetrievalAction,
                )


ACTION_PRECONDITIONS: Mapping[RetrievalAction, RetrievalActionPreconditions] = {
    RetrievalAction.MEMORY_LOOKUP: RetrievalActionPreconditions(),

    RetrievalAction.SPARSE_SEARCH: RetrievalActionPreconditions(),
    RetrievalAction.DENSE_SEARCH: RetrievalActionPreconditions(),

    RetrievalAction.FUSE: RetrievalActionPreconditions(
        requires_any=(
            RetrievalAction.SPARSE_SEARCH,
            RetrievalAction.DENSE_SEARCH,
            RetrievalAction.MEMORY_LOOKUP,
        ),
    ),

    RetrievalAction.SCORE_RERANK: RetrievalActionPreconditions(
        requires_any=(
            RetrievalAction.FUSE,
            RetrievalAction.SPARSE_SEARCH,
            RetrievalAction.DENSE_SEARCH,
            RetrievalAction.LINKED_EXPAND,
            RetrievalAction.MEMORY_LOOKUP,
        ),
    ),
    RetrievalAction.CE_RERANK: RetrievalActionPreconditions(
        requires_any=(
            RetrievalAction.FUSE,
            RetrievalAction.SPARSE_SEARCH,
            RetrievalAction.DENSE_SEARCH,
            RetrievalAction.LINKED_EXPAND,
            RetrievalAction.MEMORY_LOOKUP,
        ),
    ),

    RetrievalAction.LINKED_EXPAND: RetrievalActionPreconditions(
        requires_any=(
            RetrievalAction.FUSE,
            RetrievalAction.SCORE_RERANK,
            RetrievalAction.CE_RERANK,
            RetrievalAction.SPARSE_SEARCH,
            RetrievalAction.DENSE_SEARCH,
        ),
    ),

    RetrievalAction.ESCALATE_K: RetrievalActionPreconditions(
        requires_any=(
            RetrievalAction.SPARSE_SEARCH,
            RetrievalAction.DENSE_SEARCH,
            RetrievalAction.FUSE,
            RetrievalAction.SCORE_RERANK,
            RetrievalAction.CE_RERANK,
        ),
    ),
    RetrievalAction.ESCALATE_DENSE: RetrievalActionPreconditions(
        requires_all=(RetrievalAction.SPARSE_SEARCH,),
        forbidden_after=(RetrievalAction.DENSE_SEARCH,),
    ),

    RetrievalAction.VERIFY_EVIDENCE: RetrievalActionPreconditions(
        requires_any=(
            RetrievalAction.MEMORY_LOOKUP,
            RetrievalAction.SPARSE_SEARCH,
            RetrievalAction.DENSE_SEARCH,
            RetrievalAction.FUSE,
            RetrievalAction.SCORE_RERANK,
            RetrievalAction.CE_RERANK,
            RetrievalAction.LINKED_EXPAND,
        ),
    ),
    RetrievalAction.PACK_CONTEXT: RetrievalActionPreconditions(
        requires_any=(
            RetrievalAction.VERIFY_EVIDENCE,
            RetrievalAction.SCORE_RERANK,
            RetrievalAction.CE_RERANK,
            RetrievalAction.FUSE,
            RetrievalAction.SPARSE_SEARCH,
            RetrievalAction.DENSE_SEARCH,
            RetrievalAction.MEMORY_LOOKUP,
        ),
    ),

    RetrievalAction.STOP: RetrievalActionPreconditions(),
    RetrievalAction.FALLBACK_NO_ANSWER: RetrievalActionPreconditions(),
}


@dataclass(frozen=True)
class RetrievalActionEvaluation:
    action: RetrievalAction
    estimated_gain: float
    estimated_cost_ms: float
    score: float
    preconditions_met: bool = True
    blocked_reason: str | None = None
    metadata: Payload = field(default_factory=dict)

    @classmethod
    def from_gain_cost(
        cls,
        *,
        action: RetrievalAction,
        estimated_gain: float,
        estimated_cost_ms: float,
        preconditions_met: bool = True,
        blocked_reason: str | None = None,
        min_cost_ms: float = 1.0,
        metadata: Payload | None = None,
    ) -> RetrievalActionEvaluation:
        safe_cost = max(float(estimated_cost_ms), float(min_cost_ms))
        return cls(
            action=action,
            estimated_gain=float(estimated_gain),
            estimated_cost_ms=float(estimated_cost_ms),
            score=float(estimated_gain) / safe_cost,
            preconditions_met=preconditions_met,
            blocked_reason=blocked_reason,
            metadata=metadata or {},
        )

    def validate(self, *, max_metadata_keys: int | None = None) -> None:
        validate_enum_instance(
            "retrieval_action_evaluation.action",
            self.action,
            RetrievalAction,
        )
        validate_finite_number(
            "retrieval_action_evaluation.estimated_gain",
            self.estimated_gain,
        )
        validate_finite_number(
            "retrieval_action_evaluation.estimated_cost_ms",
            self.estimated_cost_ms,
        )
        validate_finite_number("retrieval_action_evaluation.score", self.score)
        validate_bool(
            "retrieval_action_evaluation.preconditions_met",
            self.preconditions_met,
        )
        validate_optional_clean_str(
            "retrieval_action_evaluation.blocked_reason",
            self.blocked_reason,
        )
        validate_mapping_size(
            "retrieval_action_evaluation.metadata",
            self.metadata,
            max_metadata_keys,
        )

        if self.estimated_gain < 0:
            raise ValueError("retrieval_action_evaluation.estimated_gain cannot be negative.")

        if self.estimated_cost_ms < 0:
            raise ValueError("retrieval_action_evaluation.estimated_cost_ms cannot be negative.")


@dataclass(frozen=True)
class RetrievalActionDecision:
    action: RetrievalAction
    reason: str
    score: float = 0.0
    estimated_gain: float = 0.0
    estimated_cost_ms: float = 0.0
    evaluations: tuple[RetrievalActionEvaluation, ...] = field(default_factory=tuple)
    metadata: Payload = field(default_factory=dict)

    @property
    def should_stop(self) -> bool:
        return self.action in {
            RetrievalAction.STOP,
            RetrievalAction.FALLBACK_NO_ANSWER,
        }

    def validate(
        self,
        *,
        max_evaluations: int | None = None,
        max_metadata_keys: int | None = None,
    ) -> None:
        validate_enum_instance(
            "retrieval_action_decision.action",
            self.action,
            RetrievalAction,
        )
        validate_non_empty_str("retrieval_action_decision.reason", self.reason)
        validate_finite_number("retrieval_action_decision.score", self.score)
        validate_finite_number(
            "retrieval_action_decision.estimated_gain",
            self.estimated_gain,
        )
        validate_finite_number(
            "retrieval_action_decision.estimated_cost_ms",
            self.estimated_cost_ms,
        )

        if self.estimated_gain < 0:
            raise ValueError("retrieval_action_decision.estimated_gain cannot be negative.")

        if self.estimated_cost_ms < 0:
            raise ValueError("retrieval_action_decision.estimated_cost_ms cannot be negative.")

        validate_optional_sequence_limit(
            "retrieval_action_decision.evaluations",
            self.evaluations,
            max_evaluations,
        )
        for index, evaluation in enumerate(self.evaluations):
            if not isinstance(evaluation, RetrievalActionEvaluation):
                raise ValueError(
                    f"retrieval_action_decision.evaluations[{index}] "
                    "must be RetrievalActionEvaluation."
                )
            evaluation.validate(max_metadata_keys=max_metadata_keys)

        validate_mapping_size(
            "retrieval_action_decision.metadata",
            self.metadata,
            max_metadata_keys,
        )


@dataclass(frozen=True)
class RetrievalActionRecord:
    action: RetrievalAction
    status: RetrievalActionStatus
    hop: int
    started_at: float | None = None
    ended_at: float | None = None
    latency_ms: float | None = None
    input_count: int | None = None
    output_count: int | None = None
    materially_changed: bool | None = None
    reason: str | None = None
    error: str | None = None
    metadata: Payload = field(default_factory=dict)

    @classmethod
    def started(
        cls,
        *,
        action: RetrievalAction,
        hop: int,
        reason: str | None = None,
        metadata: Payload | None = None,
    ) -> RetrievalActionRecord:
        return cls(
            action=action,
            status=RetrievalActionStatus.STARTED,
            hop=hop,
            started_at=time(),
            reason=reason,
            metadata=metadata or {},
        )

    @classmethod
    def skipped(
        cls,
        *,
        action: RetrievalAction,
        hop: int,
        reason: str,
        metadata: Payload | None = None,
    ) -> RetrievalActionRecord:
        return cls(
            action=action,
            status=RetrievalActionStatus.SKIPPED,
            hop=hop,
            reason=reason,
            metadata=metadata or {},
        )

    def succeeded(
        self,
        *,
        input_count: int | None = None,
        output_count: int | None = None,
        materially_changed: bool | None = None,
        metadata: Payload | None = None,
    ) -> RetrievalActionRecord:
        ended_at = time()
        started_at = self.started_at or ended_at
        merged_metadata = dict(self.metadata)
        merged_metadata.update(metadata or {})

        return type(self)(
            action=self.action,
            status=RetrievalActionStatus.SUCCEEDED,
            hop=self.hop,
            started_at=started_at,
            ended_at=ended_at,
            latency_ms=(ended_at - started_at) * 1000,
            input_count=input_count,
            output_count=output_count,
            materially_changed=materially_changed,
            reason=self.reason,
            metadata=merged_metadata,
        )

    def failed(
        self,
        *,
        error: str,
        metadata: Payload | None = None,
    ) -> RetrievalActionRecord:
        ended_at = time()
        started_at = self.started_at or ended_at
        merged_metadata = dict(self.metadata)
        merged_metadata.update(metadata or {})

        return type(self)(
            action=self.action,
            status=RetrievalActionStatus.FAILED,
            hop=self.hop,
            started_at=started_at,
            ended_at=ended_at,
            latency_ms=(ended_at - started_at) * 1000,
            reason=self.reason,
            error=error,
            metadata=merged_metadata,
        )

    def validate(self, *, max_metadata_keys: int | None = None) -> None:
        validate_enum_instance(
            "retrieval_action_record.action",
            self.action,
            RetrievalAction,
        )
        validate_enum_instance(
            "retrieval_action_record.status",
            self.status,
            RetrievalActionStatus,
        )
        validate_non_negative_int("retrieval_action_record.hop", self.hop)
        validate_optional_clean_str("retrieval_action_record.reason", self.reason)
        validate_optional_clean_str("retrieval_action_record.error", self.error)

        if self.input_count is not None:
            validate_non_negative_int(
                "retrieval_action_record.input_count",
                self.input_count,
            )

        if self.output_count is not None:
            validate_non_negative_int(
                "retrieval_action_record.output_count",
                self.output_count,
            )

        if self.materially_changed is not None:
            validate_bool(
                "retrieval_action_record.materially_changed",
                self.materially_changed,
            )

        for name, value in (
            ("started_at", self.started_at),
            ("ended_at", self.ended_at),
            ("latency_ms", self.latency_ms),
        ):
            if value is not None:
                validate_finite_number(f"retrieval_action_record.{name}", value)
                if value < 0:
                    raise ValueError(f"retrieval_action_record.{name} cannot be negative.")

        validate_mapping_size(
            "retrieval_action_record.metadata",
            self.metadata,
            max_metadata_keys,
        )


def action_preconditions(action: RetrievalAction) -> RetrievalActionPreconditions:
    return ACTION_PRECONDITIONS[action]


def is_terminal_action(action: RetrievalAction) -> bool:
    return action in {
        RetrievalAction.STOP,
        RetrievalAction.FALLBACK_NO_ANSWER,
    }


def non_terminal_actions() -> tuple[RetrievalAction, ...]:
    return tuple(action for action in RetrievalAction if not is_terminal_action(action))


def action_requires_embedding(action: RetrievalAction) -> bool:
    return action in {
        RetrievalAction.DENSE_SEARCH,
        RetrievalAction.SPARSE_SEARCH,
    }


def action_produces_candidates(action: RetrievalAction) -> bool:
    return action in {
        RetrievalAction.MEMORY_LOOKUP,
        RetrievalAction.SPARSE_SEARCH,
        RetrievalAction.DENSE_SEARCH,
        RetrievalAction.FUSE,
        RetrievalAction.SCORE_RERANK,
        RetrievalAction.CE_RERANK,
        RetrievalAction.LINKED_EXPAND,
        RetrievalAction.ESCALATE_K,
        RetrievalAction.ESCALATE_DENSE,
    }


def completed_actions(records: Sequence[RetrievalActionRecord]) -> set[RetrievalAction]:
    return {
        record.action
        for record in records
        if record.status == RetrievalActionStatus.SUCCEEDED
    }
