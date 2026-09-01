from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from rag.retrieval.adaptive.actions import (
    RetrievalAction,
    RetrievalActionRecord,
    RetrievalActionStatus,
    completed_actions,
)
from rag.retrieval.contracts.candidates import (
    FusedCandidate,
    RerankedCandidate,
    RetrievalCandidate,
)
from rag.retrieval.contracts.context import ContextBlock
from rag.retrieval.contracts.embeddings import QueryEmbedding
from rag.retrieval.contracts.primitives import (
    Payload,
    validate_bool,
    validate_finite_number,
    validate_mapping_size,
    validate_non_empty_str,
    validate_non_negative_int,
    validate_optional_clean_str,
    validate_optional_sequence_limit,
    validate_positive_int,
)
from rag.retrieval.contracts.query import NormalizedQuery
from rag.retrieval.contracts.request import RetrievalRequest


@dataclass(frozen=True)
class QuerySignals:
    normalized_query: str
    token_count: int
    char_count: int

    exact_token_count: int = 0
    quoted_term_count: int = 0
    code_like_token_count: int = 0
    file_hint_count: int = 0

    referential_marker_count: int = 0
    memory_marker_count: int = 0
    temporal_marker_count: int = 0
    comparison_marker_count: int = 0
    debug_marker_count: int = 0
    implementation_marker_count: int = 0
    architecture_marker_count: int = 0

    has_source_constraints: bool = False
    has_chat_history: bool = False
    has_session_context: bool = False
    wants_memory: bool = False

    lexical_specificity: float = 0.0
    corpus_affinity: float | None = None
    metadata: Payload = field(default_factory=dict)

    @classmethod
    def minimal(
        cls,
        *,
        request: RetrievalRequest,
        normalized_query: NormalizedQuery,
    ) -> QuerySignals:
        text = " ".join(normalized_query.normalized_query.strip().split())
        return cls(
            normalized_query=text,
            token_count=len(text.split()),
            char_count=len(text),
            has_source_constraints=not request.source_constraints.is_global,
            has_chat_history=bool(request.chat_history),
            has_session_context=bool(request.tenant.session_id),
            wants_memory=request.wants_memory,
        )

    @property
    def is_low_information(self) -> bool:
        return self.token_count <= 2 or self.char_count <= 8

    @property
    def is_exactish(self) -> bool:
        return (
            self.exact_token_count
            + self.quoted_term_count
            + self.code_like_token_count
            + self.file_hint_count
        ) > 0

    @property
    def is_chat_dependent(self) -> bool:
        return (
            self.referential_marker_count > 0
            or self.memory_marker_count > 0
        )

    @property
    def is_structural_or_multi_hop(self) -> bool:
        return (
            self.comparison_marker_count
            + self.implementation_marker_count
            + self.architecture_marker_count
        ) > 0

    @property
    def likely_needs_memory(self) -> bool:
        if not self.wants_memory:
            return False

        if self.memory_marker_count > 0:
            return True

        return (
            self.referential_marker_count > 0
            and (
                self.has_chat_history
                or self.has_session_context
            )
        )

    def validate(self, *, max_metadata_keys: int | None = None) -> None:
        validate_non_empty_str("query_signals.normalized_query", self.normalized_query)

        for name, value in (
            ("token_count", self.token_count),
            ("char_count", self.char_count),
            ("exact_token_count", self.exact_token_count),
            ("quoted_term_count", self.quoted_term_count),
            ("code_like_token_count", self.code_like_token_count),
            ("file_hint_count", self.file_hint_count),
            ("referential_marker_count", self.referential_marker_count),
            ("memory_marker_count", self.memory_marker_count),
            ("temporal_marker_count", self.temporal_marker_count),
            ("comparison_marker_count", self.comparison_marker_count),
            ("debug_marker_count", self.debug_marker_count),
            ("implementation_marker_count", self.implementation_marker_count),
            ("architecture_marker_count", self.architecture_marker_count),
        ):
            validate_non_negative_int(f"query_signals.{name}", value)

        for name, value in (
            ("has_source_constraints", self.has_source_constraints),
            ("has_chat_history", self.has_chat_history),
            ("has_session_context", self.has_session_context),
            ("wants_memory", self.wants_memory),
        ):
            validate_bool(f"query_signals.{name}", value)

        _validate_probability("query_signals.lexical_specificity", self.lexical_specificity)

        if self.corpus_affinity is not None:
            _validate_probability("query_signals.corpus_affinity", self.corpus_affinity)

        validate_mapping_size(
            "query_signals.metadata",
            self.metadata,
            max_metadata_keys,
        )


@dataclass(frozen=True)
class RetrievalConfidence:
    value: float = 0.0
    reason: str | None = None
    score_gap: float | None = None
    exact_coverage: float | None = None
    source_coverage: float | None = None
    metadata: Payload = field(default_factory=dict)

    @property
    def is_high(self) -> bool:
        return self.value >= 0.78

    @property
    def is_medium(self) -> bool:
        return self.value >= 0.55

    @property
    def is_low(self) -> bool:
        return self.value < 0.55

    def validate(self, *, max_metadata_keys: int | None = None) -> None:
        _validate_probability("retrieval_confidence.value", self.value)
        validate_optional_clean_str("retrieval_confidence.reason", self.reason)

        for name, value in (
            ("score_gap", self.score_gap),
            ("exact_coverage", self.exact_coverage),
            ("source_coverage", self.source_coverage),
        ):
            if value is not None:
                _validate_probability(f"retrieval_confidence.{name}", value)

        validate_mapping_size(
            "retrieval_confidence.metadata",
            self.metadata,
            max_metadata_keys,
        )


@dataclass(frozen=True)
class RetrievalStateSnapshot:
    hop: int
    last_action: RetrievalAction | None
    confidence: RetrievalConfidence
    sparse_count: int = 0
    dense_count: int = 0
    fused_count: int = 0
    expanded_count: int = 0
    reranked_count: int = 0
    context_count: int = 0
    active_top_identities: tuple[str, ...] = field(default_factory=tuple)
    remaining_latency_budget_ms: float | None = None
    metadata: Payload = field(default_factory=dict)

    def validate(self, *, max_metadata_keys: int | None = None) -> None:
        validate_non_negative_int("retrieval_state_snapshot.hop", self.hop)

        if self.last_action is not None and not isinstance(
            self.last_action,
            RetrievalAction,
        ):
            raise ValueError("retrieval_state_snapshot.last_action must be RetrievalAction.")

        self.confidence.validate(max_metadata_keys=max_metadata_keys)

        for name, value in (
            ("sparse_count", self.sparse_count),
            ("dense_count", self.dense_count),
            ("fused_count", self.fused_count),
            ("expanded_count", self.expanded_count),
            ("reranked_count", self.reranked_count),
            ("context_count", self.context_count),
        ):
            validate_non_negative_int(f"retrieval_state_snapshot.{name}", value)

        if self.remaining_latency_budget_ms is not None:
            validate_finite_number(
                "retrieval_state_snapshot.remaining_latency_budget_ms",
                self.remaining_latency_budget_ms,
            )
            if self.remaining_latency_budget_ms < 0:
                raise ValueError(
                    "retrieval_state_snapshot.remaining_latency_budget_ms cannot be negative."
                )

        validate_optional_sequence_limit(
            "retrieval_state_snapshot.active_top_identities",
            self.active_top_identities,
            None,
        )
        for index, identity in enumerate(self.active_top_identities):
            validate_non_empty_str(
                f"retrieval_state_snapshot.active_top_identities[{index}]",
                identity,
            )

        validate_mapping_size(
            "retrieval_state_snapshot.metadata",
            self.metadata,
            max_metadata_keys,
        )




def _append_state_sequence(current, item):
    """
    Tuple/list-safe append for retrieval graph state.

    The app wants list-like behavior, but graph runtimes may materialize
    state fields as tuples during snapshots or merges. This helper always
    returns a mutable list so downstream code can keep using normal list logic.
    """
    if current is None:
        return [item]

    if isinstance(current, list):
        current.append(item)
        return current

    if isinstance(current, tuple):
        return [*current, item]

    try:
        current.append(item)
        return current
    except AttributeError:
        try:
            return [*list(current), item]
        except TypeError:
            return [current, item]

@dataclass
class AdaptiveRetrievalState:
    request: RetrievalRequest
    normalized_query: NormalizedQuery
    query_signals: QuerySignals

    max_hops: int = 6
    hop: int = 0
    remaining_latency_budget_ms: float | None = None

    query_embedding: QueryEmbedding | None = None

    memory_candidates: tuple[RetrievalCandidate, ...] = field(default_factory=tuple)
    sparse_candidates: tuple[RetrievalCandidate, ...] = field(default_factory=tuple)
    dense_candidates: tuple[RetrievalCandidate, ...] = field(default_factory=tuple)
    fused_candidates: tuple[FusedCandidate, ...] = field(default_factory=tuple)
    expanded_candidates: tuple[RetrievalCandidate, ...] = field(default_factory=tuple)
    reranked_candidates: tuple[RerankedCandidate, ...] = field(default_factory=tuple)
    context_blocks: tuple[ContextBlock, ...] = field(default_factory=tuple)

    confidence: RetrievalConfidence = field(default_factory=RetrievalConfidence)
    action_records: tuple[RetrievalActionRecord, ...] = field(default_factory=tuple)
    snapshots: tuple[RetrievalStateSnapshot, ...] = field(default_factory=tuple)

    warnings: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)
    metadata: Payload = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        request: RetrievalRequest,
        normalized_query: NormalizedQuery,
        query_signals: QuerySignals | None = None,
        max_hops: int = 6,
        latency_budget_ms: float | None = None,
        metadata: Payload | None = None,
    ) -> AdaptiveRetrievalState:
        request.validate()
        normalized_query.validate()

        state = cls(
            request=request,
            normalized_query=normalized_query,
            query_signals=query_signals
            or QuerySignals.minimal(
                request=request,
                normalized_query=normalized_query,
            ),
            max_hops=max_hops,
            remaining_latency_budget_ms=latency_budget_ms,
            metadata=metadata or {},
        )
        state.validate()
        state.capture_snapshot()
        return state

    @property
    def completed_actions(self) -> set[RetrievalAction]:
        return completed_actions(self.action_records)

    @property
    def failed_actions(self) -> set[RetrievalAction]:
        return {
            record.action
            for record in self.action_records
            if record.status == RetrievalActionStatus.FAILED
        }

    @property
    def can_continue(self) -> bool:
        return self.hop < self.max_hops and not self.errors

    @property
    def has_candidates(self) -> bool:
        return bool(self.active_candidates())

    @property
    def candidate_count(self) -> int:
        return (
            len(self.memory_candidates)
            + len(self.sparse_candidates)
            + len(self.dense_candidates)
            + len(self.fused_candidates)
            + len(self.expanded_candidates)
            + len(self.reranked_candidates)
        )

    def has_completed(self, action: RetrievalAction) -> bool:
        return action in self.completed_actions

    def increment_hop(self) -> None:
        self.hop += 1

    def add_action_record(self, record: RetrievalActionRecord) -> None:
        record.validate()
        self.action_records = self.action_records + (record,)

        if record.latency_ms is not None and self.remaining_latency_budget_ms is not None:
            self.remaining_latency_budget_ms = max(
                0.0,
                self.remaining_latency_budget_ms - record.latency_ms,
            )

    def add_warning(self, warning: str) -> None:
        cleaned = warning.strip()
        if cleaned:
            self.warnings = self.warnings + (cleaned,)

    def add_error(self, error: str) -> None:
        cleaned = error.strip()
        if cleaned:
            self.errors = self.errors + (cleaned,)

    def set_confidence(
        self,
        value: float,
        *,
        reason: str | None = None,
        score_gap: float | None = None,
        exact_coverage: float | None = None,
        source_coverage: float | None = None,
        metadata: Payload | None = None,
    ) -> None:
        self.confidence = RetrievalConfidence(
            value=value,
            reason=reason,
            score_gap=score_gap,
            exact_coverage=exact_coverage,
            source_coverage=source_coverage,
            metadata=metadata or {},
        )
        self.confidence.validate()

    def active_candidates(
        self,
    ) -> tuple[
        RerankedCandidate | FusedCandidate | RetrievalCandidate,
        ...
    ]:
        if self.reranked_candidates:
            return self.reranked_candidates

        if self.expanded_candidates and self.fused_candidates:
            return self.fused_candidates + self.expanded_candidates

        if self.expanded_candidates:
            return self.expanded_candidates

        if self.fused_candidates:
            return self.fused_candidates

        merged: list[RetrievalCandidate] = []
        merged.extend(self.memory_candidates)
        merged.extend(self.sparse_candidates)
        merged.extend(self.dense_candidates)
        return tuple(merged)

    def active_top_identities(self, *, limit: int = 5) -> tuple[str, ...]:
        validate_positive_int("active_top_identities.limit", limit)
        identities: list[str] = []
        seen: set[str] = set()

        for candidate in self.active_candidates():
            identity = candidate_identity(candidate)
            if identity in seen:
                continue

            seen.add(identity)
            identities.append(identity)

            if len(identities) >= limit:
                break

        return tuple(identities)

    def capture_snapshot(self, *, metadata: Payload | None = None) -> RetrievalStateSnapshot:
        last_action = self.action_records[-1].action if self.action_records else None

        snapshot = RetrievalStateSnapshot(
            hop=self.hop,
            last_action=last_action,
            confidence=self.confidence,
            sparse_count=len(self.sparse_candidates),
            dense_count=len(self.dense_candidates),
            fused_count=len(self.fused_candidates),
            expanded_count=len(self.expanded_candidates),
            reranked_count=len(self.reranked_candidates),
            context_count=len(self.context_blocks),
            active_top_identities=self.active_top_identities(),
            remaining_latency_budget_ms=self.remaining_latency_budget_ms,
            metadata=metadata or {},
        )
        snapshot.validate()
        self.snapshots = self.snapshots + (snapshot,)
        return snapshot

    def validate(self, *, max_metadata_keys: int | None = None) -> None:
        if not isinstance(self.request, RetrievalRequest):
            raise ValueError("adaptive_retrieval_state.request must be RetrievalRequest.")
        if not isinstance(self.normalized_query, NormalizedQuery):
            raise ValueError("adaptive_retrieval_state.normalized_query must be NormalizedQuery.")
        if not isinstance(self.query_signals, QuerySignals):
            raise ValueError("adaptive_retrieval_state.query_signals must be QuerySignals.")

        self.request.validate()
        self.normalized_query.validate()
        self.query_signals.validate(max_metadata_keys=max_metadata_keys)

        validate_positive_int("adaptive_retrieval_state.max_hops", self.max_hops)
        validate_non_negative_int("adaptive_retrieval_state.hop", self.hop)

        if self.remaining_latency_budget_ms is not None:
            validate_finite_number(
                "adaptive_retrieval_state.remaining_latency_budget_ms",
                self.remaining_latency_budget_ms,
            )
            if self.remaining_latency_budget_ms < 0:
                raise ValueError(
                    "adaptive_retrieval_state.remaining_latency_budget_ms cannot be negative."
                )

        if self.query_embedding is not None:
            if not isinstance(self.query_embedding, QueryEmbedding):
                raise ValueError("adaptive_retrieval_state.query_embedding must be QueryEmbedding.")
            self.query_embedding.validate()

        _validate_candidate_tuple(
            "adaptive_retrieval_state.memory_candidates",
            self.memory_candidates,
            RetrievalCandidate,
        )
        _validate_candidate_tuple(
            "adaptive_retrieval_state.sparse_candidates",
            self.sparse_candidates,
            RetrievalCandidate,
        )
        _validate_candidate_tuple(
            "adaptive_retrieval_state.dense_candidates",
            self.dense_candidates,
            RetrievalCandidate,
        )
        _validate_candidate_tuple(
            "adaptive_retrieval_state.fused_candidates",
            self.fused_candidates,
            FusedCandidate,
        )
        _validate_candidate_tuple(
            "adaptive_retrieval_state.expanded_candidates",
            self.expanded_candidates,
            RetrievalCandidate,
        )
        _validate_candidate_tuple(
            "adaptive_retrieval_state.reranked_candidates",
            self.reranked_candidates,
            RerankedCandidate,
        )

        for index, block in enumerate(self.context_blocks):
            if not isinstance(block, ContextBlock):
                raise ValueError(
                    f"adaptive_retrieval_state.context_blocks[{index}] must be ContextBlock."
                )
            block.validate()

        self.confidence.validate(max_metadata_keys=max_metadata_keys)

        for index, record in enumerate(self.action_records):
            if not isinstance(record, RetrievalActionRecord):
                raise ValueError(
                    f"adaptive_retrieval_state.action_records[{index}] must be RetrievalActionRecord."
                )
            record.validate(max_metadata_keys=max_metadata_keys)

        for index, snapshot in enumerate(self.snapshots):
            if not isinstance(snapshot, RetrievalStateSnapshot):
                raise ValueError(
                    f"adaptive_retrieval_state.snapshots[{index}] must be RetrievalStateSnapshot."
                )
            snapshot.validate(max_metadata_keys=max_metadata_keys)

        _validate_clean_messages("adaptive_retrieval_state.warnings", self.warnings)
        _validate_clean_messages("adaptive_retrieval_state.errors", self.errors)

        validate_mapping_size(
            "adaptive_retrieval_state.metadata",
            self.metadata,
            max_metadata_keys,
        )


def candidate_identity(candidate: Any) -> str:
    base = _unwrap_candidate(candidate)

    for attr_name in (
        "chunk_id",
        "point_id",
        "source_id",
        "parser_doc_id",
        "section_id",
    ):
        value = _clean_optional(getattr(base, attr_name, None))
        if value:
            return value

    ref = getattr(base, "ref", None) or getattr(base, "source_ref", None)
    if ref is not None:
        for attr_name in (
            "chunk_id",
            "point_id",
            "source_id",
            "parser_doc_id",
            "section_id",
            "text_sha256",
        ):
            value = _clean_optional(getattr(ref, attr_name, None))
            if value:
                return value

    payload = getattr(base, "payload", {}) or {}
    if isinstance(payload, Mapping):
        for key in (
            "chunk_id",
            "point_id",
            "source_id",
            "parser_doc_id",
            "section_id",
            "text_sha256",
            "file_path",
            "file_name",
        ):
            value = _clean_optional(payload.get(key))
            if value:
                return value

    metadata = getattr(base, "metadata", {}) or {}
    if isinstance(metadata, Mapping):
        for key in (
            "chunk_id",
            "point_id",
            "source_id",
            "parser_doc_id",
            "section_id",
            "text_sha256",
            "file_path",
            "file_name",
        ):
            value = _clean_optional(metadata.get(key))
            if value:
                return value

    text = _clean_optional(getattr(base, "text", None))
    if text:
        return text[:200]

    return str(id(candidate))


def top_identity_set_changed(
    before: Sequence[Any],
    after: Sequence[Any],
    *,
    limit: int = 5,
) -> bool:
    validate_positive_int("top_identity_set_changed.limit", limit)
    before_set = tuple(dict.fromkeys(candidate_identity(item) for item in before[:limit]))
    after_set = tuple(dict.fromkeys(candidate_identity(item) for item in after[:limit]))
    return before_set != after_set


def _unwrap_candidate(candidate: Any) -> Any:
    if isinstance(candidate, RerankedCandidate):
        return candidate.candidate

    if isinstance(candidate, FusedCandidate):
        return candidate.candidate

    return candidate


def _validate_candidate_tuple(
    name: str,
    candidates: tuple[Any, ...],
    expected_type: type[Any],
) -> None:
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, expected_type):
            raise ValueError(f"{name}[{index}] must be {expected_type.__name__}.")

        candidate.validate()


def _validate_clean_messages(name: str, values: Sequence[str]) -> None:
    for index, value in enumerate(values):
        validate_non_empty_str(f"{name}[{index}]", value)


def _validate_probability(name: str, value: float) -> None:
    validate_finite_number(name, value)

    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None

    cleaned = str(value).strip()
    return cleaned or None
