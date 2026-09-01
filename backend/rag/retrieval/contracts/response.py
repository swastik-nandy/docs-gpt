from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

from rag.retrieval.contracts.candidates import RerankedCandidate
from rag.retrieval.contracts.config import (
    RetrievalContractConfig,
    get_retrieval_contract_config,
)
from rag.retrieval.contracts.context import ContextBlock, PackedContext
from rag.retrieval.contracts.enums import ConfidenceBand
from rag.retrieval.contracts.primitives import (
    Payload,
    validate_bool,
    validate_enum_instance,
    validate_finite_number,
    validate_mapping_size,
    validate_non_empty_str,
    validate_optional_clean_str,
    validate_optional_sequence_limit,
)


@dataclass(frozen=True)
class RetrievalDecision:
    is_sufficient: bool
    confidence: ConfidenceBand = ConfidenceBand.UNKNOWN
    confidence_score: float | None = None
    reason: str | None = None
    should_rewrite: bool = False
    rewrite_reason: str | None = None
    metadata: Payload = field(default_factory=dict)

    @property
    def needs_rewrite(self) -> bool:
        return self.should_rewrite

    @property
    def is_confident(self) -> bool:
        return self.confidence in {ConfidenceBand.MEDIUM, ConfidenceBand.HIGH}

    @property
    def is_low_confidence(self) -> bool:
        return self.confidence in {
            ConfidenceBand.UNKNOWN,
            ConfidenceBand.VERY_LOW,
            ConfidenceBand.LOW,
        }

    def normalized(self) -> Self:
        return type(self)(
            is_sufficient=bool(self.is_sufficient),
            confidence=self.confidence,
            confidence_score=(
                float(self.confidence_score)
                if self.confidence_score is not None
                else None
            ),
            reason=_clean_optional(self.reason),
            should_rewrite=bool(self.should_rewrite),
            rewrite_reason=_clean_optional(self.rewrite_reason),
            metadata=dict(self.metadata),
        )

    def validate(
        self,
        *,
        config: RetrievalContractConfig | None = None,
    ) -> None:
        resolved_config = config or get_retrieval_contract_config()
        max_metadata_keys = resolved_config.request_limits.max_metadata_keys

        validate_bool("retrieval_decision.is_sufficient", self.is_sufficient)
        validate_enum_instance(
            "retrieval_decision.confidence",
            self.confidence,
            ConfidenceBand,
        )
        validate_bool("retrieval_decision.should_rewrite", self.should_rewrite)
        validate_optional_clean_str("retrieval_decision.reason", self.reason)
        validate_optional_clean_str(
            "retrieval_decision.rewrite_reason",
            self.rewrite_reason,
        )

        if self.confidence_score is not None:
            validate_finite_number(
                "retrieval_decision.confidence_score",
                self.confidence_score,
            )

            if not 0 <= self.confidence_score <= 1:
                raise ValueError(
                    "retrieval_decision.confidence_score must be between 0 and 1."
                )

        if self.should_rewrite and not self.rewrite_reason:
            raise ValueError(
                "retrieval_decision.rewrite_reason is required when "
                "should_rewrite is true."
            )

        validate_mapping_size(
            "retrieval_decision.metadata",
            self.metadata,
            max_metadata_keys,
        )


@dataclass(frozen=True)
class RetrievalResponse:
    query: str
    decision: RetrievalDecision
    packed_context: PackedContext
    request_id: str | None = None
    conversation_id: str | None = None
    candidates: tuple[RerankedCandidate, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)
    trace: Payload = field(default_factory=dict)
    metadata: Payload = field(default_factory=dict)

    @property
    def context_blocks(self) -> tuple[ContextBlock, ...]:
        return self.packed_context.blocks

    @property
    def has_context(self) -> bool:
        return self.packed_context.has_blocks

    @property
    def is_successful(self) -> bool:
        return not self.errors and self.decision.is_sufficient

    @property
    def should_rewrite(self) -> bool:
        return self.decision.should_rewrite

    def normalized(self) -> Self:
        return type(self)(
            query=self.query.strip(),
            decision=self.decision.normalized(),
            packed_context=self.packed_context.normalized(),
            request_id=_clean_optional(self.request_id),
            conversation_id=_clean_optional(self.conversation_id),
            candidates=tuple(candidate.normalized() for candidate in self.candidates),
            warnings=tuple(warning.strip() for warning in self.warnings if warning.strip()),
            errors=tuple(error.strip() for error in self.errors if error.strip()),
            trace=dict(self.trace),
            metadata=dict(self.metadata),
        )

    def validate(
        self,
        *,
        config: RetrievalContractConfig | None = None,
    ) -> None:
        resolved_config = config or get_retrieval_contract_config()
        max_metadata_keys = resolved_config.request_limits.max_metadata_keys

        validate_non_empty_str("retrieval_response.query", self.query)
        validate_optional_clean_str("retrieval_response.request_id", self.request_id)
        validate_optional_clean_str(
            "retrieval_response.conversation_id",
            self.conversation_id,
        )

        if not isinstance(self.decision, RetrievalDecision):
            raise ValueError(
                "retrieval_response.decision must be RetrievalDecision."
            )

        if not isinstance(self.packed_context, PackedContext):
            raise ValueError(
                "retrieval_response.packed_context must be PackedContext."
            )

        self.decision.validate(config=resolved_config)
        self.packed_context.validate(
            config=resolved_config,
            allow_empty=not self.decision.is_sufficient,
        )

        validate_optional_sequence_limit(
            "retrieval_response.candidates",
            self.candidates,
            None,
        )

        for index, candidate in enumerate(self.candidates):
            if not isinstance(candidate, RerankedCandidate):
                raise ValueError(
                    f"retrieval_response.candidates[{index}] "
                    "must be RerankedCandidate."
                )

            candidate.validate(config=resolved_config)

        for index, warning in enumerate(self.warnings):
            validate_non_empty_str(
                f"retrieval_response.warnings[{index}]",
                warning,
            )

        for index, error in enumerate(self.errors):
            validate_non_empty_str(
                f"retrieval_response.errors[{index}]",
                error,
            )

        validate_mapping_size(
            "retrieval_response.trace",
            self.trace,
            max_metadata_keys,
        )
        validate_mapping_size(
            "retrieval_response.metadata",
            self.metadata,
            max_metadata_keys,
        )


def build_empty_retrieval_response(
    *,
    query: str,
    reason: str,
    request_id: str | None = None,
    conversation_id: str | None = None,
    config: RetrievalContractConfig | None = None,
    metadata: Payload | None = None,
) -> RetrievalResponse:
    resolved_config = config or get_retrieval_contract_config()

    response = RetrievalResponse(
        query=query,
        request_id=request_id,
        conversation_id=conversation_id,
        decision=RetrievalDecision(
            is_sufficient=False,
            confidence=ConfidenceBand.VERY_LOW,
            reason=reason,
        ),
        packed_context=PackedContext(
            query=query,
            blocks=(),
            total_tokens=0,
            max_total_tokens=resolved_config.context_defaults.max_total_tokens,
            reserve_tokens_for_answer=(
                resolved_config.context_defaults.reserve_tokens_for_answer
            ),
        ),
        metadata=metadata or {},
    )
    response.validate(config=resolved_config)
    return response


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None

    cleaned = value.strip()
    return cleaned or None