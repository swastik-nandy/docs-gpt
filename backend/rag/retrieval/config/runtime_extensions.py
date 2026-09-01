from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


_ALLOWED_RELAXATION_STEPS = frozenset(
    {
        "date",
        "document_type",
        "section_id",
        "heading_path",
        "ir_id",
        "parser_doc_id",
        "job_id",
        "source_id",
    }
)


@dataclass(frozen=True)
class AdaptiveScopeRelaxationRuntimeConfig:
    enabled: bool
    max_relaxation_steps: int
    require_escalate_k_first: bool
    relaxation_order: tuple[str, ...]
    date_filter_keys: frozenset[str]
    document_type_filter_keys: frozenset[str]

    def validate(self) -> None:
        _validate_bool(
            "adaptive.scope_relaxation.enabled",
            self.enabled,
        )
        _validate_positive_int(
            "adaptive.scope_relaxation.max_relaxation_steps",
            self.max_relaxation_steps,
        )
        _validate_bool(
            "adaptive.scope_relaxation.require_escalate_k_first",
            self.require_escalate_k_first,
        )

        if not self.relaxation_order:
            raise ValueError(
                "adaptive.scope_relaxation.relaxation_order cannot be empty."
            )

        if len(set(self.relaxation_order)) != len(self.relaxation_order):
            raise ValueError(
                "adaptive.scope_relaxation.relaxation_order "
                "cannot contain duplicates."
            )

        unsupported_steps = set(self.relaxation_order) - _ALLOWED_RELAXATION_STEPS
        if unsupported_steps:
            raise ValueError(
                "adaptive.scope_relaxation.relaxation_order contains "
                f"unsupported steps: {sorted(unsupported_steps)}."
            )

        if self.max_relaxation_steps > len(self.relaxation_order):
            raise ValueError(
                "adaptive.scope_relaxation.max_relaxation_steps cannot exceed "
                "the number of configured relaxation steps."
            )

        _validate_non_empty_strings(
            "adaptive.scope_relaxation.date_filter_keys",
            self.date_filter_keys,
        )
        _validate_non_empty_strings(
            "adaptive.scope_relaxation.document_type_filter_keys",
            self.document_type_filter_keys,
        )


@dataclass(frozen=True)
class AdaptiveSignalRuntimeConfig:
    max_metadata_keys: int
    max_metadata_terms: int
    min_long_token_chars: int
    include_term_metadata: bool
    lowercase_corpus_tokens: bool

    def validate(self) -> None:
        _validate_positive_int(
            "adaptive.signals.max_metadata_keys",
            self.max_metadata_keys,
        )
        _validate_positive_int(
            "adaptive.signals.max_metadata_terms",
            self.max_metadata_terms,
        )
        _validate_positive_int(
            "adaptive.signals.min_long_token_chars",
            self.min_long_token_chars,
        )
        _validate_bool(
            "adaptive.signals.include_term_metadata",
            self.include_term_metadata,
        )
        _validate_bool(
            "adaptive.signals.lowercase_corpus_tokens",
            self.lowercase_corpus_tokens,
        )


@dataclass(frozen=True)
class AdaptiveEntityRecencyRuntimeConfig:
    enabled: bool
    half_life_turns: float

    def validate(self) -> None:
        _validate_bool(
            "adaptive.signals.entity_recency.enabled",
            self.enabled,
        )
        _validate_positive_number(
            "adaptive.signals.entity_recency.half_life_turns",
            self.half_life_turns,
        )


@dataclass(frozen=True)
class AdaptiveConfidenceAdjustmentRuntimeConfig:
    score_gap_delta: float = 0.0
    exact_coverage_delta: float = 0.0
    source_coverage_delta: float = 0.0
    candidate_presence_delta: float = 0.0
    entropy_delta: float = 0.0

    def validate(self, *, section: str) -> None:
        for field_name, value in self.__dict__.items():
            _validate_bounded_number(
                f"{section}.{field_name}",
                value,
                minimum=-1.0,
                maximum=1.0,
            )


@dataclass(frozen=True)
class AdaptiveEntityCoverageRuntimeConfig:
    base_weight: float
    recency_multiplier: float
    entity_only_min_weight: float

    def validate(self) -> None:
        _validate_probability(
            "adaptive.confidence.entity_coverage.base_weight",
            self.base_weight,
        )
        _validate_probability(
            "adaptive.confidence.entity_coverage.recency_multiplier",
            self.recency_multiplier,
        )
        _validate_probability(
            "adaptive.confidence.entity_coverage.entity_only_min_weight",
            self.entity_only_min_weight,
        )


@dataclass(frozen=True)
class AdaptiveEvidenceVerificationRuntimeConfig:
    enabled: bool
    max_candidates: int
    max_candidate_chars: int
    grounding_threshold: float
    failed_verification_penalty: float

    def validate(self) -> None:
        _validate_bool(
            "adaptive.evidence_verification.enabled",
            self.enabled,
        )
        _validate_positive_int(
            "adaptive.evidence_verification.max_candidates",
            self.max_candidates,
        )
        _validate_positive_int(
            "adaptive.evidence_verification.max_candidate_chars",
            self.max_candidate_chars,
        )
        _validate_probability(
            "adaptive.evidence_verification.grounding_threshold",
            self.grounding_threshold,
        )
        _validate_probability(
            "adaptive.evidence_verification.failed_verification_penalty",
            self.failed_verification_penalty,
        )


@dataclass(frozen=True)
class AdaptiveCrossTurnRedundancyRuntimeConfig:
    enabled: bool
    history_turns: int
    minimum_repeat_count: int
    penalty: float
    disable_for_drill_down: bool

    def validate(self) -> None:
        _validate_bool(
            "adaptive.cross_turn_redundancy.enabled",
            self.enabled,
        )
        _validate_positive_int(
            "adaptive.cross_turn_redundancy.history_turns",
            self.history_turns,
        )
        _validate_positive_int(
            "adaptive.cross_turn_redundancy.minimum_repeat_count",
            self.minimum_repeat_count,
        )
        _validate_probability(
            "adaptive.cross_turn_redundancy.penalty",
            self.penalty,
        )
        _validate_bool(
            "adaptive.cross_turn_redundancy.disable_for_drill_down",
            self.disable_for_drill_down,
        )

        if self.minimum_repeat_count > self.history_turns:
            raise ValueError(
                "adaptive.cross_turn_redundancy.minimum_repeat_count "
                "cannot exceed history_turns."
            )


@dataclass(frozen=True)
class QueryAwareCalibrationRuntimeConfig:
    enabled: bool

    focus_coverage_weight: float
    title_coverage_weight: float
    title_compactness_weight: float
    phrase_in_title_bonus: float
    phrase_in_text_bonus: float

    broad_query_max_focus_terms: int
    specialization_threshold: float
    title_compactness_threshold: float
    specialization_penalty: float

    high_specialization_threshold: float
    high_specialization_penalty: float

    code_marker_threshold: int
    code_penalty: float

    short_chunk_chars: int
    short_chunk_penalty: float
    zero_focus_penalty: float

    minimum_adjustment: float
    maximum_adjustment: float

    def validate(self) -> None:
        _validate_bool(
            "reranker.query_aware_calibration.enabled",
            self.enabled,
        )

        for field_name in (
            "focus_coverage_weight",
            "title_coverage_weight",
            "title_compactness_weight",
            "phrase_in_title_bonus",
            "phrase_in_text_bonus",
            "specialization_penalty",
            "high_specialization_penalty",
            "code_penalty",
            "short_chunk_penalty",
            "zero_focus_penalty",
        ):
            _validate_non_negative_number(
                f"reranker.query_aware_calibration.{field_name}",
                getattr(self, field_name),
            )

        _validate_positive_int(
            "reranker.query_aware_calibration.broad_query_max_focus_terms",
            self.broad_query_max_focus_terms,
        )
        _validate_probability(
            "reranker.query_aware_calibration.specialization_threshold",
            self.specialization_threshold,
        )
        _validate_probability(
            "reranker.query_aware_calibration.title_compactness_threshold",
            self.title_compactness_threshold,
        )
        _validate_probability(
            "reranker.query_aware_calibration.high_specialization_threshold",
            self.high_specialization_threshold,
        )

        if self.high_specialization_threshold < self.specialization_threshold:
            raise ValueError(
                "reranker.query_aware_calibration."
                "high_specialization_threshold must be greater than or equal "
                "to specialization_threshold."
            )

        _validate_positive_int(
            "reranker.query_aware_calibration.code_marker_threshold",
            self.code_marker_threshold,
        )
        _validate_positive_int(
            "reranker.query_aware_calibration.short_chunk_chars",
            self.short_chunk_chars,
        )

        _validate_finite_number(
            "reranker.query_aware_calibration.minimum_adjustment",
            self.minimum_adjustment,
        )
        _validate_finite_number(
            "reranker.query_aware_calibration.maximum_adjustment",
            self.maximum_adjustment,
        )

        if self.minimum_adjustment > self.maximum_adjustment:
            raise ValueError(
                "reranker.query_aware_calibration.minimum_adjustment "
                "must be less than or equal to maximum_adjustment."
            )


@dataclass(frozen=True)
class RetrievalRuntimeExtensions:
    scope_relaxation: AdaptiveScopeRelaxationRuntimeConfig
    signals: AdaptiveSignalRuntimeConfig
    entity_recency: AdaptiveEntityRecencyRuntimeConfig

    chat_dependent_confidence: AdaptiveConfidenceAdjustmentRuntimeConfig
    structural_multi_hop_confidence: AdaptiveConfidenceAdjustmentRuntimeConfig
    entity_coverage: AdaptiveEntityCoverageRuntimeConfig

    evidence_verification: AdaptiveEvidenceVerificationRuntimeConfig
    cross_turn_redundancy: AdaptiveCrossTurnRedundancyRuntimeConfig

    query_aware_calibration: QueryAwareCalibrationRuntimeConfig

    def validate(self) -> None:
        self.scope_relaxation.validate()
        self.signals.validate()
        self.entity_recency.validate()

        self.chat_dependent_confidence.validate(
            section="adaptive.confidence.chat_dependent"
        )
        self.structural_multi_hop_confidence.validate(
            section="adaptive.confidence.structural_multi_hop"
        )
        self.entity_coverage.validate()

        self.evidence_verification.validate()
        self.cross_turn_redundancy.validate()
        self.query_aware_calibration.validate()


def parse_retrieval_runtime_extensions(
    raw: Mapping[str, Any],
) -> RetrievalRuntimeExtensions:
    adaptive = _required_mapping(raw, "adaptive")

    scope_relaxation = _required_mapping(
        adaptive,
        "scope_relaxation",
        parent="adaptive",
    )

    signals = _required_mapping(
        adaptive,
        "signals",
        parent="adaptive",
    )
    entity_recency = _required_mapping(
        signals,
        "entity_recency",
        parent="adaptive.signals",
    )

    confidence = _required_mapping(
        adaptive,
        "confidence",
        parent="adaptive",
    )
    chat_dependent = _required_mapping(
        confidence,
        "chat_dependent",
        parent="adaptive.confidence",
    )
    structural_multi_hop = _required_mapping(
        confidence,
        "structural_multi_hop",
        parent="adaptive.confidence",
    )
    entity_coverage = _required_mapping(
        confidence,
        "entity_coverage",
        parent="adaptive.confidence",
    )

    evidence_verification = _required_mapping(
        adaptive,
        "evidence_verification",
        parent="adaptive",
    )
    cross_turn_redundancy = _required_mapping(
        adaptive,
        "cross_turn_redundancy",
        parent="adaptive",
    )

    reranker = _required_mapping(raw, "reranker")
    query_aware_calibration = _required_mapping(
        reranker,
        "query_aware_calibration",
        parent="reranker",
    )

    extensions = RetrievalRuntimeExtensions(
        scope_relaxation=AdaptiveScopeRelaxationRuntimeConfig(
            enabled=_required_bool(
                scope_relaxation,
                "enabled",
                section="adaptive.scope_relaxation",
            ),
            max_relaxation_steps=_required_int(
                scope_relaxation,
                "max_relaxation_steps",
                section="adaptive.scope_relaxation",
            ),
            require_escalate_k_first=_required_bool(
                scope_relaxation,
                "require_escalate_k_first",
                section="adaptive.scope_relaxation",
            ),
            relaxation_order=_required_str_tuple(
                scope_relaxation,
                "relaxation_order",
                section="adaptive.scope_relaxation",
            ),
            date_filter_keys=frozenset(
                _required_str_tuple(
                    scope_relaxation,
                    "date_filter_keys",
                    section="adaptive.scope_relaxation",
                )
            ),
            document_type_filter_keys=frozenset(
                _required_str_tuple(
                    scope_relaxation,
                    "document_type_filter_keys",
                    section="adaptive.scope_relaxation",
                )
            ),
        ),
        signals=AdaptiveSignalRuntimeConfig(
            max_metadata_keys=_required_int(
                signals,
                "max_metadata_keys",
                section="adaptive.signals",
            ),
            max_metadata_terms=_required_int(
                signals,
                "max_metadata_terms",
                section="adaptive.signals",
            ),
            min_long_token_chars=_required_int(
                signals,
                "min_long_token_chars",
                section="adaptive.signals",
            ),
            include_term_metadata=_required_bool(
                signals,
                "include_term_metadata",
                section="adaptive.signals",
            ),
            lowercase_corpus_tokens=_required_bool(
                signals,
                "lowercase_corpus_tokens",
                section="adaptive.signals",
            ),
        ),
        entity_recency=AdaptiveEntityRecencyRuntimeConfig(
            enabled=_required_bool(
                entity_recency,
                "enabled",
                section="adaptive.signals.entity_recency",
            ),
            half_life_turns=_required_float(
                entity_recency,
                "half_life_turns",
                section="adaptive.signals.entity_recency",
            ),
        ),
        chat_dependent_confidence=AdaptiveConfidenceAdjustmentRuntimeConfig(
            score_gap_delta=_optional_float(
                chat_dependent,
                "score_gap_delta",
            ),
            exact_coverage_delta=_optional_float(
                chat_dependent,
                "exact_coverage_delta",
            ),
            source_coverage_delta=_optional_float(
                chat_dependent,
                "source_coverage_delta",
            ),
            candidate_presence_delta=_optional_float(
                chat_dependent,
                "candidate_presence_delta",
            ),
            entropy_delta=_optional_float(
                chat_dependent,
                "entropy_delta",
            ),
        ),
        structural_multi_hop_confidence=(
            AdaptiveConfidenceAdjustmentRuntimeConfig(
                score_gap_delta=_optional_float(
                    structural_multi_hop,
                    "score_gap_delta",
                ),
                exact_coverage_delta=_optional_float(
                    structural_multi_hop,
                    "exact_coverage_delta",
                ),
                source_coverage_delta=_optional_float(
                    structural_multi_hop,
                    "source_coverage_delta",
                ),
                candidate_presence_delta=_optional_float(
                    structural_multi_hop,
                    "candidate_presence_delta",
                ),
                entropy_delta=_optional_float(
                    structural_multi_hop,
                    "entropy_delta",
                ),
            )
        ),
        entity_coverage=AdaptiveEntityCoverageRuntimeConfig(
            base_weight=_required_float(
                entity_coverage,
                "base_weight",
                section="adaptive.confidence.entity_coverage",
            ),
            recency_multiplier=_required_float(
                entity_coverage,
                "recency_multiplier",
                section="adaptive.confidence.entity_coverage",
            ),
            entity_only_min_weight=_required_float(
                entity_coverage,
                "entity_only_min_weight",
                section="adaptive.confidence.entity_coverage",
            ),
        ),
        evidence_verification=AdaptiveEvidenceVerificationRuntimeConfig(
            enabled=_required_bool(
                evidence_verification,
                "enabled",
                section="adaptive.evidence_verification",
            ),
            max_candidates=_required_int(
                evidence_verification,
                "max_candidates",
                section="adaptive.evidence_verification",
            ),
            max_candidate_chars=_required_int(
                evidence_verification,
                "max_candidate_chars",
                section="adaptive.evidence_verification",
            ),
            grounding_threshold=_required_float(
                evidence_verification,
                "grounding_threshold",
                section="adaptive.evidence_verification",
            ),
            failed_verification_penalty=_required_float(
                evidence_verification,
                "failed_verification_penalty",
                section="adaptive.evidence_verification",
            ),
        ),
        cross_turn_redundancy=AdaptiveCrossTurnRedundancyRuntimeConfig(
            enabled=_required_bool(
                cross_turn_redundancy,
                "enabled",
                section="adaptive.cross_turn_redundancy",
            ),
            history_turns=_required_int(
                cross_turn_redundancy,
                "history_turns",
                section="adaptive.cross_turn_redundancy",
            ),
            minimum_repeat_count=_required_int(
                cross_turn_redundancy,
                "minimum_repeat_count",
                section="adaptive.cross_turn_redundancy",
            ),
            penalty=_required_float(
                cross_turn_redundancy,
                "penalty",
                section="adaptive.cross_turn_redundancy",
            ),
            disable_for_drill_down=_required_bool(
                cross_turn_redundancy,
                "disable_for_drill_down",
                section="adaptive.cross_turn_redundancy",
            ),
        ),
        query_aware_calibration=QueryAwareCalibrationRuntimeConfig(
            enabled=_required_bool(
                query_aware_calibration,
                "enabled",
                section="reranker.query_aware_calibration",
            ),
            focus_coverage_weight=_required_float(
                query_aware_calibration,
                "focus_coverage_weight",
                section="reranker.query_aware_calibration",
            ),
            title_coverage_weight=_required_float(
                query_aware_calibration,
                "title_coverage_weight",
                section="reranker.query_aware_calibration",
            ),
            title_compactness_weight=_required_float(
                query_aware_calibration,
                "title_compactness_weight",
                section="reranker.query_aware_calibration",
            ),
            phrase_in_title_bonus=_required_float(
                query_aware_calibration,
                "phrase_in_title_bonus",
                section="reranker.query_aware_calibration",
            ),
            phrase_in_text_bonus=_required_float(
                query_aware_calibration,
                "phrase_in_text_bonus",
                section="reranker.query_aware_calibration",
            ),
            broad_query_max_focus_terms=_required_int(
                query_aware_calibration,
                "broad_query_max_focus_terms",
                section="reranker.query_aware_calibration",
            ),
            specialization_threshold=_required_float(
                query_aware_calibration,
                "specialization_threshold",
                section="reranker.query_aware_calibration",
            ),
            title_compactness_threshold=_required_float(
                query_aware_calibration,
                "title_compactness_threshold",
                section="reranker.query_aware_calibration",
            ),
            specialization_penalty=_required_float(
                query_aware_calibration,
                "specialization_penalty",
                section="reranker.query_aware_calibration",
            ),
            high_specialization_threshold=_required_float(
                query_aware_calibration,
                "high_specialization_threshold",
                section="reranker.query_aware_calibration",
            ),
            high_specialization_penalty=_required_float(
                query_aware_calibration,
                "high_specialization_penalty",
                section="reranker.query_aware_calibration",
            ),
            code_marker_threshold=_required_int(
                query_aware_calibration,
                "code_marker_threshold",
                section="reranker.query_aware_calibration",
            ),
            code_penalty=_required_float(
                query_aware_calibration,
                "code_penalty",
                section="reranker.query_aware_calibration",
            ),
            short_chunk_chars=_required_int(
                query_aware_calibration,
                "short_chunk_chars",
                section="reranker.query_aware_calibration",
            ),
            short_chunk_penalty=_required_float(
                query_aware_calibration,
                "short_chunk_penalty",
                section="reranker.query_aware_calibration",
            ),
            zero_focus_penalty=_required_float(
                query_aware_calibration,
                "zero_focus_penalty",
                section="reranker.query_aware_calibration",
            ),
            minimum_adjustment=_required_float(
                query_aware_calibration,
                "minimum_adjustment",
                section="reranker.query_aware_calibration",
            ),
            maximum_adjustment=_required_float(
                query_aware_calibration,
                "maximum_adjustment",
                section="reranker.query_aware_calibration",
            ),
        ),
    )
    extensions.validate()
    return extensions


def _required_mapping(
    data: Mapping[str, Any],
    key: str,
    *,
    parent: str | None = None,
) -> Mapping[str, Any]:
    value = data.get(key)
    section = f"{parent}.{key}" if parent else key

    if not isinstance(value, Mapping):
        raise ValueError(f"{section} must be a TOML table.")

    return value


def _required_bool(
    data: Mapping[str, Any],
    key: str,
    *,
    section: str,
) -> bool:
    value = _required_value(data, key, section=section)

    if not isinstance(value, bool):
        raise ValueError(f"{section}.{key} must be boolean.")

    return value


def _required_int(
    data: Mapping[str, Any],
    key: str,
    *,
    section: str,
) -> int:
    value = _required_value(data, key, section=section)

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{section}.{key} must be an integer.")

    return value


def _required_float(
    data: Mapping[str, Any],
    key: str,
    *,
    section: str,
) -> float:
    value = _required_value(data, key, section=section)

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{section}.{key} must be numeric.")

    result = float(value)
    _validate_finite_number(f"{section}.{key}", result)
    return result


def _optional_float(
    data: Mapping[str, Any],
    key: str,
) -> float:
    value = data.get(key)

    if value is None:
        return 0.0

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric.")

    result = float(value)
    _validate_finite_number(key, result)
    return result


def _required_str_tuple(
    data: Mapping[str, Any],
    key: str,
    *,
    section: str,
) -> tuple[str, ...]:
    value = _required_value(data, key, section=section)

    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{section}.{key} must be an array of strings.")

    values = tuple(str(item).strip() for item in value)

    if not values or any(not item for item in values):
        raise ValueError(
            f"{section}.{key} must contain non-empty strings."
        )

    return values


def _required_value(
    data: Mapping[str, Any],
    key: str,
    *,
    section: str,
) -> Any:
    if key not in data:
        raise ValueError(f"{section}.{key} is required.")

    return data[key]


def _validate_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean.")


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_finite_number(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")


def _validate_positive_number(name: str, value: float) -> None:
    _validate_finite_number(name, value)

    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")


def _validate_non_negative_number(name: str, value: float) -> None:
    _validate_finite_number(name, value)

    if value < 0:
        raise ValueError(f"{name} cannot be negative.")


def _validate_probability(name: str, value: float) -> None:
    _validate_bounded_number(
        name,
        value,
        minimum=0.0,
        maximum=1.0,
    )


def _validate_bounded_number(
    name: str,
    value: float,
    *,
    minimum: float,
    maximum: float,
) -> None:
    _validate_finite_number(name, value)

    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}."
        )


def _validate_non_empty_strings(
    name: str,
    values: frozenset[str],
) -> None:
    if not values:
        raise ValueError(f"{name} cannot be empty.")

    if any(not value.strip() for value in values):
        raise ValueError(f"{name} cannot contain blank values.")
