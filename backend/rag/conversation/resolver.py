from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from rag.conversation.config import ReferenceResolutionConfig
from rag.conversation.entities import EntityExtractor
from rag.conversation.models import (
    ConversationContext,
    EntityMention,
    ReferenceResolutionResult,
)


Payload = dict[str, Any]


_REFERENTIAL_MARKER_RE = re.compile(
    r"\b("
    r"it|its|itself|"
    r"this|that|these|those|"
    r"same|previous|above|earlier|former|latter"
    r")\b",
    re.IGNORECASE,
)

_POSSESSIVE_REFERENCE_RE = re.compile(
    r"\bits\b",
    re.IGNORECASE,
)

_REFLEXIVE_REFERENCE_RE = re.compile(
    r"\bitself\b",
    re.IGNORECASE,
)

_PRONOUN_REFERENCE_RE = re.compile(
    r"\bit\b",
    re.IGNORECASE,
)

_DEMONSTRATIVE_REFERENCE_RE = re.compile(
    r"\b(this|that|these|those)\b",
    re.IGNORECASE,
)

_RELATIVE_REFERENCE_RE = re.compile(
    r"\b(?:the\s+)?"
    r"(same|previous|above|earlier|former|latter)"
    r"\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _AntecedentCandidate:
    mention: EntityMention
    source: str
    confidence: float

    @property
    def entity_key(self) -> str:
        return _entity_key(self.mention.entity_text)


class ConversationReferenceResolver:
    """
    Domain-independent conversational reference resolver.

    The resolver operates only on:

    - referential language
    - extracted entity identity
    - turn recency
    - entity salience
    - extraction confidence

    Runtime behavior and tunable confidence values are supplied through
    ReferenceResolutionConfig and EntityExtractor.
    """

    def __init__(
        self,
        *,
        config: ReferenceResolutionConfig,
        entity_extractor: EntityExtractor,
    ) -> None:
        config.validate()

        self._config = config
        self._entity_extractor = entity_extractor

    @property
    def config(self) -> ReferenceResolutionConfig:
        return self._config

    def resolve(
        self,
        *,
        raw_query: str,
        context: ConversationContext,
    ) -> ReferenceResolutionResult:
        query = _normalize_text(raw_query)

        if not query:
            raise ValueError("raw_query cannot be empty.")

        context.validate()

        if not self._config.enabled:
            return _create_result(
                raw_query=query,
                resolved_query=query,
                needed_resolution=False,
                strategy="reference_resolution_disabled",
                confidence=1.0,
                metadata={
                    "resolution_enabled": False,
                },
            )

        if not _contains_referential_marker(query):
            return _create_result(
                raw_query=query,
                resolved_query=query,
                needed_resolution=False,
                strategy="no_referential_marker",
                confidence=1.0,
                metadata={
                    "resolution_enabled": True,
                },
            )

        query_entities = self._entity_extractor.extract(
            query,
            max_entities=self._config.max_recent_entities,
        )

        excluded_entity_keys = {
            _entity_key(entity.entity_text)
            for entity in query_entities
        }

        candidates = self._collect_antecedent_candidates(
            context=context,
            excluded_entity_keys=excluded_entity_keys,
        )

        antecedent = _select_best_antecedent(candidates)

        if antecedent is None:
            return _create_result(
                raw_query=query,
                resolved_query=query,
                needed_resolution=True,
                strategy="unresolved_no_antecedent",
                confidence=0.0,
                metadata={
                    "resolution_enabled": True,
                    "query_entities": [
                        _entity_payload(entity)
                        for entity in query_entities
                    ],
                    "recent_entity_count": len(
                        context.recent_entities
                    ),
                    "recent_turn_count": len(
                        context.recent_turns
                    ),
                    "candidate_count": 0,
                },
            )

        resolved_query = _substitute_first_reference(
            query=query,
            entity=antecedent.mention.entity_text,
        )

        if resolved_query == query:
            return _create_result(
                raw_query=query,
                resolved_query=query,
                needed_resolution=True,
                strategy="unresolved_unsupported_marker",
                confidence=0.0,
                metadata={
                    "resolution_enabled": True,
                    "candidate": _candidate_payload(
                        antecedent
                    ),
                },
            )

        resolution_confidence = (
            self._config
            .confidence_for_recent_entity_substitution
            * antecedent.confidence
        )

        return _create_result(
            raw_query=query,
            resolved_query=resolved_query,
            needed_resolution=True,
            resolved_entity=antecedent.mention.entity_text,
            strategy=f"{antecedent.source}_substitution",
            confidence=_clamp_probability(
                resolution_confidence
            ),
            metadata={
                "resolution_enabled": True,
                "candidate": _candidate_payload(
                    antecedent
                ),
                "query_entities": [
                    _entity_payload(entity)
                    for entity in query_entities
                ],
                "candidate_count": len(candidates),
            },
        )

    def _collect_antecedent_candidates(
        self,
        *,
        context: ConversationContext,
        excluded_entity_keys: set[str],
    ) -> tuple[_AntecedentCandidate, ...]:
        candidates: list[_AntecedentCandidate] = []
        seen_entity_keys: set[str] = set()

        for entity in context.recent_entities:
            if len(candidates) >= self._config.max_recent_entities:
                break

            candidate = self._candidate_from_persisted_entity(
                entity
            )

            _append_candidate(
                candidate=candidate,
                candidates=candidates,
                seen_entity_keys=seen_entity_keys,
                excluded_entity_keys=excluded_entity_keys,
            )

        if len(candidates) >= self._config.max_recent_entities:
            return tuple(candidates)

        recent_turns = context.recent_turns[
            -self._config.max_recent_turns_for_fallback:
        ]

        for turn in reversed(recent_turns):
            if turn.role not in {
                "user",
                "assistant",
            }:
                continue

            mentions = self._entity_extractor.extract(
                turn.text_for_context,
                max_entities=self._config.max_recent_entities,
            )

            for mention in mentions:
                candidate = self._candidate_from_turn_entity(
                    mention=mention,
                    turn_id=turn.turn_id,
                    turn_index=turn.turn_index,
                    turn_role=turn.role,
                )

                _append_candidate(
                    candidate=candidate,
                    candidates=candidates,
                    seen_entity_keys=seen_entity_keys,
                    excluded_entity_keys=excluded_entity_keys,
                )

                if (
                    len(candidates)
                    >= self._config.max_recent_entities
                ):
                    return tuple(candidates)

        return tuple(candidates)

    def _candidate_from_persisted_entity(
        self,
        entity: EntityMention,
    ) -> _AntecedentCandidate:
        normalized = entity.normalized()

        confidence = _mention_confidence(
            mention=normalized,
            entity_extractor=self._entity_extractor,
        )

        return _AntecedentCandidate(
            mention=EntityMention(
                entity_text=normalized.entity_text,
                entity_type=normalized.entity_type,
                salience_rank=normalized.salience_rank,
                turn_index=normalized.turn_index,
                metadata={
                    **normalized.metadata,
                    "resolution_source": "entity_index",
                },
            ),
            source="entity_index",
            confidence=confidence,
        )

    def _candidate_from_turn_entity(
        self,
        *,
        mention: EntityMention,
        turn_id: str,
        turn_index: int,
        turn_role: str,
    ) -> _AntecedentCandidate:
        normalized = mention.normalized()

        base_confidence = _mention_confidence(
            mention=normalized,
            entity_extractor=self._entity_extractor,
        )

        confidence = (
            base_confidence
            * self._config
            .recent_turn_fallback_confidence_multiplier
        )

        return _AntecedentCandidate(
            mention=EntityMention(
                entity_text=normalized.entity_text,
                entity_type=normalized.entity_type,
                salience_rank=normalized.salience_rank,
                turn_index=turn_index,
                metadata={
                    **normalized.metadata,
                    "resolution_source": (
                        "recent_turn_fallback"
                    ),
                    "source_turn_id": turn_id,
                    "source_turn_role": turn_role,
                },
            ),
            source="recent_turn_fallback",
            confidence=_clamp_probability(confidence),
        )


def _append_candidate(
    *,
    candidate: _AntecedentCandidate,
    candidates: list[_AntecedentCandidate],
    seen_entity_keys: set[str],
    excluded_entity_keys: set[str],
) -> None:
    entity_key = candidate.entity_key

    if not entity_key:
        return

    if entity_key in excluded_entity_keys:
        return

    if entity_key in seen_entity_keys:
        return

    candidate.mention.validate()

    seen_entity_keys.add(entity_key)
    candidates.append(candidate)


def _select_best_antecedent(
    candidates: Iterable[_AntecedentCandidate],
) -> _AntecedentCandidate | None:
    available = tuple(candidates)

    if not available:
        return None

    return min(
        available,
        key=lambda candidate: (
            -(candidate.mention.turn_index or 0),
            candidate.mention.salience_rank,
            -candidate.confidence,
            -len(candidate.mention.entity_text),
            candidate.mention.entity_text.casefold(),
        ),
    )


def _mention_confidence(
    *,
    mention: EntityMention,
    entity_extractor: EntityExtractor,
) -> float:
    metadata_confidence = mention.metadata.get(
        "confidence"
    )

    if (
        isinstance(metadata_confidence, (int, float))
        and not isinstance(metadata_confidence, bool)
    ):
        return _clamp_probability(
            float(metadata_confidence)
        )

    return (
        entity_extractor
        .config
        .confidence
        .for_entity_type(mention.entity_type)
    )


def _contains_referential_marker(
    query: str,
) -> bool:
    return bool(
        _REFERENTIAL_MARKER_RE.search(query)
    )


def _substitute_first_reference(
    *,
    query: str,
    entity: str,
) -> str:
    if _POSSESSIVE_REFERENCE_RE.search(query):
        return _normalize_text(
            _POSSESSIVE_REFERENCE_RE.sub(
                _possessive_form(entity),
                query,
                count=1,
            )
        )

    if _REFLEXIVE_REFERENCE_RE.search(query):
        return _normalize_text(
            _REFLEXIVE_REFERENCE_RE.sub(
                entity,
                query,
                count=1,
            )
        )

    if _PRONOUN_REFERENCE_RE.search(query):
        return _normalize_text(
            _PRONOUN_REFERENCE_RE.sub(
                entity,
                query,
                count=1,
            )
        )

    if _DEMONSTRATIVE_REFERENCE_RE.search(query):
        return _normalize_text(
            _DEMONSTRATIVE_REFERENCE_RE.sub(
                entity,
                query,
                count=1,
            )
        )

    if _RELATIVE_REFERENCE_RE.search(query):
        return _normalize_text(
            _RELATIVE_REFERENCE_RE.sub(
                entity,
                query,
                count=1,
            )
        )

    return query


def _possessive_form(
    entity: str,
) -> str:
    return f"{entity.rstrip()}'s"


def _candidate_payload(
    candidate: _AntecedentCandidate,
) -> Payload:
    return {
        "entity": _entity_payload(
            candidate.mention
        ),
        "source": candidate.source,
        "confidence": candidate.confidence,
    }


def _entity_payload(
    entity: EntityMention,
) -> Payload:
    return {
        "entity_text": entity.entity_text,
        "entity_type": entity.entity_type,
        "salience_rank": entity.salience_rank,
        "turn_index": entity.turn_index,
        "metadata": dict(entity.metadata),
    }


def _create_result(
    *,
    raw_query: str,
    resolved_query: str,
    needed_resolution: bool,
    strategy: str,
    confidence: float,
    resolved_entity: str | None = None,
    metadata: Payload | None = None,
) -> ReferenceResolutionResult:
    result = ReferenceResolutionResult(
        raw_query=raw_query,
        resolved_query=resolved_query,
        needed_resolution=needed_resolution,
        resolved_entity=resolved_entity,
        strategy=strategy,
        confidence=_clamp_probability(confidence),
        metadata=dict(metadata or {}),
    )

    result.validate()
    return result


def _entity_key(
    value: str,
) -> str:
    return _normalize_text(value).casefold()


def _normalize_text(
    value: str,
) -> str:
    return re.sub(
        r"\s+",
        " ",
        (value or "").strip(),
    )


def _clamp_probability(
    value: float,
) -> float:
    return max(
        0.0,
        min(1.0, value),
    )