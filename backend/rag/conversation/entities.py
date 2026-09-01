from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from rag.conversation.config import EntityExtractionConfig
from rag.conversation.models import EntityMention


_BACKTICK_RE = re.compile(
    r"`(?P<value>[^`\n]+)`"
)

_DOUBLE_QUOTED_RE = re.compile(
    r'"(?P<value>[^"\n]+)"'
)

_SINGLE_QUOTED_RE = re.compile(
    r"'(?P<value>[^'\n]+)'"
)

_QUALIFIED_IDENTIFIER_RE = re.compile(
    r"\b"
    r"[A-Za-z_][A-Za-z0-9_-]*"
    r"(?:(?:\.|::)[A-Za-z_][A-Za-z0-9_-]*)+"
    r"\b"
)

_PROPER_NOUN_TOKEN_PATTERN = r"[A-Z][A-Za-z0-9_-]*"


@dataclass(frozen=True)
class _EntityCandidate:
    text: str
    entity_type: str
    start: int
    end: int
    confidence: float

    @property
    def span_length(self) -> int:
        return self.end - self.start


class EntityExtractor:
    """
    Domain-independent structural entity extractor.

    Runtime behavior, thresholds, limits, confidence values, and filtering
    vocabularies are supplied through EntityExtractionConfig.

    This class must be created once at application startup and injected into
    conversation-aware components.
    """

    def __init__(
        self,
        *,
        config: EntityExtractionConfig,
    ) -> None:
        config.validate()

        self._config = config

        self._leading_function_words = frozenset(
            value.casefold()
            for value in config.leading_function_words
        )

        self._standalone_non_entities = frozenset(
            value.casefold()
            for value in config.standalone_non_entities
        )

        self._proper_noun_pattern = self._compile_proper_noun_pattern()
        self._acronym_pattern = self._compile_acronym_pattern()

    @property
    def config(self) -> EntityExtractionConfig:
        return self._config

    def extract(
        self,
        text: str,
        *,
        max_entities: int | None = None,
    ) -> tuple[EntityMention, ...]:
        if not self._config.enabled:
            return ()

        normalized_text = _normalize_text(text)
        if not normalized_text:
            return ()

        effective_limit = self._resolve_entity_limit(max_entities)

        candidates = sorted(
            self._iter_candidates(normalized_text),
            key=lambda candidate: (
                -candidate.confidence,
                -candidate.span_length,
                candidate.start,
                candidate.text.casefold(),
            ),
        )

        selected: list[_EntityCandidate] = []
        seen_keys: set[str] = set()

        for candidate in candidates:
            normalized_candidate = self._normalize_candidate(
                candidate=candidate,
                source_text=normalized_text,
            )

            if normalized_candidate is None:
                continue

            entity_key = _entity_key(normalized_candidate.text)
            if not entity_key or entity_key in seen_keys:
                continue

            if _overlaps_selected(
                candidate=normalized_candidate,
                selected=selected,
            ):
                continue

            selected.append(normalized_candidate)
            seen_keys.add(entity_key)

            if len(selected) >= effective_limit:
                break

        selected.sort(
            key=lambda candidate: (
                candidate.start,
                -candidate.confidence,
                -candidate.span_length,
                candidate.text.casefold(),
            )
        )

        return tuple(
            EntityMention(
                entity_text=candidate.text,
                entity_type=candidate.entity_type,
                salience_rank=salience_rank,
                metadata={
                    "extractor": "structural",
                    "confidence": candidate.confidence,
                    "span_start": candidate.start,
                    "span_end": candidate.end,
                },
            )
            for salience_rank, candidate in enumerate(
                selected,
                start=1,
            )
        )

    def _iter_candidates(
        self,
        text: str,
    ) -> Iterator[_EntityCandidate]:
        if self._config.extract_backtick_phrases:
            yield from self._iter_pattern_matches(
                pattern=_BACKTICK_RE,
                text=text,
                entity_type="code_or_exact",
                named_group="value",
            )

        if self._config.extract_quoted_phrases:
            yield from self._iter_pattern_matches(
                pattern=_DOUBLE_QUOTED_RE,
                text=text,
                entity_type="quoted_phrase",
                named_group="value",
            )

            yield from self._iter_pattern_matches(
                pattern=_SINGLE_QUOTED_RE,
                text=text,
                entity_type="quoted_phrase",
                named_group="value",
            )

        if self._config.extract_qualified_identifiers:
            yield from self._iter_pattern_matches(
                pattern=_QUALIFIED_IDENTIFIER_RE,
                text=text,
                entity_type="identifier",
            )

        if self._config.extract_acronyms:
            yield from self._iter_pattern_matches(
                pattern=self._acronym_pattern,
                text=text,
                entity_type="acronym",
            )

        if self._config.extract_proper_nouns:
            yield from self._iter_pattern_matches(
                pattern=self._proper_noun_pattern,
                text=text,
                entity_type="proper_noun",
            )

    def _iter_pattern_matches(
        self,
        *,
        pattern: re.Pattern[str],
        text: str,
        entity_type: str,
        named_group: str | None = None,
    ) -> Iterator[_EntityCandidate]:
        confidence = self._config.confidence.for_entity_type(
            entity_type
        )

        for match in pattern.finditer(text):
            if named_group is None:
                candidate_text = match.group(0)
                start = match.start()
                end = match.end()
            else:
                candidate_text = match.group(named_group)
                start = match.start(named_group)
                end = match.end(named_group)

            yield _EntityCandidate(
                text=candidate_text,
                entity_type=entity_type,
                start=start,
                end=end,
                confidence=confidence,
            )

    def _normalize_candidate(
        self,
        *,
        candidate: _EntityCandidate,
        source_text: str,
    ) -> _EntityCandidate | None:
        normalized_text = _normalize_text(candidate.text)
        normalized_text = _strip_boundary_punctuation(
            normalized_text
        )

        if (
            candidate.entity_type == "proper_noun"
            and self._config.strip_leading_function_words
        ):
            normalized_text = self._strip_leading_function_words(
                normalized_text
            )

        if not normalized_text:
            return None

        if (
            len(normalized_text)
            < self._config.minimum_entity_length
        ):
            return None

        if (
            len(normalized_text)
            > self._config.max_entity_length
        ):
            return None

        if (
            normalized_text.casefold()
            in self._standalone_non_entities
        ):
            return None

        if not any(
            character.isalnum()
            for character in normalized_text
        ):
            return None

        if (
            candidate.entity_type == "acronym"
            and len(normalized_text)
            > self._config.max_acronym_length
        ):
            return None

        if candidate.entity_type == "proper_noun":
            if not self._is_valid_proper_noun(
                text=normalized_text,
                source_text=source_text,
                source_start=candidate.start,
            ):
                return None

        return _EntityCandidate(
            text=normalized_text,
            entity_type=candidate.entity_type,
            start=candidate.start,
            end=candidate.end,
            confidence=candidate.confidence,
        )

    def _strip_leading_function_words(
        self,
        text: str,
    ) -> str:
        tokens = text.split()

        while (
            len(tokens) > 1
            and tokens[0].casefold()
            in self._leading_function_words
        ):
            tokens.pop(0)

        return " ".join(tokens)

    def _is_valid_proper_noun(
        self,
        *,
        text: str,
        source_text: str,
        source_start: int,
    ) -> bool:
        tokens = text.split()

        if not tokens:
            return False

        if (
            len(tokens)
            > self._config.max_proper_noun_tokens
        ):
            return False

        if (
            len(tokens) == 1
            and self._config.reject_sentence_initial_single_token
            and _is_sentence_boundary(
                text=source_text,
                start=source_start,
            )
            and not _has_strong_orthographic_signal(text)
        ):
            return False

        return True

    def _resolve_entity_limit(
        self,
        requested_limit: int | None,
    ) -> int:
        if requested_limit is None:
            return self._config.max_entities

        if requested_limit <= 0:
            raise ValueError(
                "max_entities must be greater than zero."
            )

        return min(
            requested_limit,
            self._config.max_entities,
        )

    def _compile_proper_noun_pattern(
        self,
    ) -> re.Pattern[str]:
        additional_token_limit = (
            self._config.max_proper_noun_tokens - 1
        )

        return re.compile(
            rf"\b"
            rf"{_PROPER_NOUN_TOKEN_PATTERN}"
            rf"(?:\s+{_PROPER_NOUN_TOKEN_PATTERN})"
            rf"{{0,{additional_token_limit}}}"
            rf"\b"
        )

    def _compile_acronym_pattern(
        self,
    ) -> re.Pattern[str]:
        remaining_character_limit = (
            self._config.max_acronym_length - 1
        )

        return re.compile(
            rf"\b"
            rf"[A-Z]"
            rf"[A-Z0-9]"
            rf"{{0,{remaining_character_limit}}}"
            rf"\b"
        )


def _overlaps_selected(
    *,
    candidate: _EntityCandidate,
    selected: list[_EntityCandidate],
) -> bool:
    return any(
        candidate.start < existing.end
        and existing.start < candidate.end
        for existing in selected
    )


def _strip_boundary_punctuation(
    text: str,
) -> str:
    return text.strip(
        " \t\r\n,;:!?()[]{}"
    )


def _is_sentence_boundary(
    *,
    text: str,
    start: int,
) -> bool:
    if start <= 0:
        return True

    prefix = text[:start].rstrip()
    if not prefix:
        return True

    return prefix[-1] in {
        ".",
        "!",
        "?",
        "\n",
    }


def _has_strong_orthographic_signal(
    value: str,
) -> bool:
    if any(
        character.isdigit()
        for character in value
    ):
        return True

    if "_" in value or "-" in value:
        return True

    return any(
        character.isupper()
        for character in value[1:]
    )


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