from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from rag.retrieval.adaptive.state import QuerySignals
from rag.retrieval.contracts.primitives import (
    Payload,
    validate_bool,
    validate_finite_number,
    validate_mapping_size,
    validate_non_empty_str,
    validate_non_negative_int,
    validate_optional_sequence_limit,
    validate_positive_int,
)
from rag.retrieval.contracts.query import NormalizedQuery
from rag.retrieval.contracts.request import RetrievalRequest


_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:@#-]+")
_QUOTED_TERM_RE = re.compile(r"`([^`]{1,200})`|\"([^\"]{1,200})\"|'([^']{1,200})'")

_FILE_HINT_RE = re.compile(
    r"""
    (?<!\w)
    [\w./@:-]+
    \.
    (?:
        py|pyi|ts|tsx|js|jsx|md|mdx|json|toml|yaml|yml|sql|html|css|
        go|rs|java|kt|cpp|cc|cxx|c|h|hpp|proto|sh|dockerfile|env
    )
    (?!\w)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_FUNCTION_CALL_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\([^)]{0,120}\)")
_DOTTED_IDENTIFIER_RE = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]{1,}(?:\.[A-Za-z_][A-Za-z0-9_]{1,})+\b"
)
_SNAKE_OR_KEBAB_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[_-][A-Za-z0-9]+)+\b")
_CONSTANT_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
_CAMEL_RE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z0-9]+)+\b")

_REFERENTIAL_RE = re.compile(
    r"\b(this|that|these|those|it|same|above|previous|earlier|last|again|continue|there|here|mentioned)\b",
    re.IGNORECASE,
)

_MEMORY_RE = re.compile(
    r"\b(memory|remember|recall|conversation|chat history|we discussed|i said|you said|last time|from earlier)\b",
    re.IGNORECASE,
)

_TEMPORAL_RE = re.compile(
    r"""
    \b(
        latest|current|recent|recently|today|yesterday|tomorrow|now|
        new|old|updated|fresh|deprecated|changed|changelog
    )\b
    |
    \b\d{4}-\d{1,2}-\d{1,2}\b
    |
    \b\d{1,2}/\d{1,2}/\d{2,4}\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_COMPARISON_RE = re.compile(
    r"\b(compare|comparison|difference|different|vs|versus|better|worse|tradeoff|trade-off|choose|which one|between)\b",
    re.IGNORECASE,
)

_DEBUG_RE = re.compile(
    r"\b(error|exception|traceback|stack trace|failed|failure|bug|crash|timeout|refused|denied|wrong|broken|not working|fix)\b",
    re.IGNORECASE,
)

_IMPLEMENTATION_RE = re.compile(
    r"\b(implement|implementation|code|build|wire|integrate|add|modify|change|patch|refactor|create|write)\b",
    re.IGNORECASE,
)

_ARCHITECTURE_RE = re.compile(
    r"\b(architecture|design|pipeline|layer|service|engine|workflow|graph|orchestration|state|memory|agentic|controller|policy|router)\b",
    re.IGNORECASE,
)


class CorpusAffinityProvider(Protocol):
    def score(self, tokens: Sequence[str]) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class QuerySignalExtractorConfig:
    max_metadata_keys: int = 64
    max_metadata_terms: int = 64
    min_long_token_chars: int = 12
    include_term_metadata: bool = True
    lowercase_corpus_tokens: bool = True

    def validate(self) -> None:
        validate_positive_int(
            "query_signal_extractor.max_metadata_keys",
            self.max_metadata_keys,
        )
        validate_positive_int(
            "query_signal_extractor.max_metadata_terms",
            self.max_metadata_terms,
        )
        validate_positive_int(
            "query_signal_extractor.min_long_token_chars",
            self.min_long_token_chars,
        )
        validate_bool(
            "query_signal_extractor.include_term_metadata",
            self.include_term_metadata,
        )
        validate_bool(
            "query_signal_extractor.lowercase_corpus_tokens",
            self.lowercase_corpus_tokens,
        )


@dataclass(frozen=True)
class QuerySignalTerms:
    tokens: tuple[str, ...] = field(default_factory=tuple)
    quoted_terms: tuple[str, ...] = field(default_factory=tuple)
    file_hints: tuple[str, ...] = field(default_factory=tuple)
    code_like_terms: tuple[str, ...] = field(default_factory=tuple)
    referential_markers: tuple[str, ...] = field(default_factory=tuple)
    memory_markers: tuple[str, ...] = field(default_factory=tuple)
    temporal_markers: tuple[str, ...] = field(default_factory=tuple)
    comparison_markers: tuple[str, ...] = field(default_factory=tuple)
    debug_markers: tuple[str, ...] = field(default_factory=tuple)
    implementation_markers: tuple[str, ...] = field(default_factory=tuple)
    architecture_markers: tuple[str, ...] = field(default_factory=tuple)

    @property
    def exact_terms(self) -> tuple[str, ...]:
        return clean_unique_strings(
            (
                *self.quoted_terms,
                *self.file_hints,
                *self.code_like_terms,
            )
        )

    def validate(self, *, max_terms: int | None = None) -> None:
        for field_name, values in (
            ("tokens", self.tokens),
            ("quoted_terms", self.quoted_terms),
            ("file_hints", self.file_hints),
            ("code_like_terms", self.code_like_terms),
            ("referential_markers", self.referential_markers),
            ("memory_markers", self.memory_markers),
            ("temporal_markers", self.temporal_markers),
            ("comparison_markers", self.comparison_markers),
            ("debug_markers", self.debug_markers),
            ("implementation_markers", self.implementation_markers),
            ("architecture_markers", self.architecture_markers),
        ):
            validate_optional_sequence_limit(
                f"query_signal_terms.{field_name}",
                values,
                max_terms,
            )
            for index, value in enumerate(values):
                validate_non_empty_str(
                    f"query_signal_terms.{field_name}[{index}]",
                    value,
                )


@dataclass(frozen=True)
class SetCorpusAffinityProvider:
    vocabulary: frozenset[str]
    lowercase: bool = True

    @classmethod
    def from_terms(
        cls,
        terms: Sequence[str],
        *,
        lowercase: bool = True,
    ) -> SetCorpusAffinityProvider:
        vocabulary = frozenset(
            normalized
            for term in terms
            if (normalized := _normalize_vocab_term(term, lowercase=lowercase))
        )
        return cls(vocabulary=vocabulary, lowercase=lowercase)

    def score(self, tokens: Sequence[str]) -> float:
        normalized_tokens = tuple(
            normalized
            for token in tokens
            if (normalized := _normalize_vocab_term(token, lowercase=self.lowercase))
        )

        if not normalized_tokens:
            return 0.0

        hits = sum(1 for token in normalized_tokens if token in self.vocabulary)
        return hits / len(normalized_tokens)


@dataclass(frozen=True)
class DefaultQuerySignalExtractor:
    config: QuerySignalExtractorConfig = QuerySignalExtractorConfig()
    corpus_affinity_provider: CorpusAffinityProvider | None = None

    def extract(
        self,
        *,
        request: RetrievalRequest,
        normalized_query: NormalizedQuery,
    ) -> QuerySignals:
        request.validate()
        normalized_query.validate()
        self.config.validate()

        text = " ".join(normalized_query.normalized_query.strip().split())
        terms = extract_query_signal_terms(
            text,
            max_terms=self.config.max_metadata_terms,
        )
        terms.validate(max_terms=self.config.max_metadata_terms)

        lexical_specificity = estimate_lexical_specificity(
            terms=terms,
            min_long_token_chars=self.config.min_long_token_chars,
        )

        corpus_affinity = None
        if self.corpus_affinity_provider is not None:
            corpus_affinity = self.corpus_affinity_provider.score(terms.tokens)
            _validate_probability("query_signals.corpus_affinity", corpus_affinity)

        metadata: Payload = {
            "source": "default_query_signal_extractor",
        }

        if self.config.include_term_metadata:
            metadata.update(
                {
                    "exact_terms": list(terms.exact_terms),
                    "file_hints": list(terms.file_hints),
                    "code_like_terms": list(terms.code_like_terms),
                    "quoted_terms": list(terms.quoted_terms),
                }
            )

        signals = QuerySignals(
            normalized_query=text,
            token_count=len(terms.tokens),
            char_count=len(text),
            exact_token_count=len(terms.exact_terms),
            quoted_term_count=len(terms.quoted_terms),
            code_like_token_count=len(terms.code_like_terms),
            file_hint_count=len(terms.file_hints),
            referential_marker_count=len(terms.referential_markers),
            memory_marker_count=len(terms.memory_markers),
            temporal_marker_count=len(terms.temporal_markers),
            comparison_marker_count=len(terms.comparison_markers),
            debug_marker_count=len(terms.debug_markers),
            implementation_marker_count=len(terms.implementation_markers),
            architecture_marker_count=len(terms.architecture_markers),
            has_source_constraints=not request.source_constraints.is_global,
            has_chat_history=bool(request.chat_history),
            has_session_context=bool(request.tenant.session_id),
            wants_memory=request.wants_memory,
            lexical_specificity=lexical_specificity,
            corpus_affinity=corpus_affinity,
            metadata=metadata,
        )
        signals.validate(max_metadata_keys=self.config.max_metadata_keys)
        return signals


def extract_query_signal_terms(
    query: str,
    *,
    max_terms: int | None = None,
) -> QuerySignalTerms:
    cleaned_query = " ".join(query.strip().split())
    validate_non_empty_str("query", cleaned_query)

    tokens = clean_unique_strings(_TOKEN_RE.findall(cleaned_query), limit=max_terms)

    quoted_terms = clean_unique_strings(
        (
            group
            for match in _QUOTED_TERM_RE.finditer(cleaned_query)
            for group in match.groups()
            if group
        ),
        limit=max_terms,
    )

    file_hints = clean_unique_strings(
        (match.group(0) for match in _FILE_HINT_RE.finditer(cleaned_query)),
        limit=max_terms,
    )

    code_like_terms = clean_unique_strings(
        (
            *_FUNCTION_CALL_RE.findall(cleaned_query),
            *_DOTTED_IDENTIFIER_RE.findall(cleaned_query),
            *_SNAKE_OR_KEBAB_RE.findall(cleaned_query),
            *_CONSTANT_RE.findall(cleaned_query),
            *_CAMEL_RE.findall(cleaned_query),
            *file_hints,
        ),
        limit=max_terms,
    )

    return QuerySignalTerms(
        tokens=tokens,
        quoted_terms=quoted_terms,
        file_hints=file_hints,
        code_like_terms=code_like_terms,
        referential_markers=_marker_values(_REFERENTIAL_RE, cleaned_query, max_terms=max_terms),
        memory_markers=_marker_values(_MEMORY_RE, cleaned_query, max_terms=max_terms),
        temporal_markers=_marker_values(_TEMPORAL_RE, cleaned_query, max_terms=max_terms),
        comparison_markers=_marker_values(_COMPARISON_RE, cleaned_query, max_terms=max_terms),
        debug_markers=_marker_values(_DEBUG_RE, cleaned_query, max_terms=max_terms),
        implementation_markers=_marker_values(
            _IMPLEMENTATION_RE,
            cleaned_query,
            max_terms=max_terms,
        ),
        architecture_markers=_marker_values(
            _ARCHITECTURE_RE,
            cleaned_query,
            max_terms=max_terms,
        ),
    )


def estimate_lexical_specificity(
    *,
    terms: QuerySignalTerms,
    min_long_token_chars: int = 12,
) -> float:
    validate_positive_int("min_long_token_chars", min_long_token_chars)

    token_count = max(1, len(terms.tokens))
    exact_density = min(1.0, len(terms.exact_terms) / token_count)

    rare_shape_tokens = tuple(
        token
        for token in terms.tokens
        if _looks_rare_or_identifier(token, min_long_token_chars=min_long_token_chars)
    )
    rare_shape_density = min(1.0, len(rare_shape_tokens) / token_count)

    quoted_boost = 0.14 if terms.quoted_terms else 0.0
    file_boost = 0.18 if terms.file_hints else 0.0
    code_boost = 0.16 if terms.code_like_terms else 0.0

    low_information_penalty = 0.12 if token_count <= 2 else 0.0

    score = (
        0.08
        + 0.38 * exact_density
        + 0.34 * rare_shape_density
        + quoted_boost
        + file_boost
        + code_boost
        - low_information_penalty
    )

    return _clamp_probability(score)


def clean_unique_strings(
    values: Sequence[Any],
    *,
    limit: int | None = None,
) -> tuple[str, ...]:
    if limit is not None:
        validate_positive_int("clean_unique_strings.limit", limit)

    cleaned: list[str] = []
    seen: set[str] = set()

    for value in values:
        item = " ".join(str(value).strip().split())
        if not item:
            continue

        if item in seen:
            continue

        seen.add(item)
        cleaned.append(item)

        if limit is not None and len(cleaned) >= limit:
            break

    return tuple(cleaned)


def _marker_values(
    pattern: re.Pattern[str],
    text: str,
    *,
    max_terms: int | None,
) -> tuple[str, ...]:
    values: list[str] = []

    for match in pattern.finditer(text):
        value = next((group for group in match.groups() if group), match.group(0))
        values.append(value)

    return clean_unique_strings(values, limit=max_terms)


def _looks_rare_or_identifier(
    token: str,
    *,
    min_long_token_chars: int,
) -> bool:
    if len(token) >= min_long_token_chars:
        return True

    if any(char.isdigit() for char in token):
        return True

    if any(char in token for char in ("_", "-", ".", "/", ":", "@", "#")):
        return True

    if _CONSTANT_RE.fullmatch(token):
        return True

    if _CAMEL_RE.fullmatch(token):
        return True

    return False


def _normalize_vocab_term(term: Any, *, lowercase: bool) -> str:
    cleaned = str(term).strip()
    if lowercase:
        cleaned = cleaned.lower()
    return cleaned


def _clamp_probability(value: float) -> float:
    validate_finite_number("probability", value)
    return max(0.0, min(1.0, float(value)))


def _validate_probability(name: str, value: float) -> None:
    validate_finite_number(name, value)

    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")


def validate_query_signal_extraction_sample(
    signals: QuerySignals,
    *,
    min_tokens: int = 0,
) -> None:
    validate_non_negative_int("min_tokens", min_tokens)
    signals.validate()

    if signals.token_count < min_tokens:
        raise ValueError(
            "query signal extraction produced fewer tokens than expected. "
            f"Expected at least {min_tokens}, got {signals.token_count}."
        )
