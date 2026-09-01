from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from rag.retrieval.contracts.enums import QueryDefect, QueryIntent


Payload = dict[str, Any]


_FUNCTION_WORDS = {
    "a",
    "an",
    "and",
    "about",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_./:-]*|[0-9]+")
_QUOTED_RE = re.compile(r"`([^`]{1,200})`|\"([^\"]{1,200})\"|'([^']{1,200})'")
_PATH_LIKE_RE = re.compile(
    r"(?:^|\s)([A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,16})(?:\s|$)"
)
_CODE_LIKE_RE = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]{1,}(?:\.[A-Za-z_][A-Za-z0-9_]{1,})+\b"
    r"|\b[A-Za-z_][A-Za-z0-9_]{1,}\\([^)]{0,120}\\)\b"
)
_UPPER_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,}\b")
_REFERENTIAL_RE = re.compile(
    r"\b(it|its|this|that|these|those|same|previous|above|earlier|last)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryTerm:
    text: str
    weight: float
    features: tuple[str, ...] = field(default_factory=tuple)

    def to_payload(self) -> Payload:
        return {
            "text": self.text,
            "weight": self.weight,
            "features": list(self.features),
        }


@dataclass(frozen=True)
class QueryAnalysis:
    raw_query: str
    normalized_query: str

    # Keep enum fields for existing contracts, but do not infer intent from
    # hardcoded action dictionaries. Intent remains conservative unless a
    # purely structural signal is strong enough.
    intent: QueryIntent = QueryIntent.UNKNOWN
    defects: tuple[QueryDefect, ...] = field(default_factory=tuple)

    query_terms: tuple[str, ...] = field(default_factory=tuple)
    weighted_terms: tuple[QueryTerm, ...] = field(default_factory=tuple)

    anchor_terms: tuple[str, ...] = field(default_factory=tuple)
    focus_terms: tuple[str, ...] = field(default_factory=tuple)
    entity_terms: tuple[str, ...] = field(default_factory=tuple)
    exact_terms: tuple[str, ...] = field(default_factory=tuple)
    source_hints: tuple[str, ...] = field(default_factory=tuple)
    phrase_terms: tuple[str, ...] = field(default_factory=tuple)
    semantic_queries: tuple[str, ...] = field(default_factory=tuple)

    referential_marker_count: int = 0
    token_count: int = 0
    unique_token_ratio: float = 0.0
    avg_token_length: float = 0.0
    lexical_specificity: float = 0.0
    entity_density: float = 0.0
    code_signal_density: float = 0.0
    exact_signal_density: float = 0.0

    structural_shape: str = "statement"
    confidence: float = 0.5
    requires_memory: bool = False

    reasoning: tuple[str, ...] = field(default_factory=tuple)
    metadata: Payload = field(default_factory=dict)

    def to_payload(self) -> Payload:
        return {
            "raw_query": self.raw_query,
            "normalized_query": self.normalized_query,
            "intent": self.intent.value,
            "defects": [item.value for item in self.defects],
            "query_terms": list(self.query_terms),
            "weighted_terms": [item.to_payload() for item in self.weighted_terms],
            "anchor_terms": list(self.anchor_terms),
            "focus_terms": list(self.focus_terms),
            "entity_terms": list(self.entity_terms),
            "exact_terms": list(self.exact_terms),
            "source_hints": list(self.source_hints),
            "phrase_terms": list(self.phrase_terms),
            "semantic_queries": list(self.semantic_queries),
            "referential_marker_count": self.referential_marker_count,
            "token_count": self.token_count,
            "unique_token_ratio": self.unique_token_ratio,
            "avg_token_length": self.avg_token_length,
            "lexical_specificity": self.lexical_specificity,
            "entity_density": self.entity_density,
            "code_signal_density": self.code_signal_density,
            "exact_signal_density": self.exact_signal_density,
            "structural_shape": self.structural_shape,
            "confidence": self.confidence,
            "requires_memory": self.requires_memory,
            "reasoning": list(self.reasoning),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> QueryAnalysis:
        intent_value = str(payload.get("intent") or QueryIntent.UNKNOWN.value)
        intent = (
            QueryIntent(intent_value)
            if intent_value in {item.value for item in QueryIntent}
            else QueryIntent.UNKNOWN
        )

        defect_values = {item.value for item in QueryDefect}
        defects = tuple(
            QueryDefect(str(value))
            for value in payload.get("defects", ())
            if str(value) in defect_values
        )

        weighted_terms = tuple(
            QueryTerm(
                text=str(item.get("text") or ""),
                weight=float(item.get("weight") or 0.0),
                features=_as_tuple(item.get("features")),
            )
            for item in payload.get("weighted_terms", ())
            if isinstance(item, Mapping) and str(item.get("text") or "").strip()
        )

        return cls(
            raw_query=str(payload.get("raw_query") or ""),
            normalized_query=str(payload.get("normalized_query") or ""),
            intent=intent,
            defects=defects,
            query_terms=_as_tuple(payload.get("query_terms")),
            weighted_terms=weighted_terms,
            anchor_terms=_as_tuple(payload.get("anchor_terms")),
            focus_terms=_as_tuple(payload.get("focus_terms")),
            entity_terms=_as_tuple(payload.get("entity_terms")),
            exact_terms=_as_tuple(payload.get("exact_terms")),
            source_hints=_as_tuple(payload.get("source_hints")),
            phrase_terms=_as_tuple(payload.get("phrase_terms")),
            semantic_queries=_as_tuple(payload.get("semantic_queries")),
            referential_marker_count=int(payload.get("referential_marker_count") or 0),
            token_count=int(payload.get("token_count") or 0),
            unique_token_ratio=float(payload.get("unique_token_ratio") or 0.0),
            avg_token_length=float(payload.get("avg_token_length") or 0.0),
            lexical_specificity=float(payload.get("lexical_specificity") or 0.0),
            entity_density=float(payload.get("entity_density") or 0.0),
            code_signal_density=float(payload.get("code_signal_density") or 0.0),
            exact_signal_density=float(payload.get("exact_signal_density") or 0.0),
            structural_shape=str(payload.get("structural_shape") or "statement"),
            confidence=float(payload.get("confidence") or 0.5),
            requires_memory=bool(payload.get("requires_memory")),
            reasoning=_as_tuple(payload.get("reasoning")),
            metadata=dict(payload.get("metadata") or {}),
        )


def analyze_query(
    query: str,
    *,
    wants_memory: bool = False,
    has_chat_history: bool = False,
    has_conversation_id: bool = False,
) -> QueryAnalysis:
    normalized_query = normalize_query_text(query)
    raw_tokens = tuple(_TOKEN_RE.findall(normalized_query))
    normalized_tokens = clean_unique_strings(_normalize_token(token) for token in raw_tokens)

    quoted_terms = _quoted_terms(normalized_query)
    path_like_terms = _path_like_terms(normalized_query)
    code_like_terms = _code_like_terms(normalized_query)
    acronym_terms = _acronym_terms(normalized_query)
    capitalized_terms = _capitalized_terms(raw_tokens)

    exact_terms = clean_unique_strings(
        (*quoted_terms, *path_like_terms, *code_like_terms)
    )
    source_hints = path_like_terms

    entity_terms = clean_unique_strings((*acronym_terms, *capitalized_terms))
    weighted_terms = _weighted_terms(
        raw_tokens=raw_tokens,
        normalized_tokens=normalized_tokens,
        exact_terms=exact_terms,
        entity_terms=entity_terms,
    )

    query_terms = clean_unique_strings(item.text for item in weighted_terms)

    primary_anchor_terms = clean_unique_strings(
        item.text
        for item in weighted_terms
        if _has_any_feature(item, {"entity", "exact", "code", "path", "rare_shape"})
        and not _has_any_feature(item, {"function"})
    )

    secondary_anchor_terms = clean_unique_strings(
        item.text
        for item in weighted_terms
        if _has_any_feature(item, {"long"})
        and not _has_any_feature(item, {"function"})
    )

    anchor_terms = clean_unique_strings((*primary_anchor_terms, *secondary_anchor_terms))

    # Focus terms are the strongest anchors. If strong entity/exact/code/path
    # anchors exist, do not let generic long words compete with them.
    if primary_anchor_terms:
        focus_terms = primary_anchor_terms
    else:
        focus_terms = clean_unique_strings(
            item.text
            for item in sorted(weighted_terms, key=lambda term: (-term.weight, term.text))
            if item.weight >= _dynamic_focus_threshold(weighted_terms)
            and not _has_any_feature(item, {"function"})
        )

    phrase_terms = _phrase_terms(raw_tokens=raw_tokens, focus_terms=focus_terms)
    semantic_queries = clean_unique_strings(
        (
            normalized_query,
            " ".join(focus_terms),
            " ".join(anchor_terms),
        )
    )

    referential_marker_count = len(_REFERENTIAL_RE.findall(normalized_query))
    token_count = len(raw_tokens)
    unique_token_ratio = _ratio(len(set(token.lower() for token in raw_tokens)), max(1, token_count))
    avg_token_length = _avg(len(token) for token in raw_tokens)
    lexical_specificity = _lexical_specificity(weighted_terms=weighted_terms, token_count=token_count)
    entity_density = _ratio(len(entity_terms), max(1, token_count))
    code_signal_density = _ratio(len(code_like_terms) + len(path_like_terms), max(1, token_count))
    exact_signal_density = _ratio(len(exact_terms), max(1, token_count))

    structural_shape = _structural_shape(
        query=normalized_query,
        referential_marker_count=referential_marker_count,
    )
    defects = _defects(
        query=normalized_query,
        token_count=token_count,
        referential_marker_count=referential_marker_count,
        focus_terms=focus_terms,
    )
    intent = _conservative_intent(
        structural_shape=structural_shape,
        exact_terms=exact_terms,
        source_hints=source_hints,
    )
    requires_memory = bool(
        wants_memory
        and referential_marker_count > 0
    )

    confidence = _confidence(
        focus_terms=focus_terms,
        exact_terms=exact_terms,
        defects=defects,
        lexical_specificity=lexical_specificity,
        entity_density=entity_density,
        code_signal_density=code_signal_density,
    )

    reasoning = (
        f"structural_shape={structural_shape}",
        f"focus_terms={len(focus_terms)}",
        f"exact_terms={len(exact_terms)}",
        f"entity_density={entity_density}",
        f"lexical_specificity={lexical_specificity}",
        f"confidence={confidence}",
    )

    return QueryAnalysis(
        raw_query=query,
        normalized_query=normalized_query,
        intent=intent,
        defects=defects,
        query_terms=query_terms,
        weighted_terms=weighted_terms,
        anchor_terms=anchor_terms,
        focus_terms=focus_terms,
        entity_terms=entity_terms,
        exact_terms=exact_terms,
        source_hints=source_hints,
        phrase_terms=phrase_terms,
        semantic_queries=semantic_queries,
        referential_marker_count=referential_marker_count,
        token_count=token_count,
        unique_token_ratio=unique_token_ratio,
        avg_token_length=avg_token_length,
        lexical_specificity=lexical_specificity,
        entity_density=entity_density,
        code_signal_density=code_signal_density,
        exact_signal_density=exact_signal_density,
        structural_shape=structural_shape,
        confidence=confidence,
        requires_memory=requires_memory,
        reasoning=reasoning,
        metadata={
            "source": "generic_statistical_query_analysis",
            "version": "v1",
            "uses_domain_terms": False,
            "uses_action_dictionary": False,
        },
    )


def query_analysis_from_metadata(metadata: Mapping[str, Any]) -> QueryAnalysis | None:
    raw = metadata.get("query_analysis")
    if not isinstance(raw, Mapping):
        return None
    return QueryAnalysis.from_payload(raw)


def normalize_query_text(query: str) -> str:
    cleaned = " ".join(str(query).strip().split())
    if not cleaned:
        raise ValueError("query cannot be empty after normalization.")
    return cleaned


def clean_unique_strings(values: Sequence[Any]) -> tuple[str, ...]:
    cleaned: list[str] = []
    seen: set[str] = set()

    for value in values:
        item = str(value).strip()
        if not item:
            continue

        key = item.lower()
        if key in seen:
            continue

        seen.add(key)
        cleaned.append(item)

    return tuple(cleaned)


def _normalize_token(token: str) -> str:
    cleaned = str(token).strip().lower()
    cleaned = cleaned.strip(".,;:!?()[]{}<>")
    return cleaned


def _quoted_terms(query: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in _QUOTED_RE.finditer(query):
        for group in match.groups():
            if group:
                values.append(group.strip())
                break
    return clean_unique_strings(values)


def _path_like_terms(query: str) -> tuple[str, ...]:
    return clean_unique_strings(
        match.group(1).strip()
        for match in _PATH_LIKE_RE.finditer(f" {query} ")
    )


def _code_like_terms(query: str) -> tuple[str, ...]:
    return clean_unique_strings(match.group(0).strip() for match in _CODE_LIKE_RE.finditer(query))


def _acronym_terms(query: str) -> tuple[str, ...]:
    return clean_unique_strings(match.group(0).lower() for match in _UPPER_ACRONYM_RE.finditer(query))


def _capitalized_terms(raw_tokens: Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []

    for index, token in enumerate(raw_tokens):
        cleaned = token.strip()
        if not cleaned:
            continue

        normalized = _normalize_token(cleaned)

        if normalized in _FUNCTION_WORDS:
            continue

        if cleaned[:1].isupper() and not cleaned.isupper():
            values.append(normalized)

    return clean_unique_strings(values)


def _weighted_terms(
    *,
    raw_tokens: Sequence[str],
    normalized_tokens: Sequence[str],
    exact_terms: Sequence[str],
    entity_terms: Sequence[str],
) -> tuple[QueryTerm, ...]:
    exact_lookup = {item.lower() for item in exact_terms}
    entity_lookup = {item.lower() for item in entity_terms}

    terms: list[QueryTerm] = []

    for raw_token, token in zip(raw_tokens, normalized_tokens, strict=False):
        if not token:
            continue

        features: list[str] = []
        weight = 1.0

        if token in _FUNCTION_WORDS:
            features.append("function")
            weight = 0.25

        if token in entity_lookup:
            features.append("entity")
            weight += 0.7

        if raw_token.lower() in exact_lookup or token in exact_lookup:
            features.append("exact")
            weight += 0.8

        if _looks_code_like(raw_token):
            features.append("code")
            weight += 0.6

        if _looks_path_like(raw_token):
            features.append("path")
            weight += 0.7

        if len(token) >= 7 and token not in _FUNCTION_WORDS:
            features.append("long")
            weight += 0.35

        if _rare_shape(raw_token):
            features.append("rare_shape")
            weight += 0.35

        if raw_token.islower() and len(token) <= 3:
            features.append("short_lower")
            weight -= 0.35

        terms.append(
            QueryTerm(
                text=token,
                weight=round(max(0.05, weight), 4),
                features=tuple(dict.fromkeys(features)),
            )
        )

    best: dict[str, QueryTerm] = {}
    for term in terms:
        current = best.get(term.text)
        if current is None or term.weight > current.weight:
            best[term.text] = term

    return tuple(sorted(best.values(), key=lambda item: (-item.weight, item.text)))


def _dynamic_focus_threshold(weighted_terms: Sequence[QueryTerm]) -> float:
    if not weighted_terms:
        return math.inf

    weights = sorted(item.weight for item in weighted_terms)
    mean = sum(weights) / len(weights)
    return max(1.25, round(mean, 4))


def _phrase_terms(
    *,
    raw_tokens: Sequence[str],
    focus_terms: Sequence[str],
) -> tuple[str, ...]:
    if not raw_tokens or not focus_terms:
        return ()

    focus = {item.lower() for item in focus_terms}
    normalized = tuple(_normalize_token(token) for token in raw_tokens)

    phrases: list[str] = []
    for size in (2, 3):
        for index in range(0, max(0, len(normalized) - size + 1)):
            window = normalized[index : index + size]
            if any(token in focus for token in window):
                phrase = " ".join(token for token in window if token)
                if len(phrase.split()) >= 2:
                    phrases.append(phrase)

    return clean_unique_strings(phrases)


def _structural_shape(*, query: str, referential_marker_count: int) -> str:
    if referential_marker_count:
        return "referential_follow_up"

    stripped = query.strip()
    if stripped.endswith("?"):
        return "question"

    if len(stripped.splitlines()) > 1:
        return "multi_line"

    return "statement"


def _defects(
    *,
    query: str,
    token_count: int,
    referential_marker_count: int,
    focus_terms: Sequence[str],
) -> tuple[QueryDefect, ...]:
    defects: list[QueryDefect] = []

    if token_count <= 2:
        defects.append(QueryDefect.LOW_INFORMATION)

    if len(query) > 2048:
        defects.append(QueryDefect.TOO_LONG)

    if referential_marker_count:
        defects.append(QueryDefect.CHAT_DEPENDENT)

    if not focus_terms and token_count <= 5:
        defects.append(QueryDefect.OVER_BROAD)

    return tuple(dict.fromkeys(defects))


def _conservative_intent(
    *,
    structural_shape: str,
    exact_terms: Sequence[str],
    source_hints: Sequence[str],
) -> QueryIntent:
    # No action-word dictionary. Keep intent conservative.
    if source_hints:
        return QueryIntent.NAVIGATION

    if exact_terms:
        return QueryIntent.API_REFERENCE

    if structural_shape == "referential_follow_up":
        return QueryIntent.FOLLOW_UP

    return QueryIntent.UNKNOWN


def _confidence(
    *,
    focus_terms: Sequence[str],
    exact_terms: Sequence[str],
    defects: Sequence[QueryDefect],
    lexical_specificity: float,
    entity_density: float,
    code_signal_density: float,
) -> float:
    value = 0.42

    if focus_terms:
        value += 0.22

    if exact_terms:
        value += 0.10

    value += min(0.12, lexical_specificity * 0.12)
    value += min(0.08, entity_density * 0.16)
    value += min(0.06, code_signal_density * 0.12)

    if QueryDefect.LOW_INFORMATION in defects:
        value -= 0.12

    if QueryDefect.OVER_BROAD in defects:
        value -= 0.10

    if QueryDefect.CHAT_DEPENDENT in defects:
        value -= 0.06

    return round(max(0.0, min(1.0, value)), 4)


def _lexical_specificity(
    *,
    weighted_terms: Sequence[QueryTerm],
    token_count: int,
) -> float:
    if token_count <= 0:
        return 0.0

    signal = sum(1 for term in weighted_terms if term.weight >= 1.35)
    return _ratio(signal, token_count)


def _has_any_feature(term: QueryTerm, features: set[str]) -> bool:
    return bool(set(term.features) & features)


def _looks_code_like(value: str) -> bool:
    return bool(_CODE_LIKE_RE.search(value))


def _looks_path_like(value: str) -> bool:
    return bool(_PATH_LIKE_RE.search(f" {value} "))


def _rare_shape(value: str) -> bool:
    if any(char.isdigit() for char in value) and any(char.isalpha() for char in value):
        return True

    if any(char in value for char in ("_", "-", "/", ".", ":")):
        return True

    return False


def _avg(values: Sequence[int]) -> float:
    values = tuple(values)
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(max(0.0, min(1.0, numerator / denominator)), 4)


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()

    if isinstance(value, str):
        return (value,) if value.strip() else ()

    try:
        return tuple(str(item).strip() for item in value if str(item).strip())
    except TypeError:
        return ()
