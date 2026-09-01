from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Mapping, Sequence

from rag.retrieval.contracts.candidates import RerankedCandidate
from rag.retrieval.contracts.query import NormalizedQuery
from rag.retrieval.contracts.request import RetrievalRequest
from rag.retrieval.query_processing.analysis import query_analysis_from_metadata


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_./:-]*|[0-9]+")

_FUNCTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
    "did", "do", "does", "for", "from", "how", "in", "is", "it", "its",
    "of", "on", "or", "that", "the", "these", "this", "those", "to",
    "was", "were", "what", "when", "where", "which", "who", "why", "with",
}


def calibrate_reranked_candidates(
    *,
    request: RetrievalRequest,
    normalized_query: NormalizedQuery,
    candidates: Sequence[RerankedCandidate],
) -> tuple[RerankedCandidate, ...]:
    request.validate()
    normalized_query.validate()

    if not candidates:
        return ()

    analysis = query_analysis_from_metadata(normalized_query.metadata)
    if analysis is None:
        return tuple(candidates)

    if analysis.exact_terms or analysis.source_hints:
        return tuple(candidates)

    constraints = request.source_constraints
    if (
        constraints.source_id
        or constraints.parser_doc_id
        or constraints.section_id
        or constraints.ir_id
        or constraints.heading_path
        or constraints.payload_filters
    ):
        return tuple(candidates)

    focus_terms = _calibration_terms(
        normalized_query=normalized_query,
        analysis=analysis,
    )
    if not focus_terms:
        return tuple(candidates)

    calibrated: list[tuple[float, RerankedCandidate, dict[str, Any]]] = []

    for candidate in candidates:
        info = _calibration_info(candidate=candidate, focus_terms=focus_terms)
        calibrated.append((info["calibrated_score"], candidate, info))

    ordered = sorted(
        calibrated,
        key=lambda item: (
            -item[0],
            int(item[1].rank),
            _candidate_identity(item[1]),
        ),
    )

    output: list[RerankedCandidate] = []

    for rank, (score, candidate, info) in enumerate(ordered, start=1):
        metadata = dict(candidate.metadata)
        metadata["query_calibration"] = info

        updated = replace(
            candidate,
            rank=rank,
            rerank_score=float(score),
            metadata=metadata,
        )
        updated.validate()
        output.append(updated)

    return tuple(output)


def _calibration_terms(
    *,
    normalized_query: NormalizedQuery,
    analysis: Any,
) -> tuple[str, ...]:
    """
    Build soft calibration terms.

    Important for conversational retrieval:
    - resolved/memory entity terms are useful anchors,
    - current user-turn intent terms must not disappear.

    Example:
        raw:      "what about its security protocols?"
        resolved: "what about Amazon ECS security protocols?"

    We want calibration to include "security" and "protocols",
    not only the resolved entity/topic.
    """
    terms: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        token = _clean_token(value)
        if not token or token in seen:
            return
        seen.add(token)
        terms.append(token)

    metadata = dict(getattr(normalized_query, "metadata", {}) or {})
    stateful = metadata.get("stateful_retrieval") or {}

    raw_query = str(
        stateful.get("raw_query")
        or getattr(normalized_query, "original_query", "")
        or ""
    )
    resolved_query = str(
        stateful.get("resolved_query")
        or getattr(normalized_query, "normalized_query", "")
        or ""
    )

    # Keep existing focus terms, but do not let them become the whole universe.
    for term in getattr(analysis, "focus_terms", ()) or ():
        add(term)

    # Current user wording usually contains the actual intent:
    # security, storage, limits, pricing, troubleshooting, setup, etc.
    for term in _content_tokens(raw_query):
        add(term)

    # If an anchor term literally appears in the current turn, it is probably
    # user intent, not just memory-injected topic context.
    raw_blob = f" {raw_query.lower()} "
    for term in getattr(analysis, "anchor_terms", ()) or ():
        token = _clean_token(term)
        if token and f" {token} " in raw_blob:
            add(token)

    # If the resolved query added a useful non-pronoun term, keep it softly too.
    # This helps when resolver turns "its" into the actual entity.
    for term in _content_tokens(resolved_query):
        add(term)

    return tuple(terms)


def _calibration_info(
    *,
    candidate: RerankedCandidate,
    focus_terms: Sequence[str],
) -> dict[str, Any]:
    base_score = float(candidate.rerank_score)

    text = _candidate_text(candidate).lower()
    title = _candidate_title(candidate).lower()
    source = _source_key(candidate).lower()

    title_source = " ".join((title, source))
    full = " ".join((title_source, text))

    focus = tuple(term.lower() for term in focus_terms if term.strip())
    title_tokens = _content_tokens(title_source)

    focus_hits_full = sum(1 for term in focus if term in full)
    focus_hits_title = sum(1 for term in focus if term in title_source)

    focus_coverage = focus_hits_full / max(1, len(focus))
    title_coverage = focus_hits_title / max(1, len(focus))
    title_compactness = focus_hits_title / max(1, len(title_tokens))

    extra_title_tokens = [
        token
        for token in title_tokens
        if token not in focus
    ]
    specialization_ratio = len(extra_title_tokens) / max(1, len(title_tokens))

    focus_phrase = " ".join(focus)
    phrase_in_title = bool(focus_phrase and focus_phrase in title_source)
    phrase_in_text = bool(focus_phrase and focus_phrase in full)

    adjustment = 0.0

    # Soft relevance lift. This is intentionally bounded and never filters.
    adjustment += 0.018 * focus_coverage
    adjustment += 0.022 * title_coverage
    adjustment += 0.012 * title_compactness

    if phrase_in_title:
        adjustment += 0.018
    elif phrase_in_text:
        adjustment += 0.010

    # Mildly demote narrow/specialized titles for broad follow-up queries.
    if len(focus) <= 3 and specialization_ratio >= 0.70 and title_compactness < 0.35:
        adjustment -= 0.012
    elif specialization_ratio >= 0.85:
        adjustment -= 0.008

    # Command/code-heavy chunks can still survive, but should not outrank
    # explanatory chunks unless they strongly cover the focus.
    code_marker_count = text.count("$ ") + text.count("```") + text.count(" --")
    if code_marker_count >= 2 and focus_coverage < 1.0:
        adjustment -= 0.010

    # Short chunks are not killed, only nudged down.
    if len(text) < 120:
        adjustment -= 0.008

    # No focus coverage is suspicious, but keep the candidate alive.
    if focus_coverage <= 0.0 and title_coverage <= 0.0:
        adjustment -= 0.018

    # Guardrail: calibration can reorder close calls, not bulldoze retrieval.
    adjustment = max(-0.035, min(0.055, adjustment))
    calibrated_score = base_score + adjustment

    return {
        "enabled": True,
        "base_score": round(base_score, 8),
        "calibrated_score": round(calibrated_score, 8),
        "adjustment": round(adjustment, 8),
        "focus_coverage": round(focus_coverage, 4),
        "title_coverage": round(title_coverage, 4),
        "title_compactness": round(title_compactness, 4),
        "specialization_ratio": round(specialization_ratio, 4),
        "phrase_in_title": phrase_in_title,
        "phrase_in_text": phrase_in_text,
        "bounded": True,
    }


def _candidate_text(candidate: RerankedCandidate) -> str:
    nested = candidate.candidate

    text = _clean_optional(getattr(nested, "text", None))
    if text:
        return text

    payload = _candidate_payload(candidate)
    return _clean_optional(payload.get("text")) or ""


def _candidate_title(candidate: RerankedCandidate) -> str:
    nested = candidate.candidate
    payload = _candidate_payload(candidate)
    text = _candidate_text(candidate)

    parts: list[str] = []

    # Most of your chunks encode useful title/context inside text.
    context_match = re.search(r"Context:\s*(.*?)(?:\s+Function:|\s+Heading:|\s+Text:|$)", text)
    if context_match:
        parts.append(context_match.group(1))

    heading_match = re.search(r"Heading:\s*(.*?)(?:\s+Function:|\s+Text:|$)", text)
    if heading_match:
        parts.append(heading_match.group(1))

    for key in ("heading", "heading_path", "context", "source_title", "title", "topic"):
        value = _clean_optional(getattr(nested, key, None))
        if value:
            parts.append(value)

        value = _clean_optional(payload.get(key))
        if value:
            parts.append(value)

    return " ".join(parts)


def _source_key(candidate: RerankedCandidate) -> str:
    nested = candidate.candidate
    payload = _candidate_payload(candidate)

    for key in ("source_id", "source", "source_path", "file_path", "file_name", "parser_doc_id"):
        value = _clean_optional(getattr(nested, key, None))
        if value:
            return value

        ref = getattr(nested, "ref", None)
        if ref is not None:
            value = _clean_optional(getattr(ref, key, None))
            if value:
                return value

        value = _clean_optional(payload.get(key))
        if value:
            return value

    return "unknown_source"


def _candidate_payload(candidate: RerankedCandidate) -> dict[str, Any]:
    payload = getattr(candidate.candidate, "payload", None)
    if isinstance(payload, Mapping):
        return dict(payload)

    metadata = getattr(candidate.candidate, "metadata", None)
    if isinstance(metadata, Mapping):
        nested = metadata.get("payload")
        if isinstance(nested, Mapping):
            return dict(nested)
        return dict(metadata)

    return {}


def _content_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in (_clean_token(item) for item in _TOKEN_RE.findall(text))
        if token and token not in _FUNCTION_WORDS
    )


def _candidate_identity(candidate: RerankedCandidate) -> str:
    nested = candidate.candidate

    for attr in ("chunk_id", "point_id", "source_id", "parser_doc_id", "section_id"):
        value = _clean_optional(getattr(nested, attr, None))
        if value:
            return value

    return _candidate_text(candidate)[:160]


def _clean_token(value: Any) -> str:
    return str(value).strip().lower().strip(".,;:!?()[]{}<>")


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        joined = " ".join(str(item) for item in value if str(item).strip())
        return joined.strip() or None

    cleaned = str(value).strip()
    return cleaned or None
