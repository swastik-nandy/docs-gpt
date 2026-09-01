from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


ANSWER_NEED_GENERAL = "general"
ANSWER_NEED_DEFINITION = "definition"
ANSWER_NEED_PROCEDURE = "procedure"
ANSWER_NEED_COMPARISON = "comparison"
ANSWER_NEED_CONSTRAINT = "constraint"
ANSWER_NEED_TROUBLESHOOTING = "troubleshooting"
ANSWER_NEED_LIST = "list_lookup"
ANSWER_NEED_REFERENCE = "reference"


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "what", "which", "who", "whom", "whose", "when", "where", "why",
    "how", "do", "does", "did", "can", "could", "should", "would",
    "i", "me", "my", "we", "our", "you", "your", "it", "this", "that",
    "these", "those", "of", "for", "to", "in", "on", "at", "by", "with",
    "from", "about", "into", "over", "under", "and", "or", "but", "as",
    "using", "use", "tell", "show", "give", "explain",
}

_INTENT_WORDS = {
    "define", "definition", "meaning", "overview", "summary",
    "create", "configure", "setup", "set", "install", "deploy", "run",
    "compare", "difference", "versus", "vs", "better",
    "limit", "allowed", "supported", "support", "required", "must",
    "error", "failed", "failure", "troubleshoot", "fix", "issue",
    "list", "available", "options", "types",
}


_DEFINITION_PATTERNS = (
    r"\b(is|are)\s+(?:a|an|the)\b",
    r"\b(is|are)\s+used\s+to\b",
    r"\b(refers?\s+to|means|is\s+called|known\s+as)\b",
    r"\b(enables|allows|lets|provides|offers|helps)\b",
    r"\b(technology|service|platform|engine|feature|capability|component|resource)\b",
)

_PROCEDURE_PATTERNS = (
    r"\b(step|steps|procedure|walkthrough|tutorial)\b",
    r"\b(create|configure|set|define|run|deploy|launch|install|enable|disable|add|remove)\b",
    r"\b(command|cli|console|api|parameter|setting|task definition|configuration)\b",
    r"\b(first|next|then|after|before|finally)\b",
)

_CONSTRAINT_PATTERNS = (
    r"\b(must|must not|cannot|can't|can only|required|requires|not supported|supported)\b",
    r"\b(limit|quota|maximum|minimum|max|min|allowed|valid|invalid|only)\b",
    r"\b\d+(?:\.\d+)?\b",
)

_COMPARISON_PATTERNS = (
    r"\b(vs|versus|compare|comparison|difference|different|tradeoff|trade-off)\b",
    r"\b(whereas|while|compared|instead|rather|unlike|similar|both)\b",
    r"\b(better|worse|prefer|choose|option)\b",
)

_TROUBLESHOOTING_PATTERNS = (
    r"\b(error|exception|failed|failure|crash|timeout|denied|refused|broken|not working)\b",
    r"\b(cause|because|reason|fix|resolve|troubleshoot|debug|check|verify)\b",
    r"\b(warning|symptom|issue|problem)\b",
)

_LIST_PATTERNS = (
    r"\b(following|available|include|includes|types|options|providers|values|examples)\b",
    r"\b(list|listed|enumerated)\b",
)

_REFERENCE_PATTERNS = (
    r"\b(parameter|field|property|attribute|method|function|class|api|syntax|schema)\b",
    r"`[^`]+`",
)


_INCIDENTAL_PATTERNS = (
    r"\bclick here\b",
    r"\bsign up\b",
    r"\bworkshop\b",
    r"\bwebinar\b",
    r"\blearn more\b",
    r"\bfor more information\b",
    r"\bsee also\b",
    r"\btip:\b",
)


@dataclass(frozen=True)
class AnswerabilityAssessment:
    answer_need: str
    score: float
    answer_bearing: bool
    mention_only: bool
    entity_overlap: float
    phrase_overlap: float
    evidence_score: float
    evidence_hits: tuple[str, ...]
    reason: str


def classify_answer_need(query: str) -> str:
    q = _normalize_space(query).lower()

    if not q:
        return ANSWER_NEED_GENERAL

    if re.search(r"\b(error|exception|failed|failure|not working|broken|timeout|denied|refused|why.*fail|why.*error)\b", q):
        return ANSWER_NEED_TROUBLESHOOTING

    if re.search(r"\b(compare|difference|vs\.?|versus|tradeoff|trade-off|better|which one|choose between)\b", q):
        return ANSWER_NEED_COMPARISON

    if re.search(r"\b(max|min|maximum|minimum|limit|quota|allowed|valid|supported|not supported|required|must|cannot|can't|can i|can we|target type|which .* type)\b", q):
        return ANSWER_NEED_CONSTRAINT

    if re.search(r"\b(how to|how do i|how do we|how can i|setup|set up|configure|create|deploy|run|install|enable|disable|integrate)\b", q):
        return ANSWER_NEED_PROCEDURE

    if re.search(r"\b(list|show all|what are the available|available .* options|available .* types|types of|providers|values)\b", q):
        return ANSWER_NEED_LIST

    if re.search(r"\b(parameter|field|property|attribute|method|function|class|api|syntax|schema|where is .* configured)\b", q):
        return ANSWER_NEED_REFERENCE

    if re.search(r"^\s*(what is|what are|define|definition of|meaning of|explain what|overview of)\b", q):
        return ANSWER_NEED_DEFINITION

    return ANSWER_NEED_GENERAL


def assess_candidate_answerability(
    *,
    query: str,
    text: str,
    heading: str = "",
    answer_need: str | None = None,
) -> AnswerabilityAssessment:
    need = answer_need or classify_answer_need(query)
    query_terms = _focus_terms(query)

    candidate_text = _normalize_space(f"{heading} {text}").lower()
    body_text = _normalize_space(text).lower()

    if not candidate_text:
        return AnswerabilityAssessment(
            answer_need=need,
            score=0.0,
            answer_bearing=False,
            mention_only=False,
            entity_overlap=0.0,
            phrase_overlap=0.0,
            evidence_score=0.0,
            evidence_hits=(),
            reason="empty candidate text",
        )

    entity_overlap = _term_overlap(query_terms, _tokens(candidate_text))
    phrase_overlap = _phrase_overlap(query_terms, candidate_text)

    evidence_hits = _evidence_hits(need, body_text)
    evidence_score = min(1.0, len(evidence_hits) / 3.0)

    entity_present = entity_overlap > 0.0 or phrase_overlap > 0.0
    incidental = _has_any(_INCIDENTAL_PATTERNS, body_text)

    if need == ANSWER_NEED_GENERAL:
        score = 0.25 + (0.50 * entity_overlap) + (0.25 * phrase_overlap)
        return AnswerabilityAssessment(
            answer_need=need,
            score=round(score, 6),
            answer_bearing=True,
            mention_only=False,
            entity_overlap=entity_overlap,
            phrase_overlap=phrase_overlap,
            evidence_score=evidence_score,
            evidence_hits=evidence_hits,
            reason="general query",
        )

    answer_bearing = entity_present and evidence_score > 0.0

    # Strong definition evidence can be "X is ..." without many pattern hits.
    if need == ANSWER_NEED_DEFINITION and entity_present:
        if _looks_like_definition_sentence(body_text, query_terms):
            answer_bearing = True
            evidence_score = max(evidence_score, 0.75)
            evidence_hits = tuple(dict.fromkeys((*evidence_hits, "definition_sentence")))

    mention_only = entity_present and not answer_bearing

    if incidental and not answer_bearing:
        mention_only = True

    score = (
        (0.35 * entity_overlap)
        + (0.20 * phrase_overlap)
        + (0.45 * evidence_score)
    )

    if mention_only:
        score -= 0.55

    if incidental and not answer_bearing:
        score -= 0.20

    score = max(-1.0, min(1.5, score))

    reason_parts = [
        f"need={need}",
        f"entity_overlap={entity_overlap:.2f}",
        f"phrase_overlap={phrase_overlap:.2f}",
        f"evidence_score={evidence_score:.2f}",
    ]

    if mention_only:
        reason_parts.append("mention_only=true")
    if answer_bearing:
        reason_parts.append("answer_bearing=true")

    return AnswerabilityAssessment(
        answer_need=need,
        score=round(score, 6),
        answer_bearing=answer_bearing,
        mention_only=mention_only,
        entity_overlap=round(entity_overlap, 6),
        phrase_overlap=round(phrase_overlap, 6),
        evidence_score=round(evidence_score, 6),
        evidence_hits=evidence_hits,
        reason="; ".join(reason_parts),
    )


def should_apply_answerability(query: str) -> bool:
    return classify_answer_need(query) != ANSWER_NEED_GENERAL


def _evidence_hits(answer_need: str, text: str) -> tuple[str, ...]:
    if answer_need == ANSWER_NEED_DEFINITION:
        patterns = _DEFINITION_PATTERNS
    elif answer_need == ANSWER_NEED_PROCEDURE:
        patterns = _PROCEDURE_PATTERNS
    elif answer_need == ANSWER_NEED_CONSTRAINT:
        patterns = _CONSTRAINT_PATTERNS
    elif answer_need == ANSWER_NEED_COMPARISON:
        patterns = _COMPARISON_PATTERNS
    elif answer_need == ANSWER_NEED_TROUBLESHOOTING:
        patterns = _TROUBLESHOOTING_PATTERNS
    elif answer_need == ANSWER_NEED_LIST:
        patterns = _LIST_PATTERNS
    elif answer_need == ANSWER_NEED_REFERENCE:
        patterns = _REFERENCE_PATTERNS
    else:
        patterns = ()

    hits: list[str] = []
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits.append(pattern)

    return tuple(hits)


def _looks_like_definition_sentence(text: str, query_terms: tuple[str, ...]) -> bool:
    if not query_terms:
        return False

    # Look for "X is/are ..." where at least one focus term is near a copula.
    for term in query_terms:
        escaped = re.escape(term)
        if re.search(rf"\b{escaped}\b[^.\n]{{0,80}}\b(is|are)\s+(a|an|the|used|known|called)?\b", text):
            return True

    if re.search(r"\b(is|are)\s+(a|an|the)\s+\w+", text) and any(term in text for term in query_terms):
        return True

    return False


def _focus_terms(query: str) -> tuple[str, ...]:
    values: list[str] = []
    for token in _tokens(query):
        if token in _STOPWORDS:
            continue
        if token in _INTENT_WORDS:
            continue
        if len(token) <= 1:
            continue
        values.append(token)

    return tuple(dict.fromkeys(values))


def _phrase_overlap(query_terms: tuple[str, ...], text: str) -> float:
    if len(query_terms) < 2:
        return 0.0

    phrases: list[str] = []
    for size in (2, 3):
        if len(query_terms) < size:
            continue
        for index in range(0, len(query_terms) - size + 1):
            phrases.append(" ".join(query_terms[index : index + size]))

    if not phrases:
        return 0.0

    hits = sum(1 for phrase in phrases if phrase in text)
    return hits / len(phrases)


def _term_overlap(required: Iterable[str], present: set[str]) -> float:
    required_set = set(required)
    if not required_set:
        return 0.0

    return len(required_set & present) / len(required_set)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]{1,}", text.lower().replace("_", " ").replace("-", " "))
        if token
    }


def _normalize_space(text: str) -> str:
    return " ".join(str(text or "").split())


def _has_any(patterns: Iterable[str], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)
