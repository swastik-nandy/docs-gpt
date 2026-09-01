from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


TurnAction = Literal[
    "document_question",
    "referential_document_followup",
    "previous_answer_verification",
    "previous_answer_transform",
    "clarification_needed",
    "conversation_only",
]

AnswerScope = Literal[
    "knowledge_base",
    "assistant_profile",
    "general_conversation",
    "previous_answer",
]


_ALLOWED_ACTIONS = {
    "document_question",
    "referential_document_followup",
    "previous_answer_verification",
    "previous_answer_transform",
    "clarification_needed",
    "conversation_only",
}

_ALLOWED_ANSWER_SCOPES = {
    "knowledge_base",
    "assistant_profile",
    "general_conversation",
    "previous_answer",
}


_SMALLTALK_RE = re.compile(
    r"^\s*"
    r"(hi|hello|hey|yo|sup|namaste|thanks|thank you|ok|okay|cool|nice|great|awesome|hmm|hmmm|lol|haha)"
    r"(?:\s*[,\-!]?\s*(docs[-\s]?gpt|docsgpt|assistant|bot|bro|buddy|there))?"
    r"\s*[!.?]*\s*$",
    re.IGNORECASE,
)

_CONVERSATION_META_RE = re.compile(
    r"\b("
    r"who are you|what can you do|can you help|help me|"
    r"how are you|are you there|hello there|"
    r"explain yourself|what are your capabilities|"
    r"start|continue|go on|sounds good|got it"
    r")\b",
    re.IGNORECASE,
)

_ASSISTANT_IDENTITY_RE = re.compile(
    r"\b("
    r"who are you|what are you|what(?:'s| is) your name|"
    r"what model are you|which model are you|"
    r"what llm are you|which llm are you|"
    r"what model do you use|which model do you use|"
    r"what llm do you use|"
    r"who built you|who made you|who created you|"
    r"who developed you|"
    r"your creator|your builder|your developer|"
    r"your model|your llm|your provider|"
    r"are you from [a-z0-9_.+\- ]+|"
    r"(?:were|are) you "
    r"(?:built|made|created|developed) by "
    r"[a-z0-9_.+\- ]+|"
    r"are you "
    r"(?:using|running|powered by|based on) "
    r"[a-z0-9_.+\- ]+|"
    r"do you use [a-z0-9_.+\- ]+ as "
    r"(?:your )?(?:model|llm|provider)"
    r")\b",
    re.IGNORECASE,
)

_KNOWLEDGE_QUESTION_RE = re.compile(
    r"\b("
    r"what|which|how|why|when|where|who|"
    r"does|do|can|could|should|is|are|"
    r"show me|tell me|explain|describe|compare|list|"
    r"use|used|support|supports|work|works"
    r")\b",
    re.IGNORECASE,
)

_SOURCE_BOUND_RE = re.compile(
    r"\b("
    r"doc|docs|document|documents|provided context|provided docs|knowledge base|"
    r"retrieved|source|sources|citation|citations|according to|based on|from the docs|"
    r"ecs|fargate|aws|cluster|task definition|service|capacity provider|cloudwatch|iam|vpc"
    r")\b",
    re.IGNORECASE,
)

_VERIFICATION_RE = re.compile(
    r"\b("
    r"are you sure|sure about|verify|is that correct|is this correct|"
    r"right\??|accurate|did you check|what you saw|where did you get|"
    r"source|citation|evidence"
    r")\b",
    re.IGNORECASE,
)

_TRANSFORM_RE = re.compile(
    r"\b("
    r"shorten|make it short|summarize|simplify|explain again|rephrase|"
    r"make it|format|bullet|table|checklist|rewrite|expand this|"
    r"continue|go on|keep going|"
    r"from the provided answers|from your answer|based on your answer"
    r")\b",
    re.IGNORECASE,
)

_REFERENTIAL_RE = re.compile(
    r"\b("
    r"it|its|that|this|same|previous|earlier|above|"
    r"those|these|they|them|their|one|ones|both|"
    r"either|former|latter|first|second"
    r")\b",
    re.IGNORECASE,
)

_FOLLOWUP_EXTENSION_RE = re.compile(
    r"\b("
    r"other|more|more important|else|also|another|additional|"
    r"and|then|instead|versus|vs|compared|"
    r"what about|what else|anything else|need to know|"
    r"limits?|restrictions?|constraints?|quotas?|rules?|requirements?|length|long|size|max|maximum|min|minimum|keys?|values?|characters?|"
    r"important things|key things|main things|gotchas?|caveats?"
    r")\b",
    re.IGNORECASE,
)

_DOC_CITATION_RE = re.compile(
    r"\bdoc_[a-f0-9]+:[a-z0-9_:-]+\b"
    r"|\[doc_[^\]]+\]"
    r"|\[[^\]\n]+\.(?:"
    r"md|mdx|markdown|mdown|mkd|mdwn|"
    r"txt|pdf|doc|docx"
    r")\]",
    re.IGNORECASE,
)

_TECHNICAL_ACTION_RE = re.compile(
    r"\b("
    r"how to|how do|tell me how|spot|detect|debug|troubleshoot|fix|configure|setup|install|deploy|"
    r"errors?|issues?|failures?|exceptions?|logs?|metrics?|monitor|observe|health|status|"
    r"api|endpoint|database|db|vector|index|collection|latency|memory|storage|disk|auth|"
    r"security|protocol|quota|cluster|service|container|server|request|response"
    r")\b",
    re.IGNORECASE,
)

_TECHNICAL_OBJECT_RE = re.compile(
    r"\b("
    r"db|database|api|sdk|cli|server|service|cluster|index|collection|vector|embedding|"
    r"models?|llm|agents?|interactions?|framework|library|package|client|"
    r"code|function|method|features?|capabilit(?:y|ies)|workflow|"
    r"platform|tool|product|system|robotics|"
    r"logs?|metrics?|errors?|issues?|latency|memory|storage|auth|security|protocol|"
    r"container|deployment|configuration|config|pipeline|queue|cache|broker"
    r")\b",
    re.IGNORECASE,
)

_PRODUCTISH_TECH_RE = re.compile(
    r"\b[a-z][a-z0-9_-]{2,}\s+"
    r"(db|database|api|sdk|cli|server|service|cluster|index|collection|logs?|metrics?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class TurnPlan:
    action: TurnAction
    needs_document_retrieval: bool
    retrieval_query: str | None = None
    answer_scope: AnswerScope = "knowledge_base"
    allow_conversation_answer: bool = False
    needs_clarification: bool = False
    uses_previous_turns: bool = False
    confidence: float = 0.5
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_state_payload(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "needs_document_retrieval": self.needs_document_retrieval,
            "retrieval_query": self.retrieval_query,
            "answer_scope": self.answer_scope,
            "allow_conversation_answer": self.allow_conversation_answer,
            "needs_clarification": self.needs_clarification,
            "uses_previous_turns": self.uses_previous_turns,
            "confidence": self.confidence,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


def deterministic_turn_plan(
    *,
    current_user_message: str,
    recent_user_message: str | None,
    recent_assistant_message: str | None,
) -> TurnPlan | None:
    message = " ".join(
        (current_user_message or "")
        .strip()
        .split()
    )

    has_recent = bool(
        recent_user_message
        or recent_assistant_message
    )

    if not message:
        return TurnPlan(
            action="conversation_only",
            needs_document_retrieval=False,
            retrieval_query=None,
            answer_scope="general_conversation",
            allow_conversation_answer=True,
            confidence=1.0,
            reason=(
                "Blank message should not "
                "trigger retrieval."
            ),
            metadata={
                "deterministic": True,
            },
        )

    if _SMALLTALK_RE.match(message):
        return TurnPlan(
            action="conversation_only",
            needs_document_retrieval=False,
            retrieval_query=None,
            answer_scope="general_conversation",
            allow_conversation_answer=True,
            confidence=0.98,
            reason=(
                "Greeting or small-talk message "
                "should be handled conversationally."
            ),
            metadata={
                "deterministic": True,
            },
        )

    #------------------------PREVIOUS ANSWER TRANSFORM-------------------------

    if (
        has_recent
        and _TRANSFORM_RE.search(message)
        and not _SOURCE_BOUND_RE.search(message)
    ):
        return TurnPlan(
            action="previous_answer_transform",
            needs_document_retrieval=False,
            retrieval_query=None,
            answer_scope="previous_answer",
            allow_conversation_answer=True,
            uses_previous_turns=True,
            confidence=0.9,
            reason=(
                "User asks to transform the previous "
                "answer without requesting new source lookup."
            ),
            metadata={
                "deterministic": True,
            },
        )

    #------------------------VERIFICATION-------------------------

    if (
        has_recent
        and _VERIFICATION_RE.search(message)
    ):
        recent_text = " ".join(
            item
            for item in (
                recent_user_message or "",
                recent_assistant_message or "",
            )
            if item
        )

        if (
            _ASSISTANT_IDENTITY_RE.search(message)
            or _ASSISTANT_IDENTITY_RE.search(
                recent_text
            )
        ):
            return TurnPlan(
                action="conversation_only",
                needs_document_retrieval=False,
                retrieval_query=None,
                answer_scope="assistant_profile",
                allow_conversation_answer=True,
                uses_previous_turns=True,
                confidence=0.92,
                reason=(
                    "User is verifying assistant "
                    "identity/persona."
                ),
                metadata={
                    "deterministic": True,
                    "identity_verification": True,
                },
            )

        recent_user = (
            recent_user_message
            or ""
        )

        recent_assistant = (
            recent_assistant_message
            or ""
        )

        previous_was_source_bound = (
            _SOURCE_BOUND_RE.search(
                recent_user
            )
            is not None
            or _looks_like_knowledge_base_technical_query(
                recent_user
            )
            or _SOURCE_BOUND_RE.search(
                recent_assistant
            )
            is not None
            or _TECHNICAL_OBJECT_RE.search(
                recent_assistant
            )
            is not None
            or _DOC_CITATION_RE.search(
                recent_assistant
            )
            is not None
        )

        if previous_was_source_bound:
            retrieval_query = (
                _compose_followup_retrieval_query(
                    current_user_message=message,
                    recent_user_message=(
                        recent_user_message
                    ),
                    recent_assistant_message=(
                        recent_assistant_message
                    ),
                )
            )

            return TurnPlan(
                action=(
                    "previous_answer_verification"
                ),
                needs_document_retrieval=True,
                retrieval_query=retrieval_query,
                answer_scope="knowledge_base",
                allow_conversation_answer=False,
                uses_previous_turns=True,
                confidence=0.88,
                reason=(
                    "User asks to verify the previous "
                    "source-grounded answer."
                ),
                metadata={
                    "deterministic": True,
                },
            )

        return TurnPlan(
            action="conversation_only",
            needs_document_retrieval=False,
            retrieval_query=None,
            answer_scope="previous_answer",
            allow_conversation_answer=True,
            uses_previous_turns=True,
            confidence=0.84,
            reason=(
                "User is verifying a prior "
                "conversational answer."
            ),
            metadata={
                "deterministic": True,
                "conversation_verification": True,
            },
        )

    #------------------------ASSISTANT IDENTITY-------------------------

    identity_cleaned_message = (
        _strip_assistant_identity_terms(
            message
        )
    )

    if (
        _ASSISTANT_IDENTITY_RE.search(
            message
        )
        and not _SOURCE_BOUND_RE.search(
            identity_cleaned_message
        )
    ):
        return TurnPlan(
            action="conversation_only",
            needs_document_retrieval=False,
            retrieval_query=None,
            answer_scope="assistant_profile",
            allow_conversation_answer=True,
            uses_previous_turns=has_recent,
            confidence=0.97,
            reason=(
                "User is explicitly asking about "
                "assistant identity/persona."
            ),
            metadata={
                "deterministic": True,
                "assistant_identity_turn": True,
            },
        )

    #------------------------STATEFUL DOCUMENT FOLLOWUP-------------------------

    if (
        has_recent
        and _looks_like_source_bound_followup(
            current_user_message=message,
            recent_user_message=(
                recent_user_message
            ),
            recent_assistant_message=(
                recent_assistant_message
            ),
        )
    ):
        retrieval_query = (
            _compose_followup_retrieval_query(
                current_user_message=message,
                recent_user_message=(
                    recent_user_message
                ),
                recent_assistant_message=(
                    recent_assistant_message
                ),
            )
        )

        return TurnPlan(
            action=(
                "referential_document_followup"
            ),
            needs_document_retrieval=True,
            retrieval_query=retrieval_query,
            answer_scope="knowledge_base",
            allow_conversation_answer=False,
            uses_previous_turns=True,
            confidence=0.9,
            reason=(
                "User is asking a contextual "
                "follow-up to the previous "
                "source-bound answer."
            ),
            metadata={
                "deterministic": True,
                "source_bound_followup": True,
            },
        )

    #------------------------FRESH KNOWLEDGE QUERY-------------------------

    if _looks_like_knowledge_base_technical_query(
        message
    ):
        return TurnPlan(
            action="document_question",
            needs_document_retrieval=True,
            retrieval_query=(
                current_user_message
            ),
            answer_scope="knowledge_base",
            allow_conversation_answer=False,
            uses_previous_turns=False,
            confidence=0.9,
            reason=(
                "User is asking a concrete product, "
                "technical, or system question that "
                "should be answered from source documents."
            ),
            metadata={
                "deterministic": True,
                "generic_technical_source_bound": True,
            },
        )

    #------------------------GENERAL CONVERSATION-------------------------

    if (
        _CONVERSATION_META_RE.search(
            message
        )
        and not _SOURCE_BOUND_RE.search(
            message
        )
    ):
        return TurnPlan(
            action="conversation_only",
            needs_document_retrieval=False,
            retrieval_query=None,
            answer_scope="general_conversation",
            allow_conversation_answer=True,
            uses_previous_turns=has_recent,
            confidence=0.9,
            reason=(
                "General conversation/meta message "
                "does not need document retrieval."
            ),
            metadata={
                "deterministic": True,
            },
        )

    return None


def parse_turn_plan_response(
    raw_text: str,
    *,
    current_user_message: str,
) -> TurnPlan:
    payload = _load_json_payload(raw_text)

    if not isinstance(payload, Mapping):
        raise ValueError("turn planning response must be a JSON object.")

    action = _clean_action(payload.get("action"))
    answer_scope = _clean_answer_scope(payload.get("answer_scope"), action=action)

    needs_document_retrieval = _coerce_bool(
        payload.get("needs_document_retrieval"),
        default=(answer_scope == "knowledge_base"),
    )

    retrieval_query = _optional_text(
        payload.get("retrieval_query")
        or payload.get("standalone_query")
        or payload.get("query_for_retrieval")
    )

    if needs_document_retrieval and not retrieval_query:
        retrieval_query = current_user_message

    return TurnPlan(
        action=action,
        needs_document_retrieval=needs_document_retrieval,
        retrieval_query=retrieval_query,
        answer_scope=answer_scope,
        allow_conversation_answer=_coerce_bool(
            payload.get("allow_conversation_answer"),
            default=(answer_scope != "knowledge_base"),
        ),
        needs_clarification=_coerce_bool(
            payload.get("needs_clarification"),
            default=(action == "clarification_needed"),
        ),
        uses_previous_turns=_coerce_bool(
            payload.get("uses_previous_turns"),
            default=(action in {
                "referential_document_followup",
                "previous_answer_verification",
                "previous_answer_transform",
            }),
        ),
        confidence=_coerce_confidence(payload.get("confidence", 0.5)),
        reason=str(payload.get("reason") or "").strip(),
        metadata={
            key: value
            for key, value in payload.items()
            if key not in {
                "action",
                "needs_document_retrieval",
                "retrieval_query",
                "standalone_query",
                "query_for_retrieval",
                "answer_scope",
                "allow_conversation_answer",
                "needs_clarification",
                "uses_previous_turns",
                "confidence",
                "reason",
            }
        },
    )


def fallback_turn_plan(
    *,
    current_user_message: str,
    recent_user_message: str | None,
    recent_assistant_message: str | None,
) -> TurnPlan:
    deterministic = deterministic_turn_plan(
        current_user_message=current_user_message,
        recent_user_message=recent_user_message,
        recent_assistant_message=recent_assistant_message,
    )
    if deterministic is not None:
        return deterministic

    message = " ".join(current_user_message.lower().split())
    padded = f" {message} "
    has_recent = bool(recent_user_message or recent_assistant_message)

    if has_recent and _REFERENTIAL_RE.search(padded):
        return TurnPlan(
            action="referential_document_followup",
            needs_document_retrieval=True,
            retrieval_query=current_user_message,
            answer_scope="knowledge_base",
            allow_conversation_answer=False,
            uses_previous_turns=True,
            confidence=0.55,
            reason="Fallback detected a referential knowledge-base follow-up.",
            metadata={"fallback": True},
        )

    if _looks_like_knowledge_base_technical_query(current_user_message):
        return TurnPlan(
            action="document_question",
            needs_document_retrieval=True,
            retrieval_query=current_user_message,
            answer_scope="knowledge_base",
            allow_conversation_answer=False,
            uses_previous_turns=False,
            confidence=0.62,
            reason="Fallback detected a concrete technical/product query that should be source-bound.",
            metadata={
                "fallback": True,
                "generic_technical_source_bound": True,
            },
        )

    if not _SOURCE_BOUND_RE.search(message):
        return TurnPlan(
            action="conversation_only",
            needs_document_retrieval=False,
            retrieval_query=None,
            answer_scope="general_conversation",
            allow_conversation_answer=True,
            uses_previous_turns=has_recent,
            confidence=0.52,
            reason="Fallback treated message as general conversation because no knowledge-base signal was present.",
            metadata={"fallback": True},
        )

    return TurnPlan(
        action="document_question",
        needs_document_retrieval=True,
        retrieval_query=current_user_message,
        answer_scope="knowledge_base",
        allow_conversation_answer=False,
        uses_previous_turns=False,
        confidence=0.5,
        reason="Fallback defaulted to document retrieval for a source-bound question.",
        metadata={"fallback": True},
    )


def _looks_like_source_bound_followup(
    *,
    current_user_message: str,
    recent_user_message: str | None,
    recent_assistant_message: str | None,
) -> bool:
    message = " ".join((current_user_message or "").strip().split())
    if not message:
        return False

    if _SMALLTALK_RE.match(message):
        return False

    if _ASSISTANT_IDENTITY_RE.search(message):
        return False

    if not _FOLLOWUP_EXTENSION_RE.search(message) and not _REFERENTIAL_RE.search(message):
        return False

    recent_user = recent_user_message or ""
    recent_assistant = recent_assistant_message or ""

    previous_was_source_bound = (
        _SOURCE_BOUND_RE.search(recent_user) is not None
        or _looks_like_knowledge_base_technical_query(recent_user)
        or _SOURCE_BOUND_RE.search(recent_assistant) is not None
        or _TECHNICAL_OBJECT_RE.search(recent_assistant) is not None
        or _DOC_CITATION_RE.search(recent_assistant) is not None
    )

    return previous_was_source_bound


def _compose_followup_retrieval_query(
    *,
    current_user_message: str,
    recent_user_message: str | None,
    recent_assistant_message: str | None = None,
) -> str:
    current = " ".join((current_user_message or "").strip().split())
    previous = " ".join((recent_user_message or "").strip().split())
    assistant_context = " ".join((recent_assistant_message or "").strip().split())

    if len(assistant_context) > 260:
        assistant_context = assistant_context[:260].rstrip() + "..."

    pieces = []
    if previous:
        pieces.append(previous)
    if assistant_context:
        pieces.append(f"Previous answer context: {assistant_context}")
    if current:
        pieces.append(f"Follow-up: {current}")

    return ". ".join(pieces) or current or previous


def _looks_like_knowledge_base_technical_query(text: str) -> bool:
    """
    Generic source-bound detector.

    This intentionally avoids hardcoding vendors/products. It catches concrete
    technical/product questions such as:
    - "how to spot errors in qdrant db"
    - "debug redis latency"
    - "configure postgres auth"
    - "kafka consumer lag issue"

    Normal chat like "hi docs-gpt" or identity questions stay conversational.
    """
    message = " ".join((text or "").strip().split())
    if not message:
        return False

    if _SMALLTALK_RE.match(message):
        return False

    identity_cleaned = _strip_assistant_identity_terms(message)
    if _ASSISTANT_IDENTITY_RE.search(message) and not _SOURCE_BOUND_RE.search(identity_cleaned):
        return False

    if _SOURCE_BOUND_RE.search(message):
        return True

    if _PRODUCTISH_TECH_RE.search(message) and _TECHNICAL_ACTION_RE.search(message):
        return True

    if (
        _TECHNICAL_ACTION_RE.search(message)
        and _TECHNICAL_OBJECT_RE.search(message)
    ):
        return True

    if (
        _KNOWLEDGE_QUESTION_RE.search(message)
        and _TECHNICAL_OBJECT_RE.search(message)
    ):
        return True

    return False


def _load_json_payload(raw_text: str) -> Any:
    cleaned = raw_text.strip()
    cleaned = _strip_markdown_json_fence(cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        blob = _extract_first_json_blob(cleaned)
        if blob is None:
            raise
        return json.loads(blob)


def _strip_markdown_json_fence(text: str) -> str:
    fence = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence:
        return fence.group(1).strip()
    return text


def _extract_first_json_blob(text: str) -> str | None:
    start_candidates = [
        position
        for position in (text.find("{"), text.find("["))
        if position != -1
    ]
    if not start_candidates:
        return None

    start = min(start_candidates)
    opener = text[start]
    closer = "}" if opener == "{" else "]"

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]

        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == "\"":
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def _clean_action(value: Any) -> TurnAction:
    action = str(value or "document_question").strip().lower()
    if action not in _ALLOWED_ACTIONS:
        action = "document_question"
    return action  # type: ignore[return-value]


def _clean_answer_scope(value: Any, *, action: str) -> AnswerScope:
    answer_scope = str(value or "").strip().lower()

    if answer_scope in _ALLOWED_ANSWER_SCOPES:
        return answer_scope  # type: ignore[return-value]

    if action == "conversation_only":
        return "general_conversation"
    if action == "previous_answer_transform":
        return "previous_answer"

    return "knowledge_base"


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False

    return default


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.5

    return max(0.0, min(1.0, confidence))


def _strip_assistant_identity_terms(text: str) -> str:
    cleaned = re.sub(
        r"\b(docs[-\s]?gpt|docsgpt|docs gpt)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b("
        r"built by swastik|made by swastik|created by swastik|"
        r"built for embitel|made for embitel|created for embitel|"
        r"swastik nandy|swastik|embitel technologies|embitel|"
        r"google|gemini|gemma|openai|chatgpt|groq|anthropic|claude|"
        r"llm provider|model provider"
        r")\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    return " ".join(cleaned.split())
