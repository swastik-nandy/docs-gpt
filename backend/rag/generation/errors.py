from __future__ import annotations

import json
import logging
import re
import traceback
from datetime import UTC, datetime
from typing import Any, Mapping
from uuid import uuid4


Payload = dict[str, Any]


_SLOT_FAILURE_RE = re.compile(
    r"(?P<slot>LLM_\d+)=(?P<model>.*?) via (?P<provider>.*?) failed: (?P<error>.*?)(?=\s+\|\s+LLM_\d+=|$)",
    re.DOTALL,
)


def build_generation_error_event(
    exc: Exception,
    *,
    role: str,
    state: Mapping[str, Any] | None = None,
) -> Payload:
    message = str(exc)
    lowered = message.lower()
    slot_failures = _parse_slot_failures(message)

    if slot_failures:
        error_class = "all_model_slots_failed"
        error_code = "GENERATION_ALL_MODEL_SLOTS_FAILED"
        component = "model_router"
        retryable = any(bool(item.get("retryable")) for item in slot_failures)
        requires_admin_action = any(
            bool(item.get("requires_admin_action"))
            for item in slot_failures
        )
        frontend_message_key = (
            "answer_model_configuration_error"
            if requires_admin_action and not retryable
            else "answer_model_unavailable"
        )
    else:
        classified = _classify_error_message(lowered)
        error_class = classified["error_class"]
        error_code = classified["error_code"]
        component = classified["component"]
        retryable = classified["retryable"]
        requires_admin_action = classified["requires_admin_action"]
        frontend_message_key = classified["frontend_message_key"]

    return {
        "error_id": str(uuid4()),
        "timestamp": datetime.now(UTC).isoformat(),
        "layer": "generation",
        "component": component,
        "role": role,
        "error_type": type(exc).__name__,
        "error_class": error_class,
        "error_code": error_code,
        "retryable": retryable,
        "requires_admin_action": requires_admin_action,
        "frontend": {
            "message_key": frontend_message_key,
            "severity": "error",
            "retryable": retryable,
        },
        "llm_failures": slot_failures,
        "message": _truncate(message, 6000),
        "traceback": _truncate(
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            12000,
        ),
        "state_context": _safe_state_context(state or {}),
    }


def log_generation_exception(
    *,
    logger_name: str,
    message: str,
    error_event: Mapping[str, Any],
    exc: Exception,
) -> None:
    logger = logging.getLogger(logger_name)

    # Put the structured event directly in the log line so dev terminals,
    # Docker logs, CloudWatch, ECS logs, etc. can capture it without custom setup.
    logger.exception(
        "%s %s",
        message,
        json.dumps(error_event, default=str, ensure_ascii=False),
        exc_info=exc,
    )


def _parse_slot_failures(message: str) -> list[Payload]:
    failures: list[Payload] = []

    for match in _SLOT_FAILURE_RE.finditer(message):
        provider_error = " ".join(match.group("error").strip().split())
        provider_error_lowered = provider_error.lower()
        classified = _classify_error_message(provider_error_lowered)

        failures.append(
            {
                "slot": match.group("slot").strip(),
                "model": match.group("model").strip(),
                "provider": match.group("provider").strip(),
                "provider_error": _truncate(provider_error, 3000),
                "error_class": classified["error_class"],
                "error_code": classified["error_code"],
                "retryable": classified["retryable"],
                "requires_admin_action": classified["requires_admin_action"],
                "http_status": _extract_http_status(provider_error_lowered),
            }
        )

    return failures


def _classify_error_message(lowered: str) -> Payload:
    if (
        "invalid api key" in lowered
        or "unauthorized" in lowered
        or "permission denied" in lowered
        or "401" in lowered
    ):
        return {
            "component": "llm_provider",
            "error_class": "model_auth_error",
            "error_code": "GENERATION_MODEL_AUTH_ERROR",
            "frontend_message_key": "answer_model_configuration_error",
            "retryable": False,
            "requires_admin_action": True,
        }

    if "rate limit" in lowered or "quota" in lowered or "429" in lowered:
        return {
            "component": "llm_provider",
            "error_class": "model_rate_limited",
            "error_code": "GENERATION_MODEL_RATE_LIMITED",
            "frontend_message_key": "answer_model_rate_limited",
            "retryable": True,
            "requires_admin_action": False,
        }

    if "timeout" in lowered or "timed out" in lowered:
        return {
            "component": "llm_provider",
            "error_class": "model_timeout",
            "error_code": "GENERATION_MODEL_TIMEOUT",
            "frontend_message_key": "answer_model_timeout",
            "retryable": True,
            "requires_admin_action": False,
        }

    if (
        "500" in lowered
        or "internal" in lowered
        or "temporarily unavailable" in lowered
        or "503" in lowered
        or "502" in lowered
        or "504" in lowered
    ):
        return {
            "component": "llm_provider",
            "error_class": "model_provider_transient_error",
            "error_code": "GENERATION_MODEL_PROVIDER_TRANSIENT_ERROR",
            "frontend_message_key": "answer_model_temporarily_unavailable",
            "retryable": True,
            "requires_admin_action": False,
        }

    if "context length" in lowered or "token" in lowered and "limit" in lowered:
        return {
            "component": "prompt_builder",
            "error_class": "model_context_limit_error",
            "error_code": "GENERATION_MODEL_CONTEXT_LIMIT_ERROR",
            "frontend_message_key": "answer_context_too_large",
            "retryable": False,
            "requires_admin_action": False,
        }

    if "json" in lowered or "parse" in lowered:
        return {
            "component": "generation_parser",
            "error_class": "generation_parse_error",
            "error_code": "GENERATION_PARSE_ERROR",
            "frontend_message_key": "answer_generation_failed",
            "retryable": False,
            "requires_admin_action": False,
        }

    return {
        "component": "generation",
        "error_class": "generation_unknown_error",
        "error_code": "GENERATION_UNKNOWN_ERROR",
        "frontend_message_key": "answer_generation_failed",
        "retryable": False,
        "requires_admin_action": False,
    }


def _extract_http_status(lowered: str) -> int | None:
    match = re.search(r"\b(400|401|403|404|408|409|422|429|500|502|503|504)\b", lowered)
    if not match:
        return None

    return int(match.group(1))


def _safe_state_context(state: Mapping[str, Any]) -> Payload:
    turn_plan = state.get("turn_plan") or {}
    retrieval_state = state.get("retrieval_state") or {}

    return {
        "tenant_id": state.get("tenant_id"),
        "user_id": state.get("user_id"),
        "conversation_id": state.get("conversation_id"),
        "turn_id": state.get("turn_id"),
        "answer_scope": (
            turn_plan.get("answer_scope")
            if isinstance(turn_plan, Mapping)
            else None
        ),
        "turn_action": (
            turn_plan.get("action")
            if isinstance(turn_plan, Mapping)
            else None
        ),
        "needs_document_retrieval": (
            turn_plan.get("needs_document_retrieval")
            if isinstance(turn_plan, Mapping)
            else None
        ),
        "retrieval_trace_id": (
            retrieval_state.get("retrieval_trace_id")
            if isinstance(retrieval_state, Mapping)
            else None
        ),
        "has_retrieval_state": bool(retrieval_state),
        "raw_user_message_length": len(str(state.get("raw_user_message") or "")),
        "current_user_message_length": len(str(state.get("current_user_message") or "")),
    }


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value

    return value[:max_chars].rstrip() + "...[truncated]"
