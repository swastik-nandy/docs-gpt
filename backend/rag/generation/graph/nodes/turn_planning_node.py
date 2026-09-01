from __future__ import annotations

import os

from typing import Any

from rag.generation.graph.state import ConversationState
from rag.generation.llm.router import ModelRouter
from rag.generation.llm.schema import ChatMessage, ModelRequest
from rag.generation.prompt.prompt_config import PromptRegistry
from rag.generation.turn_planning.schema import (
    deterministic_turn_plan,
    fallback_turn_plan,
    parse_turn_plan_response,
)


def build_turn_planning_node(
    *,
    model_router: ModelRouter,
    prompt_registry: PromptRegistry | None = None,
):
    registry = prompt_registry or PromptRegistry.from_toml()

    async def turn_planning_node(state: ConversationState) -> dict[str, Any]:
        current_user_message = str(
            state.get("raw_user_message")
            or state.get("current_user_message")
            or ""
        ).strip()

        if not current_user_message:
            current_user_message = ""

        recent_turns = list(state.get("recent_turns") or [])
        recent_user_message = _last_message_by_role(recent_turns, "user")
        recent_assistant_message = _last_message_by_role(recent_turns, "assistant")

        deterministic = deterministic_turn_plan(
            current_user_message=current_user_message,
            recent_user_message=recent_user_message,
            recent_assistant_message=recent_assistant_message,
        )

        if deterministic is not None:
            payload = deterministic.to_state_payload()
            payload["metadata"] = {
                **dict(payload.get("metadata") or {}),
                "planner_provider": "system",
                "planner_model": "deterministic_chat_mode",
                "planner_fallback_used": False,
                "recent_user_message_available": recent_user_message is not None,
                "recent_assistant_message_available": recent_assistant_message is not None,
            }

            return {
                "turn_plan": payload,
                "retrieval_query": payload.get("retrieval_query"),
                "raw_user_message": current_user_message,
            }

        fast_demo = (
            os.getenv(
                "DOCS_GPT_FAST_DEMO",
                "",
            )
            .strip()
            .lower()
            in {
                "1",
                "true",
                "yes",
                "on",
            }
        )

        if fast_demo:
            payload = {
                "action": "document_question",
                "needs_document_retrieval": True,
                "retrieval_query": current_user_message,
                "answer_scope": "knowledge_base",
                "allow_conversation_answer": False,
                "needs_clarification": False,
                "uses_previous_turns": bool(
                    recent_user_message
                    or recent_assistant_message
                ),
                "confidence": 0.75,
                "reason": (
                    "Fast demo mode defaults ambiguous "
                    "turns to document retrieval."
                ),
                "metadata": {
                    "planner_provider": "system",
                    "planner_model": "fast_demo_router",
                    "planner_fallback_used": False,
                    "fast_demo": True,
                    "recent_user_message_available": (
                        recent_user_message is not None
                    ),
                    "recent_assistant_message_available": (
                        recent_assistant_message is not None
                    ),
                },
            }

            return {
                "turn_plan": payload,
                "retrieval_query": (
                    current_user_message
                    or None
                ),
                "raw_user_message": (
                    current_user_message
                ),
            }

        rendered_recent_turns = _render_recent_turns(recent_turns)

        try:
            response = await model_router.generate(
                ModelRequest(
                    role="turn_planning",
                    system=registry.turn_planning.system,
                    messages=[
                        ChatMessage(
                            role="user",
                            content=registry.turn_planning.user_template.format(
                                current_user_message=current_user_message,
                                recent_turns=rendered_recent_turns,
                                recent_user_message=recent_user_message or registry.rendering.empty_value,
                                recent_assistant_message=recent_assistant_message or registry.rendering.empty_value,
                                conversation_summary=state.get("conversation_summary") or registry.rendering.empty_value,
                            ),
                        )
                    ],
                    response_mime_type="application/json",
                )
            )
            plan = parse_turn_plan_response(
                response.text,
                current_user_message=current_user_message,
            )
            planner_metadata = {
                "planner_provider": response.provider,
                "planner_model": response.model,
                "planner_finish_reason": response.raw_finish_reason,
                "planner_fallback_used": False,
            }
        except Exception as exc:
            plan = fallback_turn_plan(
                current_user_message=current_user_message,
                recent_user_message=recent_user_message,
                recent_assistant_message=recent_assistant_message,
            )
            planner_metadata = {
                "planner_provider": "system",
                "planner_model": "fallback_turn_plan",
                "planner_fallback_used": True,
                "planner_error": f"{type(exc).__name__}: {exc}",
            }

        payload = plan.to_state_payload()
        payload["metadata"] = {
            **dict(payload.get("metadata") or {}),
            **planner_metadata,
            "recent_user_message_available": recent_user_message is not None,
            "recent_assistant_message_available": recent_assistant_message is not None,
        }

        retrieval_query = (
            str(payload.get("retrieval_query") or "").strip()
            if payload.get("needs_document_retrieval", True)
            else ""
        )

        if payload.get("needs_document_retrieval", True) and not retrieval_query:
            retrieval_query = current_user_message

        return {
            "turn_plan": payload,
            "retrieval_query": retrieval_query or None,
            "raw_user_message": current_user_message,
        }

    return turn_planning_node


def _last_message_by_role(messages: list[ChatMessage], role: str) -> str | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.role == role and message.content.strip():
            return message.content.strip()

    return None


def _render_recent_turns(messages: list[ChatMessage], *, max_turns: int = 8) -> str:
    if not messages:
        return "None"

    rendered: list[str] = []
    for index, message in enumerate(messages[-max_turns:], start=1):
        content = " ".join(message.content.strip().split())
        if len(content) > 900:
            content = content[:900] + "..."
        rendered.append(f"{index}. {message.role}: {content}")

    return "\n".join(rendered)
