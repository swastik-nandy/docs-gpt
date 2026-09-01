from __future__ import annotations

from typing import Any

from rag.generation.errors import build_generation_error_event
from rag.generation.graph.state import ConversationState
from rag.generation.llm.router import ModelRouter
from rag.generation.llm.schema import ChatMessage
from rag.generation.llm.schema import ModelRequest
from rag.generation.prompt.prompt_config import PromptRegistry
from rag.generation.prompt.rag_prompt import build_rag_user_prompt
from rag.generation.prompt.rag_prompt import get_rag_system_prompt
from rag.generation.streaming import GenerationStreamEvent
from rag.generation.streaming import GenerationStreamEventType
from rag.generation.streaming import emit_stream_event


#--------------------NODE--------------------

def build_generation_node(
    *,
    model_router: ModelRouter,
    prompt_registry: PromptRegistry | None = None,
):
    registry = (
        prompt_registry
        or PromptRegistry.from_toml()
    )

    async def generation_node(
        state: ConversationState,
    ) -> dict[str, Any]:
        context_bundle = state.get(
            "context_bundle"
        ) or {}

        if not _has_grounding_evidence(
            context_bundle,
            state,
        ):
            return {
                "draft_answer": (
                    registry
                    .fallbacks
                    .insufficient_grounding
                ),
                "generation_metadata": {
                    "provider": "system",
                    "model": "no_evidence_gate",
                    "role": "generation_primary",
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                    "raw_finish_reason": (
                        "no_grounding_evidence"
                    ),
                },
            }

        request = ModelRequest(
            role="generation_primary",
            system=get_rag_system_prompt(
                registry
            ),
            messages=[
                ChatMessage(
                    role="user",
                    content=build_rag_user_prompt(
                        state,
                        prompt_registry=registry,
                    ),
                )
            ],
        )

        parts: list[str] = []

        generation_metadata: dict[
            str,
            Any,
        ] = {}

        try:
            async for event in model_router.stream(
                request
            ):
                if event.event_type == "started":
                    emit_stream_event(
                        GenerationStreamEvent(
                            event_type=(
                                GenerationStreamEventType
                                .MODEL_STARTED
                            ),
                            conversation_id=state.get(
                                "conversation_id"
                            ),
                            turn_id=state.get(
                                "turn_id"
                            ),
                            payload={
                                "provider": event.provider,
                                "model": event.model,
                                "role": event.role,
                                "slot_name": event.slot_name,
                                "fallback_used": (
                                    event.fallback_used
                                ),
                            },
                        ),
                    )

                    continue

                if event.event_type == "delta":
                    if not event.text:
                        continue

                    parts.append(
                        event.text
                    )

                    emit_stream_event(
                        GenerationStreamEvent(
                            event_type=(
                                GenerationStreamEventType
                                .ANSWER_DELTA
                            ),
                            conversation_id=state.get(
                                "conversation_id"
                            ),
                            turn_id=state.get(
                                "turn_id"
                            ),
                            text=event.text,
                            payload={
                                "provider": event.provider,
                                "model": event.model,
                            },
                        ),
                    )

                    continue

                if event.event_type == "completed":
                    generation_metadata = {
                        "provider": event.provider,
                        "model": event.model,
                        "role": event.role,
                        "slot_name": event.slot_name,
                        "fallback_used": (
                            event.fallback_used
                        ),
                        "usage": {
                            "input_tokens": (
                                event
                                .usage
                                .input_tokens
                            ),
                            "output_tokens": (
                                event
                                .usage
                                .output_tokens
                            ),
                            "total_tokens": (
                                event
                                .usage
                                .total_tokens
                            ),
                        },
                        "raw_finish_reason": (
                            event.raw_finish_reason
                        ),
                    }

                    emit_stream_event(
                        GenerationStreamEvent(
                            event_type=(
                                GenerationStreamEventType
                                .MODEL_COMPLETED
                            ),
                            conversation_id=state.get(
                                "conversation_id"
                            ),
                            turn_id=state.get(
                                "turn_id"
                            ),
                            payload=dict(
                                generation_metadata
                            ),
                        ),
                    )

        except Exception as exc:
            error_event = (
                build_generation_error_event(
                    exc,
                    role="generation_primary",
                    state=state,
                )
            )

            emit_stream_event(
                GenerationStreamEvent(
                    event_type=(
                        GenerationStreamEventType
                        .STREAM_ERROR
                    ),
                    conversation_id=state.get(
                        "conversation_id"
                    ),
                    turn_id=state.get(
                        "turn_id"
                    ),
                    payload={
                        "error_id": error_event.get(
                            "error_id"
                        ),
                        "error_code": error_event.get(
                            "error_code"
                        ),
                        "error_class": error_event.get(
                            "error_class"
                        ),
                        "retryable": error_event.get(
                            "retryable"
                        ),
                        "frontend": error_event.get(
                            "frontend"
                        ),
                    },
                ),
            )

            return {
                "draft_answer": (
                    registry
                    .fallbacks
                    .generation_unavailable
                ),
                "generation_metadata": {
                    "provider": "system",
                    "model": (
                        "generation_error_fallback"
                    ),
                    "role": "generation_primary",
                    "status": (
                        "generation_failed_after_retrieval"
                    ),
                    "frontend": error_event.get(
                        "frontend"
                    ),
                    "error_id": error_event.get(
                        "error_id"
                    ),
                    "error_class": error_event.get(
                        "error_class"
                    ),
                    "error_code": error_event.get(
                        "error_code"
                    ),
                    "retryable": error_event.get(
                        "retryable"
                    ),
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                    "raw_finish_reason": (
                        "generation_failed:"
                        f"{type(exc).__name__}"
                    ),
                    "error_details": error_event,
                },
            }

        answer = "".join(
            parts
        ).strip()

        if not answer:
            return {
                "draft_answer": (
                    registry
                    .fallbacks
                    .generation_unavailable
                ),
                "generation_metadata": {
                    "provider": "system",
                    "model": (
                        "empty_generation_fallback"
                    ),
                    "role": "generation_primary",
                    "status": (
                        "empty_generation_stream"
                    ),
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                    "raw_finish_reason": (
                        "empty_generation_stream"
                    ),
                },
            }

        return {
            "draft_answer": answer,
            "generation_metadata": (
                generation_metadata
            ),
        }

    return generation_node


#--------------------GROUNDING GATE--------------------

def _has_grounding_evidence(
    context_bundle: dict[str, Any],
    state: ConversationState,
) -> bool:
    retrieved_documents = (
        context_bundle.get(
            "retrieved_documents"
        )
        or []
    )

    retrieved_memory = (
        context_bundle.get(
            "retrieved_memory"
        )
        or []
    )

    if (
        retrieved_documents
        or retrieved_memory
    ):
        return True

    turn_plan = state.get(
        "turn_plan"
    ) or {}

    if (
        turn_plan.get(
            "allow_conversation_answer"
        )
        and not turn_plan.get(
            "needs_document_retrieval",
            True,
        )
    ):
        return True

    return False
