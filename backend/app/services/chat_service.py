from __future__ import annotations

import asyncio
import logging
import time

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from rag.conversation import PostgresConversationRepository
from rag.conversation import extract_entity_mentions
from rag.conversation.models import EntityMention
from rag.generation.conversation_io import build_generation_state_from_turn
from rag.generation.conversation_io import persist_assistant_turn_from_generation_state
from rag.generation.service import GenerationRuntime
from rag.generation.streaming import GenerationStreamEvent
from rag.generation.streaming import GenerationStreamEventType

from app.services.redis_event_bus import RedisEventBus


logger = logging.getLogger(__name__)


#------------------------ERRORS-------------------------

class ChatServiceError(RuntimeError):
    pass


class ConversationBusyError(ChatServiceError):
    pass


#------------------------RESULTS-------------------------

@dataclass(frozen=True)
class ChatConversation:
    conversation_id: str


@dataclass(frozen=True)
class ChatSubmission:
    conversation_id: str
    user_turn_id: str


#------------------------SERVICE-------------------------

class ChatService:
    def __init__(
        self,
        *,
        repository: PostgresConversationRepository,
        runtime: GenerationRuntime,
        event_bus: RedisEventBus,
        top_k: int,
        enable_long_term_memory: bool,
        enable_semantic_memory_write: bool,
    ) -> None:
        self._repository = repository
        self._runtime = runtime
        self._event_bus = event_bus
        self._top_k = top_k
        self._enable_long_term_memory = (
            enable_long_term_memory
        )
        self._enable_semantic_memory_write = (
            enable_semantic_memory_write
        )

        self._tasks: set[
            asyncio.Task[None]
        ] = set()

        self._active_conversations: set[str] = set()
        self._submission_lock = asyncio.Lock()

    #------------------------CONVERSATION-------------------------

    async def create_conversation(
        self,
        *,
        tenant_id: str,
        user_id: str,
    ) -> ChatConversation:
        conversation_id = (
            "cnv_web_"
            f"{int(time.time())}_"
            f"{uuid4().hex[:8]}"
        )

        await self._repository.ensure_conversation(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            user_id=user_id,
            metadata={
                "source": "app_chat_api",
                "fresh_session": True,
                "top_k": self._top_k,
                "long_term_memory_enabled": (
                    self._enable_long_term_memory
                ),
                "semantic_memory_write_enabled": (
                    self._enable_semantic_memory_write
                ),
            },
        )

        return ChatConversation(
            conversation_id=conversation_id,
        )

    #------------------------SUBMIT MESSAGE-------------------------

    async def submit_message(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        message: str,
    ) -> ChatSubmission:
        query = str(
            message or ""
        ).strip()

        if not query:
            raise ValueError(
                "message cannot be empty."
            )

        await self._reserve_conversation(
            conversation_id
        )

        try:
            await self._repository.ensure_conversation(
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                user_id=user_id,
                metadata={
                    "source": "app_chat_api",
                },
            )

            raw_entities = (
                extract_entity_mentions(
                    query
                )
            )

            user_turn = (
                await self._repository.insert_turn(
                    conversation_id=conversation_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    role="user",
                    raw_text=query,
                    entities_extracted=[
                        {
                            "entity_text": (
                                item.entity_text
                            ),
                            "entity_type": (
                                item.entity_type
                            ),
                            "salience_rank": (
                                item.salience_rank
                            ),
                        }
                        for item in raw_entities
                    ],
                    metadata={
                        "source": "app_chat_api",
                        "phase": (
                            "pre_graph_user_turn"
                        ),
                    },
                )
            )

            await self._repository.insert_entities(
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                user_id=user_id,
                turn_index=user_turn.turn_index,
                entities=raw_entities,
            )

            state = (
                await build_generation_state_from_turn(
                    repository=self._repository,
                    conversation_id=conversation_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    user_turn_id=user_turn.turn_id,
                    current_user_message=query,
                )
            )

            state["session_id"] = (
                _session_id(
                    conversation_id
                )
            )

            state["top_k"] = self._top_k

            # Full tenant document collection.
            state[
                "retrieval_source_constraints"
            ] = {}

            state[
                "disable_long_term_memory"
            ] = not self._enable_long_term_memory

            state[
                "disable_memory_write"
            ] = (
                not
                self._enable_semantic_memory_write
            )

            task = asyncio.create_task(
                self._run_turn(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    query=query,
                    user_turn=user_turn,
                    raw_entities=raw_entities,
                    state=state,
                ),
                name=(
                    "chat-turn:"
                    f"{conversation_id}:"
                    f"{user_turn.turn_id}"
                ),
            )

            self._track_task(
                task
            )

            return ChatSubmission(
                conversation_id=conversation_id,
                user_turn_id=user_turn.turn_id,
            )

        except Exception:
            await self._release_conversation(
                conversation_id
            )

            raise

    #------------------------TURN EXECUTION-------------------------

    async def _run_turn(
        self,
        *,
        tenant_id: str,
        user_id: str,
        query: str,
        user_turn: Any,
        raw_entities: tuple[
            EntityMention,
            ...,
        ],
        state: dict[str, Any],
    ) -> None:
        conversation_id = str(
            state["conversation_id"]
        )

        completion_event: (
            GenerationStreamEvent
            | None
        ) = None

        last_sequence = 0

        try:
            async for event in self._runtime.stream(
                state
            ):
                last_sequence = max(
                    last_sequence,
                    event.sequence,
                )

                if (
                    event.event_type
                    == GenerationStreamEventType
                    .STREAM_COMPLETED
                ):
                    completion_event = event

                    # Completion is held until the final
                    # assistant turn is durable in Postgres.
                    continue

                await self._publish_event(
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    event=event,
                )

                if (
                    event.event_type
                    == GenerationStreamEventType
                    .STREAM_ERROR
                ):
                    return

            if completion_event is None:
                raise ChatServiceError(
                    "Generation stream ended without "
                    "stream.completed."
                )

            final_state = (
                completion_event.final_state
            )

            if final_state is None:
                raise ChatServiceError(
                    "stream.completed did not contain "
                    "the final generation state."
                )

            final_state = dict(
                final_state
            )

            #------------------------POST RETRIEVAL-------------------------

            retrieval_state = dict(
                final_state.get(
                    "retrieval_state"
                )
                or {}
            )

            stateful = dict(
                retrieval_state.get(
                    "stateful_retrieval"
                )
                or {}
            )

            resolved_query = str(
                stateful.get(
                    "resolved_query"
                )
                or retrieval_state.get(
                    "rewritten_query"
                )
                or query
            )

            query_changed = bool(
                stateful.get(
                    "query_changed"
                )
            )

            retrieval_trace_id = (
                retrieval_state.get(
                    "retrieval_trace_id"
                )
            )

            updated_user_turn = (
                await self._repository
                .update_turn_after_retrieval(
                    turn_id=user_turn.turn_id,
                    resolved_text=(
                        resolved_query
                        if query_changed
                        else None
                    ),
                    retrieval_trace_id=(
                        str(
                            retrieval_trace_id
                        )
                        if retrieval_trace_id
                        else None
                    ),
                    metadata={
                        "source": "app_chat_api",
                        "phase": (
                            "post_graph_user_turn"
                        ),
                        "stateful_retrieval": (
                            stateful
                        ),
                        "turn_plan": (
                            final_state.get(
                                "turn_plan"
                            )
                            or {}
                        ),
                    },
                )
            )

            if retrieval_trace_id:
                await (
                    self._repository
                    .attach_retrieval_trace_to_turn(
                        retrieval_trace_id=str(
                            retrieval_trace_id
                        ),
                        turn_id=user_turn.turn_id,
                    )
                )

            #------------------------RESOLVED ENTITIES-------------------------

            resolved_entities = (
                extract_entity_mentions(
                    resolved_query
                )
            )

            new_entities = _entities_not_in(
                raw_entities,
                resolved_entities,
            )

            await self._repository.insert_entities(
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                user_id=user_id,
                turn_index=(
                    updated_user_turn.turn_index
                ),
                entities=new_entities,
            )

            #------------------------ASSISTANT TURN-------------------------

            assistant_turn = (
                await
                persist_assistant_turn_from_generation_state(
                    repository=self._repository,
                    state=final_state,
                    metadata={
                        "source": "app_chat_api",
                        "retrieval_trace_id": (
                            retrieval_trace_id
                        ),
                        "turn_plan": (
                            final_state.get(
                                "turn_plan"
                            )
                            or {}
                        ),
                    },
                )
            )

            #------------------------COMPLETION-------------------------

            completed = (
                completion_event.to_dict()
            )

            completed_payload = dict(
                completed.get(
                    "payload"
                )
                or {}
            )

            completed_payload.update(
                {
                    "user_turn_id": (
                        user_turn.turn_id
                    ),
                    "assistant_turn_id": (
                        assistant_turn.turn_id
                    ),
                    "resolved_query": (
                        resolved_query
                    ),
                    "query_changed": (
                        query_changed
                    ),
                }
            )

            completed["payload"] = (
                completed_payload
            )

            await (
                self._event_bus
                .publish_chat_event(
                    tenant_id=tenant_id,
                    conversation_id=(
                        conversation_id
                    ),
                    event=completed,
                )
            )

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            logger.exception(
                "Chat turn failed: "
                "conversation_id=%s "
                "user_turn_id=%s",
                conversation_id,
                user_turn.turn_id,
            )

            error_event = GenerationStreamEvent(
                event_type=(
                    GenerationStreamEventType
                    .STREAM_ERROR
                ),
                sequence=last_sequence + 1,
                conversation_id=conversation_id,
                turn_id=user_turn.turn_id,
                payload={
                    "error_type": (
                        type(
                            exc
                        ).__name__
                    ),
                    "message": str(
                        exc
                    ),
                    "retryable": False,
                    "phase": (
                        "chat_turn_orchestration"
                    ),
                },
            )

            try:
                await self._publish_event(
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    event=error_event,
                )

            except Exception:
                logger.exception(
                    "Failed to publish chat "
                    "stream.error event."
                )

        finally:
            await self._release_conversation(
                conversation_id
            )

    #------------------------REDIS-------------------------

    async def _publish_event(
        self,
        *,
        tenant_id: str,
        conversation_id: str,
        event: GenerationStreamEvent,
    ) -> None:
        await (
            self._event_bus
            .publish_chat_event(
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                event=event.to_dict(),
            )
        )

    #------------------------CONCURRENCY-------------------------

    async def _reserve_conversation(
        self,
        conversation_id: str,
    ) -> None:
        async with self._submission_lock:
            if (
                conversation_id
                in self._active_conversations
            ):
                raise ConversationBusyError(
                    "A response is already being "
                    "generated for this conversation."
                )

            self._active_conversations.add(
                conversation_id
            )

    async def _release_conversation(
        self,
        conversation_id: str,
    ) -> None:
        async with self._submission_lock:
            self._active_conversations.discard(
                conversation_id
            )

    #------------------------TASKS-------------------------

    def _track_task(
        self,
        task: asyncio.Task[None],
    ) -> None:
        self._tasks.add(
            task
        )

        task.add_done_callback(
            self._tasks.discard
        )

    async def close(
        self,
    ) -> None:
        tasks = tuple(
            self._tasks
        )

        for task in tasks:
            task.cancel()

        if tasks:
            await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )

        self._tasks.clear()
        self._active_conversations.clear()


#------------------------HELPERS-------------------------

def _session_id(
    conversation_id: str,
) -> str:
    return (
        "ses_web_"
        f"{conversation_id}"
    )


def _entities_not_in(
    existing: tuple[
        EntityMention,
        ...,
    ],
    candidates: tuple[
        EntityMention,
        ...,
    ],
) -> tuple[
    EntityMention,
    ...,
]:
    existing_keys = {
        item.entity_text.lower()
        for item in existing
    }

    result: list[
        EntityMention
    ] = []

    for item in candidates:
        if (
            item.entity_text.lower()
            in existing_keys
        ):
            continue

        result.append(
            item
        )

    return tuple(
        result
    )


__all__ = [
    "ChatConversation",
    "ChatService",
    "ChatServiceError",
    "ChatSubmission",
    "ConversationBusyError",
]
