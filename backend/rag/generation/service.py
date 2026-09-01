from __future__ import annotations

import asyncio

from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path
from typing import Any

from rag.generation.graph.graph import GraphNode
from rag.generation.graph.graph import build_conversation_graph
from rag.generation.graph.nodes.memory_nodes import QueryEmbedder
from rag.generation.graph.state import ConversationState
from rag.generation.llm.config import ModelRegistry
from rag.generation.llm.router import ModelRouter
from rag.generation.memory.qdrant_memory_store import QdrantMemoryStore
from rag.generation.runtime_config import GenerationRuntimeConfig
from rag.generation.streaming import GenerationStreamEvent
from rag.generation.streaming import GenerationStreamEventType
from rag.generation.streaming import bind_stream_sink


#--------------------RUNTIME--------------------

class GenerationRuntime:
    def __init__(
        self,
        *,
        graph: Any,
        model_router: ModelRouter,
        memory_store: QdrantMemoryStore,
        config: GenerationRuntimeConfig,
    ) -> None:
        self.graph = graph
        self.model_router = model_router
        self.memory_store = memory_store
        self.config = config

    #--------------------NORMAL CALL--------------------

    async def invoke(
        self,
        state: ConversationState,
    ) -> ConversationState:
        final_state = await self.graph.ainvoke(
            state
        )

        return _clean_state(
            final_state
        )

    #--------------------STREAMING CALL--------------------

    async def stream(
        self,
        state: ConversationState,
    ) -> AsyncIterator[
        GenerationStreamEvent
    ]:
        queue: asyncio.Queue[
            GenerationStreamEvent
        ] = asyncio.Queue()

        sequence = 0

        def sink(
            event: GenerationStreamEvent,
        ) -> None:
            queue.put_nowait(
                event
            )

        async def execute_graph():
            with bind_stream_sink(
                sink
            ):
                return await self.graph.ainvoke(
                    state
                )

        task = asyncio.create_task(
            execute_graph()
        )

        sequence += 1

        yield GenerationStreamEvent(
            event_type=(
                GenerationStreamEventType
                .STREAM_STARTED
            ),
            sequence=sequence,
            conversation_id=state.get(
                "conversation_id"
            ),
            turn_id=state.get(
                "turn_id"
            ),
            payload={
                "status": "processing",
            },
        )

        streamed_parts: list[str] = []

        try:
            while (
                not task.done()
                or not queue.empty()
            ):
                try:
                    event = await asyncio.wait_for(
                        queue.get(),
                        timeout=0.05,
                    )

                except asyncio.TimeoutError:
                    continue

                sequence += 1

                event = event.with_sequence(
                    sequence
                )

                if (
                    event.event_type
                    == GenerationStreamEventType
                    .ANSWER_DELTA
                ):
                    streamed_parts.append(
                        event.text
                    )

                yield event

            try:
                raw_final_state = await task

            except Exception as exc:
                sequence += 1

                yield GenerationStreamEvent(
                    event_type=(
                        GenerationStreamEventType
                        .STREAM_ERROR
                    ),
                    sequence=sequence,
                    conversation_id=state.get(
                        "conversation_id"
                    ),
                    turn_id=state.get(
                        "turn_id"
                    ),
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
                    },
                )

                return

            final_state = _clean_state(
                raw_final_state
            )

            final_answer = str(
                final_state.get(
                    "final_answer"
                )
                or final_state.get(
                    "draft_answer"
                )
                or ""
            ).strip()

            streamed_answer = "".join(
                streamed_parts
            ).strip()

            if (
                final_answer
                and final_answer
                != streamed_answer
            ):
                sequence += 1

                yield GenerationStreamEvent(
                    event_type=(
                        GenerationStreamEventType
                        .ANSWER_REPLACE
                    ),
                    sequence=sequence,
                    conversation_id=state.get(
                        "conversation_id"
                    ),
                    turn_id=state.get(
                        "turn_id"
                    ),
                    text=final_answer,
                    payload={
                        "reason": (
                            "post_generation_finalization"
                        ),
                    },
                )

            retrieval_state = (
                final_state.get(
                    "retrieval_state"
                )
                or {}
            )

            sequence += 1

            yield GenerationStreamEvent(
                event_type=(
                    GenerationStreamEventType
                    .STREAM_COMPLETED
                ),
                sequence=sequence,
                conversation_id=state.get(
                    "conversation_id"
                ),
                turn_id=state.get(
                    "turn_id"
                ),
                payload={
                    "status": "completed",
                    "generation_metadata": (
                        final_state.get(
                            "generation_metadata"
                        )
                        or {}
                    ),
                    "grounding_result": (
                        final_state.get(
                            "grounding_result"
                        )
                        or {}
                    ),
                    "citations": (
                        retrieval_state.get(
                            "citations"
                        )
                        or []
                    ),
                    "retrieval_trace_id": (
                        retrieval_state.get(
                            "retrieval_trace_id"
                        )
                    ),
                },
                final_state=final_state,
            )

        finally:
            if not task.done():
                task.cancel()

                with suppress(
                    asyncio.CancelledError
                ):
                    await task

    #--------------------LIFECYCLE--------------------

    async def close(
        self,
    ) -> None:
        await self.model_router.close()
        await self.memory_store.close()


#--------------------BUILDING--------------------

async def build_generation_runtime(
    *,
    retrieval_node: GraphNode,
    embedder: QueryEmbedder,
    runtime_config_path: str | Path | None = None,
) -> GenerationRuntime:
    runtime_config = (
        GenerationRuntimeConfig.from_toml(
            runtime_config_path
        )
    )

    registry = ModelRegistry.from_toml(
        runtime_config.models.registry_path
    )

    model_router = (
        ModelRouter.from_registry(
            registry
        )
    )

    qdrant_config = (
        runtime_config.memory.qdrant
    )

    memory_store = QdrantMemoryStore(
        url=qdrant_config.url,
        collection_name=(
            qdrant_config.collection_name
        ),
        vector_size=(
            qdrant_config.vector_size
        ),
        timeout_seconds=(
            qdrant_config.timeout_seconds
        ),
    )

    await memory_store.setup()

    graph = build_conversation_graph(
        memory_store=memory_store,
        embedder=embedder,
        model_router=model_router,
        retrieval_node=retrieval_node,
    )

    return GenerationRuntime(
        graph=graph,
        model_router=model_router,
        memory_store=memory_store,
        config=runtime_config,
    )


#--------------------STATE--------------------

def _clean_state(
    value: Any,
) -> ConversationState:
    if not isinstance(
        value,
        dict,
    ):
        raise TypeError(
            "Generation graph returned "
            "a non-dict state."
        )

    return dict(
        value
    )
