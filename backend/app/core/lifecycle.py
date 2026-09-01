from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag.conversation import PostgresConversationRepository
from rag.generation.embedder import RetrievalDenseGenerationEmbedder
from rag.generation.retrieval_node import build_generation_retrieval_node
from rag.generation.service import build_generation_runtime

from app.core.settings import get_settings
from app.services.chat_service import ChatService
from app.services.redis_event_bus import RedisEventBus


#------------------------APPLICATION LIFECYCLE-------------------------

@asynccontextmanager
async def app_lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    settings = get_settings()

    event_bus = RedisEventBus(
        settings.redis
    )

    repository = None
    runtime = None
    chat_service = None

    try:
        await event_bus.start()

        repository = (
            await
            PostgresConversationRepository
            .connect()
        )

        runtime = (
            await build_generation_runtime(
                retrieval_node=(
                    build_generation_retrieval_node(
                        top_k=(
                            settings.chat_top_k
                        ),
                    )
                ),
                embedder=(
                    RetrievalDenseGenerationEmbedder()
                ),
            )
        )

        chat_service = ChatService(
            repository=repository,
            runtime=runtime,
            event_bus=event_bus,
            top_k=settings.chat_top_k,
            enable_long_term_memory=(
                settings
                .chat_enable_long_term_memory
            ),
            enable_semantic_memory_write=(
                settings
                .chat_enable_semantic_memory_write
            ),
        )

        app.state.redis_event_bus = (
            event_bus
        )

        app.state.conversation_repository = (
            repository
        )

        app.state.generation_runtime = (
            runtime
        )

        app.state.chat_service = (
            chat_service
        )

        yield

    finally:
        if chat_service is not None:
            await chat_service.close()

        if runtime is not None:
            await runtime.close()

        if repository is not None:
            await repository.close()

        await event_bus.close()


__all__ = [
    "app_lifespan",
]
