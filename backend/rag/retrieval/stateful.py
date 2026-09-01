from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag.conversation.config import load_conversation_runtime_config
from rag.conversation.entities import EntityExtractor
from rag.conversation.models import (
    ConversationContext,
    ReferenceResolutionResult,
)
from rag.conversation.repository import PostgresConversationRepository
from rag.conversation.resolver import ConversationReferenceResolver
from rag.retrieval.contracts.request import ChatTurn, RetrievalRequest


Payload = dict[str, Any]


@dataclass(frozen=True)
class StatefulRetrievalRequest:
    """
    Result of preparing a retrieval request with conversation state.

    original_request:
        The raw request created from the user's literal message.

    request:
        The request that should be sent into retrieval. Its query may be
        rewritten into a standalone form.

    resolution:
        Details of the generic reference-resolution step.

    context:
        Postgres-backed conversation context used for the rewrite.
    """

    original_request: RetrievalRequest
    request: RetrievalRequest
    resolution: ReferenceResolutionResult
    context: ConversationContext | None = None

    @property
    def query_changed(self) -> bool:
        return self.resolution.changed



def _build_default_reference_resolver() -> ConversationReferenceResolver:
    runtime_config = load_conversation_runtime_config()

    return ConversationReferenceResolver(
        config=runtime_config.reference_resolution,
        entity_extractor=EntityExtractor(
            config=runtime_config.entity_extraction,
        ),
    )


class StatefulRetrievalPreprocessor:
    """
    Conversation-aware retrieval request preprocessor.

    This layer is intentionally generic:
    - no document-specific vocabulary
    - no ECS/AWS/product-specific rules
    - no Qdrant calls
    - no context-packing behavior

    It only converts a weak conversational query into a standalone retrieval query
    using recent conversation entities and turns from Postgres.
    """

    def __init__(
        self,
        *,
        repository: PostgresConversationRepository | None = None,
        resolver: ConversationReferenceResolver | None = None,
        turn_limit: int = 8,
        entity_limit: int = 8,
    ) -> None:
        self.repository = repository
        self.resolver = resolver or _build_default_reference_resolver()
        self.turn_limit = turn_limit
        self.entity_limit = entity_limit

    async def prepare(
        self,
        request: RetrievalRequest,
    ) -> StatefulRetrievalRequest:
        request.validate()

        if not request.conversation_id:
            resolution = ReferenceResolutionResult(
                raw_query=request.query,
                resolved_query=request.query,
                needed_resolution=False,
                strategy="no_conversation_id",
                confidence=1.0,
            )
            resolution.validate()

            return StatefulRetrievalRequest(
                original_request=request,
                request=_with_stateful_metadata(
                    request=request,
                    resolution=resolution,
                    context=None,
                ),
                resolution=resolution,
                context=None,
            )

        repository = self.repository
        owns_repository = repository is None

        if repository is None:
            repository = await PostgresConversationRepository.connect()

        try:
            context = await repository.fetch_context(
                conversation_id=request.conversation_id,
                tenant_id=request.tenant.tenant_id,
                user_id=request.tenant.user_id,
                turn_limit=self.turn_limit,
                entity_limit=self.entity_limit,
            )

            resolution = self.resolver.resolve(
                raw_query=request.query,
                context=context,
            )
            resolution.validate()

            prepared_request = _build_resolved_request(
                request=request,
                resolution=resolution,
                context=context,
            )

            return StatefulRetrievalRequest(
                original_request=request,
                request=prepared_request,
                resolution=resolution,
                context=context,
            )

        finally:
            if owns_repository:
                await repository.close()


async def prepare_stateful_retrieval_request(
    request: RetrievalRequest,
    *,
    repository: PostgresConversationRepository | None = None,
    resolver: ConversationReferenceResolver | None = None,
    turn_limit: int = 8,
    entity_limit: int = 8,
) -> StatefulRetrievalRequest:
    return await StatefulRetrievalPreprocessor(
        repository=repository,
        resolver=resolver,
        turn_limit=turn_limit,
        entity_limit=entity_limit,
    ).prepare(request)


def _build_resolved_request(
    *,
    request: RetrievalRequest,
    resolution: ReferenceResolutionResult,
    context: ConversationContext,
) -> RetrievalRequest:
    chat_history = _chat_history_from_context(context) or request.chat_history

    metadata = _stateful_metadata(
        request=request,
        resolution=resolution,
        context=context,
    )

    return RetrievalRequest.create(
        query=resolution.resolved_query,
        tenant=request.tenant,
        mode=request.mode,
        top_k=request.top_k,
        memory_policy=request.memory_policy,
        query_processing=request.query_processing,
        request_id=request.request_id,
        conversation_id=request.conversation_id,
        source_constraints=request.source_constraints,
        chat_history=chat_history,
        metadata=metadata,
    )


def _with_stateful_metadata(
    *,
    request: RetrievalRequest,
    resolution: ReferenceResolutionResult,
    context: ConversationContext | None,
) -> RetrievalRequest:
    return RetrievalRequest.create(
        query=resolution.resolved_query,
        tenant=request.tenant,
        mode=request.mode,
        top_k=request.top_k,
        memory_policy=request.memory_policy,
        query_processing=request.query_processing,
        request_id=request.request_id,
        conversation_id=request.conversation_id,
        source_constraints=request.source_constraints,
        chat_history=request.chat_history,
        metadata=_stateful_metadata(
            request=request,
            resolution=resolution,
            context=context,
        ),
    )


def _chat_history_from_context(
    context: ConversationContext,
) -> tuple[ChatTurn, ...]:
    turns: list[ChatTurn] = []

    for turn in context.recent_turns:
        turns.append(
            ChatTurn(
                role=turn.role,
                content=turn.text_for_context,
                turn_id=turn.turn_id,
                created_at=None,
                metadata={
                    "source": "postgres_conversation_turns",
                    "turn_index": turn.turn_index,
                    "raw_text": turn.raw_text,
                    "resolved_text": turn.resolved_text,
                },
            )
        )

    return tuple(turn.normalized() for turn in turns)


def _stateful_metadata(
    *,
    request: RetrievalRequest,
    resolution: ReferenceResolutionResult,
    context: ConversationContext | None,
) -> Payload:
    metadata = dict(request.metadata)

    metadata["stateful_retrieval"] = {
        "enabled": True,
        "raw_query": resolution.raw_query,
        "resolved_query": resolution.resolved_query,
        "query_changed": resolution.changed,
        "needed_resolution": resolution.needed_resolution,
        "resolved_entity": resolution.resolved_entity,
        "strategy": resolution.strategy,
        "confidence": resolution.confidence,
        "metadata": dict(resolution.metadata),
        "conversation_context": (
            {
                "conversation_id": context.conversation_id,
                "recent_turn_count": len(context.recent_turns),
                "recent_entity_count": len(context.recent_entities),
                "has_summary": context.has_summary,
                "working_memory_count": len(context.working_memory),
            }
            if context is not None
            else None
        ),
    }

    return metadata
