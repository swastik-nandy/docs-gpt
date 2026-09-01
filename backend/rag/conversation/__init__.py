from __future__ import annotations

from rag.conversation.config import ConversationDbConfig
try:
    from rag.conversation.entities import extract_entity_mentions
except ImportError:
    def extract_entity_mentions(*args, **kwargs):
        return []
from rag.conversation.models import (
    ConversationContext,
    ConversationSummary,
    ConversationTurn,
    EntityMention,
    ReferenceResolutionResult,
    WorkingMemoryItem,
)
from rag.conversation.repository import PostgresConversationRepository
from rag.conversation.resolver import ConversationReferenceResolver

__all__ = [
    "ConversationContext",
    "ConversationDbConfig",
    "ConversationReferenceResolver",
    "ConversationSummary",
    "ConversationTurn",
    "EntityMention",
    "PostgresConversationRepository",
    "ReferenceResolutionResult",
    "WorkingMemoryItem",
    "extract_entity_mentions",
]
