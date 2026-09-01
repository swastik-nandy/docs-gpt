from __future__ import annotations

from typing import Protocol

from rag.generation.graph.state import ConversationState
from rag.generation.memory.memory_gate import should_retrieve_long_term_memory
from rag.generation.memory.qdrant_memory_store import QdrantMemoryStore


class QueryEmbedder(Protocol):
    async def embed_query(self, text: str) -> list[float]:
        ...


def build_memory_retrieval_node(
    *,
    memory_store: QdrantMemoryStore,
    embedder: QueryEmbedder,
    limit: int = 8,
    min_score: float = 0.45,
):
    async def retrieve_memory_node(state: ConversationState) -> dict:
        if state.get("disable_long_term_memory"):
            return {"retrieved_memory": []}

        user_message = state["current_user_message"]

        if not should_retrieve_long_term_memory(user_message):
            return {"retrieved_memory": []}

        query_vector = await embedder.embed_query(user_message)

        memories = await memory_store.find_similar(
            tenant_id=state["tenant_id"],
            user_id=state["user_id"],
            query_vector=query_vector,
            limit=limit,
            min_score=min_score,
        )

        return {"retrieved_memory": memories}

    return retrieve_memory_node
