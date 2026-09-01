from __future__ import annotations

from dataclasses import asdict

from rag.generation.graph.state import ConversationState


async def assemble_context_node(state: ConversationState) -> dict:
    retrieval_state = state.get("retrieval_state") or {}
    retrieved_memory = state.get("retrieved_memory") or []

    context_bundle = {
        "retrieved_documents": retrieval_state.get("selected_context", []),
        "retrieved_memory": [asdict(memory) for memory in retrieved_memory],
        "working_memory": [],
        "conversation_summary": state.get("conversation_summary"),
        "provenance_map": {},
    }

    return {"context_bundle": context_bundle}
