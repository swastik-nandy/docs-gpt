from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from rag.generation.graph.nodes.context_nodes import assemble_context_node
from rag.generation.graph.nodes.generation_nodes import build_generation_node
from rag.generation.graph.nodes.grounding_nodes import basic_grounding_node
from rag.generation.graph.nodes.memory_nodes import QueryEmbedder, build_memory_retrieval_node
from rag.generation.graph.nodes.memory_write_nodes import build_memory_write_node
from rag.generation.graph.nodes.turn_planning_node import build_turn_planning_node
from rag.generation.graph.state import ConversationState
from rag.generation.llm.router import ModelRouter
from rag.generation.memory.qdrant_memory_store import QdrantMemoryStore
from rag.generation.prompt.prompt_config import PromptRegistry


GraphNode = Callable[[ConversationState], Awaitable[dict[str, Any]]]


def build_conversation_graph(
    *,
    memory_store: QdrantMemoryStore,
    embedder: QueryEmbedder,
    model_router: ModelRouter,
    retrieval_node: GraphNode,
    prompt_registry: PromptRegistry | None = None,
):
    registry = prompt_registry or PromptRegistry.from_toml()

    builder = StateGraph(ConversationState)

    builder.add_node(
        "turn_planning",
        build_turn_planning_node(
            model_router=model_router,
            prompt_registry=registry,
        ),
    )
    builder.add_node("retrieval", retrieval_node)
    builder.add_node(
        "memory_retrieval",
        build_memory_retrieval_node(memory_store=memory_store, embedder=embedder),
    )
    builder.add_node("context_assembly", assemble_context_node)
    builder.add_node(
        "generation",
        build_generation_node(
            model_router=model_router,
            prompt_registry=registry,
        ),
    )
    builder.add_node("grounding", basic_grounding_node)
    builder.add_node(
        "memory_write",
        build_memory_write_node(
            memory_store=memory_store,
            embedder=embedder,
            model_router=model_router,
            prompt_registry=registry,
        ),
    )
    builder.add_node("response_formatting", _build_response_formatting_node(registry))

    builder.add_edge(START, "turn_planning")
    builder.add_conditional_edges(
        "turn_planning",
        _route_after_turn_planning,
        {
            "retrieve": "retrieval",
            "skip_retrieval": "memory_retrieval",
        },
    )
    builder.add_edge("retrieval", "memory_retrieval")
    builder.add_edge("memory_retrieval", "context_assembly")
    builder.add_edge("context_assembly", "generation")
    builder.add_edge("generation", "grounding")
    builder.add_edge("grounding", "memory_write")
    builder.add_edge("memory_write", "response_formatting")
    builder.add_edge("response_formatting", END)

    return builder.compile()


def _route_after_turn_planning(state: ConversationState) -> str:
    turn_plan = state.get("turn_plan") or {}
    if turn_plan.get("needs_document_retrieval", True):
        return "retrieve"

    return "skip_retrieval"


def _build_response_formatting_node(prompt_registry: PromptRegistry):
    async def response_formatting_node(state: ConversationState) -> dict:
        answer = state.get("draft_answer", "").strip()
        grounding_result = state.get("grounding_result") or {}

        if not answer:
            answer = prompt_registry.fallbacks.insufficient_grounding

        if grounding_result and not grounding_result.get("passed", True):
            answer = prompt_registry.fallbacks.insufficient_grounding

        return {
            "final_answer": answer,
            "grounding_result": grounding_result,
        }

    return response_formatting_node
