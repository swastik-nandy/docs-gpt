from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
from pathlib import Path

import rtoml

from rag.generation.graph.graph import build_conversation_graph
from rag.generation.llm.schema import ModelResponse
from rag.generation.memory.qdrant_memory_store import QdrantMemoryStore


class FakeModelRouter:
    async def generate(self, request):
        if request.role == "memory_extraction":
            text = '{"facts":[{"text":"User prefers uv and python3 for Python projects.","fact_type":"preference","confidence":0.95,"metadata":{"source_test":"fake_graph"}}]}'
        else:
            text = "Test answer generated from the fake model router."

        return ModelResponse(
            text=text,
            provider="fake",
            model="fake-model",
            role=request.role,
        )


class FakeEmbedder:
    def __init__(self, vector_size: int) -> None:
        self.vector_size = vector_size

    async def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.vector_size
        seed = sum(ord(char) for char in text) % self.vector_size
        vector[seed] = 1.0
        return vector


async def fake_retrieval_node(state):
    return {
        "retrieval_state": {
            "query": state["current_user_message"],
            "selected_context": [],
            "citations": [],
            "warnings": [],
        }
    }


async def main() -> None:
    raw = rtoml.load(Path("rag/generation/config/runtime.toml"))
    qdrant = raw["memory"]["qdrant"]
    vector_size = int(qdrant["vector_size"])

    memory_store = QdrantMemoryStore(
        url=str(qdrant["url"]),
        collection_name=str(qdrant["collection_name"]),
        vector_size=vector_size,
        timeout_seconds=float(qdrant.get("timeout_seconds", 10.0)),
    )
    await memory_store.setup()

    graph = build_conversation_graph(
        memory_store=memory_store,
        embedder=FakeEmbedder(vector_size),
        model_router=FakeModelRouter(),
        retrieval_node=fake_retrieval_node,
    )

    result = await graph.ainvoke(
        {
            "tenant_id": "smoke_tenant",
            "user_id": "smoke_user",
            "conversation_id": "smoke_conversation",
            "turn_id": "smoke_turn_full_graph",
            "current_user_message": "Remember that I prefer uv and python3 for Python projects.",
        }
    )

    await memory_store.close()

    print(result)

    if not result.get("final_answer"):
        raise SystemExit("Full graph smoke test failed: no final_answer.")

    if not result.get("written_memory_ids") and not result.get("skipped_memory_ids"):
        raise SystemExit("Full graph smoke test failed: memory write did not happen.")

    print("Full graph smoke test passed.")


if __name__ == "__main__":
    asyncio.run(main())
