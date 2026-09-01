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

from rag.generation.graph.nodes.memory_write_nodes import build_memory_write_node
from rag.generation.llm.schema import ModelResponse
from rag.generation.memory.qdrant_memory_store import QdrantMemoryStore


class FakeModelRouter:
    async def generate(self, request):
        return ModelResponse(
            text='{"facts":[{"text":"User prefers uv and python3 for Python projects.","fact_type":"preference","confidence":0.95,"metadata":{"source_test":"fake_router"}}]}',
            provider="fake",
            model="fake-memory-extractor",
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

    node = build_memory_write_node(
        memory_store=memory_store,
        embedder=FakeEmbedder(vector_size),
        model_router=FakeModelRouter(),
    )

    state = {
        "tenant_id": "smoke_tenant",
        "user_id": "smoke_user",
        "conversation_id": "smoke_conversation",
        "turn_id": "smoke_turn_memory_write",
        "current_user_message": "Remember that I prefer uv and python3 for Python projects.",
        "draft_answer": "Got it.",
    }

    result = await node(state)
    print(result)

    written = result.get("written_memory_ids", [])
    skipped = result.get("skipped_memory_ids", [])

    if not written and not skipped:
        raise SystemExit("Memory write smoke test failed: neither written nor skipped duplicate.")

    await memory_store.close()
    print("Memory write smoke test passed.")


if __name__ == "__main__":
    asyncio.run(main())
