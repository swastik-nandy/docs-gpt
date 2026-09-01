from __future__ import annotations

import asyncio
from pathlib import Path

import rtoml

from rag.generation.memory.qdrant_memory_store import QdrantMemoryStore
from rag.generation.memory.schema import MemoryCandidate


async def main() -> None:
    runtime_config_path = Path("rag/generation/config/runtime.toml")
    raw = rtoml.load(runtime_config_path)

    qdrant = raw["memory"]["qdrant"]
    vector_size = int(qdrant["vector_size"])

    store = QdrantMemoryStore(
        url=str(qdrant["url"]),
        collection_name=str(qdrant["collection_name"]),
        vector_size=vector_size,
        timeout_seconds=float(qdrant.get("timeout_seconds", 10.0)),
    )

    await store.setup()

    vector = [0.0] * vector_size
    vector[0] = 1.0

    memory_id = await store.put(
        MemoryCandidate(
            tenant_id="smoke_tenant",
            user_id="smoke_user",
            text="User prefers uv and python3 for Python projects.",
            vector=vector,
            fact_type="preference",
            conversation_id="smoke_conversation",
            source_turn_id="smoke_turn",
            confidence=0.95,
            metadata={"test": True},
        )
    )

    matches = await store.find_similar(
        tenant_id="smoke_tenant",
        user_id="smoke_user",
        query_vector=vector,
        limit=3,
        min_score=0.1,
    )

    await store.close()

    print(f"inserted_memory_id={memory_id}")
    print(f"matches={len(matches)}")

    for match in matches:
        print(
            f"match id={match.memory_id} score={match.score:.4f} "
            f"type={match.fact_type} text={match.text}"
        )

    if not matches:
        raise SystemExit("Memory smoke test failed: no matches returned.")

    print("Memory smoke test passed.")


if __name__ == "__main__":
    asyncio.run(main())
