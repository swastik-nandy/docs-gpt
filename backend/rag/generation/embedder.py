from __future__ import annotations

from rag.retrieval.contracts.query import NormalizedQuery
from rag.retrieval.embeddings import RetrievalQueryEmbedder


class RetrievalDenseGenerationEmbedder:
    """
    Adapter from retrieval's query embedder to generation memory's QueryEmbedder protocol.

    Generation memory wants embed_query(text) -> list[float].
    RetrievalQueryEmbedder returns a full QueryEmbedding object.
    """

    def __init__(self, embedder: RetrievalQueryEmbedder | None = None) -> None:
        self.embedder = embedder or RetrievalQueryEmbedder()

    async def embed_query(self, text: str) -> list[float]:
        query = " ".join(text.strip().split())
        if not query:
            raise ValueError("text cannot be empty.")

        embedding = await self.embedder.embed(
            NormalizedQuery(
                original_query=query,
                normalized_query=query,
            )
        )

        if embedding.dense_vector is None:
            raise ValueError("retrieval embedder did not return a dense vector.")

        return [float(value) for value in embedding.dense_vector]
