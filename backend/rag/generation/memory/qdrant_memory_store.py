from __future__ import annotations

from datetime import datetime
from typing import Iterable
from uuid import uuid4

from qdrant_client import AsyncQdrantClient, models

from rag.generation.memory.schema import MemoryCandidate, RetrievedMemory, utc_now_iso


class QdrantMemoryStore:
    def __init__(
        self,
        *,
        url: str,
        collection_name: str,
        vector_size: int,
        api_key: str | None = None,
        distance: models.Distance = models.Distance.COSINE,
        timeout_seconds: float = 10.0,
    ) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size must be positive.")

        self._client = AsyncQdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout_seconds,
        )
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._distance = distance

    async def setup(self) -> None:
        exists = await self._client.collection_exists(self._collection_name)
        if exists:
            return

        await self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(
                size=self._vector_size,
                distance=self._distance,
            ),
        )

        await self._create_payload_index("tenant_id")
        await self._create_payload_index("user_id")
        await self._create_payload_index("fact_type")
        await self._create_payload_index("conversation_id")

    async def put(self, candidate: MemoryCandidate) -> str:
        self._validate_vector(candidate.vector)

        memory_id = str(uuid4())
        now = utc_now_iso()

        payload = {
            "schema_version": 1,
            "memory_id": memory_id,
            "tenant_id": candidate.tenant_id,
            "user_id": candidate.user_id,
            "conversation_id": candidate.conversation_id,
            "source_turn_id": candidate.source_turn_id,
            "text": candidate.text,
            "fact_type": candidate.fact_type,
            "confidence": candidate.confidence,
            "created_at": now,
            "last_confirmed_at": now,
            "metadata": dict(candidate.metadata),
        }

        await self._client.upsert(
            collection_name=self._collection_name,
            points=[
                models.PointStruct(
                    id=memory_id,
                    vector=candidate.vector,
                    payload=payload,
                )
            ],
            wait=True,
        )

        return memory_id

    async def find_similar(
        self,
        *,
        tenant_id: str,
        user_id: str,
        query_vector: list[float],
        limit: int = 8,
        min_score: float | None = None,
        fact_types: Iterable[str] | None = None,
    ) -> list[RetrievedMemory]:
        self._validate_vector(query_vector)

        must_conditions: list[models.Condition] = [
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=tenant_id),
            ),
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id),
            ),
        ]

        fact_type_list = list(fact_types or [])
        if fact_type_list:
            must_conditions.append(
                models.FieldCondition(
                    key="fact_type",
                    match=models.MatchAny(any=fact_type_list),
                )
            )

        response = await self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            query_filter=models.Filter(must=must_conditions),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        memories: list[RetrievedMemory] = []
        for point in response.points:
            if min_score is not None and point.score < min_score:
                continue

            payload = point.payload or {}
            memories.append(
                self._payload_to_memory(
                    point_id=str(point.id),
                    score=float(point.score),
                    payload=payload,
                )
            )

        return memories

    async def find_duplicate(
        self,
        *,
        tenant_id: str,
        user_id: str,
        query_vector: list[float],
        fact_type: str | None = None,
        min_score: float = 0.92,
    ) -> RetrievedMemory | None:
        fact_types = [fact_type] if fact_type else None

        matches = await self.find_similar(
            tenant_id=tenant_id,
            user_id=user_id,
            query_vector=query_vector,
            limit=1,
            min_score=min_score,
            fact_types=fact_types,
        )

        return matches[0] if matches else None

    async def close(self) -> None:
        await self._client.close()

    async def _create_payload_index(self, field_name: str) -> None:
        try:
            await self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        except Exception:
            return

    def _validate_vector(self, vector: list[float]) -> None:
        if len(vector) != self._vector_size:
            raise ValueError(
                f"Expected vector size {self._vector_size}, got {len(vector)}."
            )

    @staticmethod
    def _payload_to_memory(
        *,
        point_id: str,
        score: float,
        payload: dict,
    ) -> RetrievedMemory:
        return RetrievedMemory(
            memory_id=str(payload.get("memory_id") or point_id),
            tenant_id=str(payload["tenant_id"]),
            user_id=str(payload["user_id"]),
            text=str(payload["text"]),
            fact_type=str(payload["fact_type"]),
            score=score,
            confidence=float(payload.get("confidence", 0.0)),
            conversation_id=payload.get("conversation_id"),
            source_turn_id=payload.get("source_turn_id"),
            created_at=_parse_dt(payload.get("created_at")),
            last_confirmed_at=_parse_dt(payload.get("last_confirmed_at")),
            metadata=payload.get("metadata") or {},
        )


def _parse_dt(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
