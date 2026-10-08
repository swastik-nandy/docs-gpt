from __future__ import annotations

from environment import load_environment
import os

load_environment()

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any


from qdrant_client import AsyncQdrantClient
from qdrant_client import models


# --------- ENVIRONMENT ------------------------


# --------- COLLECTION ------------------------

COLLECTION_NAME = (
    "qwen3_embeddings_0.6b"
)

VECTOR_SIZE = 1024

DISTANCE = (
    models.Distance.COSINE
)


# --------- RESULT ------------------------

@dataclass(frozen=True)
class QdrantUpsertResult:
    point_count: int
    elapsed_seconds: float


# --------- CLIENT ------------------------

class QdrantCloud:
    def __init__(
        self,
    ) -> None:
        endpoint = (
            os.getenv(
                "QDRANT_CLOUD_CLUSTER_ENDPOINT"
            )
            or ""
        ).strip().rstrip("/")

        api_key = (
            os.getenv(
                "QDRANT_CLOUD_API_KEY"
            )
            or ""
        ).strip()

        if not endpoint:
            raise RuntimeError(
                "Missing "
                "QDRANT_CLOUD_CLUSTER_ENDPOINT"
            )

        if not api_key:
            raise RuntimeError(
                "Missing "
                "QDRANT_CLOUD_API_KEY"
            )

        self._client = (
            AsyncQdrantClient(
                url=endpoint,
                api_key=api_key,
                timeout=60,
                prefer_grpc=True,
            )
        )

        self._collection_ready = (
            False
        )

    # --------- PROPERTIES ------------------------

    @property
    def collection_name(
        self,
    ) -> str:
        return COLLECTION_NAME

    # --------- LIFECYCLE ------------------------

    async def __aenter__(
        self,
    ) -> "QdrantCloud":
        return self

    async def __aexit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> None:
        await self.close()

    async def close(
        self,
    ) -> None:
        await self._client.close()

    # --------- VERIFY COLLECTION ------------------------

    async def ensure_collection(
        self,
    ) -> None:
        if self._collection_ready:
            return

        exists = (
            await self
            ._client
            .collection_exists(
                collection_name=(
                    COLLECTION_NAME
                )
            )
        )

        if not exists:
            raise RuntimeError(
                "Required Qdrant collection "
                f"{COLLECTION_NAME!r} "
                "does not exist"
            )

        info = (
            await self
            ._client
            .get_collection(
                collection_name=(
                    COLLECTION_NAME
                )
            )
        )

        vectors = (
            info
            .config
            .params
            .vectors
        )

        if not isinstance(
            vectors,
            models.VectorParams,
        ):
            raise RuntimeError(
                "Expected a single unnamed "
                "dense vector configuration"
            )

        if vectors.size != VECTOR_SIZE:
            raise RuntimeError(
                "Qdrant vector size mismatch: "
                f"{vectors.size} != "
                f"{VECTOR_SIZE}"
            )

        if vectors.distance != DISTANCE:
            raise RuntimeError(
                "Qdrant distance mismatch: "
                f"{vectors.distance} != "
                f"{DISTANCE}"
            )

        self._collection_ready = True

    # --------- UPSERT ------------------------

    async def upsert(
        self,
        points: list[
            dict[str, Any]
        ],
    ) -> QdrantUpsertResult:
        if not points:
            raise ValueError(
                "points cannot be empty"
            )

        await self.ensure_collection()

        qdrant_points: list[
            models.PointStruct
        ] = []

        for index, point in enumerate(
            points
        ):
            vector = point.get(
                "vector"
            )

            if not isinstance(
                vector,
                list,
            ):
                raise TypeError(
                    f"Point {index} "
                    "has no vector"
                )

            if len(vector) != VECTOR_SIZE:
                raise ValueError(
                    f"Point {index} vector "
                    f"size={len(vector)}, "
                    f"expected={VECTOR_SIZE}"
                )

            point_id = point.get(
                "id"
            )

            if point_id is None:
                raise ValueError(
                    f"Point {index} "
                    "has no id"
                )

            payload = point.get(
                "payload"
            )

            if payload is None:
                payload = {}

            qdrant_points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        started = perf_counter()

        await self._client.upsert(
            collection_name=(
                COLLECTION_NAME
            ),
            points=qdrant_points,
            wait=True,
        )

        elapsed = (
            perf_counter()
            - started
        )

        return QdrantUpsertResult(
            point_count=len(
                qdrant_points
            ),
            elapsed_seconds=elapsed,
        )
