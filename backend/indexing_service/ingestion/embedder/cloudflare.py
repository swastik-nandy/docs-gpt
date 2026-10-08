from __future__ import annotations

from environment import load_environment
import asyncio
import os

load_environment()

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import httpx


from .config import get_embedding_config


# --------- ENVIRONMENT ------------------------

# --------- RESULT ------------------------

@dataclass(frozen=True)
class EmbeddingBatchResult:
    vectors: list[list[float]]
    elapsed_seconds: float
    input_count: int
    vector_size: int


# --------- CLIENT ------------------------

class CloudflareEmbedder:
    def __init__(
        self,
    ) -> None:
        config = (
            get_embedding_config()
        )

        self._config = (
            config.embedder
        )

        self._account_id = (
            os.getenv(
                "CLOUDFLARE_ACCOUNT_ID"
            )
            or ""
        ).strip()

        self._token = (
            os.getenv(
                "CLOUDFLARE_TOKEN"
            )
            or os.getenv(
                "CLOUDFLARE_API_TOKEN"
            )
            or ""
        ).strip()

        if not self._account_id:
            raise RuntimeError(
                "Missing CLOUDFLARE_ACCOUNT_ID"
            )

        if not self._token:
            raise RuntimeError(
                "Missing CLOUDFLARE_TOKEN"
            )

        model = (
            self._config
            .model
            .name
        )

        self._url = (
            "https://api.cloudflare.com/"
            "client/v4/accounts/"
            f"{self._account_id}/"
            f"ai/run/{model}"
        )

        self._client = (
            httpx.AsyncClient(
                timeout=httpx.Timeout(
                    self._config
                    .request
                    .timeout_seconds
                ),
                headers={
                    "Authorization": (
                        f"Bearer {self._token}"
                    ),
                    "Content-Type": (
                        "application/json"
                    ),
                },
            )
        )

    # --------- LIFECYCLE ------------------------

    async def __aenter__(
        self,
    ) -> "CloudflareEmbedder":
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
        await self._client.aclose()

    # --------- EMBED ------------------------

    async def embed(
        self,
        texts: list[str],
    ) -> EmbeddingBatchResult:
        if not texts:
            raise ValueError(
                "texts cannot be empty"
            )

        for index, text in enumerate(
            texts
        ):
            if not isinstance(
                text,
                str,
            ):
                raise TypeError(
                    "Embedding input at index "
                    f"{index} is not a string"
                )

            if not text.strip():
                raise ValueError(
                    "Embedding input at index "
                    f"{index} is empty"
                )

        started = perf_counter()

        payload = await self._request(
            texts
        )

        vectors = (
            _extract_vectors(
                payload
            )
        )

        expected_size = (
            self._config
            .model
            .vector_size
        )

        if len(vectors) != len(
            texts
        ):
            raise RuntimeError(
                "Cloudflare returned "
                f"{len(vectors)} vectors "
                f"for {len(texts)} inputs"
            )

        for index, vector in enumerate(
            vectors
        ):
            if len(vector) != expected_size:
                raise RuntimeError(
                    "Unexpected vector size at "
                    f"index {index}: "
                    f"{len(vector)} "
                    f"(expected {expected_size})"
                )

        elapsed = (
            perf_counter()
            - started
        )

        return EmbeddingBatchResult(
            vectors=vectors,
            elapsed_seconds=elapsed,
            input_count=len(
                texts
            ),
            vector_size=expected_size,
        )

    # --------- HTTP ------------------------

    async def _request(
        self,
        texts: list[str],
    ) -> dict[str, Any]:
        retries = (
            self._config
            .request
            .max_retries
        )

        last_error: Exception | None = (
            None
        )

        for attempt in range(
            retries + 1
        ):
            try:
                response = (
                    await self
                    ._client
                    .post(
                        self._url,
                        json={
                            "text": texts,
                        },
                    )
                )

                response.raise_for_status()

                payload = response.json()

                if not isinstance(
                    payload,
                    dict,
                ):
                    raise RuntimeError(
                        "Unexpected Cloudflare "
                        "response type"
                    )

                if (
                    "success" in payload
                    and payload[
                        "success"
                    ] is False
                ):
                    raise RuntimeError(
                        "Cloudflare API reported "
                        f"failure: {payload}"
                    )

                return payload

            except (
                httpx.HTTPError,
                RuntimeError,
            ) as exc:
                last_error = exc

                if attempt >= retries:
                    break

                await asyncio.sleep(
                    0.25
                    * (
                        2 ** attempt
                    )
                )

        raise RuntimeError(
            "Cloudflare embedding request "
            "failed after retries"
        ) from last_error


# --------- RESPONSE PARSING ------------------------

def _extract_vectors(
    payload: dict[str, Any],
) -> list[list[float]]:
    result = payload.get(
        "result",
        payload,
    )

    if not isinstance(
        result,
        dict,
    ):
        raise RuntimeError(
            "Cloudflare response has no "
            "embedding result object"
        )

    data = result.get(
        "data"
    )

    if not isinstance(
        data,
        list,
    ):
        raise RuntimeError(
            "Cloudflare response has no "
            "embedding data array"
        )

    vectors: list[
        list[float]
    ] = []

    for index, vector in enumerate(
        data
    ):
        if not isinstance(
            vector,
            list,
        ):
            raise RuntimeError(
                "Embedding at index "
                f"{index} is not an array"
            )

        vectors.append(
            [
                float(value)
                for value in vector
            ]
        )

    return vectors
