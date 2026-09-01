from __future__ import annotations

import json
import re

from collections.abc import AsyncIterator
from collections.abc import Mapping
from typing import Any

import redis.asyncio as redis_async

from app.core.settings import RedisSettings


#------------------------ERRORS-------------------------

class RedisEventBusError(RuntimeError):
    pass


#------------------------CHANNELS-------------------------

_CHANNEL_SEGMENT_PATTERN = re.compile(
    r"^[A-Za-z0-9_-]{1,160}$"
)


def _validate_channel_segment(
    value: str,
    *,
    field_name: str,
) -> str:
    normalized = str(value or "").strip()

    if not normalized:
        raise RedisEventBusError(
            f"{field_name} cannot be empty."
        )

    if not _CHANNEL_SEGMENT_PATTERN.fullmatch(
        normalized
    ):
        raise RedisEventBusError(
            f"Invalid {field_name}: {normalized!r}"
        )

    return normalized


#------------------------EVENT BUS-------------------------

class RedisEventBus:
    def __init__(
        self,
        settings: RedisSettings,
    ) -> None:
        self._settings = settings
        self._client: redis_async.Redis | None = None

    @property
    def is_started(self) -> bool:
        return self._client is not None

    #------------------------LIFECYCLE-------------------------

    async def start(self) -> None:
        if self._client is not None:
            return

        client = redis_async.from_url(
            self._settings.url,
            encoding="utf-8",
            decode_responses=True,
        )

        try:
            await client.ping()

        except Exception as exc:
            await client.aclose()

            raise RedisEventBusError(
                "Redis connection failed for "
                f"{self._settings.url}: {exc}"
            ) from exc

        self._client = client

    async def close(self) -> None:
        client = self._client
        self._client = None

        if client is not None:
            await client.aclose()

    async def ping(self) -> bool:
        client = self._require_client()

        return bool(
            await client.ping()
        )

    #------------------------CHANNELS-------------------------

    def chat_channel(
        self,
        *,
        tenant_id: str,
        conversation_id: str,
    ) -> str:
        tenant = _validate_channel_segment(
            tenant_id,
            field_name="tenant_id",
        )

        conversation = _validate_channel_segment(
            conversation_id,
            field_name="conversation_id",
        )

        return (
            f"{self._settings.channel_prefix}:"
            f"{tenant}:"
            f"{conversation}"
        )

    #------------------------PUBLISH-------------------------

    async def publish_chat_event(
        self,
        *,
        tenant_id: str,
        conversation_id: str,
        event: Mapping[str, Any],
    ) -> int:
        client = self._require_client()

        channel = self.chat_channel(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
        )

        payload = json.dumps(
            dict(event),
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )

        subscriber_count = await client.publish(
            channel,
            payload,
        )

        return int(subscriber_count)

    #------------------------SUBSCRIBE-------------------------

    async def subscribe_chat(
        self,
        *,
        tenant_id: str,
        conversation_id: str,
    ) -> AsyncIterator[dict[str, Any]]:
        client = self._require_client()

        channel = self.chat_channel(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
        )

        async with client.pubsub() as pubsub:
            await pubsub.subscribe(
                channel
            )

            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue

                raw_data = message.get("data")

                if not isinstance(raw_data, str):
                    continue

                try:
                    event = json.loads(
                        raw_data
                    )

                except json.JSONDecodeError:
                    continue

                if not isinstance(event, dict):
                    continue

                yield event

    #------------------------CLIENT-------------------------

    def _require_client(
        self,
    ) -> redis_async.Redis:
        client = self._client

        if client is None:
            raise RedisEventBusError(
                "Redis event bus has not been started."
            )

        return client


__all__ = [
    "RedisEventBus",
    "RedisEventBusError",
]
