from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import psycopg
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from rag.conversation.config import ConversationDbConfig, load_conversation_runtime_config
from rag.conversation.models import (
    ConversationContext,
    ConversationSummary,
    ConversationTurn,
    EntityMention,
    Payload,
    TurnRole,
    WorkingMemoryItem,
)


class PostgresConversationRepository:
    def __init__(self, connection: AsyncConnection[Any]) -> None:
        self.connection = connection

    @classmethod
    async def connect(
        cls,
        config: ConversationDbConfig | None = None,
    ) -> PostgresConversationRepository:
        runtime_config = load_conversation_runtime_config()
        resolved = config or ConversationDbConfig.from_env(
            runtime_config=runtime_config,
        )
        resolved.validate()

        connection = await psycopg.AsyncConnection.connect(
            resolved.dsn,
            row_factory=dict_row,
        )
        return cls(connection)

    async def close(self) -> None:
        await self.connection.close()

    async def ensure_conversation(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        user_id: str | None = None,
        metadata: Payload | None = None,
    ) -> None:
        _require_clean("conversation_id", conversation_id)
        _require_clean("tenant_id", tenant_id)

        await self.connection.execute(
            """
            INSERT INTO chat.conversations (
                conversation_id,
                tenant_id,
                user_id,
                metadata
            )
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (conversation_id)
            DO UPDATE SET
                tenant_id = EXCLUDED.tenant_id,
                user_id = EXCLUDED.user_id,
                last_active_at = now(),
                metadata = chat.conversations.metadata || EXCLUDED.metadata
            """,
            (
                conversation_id,
                tenant_id,
                user_id,
                Jsonb(metadata or {}),
            ),
        )
        await self.connection.commit()

    async def next_turn_index(self, *, conversation_id: str) -> int:
        _require_clean("conversation_id", conversation_id)

        row = await self._fetch_one(
            """
            SELECT COALESCE(MAX(turn_index), 0) + 1 AS next_turn_index
            FROM chat.turns
            WHERE conversation_id = %s
            """,
            (conversation_id,),
        )

        return int(row["next_turn_index"])

    async def insert_turn(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        role: TurnRole,
        raw_text: str,
        user_id: str | None = None,
        turn_index: int | None = None,
        resolved_text: str | None = None,
        entities_extracted: Sequence[Mapping[str, Any]] | None = None,
        citations: Sequence[Mapping[str, Any]] | None = None,
        model_used: str | None = None,
        latency_ms: int | None = None,
        hallucination_check_result: Mapping[str, Any] | None = None,
        retrieval_trace_id: str | None = None,
        metadata: Payload | None = None,
    ) -> ConversationTurn:
        _require_clean("conversation_id", conversation_id)
        _require_clean("tenant_id", tenant_id)
        _require_clean("raw_text", raw_text)

        await self.ensure_conversation(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            user_id=user_id,
        )

        resolved_turn_index = turn_index or await self.next_turn_index(
            conversation_id=conversation_id,
        )

        row = await self._fetch_one(
            """
            INSERT INTO chat.turns (
                conversation_id,
                tenant_id,
                user_id,
                turn_index,
                role,
                raw_text,
                resolved_text,
                entities_extracted,
                citations,
                model_used,
                latency_ms,
                hallucination_check_result,
                retrieval_trace_id,
                metadata
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING
                turn_id,
                conversation_id,
                tenant_id,
                user_id,
                turn_index,
                role,
                raw_text,
                resolved_text,
                entities_extracted,
                citations,
                model_used,
                latency_ms,
                hallucination_check_result,
                retrieval_trace_id,
                metadata
            """,
            (
                conversation_id,
                tenant_id,
                user_id,
                resolved_turn_index,
                role,
                raw_text,
                resolved_text,
                Jsonb(list(entities_extracted or [])),
                Jsonb(list(citations or [])),
                model_used,
                latency_ms,
                Jsonb(dict(hallucination_check_result or {})),
                retrieval_trace_id,
                Jsonb(metadata or {}),
            ),
        )

        await self.connection.commit()

        turn = _turn_from_row(row)
        turn.validate()
        return turn

    async def insert_entities(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        user_id: str | None,
        turn_index: int,
        entities: Sequence[EntityMention],
    ) -> tuple[EntityMention, ...]:
        if not entities:
            return ()

        normalized = tuple(entity.normalized() for entity in entities)
        for entity in normalized:
            entity.validate()

        for entity in normalized:
            await self.connection.execute(
                """
                INSERT INTO chat.entity_index (
                    conversation_id,
                    tenant_id,
                    user_id,
                    turn_index,
                    entity_text,
                    entity_type,
                    salience_rank,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    conversation_id,
                    tenant_id,
                    user_id,
                    turn_index,
                    entity.entity_text,
                    entity.entity_type,
                    entity.salience_rank,
                    Jsonb(entity.metadata),
                ),
            )

        await self.connection.commit()
        return normalized

    async def insert_retrieval_trace(
        self,
        *,
        tenant_id: str,
        raw_query: str,
        conversation_id: str | None = None,
        turn_id: str | None = None,
        user_id: str | None = None,
        resolved_query: str | None = None,
        query_signals: Mapping[str, Any] | None = None,
        retrieval_decision: Mapping[str, Any] | None = None,
        selected_context_refs: Sequence[Mapping[str, Any]] | None = None,
        trace: Sequence[Mapping[str, Any]] | None = None,
        metadata: Payload | None = None,
    ) -> str:
        _require_clean("tenant_id", tenant_id)
        _require_clean("raw_query", raw_query)

        row = await self._fetch_one(
            """
            INSERT INTO chat.retrieval_traces (
                conversation_id,
                turn_id,
                tenant_id,
                user_id,
                raw_query,
                resolved_query,
                query_signals,
                retrieval_decision,
                selected_context_refs,
                trace,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING retrieval_trace_id
            """,
            (
                conversation_id,
                turn_id,
                tenant_id,
                user_id,
                raw_query,
                resolved_query,
                Jsonb(dict(query_signals or {})),
                Jsonb(dict(retrieval_decision or {})),
                Jsonb(list(selected_context_refs or [])),
                Jsonb(list(trace or [])),
                Jsonb(metadata or {}),
            ),
        )

        await self.connection.commit()
        return str(row["retrieval_trace_id"])

    async def update_turn_after_retrieval(
        self,
        *,
        turn_id: str,
        resolved_text: str | None,
        retrieval_trace_id: str | None,
        metadata: Payload | None = None,
    ) -> ConversationTurn:
        _require_clean("turn_id", turn_id)

        row = await self._fetch_one(
            """
            UPDATE chat.turns
            SET
                resolved_text = %s,
                retrieval_trace_id = %s,
                metadata = metadata || %s
            WHERE turn_id = %s
            RETURNING
                turn_id,
                conversation_id,
                tenant_id,
                user_id,
                turn_index,
                role,
                raw_text,
                resolved_text,
                entities_extracted,
                citations,
                model_used,
                latency_ms,
                hallucination_check_result,
                retrieval_trace_id,
                metadata
            """,
            (
                resolved_text,
                retrieval_trace_id,
                Jsonb(metadata or {}),
                turn_id,
            ),
        )

        await self.connection.commit()

        turn = _turn_from_row(row)
        turn.validate()
        return turn

    async def attach_retrieval_trace_to_turn(
        self,
        *,
        retrieval_trace_id: str,
        turn_id: str,
    ) -> None:
        _require_clean("retrieval_trace_id", retrieval_trace_id)
        _require_clean("turn_id", turn_id)

        await self.connection.execute(
            """
            UPDATE chat.retrieval_traces
            SET turn_id = %s
            WHERE retrieval_trace_id = %s
            """,
            (
                turn_id,
                retrieval_trace_id,
            ),
        )
        await self.connection.commit()

    async def fetch_recent_turns(
        self,
        *,
        conversation_id: str,
        limit: int = 8,
    ) -> tuple[ConversationTurn, ...]:
        rows = await self._fetch_all(
            """
            SELECT
                turn_id,
                conversation_id,
                tenant_id,
                user_id,
                turn_index,
                role,
                raw_text,
                resolved_text,
                entities_extracted,
                citations,
                model_used,
                latency_ms,
                hallucination_check_result,
                retrieval_trace_id,
                metadata
            FROM chat.turns
            WHERE conversation_id = %s
            ORDER BY turn_index DESC
            LIMIT %s
            """,
            (conversation_id, max(1, limit)),
        )

        turns = tuple(_turn_from_row(row) for row in reversed(rows))
        for turn in turns:
            turn.validate()
        return turns

    async def fetch_recent_entities(
        self,
        *,
        conversation_id: str,
        limit: int = 8,
    ) -> tuple[EntityMention, ...]:
        rows = await self._fetch_all(
            """
            SELECT
                entity_text,
                entity_type,
                salience_rank,
                turn_index,
                metadata
            FROM chat.entity_index
            WHERE conversation_id = %s
            ORDER BY turn_index DESC, salience_rank ASC, created_at DESC
            LIMIT %s
            """,
            (conversation_id, max(1, limit)),
        )

        entities = tuple(
            EntityMention(
                entity_text=str(row["entity_text"]),
                entity_type=str(row["entity_type"] or "unknown"),
                salience_rank=int(row["salience_rank"]),
                turn_index=int(row["turn_index"]),
                metadata=dict(row["metadata"] or {}),
            )
            for row in rows
        )

        for entity in entities:
            entity.validate()

        return entities

    async def fetch_latest_summary(
        self,
        *,
        conversation_id: str,
    ) -> ConversationSummary | None:
        rows = await self._fetch_all(
            """
            SELECT
                summary_id,
                conversation_id,
                tenant_id,
                user_id,
                covers_up_to_turn_index,
                summary_text,
                model_used,
                metadata
            FROM chat.conversation_summaries
            WHERE conversation_id = %s
            ORDER BY covers_up_to_turn_index DESC, created_at DESC
            LIMIT 1
            """,
            (conversation_id,),
        )

        if not rows:
            return None

        summary = _summary_from_row(rows[0])
        summary.validate()
        return summary

    async def fetch_working_memory(
        self,
        *,
        conversation_id: str,
    ) -> tuple[WorkingMemoryItem, ...]:
        rows = await self._fetch_all(
            """
            SELECT key, value, metadata
            FROM chat.working_memory
            WHERE conversation_id = %s
              AND (expires_at IS NULL OR expires_at > now())
            ORDER BY created_at ASC
            """,
            (conversation_id,),
        )

        items = tuple(
            WorkingMemoryItem(
                key=str(row["key"]),
                value=dict(row["value"] or {}),
                metadata=dict(row["metadata"] or {}),
            )
            for row in rows
        )

        for item in items:
            item.validate()

        return items

    async def fetch_context(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        user_id: str | None,
        turn_limit: int = 8,
        entity_limit: int = 8,
    ) -> ConversationContext:
        context = ConversationContext(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            user_id=user_id,
            recent_turns=await self.fetch_recent_turns(
                conversation_id=conversation_id,
                limit=turn_limit,
            ),
            latest_summary=await self.fetch_latest_summary(
                conversation_id=conversation_id,
            ),
            recent_entities=await self.fetch_recent_entities(
                conversation_id=conversation_id,
                limit=entity_limit,
            ),
            working_memory=await self.fetch_working_memory(
                conversation_id=conversation_id,
            ),
        )
        context.validate()
        return context

    async def _fetch_one(
        self,
        query: str,
        params: Sequence[Any],
    ) -> Mapping[str, Any]:
        async with self.connection.cursor() as cursor:
            await cursor.execute(query, params)
            row = await cursor.fetchone()

        if row is None:
            raise LookupError("Expected one row, got none.")

        return row

    async def _fetch_all(
        self,
        query: str,
        params: Sequence[Any],
    ) -> list[Mapping[str, Any]]:
        async with self.connection.cursor() as cursor:
            await cursor.execute(query, params)
            rows = await cursor.fetchall()

        return list(rows)


def _turn_from_row(row: Mapping[str, Any]) -> ConversationTurn:
    return ConversationTurn(
        turn_id=str(row["turn_id"]),
        conversation_id=str(row["conversation_id"]),
        tenant_id=str(row["tenant_id"]),
        user_id=_optional_str(row.get("user_id")),
        turn_index=int(row["turn_index"]),
        role=str(row["role"]),  # type: ignore[arg-type]
        raw_text=str(row["raw_text"]),
        resolved_text=_optional_str(row.get("resolved_text")),
        entities_extracted=list(row.get("entities_extracted") or []),
        citations=list(row.get("citations") or []),
        model_used=_optional_str(row.get("model_used")),
        latency_ms=(
            int(row["latency_ms"])
            if row.get("latency_ms") is not None
            else None
        ),
        hallucination_check_result=dict(row.get("hallucination_check_result") or {}),
        retrieval_trace_id=_optional_str(row.get("retrieval_trace_id")),
        metadata=dict(row.get("metadata") or {}),
    )


def _summary_from_row(row: Mapping[str, Any]) -> ConversationSummary:
    return ConversationSummary(
        summary_id=str(row["summary_id"]),
        conversation_id=str(row["conversation_id"]),
        tenant_id=str(row["tenant_id"]),
        user_id=_optional_str(row.get("user_id")),
        covers_up_to_turn_index=int(row["covers_up_to_turn_index"]),
        summary_text=str(row["summary_text"]),
        model_used=_optional_str(row.get("model_used")),
        metadata=dict(row.get("metadata") or {}),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None

    cleaned = str(value).strip()
    return cleaned or None


def _require_clean(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty.")
