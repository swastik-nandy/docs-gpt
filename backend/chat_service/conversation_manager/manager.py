from __future__ import annotations

from collections.abc import (
    Mapping,
    Sequence,
)
from typing import Any
from uuid import uuid4

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

from .database import ConversationDatabase
from .models import (
    Conversation,
    Message,
    MessageRole,
    Payload,
    ToolCall,
    ToolCallStatus,
)


# --------- MANAGER ------------------------

class ConversationManager:
    def __init__(
        self,
        database: ConversationDatabase,
    ) -> None:
        self.database = database

    @classmethod
    async def connect(
        cls,
        dsn: str | None = None,
    ) -> "ConversationManager":
        database = ConversationDatabase(
            dsn
        )

        await database.open()

        return cls(
            database
        )

    # --------- LIFECYCLE ------------------------

    async def __aenter__(
        self,
    ) -> "ConversationManager":
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
        await self.database.close()

    # --------- CONVERSATIONS ------------------------

    async def ensure_conversation(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        user_id: str | None = None,
        title: str | None = None,
        metadata: Payload | None = None,
    ) -> Conversation:
        _require_text(
            "conversation_id",
            conversation_id,
        )

        _require_text(
            "tenant_id",
            tenant_id,
        )

        async with (
            self.database
            .pool
            .connection()
        ) as connection:
            async with (
                connection.transaction()
            ):
                return await (
                    self
                    ._ensure_conversation(
                        connection,
                        conversation_id=(
                            conversation_id
                        ),
                        tenant_id=tenant_id,
                        user_id=user_id,
                        title=title,
                        metadata=metadata,
                    )
                )

    async def _ensure_conversation(
        self,
        connection: AsyncConnection[Any],
        *,
        conversation_id: str,
        tenant_id: str,
        user_id: str | None,
        title: str | None,
        metadata: Payload | None,
    ) -> Conversation:
        cursor = await connection.execute(
            """
            INSERT INTO
                conversation_manager.conversations
                AS c (
                    conversation_id,
                    tenant_id,
                    user_id,
                    title,
                    metadata
                )
            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s
            )
            ON CONFLICT (
                conversation_id
            )
            DO UPDATE SET
                user_id = COALESCE(
                    c.user_id,
                    EXCLUDED.user_id
                ),
                title = COALESCE(
                    EXCLUDED.title,
                    c.title
                ),
                metadata = (
                    c.metadata
                    || EXCLUDED.metadata
                ),
                last_active_at = now()
            WHERE
                c.tenant_id
                = EXCLUDED.tenant_id
                AND (
                    c.user_id IS NULL
                    OR EXCLUDED.user_id IS NULL
                    OR c.user_id
                       = EXCLUDED.user_id
                )
            RETURNING *
            """,
            (
                conversation_id,
                tenant_id,
                user_id,
                title,
                Jsonb(
                    metadata
                    or {}
                ),
            ),
        )

        row = await cursor.fetchone()

        if row is None:
            raise PermissionError(
                "Conversation belongs to "
                "another tenant or user"
            )

        return _conversation_from_row(
            row
        )

    async def get_conversation(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
    ) -> Conversation | None:
        async with (
            self.database
            .pool
            .connection()
        ) as connection:
            cursor = await connection.execute(
                """
                SELECT *
                FROM conversation_manager.conversations
                WHERE conversation_id = %s
                  AND tenant_id = %s
                """,
                (
                    conversation_id,
                    tenant_id,
                ),
            )

            row = await cursor.fetchone()

        if row is None:
            return None

        return _conversation_from_row(
            row
        )

    # --------- MESSAGES ------------------------

    async def append_message(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        role: MessageRole,
        content: str,
        user_id: str | None = None,
        search_scope: Payload | None = None,
        citations: Sequence[
            Mapping[str, Any]
        ] | None = None,
        model_used: str | None = None,
        latency_ms: int | None = None,
        metadata: Payload | None = None,
    ) -> Message:
        _require_text(
            "conversation_id",
            conversation_id,
        )

        _require_text(
            "tenant_id",
            tenant_id,
        )

        _require_text(
            "content",
            content,
        )

        if role not in {
            "user",
            "assistant",
        }:
            raise ValueError(
                "Unsupported message role"
            )

        if (
            latency_ms is not None
            and latency_ms < 0
        ):
            raise ValueError(
                "latency_ms cannot be negative"
            )

        async with (
            self.database
            .pool
            .connection()
        ) as connection:
            async with (
                connection.transaction()
            ):
                await self._ensure_conversation(
                    connection,
                    conversation_id=(
                        conversation_id
                    ),
                    tenant_id=tenant_id,
                    user_id=user_id,
                    title=None,
                    metadata=None,
                )

                cursor = await connection.execute(
                    """
                    UPDATE
                        conversation_manager.conversations
                    SET
                        next_sequence =
                            next_sequence + 1,
                        last_active_at = now()
                    WHERE conversation_id = %s
                      AND tenant_id = %s
                    RETURNING
                        next_sequence - 1
                        AS sequence_number
                    """,
                    (
                        conversation_id,
                        tenant_id,
                    ),
                )

                sequence_row = (
                    await cursor.fetchone()
                )

                if sequence_row is None:
                    raise LookupError(
                        "Conversation not found"
                    )

                sequence_number = int(
                    sequence_row[
                        "sequence_number"
                    ]
                )

                message_id = str(
                    uuid4()
                )

                cursor = await connection.execute(
                    """
                    INSERT INTO
                        conversation_manager.messages (
                            message_id,
                            conversation_id,
                            sequence_number,
                            role,
                            content,
                            search_scope,
                            citations,
                            model_used,
                            latency_ms,
                            metadata
                        )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s
                    )
                    RETURNING *
                    """,
                    (
                        message_id,
                        conversation_id,
                        sequence_number,
                        role,
                        content,
                        Jsonb(
                            search_scope
                            or {}
                        ),
                        Jsonb(
                            list(
                                citations
                                or []
                            )
                        ),
                        model_used,
                        latency_ms,
                        Jsonb(
                            metadata
                            or {}
                        ),
                    ),
                )

                row = await cursor.fetchone()

                if row is None:
                    raise RuntimeError(
                        "Failed to insert message"
                    )

        return _message_from_row(
            row
        )

    async def recent_messages(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        limit: int = 24,
    ) -> tuple[Message, ...]:
        if limit <= 0:
            return ()

        safe_limit = min(
            int(limit),
            100,
        )

        async with (
            self.database
            .pool
            .connection()
        ) as connection:
            cursor = await connection.execute(
                """
                SELECT m.*
                FROM (
                    SELECT *
                    FROM conversation_manager.messages
                    WHERE conversation_id = %s
                    ORDER BY
                        sequence_number DESC
                    LIMIT %s
                ) AS m
                JOIN
                    conversation_manager.conversations
                    AS c
                  ON c.conversation_id
                   = m.conversation_id
                WHERE c.tenant_id = %s
                ORDER BY
                    m.sequence_number ASC
                """,
                (
                    conversation_id,
                    safe_limit,
                    tenant_id,
                ),
            )

            rows = await cursor.fetchall()

        return tuple(
            _message_from_row(
                row
            )
            for row in rows
        )

    async def messages_after(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        after_sequence: int,
        limit: int = 100,
    ) -> tuple[Message, ...]:
        safe_limit = max(
            1,
            min(
                int(limit),
                500,
            ),
        )

        async with (
            self.database
            .pool
            .connection()
        ) as connection:
            cursor = await connection.execute(
                """
                SELECT m.*
                FROM
                    conversation_manager.messages
                    AS m
                JOIN
                    conversation_manager.conversations
                    AS c
                  ON c.conversation_id
                   = m.conversation_id
                WHERE m.conversation_id = %s
                  AND c.tenant_id = %s
                  AND m.sequence_number > %s
                ORDER BY
                    m.sequence_number ASC
                LIMIT %s
                """,
                (
                    conversation_id,
                    tenant_id,
                    after_sequence,
                    safe_limit,
                ),
            )

            rows = await cursor.fetchall()

        return tuple(
            _message_from_row(
                row
            )
            for row in rows
        )

    # --------- SUMMARY ------------------------

    async def update_summary(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        summary: str,
        summarized_through_sequence: int,
    ) -> None:
        _require_text(
            "summary",
            summary,
        )

        if (
            summarized_through_sequence
            < 0
        ):
            raise ValueError(
                "summarized_through_sequence "
                "cannot be negative"
            )

        async with (
            self.database
            .pool
            .connection()
        ) as connection:
            async with (
                connection.transaction()
            ):
                cursor = await connection.execute(
                    """
                    UPDATE
                        conversation_manager.conversations
                    SET
                        summary = %s,
                        summarized_through_sequence = %s,
                        summary_updated_at = now(),
                        last_active_at = now()
                    WHERE conversation_id = %s
                      AND tenant_id = %s
                    """,
                    (
                        summary,
                        summarized_through_sequence,
                        conversation_id,
                        tenant_id,
                    ),
                )

                if cursor.rowcount != 1:
                    raise LookupError(
                        "Conversation not found"
                    )

    # --------- TOOL CALLS ------------------------

    async def insert_tool_call(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        tool_name: str,
        arguments: Payload,
        result: Payload | None = None,
        message_id: str | None = None,
        status: ToolCallStatus = "completed",
        latency_ms: int | None = None,
    ) -> ToolCall:
        _require_text(
            "tool_name",
            tool_name,
        )

        if status not in {
            "pending",
            "completed",
            "failed",
        }:
            raise ValueError(
                "Unsupported tool call status"
            )

        async with (
            self.database
            .pool
            .connection()
        ) as connection:
            async with (
                connection.transaction()
            ):
                owner = await connection.execute(
                    """
                    SELECT 1
                    FROM
                        conversation_manager.conversations
                    WHERE conversation_id = %s
                      AND tenant_id = %s
                    """,
                    (
                        conversation_id,
                        tenant_id,
                    ),
                )

                if (
                    await owner.fetchone()
                    is None
                ):
                    raise LookupError(
                        "Conversation not found"
                    )

                tool_call_id = str(
                    uuid4()
                )

                cursor = await connection.execute(
                    """
                    INSERT INTO
                        conversation_manager.tool_calls (
                            tool_call_id,
                            conversation_id,
                            message_id,
                            tool_name,
                            arguments,
                            result,
                            status,
                            latency_ms
                        )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s
                    )
                    RETURNING *
                    """,
                    (
                        tool_call_id,
                        conversation_id,
                        message_id,
                        tool_name,
                        Jsonb(
                            arguments
                        ),
                        Jsonb(
                            result
                            or {}
                        ),
                        status,
                        latency_ms,
                    ),
                )

                row = await cursor.fetchone()

                if row is None:
                    raise RuntimeError(
                        "Failed to insert tool call"
                    )

        return _tool_call_from_row(
            row
        )

    async def recent_tool_calls(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        tool_names: Sequence[str] | None = None,
        limit: int = 4,
    ) -> tuple[ToolCall, ...]:
        safe_limit = max(
            1,
            min(
                int(limit),
                20,
            ),
        )

        names = [
            str(name).strip()
            for name in (
                tool_names
                or []
            )
            if str(name).strip()
        ]

        async with (
            self.database
            .pool
            .connection()
        ) as connection:
            if names:
                cursor = await connection.execute(
                    """
                    SELECT tc.*
                    FROM
                        conversation_manager.tool_calls
                        AS tc
                    JOIN
                        conversation_manager.conversations
                        AS c
                      ON c.conversation_id
                       = tc.conversation_id
                    WHERE tc.conversation_id = %s
                      AND c.tenant_id = %s
                      AND tc.status = 'completed'
                      AND tc.tool_name = ANY(%s)
                    ORDER BY
                        tc.created_at DESC
                    LIMIT %s
                    """,
                    (
                        conversation_id,
                        tenant_id,
                        names,
                        safe_limit,
                    ),
                )

            else:
                cursor = await connection.execute(
                    """
                    SELECT tc.*
                    FROM
                        conversation_manager.tool_calls
                        AS tc
                    JOIN
                        conversation_manager.conversations
                        AS c
                      ON c.conversation_id
                       = tc.conversation_id
                    WHERE tc.conversation_id = %s
                      AND c.tenant_id = %s
                      AND tc.status = 'completed'
                    ORDER BY
                        tc.created_at DESC
                    LIMIT %s
                    """,
                    (
                        conversation_id,
                        tenant_id,
                        safe_limit,
                    ),
                )

            rows = await cursor.fetchall()

        return tuple(
            _tool_call_from_row(
                row
            )
            for row in rows
        )

    # --------- CONVERSATION LIST ------------------------

    async def list_conversations(
        self,
        *,
        tenant_id: str,
        user_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        safe_limit = max(
            1,
            min(
                int(limit),
                100,
            ),
        )

        async with (
            self.database
            .pool
            .connection()
        ) as connection:
            cursor = await connection.execute(
                """
                SELECT
                    c.conversation_id,

                    COALESCE(
                        NULLIF(
                            btrim(c.title),
                            ''
                        ),
                        LEFT(
                            (
                                SELECT m.content
                                FROM
                                    conversation_manager.messages
                                    AS m
                                WHERE
                                    m.conversation_id
                                    = c.conversation_id
                                  AND m.role = 'user'
                                ORDER BY
                                    m.sequence_number ASC
                                LIMIT 1
                            ),
                            72
                        ),
                        'New conversation'
                    ) AS title,

                    LEFT(
                        (
                            SELECT m.content
                            FROM
                                conversation_manager.messages
                                AS m
                            WHERE
                                m.conversation_id
                                = c.conversation_id
                            ORDER BY
                                m.sequence_number DESC
                            LIMIT 1
                        ),
                        120
                    ) AS preview,

                    c.last_active_at,

                    (
                        SELECT COUNT(*)
                        FROM
                            conversation_manager.messages
                            AS m
                        WHERE
                            m.conversation_id
                            = c.conversation_id
                          AND m.role = 'user'
                    ) AS user_message_count

                FROM
                    conversation_manager.conversations
                    AS c

                WHERE c.tenant_id = %s
                  AND c.user_id = %s

                ORDER BY
                    c.last_active_at DESC

                LIMIT %s
                """,
                (
                    tenant_id,
                    user_id,
                    safe_limit,
                ),
            )

            rows = await cursor.fetchall()

        return [
            dict(row)
            for row in rows
        ]

    # --------- COUNT ------------------------

    async def count_user_messages(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        user_id: str,
    ) -> int:
        async with (
            self.database
            .pool
            .connection()
        ) as connection:
            cursor = await connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM
                    conversation_manager.messages
                    AS m
                JOIN
                    conversation_manager.conversations
                    AS c
                  ON c.conversation_id
                   = m.conversation_id
                WHERE m.conversation_id = %s
                  AND c.tenant_id = %s
                  AND c.user_id = %s
                  AND m.role = 'user'
                """,
                (
                    conversation_id,
                    tenant_id,
                    user_id,
                ),
            )

            row = await cursor.fetchone()

        return int(
            (
                row
                or {}
            ).get(
                "count",
                0,
            )
        )

    # --------- DELETE ------------------------

    async def delete_conversation(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        user_id: str,
    ) -> bool:
        async with (
            self.database
            .pool
            .connection()
        ) as connection:
            async with (
                connection.transaction()
            ):
                cursor = await connection.execute(
                    """
                    DELETE FROM
                        conversation_manager.conversations
                    WHERE conversation_id = %s
                      AND tenant_id = %s
                      AND user_id = %s
                    """,
                    (
                        conversation_id,
                        tenant_id,
                        user_id,
                    ),
                )

                return bool(
                    cursor.rowcount
                )

    async def clear_conversations(
        self,
        *,
        tenant_id: str,
        user_id: str,
    ) -> int:
        async with (
            self.database
            .pool
            .connection()
        ) as connection:
            async with (
                connection.transaction()
            ):
                cursor = await connection.execute(
                    """
                    DELETE FROM
                        conversation_manager.conversations
                    WHERE tenant_id = %s
                      AND user_id = %s
                    """,
                    (
                        tenant_id,
                        user_id,
                    ),
                )

                return int(
                    cursor.rowcount
                    or 0
                )


# --------- ROW MAPPERS ------------------------

def _conversation_from_row(
    row: Mapping[str, Any],
) -> Conversation:
    return Conversation(
        conversation_id=str(
            row["conversation_id"]
        ),
        tenant_id=str(
            row["tenant_id"]
        ),
        user_id=row.get(
            "user_id"
        ),
        title=row.get(
            "title"
        ),
        summary=row.get(
            "summary"
        ),
        summarized_through_sequence=int(
            row.get(
                "summarized_through_sequence"
            )
            or 0
        ),
        next_sequence=int(
            row.get(
                "next_sequence"
            )
            or 1
        ),
        status=str(
            row["status"]
        ),
        metadata=dict(
            row.get(
                "metadata"
            )
            or {}
        ),
        created_at=row.get(
            "created_at"
        ),
        last_active_at=row.get(
            "last_active_at"
        ),
        summary_updated_at=row.get(
            "summary_updated_at"
        ),
    )


def _message_from_row(
    row: Mapping[str, Any],
) -> Message:
    return Message(
        message_id=str(
            row["message_id"]
        ),
        conversation_id=str(
            row["conversation_id"]
        ),
        sequence_number=int(
            row["sequence_number"]
        ),
        role=str(
            row["role"]
        ),
        content=str(
            row["content"]
        ),
        search_scope=dict(
            row.get(
                "search_scope"
            )
            or {}
        ),
        citations=list(
            row.get(
                "citations"
            )
            or []
        ),
        model_used=row.get(
            "model_used"
        ),
        latency_ms=row.get(
            "latency_ms"
        ),
        metadata=dict(
            row.get(
                "metadata"
            )
            or {}
        ),
        created_at=row.get(
            "created_at"
        ),
    )


def _tool_call_from_row(
    row: Mapping[str, Any],
) -> ToolCall:
    return ToolCall(
        tool_call_id=str(
            row["tool_call_id"]
        ),
        conversation_id=str(
            row["conversation_id"]
        ),
        message_id=row.get(
            "message_id"
        ),
        tool_name=str(
            row["tool_name"]
        ),
        arguments=dict(
            row.get(
                "arguments"
            )
            or {}
        ),
        result=dict(
            row.get(
                "result"
            )
            or {}
        ),
        status=str(
            row["status"]
        ),
        latency_ms=row.get(
            "latency_ms"
        ),
        created_at=row.get(
            "created_at"
        ),
    )


# --------- VALIDATION ------------------------

def _require_text(
    name: str,
    value: str,
) -> None:
    if not str(
        value
        or ""
    ).strip():
        raise ValueError(
            f"{name} cannot be empty"
        )
