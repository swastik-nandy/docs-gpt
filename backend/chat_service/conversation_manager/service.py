from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    delete,
    func,
    select,
)

from .database import session_scope
from .models import (
    Conversation,
    Message,
    ToolCall,
)


# --------- HELPERS ------------------------

def _now() -> datetime:
    return datetime.now(
        timezone.utc
    )


def _require_text(
    name: str,
    value: str,
) -> str:
    cleaned = str(
        value
        or ""
    ).strip()

    if not cleaned:
        raise ValueError(
            f"{name} cannot be empty"
        )

    return cleaned


# --------- CONVERSATION SERVICE ------------------------

class ConversationService:

    # --------- ENSURE CONVERSATION ------------------------

    async def ensure_conversation(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        user_id: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Conversation:
        conversation_id = _require_text(
            "conversation_id",
            conversation_id,
        )

        tenant_id = _require_text(
            "tenant_id",
            tenant_id,
        )

        async with session_scope() as session:
            conversation = await session.get(
                Conversation,
                conversation_id,
            )

            if conversation is None:
                conversation = Conversation(
                    conversation_id=conversation_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    title=title,
                    metadata_json=metadata or {},
                )

                session.add(
                    conversation
                )

                await session.flush()

                return conversation

            if (
                conversation.tenant_id
                != tenant_id
            ):
                raise PermissionError(
                    "Conversation belongs to "
                    "another tenant"
                )

            if (
                conversation.user_id is not None
                and user_id is not None
                and conversation.user_id
                != user_id
            ):
                raise PermissionError(
                    "Conversation belongs to "
                    "another user"
                )

            if (
                conversation.user_id is None
                and user_id is not None
            ):
                conversation.user_id = (
                    user_id
                )

            if title is not None:
                conversation.title = title

            if metadata:
                conversation.metadata_json = {
                    **conversation.metadata_json,
                    **metadata,
                }

            conversation.last_active_at = (
                _now()
            )

            await session.flush()

            return conversation

    # --------- GET CONVERSATION ------------------------

    async def get_conversation(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
    ) -> Conversation | None:
        async with session_scope() as session:
            statement = (
                select(
                    Conversation
                )
                .where(
                    Conversation.conversation_id
                    == conversation_id,
                    Conversation.tenant_id
                    == tenant_id,
                )
            )

            result = await session.execute(
                statement
            )

            return (
                result
                .scalar_one_or_none()
            )

    # --------- APPEND MESSAGE ------------------------

    async def append_message(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        role: str,
        content: str,
        user_id: str | None = None,
        search_scope: dict[str, Any] | None = None,
        citations: list[
            dict[str, Any]
        ] | None = None,
        model_used: str | None = None,
        latency_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        if role not in {
            "user",
            "assistant",
        }:
            raise ValueError(
                f"Unsupported role: {role}"
            )

        content = _require_text(
            "content",
            content,
        )

        async with session_scope() as session:
            statement = (
                select(
                    Conversation
                )
                .where(
                    Conversation.conversation_id
                    == conversation_id,
                    Conversation.tenant_id
                    == tenant_id,
                )
                .with_for_update()
            )

            result = await session.execute(
                statement
            )

            conversation = (
                result
                .scalar_one_or_none()
            )

            if conversation is None:
                conversation = Conversation(
                    conversation_id=(
                        conversation_id
                    ),
                    tenant_id=tenant_id,
                    user_id=user_id,
                )

                session.add(
                    conversation
                )

                await session.flush()

            elif (
                conversation.user_id is not None
                and user_id is not None
                and conversation.user_id
                != user_id
            ):
                raise PermissionError(
                    "Conversation belongs to "
                    "another user"
                )

            sequence_number = (
                conversation
                .next_sequence
            )

            conversation.next_sequence += 1
            conversation.last_active_at = (
                _now()
            )

            message = Message(
                message_id=str(
                    uuid4()
                ),
                conversation_id=(
                    conversation_id
                ),
                sequence_number=(
                    sequence_number
                ),
                role=role,
                content=content,
                search_scope=(
                    search_scope
                    or {}
                ),
                citations=(
                    citations
                    or []
                ),
                model_used=model_used,
                latency_ms=latency_ms,
                metadata_json=(
                    metadata
                    or {}
                ),
            )

            session.add(
                message
            )

            await session.flush()

            return message

    # --------- RECENT MESSAGES ------------------------

    async def recent_messages(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        limit: int = 24,
    ) -> tuple[Message, ...]:
        if limit <= 0:
            return ()

        limit = min(
            limit,
            100,
        )

        async with session_scope() as session:
            statement = (
                select(
                    Message
                )
                .join(
                    Conversation
                )
                .where(
                    Message.conversation_id
                    == conversation_id,
                    Conversation.tenant_id
                    == tenant_id,
                )
                .order_by(
                    Message.sequence_number.desc()
                )
                .limit(
                    limit
                )
            )

            result = await session.execute(
                statement
            )

            messages = list(
                result.scalars().all()
            )

        messages.reverse()

        return tuple(
            messages
        )

    # --------- MESSAGES AFTER ------------------------

    async def messages_after(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        after_sequence: int,
        limit: int = 100,
    ) -> tuple[Message, ...]:
        limit = max(
            1,
            min(
                limit,
                500,
            ),
        )

        async with session_scope() as session:
            statement = (
                select(
                    Message
                )
                .join(
                    Conversation
                )
                .where(
                    Message.conversation_id
                    == conversation_id,
                    Conversation.tenant_id
                    == tenant_id,
                    Message.sequence_number
                    > after_sequence,
                )
                .order_by(
                    Message.sequence_number
                )
                .limit(
                    limit
                )
            )

            result = await session.execute(
                statement
            )

            return tuple(
                result.scalars().all()
            )

    # --------- UPDATE SUMMARY ------------------------

    async def update_summary(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        summary: str,
        summarized_through_sequence: int,
    ) -> None:
        summary = _require_text(
            "summary",
            summary,
        )

        async with session_scope() as session:
            statement = (
                select(
                    Conversation
                )
                .where(
                    Conversation.conversation_id
                    == conversation_id,
                    Conversation.tenant_id
                    == tenant_id,
                )
                .with_for_update()
            )

            result = await session.execute(
                statement
            )

            conversation = (
                result
                .scalar_one_or_none()
            )

            if conversation is None:
                raise LookupError(
                    "Conversation not found"
                )

            conversation.summary = summary

            conversation.summarized_through_sequence = (
                summarized_through_sequence
            )

            conversation.summary_updated_at = (
                _now()
            )

            conversation.last_active_at = (
                _now()
            )

    # --------- INSERT TOOL CALL ------------------------

    async def insert_tool_call(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any] | None = None,
        message_id: str | None = None,
        status: str = "completed",
        latency_ms: int | None = None,
    ) -> ToolCall:
        tool_name = _require_text(
            "tool_name",
            tool_name,
        )

        if status not in {
            "pending",
            "completed",
            "failed",
        }:
            raise ValueError(
                f"Unsupported status: {status}"
            )

        async with session_scope() as session:
            statement = (
                select(
                    Conversation.conversation_id
                )
                .where(
                    Conversation.conversation_id
                    == conversation_id,
                    Conversation.tenant_id
                    == tenant_id,
                )
            )

            conversation_result = (
                await session.execute(
                    statement
                )
            )

            if (
                conversation_result
                .scalar_one_or_none()
                is None
            ):
                raise LookupError(
                    "Conversation not found"
                )

            tool_call = ToolCall(
                tool_call_id=str(
                    uuid4()
                ),
                conversation_id=(
                    conversation_id
                ),
                message_id=message_id,
                tool_name=tool_name,
                arguments=arguments,
                result=result or {},
                status=status,
                latency_ms=latency_ms,
            )

            session.add(
                tool_call
            )

            await session.flush()

            return tool_call

    # --------- RECENT TOOL CALLS ------------------------

    async def recent_tool_calls(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        tool_names: list[str] | None = None,
        limit: int = 4,
    ) -> tuple[ToolCall, ...]:
        limit = max(
            1,
            min(
                limit,
                20,
            ),
        )

        async with session_scope() as session:
            statement = (
                select(
                    ToolCall
                )
                .join(
                    Conversation
                )
                .where(
                    ToolCall.conversation_id
                    == conversation_id,
                    Conversation.tenant_id
                    == tenant_id,
                    ToolCall.status
                    == "completed",
                )
            )

            if tool_names:
                statement = (
                    statement.where(
                        ToolCall.tool_name.in_(
                            tool_names
                        )
                    )
                )

            statement = (
                statement
                .order_by(
                    ToolCall.created_at.desc()
                )
                .limit(
                    limit
                )
            )

            result = await session.execute(
                statement
            )

            return tuple(
                result.scalars().all()
            )

    # --------- LIST CONVERSATIONS ------------------------

    async def list_conversations(
        self,
        *,
        tenant_id: str,
        user_id: str,
        limit: int = 20,
    ) -> tuple[Conversation, ...]:
        limit = max(
            1,
            min(
                limit,
                100,
            ),
        )

        async with session_scope() as session:
            statement = (
                select(
                    Conversation
                )
                .where(
                    Conversation.tenant_id
                    == tenant_id,
                    Conversation.user_id
                    == user_id,
                    Conversation.status
                    == "active",
                )
                .order_by(
                    Conversation
                    .last_active_at
                    .desc()
                )
                .limit(
                    limit
                )
            )

            result = await session.execute(
                statement
            )

            return tuple(
                result.scalars().all()
            )

    # --------- COUNT USER MESSAGES ------------------------

    async def count_user_messages(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        user_id: str,
    ) -> int:
        async with session_scope() as session:
            statement = (
                select(
                    func.count(
                        Message.message_id
                    )
                )
                .join(
                    Conversation
                )
                .where(
                    Message.conversation_id
                    == conversation_id,
                    Message.role
                    == "user",
                    Conversation.tenant_id
                    == tenant_id,
                    Conversation.user_id
                    == user_id,
                )
            )

            result = await session.execute(
                statement
            )

            return int(
                result.scalar_one()
            )

    # --------- DELETE CONVERSATION ------------------------

    async def delete_conversation(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        user_id: str,
    ) -> bool:
        async with session_scope() as session:
            statement = (
                delete(
                    Conversation
                )
                .where(
                    Conversation.conversation_id
                    == conversation_id,
                    Conversation.tenant_id
                    == tenant_id,
                    Conversation.user_id
                    == user_id,
                )
            )

            result = await session.execute(
                statement
            )

            return bool(
                result.rowcount
            )
