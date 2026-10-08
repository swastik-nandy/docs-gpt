from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


# --------- BASE ------------------------

class Base(DeclarativeBase):
    pass


# --------- CONVERSATION ------------------------

class Conversation(Base):
    __tablename__ = "conversations"

    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'archived')",
            name="ck_conversation_status",
        ),
        Index(
            "ix_conversation_owner_activity",
            "tenant_id",
            "user_id",
            "last_active_at",
        ),
        {
            "schema": "conversation_manager",
        },
    )

    conversation_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    tenant_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )

    user_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )

    title: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    summarized_through_sequence: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        server_default="0",
    )

    next_sequence: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=1,
        server_default="1",
    )

    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="active",
        server_default="active",
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    last_active_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    summary_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.sequence_number",
    )

    tool_calls: Mapped[list["ToolCall"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


# --------- MESSAGE ------------------------

class Message(Base):
    __tablename__ = "messages"

    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            "sequence_number",
            name="uq_message_conversation_sequence",
        ),
        CheckConstraint(
            "role IN ('user', 'assistant')",
            name="ck_message_role",
        ),
        Index(
            "ix_message_conversation_sequence",
            "conversation_id",
            "sequence_number",
        ),
        {
            "schema": "conversation_manager",
        },
    )

    message_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: str(uuid4()),
    )

    conversation_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "conversation_manager.conversations.conversation_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    sequence_number: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    role: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    search_scope: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )

    citations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
    )

    model_used: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    latency_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    conversation: Mapped["Conversation"] = relationship(
        back_populates="messages",
    )

    tool_calls: Mapped[list["ToolCall"]] = relationship(
        back_populates="message",
    )


# --------- TOOL CALL ------------------------

class ToolCall(Base):
    __tablename__ = "tool_calls"

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'completed', 'failed')",
            name="ck_tool_call_status",
        ),
        Index(
            "ix_tool_call_conversation_created",
            "conversation_id",
            "created_at",
        ),
        {
            "schema": "conversation_manager",
        },
    )

    tool_call_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: str(uuid4()),
    )

    conversation_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "conversation_manager.conversations.conversation_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    message_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey(
            "conversation_manager.messages.message_id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    tool_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    arguments: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )

    result: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )

    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="completed",
        server_default="completed",
    )

    latency_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    conversation: Mapped["Conversation"] = relationship(
        back_populates="tool_calls",
    )

    message: Mapped["Message | None"] = relationship(
        back_populates="tool_calls",
    )
