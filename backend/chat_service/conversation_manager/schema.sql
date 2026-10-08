CREATE SCHEMA IF NOT EXISTS conversation_manager;


-- --------- CONVERSATIONS ------------------------

CREATE TABLE IF NOT EXISTS conversation_manager.conversations (
    conversation_id TEXT PRIMARY KEY,

    tenant_id TEXT NOT NULL,
    user_id TEXT,

    title TEXT,
    summary TEXT,

    summarized_through_sequence BIGINT NOT NULL DEFAULT 0,
    next_sequence BIGINT NOT NULL DEFAULT 1,

    status TEXT NOT NULL DEFAULT 'active'
        CHECK (
            status IN (
                'active',
                'archived'
            )
        ),

    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    summary_updated_at TIMESTAMPTZ,

    CHECK (
        jsonb_typeof(metadata) = 'object'
    )
);


CREATE INDEX IF NOT EXISTS
    idx_conversations_owner_activity
ON conversation_manager.conversations (
    tenant_id,
    user_id,
    last_active_at DESC
);


-- --------- MESSAGES ------------------------

CREATE TABLE IF NOT EXISTS conversation_manager.messages (
    message_id TEXT PRIMARY KEY,

    conversation_id TEXT NOT NULL
        REFERENCES conversation_manager.conversations(
            conversation_id
        )
        ON DELETE CASCADE,

    sequence_number BIGINT NOT NULL,

    role TEXT NOT NULL
        CHECK (
            role IN (
                'user',
                'assistant'
            )
        ),

    content TEXT NOT NULL,

    search_scope JSONB NOT NULL
        DEFAULT '{}'::jsonb,

    citations JSONB NOT NULL
        DEFAULT '[]'::jsonb,

    model_used TEXT,

    latency_ms INTEGER
        CHECK (
            latency_ms IS NULL
            OR latency_ms >= 0
        ),

    metadata JSONB NOT NULL
        DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL
        DEFAULT now(),

    UNIQUE (
        conversation_id,
        sequence_number
    ),

    CHECK (
        jsonb_typeof(search_scope)
        = 'object'
    ),

    CHECK (
        jsonb_typeof(citations)
        = 'array'
    ),

    CHECK (
        jsonb_typeof(metadata)
        = 'object'
    )
);


CREATE INDEX IF NOT EXISTS
    idx_messages_conversation_sequence
ON conversation_manager.messages (
    conversation_id,
    sequence_number
);


-- --------- TOOL CALLS ------------------------

CREATE TABLE IF NOT EXISTS conversation_manager.tool_calls (
    tool_call_id TEXT PRIMARY KEY,

    conversation_id TEXT NOT NULL
        REFERENCES conversation_manager.conversations(
            conversation_id
        )
        ON DELETE CASCADE,

    message_id TEXT
        REFERENCES conversation_manager.messages(
            message_id
        )
        ON DELETE SET NULL,

    tool_name TEXT NOT NULL,

    arguments JSONB NOT NULL
        DEFAULT '{}'::jsonb,

    result JSONB NOT NULL
        DEFAULT '{}'::jsonb,

    status TEXT NOT NULL
        DEFAULT 'completed'
        CHECK (
            status IN (
                'pending',
                'completed',
                'failed'
            )
        ),

    latency_ms INTEGER
        CHECK (
            latency_ms IS NULL
            OR latency_ms >= 0
        ),

    created_at TIMESTAMPTZ NOT NULL
        DEFAULT now(),

    CHECK (
        jsonb_typeof(arguments)
        = 'object'
    ),

    CHECK (
        jsonb_typeof(result)
        = 'object'
    )
);


CREATE INDEX IF NOT EXISTS
    idx_tool_calls_conversation_created
ON conversation_manager.tool_calls (
    conversation_id,
    created_at DESC
);
