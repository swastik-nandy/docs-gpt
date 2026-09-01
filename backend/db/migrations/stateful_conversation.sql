BEGIN;

CREATE SCHEMA IF NOT EXISTS chat;
CREATE SCHEMA IF NOT EXISTS memory;

-- Required for gen_random_uuid().
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- pgvector is intentionally not required in this migration.
-- Long-term memory can store embedding payload metadata now.
-- A later migration can add semantic long-term memory search if needed.
-- after pgvector is installed.

CREATE TABLE IF NOT EXISTS chat.conversations (
    conversation_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    user_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    status TEXT NOT NULL DEFAULT 'active',
    current_summary_id TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    CONSTRAINT conversations_status_check
        CHECK (status IN ('active', 'archived')),

    CONSTRAINT conversations_tenant_id_not_blank
        CHECK (btrim(tenant_id) <> ''),

    CONSTRAINT conversations_conversation_id_not_blank
        CHECK (btrim(conversation_id) <> '')
);

CREATE INDEX IF NOT EXISTS idx_conversations_tenant_user_active
    ON chat.conversations (tenant_id, user_id, last_active_at DESC);

CREATE INDEX IF NOT EXISTS idx_conversations_status
    ON chat.conversations (status);


CREATE TABLE IF NOT EXISTS chat.turns (
    turn_id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    conversation_id TEXT NOT NULL REFERENCES chat.conversations(conversation_id) ON DELETE CASCADE,
    tenant_id TEXT NOT NULL,
    user_id TEXT,
    turn_index INTEGER NOT NULL,
    role TEXT NOT NULL,

    raw_text TEXT NOT NULL,
    resolved_text TEXT,

    entities_extracted JSONB NOT NULL DEFAULT '[]'::jsonb,

    citations JSONB NOT NULL DEFAULT '[]'::jsonb,
    model_used TEXT,
    latency_ms INTEGER,
    hallucination_check_result JSONB NOT NULL DEFAULT '{}'::jsonb,
    retrieval_trace_id TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    CONSTRAINT turns_role_check
        CHECK (role IN ('user', 'assistant', 'system', 'tool')),

    CONSTRAINT turns_turn_index_positive
        CHECK (turn_index > 0),

    CONSTRAINT turns_latency_non_negative
        CHECK (latency_ms IS NULL OR latency_ms >= 0),

    CONSTRAINT turns_tenant_id_not_blank
        CHECK (btrim(tenant_id) <> ''),

    CONSTRAINT turns_raw_text_not_blank
        CHECK (btrim(raw_text) <> ''),

    CONSTRAINT turns_unique_conversation_turn_index
        UNIQUE (conversation_id, turn_index)
);

CREATE INDEX IF NOT EXISTS idx_turns_conversation_index_desc
    ON chat.turns (conversation_id, turn_index DESC);

CREATE INDEX IF NOT EXISTS idx_turns_tenant_user_created
    ON chat.turns (tenant_id, user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_turns_entities_gin
    ON chat.turns USING gin (entities_extracted);

CREATE INDEX IF NOT EXISTS idx_turns_citations_gin
    ON chat.turns USING gin (citations);


CREATE TABLE IF NOT EXISTS chat.conversation_summaries (
    summary_id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    conversation_id TEXT NOT NULL REFERENCES chat.conversations(conversation_id) ON DELETE CASCADE,
    tenant_id TEXT NOT NULL,
    user_id TEXT,

    covers_up_to_turn_index INTEGER NOT NULL,
    summary_text TEXT NOT NULL,

    model_used TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    CONSTRAINT conversation_summaries_turn_index_positive
        CHECK (covers_up_to_turn_index > 0),

    CONSTRAINT conversation_summaries_summary_not_blank
        CHECK (btrim(summary_text) <> ''),

    CONSTRAINT conversation_summaries_tenant_id_not_blank
        CHECK (btrim(tenant_id) <> '')
);

CREATE INDEX IF NOT EXISTS idx_conversation_summaries_latest
    ON chat.conversation_summaries (conversation_id, covers_up_to_turn_index DESC, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_conversation_summaries_tenant_user
    ON chat.conversation_summaries (tenant_id, user_id, created_at DESC);


DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'conversations_current_summary_fk'
    ) THEN
        ALTER TABLE chat.conversations
        ADD CONSTRAINT conversations_current_summary_fk
        FOREIGN KEY (current_summary_id)
        REFERENCES chat.conversation_summaries(summary_id)
        ON DELETE SET NULL;
    END IF;
END $$;


CREATE TABLE IF NOT EXISTS chat.working_memory (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    conversation_id TEXT NOT NULL REFERENCES chat.conversations(conversation_id) ON DELETE CASCADE,
    tenant_id TEXT NOT NULL,
    user_id TEXT,

    key TEXT NOT NULL,
    value JSONB NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    CONSTRAINT working_memory_key_not_blank
        CHECK (btrim(key) <> ''),

    CONSTRAINT working_memory_tenant_id_not_blank
        CHECK (btrim(tenant_id) <> '')
);

CREATE INDEX IF NOT EXISTS idx_working_memory_conversation_active
    ON chat.working_memory (conversation_id, key, expires_at);

CREATE INDEX IF NOT EXISTS idx_working_memory_tenant_user
    ON chat.working_memory (tenant_id, user_id, created_at DESC);


CREATE TABLE IF NOT EXISTS chat.entity_index (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    conversation_id TEXT NOT NULL REFERENCES chat.conversations(conversation_id) ON DELETE CASCADE,
    tenant_id TEXT NOT NULL,
    user_id TEXT,

    turn_index INTEGER NOT NULL,
    entity_text TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'unknown',
    salience_rank INTEGER NOT NULL DEFAULT 1,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    CONSTRAINT entity_index_turn_index_positive
        CHECK (turn_index > 0),

    CONSTRAINT entity_index_salience_rank_positive
        CHECK (salience_rank > 0),

    CONSTRAINT entity_index_entity_text_not_blank
        CHECK (btrim(entity_text) <> ''),

    CONSTRAINT entity_index_entity_type_not_blank
        CHECK (btrim(entity_type) <> ''),

    CONSTRAINT entity_index_tenant_id_not_blank
        CHECK (btrim(tenant_id) <> '')
);

CREATE INDEX IF NOT EXISTS idx_entity_index_resolution
    ON chat.entity_index (conversation_id, turn_index DESC, salience_rank ASC);

CREATE INDEX IF NOT EXISTS idx_entity_index_entity_text
    ON chat.entity_index (tenant_id, lower(entity_text));


CREATE TABLE IF NOT EXISTS memory.long_term_memory (
    fact_id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,

    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,

    fact_text TEXT NOT NULL,
    fact_type TEXT NOT NULL DEFAULT 'context',
    embedding_json JSONB,
    embedding_model TEXT,
    embedding_dimension INTEGER,


    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.75,
    source_turn_id TEXT REFERENCES chat.turns(turn_id) ON DELETE SET NULL,
    superseded_by TEXT REFERENCES memory.long_term_memory(fact_id) ON DELETE SET NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_confirmed_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    CONSTRAINT long_term_memory_fact_text_not_blank
        CHECK (btrim(fact_text) <> ''),

    CONSTRAINT long_term_memory_fact_type_check
        CHECK (fact_type IN ('preference', 'context', 'correction', 'identity')),

    CONSTRAINT long_term_memory_confidence_range
        CHECK (confidence >= 0.0 AND confidence <= 1.0),

    CONSTRAINT long_term_memory_tenant_id_not_blank
        CHECK (btrim(tenant_id) <> ''),

    CONSTRAINT long_term_memory_user_id_not_blank
        CHECK (btrim(user_id) <> '')
);

CREATE INDEX IF NOT EXISTS idx_long_term_memory_tenant_user
    ON memory.long_term_memory (tenant_id, user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_long_term_memory_active_facts
    ON memory.long_term_memory (tenant_id, user_id, fact_type)
    WHERE superseded_by IS NULL;

-- Vector index intentionally deferred until pgvector is installed.
-- Future migration:
-- CREATE INDEX idx_long_term_memory_embedding_hnsw
-- ON memory.long_term_memory
-- vector index details intentionally deferred.
-- WHERE embedding IS NOT NULL;

CREATE TABLE IF NOT EXISTS chat.retrieval_traces (
    retrieval_trace_id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,

    conversation_id TEXT REFERENCES chat.conversations(conversation_id) ON DELETE SET NULL,
    turn_id TEXT REFERENCES chat.turns(turn_id) ON DELETE SET NULL,

    tenant_id TEXT NOT NULL,
    user_id TEXT,

    raw_query TEXT NOT NULL,
    resolved_query TEXT,
    query_signals JSONB NOT NULL DEFAULT '{}'::jsonb,
    retrieval_decision JSONB NOT NULL DEFAULT '{}'::jsonb,
    selected_context_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
    trace JSONB NOT NULL DEFAULT '[]'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    CONSTRAINT retrieval_traces_tenant_id_not_blank
        CHECK (btrim(tenant_id) <> ''),

    CONSTRAINT retrieval_traces_raw_query_not_blank
        CHECK (btrim(raw_query) <> '')
);

CREATE INDEX IF NOT EXISTS idx_retrieval_traces_conversation_created
    ON chat.retrieval_traces (conversation_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_retrieval_traces_tenant_user_created
    ON chat.retrieval_traces (tenant_id, user_id, created_at DESC);


COMMIT;
