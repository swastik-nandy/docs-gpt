BEGIN;


--==============================================================================
-- SCHEMAS
--==============================================================================

CREATE SCHEMA IF NOT EXISTS ingestion;
CREATE SCHEMA IF NOT EXISTS knowledge;


-- Required for gen_random_uuid().
CREATE EXTENSION IF NOT EXISTS pgcrypto;


--==============================================================================
-- INGESTION JOBS
--==============================================================================

CREATE TABLE IF NOT EXISTS ingestion.jobs (
    job_id TEXT PRIMARY KEY,

    tenant_id TEXT NOT NULL,

    status TEXT NOT NULL DEFAULT 'queued',

    upload_count INTEGER NOT NULL DEFAULT 0,
    source_count INTEGER NOT NULL DEFAULT 0,
    duplicate_count INTEGER NOT NULL DEFAULT 0,

    source_artifact_root TEXT,
    output_artifact_root TEXT,

    queued_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    processing_started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    error_code TEXT,
    error_message TEXT,

    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    CONSTRAINT ingestion_jobs_status_check
        CHECK (
            status IN (
                'queued',
                'processing',
                'succeeded',
                'failed'
            )
        ),

    CONSTRAINT ingestion_jobs_job_id_not_blank
        CHECK (
            btrim(job_id) <> ''
        ),

    CONSTRAINT ingestion_jobs_tenant_id_not_blank
        CHECK (
            btrim(tenant_id) <> ''
        ),

    CONSTRAINT ingestion_jobs_upload_count_non_negative
        CHECK (
            upload_count >= 0
        ),

    CONSTRAINT ingestion_jobs_source_count_non_negative
        CHECK (
            source_count >= 0
        ),

    CONSTRAINT ingestion_jobs_duplicate_count_non_negative
        CHECK (
            duplicate_count >= 0
        ),

    CONSTRAINT ingestion_jobs_source_count_valid
        CHECK (
            source_count <= upload_count
        ),

    CONSTRAINT ingestion_jobs_duplicate_count_valid
        CHECK (
            duplicate_count <= upload_count
        ),

    CONSTRAINT ingestion_jobs_accounting_check
        CHECK (
            source_count + duplicate_count = upload_count
        ),

    CONSTRAINT ingestion_jobs_tenant_job_unique
        UNIQUE (
            tenant_id,
            job_id
        )
);


CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_tenant_status
    ON ingestion.jobs (
        tenant_id,
        status,
        queued_at DESC
    );


CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_tenant_created
    ON ingestion.jobs (
        tenant_id,
        queued_at DESC
    );


--==============================================================================
-- KNOWLEDGE SOURCES
--==============================================================================

CREATE TABLE IF NOT EXISTS knowledge.sources (
    knowledge_source_id TEXT PRIMARY KEY
        DEFAULT gen_random_uuid()::text,

    tenant_id TEXT NOT NULL,

    job_id TEXT NOT NULL,

    source_id TEXT NOT NULL,

    filename TEXT NOT NULL,

    relative_path TEXT NOT NULL,

    media_type TEXT NOT NULL,

    size_bytes BIGINT NOT NULL,

    content_sha256 TEXT NOT NULL,

    status TEXT NOT NULL DEFAULT 'queued',

    source_artifact_path TEXT,

    parser_artifact_path TEXT,

    chunk_artifact_path TEXT,

    qdrant_collection TEXT,

    indexed_chunk_count INTEGER,

    indexed_at TIMESTAMPTZ,

    unindexed_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    last_error_code TEXT,

    last_error_message TEXT,

    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    CONSTRAINT knowledge_sources_job_fk
        FOREIGN KEY (
            tenant_id,
            job_id
        )
        REFERENCES ingestion.jobs (
            tenant_id,
            job_id
        )
        ON DELETE RESTRICT,

    CONSTRAINT knowledge_sources_status_check
        CHECK (
            status IN (
                'queued',
                'transforming',
                'indexing',
                'indexed',
                'failed',
                'unindexing',
                'unindexed'
            )
        ),

    CONSTRAINT knowledge_sources_tenant_id_not_blank
        CHECK (
            btrim(tenant_id) <> ''
        ),

    CONSTRAINT knowledge_sources_job_id_not_blank
        CHECK (
            btrim(job_id) <> ''
        ),

    CONSTRAINT knowledge_sources_source_id_not_blank
        CHECK (
            btrim(source_id) <> ''
        ),

    CONSTRAINT knowledge_sources_filename_not_blank
        CHECK (
            btrim(filename) <> ''
        ),

    CONSTRAINT knowledge_sources_relative_path_not_blank
        CHECK (
            btrim(relative_path) <> ''
        ),

    CONSTRAINT knowledge_sources_media_type_not_blank
        CHECK (
            btrim(media_type) <> ''
        ),

    CONSTRAINT knowledge_sources_content_sha256_not_blank
        CHECK (
            btrim(content_sha256) <> ''
        ),

    CONSTRAINT knowledge_sources_size_bytes_non_negative
        CHECK (
            size_bytes >= 0
        ),

    CONSTRAINT knowledge_sources_chunk_count_non_negative
        CHECK (
            indexed_chunk_count IS NULL
            OR indexed_chunk_count >= 0
        ),

    CONSTRAINT knowledge_sources_tenant_job_source_unique
        UNIQUE (
            tenant_id,
            job_id,
            source_id
        )
);


--==============================================================================
-- KNOWLEDGE SOURCE INDEXES
--==============================================================================

CREATE INDEX IF NOT EXISTS idx_knowledge_sources_tenant_status
    ON knowledge.sources (
        tenant_id,
        status,
        updated_at DESC
    );


CREATE INDEX IF NOT EXISTS idx_knowledge_sources_tenant_created
    ON knowledge.sources (
        tenant_id,
        created_at DESC
    );


CREATE INDEX IF NOT EXISTS idx_knowledge_sources_tenant_content_sha256
    ON knowledge.sources (
        tenant_id,
        content_sha256
    );


CREATE INDEX IF NOT EXISTS idx_knowledge_sources_job
    ON knowledge.sources (
        tenant_id,
        job_id
    );


CREATE INDEX IF NOT EXISTS idx_knowledge_sources_filename
    ON knowledge.sources (
        tenant_id,
        lower(filename)
    );


CREATE INDEX IF NOT EXISTS idx_knowledge_sources_indexed
    ON knowledge.sources (
        tenant_id,
        indexed_at DESC
    )
    WHERE status = 'indexed';


--==============================================================================
-- UPDATED AT TRIGGER
--==============================================================================

CREATE OR REPLACE FUNCTION knowledge.touch_source_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = now();

    RETURN NEW;
END;
$$;


DROP TRIGGER IF EXISTS trg_knowledge_sources_updated_at
    ON knowledge.sources;


CREATE TRIGGER trg_knowledge_sources_updated_at
BEFORE UPDATE
ON knowledge.sources
FOR EACH ROW
EXECUTE FUNCTION knowledge.touch_source_updated_at();


COMMIT;