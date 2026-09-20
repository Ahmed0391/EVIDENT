-- Evident database schema (blueprint §14).
-- Relational where data is structured; pgvector where it is semantic.
-- The single most important change vs the original: extracted fields and
-- anomalies are TYPED ROWS, not JSON dumped into a CLOB — so results are queryable.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    id          BIGSERIAL PRIMARY KEY,
    email       TEXT UNIQUE NOT NULL,
    role        TEXT NOT NULL DEFAULT 'analyst',   -- analyst | admin
    pw_hash     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS cases (
    id          BIGSERIAL PRIMARY KEY,
    external_id TEXT UNIQUE,
    status      TEXT NOT NULL DEFAULT 'created',
    confidence  REAL,
    created_by  BIGINT REFERENCES users(id),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    case_id     BIGINT REFERENCES cases(id) ON DELETE CASCADE,
    doc_type    TEXT NOT NULL,
    object_path TEXT NOT NULL,          -- encrypted object store
    sha256      TEXT,
    ocr_text    TEXT,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS extracted_fields (
    id                    BIGSERIAL PRIMARY KEY,
    document_id           BIGINT REFERENCES documents(id) ON DELETE CASCADE,
    field_name            TEXT NOT NULL,
    value                 TEXT,
    ocr_confidence        REAL,
    extraction_confidence REAL,
    source_bbox           JSONB,        -- links value -> pixels (evidence #1)
    page                  INT
);

CREATE TABLE IF NOT EXISTS validation_results (
    id          BIGSERIAL PRIMARY KEY,
    case_id     BIGINT REFERENCES cases(id) ON DELETE CASCADE,
    rule_id     TEXT NOT NULL,
    passed      BOOLEAN NOT NULL,
    detail      TEXT
);

CREATE TABLE IF NOT EXISTS anomalies (
    id          BIGSERIAL PRIMARY KEY,
    case_id     BIGINT REFERENCES cases(id) ON DELETE CASCADE,
    type        TEXT NOT NULL,
    severity    TEXT NOT NULL DEFAULT 'medium',
    detail      TEXT,
    cited_chunk_ids TEXT[]              -- links to knowledge_chunks (evidence #5)
);

CREATE TABLE IF NOT EXISTS decisions (
    id          BIGSERIAL PRIMARY KEY,
    case_id     BIGINT REFERENCES cases(id) ON DELETE CASCADE,
    source      TEXT NOT NULL,          -- auto | agent
    recommendation TEXT NOT NULL,       -- advisory only
    rationale   TEXT,
    confidence  REAL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS human_reviews (
    id          BIGSERIAL PRIMARY KEY,
    case_id     BIGINT REFERENCES cases(id) ON DELETE CASCADE,
    analyst_id  BIGINT REFERENCES users(id),
    action      TEXT NOT NULL,          -- approve | reject | request_correction
    edited_fields JSONB,
    reason      TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feedback (
    id          BIGSERIAL PRIMARY KEY,
    case_id     BIGINT REFERENCES cases(id) ON DELETE CASCADE,
    field_name  TEXT,
    corrected_value TEXT,
    label_kind  TEXT,                   -- extraction | anomaly | decision
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id          BIGSERIAL PRIMARY KEY,
    actor_id    BIGINT REFERENCES users(id),
    action      TEXT NOT NULL,
    target      TEXT,
    at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The knowledge base: the ONLY place embeddings live. Contains NO customer PII.
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id          TEXT PRIMARY KEY,       -- e.g. 'amld5-art13-1a' -> the citation id
    text        TEXT NOT NULL,
    breadcrumb  TEXT,
    parent_id   TEXT,
    embedding   vector(1024),           -- bge-m3
    ts          tsvector,               -- lexical half of hybrid search
    meta        JSONB                   -- jurisdiction, applies_to, version, effective_date, lang
);

CREATE INDEX IF NOT EXISTS idx_fields_name  ON extracted_fields(field_name);
CREATE INDEX IF NOT EXISTS idx_anom_type    ON anomalies(type);
CREATE INDEX IF NOT EXISTS idx_kc_ts        ON knowledge_chunks USING GIN(ts);
-- HNSW vector index (pgvector >= 0.5):
CREATE INDEX IF NOT EXISTS idx_kc_embedding ON knowledge_chunks
    USING hnsw (embedding vector_cosine_ops);
