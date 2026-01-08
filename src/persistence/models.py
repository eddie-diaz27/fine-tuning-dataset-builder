"""
Database models and schema for SQLite persistence.
"""

# SQLite schema definition
CREATE_TABLES_SQL = """
-- Datasets table
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    mode TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_samples INTEGER,
    total_tokens INTEGER,
    config TEXT,
    config_id TEXT,
    FOREIGN KEY (config_id) REFERENCES saved_configs(id) ON DELETE SET NULL
);

-- Samples table
CREATE TABLE IF NOT EXISTS samples (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    content TEXT NOT NULL,
    token_length INTEGER,
    topic_keywords TEXT,
    source_file TEXT,
    split TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);

-- Indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_dataset_id ON samples(dataset_id);
CREATE INDEX IF NOT EXISTS idx_split ON samples(split);
CREATE INDEX IF NOT EXISTS idx_source_file ON samples(source_file);
CREATE INDEX IF NOT EXISTS idx_created_at ON datasets(created_at);
"""

# Additional metadata table for future extensibility
CREATE_METADATA_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Saved configurations table
CREATE_CONFIGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS saved_configs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    config TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_config_last_used ON saved_configs(last_used);
"""
