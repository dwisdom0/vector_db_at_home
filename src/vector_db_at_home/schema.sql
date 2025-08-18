PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS vector (
    id INTEGER PRIMARY KEY,
    vec BLOB NOT NULL,
    doc TEXT
);