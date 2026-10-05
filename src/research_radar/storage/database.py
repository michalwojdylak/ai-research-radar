import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    arxiv_id TEXT,
    doi TEXT,
    title TEXT NOT NULL,
    published_at TEXT NOT NULL,
    record_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS papers_arxiv ON papers(arxiv_id);
CREATE INDEX IF NOT EXISTS papers_doi ON papers(doi);
CREATE TABLE IF NOT EXISTS metrics_snapshots (
    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    timestamp TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    PRIMARY KEY (paper_id, timestamp)
);
CREATE TABLE IF NOT EXISTS analyses (
    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    cache_key TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    analysis_json TEXT NOT NULL,
    PRIMARY KEY (paper_id, cache_key)
);
CREATE INDEX IF NOT EXISTS analyses_latest ON analyses(paper_id, generated_at);
CREATE TABLE IF NOT EXISTS rankings (
    timestamp TEXT PRIMARY KEY,
    ranking_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    run_json TEXT NOT NULL
);
PRAGMA user_version = 1;
"""


class Database:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1):
            self.connection.close()
            raise ValueError(f"Unsupported database schema version: {version}")
        self.connection.executescript(SCHEMA)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
