import sqlite3
from pathlib import Path

from research_radar.models.ranking import PaperRecord
from research_radar.ranking.deduplication import canonical_arxiv_id, normalize_doi, normalize_title

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
"""

MIGRATION_2 = """
ALTER TABLE papers ADD COLUMN canonical_arxiv_id TEXT;
ALTER TABLE papers ADD COLUMN normalized_doi TEXT;
ALTER TABLE papers ADD COLUMN normalized_title TEXT;
CREATE INDEX papers_canonical_arxiv ON papers(canonical_arxiv_id);
CREATE INDEX papers_normalized_doi ON papers(normalized_doi);
CREATE INDEX papers_normalized_title ON papers(normalized_title);
CREATE INDEX papers_publication ON papers(published_at);
CREATE TABLE weekly_radars (
    id TEXT NOT NULL, mode TEXT NOT NULL, preview INTEGER NOT NULL DEFAULT 0,
    year INTEGER NOT NULL, week INTEGER NOT NULL, start_date TEXT NOT NULL,
    end_date TEXT NOT NULL, anchor_year INTEGER NOT NULL, anchor_month INTEGER NOT NULL,
    radar_json TEXT NOT NULL, PRIMARY KEY(id, mode, preview)
);
CREATE INDEX weekly_period ON weekly_radars(mode, preview, year, start_date);
CREATE INDEX weekly_month ON weekly_radars(mode, preview, anchor_year, anchor_month);
CREATE TABLE monthly_radars (
    id TEXT NOT NULL, mode TEXT NOT NULL, year INTEGER NOT NULL, month INTEGER NOT NULL,
    radar_json TEXT NOT NULL, PRIMARY KEY(id, mode)
);
CREATE TABLE yearly_radars (
    id TEXT NOT NULL, mode TEXT NOT NULL, year INTEGER NOT NULL,
    radar_json TEXT NOT NULL, PRIMARY KEY(id, mode)
);
CREATE TABLE backfill_runs (
    job_id TEXT NOT NULL, period_id TEXT NOT NULL, start_date TEXT NOT NULL,
    end_date TEXT NOT NULL, checkpoint_json TEXT NOT NULL, PRIMARY KEY(job_id, period_id)
);
CREATE TABLE relevance_observations (
    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    observed_at TEXT NOT NULL, observation_json TEXT NOT NULL,
    PRIMARY KEY(paper_id, observed_at)
);
"""


class Database:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.row_factory = sqlite3.Row
        ready = False
        try:
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.execute("PRAGMA journal_mode = WAL")
            version = self.connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2):
                raise ValueError(f"Unsupported database schema version: {version}")
            if version == 0:
                self.connection.executescript(
                    "BEGIN IMMEDIATE;\n" + SCHEMA + "\nPRAGMA user_version = 1;\nCOMMIT;"
                )
                version = 1
            if version == 1:
                self.connection.executescript("BEGIN IMMEDIATE;\n" + MIGRATION_2)
                cursor = self.connection.execute("SELECT id, record_json FROM papers")
                while rows := cursor.fetchmany(500):
                    for row in rows:
                        paper = PaperRecord.model_validate_json(row["record_json"]).paper
                        self.connection.execute(
                            "UPDATE papers SET canonical_arxiv_id=?, normalized_doi=?, "
                            "normalized_title=? WHERE id=?",
                            (
                                canonical_arxiv_id(paper.arxiv_id or paper.id),
                                normalize_doi(paper.doi),
                                normalize_title(paper.title),
                                row["id"],
                            ),
                        )
                self.connection.execute("PRAGMA user_version = 2")
                self.connection.commit()
            ready = True
        finally:
            if not ready:
                self.connection.rollback()
                self.connection.close()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
