import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from research_radar.models.history import BackfillCheckpoint, RelevanceObservation
from research_radar.models.metrics import MetricsSnapshot
from research_radar.models.paper import Paper
from research_radar.models.ranking import PaperRecord, RelevanceResult
from research_radar.storage.database import SCHEMA, Database
from research_radar.storage.repository import Repository

NOW = datetime(2024, 2, 20, tzinfo=UTC)


def paper(paper_id: str = "2402.00001") -> PaperRecord:
    return PaperRecord(
        paper=Paper(
            id=paper_id,
            arxiv_id=paper_id,
            title="Neural method",
            abstract="Neural learning",
            authors=["Researcher"],
            categories=["cs.LG"],
            published_at=NOW,
        )
    )


def legacy_database(path: Path, *, malformed: bool = False) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA)
        connection.execute("PRAGMA user_version=1")
        record = paper()
        connection.execute(
            "INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?)",
            (
                record.paper.id,
                record.paper.arxiv_id,
                None,
                record.paper.title,
                NOW.isoformat(),
                "{}" if malformed else record.model_dump_json(),
            ),
        )
        connection.execute(
            "INSERT INTO metrics_snapshots VALUES (?, ?, ?)",
            (
                record.paper.id,
                NOW.isoformat(),
                MetricsSnapshot(timestamp=NOW, citations=5).model_dump_json(),
            ),
        )
        connection.execute(
            "INSERT INTO pipeline_runs VALUES (?, ?, ?, ?, ?, ?)",
            ("existing-run", NOW.isoformat(), None, "test", "success", '{"untouched":true}'),
        )


def test_version_one_migration_preserves_operational_state(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    legacy_database(path)
    with Database(path) as database:
        repository = Repository(database)
        assert database.connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert repository.get_record("2402.00001") == paper()
        assert repository.snapshots("2402.00001")[0].citations == 5
        assert (
            database.connection.execute(
                "SELECT run_json FROM pipeline_runs WHERE id='existing-run'"
            ).fetchone()[0]
            == '{"untouched":true}'
        )
        assert (
            database.connection.execute("SELECT normalized_title FROM papers").fetchone()[0]
            == "neural method"
        )
    with Database(path) as database:
        assert database.connection.execute("SELECT count(*) FROM papers").fetchone()[0] == 1
        assert database.connection.execute("PRAGMA user_version").fetchone()[0] == 2


def test_failed_migration_rolls_back_without_deleting_data(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    legacy_database(path, malformed=True)
    with pytest.raises(ValueError):
        Database(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert connection.execute("SELECT record_json FROM papers").fetchone()[0] == "{}"
        assert "normalized_title" not in [
            row[1] for row in connection.execute("PRAGMA table_info(papers)")
        ]


def test_phase_checkpoints_and_quality_history_are_persistent(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    with Database(path) as database:
        repository = Repository(database)
        record = paper()
        repository.save_record(record)
        value = RelevanceResult(relevance_score=90, reason="Primary evidence", confidence=0.8)
        repository.save_relevance_observation(
            record.paper.id,
            RelevanceObservation(observed_at=NOW, paper=record.paper, relevance=value),
        )
        repository.save_relevance_observation(
            record.paper.id,
            RelevanceObservation(
                observed_at=NOW.replace(year=2026),
                paper=record.paper,
                relevance=value.model_copy(update={"relevance_score": 20}),
            ),
        )
        checkpoint = BackfillCheckpoint(
            job_id="job",
            period_id="2024-W08",
            start_date=NOW.date(),
            end_date=NOW.date(),
            phase="enrichment",
            processed_ids=[record.paper.id],
            updated_at=NOW,
        )
        repository.save_checkpoint(checkpoint)
    with Database(path) as database:
        repository = Repository(database)
        assert repository.get_checkpoint("job", "2024-W08") == checkpoint
        assert len(repository.relevance_observations("2402.00001", as_of=NOW)) == 1
        assert len(repository.records_between(NOW, NOW.replace(day=21))) == 1
        assert repository.records_between(NOW.replace(day=21), NOW.replace(day=22)) == []


def test_duplicate_discovery_keeps_identity_and_snapshots(tmp_path: Path) -> None:
    with Database(tmp_path / "state.sqlite3") as database:
        repository = Repository(database)
        original = paper()
        repository.save_discovered([original.paper])
        repository.save_snapshot(original.paper.id, MetricsSnapshot(timestamp=NOW, citations=5))
        revision = original.paper.model_copy(
            update={
                "arxiv_id": "2402.00001v2",
                "updated_at": NOW.replace(day=21),
                "abstract": "A revised abstract.",
            }
        )
        for _ in range(2):
            assert repository.save_discovered([revision]) == ["2402.00001"]
        assert len(repository.records_between(NOW, NOW.replace(day=22))) == 1
        assert len(repository.snapshots(original.paper.id)) == 1
        assert repository.get_record(original.paper.id).paper.abstract == "A revised abstract."


def test_relevance_updates_freeze_the_previous_observation(tmp_path: Path) -> None:
    with Database(tmp_path / "state.sqlite3") as database:
        repository = Repository(database)
        record = paper()
        record.relevance = RelevanceResult(
            relevance_score=90, reason="Original evidence", confidence=0.8
        )
        record.relevance_observed_at = NOW
        record.relevance_primary_only = True
        repository.save_record(record)
        record.relevance = RelevanceResult(
            relevance_score=30, reason="Later assessment", confidence=0.8
        )
        record.relevance_observed_at = NOW.replace(year=2026)
        repository.save_record(record)
        history = repository.relevance_observations(record.paper.id, as_of=NOW)
        assert len(history) == 1
        assert history[0].relevance.relevance_score == 90
