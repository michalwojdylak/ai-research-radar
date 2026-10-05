from datetime import UTC, datetime, timedelta
from pathlib import Path

from research_radar.models.metrics import MetricsSnapshot
from research_radar.models.paper import Paper
from research_radar.models.ranking import PaperRecord, Rankings
from research_radar.models.run import RunReport
from research_radar.storage.database import Database
from research_radar.storage.repository import Repository


def test_sqlite_persistence_and_historical_cutoff(tmp_path: Path) -> None:
    timestamp = datetime(2026, 10, 5, tzinfo=UTC)
    path = tmp_path / "state.sqlite3"
    paper = Paper(
        id="2610.00001",
        arxiv_id="2610.00001",
        title="Reasoning model",
        abstract="AI reasoning",
        authors=["Researcher"],
        published_at=timestamp,
        categories=["cs.AI"],
    )
    with Database(path) as database:
        repository = Repository(database)
        repository.save_record(PaperRecord(paper=paper))
        repository.save_snapshot(paper.id, MetricsSnapshot(timestamp=timestamp, github_stars=10))
        repository.save_snapshot(
            paper.id, MetricsSnapshot(timestamp=timestamp + timedelta(days=1), github_stars=20)
        )
        repository.save_rankings(Rankings(timestamp=timestamp, selected_ids=[paper.id]))
        repository.save_run(RunReport(id="run", started_at=timestamp, stage="test"))
    with Database(path) as database:
        repository = Repository(database)
        assert repository.get_record(paper.id) == PaperRecord(paper=paper)
        assert len(repository.snapshots(paper.id, as_of=timestamp)) == 1
        rankings = repository.latest_rankings()
        assert rankings is not None and rankings.selected_ids == [paper.id]
        assert database.connection.execute("SELECT count(*) FROM pipeline_runs").fetchone()[0] == 1


def test_identity_merge_preserves_snapshots(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    with Database(tmp_path / "state.sqlite3") as database:
        repository = Repository(database)
        for paper_id in ("old", "new"):
            repository.save_record(
                PaperRecord(
                    paper=Paper(
                        id=paper_id,
                        title="A",
                        abstract="AI",
                        authors=[],
                        categories=["cs.AI"],
                        published_at=now,
                    )
                )
            )
        repository.save_snapshot("old", MetricsSnapshot(timestamp=now, citations=12))
        repository.merge_identity("old", "new")
        assert repository.get_record("old") is None
        assert repository.snapshots("new")[0].citations == 12
