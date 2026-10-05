from datetime import UTC, datetime
from pathlib import Path

import pytest

from research_radar.collectors.arxiv import ArxivCollector
from research_radar.models.metrics import MetricsSnapshot
from research_radar.models.paper import Paper
from research_radar.models.ranking import PaperRecord
from research_radar.models.run import RunReport
from research_radar.pipeline.discover import discover
from research_radar.storage.database import Database
from research_radar.storage.repository import Repository

NOW = datetime(2026, 10, 5, tzinfo=UTC)


class FixtureArxiv(ArxivCollector):
    def __init__(self, papers: list[Paper]) -> None:
        self.papers = papers

    async def discover(
        self, *, days: int | None = None, now: datetime | None = None
    ) -> list[Paper]:
        return self.papers


def paper(paper_id: str, *, arxiv_id: str | None = None) -> Paper:
    return Paper(
        id=paper_id,
        arxiv_id=arxiv_id,
        title="A neural evaluation paper",
        abstract="AI evaluation.",
        authors=["Researcher"],
        published_at=NOW,
        categories=["cs.AI"],
        doi="10.1234/example",
    )


@pytest.mark.asyncio
async def test_discovery_does_not_merge_conflicting_arxiv_ids(tmp_path: Path) -> None:
    with Database(tmp_path / "state.sqlite3") as database:
        repository = Repository(database)
        repository.save_record(PaperRecord(paper=paper("2610.00001", arxiv_id="2610.00001")))
        report = RunReport(id="test", started_at=NOW, stage="discover")
        records = await discover(
            repository,
            FixtureArxiv([paper("2610.00002", arxiv_id="2610.00002")]),
            report,
            days=7,
            now=NOW,
        )
        assert {record.paper.id for record in records} == {"2610.00001", "2610.00002"}


@pytest.mark.asyncio
async def test_discovery_migrates_exact_group_history(tmp_path: Path) -> None:
    with Database(tmp_path / "state.sqlite3") as database:
        repository = Repository(database)
        repository.save_record(PaperRecord(paper=paper("legacy")))
        repository.save_snapshot("legacy", MetricsSnapshot(timestamp=NOW, github_stars=123))
        report = RunReport(id="test", started_at=NOW, stage="discover")
        records = await discover(
            repository,
            FixtureArxiv([paper("2610.00001", arxiv_id="2610.00001v2")]),
            report,
            days=7,
            now=NOW,
        )
        assert len(records) == 1
        assert repository.get_record("legacy") is None
        assert records[0].paper.arxiv_id == "2610.00001v2"
        assert repository.snapshots("2610.00001")[0].github_stars == 123
