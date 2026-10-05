from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from research_radar.analysis.analyzer import analysis_input_fingerprint
from research_radar.config import Settings
from research_radar.models.analysis import (
    AnalysisContent,
    AnalysisResult,
    CriticResult,
    PromptVersions,
)
from research_radar.models.paper import Enrichment, Paper, SourceReference
from research_radar.models.ranking import (
    PaperRecord,
    QualityDimensions,
    RelevanceResult,
    ScoreDimension,
)
from research_radar.models.run import RunReport
from research_radar.pipeline import weekly
from research_radar.pipeline.generate import validate_generated
from research_radar.storage.database import Database
from research_radar.storage.repository import Repository

NOW = datetime(2024, 1, 8, 15, tzinfo=UTC)


def config(path: Path, **values: object) -> Settings:
    return Settings.model_validate(
        {
            "data_dir": path / "data",
            "database_path": path / "state.sqlite3",
            "cache_dir": path / "cache",
            "openai_api_key": "test-not-a-secret",
            "dry_run": True,
            **values,
        }
    )


def input_paper(identifier: str = "2401.00001", published: datetime | None = None) -> Paper:
    return Paper(
        id=identifier,
        arxiv_id=identifier + "v1",
        title="Neural evaluation method",
        abstract="A neural language model evaluation method.",
        authors=["Researcher"],
        categories=["cs.LG"],
        published_at=published or NOW - timedelta(days=5),
        updated_at=published or NOW - timedelta(days=5),
        abs_url=f"https://arxiv.org/abs/{identifier}",
    )


def content_for(paper: Paper) -> AnalysisContent:
    result = AnalysisContent.model_validate_json(
        (Path(__file__).parent / "fixtures" / "analysis.json").read_text(encoding="utf-8")
    )
    source = SourceReference(title=paper.title, url=paper.abs_url, source_type="paper")
    result.sources = [source]
    for section in result.section_sources:
        section.sources = [source]
    return result


class FixtureProvider:
    score = 95
    calls: list[str] = []

    def __init__(self, settings: Settings, model: str) -> None:
        self.model = model

    async def generate_structured[T: BaseModel](
        self, *, system_prompt: str, user_prompt: str, schema: type[T]
    ) -> T:
        self.calls.append(self.model)
        if schema is RelevanceResult:
            dimension = ScoreDimension(score=self.score, confidence=0.8, reason="Primary evidence.")
            value: BaseModel = RelevanceResult(
                relevance_score=90,
                confidence=0.8,
                reason="AI contribution",
                quality=QualityDimensions(
                    scientific_impact=dimension,
                    novelty=dimension,
                    empirical_quality=dimension,
                    practical_relevance=dimension,
                ),
            )
        elif schema is AnalysisContent:
            value = content_for(input_paper())
        elif schema is CriticResult:
            value = CriticResult(passed=True, issues=[])
        else:
            raise AssertionError("Unexpected structured schema")
        return schema.model_validate(value.model_dump())

    async def aclose(self) -> None:
        pass


class FixtureCollector:
    windows: list[tuple[datetime, datetime]] = []
    papers: list[Paper] = []

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        pass

    async def discover_period(self, *, start: datetime, end: datetime) -> list[Paper]:
        self.windows.append((start, end))
        return [paper for paper in self.papers if start <= paper.published_at < end]


class FixtureEnrichment:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        pass

    async def enrich(self, paper: Paper) -> Enrichment:
        return Enrichment(
            openalex_id="https://openalex.org/W123",
            openalex_citations=10000,
            github_stars=100000,
            hf_trending=True,
            sources=[
                SourceReference(
                    title="OpenAlex", url="https://openalex.org/W123", source_type="openalex"
                )
            ],
        )


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> None:
    FixtureProvider.calls = []
    FixtureProvider.score = 95
    FixtureCollector.windows = []
    FixtureCollector.papers = [input_paper()]
    monkeypatch.setattr(weekly, "OpenAIProvider", FixtureProvider)
    monkeypatch.setattr(weekly, "ArxivCollector", FixtureCollector)
    monkeypatch.setattr(weekly, "EnrichmentPipeline", FixtureEnrichment)


async def test_normal_run_uses_previous_complete_week(tmp_path: Path, patched: None) -> None:
    settings = config(tmp_path)
    report = await weekly.run_weekly(settings, now=NOW)
    assert report.period_id == "2024-W01"
    assert report.discovered == report.deep_analyzed == report.generated == 1
    assert FixtureCollector.windows == [
        (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 8, tzinfo=UTC))
    ]
    with Database(settings.database_path) as database:
        repository = Repository(database)
        radar = repository.get_weekly("2024-W01")
        assert radar is not None and radar.paper_count == 1
        assert radar.papers[0].publication_score == 95
        assert radar.papers[0].quality_basis == "publication_document_proxy"
        assert radar.data_cutoff == datetime(2024, 1, 8, tzinfo=UTC)
        assert radar.status == "review"
        assert len(repository.list_monthly(year=2024)) == 1
        assert len(repository.list_yearly()) == 1
        assert repository.snapshots(input_paper().id)[0].timestamp == NOW
    assert validate_generated(settings.data_dir) == 4


async def test_weak_week_is_not_padded_or_deep_analyzed(tmp_path: Path, patched: None) -> None:
    FixtureProvider.score = 40
    settings = config(tmp_path)
    report = await weekly.run_weekly(settings, now=NOW)
    assert report.deep_analyzed == report.generated == 0
    assert FixtureProvider.calls == [settings.relevance_model]
    with Database(settings.database_path) as database:
        radar = Repository(database).get_weekly("2024-W01")
        assert radar is not None and radar.paper_count == 0
        assert radar.radar_score == 0


async def test_missing_historical_quality_is_unknown_not_a_weak_week(
    tmp_path: Path, patched: None
) -> None:
    settings = config(tmp_path, openai_api_key=None, publication_proxy=False)
    await weekly.run_weekly(settings, now=NOW)
    with Database(settings.database_path) as database:
        radar = Repository(database).get_weekly("2024-W01")
        assert radar is not None and radar.paper_count == 0
        assert radar.radar_score is None
        assert radar.status == "draft"


async def test_preview_does_not_create_final_aggregate(tmp_path: Path, patched: None) -> None:
    FixtureCollector.papers = [input_paper(published=NOW - timedelta(hours=3))]
    settings = config(tmp_path)
    report = await weekly.run_weekly(settings, preview=True, now=NOW)
    assert report.deep_analyzed == 0
    with Database(settings.database_path) as database:
        repository = Repository(database)
        assert repository.get_weekly("2024-W02") is None
        radar = repository.get_weekly("2024-W02", preview=True)
        assert radar is not None and radar.is_preview and not radar.is_complete
        assert radar.status == "draft"
        assert repository.list_monthly() == []
        assert repository.list_yearly() == []
    assert (settings.data_dir / "radars" / "previews" / "weeks" / "2024" / "2024-W02.json").exists()


async def test_targeted_analysis_budget_skips_cached_papers(tmp_path: Path, patched: None) -> None:
    settings = config(tmp_path, top_papers=2)

    class FixtureService:
        calls: list[str] = []

        async def analyze(self, record: PaperRecord) -> AnalysisResult:
            self.calls.append(record.paper.id)
            return AnalysisResult(
                paper_id=record.paper.id,
                content=content_for(record.paper),
                critic=CriticResult(passed=True, issues=[]),
                status="review",
                full_text_available=False,
                confidence="reduced",
                generated_at=NOW,
                model=settings.analysis_model,
                critic_model=settings.critic_model,
                prompt_versions=PromptVersions(relevance="v1", analysis="v1", critic="v1"),
                cache_key=record.paper.id,
                input_fingerprint=analysis_input_fingerprint(record),
            )

    with Database(settings.database_path) as database:
        repository = Repository(database)
        identifiers = [f"2401.{number:05}" for number in range(1, 6)]
        for identifier in identifiers:
            repository.save_record(PaperRecord(paper=input_paper(identifier)))
        service = FixtureService()
        first = repository.get_record(identifiers[0])
        repository.save_analysis(await service.analyze(first))
        service.calls.clear()
        report = RunReport(id="test", started_at=NOW, stage="targeted")
        attempted = await weekly.analyze_paper_ids(
            repository, settings, identifiers, report, service=service
        )
        assert attempted == identifiers[1:3]
        assert service.calls == identifiers[1:3]
        assert report.deep_analyzed == 2


async def test_retrospective_does_not_replace_primary_documents(
    tmp_path: Path, patched: None
) -> None:
    settings = config(tmp_path)
    await weekly.run_weekly(settings, now=NOW)
    primary = settings.data_dir / "radars" / "weeks" / "2024" / "2024-W01.json"
    before = primary.read_bytes()
    await weekly.run_retrospective(
        settings,
        start_date=datetime(2024, 1, 1).date(),
        end_date=datetime(2024, 1, 7).date(),
        now=NOW + timedelta(days=1),
    )
    assert primary.read_bytes() == before
    with Database(settings.database_path) as database:
        repository = Repository(database)
        assert repository.get_weekly("2024-W01").mode == "publication"
        retro = repository.get_weekly("2024-W01", mode="retrospective")
        assert retro is not None and retro.radar_score is None
        assert retro.retrospective_score is not None
