from datetime import UTC, datetime
from pathlib import Path
from xml.etree.ElementTree import ParseError

import httpx
import pytest
from pydantic import BaseModel

from research_radar.analysis.llm import LLMResponseError
from research_radar.config import Settings
from research_radar.models.analysis import AnalysisContent, CriticResult, ResearchPaper
from research_radar.models.paper import Enrichment, Paper, SourceReference
from research_radar.models.ranking import QualityDimensions, RelevanceResult, ScoreDimension
from research_radar.pipeline import runner
from research_radar.pipeline.generate import validate_generated
from research_radar.storage.database import Database
from research_radar.storage.repository import Repository

NOW = datetime(2026, 10, 5, tzinfo=UTC)


class FixtureProvider:
    def __init__(self, settings: Settings, model: str) -> None:
        self.model = model

    async def generate_structured[T: BaseModel](
        self, *, system_prompt: str, user_prompt: str, schema: type[T]
    ) -> T:
        if schema is RelevanceResult:
            value: BaseModel = RelevanceResult(
                relevance_score=80,
                confidence=0.8,
                reason="AI evaluation",
                quality=QualityDimensions(
                    scientific_impact=ScoreDimension(score=70, confidence=0.6, reason="Abstract"),
                    novelty=ScoreDimension(score=60, confidence=0.5, reason="Abstract"),
                    empirical_quality=ScoreDimension(score=60, confidence=0.4, reason="Abstract"),
                    practical_relevance=ScoreDimension(score=80, confidence=0.6, reason="Abstract"),
                ),
            )
        elif schema is AnalysisContent:
            value = AnalysisContent.model_validate_json(
                (Path(__file__).parent / "fixtures" / "analysis.json").read_text(encoding="utf-8")
            )
        elif schema is CriticResult:
            value = CriticResult(passed=True, issues=[])
        else:
            raise AssertionError(f"Unexpected schema {schema}")
        return schema.model_validate(value.model_dump())

    async def aclose(self) -> None:
        pass


class FixtureCollector:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        pass

    async def discover(self, *, days: int, now: datetime) -> list[Paper]:
        return [
            Paper(
                id="2610.00001",
                arxiv_id="2610.00001",
                title="Evaluation for AI",
                abstract="A neural language model evaluation method with new benchmarks.",
                authors=["Researcher"],
                published_at=NOW,
                categories=["cs.AI"],
                abs_url="https://arxiv.org/abs/2610.00001",
            )
        ]


class FixtureEnrichment:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        pass

    async def enrich(self, paper: Paper) -> Enrichment:
        return Enrichment(
            openalex_id="https://openalex.org/W123",
            openalex_citations=3,
            sources=[
                SourceReference(
                    title="OpenAlex metadata",
                    url="https://openalex.org/W123",
                    source_type="openalex",
                )
            ],
        )


@pytest.mark.asyncio
async def test_complete_offline_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "OpenAIProvider", FixtureProvider)
    monkeypatch.setattr(runner, "ArxivCollector", FixtureCollector)
    monkeypatch.setattr(runner, "EnrichmentPipeline", FixtureEnrichment)
    settings = Settings(
        _env_file=None,
        openai_api_key="test-not-a-secret",
        data_dir=tmp_path / "data",
        database_path=tmp_path / "state.sqlite3",
        cache_dir=tmp_path / "cache",
        top_papers=1,
    )
    report = await runner.run_pipeline(settings, now=NOW)
    assert report.status == "success"
    assert report.discovered == report.relevant == report.deep_analyzed == report.generated == 1
    assert report.critic_passed == 1
    assert validate_generated(settings.data_dir) == 2
    paper_path = next((settings.data_dir / "papers").rglob("*.json"))
    output = ResearchPaper.model_validate_json(paper_path.read_text(encoding="utf-8"))
    assert output.status == "review"
    assert output.michaels_take is None
    assert output.metrics.github_stars is None
    with Database(settings.database_path) as database:
        repository = Repository(database)
        assert len(repository.snapshots(output.id)) == 1
        assert repository.latest_analysis(output.id) is not None
        assert (
            database.connection.execute("SELECT status FROM pipeline_runs").fetchone()[0]
            == "success"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [httpx.ConnectError("Unavailable"), ParseError("Malformed Atom")])
async def test_discovery_failure_keeps_historical_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    class FailedCollector(FixtureCollector):
        async def discover(self, *, days: int, now: datetime) -> list[Paper]:
            raise error

    monkeypatch.setattr(runner, "ArxivCollector", FailedCollector)
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        database_path=tmp_path / "state.sqlite3",
        cache_dir=tmp_path / "cache",
    )
    report = await runner.run_pipeline(settings, "discover", now=NOW)
    assert report.status == "partial"
    assert report.failures[0].stage == "discover"


@pytest.mark.asyncio
async def test_one_failed_paper_does_not_abort_other_papers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class MixedCollector(FixtureCollector):
        async def discover(self, *, days: int, now: datetime) -> list[Paper]:
            good = (await super().discover(days=days, now=now))[0]
            bad = good.model_copy(
                update={
                    "id": "2610.00002",
                    "arxiv_id": "2610.00002",
                    "title": "Broken neural model",
                    "abs_url": "https://arxiv.org/abs/2610.00002",
                }
            )
            return [good, bad]

    class MixedProvider(FixtureProvider):
        async def generate_structured[T: BaseModel](
            self, *, system_prompt: str, user_prompt: str, schema: type[T]
        ) -> T:
            if schema is RelevanceResult and "Broken neural model" in user_prompt:
                raise LLMResponseError("Malformed structured response")
            return await super().generate_structured(
                system_prompt=system_prompt, user_prompt=user_prompt, schema=schema
            )

    monkeypatch.setattr(runner, "ArxivCollector", MixedCollector)
    monkeypatch.setattr(runner, "OpenAIProvider", MixedProvider)
    monkeypatch.setattr(runner, "EnrichmentPipeline", FixtureEnrichment)
    settings = Settings(
        _env_file=None,
        openai_api_key="test-not-a-secret",
        data_dir=tmp_path / "data",
        database_path=tmp_path / "state.sqlite3",
        cache_dir=tmp_path / "cache",
        top_papers=1,
    )
    report = await runner.run_pipeline(settings, now=NOW)
    assert report.status == "partial"
    assert report.generated == 1
    assert report.failures[0].paper_id == "2610.00002"
