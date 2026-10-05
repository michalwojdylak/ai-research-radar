import shutil
import uuid
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, SecretStr, ValidationError

import research_radar.pipeline.backfill as module
from research_radar.analysis.analyzer import analysis_input_fingerprint
from research_radar.collectors.http import CachedHTTP
from research_radar.config import Settings
from research_radar.models.analysis import (
    AnalysisContent,
    AnalysisResult,
    CriticResult,
    PromptVersions,
)
from research_radar.models.history import BackfillCheckpoint, RelevanceObservation
from research_radar.models.metrics import MetricsSnapshot
from research_radar.models.paper import Enrichment, Paper, SourceReference
from research_radar.models.ranking import (
    PaperRecord,
    QualityDimensions,
    RelevanceResult,
    ScoreDimension,
)
from research_radar.models.run import RunReport
from research_radar.periods import WeekPeriod
from research_radar.pipeline.backfill import (
    BackfillOptions,
    CostProtectionError,
    backfill_job_id,
    run_backfill,
)
from research_radar.storage.database import Database
from research_radar.storage.repository import Repository

NOW = datetime(2025, 2, 1, 12, tzinfo=UTC)


@pytest.fixture
def settings() -> Iterator[Settings]:
    directory = Path(".cache/backfill-tests") / uuid.uuid4().hex
    yield Settings(
        _env_file=None,
        database_path=directory / "radar.sqlite3",
        data_dir=directory / "data",
        cache_dir=directory / "http-cache",
        openai_api_key=None,
        github_token=None,
        huggingface_token=None,
        semantic_scholar_api_key=None,
        http_cache_ttl=0,
    )
    shutil.rmtree(directory, ignore_errors=True)


def options(**changes: Any) -> BackfillOptions:
    return BackfillOptions.model_validate(
        {"start_date": date(2023, 1, 1), "end_date": date(2023, 1, 15), **changes}
    )


def paper(moment: datetime, suffix: int = 0) -> Paper:
    return Paper(
        id=f"{moment:%y%m}.{moment.day:02}{suffix:03}",
        arxiv_id=f"{moment:%y%m}.{moment.day:02}{suffix:03}v1",
        title=f"Machine learning investigation {moment:%Y-%m-%d} {suffix}",
        abstract=f"Neural networks with distinct findings on {moment:%Y-%m-%d} {suffix}.",
        authors=["Researcher"],
        published_at=moment,
        updated_at=moment,
        categories=["cs.AI"],
    )


class MockCollector:
    calls: list[tuple[datetime, datetime]]
    failure_start: datetime | None
    count: int

    def __init__(
        self, settings: Settings, client: httpx.AsyncClient, *, http: CachedHTTP | None = None
    ) -> None:
        self.http = http

    async def discover_period(self, *, start: datetime, end: datetime) -> list[Paper]:
        self.calls.append((start, end))
        if start == self.failure_start:
            raise ValueError("sensitive provider response")
        return [paper(start + timedelta(hours=1), index) for index in range(self.count)]


class MockEnrichment:
    calls: list[str]
    fail_id: str | None

    def __init__(
        self, settings: Settings, client: httpx.AsyncClient, *, http: CachedHTTP | None = None
    ) -> None:
        self.http = http

    async def enrich(self, incoming: Paper) -> Enrichment:
        self.calls.append(incoming.id)
        if incoming.id == self.fail_id:
            raise ValueError("provider secret")
        return Enrichment(openalex_citations=12)


@pytest.fixture
def mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    MockCollector.calls = []
    MockCollector.failure_start = None
    MockCollector.count = 1
    MockEnrichment.calls = []
    MockEnrichment.fail_id = None
    monkeypatch.setattr(module, "ArxivCollector", MockCollector)
    monkeypatch.setattr(module, "EnrichmentPipeline", MockEnrichment)

    def no_provider(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Default backfills must not construct a paid provider")

    monkeypatch.setattr(module, "OpenAIProvider", no_provider)
    monkeypatch.setattr(
        Repository, "list_records", lambda self: pytest.fail("Do not load the whole corpus")
    )


@pytest.mark.parametrize("worker_count", [0, 17])
def test_worker_validation(worker_count: int) -> None:
    with pytest.raises(ValidationError):
        options(workers=worker_count)


@pytest.mark.asyncio
@pytest.mark.parametrize("requested_phase", ["classify", "analyze"])
async def test_cost_gate_precedes_any_side_effect(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, requested_phase: str
) -> None:
    monkeypatch.setattr(
        module, "Database", lambda *args: pytest.fail("Do not open database before confirmation")
    )
    monkeypatch.setattr(
        module.httpx,
        "AsyncClient",
        lambda *args, **kwargs: pytest.fail("Do not construct network client"),
    )
    with pytest.raises(CostProtectionError, match="requires --confirm-expensive"):
        await run_backfill(settings, options(**{requested_phase: True}), now=NOW)


@pytest.mark.asyncio
async def test_default_key_present_no_llm_and_completed_resume_no_requests(
    settings: Settings, mocked: None
) -> None:
    settings.openai_api_key = SecretStr("fixture-not-a-real-key")
    report = await run_backfill(settings, options(), now=NOW)
    assert report.status == "success"
    assert report.completed_weeks == report.total_weeks == 3
    assert report.deep_analyzed == 0 and not report.llm_relevance_enabled
    assert len(MockCollector.calls) == len(MockEnrichment.calls) == 3
    with Database(settings.database_path) as database:
        repository = Repository(database)
        for start, end in MockCollector.calls:
            record = repository.records_between(start, end)[0]
            assert record.relevance and record.relevance.classifier == "deterministic"
            assert record.relevance.quality.novelty is None
            assert record.relevance_observed_at == NOW
            assert record.relevance_primary_only
            assert len(repository.relevance_observations(record.paper.id)) == 1
            assert [snapshot.timestamp for snapshot in repository.snapshots(record.paper.id)] == [
                NOW
            ]
    second = await run_backfill(settings, options(), now=NOW)
    assert second.completed_weeks == 3 and second.processed_weeks == 0
    assert len(MockCollector.calls) == len(MockEnrichment.calls) == 3


@pytest.mark.asyncio
async def test_iso_year_boundary_and_partial_coverage(settings: Settings, mocked: None) -> None:
    report = await run_backfill(settings, options(end_date=date(2023, 1, 2)), now=NOW)
    assert report.total_weeks == 2
    assert MockCollector.calls == [
        (datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 1, 2, tzinfo=UTC)),
        (datetime(2023, 1, 2, tzinfo=UTC), datetime(2023, 1, 3, tzinfo=UTC)),
    ]
    with Database(settings.database_path) as database:
        radar = Repository(database).get_weekly("2022-W52")
        assert radar and radar.coverage_start == radar.coverage_end == date(2023, 1, 1)
        assert radar.discovery_complete and not radar.is_complete


@pytest.mark.asyncio
async def test_current_week_cannot_masquerade_as_complete(settings: Settings, mocked: None) -> None:
    await run_backfill(
        settings, options(start_date=date(2025, 1, 27), end_date=date(2025, 2, 1)), now=NOW
    )
    assert MockCollector.calls[0][1] == NOW
    with Database(settings.database_path) as database:
        radar = Repository(database).get_weekly("2025-W05", preview=True)
        assert radar and radar.is_preview and not radar.is_complete


@pytest.mark.asyncio
async def test_same_day_partial_resume_invalidates_actual_discovery_cutoff(
    settings: Settings, mocked: None
) -> None:
    chosen = options(start_date=date(2025, 1, 27), end_date=date(2025, 2, 1))
    first = await run_backfill(settings, chosen, now=NOW)
    later = NOW + timedelta(hours=1)
    second = await run_backfill(settings, chosen, now=later)
    assert first.job_id == second.job_id and second.processed_weeks == 1
    assert [end for _, end in MockCollector.calls] == [NOW, later]
    with Database(settings.database_path) as database:
        checkpoint = Repository(database).get_checkpoint(first.job_id, "2025-W05")
        assert checkpoint and checkpoint.discovery_cutoff == later


@pytest.mark.asyncio
async def test_contiguous_clipped_jobs_complete_shared_iso_boundary_week(
    settings: Settings, mocked: None
) -> None:
    await run_backfill(
        settings, options(start_date=date(2020, 12, 28), end_date=date(2020, 12, 31)), now=NOW
    )
    with Database(settings.database_path) as database:
        first = Repository(database).get_weekly("2020-W53")
        assert first and first.discovery_complete and not first.is_complete
        assert first.coverage_end == date(2020, 12, 31)
    await run_backfill(
        settings, options(start_date=date(2021, 1, 1), end_date=date(2021, 1, 3)), now=NOW
    )
    with Database(settings.database_path) as database:
        complete = Repository(database).get_weekly("2020-W53")
        assert complete and complete.discovery_complete and complete.is_complete
        assert complete.coverage_start == date(2020, 12, 28)
        assert complete.coverage_end == date(2021, 1, 3)
    assert len(MockCollector.calls) == 2


@pytest.mark.asyncio
async def test_discovery_failure_resume_and_other_weeks_continue(
    settings: Settings, mocked: None, caplog: pytest.LogCaptureFixture
) -> None:
    MockCollector.failure_start = datetime(2023, 1, 2, tzinfo=UTC)
    first = await run_backfill(settings, options(), now=NOW)
    assert first.status == "partial" and first.failed_weeks == ["2023-W01"]
    assert first.completed_weeks == 2
    assert "sensitive provider response" not in caplog.text
    with Database(settings.database_path) as database:
        checkpoint = Repository(database).get_checkpoint(first.job_id, "2023-W01")
        assert checkpoint and checkpoint.phase == "discovery" and checkpoint.status == "failed"
        assert checkpoint.error_type == "ValueError"
    MockCollector.failure_start = None
    second = await run_backfill(settings, options(), now=NOW)
    assert second.status == "success" and second.processed_weeks == 1
    assert len(MockCollector.calls) == 4


@pytest.mark.asyncio
async def test_relevance_failure_resumes_only_failed_paper(
    settings: Settings, mocked: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    MockCollector.count = 2
    original = module.RelevanceFilter.classify
    called: list[str] = []
    failed = [False]

    async def classify(self: module.RelevanceFilter, incoming: Paper) -> RelevanceResult | None:
        called.append(incoming.id)
        if incoming.id.endswith("001") and not failed[0]:
            failed[0] = True
            raise ValueError("classifier unavailable")
        return await original(self, incoming)

    monkeypatch.setattr(module.RelevanceFilter, "classify", classify)
    chosen = options(end_date=date(2023, 1, 1))
    first = await run_backfill(settings, chosen, now=NOW)
    with Database(settings.database_path) as database:
        checkpoint = Repository(database).get_checkpoint(first.job_id, "2022-W52")
        assert checkpoint and checkpoint.phase == "relevance"
        assert checkpoint.processed_ids == ["2301.01000"]
    second = await run_backfill(settings, chosen, now=NOW)
    assert second.status == "success"
    assert len(MockCollector.calls) == 1
    assert called.count("2301.01000") == 1 and called.count("2301.01001") == 2


@pytest.mark.asyncio
async def test_enrichment_failure_preserves_successful_snapshots_on_resume(
    settings: Settings, mocked: None
) -> None:
    MockCollector.count = 2
    MockEnrichment.fail_id = "2301.01001"
    chosen = options(end_date=date(2023, 1, 1))
    first = await run_backfill(settings, chosen, now=NOW)
    with Database(settings.database_path) as database:
        repository = Repository(database)
        checkpoint = repository.get_checkpoint(first.job_id, "2022-W52")
        assert checkpoint and checkpoint.phase == "enrichment"
        assert checkpoint.processed_ids == ["2301.01000"]
    MockEnrichment.fail_id = None
    assert (await run_backfill(settings, chosen, now=NOW + timedelta(days=1))).status == "success"
    assert len(MockCollector.calls) == 1
    assert MockEnrichment.calls.count("2301.01000") == 1
    with Database(settings.database_path) as database:
        assert len(Repository(database).snapshots("2301.01000")) == 1


@pytest.mark.asyncio
async def test_expanding_partial_end_requeries_only_changed_week_without_duplicate_metrics(
    settings: Settings, mocked: None
) -> None:
    first = await run_backfill(settings, options(end_date=date(2023, 1, 3)), now=NOW)
    second = await run_backfill(settings, options(end_date=date(2023, 1, 15)), now=NOW)
    assert first.job_id == second.job_id
    assert len(MockCollector.calls) == 4
    assert MockCollector.calls[2] == (
        datetime(2023, 1, 2, tzinfo=UTC),
        datetime(2023, 1, 9, tzinfo=UTC),
    )
    assert MockEnrichment.calls.count("2301.02000") == 1
    with Database(settings.database_path) as database:
        assert len(Repository(database).snapshots("2301.02000")) == 1


@pytest.mark.parametrize(
    "changes",
    [{"skip_enrichment": True}, {"classify": True}, {"publication_proxy": True}],
)
def test_relevant_option_changes_isolate_checkpoint_namespace(
    settings: Settings, changes: dict[str, bool]
) -> None:
    assert backfill_job_id(settings, options()) != backfill_job_id(settings, options(**changes))
    assert backfill_job_id(settings, options()) == backfill_job_id(
        settings, options(analyze=True, confirm_expensive=True, end_date=date(2023, 1, 30))
    )


@pytest.mark.asyncio
async def test_changed_scoring_reranks_without_rediscovery_or_enrichment(
    settings: Settings, mocked: None
) -> None:
    chosen = options(end_date=date(2023, 1, 1))
    first = await run_backfill(settings, chosen, now=NOW)
    with Database(settings.database_path) as database:
        checkpoint = Repository(database).get_checkpoint(first.job_id, "2022-W52")
        assert checkpoint and checkpoint.ranking_fingerprint
        old_fingerprint = checkpoint.ranking_fingerprint
    settings.weekly_min_score = 85
    settings.scoring_version = "changed-scoring-protocol"
    second = await run_backfill(settings, chosen, now=NOW)
    assert second.job_id == first.job_id and second.processed_weeks == 1
    assert second.completed_weeks == 1
    assert len(MockCollector.calls) == len(MockEnrichment.calls) == 1
    with Database(settings.database_path) as database:
        repository = Repository(database)
        checkpoint = repository.get_checkpoint(first.job_id, "2022-W52")
        assert checkpoint and checkpoint.ranking_fingerprint != old_fingerprint
        radar = repository.get_weekly("2022-W52")
        assert radar and radar.scoring_version == settings.scoring_version
        assert len(repository.snapshots("2301.01000")) == 1
        assert len(repository.relevance_observations("2301.01000")) == 1


@pytest.mark.asyncio
async def test_observation_and_rebuild_clocks_advance_after_metadata(
    settings: Settings, mocked: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = [NOW]
    ranked_at: list[datetime] = []
    original = module.rebuild_week

    class ClockDateTime(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            return receipt[0]

    async def delayed_metadata(self: MockEnrichment, incoming: Paper) -> Enrichment:
        receipt[0] = NOW + timedelta(minutes=10)
        return Enrichment(openalex_citations=12)

    def rebuild(
        repository: Repository, settings: Settings, period: WeekPeriod, **kwargs: Any
    ) -> Any:
        ranked_at.append(kwargs["now"])
        return original(repository, settings, period, **kwargs)

    monkeypatch.setattr(module, "datetime", ClockDateTime)
    monkeypatch.setattr(MockEnrichment, "enrich", delayed_metadata)
    monkeypatch.setattr(module, "rebuild_week", rebuild)
    await run_backfill(settings, options(end_date=date(2023, 1, 1)))
    assert ranked_at == [NOW + timedelta(minutes=10)]
    with Database(settings.database_path) as database:
        repository = Repository(database)
        snapshots = repository.snapshots("2301.01000")
        assert len(snapshots) == 1 and snapshots[0].timestamp == NOW + timedelta(minutes=10)
        observations = repository.relevance_observations("2301.01000")
        assert len(observations) == 1 and observations[0].observed_at == NOW


@pytest.mark.asyncio
async def test_skip_enrichment_and_unknown_strict_historical_quality(
    settings: Settings, mocked: None
) -> None:
    await run_backfill(settings, options(skip_enrichment=True), now=NOW)
    assert MockEnrichment.calls == []
    with Database(settings.database_path) as database:
        repository = Repository(database)
        record = repository.get_record("2301.02000")
        assert record and record.scores.publication_score is None
        assert repository.snapshots(record.paper.id) == []
        radar = repository.get_weekly("2023-W01")
        assert radar and radar.papers == [] and radar.radar_score is None


@pytest.mark.asyncio
async def test_existing_quality_and_identity_history_are_preserved(
    settings: Settings, mocked: None
) -> None:
    old_paper = paper(datetime(2023, 1, 2, 1, tzinfo=UTC))
    result = RelevanceResult(relevance_score=90, confidence=0.9, reason="Stored primary evidence")
    result.quality.novelty = ScoreDimension(score=90, reason="Original method", confidence=0.9)
    with Database(settings.database_path) as database:
        repository = Repository(database)
        repository.save_record(
            PaperRecord(paper=old_paper, relevance=result, relevance_primary_only=True)
        )
        repository.save_snapshot(
            old_paper.id, MetricsSnapshot(timestamp=datetime(2023, 1, 5, tzinfo=UTC), citations=4)
        )
        repository.save_relevance_observation(
            old_paper.id,
            RelevanceObservation(
                observed_at=datetime(2023, 1, 3, tzinfo=UTC), paper=old_paper, relevance=result
            ),
        )
    await run_backfill(settings, options(), now=NOW)
    with Database(settings.database_path) as database:
        repository = Repository(database)
        record = repository.get_record(old_paper.id)
        assert record and record.relevance == result
        assert len(repository.relevance_observations(old_paper.id)) == 1
        assert len(repository.snapshots(old_paper.id)) == 1
    assert old_paper.id not in MockEnrichment.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [10, 12])
async def test_analysis_budget_is_global_and_resumed_without_discovery_or_enrichment(
    settings: Settings, mocked: None, monkeypatch: pytest.MonkeyPatch, budget: int
) -> None:
    MockCollector.count = 6
    settings.openai_api_key = SecretStr("fixture-not-a-real-key")
    settings.top_papers = budget
    done: set[str] = set()
    attempted: list[str] = []

    def rebuild(
        repository: Repository, settings: Settings, period: WeekPeriod, **kwargs: Any
    ) -> tuple[SimpleNamespace, list[Path]]:
        return (
            SimpleNamespace(
                papers=[
                    SimpleNamespace(paper_id=record.paper.id)
                    for record in repository.records_between(period.start, period.end)
                ]
            ),
            [],
        )

    async def analyze(
        repository: Repository, settings: Settings, ids: list[str], report: RunReport
    ) -> list[str]:
        assert settings.top_papers == 1
        attempted.extend(ids)
        done.update(ids)
        report.deep_analyzed += len(ids)
        return ids

    monkeypatch.setattr(module, "rebuild_week", rebuild)
    monkeypatch.setattr(module, "analyze_paper_ids", analyze)
    monkeypatch.setattr(
        module, "_current_analysis", lambda record, analysis, settings: record.paper.id in done
    )
    chosen = options(analyze=True, confirm_expensive=True)
    first = await run_backfill(settings, chosen, now=NOW)
    assert first.deep_analyzed == len(attempted) == budget
    assert first.status == "partial" and first.completed_weeks == budget // 6
    with Database(settings.database_path) as database:
        pending_period = ["2022-W52", "2023-W01", "2023-W02"][budget // 6]
        checkpoint = Repository(database).get_checkpoint(first.job_id, pending_period)
        assert checkpoint and checkpoint.phase == "analysis" and checkpoint.status == "pending"
        assert len(checkpoint.analyses_done) == budget % 6
    second = await run_backfill(settings, chosen, now=NOW)
    assert second.deep_analyzed == 18 - budget and second.status == "success"
    assert len(attempted) == len(set(attempted)) == 18
    assert len(MockCollector.calls) == 3 and len(MockEnrichment.calls) == 18


@pytest.mark.asyncio
async def test_analysis_confirmation_required_even_after_default_completion(
    settings: Settings, mocked: None
) -> None:
    await run_backfill(settings, options(), now=NOW)
    with pytest.raises(CostProtectionError, match="Historical analysis"):
        await run_backfill(settings, options(analyze=True), now=NOW)
    assert len(MockCollector.calls) == 3


@pytest.mark.asyncio
async def test_confirmed_backfill_analysis_generates_canonical_paper_content(
    settings: Settings, mocked: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = paper(datetime(2023, 1, 2, 1, tzinfo=UTC))
    dimension = ScoreDimension(score=95, reason="Contemporaneous paper evidence", confidence=0.9)
    relevance = RelevanceResult(
        relevance_score=90,
        reason="AI method",
        confidence=0.9,
        quality=QualityDimensions(
            scientific_impact=dimension,
            novelty=dimension,
            empirical_quality=dimension,
            practical_relevance=dimension,
        ),
    )
    with Database(settings.database_path) as database:
        repository = Repository(database)
        repository.save_record(PaperRecord(paper=original, relevance=relevance))
        repository.save_relevance_observation(
            original.id,
            RelevanceObservation(
                observed_at=datetime(2023, 1, 3, tzinfo=UTC),
                paper=original,
                relevance=relevance,
            ),
        )

    async def fake_analysis(
        repository: Repository, settings: Settings, ids: list[str], report: RunReport
    ) -> list[str]:
        for paper_id in ids:
            record = repository.get_record(paper_id)
            content = AnalysisContent.model_validate_json(
                (Path(__file__).parent / "fixtures" / "analysis.json").read_text(encoding="utf-8")
            )
            source = SourceReference(
                title=record.paper.title,
                url=f"https://arxiv.org/abs/{paper_id}",
                source_type="paper",
            )
            content.sources = [source]
            for section in content.section_sources:
                section.sources = [source]
            repository.save_analysis(
                AnalysisResult(
                    paper_id=paper_id,
                    content=content,
                    critic=CriticResult(passed=True, issues=[]),
                    status="review",
                    full_text_available=False,
                    confidence="reduced",
                    generated_at=NOW,
                    model=settings.analysis_model,
                    critic_model=settings.critic_model,
                    prompt_versions=PromptVersions(relevance="v1", analysis="v1", critic="v1"),
                    cache_key=paper_id,
                    input_fingerprint=analysis_input_fingerprint(record),
                )
            )
            report.deep_analyzed += 1
        return ids

    monkeypatch.setattr(module, "analyze_paper_ids", fake_analysis)
    report = await run_backfill(
        settings,
        options(
            start_date=date(2023, 1, 2),
            end_date=date(2023, 1, 8),
            skip_enrichment=True,
            analyze=True,
            confirm_expensive=True,
        ),
        now=NOW,
    )
    assert report.deep_analyzed == 1
    assert len(list((settings.data_dir / "papers").rglob("*.json"))) == 1
    assert len(list((settings.data_dir / "articles" / "drafts").glob("*.md"))) == 1


@pytest.mark.asyncio
async def test_optional_enrichment_outages_remain_unknown_observed_now(
    settings: Settings, mocked: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unavailable(self: MockEnrichment, incoming: Paper) -> Enrichment:
        return Enrichment()

    monkeypatch.setattr(MockEnrichment, "enrich", unavailable)
    report = await run_backfill(settings, options(), now=NOW)
    assert report.status == "success"
    with Database(settings.database_path) as database:
        snapshots = Repository(database).snapshots("2301.02000")
        assert len(snapshots) == 1 and snapshots[0].timestamp == NOW
        assert snapshots[0].citations is None and snapshots[0].github_stars is None


@pytest.mark.asyncio
async def test_ranking_failure_resumes_without_repeat_observations(
    settings: Settings, mocked: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = module.rebuild_week
    unavailable = [True]

    def rebuild(
        repository: Repository, settings: Settings, period: WeekPeriod, **kwargs: Any
    ) -> Any:
        if unavailable[0]:
            raise ValueError("ranking unavailable")
        return original(repository, settings, period, **kwargs)

    monkeypatch.setattr(module, "rebuild_week", rebuild)
    chosen = options(end_date=date(2023, 1, 1))
    first = await run_backfill(settings, chosen, now=NOW)
    with Database(settings.database_path) as database:
        checkpoint = Repository(database).get_checkpoint(first.job_id, "2022-W52")
        assert checkpoint and checkpoint.phase == "ranking" and checkpoint.processed_ids == []
    unavailable[0] = False
    second = await run_backfill(settings, chosen, now=NOW + timedelta(days=1))
    assert second.status == "success"
    assert len(MockCollector.calls) == len(MockEnrichment.calls) == 1
    with Database(settings.database_path) as database:
        repository = Repository(database)
        assert len(repository.snapshots("2301.01000")) == 1
        assert len(repository.relevance_observations("2301.01000")) == 1


@pytest.mark.asyncio
async def test_default_resume_does_not_erase_failed_analysis_phase(
    settings: Settings, mocked: None
) -> None:
    chosen = options(end_date=date(2023, 1, 1))
    with Database(settings.database_path) as database:
        Repository(database).save_checkpoint(
            BackfillCheckpoint(
                job_id=backfill_job_id(settings, chosen),
                period_id="2022-W52",
                start_date=date(2023, 1, 1),
                end_date=date(2023, 1, 1),
                discovery_cutoff=datetime(2023, 1, 2, tzinfo=UTC),
                phase="analysis",
                status="failed",
                processed_ids=["stored-success"],
                analyses_done=["stored-success"],
                error_type="ValueError",
                updated_at=NOW,
            )
        )
    report = await run_backfill(settings, chosen, now=NOW)
    assert report.status == "failed" and report.failed_weeks == ["2022-W52"]
    assert report.completed_weeks == 0 and MockCollector.calls == []
    with Database(settings.database_path) as database:
        checkpoint = Repository(database).get_checkpoint(report.job_id, "2022-W52")
        assert checkpoint and checkpoint.status == "failed" and checkpoint.phase == "analysis"
        assert checkpoint.processed_ids == ["stored-success"]


@pytest.mark.asyncio
async def test_real_collector_with_mock_transport_and_completed_http_free_resume(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[httpx.Request] = []
    settings.openai_api_key = SecretStr("fixture-not-a-real-key")

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.url.host == "export.arxiv.org"
        return httpx.Response(
            200, text='<feed xmlns="http://www.w3.org/2005/Atom"><title>Empty</title></feed>'
        )

    monkeypatch.setattr(
        module, "OpenAIProvider", lambda *args: pytest.fail("Default must never construct OpenAI")
    )
    chosen = options(end_date=date(2023, 1, 1), skip_enrichment=True)
    transport = httpx.MockTransport(handle)
    first = await run_backfill(settings, chosen, now=NOW, transport=transport)
    second = await run_backfill(settings, chosen, now=NOW, transport=transport)
    assert first.status == second.status == "success"
    assert len(calls) == 1
    assert "submittedDate:[202301010000 TO 202301012359]" in calls[0].url.params["search_query"]


@pytest.mark.asyncio
async def test_confirmed_classification_uses_fake_provider_and_records_rejections(
    settings: Settings, mocked: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.openai_api_key = SecretStr("fixture-not-a-real-key")
    MockCollector.count = 2
    calls: list[str] = []
    closed: list[str] = []

    class FakeProvider:
        def __init__(self, settings: Settings, model: str) -> None:
            self.model = model

        async def generate_structured[T: BaseModel](
            self, *, system_prompt: str, user_prompt: str, schema: type[T]
        ) -> T:
            calls.append(user_prompt)
            return schema.model_validate(
                {
                    "relevance_score": 20 if "01001" in user_prompt else 90,
                    "reason": "Fixture assessment",
                    "confidence": 0.9,
                }
            )

        async def aclose(self) -> None:
            closed.append(self.model)

    monkeypatch.setattr(module, "OpenAIProvider", FakeProvider)
    chosen = options(end_date=date(2023, 1, 1), classify=True, confirm_expensive=True)
    report = await run_backfill(settings, chosen, now=NOW)
    assert report.status == "success" and report.llm_relevance_enabled
    assert len(calls) == 2 and closed == [settings.relevance_model]
    with Database(settings.database_path) as database:
        repository = Repository(database)
        accepted = repository.get_record("2301.01000")
        assert accepted and accepted.relevance and accepted.relevance.classifier == "llm"
        assert accepted.relevance_observed_at == NOW
        assert accepted.relevance_model == settings.relevance_model
        rejected = repository.relevance_observations("2301.01001")
        assert len(rejected) == 1 and rejected[0].relevance is None
        assert rejected[0].observed_at == NOW and rejected[0].primary_only
    await run_backfill(settings, chosen, now=NOW)
    assert len(calls) == 2 and len(closed) == 1
