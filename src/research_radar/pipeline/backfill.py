"""Incremental, checkpointed historical discovery; paid work is always opt-in."""

import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, time, timedelta
from uuid import uuid4

import httpx
from pydantic import Field, model_validator
from rich.console import Console

from research_radar.analysis.llm import OpenAIProvider
from research_radar.collectors.arxiv import ArxivCollector
from research_radar.collectors.http import CachedHTTP
from research_radar.config import Settings
from research_radar.enrichment.pipeline import EnrichmentPipeline
from research_radar.models.history import (
    BackfillCheckpoint,
    BackfillPhase,
    BackfillReport,
    RelevanceObservation,
)
from research_radar.models.metrics import MetricsSnapshot
from research_radar.models.paper import RadarModel
from research_radar.models.ranking import PaperRecord
from research_radar.models.run import RunReport
from research_radar.periods import WeekPeriod, iter_weeks, utc, week_for
from research_radar.pipeline.common import PAPER_ERRORS, bounded_map, report_failure
from research_radar.pipeline.generate import generate
from research_radar.pipeline.weekly import _current_analysis, analyze_paper_ids, rebuild_week
from research_radar.ranking.relevance import RelevanceFilter
from research_radar.ranking.velocity import calculate_metrics
from research_radar.storage.database import Database
from research_radar.storage.repository import Repository

logger = logging.getLogger(__name__)
console = Console()
BACKFILL_PROTOCOL = "historical-v1"


class CostProtectionError(ValueError):
    """A paid historical phase was requested without explicit confirmation."""


class BackfillOptions(RadarModel):
    start_date: date
    end_date: date
    workers: int = Field(default=4, ge=1, le=16)
    skip_enrichment: bool = False
    analyze: bool = False
    confirm_expensive: bool = False
    resume: bool = True
    classify: bool = False
    publication_proxy: bool = False

    @model_validator(mode="after")
    def ordered_dates(self) -> "BackfillOptions":
        if self.start_date > self.end_date:
            raise ValueError("Start date must not follow end date")
        return self


def backfill_job_id(settings: Settings, options: BackfillOptions) -> str:
    """Changing the end or requesting analysis does not discard completed discovery."""
    configuration = {
        "protocol": BACKFILL_PROTOCOL,
        "start_date": options.start_date.isoformat(),
        "categories": sorted(settings.arxiv_categories),
        "classify": options.classify,
        "llm_available": bool(settings.openai_api_key) if options.classify else False,
        "skip_enrichment": options.skip_enrichment,
        "publication_proxy": options.publication_proxy,
        "relevance_threshold": settings.relevance_threshold,
        "relevance_model": settings.relevance_model if options.classify else None,
        "relevance_prompt": settings.relevance_prompt_version,
    }
    return hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:24]


def _ranking_fingerprint(settings: Settings, options: BackfillOptions) -> str:
    configuration = {
        "publication_proxy": options.publication_proxy,
        "ranking_version": settings.ranking_version,
        "scoring_version": settings.scoring_version,
        "weights": settings.weights.model_dump(),
        "radar_scoring": settings.radar_scoring.model_dump(),
        "retrospective_weights": settings.retrospective_weights.model_dump(),
        "weekly_limits": [
            settings.weekly_min_score,
            settings.weekly_min_papers,
            settings.weekly_max_papers,
        ],
        "analysis_model": settings.analysis_model,
        "critic_model": settings.critic_model,
        "analysis_prompt_version": settings.analysis_prompt_version,
        "critic_prompt_version": settings.critic_prompt_version,
    }
    return hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()


def _save(repository: Repository, checkpoint: BackfillCheckpoint, now: datetime) -> None:
    checkpoint.updated_at = now
    repository.save_checkpoint(checkpoint)


def _advance(
    repository: Repository,
    checkpoint: BackfillCheckpoint,
    phase: BackfillPhase,
    now: datetime,
) -> None:
    checkpoint.phase = phase
    checkpoint.status = "complete" if phase == "complete" else "running"
    checkpoint.processed_ids = []
    checkpoint.error_type = None
    _save(repository, checkpoint, now)


async def _paper_phase(
    repository: Repository,
    checkpoint: BackfillCheckpoint,
    records: list[PaperRecord],
    operation: Callable[[PaperRecord], Awaitable[None]],
    report: RunReport,
    *,
    workers: int,
    clock: Callable[[], datetime],
) -> bool:
    processed = set(checkpoint.processed_ids)
    failures: list[str] = []

    async def process(record: PaperRecord) -> None:
        if record.paper.id in processed:
            return
        try:
            await operation(record)
        except PAPER_ERRORS as error:
            report_failure(report, checkpoint.phase, error, record.paper.id)
            failures.append(type(error).__name__)
            return
        processed.add(record.paper.id)
        checkpoint.processed_ids.append(record.paper.id)
        _save(repository, checkpoint, clock())

    await bounded_map(records, process, concurrency=workers)
    if failures:
        checkpoint.status = "failed"
        checkpoint.error_type = failures[0]
        _save(repository, checkpoint, clock())
        return False
    return True


def _reusable_relevance(record: PaperRecord, settings: Settings, classify: bool) -> bool:
    if record.relevance is None:
        return False
    if not classify:
        return True
    return (
        record.relevance.classifier == "llm"
        and record.relevance_prompt_version == settings.relevance_prompt_version
        and record.relevance_model == settings.relevance_model
        and record.relevance_observed_at is not None
        and (record.paper.updated_at or record.paper.published_at) <= record.relevance_observed_at
    )


async def run_backfill(
    settings: Settings,
    options: BackfillOptions,
    *,
    now: datetime | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> BackfillReport:
    if options.classify and not options.confirm_expensive:
        raise CostProtectionError("Historical classification requires --confirm-expensive")
    if options.analyze and not options.confirm_expensive:
        raise CostProtectionError("Historical analysis requires --confirm-expensive")
    injected_now = utc(now) if now is not None else None

    def clock() -> datetime:
        return injected_now if injected_now is not None else datetime.now(UTC)

    moment = clock()
    end_date = min(options.end_date, moment.date())
    if options.start_date > end_date:
        raise ValueError("Historical start date cannot be in the future")
    settings = settings.model_copy(update={"concurrency": options.workers})
    report = BackfillReport(
        job_id=backfill_job_id(settings, options),
        start_date=options.start_date,
        end_date=options.end_date,
        total_weeks=(week_for(end_date).start_date - week_for(options.start_date).start_date).days
        // 7
        + 1,
        llm_relevance_enabled=options.classify and settings.openai_api_key is not None,
    )
    budget = settings.top_papers
    attempted_total = 0
    ranking_fingerprint = _ranking_fingerprint(settings, options)
    relevance_provider: OpenAIProvider | None = None
    try:
        with Database(settings.database_path) as database:
            repository = Repository(database)
            async with httpx.AsyncClient(
                timeout=settings.http_timeout, transport=transport, follow_redirects=False
            ) as client:
                shared_http = CachedHTTP(settings, client)
                collector = ArxivCollector(settings, client, http=shared_http)
                enrichment = EnrichmentPipeline(settings, client, http=shared_http)
                for period in iter_weeks(options.start_date, end_date):
                    start_day = max(period.start_date, options.start_date)
                    end_day = min(period.end_date, end_date)
                    start = datetime.combine(start_day, time.min, UTC)
                    end = min(datetime.combine(end_day + timedelta(days=1), time.min, UTC), moment)
                    checkpoint = (
                        repository.get_checkpoint(report.job_id, period.id)
                        if options.resume
                        else None
                    )
                    if checkpoint is not None and (
                        checkpoint.start_date != start_day
                        or checkpoint.end_date != end_day
                        or checkpoint.discovery_cutoff != end
                    ):
                        checkpoint = None
                    if checkpoint is None:
                        checkpoint = BackfillCheckpoint(
                            job_id=report.job_id,
                            period_id=period.id,
                            start_date=start_day,
                            end_date=end_day,
                            updated_at=clock(),
                            discovery_cutoff=end,
                        )
                    else:
                        console.print(f"Resuming {period.id}: {checkpoint.phase}")
                    if checkpoint.ranking_fingerprint != ranking_fingerprint and (
                        checkpoint.phase == "complete"
                        or (checkpoint.phase == "analysis" and options.analyze)
                    ):
                        _advance(repository, checkpoint, "ranking", clock())
                    if checkpoint.phase == "complete" and not options.analyze:
                        report.completed_weeks += 1
                        console.print(
                            f"{period.id} ✓ {report.completed_weeks}/{report.total_weeks}"
                        )
                        continue
                    if checkpoint.phase == "analysis" and not options.analyze:
                        if checkpoint.status == "failed":
                            report.failed_weeks.append(period.id)
                            console.print(f"{period.id} failed: analysis (not requested)")
                        else:
                            console.print(f"{period.id} analysis deferred: not requested")
                        continue
                    if checkpoint.phase == "complete":
                        _advance(repository, checkpoint, "analysis", clock())
                    checkpoint.status = "running"
                    checkpoint.error_type = None
                    _save(repository, checkpoint, clock())
                    run = RunReport(
                        id=str(uuid4()),
                        started_at=clock(),
                        stage="backfill",
                        period_id=period.id,
                    )
                    report.processed_weeks += 1

                    def rebuild(
                        current_period: WeekPeriod = period,
                        coverage_start: date = start_day,
                        coverage_end: date = end_day,
                        discovery_end: datetime = end,
                    ) -> list[str]:
                        radar, _ = rebuild_week(
                            repository,
                            settings,
                            current_period,
                            now=clock(),
                            is_preview=moment < current_period.end,
                            coverage_start=coverage_start,
                            coverage_end=coverage_end,
                            is_complete=(
                                coverage_start == current_period.start_date
                                and coverage_end == current_period.end_date
                                and discovery_end >= current_period.end
                            ),
                            discovery_complete=True,
                            allow_publication_proxy=options.publication_proxy,
                        )
                        return [paper.paper_id for paper in radar.papers]

                    try:
                        if checkpoint.phase == "discovery":
                            papers = await collector.discover_period(start=start, end=end)
                            repository.save_discovered(papers)
                            run.discovered = len(papers)
                            _advance(repository, checkpoint, "relevance", clock())
                        if checkpoint.phase == "relevance":
                            if (
                                options.classify
                                and settings.openai_api_key is not None
                                and relevance_provider is None
                            ):
                                relevance_provider = OpenAIProvider(
                                    settings, settings.relevance_model
                                )
                            relevance = RelevanceFilter(settings, relevance_provider)

                            async def classify(
                                record: PaperRecord, classifier: RelevanceFilter = relevance
                            ) -> None:
                                if _reusable_relevance(record, settings, options.classify):
                                    return
                                record.relevance = await classifier.classify(record.paper)
                                record.relevance_observed_at = clock()
                                record.relevance_primary_only = True
                                record.relevance_prompt_version = settings.relevance_prompt_version
                                record.relevance_model = (
                                    settings.relevance_model if classifier.provider else None
                                )
                                repository.save_relevance_observation(
                                    record.paper.id,
                                    RelevanceObservation(
                                        observed_at=record.relevance_observed_at,
                                        paper=record.paper.model_copy(deep=True),
                                        relevance=record.relevance,
                                        primary_only=True,
                                        model=record.relevance_model,
                                        prompt_version=record.relevance_prompt_version,
                                    ),
                                )
                                repository.save_record(record)

                            if not await _paper_phase(
                                repository,
                                checkpoint,
                                repository.records_between(start, end),
                                classify,
                                run,
                                workers=options.workers,
                                clock=clock,
                            ):
                                report.failed_weeks.append(period.id)
                                console.print(f"{period.id} failed: relevance")
                                continue
                            _advance(repository, checkpoint, "enrichment", clock())
                        if checkpoint.phase == "enrichment":

                            async def enrich_record(record: PaperRecord) -> None:
                                if (
                                    options.skip_enrichment
                                    or record.relevance is None
                                    or record.relevance.relevance_score
                                    < settings.relevance_threshold
                                    or (options.resume and repository.snapshots(record.paper.id))
                                ):
                                    return
                                data = await enrichment.enrich(record.paper)
                                observed = clock()
                                citation_source = (
                                    "openalex"
                                    if data.openalex_citations is not None
                                    and (
                                        data.semantic_scholar_citations is None
                                        or data.openalex_citations
                                        >= data.semantic_scholar_citations
                                    )
                                    else "semantic_scholar"
                                    if data.semantic_scholar_citations is not None
                                    else None
                                )
                                snapshot = MetricsSnapshot(
                                    timestamp=observed,
                                    citations=data.citations,
                                    github_stars=data.github_stars,
                                    hf_upvotes=data.hf_upvotes,
                                    github_url=data.github_url,
                                    citation_source=citation_source,
                                    hf_paper_url=data.hf_paper_url,
                                    hf_trending=data.hf_trending,
                                )
                                record.enrichment = data
                                record.metrics = calculate_metrics(
                                    data,
                                    [
                                        *[
                                            old
                                            for old in repository.snapshots(
                                                record.paper.id, as_of=observed
                                            )
                                            if old.timestamp != observed
                                        ],
                                        snapshot,
                                    ],
                                    observed,
                                )
                                repository.save_record(record)
                                repository.save_snapshot(record.paper.id, snapshot)

                            if not await _paper_phase(
                                repository,
                                checkpoint,
                                repository.records_between(start, end),
                                enrich_record,
                                run,
                                workers=options.workers,
                                clock=clock,
                            ):
                                report.failed_weeks.append(period.id)
                                console.print(f"{period.id} failed: enrichment")
                                continue
                            _advance(repository, checkpoint, "ranking", clock())
                        if checkpoint.phase == "ranking":
                            rebuild()
                            checkpoint.ranking_fingerprint = ranking_fingerprint
                            _advance(
                                repository,
                                checkpoint,
                                "analysis" if options.analyze else "complete",
                                clock(),
                            )
                        if checkpoint.phase == "analysis" and options.analyze:
                            saved_radar = repository.get_weekly(
                                period.id, preview=moment < period.end
                            )
                            selected = (
                                [paper.paper_id for paper in saved_radar.papers]
                                if saved_radar is not None
                                else rebuild()
                            )
                            deferred = False
                            analysis_failed = False
                            for paper_id in selected:
                                record = repository.get_record(paper_id)
                                if record is None:
                                    continue
                                if _current_analysis(
                                    record, repository.latest_analysis(paper_id), settings
                                ):
                                    if paper_id not in checkpoint.processed_ids:
                                        checkpoint.processed_ids.append(paper_id)
                                        _save(repository, checkpoint, clock())
                                    continue
                                if attempted_total >= budget:
                                    deferred = True
                                    continue
                                before = len(run.failures)
                                attempted = await analyze_paper_ids(
                                    repository,
                                    settings.model_copy(update={"top_papers": 1}),
                                    [paper_id],
                                    run,
                                )
                                attempted_total += len(attempted)
                                if len(run.failures) > before:
                                    analysis_failed = True
                                    checkpoint.error_type = run.failures[-1].error_type
                                    continue
                                if not attempted:
                                    deferred = True
                                    break
                                checkpoint.processed_ids.append(paper_id)
                                checkpoint.analyses_done = list(
                                    dict.fromkeys([*checkpoint.analyses_done, *attempted])
                                )
                                _save(repository, checkpoint, clock())
                            generate(repository, settings, run, paper_ids=selected)
                            if run.deep_analyzed:
                                rebuild()
                            report.deep_analyzed += run.deep_analyzed
                            if analysis_failed:
                                checkpoint.status = "failed"
                                report.failed_weeks.append(period.id)
                            elif deferred:
                                checkpoint.status = "pending"
                            else:
                                _advance(repository, checkpoint, "complete", clock())
                            _save(repository, checkpoint, clock())
                        if checkpoint.phase == "complete":
                            report.completed_weeks += 1
                            console.print(
                                f"{period.id} ✓ {report.completed_weeks}/{report.total_weeks}"
                            )
                        elif checkpoint.status == "failed":
                            console.print(f"{period.id} failed: analysis")
                        else:
                            console.print(f"{period.id} analysis deferred: invocation budget")
                    except PAPER_ERRORS as error:
                        checkpoint.status = "failed"
                        checkpoint.error_type = type(error).__name__
                        _save(repository, checkpoint, clock())
                        report_failure(run, checkpoint.phase, error)
                        report.failed_weeks.append(period.id)
                        console.print(f"{period.id} failed: {checkpoint.phase}")
                    finally:
                        run.finished_at = clock()
                        run.status = (
                            "partial" if run.failures or run.skipped_analysis else "success"
                        )
                        repository.save_run(run)
    finally:
        if relevance_provider is not None:
            await relevance_provider.aclose()
    if report.failed_weeks:
        report.status = "partial" if report.completed_weeks else "failed"
    elif report.completed_weeks < report.total_weeks:
        report.status = "partial"
    return report
