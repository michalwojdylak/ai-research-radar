"""Period-scoped orchestration above the unchanged deterministic paper scorer."""

import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import httpx

from research_radar.analysis.analyzer import AnalysisService, analysis_input_fingerprint
from research_radar.analysis.llm import OpenAIProvider
from research_radar.collectors.arxiv import ArxivCollector
from research_radar.config import Settings
from research_radar.enrichment.pipeline import EnrichmentPipeline
from research_radar.models.analysis import AnalysisResult, PromptVersions
from research_radar.models.radars import (
    MonthlyRadar,
    RadarStatus,
    RankingMode,
    WeeklyRadar,
    YearlyRadar,
)
from research_radar.models.ranking import PaperRecord
from research_radar.models.run import RunReport
from research_radar.periods import WeekPeriod, iter_weeks, parse_week, previous_week, utc, week_for
from research_radar.pipeline.analyze import analyze
from research_radar.pipeline.common import PAPER_ERRORS, report_failure
from research_radar.pipeline.enrich import enrich
from research_radar.pipeline.generate import generate, validate_generated
from research_radar.publishing import create_review_pr
from research_radar.radars.hierarchy import build_monthly, build_weekly, build_yearly
from research_radar.radars.output import write_radar
from research_radar.ranking.relevance import RelevanceFilter
from research_radar.ranking.scoring import score_record
from research_radar.ranking.time_aware import score_at
from research_radar.storage.database import Database
from research_radar.storage.repository import Repository

logger = logging.getLogger(__name__)


def _current_analysis(
    record: PaperRecord, analysis: AnalysisResult | None, settings: Settings
) -> bool:
    return (
        analysis is not None
        and analysis.input_fingerprint == analysis_input_fingerprint(record)
        and analysis.model == settings.analysis_model
        and analysis.critic_model == settings.critic_model
        and analysis.prompt_versions
        == PromptVersions(
            relevance=settings.relevance_prompt_version,
            analysis=settings.analysis_prompt_version,
            critic=settings.critic_prompt_version,
        )
    )


def _configuration[T: WeeklyRadar | MonthlyRadar | YearlyRadar](radar: T, settings: Settings) -> T:
    if radar.status in {"approved", "published"}:
        return radar
    radar.paper_scoring_weights = settings.weights.model_copy(deep=True)
    radar.retrospective_weights = settings.retrospective_weights.model_copy(deep=True)
    return radar


def rebuild_week(
    repository: Repository,
    settings: Settings,
    period: WeekPeriod,
    *,
    now: datetime,
    mode: RankingMode = "publication",
    is_preview: bool = False,
    coverage_start: date | None = None,
    coverage_end: date | None = None,
    is_complete: bool = True,
    allow_publication_proxy: bool = False,
    discovery_complete: bool | None = None,
) -> tuple[WeeklyRadar, list[Path]]:
    now = utc(now)
    if period.start > now:
        raise ValueError("Cannot generate a future publication week")
    is_preview = is_preview or now < period.end
    start = coverage_start or period.start_date
    end = min(coverage_end or period.end_date, now.date())
    cutoff = min(period.end, now) if mode == "publication" else now
    existing = repository.get_weekly(period.id, mode=mode, preview=is_preview)
    discovery_known = is_complete if discovery_complete is None else discovery_complete
    if existing and not existing.is_preview:
        if (
            discovery_known
            and existing.discovery_complete
            and (
                start <= existing.coverage_end + timedelta(days=1)
                and existing.coverage_start <= end + timedelta(days=1)
            )
        ):
            start, end = min(start, existing.coverage_start), max(end, existing.coverage_end)
        elif not discovery_known and existing.discovery_complete:
            start, end = existing.coverage_start, existing.coverage_end
            discovery_known = True
    complete = (
        not is_preview and start == period.start_date and end == period.end_date and discovery_known
    )
    records = repository.records_between(
        period.start, min(period.end, now + timedelta(microseconds=1))
    )
    scored = [
        score_at(
            record,
            repository.snapshots(record.paper.id, as_of=cutoff),
            data_cutoff=cutoff,
            weights=settings.weights,
            mode=mode,
            observations=repository.relevance_observations(record.paper.id, as_of=now),
            allow_publication_proxy=allow_publication_proxy,
            observation_cutoff=now,
            retrospective_weights=settings.retrospective_weights,
            relevance_threshold=settings.relevance_threshold,
        )
        for record in records
    ]
    eligible_records: list[PaperRecord] = []
    eligible_scores = []
    statuses: dict[str, RadarStatus] = {}
    for record, score in zip(records, scored, strict=True):
        saved = record.scores
        current = score_record(record, settings.weights, now)
        current.scored_at = now
        current.publication_score = saved.publication_score
        current.publication_data_cutoff = saved.publication_data_cutoff
        current.publication_basis = saved.publication_basis
        current.retrospective_score = saved.retrospective_score
        current.retrospective_data_cutoff = saved.retrospective_data_cutoff
        current.retrospective_breakdown = saved.retrospective_breakdown
        if mode == "publication":
            current.publication_score = score.publication_score
            current.publication_data_cutoff = score.data_cutoff
            current.publication_basis = score.quality_basis
        else:
            current.retrospective_score = score.retrospective_score
            current.retrospective_data_cutoff = score.data_cutoff
            current.retrospective_breakdown = score.scores.retrospective_breakdown
            current.publication_score = score.publication_score
            current.publication_data_cutoff = score.scores.publication_data_cutoff
            current.publication_basis = score.scores.publication_basis
        record.scores = current
        repository.save_record(record)
        if score.scores.relevance_score is not None and (
            score.scores.relevance_score < settings.relevance_threshold
        ):
            continue
        display = record.model_copy(deep=True)
        if score.source_paper is not None:
            display.paper = score.source_paper.model_copy(
                update={
                    "id": record.paper.id,
                    "published_at": record.paper.published_at,
                }
            )
        eligible_records.append(display)
        eligible_scores.append(score)
        analysis = repository.latest_analysis(record.paper.id)
        if _current_analysis(record, analysis, settings) and analysis is not None:
            statuses[record.paper.id] = analysis.status
    radar = build_weekly(
        period,
        eligible_records,
        eligible_scores,
        settings,
        now=now,
        data_cutoff=cutoff,
        mode=mode,
        coverage_start=start,
        coverage_end=end,
        is_preview=is_preview,
        is_complete=complete,
        analysis_statuses=statuses,
    )
    if not radar.papers and radar.scored_candidate_count < radar.candidate_count:
        radar.radar_score = None
        radar.retrospective_score = None
        radar.status = "draft"
        radar.explanation = (
            "No qualifying selection can be certified: some candidate scores are unknown."
        )
    radar.discovery_complete = discovery_known
    by_id = {record.paper.id: record for record in eligible_records}
    for selection in radar.papers:
        selection.paper_version = by_id[selection.paper_id].paper.arxiv_id
    if existing and existing.status in {"approved", "published"}:
        radar = existing
    radar, paths = write_radar(_configuration(radar, settings), settings)
    radar = repository.save_weekly(radar)
    if radar.is_preview:
        return radar, paths
    year, month = period.month_key
    weeks = repository.weeks_for_month(year, month, mode=mode)
    monthly = build_monthly(year, month, weeks, settings, now=now, mode=mode)
    previous_month = repository.get_monthly(monthly.id, mode=mode)
    if previous_month and previous_month.status in {"approved", "published"}:
        monthly = previous_month
    monthly, month_paths = write_radar(_configuration(monthly, settings), settings)
    repository.save_monthly(monthly)
    yearly = build_yearly(
        year,
        repository.list_monthly(year=year, mode=mode),
        repository.list_weekly(year=year, mode=mode),
        settings,
        now=now,
        mode=mode,
    )
    previous_year = repository.get_yearly(yearly.id, mode=mode)
    if previous_year and previous_year.status in {"approved", "published"}:
        yearly = previous_year
    yearly, year_paths = write_radar(_configuration(yearly, settings), settings)
    repository.save_yearly(yearly)
    return radar, paths + month_paths + year_paths


async def analyze_paper_ids(
    repository: Repository,
    settings: Settings,
    paper_ids: list[str],
    report: RunReport,
    *,
    service: AnalysisService | None = None,
) -> list[str]:
    targets: list[str] = []
    for paper_id in dict.fromkeys(paper_ids):
        record = repository.get_record(paper_id)
        if record and not _current_analysis(record, repository.latest_analysis(paper_id), settings):
            targets.append(paper_id)
    targets = targets[: settings.top_papers]
    if not targets:
        return []
    if service is not None:
        await analyze(repository, service, settings, report, paper_ids=targets)
        return targets
    if settings.openai_api_key is None:
        await analyze(repository, None, settings, report, paper_ids=targets)
        return []
    analyst = OpenAIProvider(settings, settings.analysis_model)
    critic = OpenAIProvider(settings, settings.critic_model)
    owned_service = AnalysisService(settings, analyst, critic)
    try:
        await analyze(repository, owned_service, settings, report, paper_ids=targets)
    finally:
        await owned_service.aclose()
        await analyst.aclose()
        await critic.aclose()
    return targets


async def run_weekly(
    settings: Settings,
    *,
    period_id: str | None = None,
    preview: bool = False,
    now: datetime | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> RunReport:
    injected_now = utc(now) if now is not None else None

    def clock() -> datetime:
        return injected_now if injected_now is not None else datetime.now(UTC)

    now = clock()
    period = (
        parse_week(period_id) if period_id else week_for(now) if preview else previous_week(now)
    )
    if not preview and period.end > now:
        raise ValueError("Normal weekly runs require a complete ISO week; use preview-week")
    report = RunReport(
        id=str(uuid4()), started_at=now, stage="weekly", period_id=period.id, is_preview=preview
    )
    relevance_provider = (
        OpenAIProvider(settings, settings.relevance_model) if settings.openai_api_key else None
    )
    with Database(settings.database_path) as database:
        repository = Repository(database)
        repository.save_run(report)
        try:
            async with httpx.AsyncClient(
                timeout=settings.http_timeout, transport=transport, follow_redirects=False
            ) as client:
                discovery_ok = False
                try:
                    papers = await ArxivCollector(settings, client).discover_period(
                        start=period.start, end=min(period.end, now)
                    )
                    repository.save_discovered(papers)
                    report.discovered = len(papers)
                    discovery_ok = True
                except PAPER_ERRORS as error:
                    report_failure(report, "discover", error)
                await enrich(
                    repository,
                    RelevanceFilter(settings, relevance_provider),
                    EnrichmentPipeline(settings, client),
                    settings,
                    report,
                    now=now,
                    records=repository.records_between(period.start, min(period.end, now)),
                    observed_clock=clock,
                )
                radar, paths = rebuild_week(
                    repository,
                    settings,
                    period,
                    now=clock(),
                    is_preview=preview,
                    is_complete=discovery_ok,
                    allow_publication_proxy=settings.publication_proxy,
                )
                report.ranked = radar.scored_candidate_count
                selected = [paper.paper_id for paper in radar.papers]
                if not preview:
                    await analyze_paper_ids(repository, settings, selected, report)
                    paths += generate(repository, settings, report, paper_ids=selected)
                    radar, changed = rebuild_week(
                        repository,
                        settings,
                        period,
                        now=clock(),
                        is_complete=discovery_ok,
                        allow_publication_proxy=settings.publication_proxy,
                    )
                    paths += changed
                validate_generated(settings.data_dir)
                if not settings.dry_run and not preview:
                    try:
                        url = await create_review_pr(settings, client, paths, now=now)
                        if url:
                            logger.info("Weekly review PR: %s", url)
                    except PAPER_ERRORS as error:
                        report_failure(report, "submit", error)
            report.status = "partial" if report.failures or report.skipped_analysis else "success"
        finally:
            if report.status == "running":
                report.status = "failed"
            report.finished_at = datetime.now(UTC)
            repository.save_run(report)
            if relevance_provider is not None:
                await relevance_provider.aclose()
    return report


async def analyze_selected(
    settings: Settings,
    *,
    period_id: str | None = None,
    best_weeks: int | None = None,
    mode: RankingMode = "publication",
    confirm_expensive: bool = False,
    now: datetime | None = None,
) -> RunReport:
    from research_radar.pipeline.backfill import CostProtectionError

    if (period_id is None) == (best_weeks is None):
        raise ValueError("Select either one period or a number of best weeks")
    if settings.top_papers > 10 and not confirm_expensive:
        raise CostProtectionError("More than ten historical analyses require --confirm-expensive")
    if best_weeks is not None and best_weeks < 1:
        raise ValueError("--best-weeks must be positive")
    now = utc(now or datetime.now(UTC))
    report = RunReport(
        id=str(uuid4()), started_at=now, stage="analyze-selected", period_id=period_id
    )
    with Database(settings.database_path) as database:
        repository = Repository(database)
        if period_id:
            parse_week(period_id)
            period = repository.get_weekly(period_id, mode=mode)
            if period is None:
                raise ValueError("Requested Radar does not exist; populate its week first")
            radars = [period]
        else:
            radars = sorted(
                [x for x in repository.list_weekly(mode=mode) if x.ranking_score is not None],
                key=lambda x: (-float(x.ranking_score or 0), x.id),
            )[:best_weeks]
        paper_ids = list(dict.fromkeys(p.paper_id for radar in radars for p in radar.papers))
        repository.save_run(report)
        try:
            await analyze_paper_ids(repository, settings, paper_ids, report)
            generate(repository, settings, report, paper_ids=paper_ids)
            for radar in radars:
                rebuild_week(
                    repository,
                    settings,
                    parse_week(radar.id),
                    now=now,
                    mode=mode,
                    coverage_start=radar.coverage_start,
                    coverage_end=radar.coverage_end,
                    is_complete=radar.is_complete,
                    allow_publication_proxy=settings.publication_proxy,
                )
            report.status = "partial" if report.failures or report.skipped_analysis else "success"
        finally:
            if report.status == "running":
                report.status = "failed"
            report.finished_at = datetime.now(UTC)
            repository.save_run(report)
    return report


async def run_retrospective(
    settings: Settings, *, start_date: date, end_date: date, now: datetime | None = None
) -> list[WeeklyRadar]:
    now = utc(now or datetime.now(UTC))
    if end_date > now.date():
        raise ValueError("Retrospective range cannot extend into the future")
    radars: list[WeeklyRadar] = []
    with Database(settings.database_path) as database:
        repository = Repository(database)
        for period in iter_weeks(start_date, end_date):
            prior = repository.get_weekly(period.id)
            radar, _ = rebuild_week(
                repository,
                settings,
                period,
                now=now,
                mode="retrospective",
                coverage_start=max(start_date, period.start_date),
                coverage_end=min(end_date, period.end_date),
                is_complete=bool(prior and prior.is_complete),
                allow_publication_proxy=settings.publication_proxy,
            )
            radars.append(radar)
    return radars
