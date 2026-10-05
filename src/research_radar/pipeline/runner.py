import logging
from datetime import UTC, datetime
from uuid import uuid4

import httpx

from research_radar.analysis.analyzer import AnalysisService
from research_radar.analysis.llm import OpenAIProvider
from research_radar.collectors.arxiv import ArxivCollector
from research_radar.config import Settings
from research_radar.enrichment.pipeline import EnrichmentPipeline
from research_radar.models.run import RunReport
from research_radar.pipeline.analyze import analyze
from research_radar.pipeline.common import PAPER_ERRORS, report_failure
from research_radar.pipeline.discover import discover
from research_radar.pipeline.enrich import enrich
from research_radar.pipeline.generate import generate, validate_generated
from research_radar.pipeline.rank import rank
from research_radar.publishing import create_review_pr
from research_radar.ranking.relevance import RelevanceFilter
from research_radar.storage.database import Database
from research_radar.storage.repository import Repository

logger = logging.getLogger(__name__)
STAGES = ("discover", "enrich", "rank", "analyze", "generate")


async def run_pipeline(
    settings: Settings,
    stage: str = "run",
    *,
    now: datetime | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> RunReport:
    if stage != "run" and stage not in STAGES:
        raise ValueError(f"Unknown pipeline stage: {stage}")
    now = now or datetime.now(UTC)
    now = now.replace(tzinfo=UTC) if now.tzinfo is None else now.astimezone(UTC)
    report = RunReport(id=str(uuid4()), started_at=now, stage=stage)
    providers: list[OpenAIProvider] = []
    relevance_provider: OpenAIProvider | None = None
    service: AnalysisService | None = None
    needs_llm = stage in {"run", "enrich", "analyze"}
    if settings.openai_api_key and needs_llm:
        relevance_provider = OpenAIProvider(settings, settings.relevance_model)
        providers.append(relevance_provider)
        if stage in {"run", "analyze"}:
            analysis_provider = OpenAIProvider(settings, settings.analysis_model)
            critic_provider = OpenAIProvider(settings, settings.critic_model)
            providers.extend([analysis_provider, critic_provider])
            service = AnalysisService(settings, analysis_provider, critic_provider)
    steps = STAGES if stage == "run" else (stage,)
    with Database(settings.database_path) as database:
        repository = Repository(database)
        repository.save_run(report)
        try:
            async with httpx.AsyncClient(
                timeout=settings.http_timeout,
                transport=transport,
                follow_redirects=False,
                limits=httpx.Limits(
                    max_connections=settings.concurrency * 2,
                    max_keepalive_connections=settings.concurrency,
                ),
                headers={"User-Agent": "AI-Research-Radar/0.1 (research discovery)"},
            ) as client:
                for step in steps:
                    if step == "discover":
                        await discover(
                            repository,
                            ArxivCollector(settings, client),
                            report,
                            days=settings.research_days,
                            now=now,
                        )
                    elif step == "enrich":
                        await enrich(
                            repository,
                            RelevanceFilter(settings, relevance_provider),
                            EnrichmentPipeline(settings, client),
                            settings,
                            report,
                            now=now,
                        )
                    elif step == "rank":
                        rank(repository, settings, report, now=now)
                    elif step == "analyze":
                        await analyze(repository, service, settings, report)
                    elif step == "generate":
                        paths = generate(repository, settings, report)
                        validate_generated(settings.data_dir)
                        if rankings := repository.latest_rankings():
                            ranking_path = (
                                settings.data_dir
                                / "rankings"
                                / (f"{rankings.timestamp:%Y-%m-%d}.json")
                            )
                            if ranking_path.exists():
                                paths.append(ranking_path)
                        if not settings.dry_run:
                            try:
                                url = await create_review_pr(settings, client, paths, now=now)
                                if url:
                                    logger.info("Draft review PR: %s", url)
                            except PAPER_ERRORS as error:
                                report_failure(report, "submit", error)
                report.status = (
                    "partial" if report.failures or report.skipped_analysis else "success"
                )
        finally:
            if report.status == "running":
                report.status = "failed"
            report.finished_at = datetime.now(UTC)
            repository.save_run(report)
            if service is not None:
                await service.aclose()
            for provider in providers:
                await provider.aclose()
    return report
