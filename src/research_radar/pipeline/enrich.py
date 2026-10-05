from collections.abc import Callable
from datetime import datetime

from research_radar.config import Settings
from research_radar.enrichment.pipeline import EnrichmentPipeline
from research_radar.models.history import RelevanceObservation
from research_radar.models.metrics import MetricsSnapshot
from research_radar.models.ranking import PaperRecord
from research_radar.models.run import RunReport
from research_radar.pipeline.common import PAPER_ERRORS, bounded_map, report_failure
from research_radar.ranking.relevance import RelevanceFilter
from research_radar.ranking.velocity import calculate_metrics
from research_radar.storage.repository import Repository


async def enrich(
    repository: Repository,
    relevance_filter: RelevanceFilter,
    enrichment: EnrichmentPipeline,
    settings: Settings,
    report: RunReport,
    *,
    now: datetime,
    records: list[PaperRecord] | None = None,
    observed_clock: Callable[[], datetime] | None = None,
) -> list[PaperRecord]:
    async def process(record: PaperRecord) -> PaperRecord | None:
        try:
            record.relevance = await relevance_filter.classify(record.paper)
            relevance_observed_at = observed_clock() if observed_clock else now
            record.relevance_observed_at = relevance_observed_at
            record.relevance_primary_only = True
            record.relevance_prompt_version = settings.relevance_prompt_version
            record.relevance_model = (
                settings.relevance_model if relevance_filter.provider is not None else None
            )
            repository.save_record(record)
            repository.save_relevance_observation(
                record.paper.id,
                RelevanceObservation(
                    observed_at=relevance_observed_at,
                    paper=record.paper.model_copy(deep=True),
                    relevance=record.relevance,
                    model=record.relevance_model,
                    prompt_version=record.relevance_prompt_version,
                    primary_only=True,
                ),
            )
            if record.relevance is None:
                return None
            if record.relevance.relevance_score < settings.relevance_threshold:
                return None
            report.relevant += 1
            record.enrichment = await enrichment.enrich(record.paper)
            data = record.enrichment
            citation_source = (
                "openalex"
                if data.openalex_citations is not None
                and (
                    data.semantic_scholar_citations is None
                    or data.openalex_citations >= data.semantic_scholar_citations
                )
                else "semantic_scholar"
                if data.semantic_scholar_citations is not None
                else None
            )
            metric_observed_at = observed_clock() if observed_clock else now
            snapshot = MetricsSnapshot(
                timestamp=metric_observed_at,
                citations=data.citations,
                github_stars=data.github_stars,
                hf_upvotes=data.hf_upvotes,
                github_url=data.github_url,
                citation_source=citation_source,
                hf_paper_url=data.hf_paper_url,
                hf_trending=data.hf_trending,
            )
            repository.save_snapshot(record.paper.id, snapshot)
            record.metrics = calculate_metrics(
                data,
                repository.snapshots(record.paper.id, as_of=metric_observed_at),
                metric_observed_at,
            )
            repository.save_record(record)
            if data.sources:
                report.enriched += 1
            return record
        except PAPER_ERRORS as error:
            report_failure(report, "enrich", error, record.paper.id)
            return None

    values = await bounded_map(
        records if records is not None else repository.list_records(),
        process,
        concurrency=settings.concurrency,
    )
    return [record for record in values if record is not None]
