import logging

from research_radar.analysis.analyzer import AnalysisService
from research_radar.config import Settings
from research_radar.models.ranking import PaperRecord
from research_radar.models.run import RunReport
from research_radar.pipeline.common import PAPER_ERRORS, bounded_map, report_failure
from research_radar.storage.repository import Repository

logger = logging.getLogger(__name__)


async def analyze(
    repository: Repository,
    service: AnalysisService | None,
    settings: Settings,
    report: RunReport,
    *,
    paper_ids: list[str] | None = None,
) -> None:
    if paper_ids is None:
        rankings = repository.latest_rankings()
        if rankings is None:
            logger.warning("No rankings available; run rank before analyze.")
            return
        paper_ids = rankings.selected_ids
    records = [
        record
        for paper_id in paper_ids[: settings.top_papers]
        if (record := repository.get_record(paper_id)) is not None
    ]
    if service is None:
        report.skipped_analysis = len(records)
        logger.warning("OPENAI_API_KEY unavailable: deep analysis skipped, not fabricated.")
        return

    async def process(record: PaperRecord) -> None:
        try:
            result = await service.analyze(record)
            repository.save_analysis(result)
            report.deep_analyzed += 1
            if result.critic.passed:
                report.critic_passed += 1
            if result.status == "needs_review":
                report.needs_review += 1
        except PAPER_ERRORS as error:
            report_failure(report, "analyze", error, record.paper.id)

    await bounded_map(records, process, concurrency=settings.concurrency)
