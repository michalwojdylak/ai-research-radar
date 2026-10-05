from datetime import datetime

from research_radar.collectors.arxiv import ArxivCollector
from research_radar.models.ranking import PaperRecord
from research_radar.models.run import RunReport
from research_radar.pipeline.common import PAPER_ERRORS, report_failure
from research_radar.ranking.deduplication import deduplicate_groups
from research_radar.storage.repository import Repository


async def discover(
    repository: Repository,
    collector: ArxivCollector,
    report: RunReport,
    *,
    days: int,
    now: datetime,
) -> list[PaperRecord]:
    existing = repository.list_records()
    try:
        papers = await collector.discover(days=days, now=now)
    except PAPER_ERRORS as error:
        report_failure(report, "discover", error)
        return existing
    report.discovered = len(papers)
    old_records = {record.paper.id: record for record in existing}
    groups = deduplicate_groups([record.paper for record in existing] + papers)
    for group in groups:
        paper = group.survivor
        member_ids = {member.id for member in group.members}
        identities = sorted(
            [old_records[paper_id] for paper_id in member_ids if paper_id in old_records],
            key=lambda record: (
                record.paper.id == paper.id,
                record.paper.updated_at or record.paper.published_at,
                record.paper.id,
            ),
            reverse=True,
        )
        record = identities[0].model_copy(deep=True) if identities else PaperRecord(paper=paper)
        if record.paper.title != paper.title or record.paper.abstract != paper.abstract:
            record.relevance = None
        record.paper = paper
        repository.save_record(record)
        for old in identities:
            repository.merge_identity(old.paper.id, paper.id)
    return repository.list_records()
