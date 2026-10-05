import os
from datetime import datetime
from pathlib import Path

from research_radar.config import Settings
from research_radar.models.ranking import Rankings
from research_radar.models.run import RunReport
from research_radar.ranking.ranking import rank_records
from research_radar.ranking.velocity import calculate_metrics
from research_radar.storage.repository import Repository


def atomic_text(path: Path, content: str) -> None:
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def rank(
    repository: Repository, settings: Settings, report: RunReport, *, now: datetime
) -> Rankings:
    records = [
        record
        for record in repository.list_records()
        if record.relevance is not None
        and record.relevance.relevance_score >= settings.relevance_threshold
    ]
    for record in records:
        record.metrics = calculate_metrics(
            record.enrichment, repository.snapshots(record.paper.id, as_of=now), now
        )
    rankings = rank_records(records, top=settings.top_papers, now=now, weights=settings.weights)
    for record in records:
        repository.save_record(record)
    repository.save_rankings(rankings)
    report.ranked = sum(
        record.scores.importance is not None
        or record.scores.trending is not None
        or record.scores.overall is not None
        for record in records
    )
    atomic_text(
        settings.data_dir / "rankings" / f"{now:%Y-%m-%d}.json",
        rankings.model_dump_json(indent=2) + "\n",
    )
    return rankings
