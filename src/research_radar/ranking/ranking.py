"""Stable rankings and a single bounded analysis shortlist."""

from datetime import datetime

from research_radar.config import ScoringWeights
from research_radar.models.ranking import PaperRecord, RankingEntry, Rankings
from research_radar.ranking.scoring import score_record


def _ordered(records: list[PaperRecord], score_name: str) -> list[PaperRecord]:
    eligible = [record for record in records if getattr(record.scores, score_name) is not None]
    return sorted(
        eligible, key=lambda record: (-getattr(record.scores, score_name), record.paper.id)
    )


def _entries(records: list[PaperRecord], score_name: str) -> list[RankingEntry]:
    return [
        RankingEntry(
            paper_id=record.paper.id,
            title=record.paper.title,
            rank=rank,
            score=getattr(record.scores, score_name),
            coverage=getattr(record.scores, f"{score_name}_breakdown").coverage,
        )
        for rank, record in enumerate(records, start=1)
    ]


def rank_records(
    records: list[PaperRecord], *, top: int, now: datetime, weights: ScoringWeights
) -> Rankings:
    """Update every record's scores and select at most ``top`` distinct IDs.

    Chart lists each contain up to top known scores, with descending score and
    ascending stable ID ties. Selection takes supported overall scores first,
    then interleaves importance/trending ranks, skipping already selected IDs.
    Neither publication age nor a missing chart excludes a supported paper.
    """
    if top < 0:
        raise ValueError("top must be nonnegative")
    for record in records:
        record.scores = score_record(record, weights, now)
    unique = {
        record.paper.id: record
        for record in sorted(
            records,
            key=lambda item: (
                item.paper.updated_at or item.paper.published_at,
                item.model_dump_json(),
            ),
        )
    }
    candidates = list(unique.values())
    important = _ordered(candidates, "importance")
    trending = _ordered(candidates, "trending")
    overall = _ordered(candidates, "overall")
    selected: list[str] = []

    def select(record: PaperRecord) -> None:
        if len(selected) < top and record.paper.id not in selected:
            selected.append(record.paper.id)

    for record in overall:
        select(record)
    for index in range(max(len(important), len(trending))):
        if index < len(important):
            select(important[index])
        if index < len(trending):
            select(trending[index])
    return Rankings(
        timestamp=now,
        trending=_entries(trending[:top], "trending"),
        most_important=_entries(important[:top], "importance"),
        selected_ids=selected,
    )
