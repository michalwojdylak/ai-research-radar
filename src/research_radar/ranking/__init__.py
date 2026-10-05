"""Deterministic identity, metric, and ranking calculations."""

from research_radar.ranking.deduplication import (
    PaperIdentityGroup,
    canonical_arxiv_id,
    deduplicate,
    deduplicate_groups,
    normalize_doi,
    normalize_title,
)
from research_radar.ranking.ranking import rank_records
from research_radar.ranking.scoring import freshness_multiplier, score_record
from research_radar.ranking.velocity import calculate_metrics

__all__ = [
    "PaperIdentityGroup",
    "calculate_metrics",
    "canonical_arxiv_id",
    "deduplicate",
    "deduplicate_groups",
    "freshness_multiplier",
    "normalize_doi",
    "normalize_title",
    "rank_records",
    "score_record",
]
