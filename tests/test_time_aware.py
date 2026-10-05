from datetime import UTC, datetime, timedelta

import pytest

from research_radar.config import ScoringWeights
from research_radar.models.history import RelevanceObservation
from research_radar.models.metrics import MetricsSnapshot, PaperMetrics
from research_radar.models.paper import Enrichment, Paper
from research_radar.models.ranking import (
    PaperRecord,
    QualityDimensions,
    RelevanceResult,
    ScoreDimension,
)
from research_radar.ranking.time_aware import score_at

PUBLISHED = datetime(2023, 4, 10, tzinfo=UTC)
CUTOFF = datetime(2023, 4, 17, tzinfo=UTC)
TODAY = datetime(2026, 10, 5, tzinfo=UTC)


def record(score: float = 90) -> PaperRecord:
    dimension = ScoreDimension(score=score, confidence=0.8, reason="Primary paper evidence.")
    return PaperRecord(
        paper=Paper(
            id="2304.00001",
            arxiv_id="2304.00001v1",
            title="Original neural method",
            abstract="Neural representations.",
            authors=["Researcher"],
            published_at=PUBLISHED,
            updated_at=PUBLISHED,
            categories=["cs.LG"],
            pdf_url="https://arxiv.org/pdf/2304.00001v1",
        ),
        relevance=RelevanceResult(
            relevance_score=90,
            confidence=0.8,
            reason="AI method",
            quality=QualityDimensions(
                scientific_impact=dimension,
                novelty=dimension,
                empirical_quality=dimension,
                practical_relevance=dimension,
            ),
        ),
        enrichment=Enrichment(
            openalex_citations=10000,
            github_stars=100000,
            hf_upvotes=10000,
            hf_trending=True,
        ),
        metrics=PaperMetrics(
            citations=10000,
            github_stars=100000,
            hf_upvotes=10000,
            github_star_velocity_24h=100000,
            observed_at=TODAY,
        ),
    )


def observation(
    paper: PaperRecord, moment: datetime = PUBLISHED + timedelta(days=1)
) -> RelevanceObservation:
    return RelevanceObservation(
        observed_at=moment,
        paper=paper.paper.model_copy(deep=True),
        relevance=paper.relevance.model_copy(deep=True) if paper.relevance else None,
        primary_only=True,
        model="historical-test",
        prompt_version="v1",
    )


def test_current_popularity_and_trending_flag_do_not_leak() -> None:
    paper = record()
    previous = paper.model_copy(deep=True)
    future = MetricsSnapshot(
        timestamp=TODAY,
        citations=10000,
        github_stars=100000,
        hf_upvotes=10000,
        hf_trending=True,
        github_url="https://github.com/author/repo",
        citation_source="openalex",
        hf_paper_url="https://huggingface.co/papers/2304.00001",
    )
    scored = score_at(
        paper,
        [future],
        data_cutoff=CUTOFF,
        weights=ScoringWeights(),
        observations=[observation(paper)],
    )
    assert scored.publication_score == pytest.approx(90)
    assert scored.scores.trending is None
    assert scored.scores.overall_breakdown.coverage == pytest.approx(0.7)
    assert (
        next(
            x for x in scored.scores.importance_breakdown.components if x.name == "citation_signal"
        ).value
        is None
    )
    assert paper == previous


def test_current_counts_cannot_dominate_publication_ranking() -> None:
    high, low = record(), record()
    high.metrics.citations, low.metrics.citations = 10000, 100
    values = [
        score_at(
            paper,
            [MetricsSnapshot(timestamp=TODAY, citations=paper.metrics.citations)],
            data_cutoff=CUTOFF,
            weights=ScoringWeights(),
            observations=[observation(paper)],
        )
        for paper in (high, low)
    ]
    assert values[0].publication_score == values[1].publication_score == 90


def test_later_quality_estimates_require_explicit_proxy() -> None:
    paper = record()
    paper.relevance_observed_at = TODAY
    paper.relevance_primary_only = True
    strict = score_at(paper, [], data_cutoff=CUTOFF, weights=ScoringWeights())
    assert strict.publication_score is None
    assert strict.quality_basis == "unavailable"
    proxy = score_at(
        paper,
        [],
        data_cutoff=CUTOFF,
        weights=ScoringWeights(),
        allow_publication_proxy=True,
        observation_cutoff=TODAY,
    )
    assert proxy.publication_score == 90
    assert proxy.quality_basis == "publication_document_proxy"
    assert proxy.quality_observed_at == TODAY


def test_later_paper_revisions_are_not_publication_documents() -> None:
    paper = record()
    old = observation(paper)
    paper.paper.updated_at = TODAY
    paper.paper.title = "Future title with new experiments"
    paper.paper.abstract = "New 2026 experiments."
    paper.paper.arxiv_id = "2304.00001v4"
    paper.relevance_observed_at = TODAY
    paper.relevance_primary_only = True
    unknown = score_at(
        paper,
        [],
        data_cutoff=CUTOFF,
        weights=ScoringWeights(),
        allow_publication_proxy=True,
        observation_cutoff=TODAY,
    )
    assert unknown.publication_score is None
    historical = score_at(
        paper,
        [],
        data_cutoff=CUTOFF,
        weights=ScoringWeights(),
        observations=[old],
    )
    assert historical.publication_score == 90
    assert historical.source_paper is not None
    assert historical.source_paper.title == "Original neural method"
    assert historical.source_paper.arxiv_id == "2304.00001v1"


def test_future_quality_observation_does_not_replace_old_quality() -> None:
    paper = record(80)
    old = observation(paper)
    future = observation(record(100), TODAY)
    result = score_at(
        paper,
        [],
        data_cutoff=CUTOFF,
        weights=ScoringWeights(),
        observations=[old, future],
    )
    assert result.publication_score == 80
    assert result.quality_observed_at == old.observed_at


def test_retrospective_is_separate_from_publication_score() -> None:
    paper = record()
    observations = [observation(paper)]
    snapshots = [
        MetricsSnapshot(timestamp=TODAY, citations=10000, github_stars=100000),
    ]
    result = score_at(
        paper,
        snapshots,
        data_cutoff=TODAY,
        weights=ScoringWeights(),
        observations=observations,
        mode="retrospective",
    )
    assert result.retrospective_score is not None and result.retrospective_score > 90
    assert result.publication_score == 90
    assert result.scores.publication_score == 90
    assert result.mode == "retrospective"


def test_historical_rejection_is_not_overridden_by_current_relevance() -> None:
    paper = record()
    rejected = observation(paper)
    rejected.relevance = None
    result = score_at(
        paper,
        [MetricsSnapshot(timestamp=CUTOFF, github_stars=100000, hf_upvotes=500)],
        data_cutoff=CUTOFF,
        weights=ScoringWeights(),
        observations=[rejected],
    )
    assert result.publication_score is None
    assert result.scores.relevance_score == 0


def test_cutoff_keeps_real_early_observations_only() -> None:
    paper = record()
    snapshots = [
        MetricsSnapshot(timestamp=CUTOFF, citations=2, github_stars=50, hf_trending=False),
        MetricsSnapshot(timestamp=TODAY, citations=10000, github_stars=100000, hf_trending=True),
    ]
    result = score_at(
        paper,
        snapshots,
        data_cutoff=CUTOFF,
        weights=ScoringWeights(),
        observations=[observation(paper)],
    )
    citation = next(
        x for x in result.scores.importance_breakdown.components if x.name == "citation_signal"
    )
    assert citation.value == pytest.approx(100 * 2 / 102)
    community = next(
        x for x in result.scores.trending_breakdown.components if x.name == "community_attention"
    )
    assert community.value == 0
