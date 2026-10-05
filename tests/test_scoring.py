from datetime import UTC, datetime, timedelta

import pytest

from research_radar.config import ImportanceWeights, OverallWeights, ScoringWeights, TrendingWeights
from research_radar.models.metrics import PaperMetrics
from research_radar.models.paper import Enrichment, Paper
from research_radar.models.ranking import (
    PaperRecord,
    QualityDimensions,
    RelevanceResult,
    ScoreDimension,
)
from research_radar.ranking.scoring import freshness_multiplier, saturating_score, score_record

NOW = datetime(2026, 10, 5, 12, tzinfo=UTC)


def record(
    *, quality: QualityDimensions | None = None, metrics: PaperMetrics | None = None
) -> PaperRecord:
    return PaperRecord(
        paper=Paper(
            id="2601.00001",
            title="Research",
            abstract="Abstract",
            authors=["Researcher"],
            categories=["cs.AI"],
            published_at=NOW - timedelta(days=1),
        ),
        relevance=RelevanceResult(
            relevance_score=90,
            reason="Relevant",
            confidence=0.9,
            quality=quality or QualityDimensions(),
        ),
        metrics=metrics or PaperMetrics(),
    )


def dimension(score: float, confidence: float = 0.9) -> ScoreDimension:
    return ScoreDimension(score=score, reason="Source-supported assessment", confidence=confidence)


@pytest.mark.parametrize(
    ("age", "expected"),
    [
        (-1, 1.5),
        (0, 1.5),
        (2, 1.5),
        (2.999, 1.5),
        (3, 1.25),
        (7.999, 1.25),
        (8, 1.1),
        (14.999, 1.1),
        (15, 1),
        (30.999, 1),
        (31, 0.8),
        (3650, 0.8),
    ],
)
def test_exact_freshness_bins(age: float, expected: float) -> None:
    assert freshness_multiplier(NOW - timedelta(days=age), NOW) == expected
    assert freshness_multiplier((NOW - timedelta(days=age)).replace(tzinfo=None), NOW) == expected


def test_fixed_saturation_and_nonpositive_growth() -> None:
    assert saturating_score(None, 250) is None
    assert saturating_score(0, 250) == 0
    assert saturating_score(-100, 250) == 0
    assert saturating_score(250, 250) == 50
    assert saturating_score(100000, 250) == pytest.approx(99.75062344)


def test_reweights_observed_dimensions_and_exposes_coverage() -> None:
    item = record(
        quality=QualityDimensions(
            scientific_impact=dimension(80),
            novelty=dimension(60),
        )
    )
    result = score_record(item, ScoringWeights(), NOW)
    assert result.importance == pytest.approx((80 * 0.30 + 60 * 0.25) / 0.55)
    assert result.importance_breakdown.coverage == pytest.approx(0.55)
    components = {component.name: component for component in result.importance_breakdown.components}
    assert components["scientific_impact"].weight == 0.30
    assert components["scientific_impact"].contribution == pytest.approx(80 * 0.30 / 0.55)
    assert components["empirical_quality"].value is None
    assert components["empirical_quality"].contribution is None
    assert result.quality.scientific_impact == item.relevance.quality.scientific_impact  # type: ignore[union-attr]
    assert result.trending is None
    assert result.overall is None


def test_missing_quality_does_not_rank_on_citations_or_venue_alone() -> None:
    item = record(
        quality=QualityDimensions(venue=dimension(100), author_lab=dimension(100)),
        metrics=PaperMetrics(citations=100000),
    )
    scores = score_record(item, ScoringWeights(), NOW)
    assert scores.importance is None
    assert scores.overall is None
    assert scores.importance_breakdown.coverage == pytest.approx(0.15)


def test_sparse_or_zero_confidence_quality_does_not_dominate() -> None:
    sparse = record(quality=QualityDimensions(scientific_impact=dimension(100)))
    scores = score_record(sparse, ScoringWeights(), NOW)
    assert scores.importance is None
    assert scores.importance_breakdown.coverage == pytest.approx(0.3)
    uncertain = record(
        quality=QualityDimensions(
            scientific_impact=dimension(100, 0),
            novelty=dimension(100, 0),
        )
    )
    assert score_record(uncertain, ScoringWeights(), NOW).importance_breakdown.coverage == 0


def test_freshness_cannot_manufacture_attention() -> None:
    scores = score_record(record(), ScoringWeights(), NOW)
    assert scores.trending is None
    assert scores.trending_breakdown.coverage == pytest.approx(0.1)
    assert scores.trending_breakdown.components[4].value == 100
    only_stars = record(metrics=PaperMetrics(github_stars=100000))
    assert score_record(only_stars, ScoringWeights(), NOW).trending is None


def test_fast_growing_smaller_repository_beats_large_slow_repository() -> None:
    large = record(metrics=PaperMetrics(github_stars=100000, github_star_velocity_24h=50))
    fast = record(metrics=PaperMetrics(github_stars=1000, github_star_velocity_24h=1500))
    left, right = [score_record(item, ScoringWeights(), NOW) for item in (large, fast)]
    assert left.trending is not None and right.trending is not None
    assert right.trending > left.trending
    assert right.trending_breakdown.coverage == pytest.approx(0.5)


def test_configured_weights_are_used_without_default_weight_leaks() -> None:
    importance = ImportanceWeights(
        **{name: 1.0 if name == "novelty" else 0.0 for name in ImportanceWeights.model_fields}
    )
    trending = TrendingWeights(
        **{name: 1.0 if name == "github_velocity" else 0.0 for name in TrendingWeights.model_fields}
    )
    overall = OverallWeights(
        **{
            name: 1.0 if name == "citation_velocity" else 0.0
            for name in OverallWeights.model_fields
        }
    )
    weights = ScoringWeights(importance=importance, trending=trending, overall=overall)
    item = record(
        quality=QualityDimensions(novelty=dimension(73)),
        metrics=PaperMetrics(github_star_velocity_24h=250, citation_velocity=5),
    )
    scores = score_record(item, weights, NOW)
    assert scores.importance == 73
    assert scores.trending == 50
    assert scores.overall == 50
    assert scores.importance_breakdown.coverage == 1


def test_hf_attention_and_negative_growth_have_distinct_observed_values() -> None:
    item = record(metrics=PaperMetrics(hf_upvotes=100, hf_velocity=-10))
    item.enrichment = Enrichment(hf_trending=False)
    scores = score_record(item, ScoringWeights(), NOW)
    components = {component.name: component for component in scores.trending_breakdown.components}
    assert components["community_attention"].value == 25
    assert components["hf_velocity"].value == 0
    assert components["github_velocity"].value is None
    assert scores.trending is not None


def test_no_relevance_or_metrics_means_unknown_scores() -> None:
    item = record()
    item.relevance = None
    scores = score_record(item, ScoringWeights(), NOW)
    assert scores.relevance_score is None
    assert scores.importance is scores.trending is scores.overall is None
