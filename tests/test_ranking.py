from datetime import UTC, datetime, timedelta
from itertools import permutations

import pytest

from research_radar.config import ScoringWeights
from research_radar.models.metrics import PaperMetrics
from research_radar.models.paper import Paper
from research_radar.models.ranking import (
    PaperRecord,
    QualityDimensions,
    RelevanceResult,
    ScoreDimension,
)
from research_radar.ranking.ranking import rank_records

NOW = datetime(2026, 10, 5, 12, tzinfo=UTC)


def record(
    identifier: str, *, quality: float | None = None, growth: float | None = None
) -> PaperRecord:
    dimension = (
        ScoreDimension(score=quality, reason="Verified quality", confidence=0.9)
        if quality is not None
        else None
    )
    return PaperRecord(
        paper=Paper(
            id=identifier,
            title=f"Paper {identifier}",
            abstract="Abstract",
            authors=["Researcher"],
            categories=["cs.AI"],
            published_at=NOW - timedelta(days=1),
        ),
        relevance=RelevanceResult(
            relevance_score=90,
            reason="Relevant",
            confidence=0.9,
            quality=QualityDimensions(
                scientific_impact=dimension,
                novelty=dimension,
                empirical_quality=dimension,
                practical_relevance=dimension,
            ),
        ),
        metrics=PaperMetrics(
            github_stars=1000 if growth is not None else None,
            github_star_velocity_24h=growth,
        ),
    )


def test_ties_and_entire_output_are_deterministic() -> None:
    inputs = [
        record("c", quality=80, growth=100),
        record("b", quality=80, growth=100),
        record("a", quality=80, growth=100),
    ]
    expected = rank_records(inputs, top=3, now=NOW, weights=ScoringWeights())
    assert expected.selected_ids == ["a", "b", "c"]
    assert [entry.paper_id for entry in expected.trending] == ["a", "b", "c"]
    for permutation in permutations(inputs):
        assert rank_records(list(permutation), top=3, now=NOW, weights=ScoringWeights()) == expected


def test_unknown_scores_are_not_listed() -> None:
    result = rank_records([record("unknown")], top=10, now=NOW, weights=ScoringWeights())
    assert result.selected_ids == []
    assert result.trending == result.most_important == []


def test_updates_input_scores_and_prefers_supported_overall() -> None:
    high = record("quality", quality=95)
    viral = record("viral", growth=1500)
    result = rank_records([viral, high], top=1, now=NOW, weights=ScoringWeights())
    assert high.scores.overall == pytest.approx(95)
    assert viral.scores.overall is None
    assert result.selected_ids == ["quality"]
    assert result.trending[0].paper_id == "viral"
    assert result.most_important[0].paper_id == "quality"


def test_bounded_total_selection_covers_both_charts_without_overall() -> None:
    important = record("important", quality=90)
    important.relevance.quality.empirical_quality = None  # type: ignore[union-attr]
    important.relevance.quality.practical_relevance = None  # type: ignore[union-attr]
    viral = record("viral", growth=1500)
    unknown = record("unknown")
    result = rank_records([unknown, viral, important], top=2, now=NOW, weights=ScoringWeights())
    assert important.scores.importance is not None and important.scores.overall is None
    assert result.selected_ids == ["important", "viral"]
    assert len(result.selected_ids) <= 2
    assert len(set(result.selected_ids)) == 2


def test_archived_paper_returns_when_velocity_increases() -> None:
    archived = record("archived", growth=1500)
    archived.paper.published_at = NOW - timedelta(days=1000)
    recent = record("recent", growth=50)
    result = rank_records([recent, archived], top=1, now=NOW, weights=ScoringWeights())
    assert result.selected_ids == ["archived"]
    assert result.trending[0].paper_id == "archived"
    assert archived.scores.freshness_multiplier == 0.8


@pytest.mark.parametrize("top", [0, 1, 2, 10])
def test_selection_never_exceeds_total_top(top: int) -> None:
    records = [
        record("important", quality=90),
        record("viral", growth=2000),
        record("both", quality=80, growth=1000),
        record("none"),
    ]
    result = rank_records(records, top=top, now=NOW, weights=ScoringWeights())
    assert len(result.selected_ids) == min(top, 3)
    assert len(set(result.selected_ids)) == len(result.selected_ids)
    assert len(result.trending) <= top
    assert len(result.most_important) <= top


def test_duplicate_ids_do_not_duplicate_rank_entries() -> None:
    result = rank_records(
        [record("same", quality=80), record("same", quality=80)],
        top=5,
        now=NOW,
        weights=ScoringWeights(),
    )
    assert result.selected_ids == ["same"]
    assert len(result.most_important) == 1


def test_negative_top_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        rank_records([], top=-1, now=NOW, weights=ScoringWeights())
