from datetime import UTC, datetime, timedelta

import pytest

from research_radar.models.metrics import MetricsSnapshot
from research_radar.models.paper import Enrichment
from research_radar.ranking.velocity import calculate_metrics

NOW = datetime(2026, 10, 5, 12, tzinfo=UTC)


def snapshot(age: timedelta, **changes: object) -> MetricsSnapshot:
    return MetricsSnapshot.model_validate(
        {
            "timestamp": NOW - age,
            "github_stars": 1000,
            "citations": 100,
            "hf_upvotes": 50,
            "github_url": "https://github.com/team/repo",
            "citation_source": "openalex",
            "hf_paper_url": "https://huggingface.co/papers/2601.00001",
            **changes,
        }
    )


def test_velocities_are_per_day_and_use_target_horizons() -> None:
    current = snapshot(timedelta(), github_stars=2000, citations=106, hf_upvotes=75)
    previous = snapshot(timedelta(days=1), github_stars=1700, citations=100, hf_upvotes=50)
    weekly = snapshot(timedelta(days=7), github_stars=600)
    recent = snapshot(timedelta(hours=2), github_stars=1990, citations=105, hf_upvotes=70)
    result = calculate_metrics(Enrichment(), [recent, weekly, current, previous], NOW)
    assert result.github_stars == 2000
    assert result.github_star_velocity_24h == 300
    assert result.github_star_velocity_7d == 200
    assert result.citation_velocity == 6
    assert result.hf_velocity == 25
    assert result.observed_at == NOW


def test_irregular_sampling_uses_actual_elapsed_days() -> None:
    current = snapshot(timedelta(), github_stars=1300, citations=106, hf_upvotes=80)
    baseline = snapshot(timedelta(hours=30))
    result = calculate_metrics(Enrichment(), [current, baseline], NOW)
    assert result.github_star_velocity_24h == 240
    assert result.citation_velocity == pytest.approx(4.8)
    assert result.hf_velocity == 24
    assert result.github_star_velocity_7d is None


@pytest.mark.parametrize("age", [timedelta(hours=23), timedelta(days=2), timedelta(days=6)])
def test_insufficient_or_off_horizon_history_is_missing(age: timedelta) -> None:
    result = calculate_metrics(
        Enrichment(), [snapshot(timedelta()), snapshot(age, github_stars=1)], NOW
    )
    assert result.github_star_velocity_24h is None
    assert result.github_star_velocity_7d is None
    assert result.citation_velocity is None
    assert result.hf_velocity is None


def test_provenance_is_required_and_sources_cannot_cross() -> None:
    baseline = snapshot(
        timedelta(days=1),
        github_url="https://github.com/team/old-repo",
        citation_source="semantic_scholar",
        hf_paper_url="https://huggingface.co/papers/2601.99999",
    )
    result = calculate_metrics(Enrichment(), [baseline, snapshot(timedelta())], NOW)
    assert result.github_star_velocity_24h is None
    assert result.citation_velocity is None
    assert result.hf_velocity is None
    unknown = [
        snapshot(age, github_url=None, citation_source=None, hf_paper_url=None)
        for age in [timedelta(), timedelta(days=1)]
    ]
    assert calculate_metrics(Enrichment(), unknown, NOW).github_star_velocity_24h is None


def test_normalized_provenance_and_closest_eligible_baseline() -> None:
    current = snapshot(
        timedelta(),
        github_stars=1100,
        github_url="http://github.com/TEAM/REPO.git/",
    )
    history = [
        snapshot(timedelta(hours=30), github_stars=600),
        snapshot(timedelta(days=1), github_stars=1000),
        current,
    ]
    result = calculate_metrics(Enrichment(), history, NOW)
    assert result.github_star_velocity_24h == 100


def test_decreasing_counts_remain_signed() -> None:
    history = [
        snapshot(timedelta(), github_stars=900, citations=90, hf_upvotes=40),
        snapshot(timedelta(days=1)),
    ]
    result = calculate_metrics(Enrichment(), history, NOW)
    assert result.github_star_velocity_24h == -100
    assert result.citation_velocity == -10
    assert result.hf_velocity == -10


def test_historical_replay_ignores_future_counts_and_live_enrichment() -> None:
    current = snapshot(timedelta(), github_stars=1100)
    past = snapshot(timedelta(days=1))
    future = snapshot(timedelta(days=-1), github_stars=9000)
    enrichment = Enrichment(github_stars=10000, openalex_citations=10000, hf_upvotes=10000)
    first = calculate_metrics(enrichment, [future, current, past], NOW)
    second = calculate_metrics(enrichment, [past, current, future], NOW.replace(tzinfo=None))
    assert first == second
    assert first.github_stars == 1100
    assert first.citations == 100
    assert first.hf_upvotes == 50
    assert first.github_star_velocity_24h == 100


def test_no_history_and_missing_current_counts_are_not_fabricated() -> None:
    enrichment = Enrichment(github_stars=100000, openalex_citations=10)
    result = calculate_metrics(enrichment, [], NOW)
    assert result.github_stars == 100000
    assert result.github_star_velocity_24h is None
    assert result.citation_velocity is None
    history = [snapshot(timedelta(days=1)), snapshot(timedelta(), github_stars=None)]
    assert calculate_metrics(enrichment, history, NOW).github_stars is None
    assert calculate_metrics(enrichment, history, NOW).github_star_velocity_24h is None


def test_stale_endpoint_reports_actual_observation_time() -> None:
    current = snapshot(timedelta(days=1), github_stars=1200)
    baseline = snapshot(timedelta(days=2), github_stars=1000)
    result = calculate_metrics(Enrichment(), [current, baseline], NOW)
    assert result.observed_at == current.timestamp
    assert result.github_star_velocity_24h == 200
