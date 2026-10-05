"""Fixed-scale scoring; final editorial analysis never supplies ranking signals."""

from datetime import UTC, datetime

from research_radar.config import ScoringWeights
from research_radar.models.ranking import (
    PaperRecord,
    PaperScores,
    QualityDimensions,
    ScoreBreakdown,
    ScoreComponent,
)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def freshness_multiplier(published_at: datetime, now: datetime) -> float:
    """Elapsed floor-day bins: 0–2:1.5, 3–7:1.25, 8–14:1.1, 15–30:1, 31+:.8."""
    days = max(0, (_utc(now) - _utc(published_at)).days)
    if days <= 2:
        return 1.5
    if days <= 7:
        return 1.25
    if days <= 14:
        return 1.1
    return 1.0 if days <= 30 else 0.8


def saturating_score(value: float | int | None, half_saturation: float) -> float | None:
    """Map observed raw counts/day to 100*x/(x+h); nonpositive growth scores 0."""
    if value is None:
        return None
    value = max(0.0, float(value))
    return 100 * value / (value + half_saturation)


def _mean(values: list[float | None]) -> float | None:
    observed = [value for value in values if value is not None]
    return sum(observed) / len(observed) if observed else None


def _github_velocity(daily: float | None, weekly: float | None) -> float | None:
    observed = [
        (value, weight) for value, weight in ((daily, 0.7), (weekly, 0.3)) if value is not None
    ]
    return (
        sum(value * weight for value, weight in observed) / sum(weight for _, weight in observed)
        if observed
        else None
    )


def _breakdown(
    values: dict[str, float | None],
    weights: dict[str, float],
    *,
    supported: bool,
    minimum_coverage: float,
) -> ScoreBreakdown:
    coverage = sum(weight for name, weight in weights.items() if values[name] is not None)
    total_weight = sum(weights.values())
    coverage_fraction = min(1.0, coverage / total_weight) if total_weight else 0.0
    eligible = supported and coverage > 0 and coverage_fraction + 1e-9 >= minimum_coverage
    components = [
        ScoreComponent(
            name=name,
            value=value,
            weight=weight,
            contribution=(value * weight / coverage if eligible and value is not None else None),
        )
        for name, weight in weights.items()
        for value in [values[name]]
    ]
    score = (
        min(100.0, sum(component.contribution or 0.0 for component in components))
        if eligible
        else None
    )
    return ScoreBreakdown(score=score, coverage=coverage_fraction, components=components)


def score_record(record: PaperRecord, weights: ScoringWeights, now: datetime) -> PaperScores:
    """Score on fixed 0–100 scales, reweighting only observed components.

    Quality uses the cheap relevance pass's score/reason/confidence, never final
    analysis. Zero-confidence dimensions are unavailable; otherwise their scores
    are unchanged. Importance requires scientific quality and >=50% weight
    coverage; overall also requires >=50% and scientific quality when configured.
    Trending requires real attention and >=30% coverage, so freshness alone
    cannot rank. Below-threshold components remain visible but score is None.

    Saturation half-points: GitHub growth 250/day, citations 5/day, HF 25/day;
    absolute stars 5000, citations 100, HF upvotes 100. GitHub growth blends
    24h:7d at 70:30, reweighted if one is missing. Community attention averages
    observed HF upvote score and HF trending boolean (100/0). Absolute attention
    averages observed count scores. Overall community interest averages community
    attention, HF growth, and absolute attention; venue_author averages quality
    venue/author_lab. Freshness is a bounded component, 100*multiplier/1.5,
    not a post-score multiplier that can inflate attention-free papers.
    """
    quality = record.relevance.quality if record.relevance else QualityDimensions()
    quality_values = {
        name: dimension.score if dimension is not None and dimension.confidence > 0 else None
        for name in QualityDimensions.model_fields
        for dimension in [getattr(quality, name)]
    }
    scientific_names = ("scientific_impact", "novelty", "empirical_quality", "practical_relevance")
    scientific_available = any(quality_values[name] is not None for name in scientific_names)
    metrics = record.metrics
    github = _github_velocity(
        saturating_score(metrics.github_star_velocity_24h, 250),
        saturating_score(metrics.github_star_velocity_7d, 250),
    )
    citation = saturating_score(metrics.citations, 100)
    hf_count = saturating_score(metrics.hf_upvotes, 100)
    hf_growth = saturating_score(metrics.hf_velocity, 25)
    citation_growth = saturating_score(metrics.citation_velocity, 5)
    community = _mean(
        [
            hf_count,
            (100.0 if record.enrichment.hf_trending else 0.0)
            if record.enrichment.hf_trending is not None
            else None,
        ]
    )
    absolute = _mean([saturating_score(metrics.github_stars, 5000), citation, hf_count])
    freshness = freshness_multiplier(record.paper.published_at, now)
    importance_values = dict(quality_values, citation_signal=citation)
    trending_values = {
        "github_velocity": github,
        "community_attention": community,
        "hf_velocity": hf_growth,
        "citation_velocity": citation_growth,
        "freshness": 100 * freshness / 1.5,
        "absolute_attention": absolute,
    }
    overall_values = {
        **{name: quality_values[name] for name in scientific_names},
        "github_velocity": github,
        "community_interest": _mean([community, hf_growth, absolute]),
        "citation_velocity": citation_growth,
        "venue_author": _mean([quality_values["venue"], quality_values["author_lab"]]),
    }
    importance = _breakdown(
        importance_values,
        weights.importance.model_dump(),
        supported=scientific_available,
        minimum_coverage=0.5,
    )
    attention_available = any(
        value is not None and getattr(weights.trending, name) > 0
        for name, value in trending_values.items()
        if name != "freshness"
    )
    trending = _breakdown(
        trending_values,
        weights.trending.model_dump(),
        supported=attention_available,
        minimum_coverage=0.3,
    )
    scientific_configured = any(getattr(weights.overall, name) > 0 for name in scientific_names)
    overall_attention = any(
        overall_values[name] is not None and getattr(weights.overall, name) > 0
        for name in ("github_velocity", "community_interest", "citation_velocity")
    )
    overall = _breakdown(
        overall_values,
        weights.overall.model_dump(),
        supported=scientific_available if scientific_configured else overall_attention,
        minimum_coverage=0.5,
    )
    return PaperScores(
        relevance_score=record.relevance.relevance_score if record.relevance else None,
        importance=importance.score,
        trending=trending.score,
        overall=overall.score,
        freshness_multiplier=freshness,
        quality=quality.model_copy(deep=True),
        importance_breakdown=importance,
        trending_breakdown=trending,
        overall_breakdown=overall,
        scored_at=now,
        publication_score=record.scores.publication_score,
        publication_data_cutoff=record.scores.publication_data_cutoff,
        publication_basis=record.scores.publication_basis,
        retrospective_score=record.scores.retrospective_score,
        retrospective_data_cutoff=record.scores.retrospective_data_cutoff,
        retrospective_breakdown=record.scores.retrospective_breakdown.model_copy(deep=True),
    )
