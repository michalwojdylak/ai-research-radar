"""Cutoff-safe paper scoring; current provider values cannot masquerade as history."""

from datetime import datetime

from research_radar.config import RetrospectiveWeights, ScoringWeights
from research_radar.models.history import RelevanceObservation
from research_radar.models.metrics import MetricsSnapshot, PaperMetrics
from research_radar.models.paper import Enrichment, Paper
from research_radar.models.radars import PeriodPaperScore, QualityBasis, RankingMode
from research_radar.models.ranking import (
    PaperRecord,
    PaperScores,
    RelevanceResult,
    ScoreBreakdown,
    ScoreComponent,
)
from research_radar.periods import utc, week_for
from research_radar.ranking.relevance import deterministic_prefilter
from research_radar.ranking.scoring import saturating_score, score_record
from research_radar.ranking.velocity import calculate_metrics


def _document_available(paper: Paper, cutoff: datetime) -> bool:
    return paper.published_at <= cutoff and (paper.updated_at or paper.published_at) <= cutoff


def _quality_at(
    record: PaperRecord,
    observations: list[RelevanceObservation],
    cutoff: datetime,
    *,
    allow_proxy: bool,
    observation_cutoff: datetime,
) -> tuple[Paper | None, RelevanceResult | None, QualityBasis, datetime | None]:
    relevance: RelevanceResult | None
    historical = [
        x for x in observations if x.observed_at <= cutoff and _document_available(x.paper, cutoff)
    ]
    if historical:
        latest = max(historical, key=lambda x: (x.observed_at, x.model_dump_json()))
        relevance = latest.relevance or RelevanceResult(
            relevance_score=0,
            confidence=1,
            classifier="deterministic",
            reason="A recorded historical classification rejected this paper.",
        )
        return latest.paper, relevance, "historical_observation", latest.observed_at
    if (
        record.relevance_observed_at is not None
        and record.relevance_observed_at <= cutoff
        and _document_available(record.paper, cutoff)
    ):
        return (
            record.paper,
            record.relevance,
            "historical_observation",
            record.relevance_observed_at,
        )
    if allow_proxy:
        proxies = [
            x
            for x in observations
            if x.primary_only
            and x.observed_at <= observation_cutoff
            and _document_available(x.paper, cutoff)
        ]
        if (
            record.relevance_primary_only
            and record.relevance_observed_at is not None
            and record.relevance_observed_at <= observation_cutoff
            and _document_available(record.paper, cutoff)
        ):
            proxies.append(
                RelevanceObservation(
                    observed_at=record.relevance_observed_at,
                    paper=record.paper,
                    relevance=record.relevance,
                    primary_only=True,
                    model=record.relevance_model,
                    prompt_version=record.relevance_prompt_version,
                )
            )
        if proxies:
            latest = max(proxies, key=lambda x: (x.observed_at, x.model_dump_json()))
            relevance = latest.relevance.model_copy(deep=True) if latest.relevance else None
            if relevance is not None:
                relevance.quality = relevance.quality.model_copy(
                    update={"venue": None, "author_lab": None}
                )
            return latest.paper, relevance, "publication_document_proxy", latest.observed_at
    if _document_available(record.paper, cutoff):
        relevance = (
            RelevanceResult(
                relevance_score=70,
                confidence=0.35,
                classifier="deterministic",
                reason="Publication category/keyword filter only; research quality is unknown.",
            )
            if deterministic_prefilter(record.paper)
            else None
        )
        return record.paper, relevance, "unavailable", None
    return None, None, "unavailable", None


def _retrospective_breakdown(record: PaperRecord, weights: RetrospectiveWeights) -> ScoreBreakdown:
    values = {
        "research_quality": record.scores.importance,
        "citations": saturating_score(record.metrics.citations, 100),
        "github_adoption": saturating_score(record.metrics.github_stars, 5000),
        "community": saturating_score(record.metrics.hf_upvotes, 100),
    }
    configured = {
        "research_quality": weights.research_quality,
        "citations": weights.citations,
        "github_adoption": weights.github_adoption,
        "community": weights.community,
    }
    coverage = sum(configured[name] for name, value in values.items() if value is not None)
    components = [
        ScoreComponent(
            name=name,
            value=value,
            weight=configured[name],
            contribution=value * configured[name] / coverage
            if value is not None and coverage > 0
            else None,
        )
        for name, value in values.items()
    ]
    return ScoreBreakdown(
        score=sum(x.contribution or 0 for x in components) if coverage > 0 else None,
        coverage=min(1, coverage),
        components=components,
    )


def score_at(
    record: PaperRecord,
    snapshots: list[MetricsSnapshot],
    *,
    data_cutoff: datetime,
    weights: ScoringWeights,
    mode: RankingMode = "publication",
    observations: list[RelevanceObservation] | None = None,
    allow_publication_proxy: bool = False,
    observation_cutoff: datetime | None = None,
    retrospective_weights: RetrospectiveWeights | None = None,
    relevance_threshold: int = 60,
) -> PeriodPaperScore:
    """Only timestamped facts at/before the cutoff enter either scoring mode.

    Strict mode requires contemporaneous quality observations. Explicit proxies
    permit later abstract-only estimates of documents available by the publication
    cutoff, labelled as approximations rather than genuine historical observations.
    Undated legacy estimates and revised future documents are not historical facts.
    """
    cutoff = utc(data_cutoff)
    observation_end = utc(observation_cutoff) if observation_cutoff else cutoff
    paper, relevance, basis, quality_at = _quality_at(
        record,
        observations or [],
        cutoff,
        allow_proxy=allow_publication_proxy,
        observation_cutoff=observation_end,
    )
    scoped = record.model_copy(deep=True)
    scoped.paper = paper if paper is not None else record.paper
    scoped.relevance = relevance
    history = [snapshot for snapshot in snapshots if snapshot.timestamp <= cutoff]
    latest = max(history, key=lambda x: (x.timestamp, x.model_dump_json()), default=None)
    scoped.enrichment = Enrichment(hf_trending=latest.hf_trending if latest else None)
    scoped.metrics = (
        calculate_metrics(scoped.enrichment, history, cutoff) if latest else PaperMetrics()
    )
    scoped.scores = PaperScores()
    scoped.scores = score_record(scoped, weights, cutoff)
    scoped.scores.scope = mode
    scoped.scores.scored_at = cutoff
    eligible = (
        paper is not None
        and relevance is not None
        and (relevance.relevance_score >= relevance_threshold)
    )
    if mode == "publication":
        publication_score = (
            next(
                (
                    value
                    for value in (
                        scoped.scores.overall,
                        scoped.scores.importance,
                        scoped.scores.trending,
                    )
                    if value is not None
                ),
                None,
            )
            if eligible
            else None
        )
        breakdown = (
            scoped.scores.overall_breakdown
            if scoped.scores.overall is not None
            else scoped.scores.importance_breakdown
            if scoped.scores.importance is not None
            else scoped.scores.trending_breakdown
        )
        scoped.scores.publication_score = publication_score
        scoped.scores.publication_data_cutoff = cutoff
        scoped.scores.publication_basis = basis
        return PeriodPaperScore(
            paper_id=record.paper.id,
            mode=mode,
            data_cutoff=cutoff,
            scores=scoped.scores,
            publication_score=publication_score,
            quality_basis=basis,
            quality_observed_at=quality_at,
            coverage=breakdown.coverage,
            source_paper=paper,
        )
    retrospective = _retrospective_breakdown(
        scoped, retrospective_weights or RetrospectiveWeights()
    )
    scoped.scores.retrospective_score = retrospective.score if eligible else None
    scoped.scores.retrospective_data_cutoff = cutoff
    scoped.scores.retrospective_breakdown = retrospective
    publication = score_at(
        record,
        snapshots,
        data_cutoff=min(cutoff, week_for(record.paper.published_at).end),
        weights=weights,
        observations=observations,
        mode="publication",
        allow_publication_proxy=allow_publication_proxy,
        observation_cutoff=observation_end,
        relevance_threshold=relevance_threshold,
    )
    scoped.scores.publication_score = publication.publication_score
    scoped.scores.publication_data_cutoff = publication.data_cutoff
    scoped.scores.publication_basis = publication.quality_basis
    return PeriodPaperScore(
        paper_id=record.paper.id,
        mode=mode,
        data_cutoff=cutoff,
        scores=scoped.scores,
        retrospective_score=scoped.scores.retrospective_score,
        publication_score=publication.publication_score,
        quality_basis=basis,
        quality_observed_at=quality_at,
        coverage=retrospective.coverage,
        source_paper=paper,
    )
