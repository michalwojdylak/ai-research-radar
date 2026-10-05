from calendar import monthrange
from datetime import date, datetime
from typing import Literal

from research_radar.config import Settings
from research_radar.models.analysis import PromptVersions
from research_radar.models.radars import (
    MonthlyRadar,
    PeriodPaperScore,
    PeriodReference,
    RadarProvenance,
    RadarStatus,
    RankingMode,
    WeeklyPaperSelection,
    WeeklyRadar,
    YearlyRadar,
)
from research_radar.models.ranking import PaperRecord
from research_radar.periods import WeekPeriod, iter_weeks, utc, week_for
from research_radar.radars.scoring import radar_score

_REVIEWED = {"review", "approved", "published"}


def _provenance(
    settings: Settings,
    now: datetime,
    cutoff: datetime,
    mode: RankingMode,
    status: RadarStatus,
    complete: bool,
    explanation: str,
) -> RadarProvenance:
    return RadarProvenance(
        mode=mode,
        ranking_version=settings.ranking_version,
        scoring_version=settings.scoring_version,
        prompt_versions=PromptVersions(
            relevance=settings.relevance_prompt_version,
            analysis=settings.analysis_prompt_version,
            critic=settings.critic_prompt_version,
        ),
        generated_at=now,
        data_cutoff=cutoff,
        scoring_configuration=settings.radar_scoring,
        paper_scoring_weights=settings.weights,
        retrospective_weights=settings.retrospective_weights,
        status=status,
        is_complete=complete,
        explanation=explanation,
    )


def _selection_reason(score: PeriodPaperScore) -> str:
    value = score.ranking_score
    assert value is not None
    parts = [
        f"{score.mode.title()} score {value:.2f}/100",
        f"evidence coverage {score.coverage:.0%}",
        f"quality basis: {score.quality_basis}",
    ]
    dimensions = score.scores.quality
    for name in type(dimensions).model_fields:
        dimension = getattr(dimensions, name)
        if dimension is not None:
            parts.append(
                f"{name.replace('_', ' ')} {dimension.score:.2f}/100 "
                f"(confidence {dimension.confidence:.0%}): {dimension.reason}"
            )
    breakdown = (
        score.scores.retrospective_breakdown
        if score.mode == "retrospective"
        else score.scores.overall_breakdown
        if score.scores.overall is not None
        else score.scores.importance_breakdown
        if score.scores.importance is not None
        else score.scores.trending_breakdown
    )
    for component in sorted(breakdown.components, key=lambda item: item.name):
        if component.value is not None:
            contribution = (
                f", contribution {component.contribution:.2f}"
                if component.contribution is not None
                else ""
            )
            parts.append(
                f"{component.name.replace('_', ' ')} {component.value:.2f}/100 "
                f"(weight {component.weight:g}{contribution})"
            )
    if score.quality_observed_at is not None:
        parts.append(f"quality observed at {score.quality_observed_at.isoformat()}")
    return "; ".join(parts) + "."


def build_weekly(
    period: WeekPeriod,
    records: list[PaperRecord],
    scores: list[PeriodPaperScore],
    settings: Settings,
    *,
    now: datetime,
    data_cutoff: datetime,
    mode: RankingMode = "publication",
    coverage_start: date | None = None,
    coverage_end: date | None = None,
    is_preview: bool = False,
    is_complete: bool = True,
    analysis_statuses: dict[str, RadarStatus] | None = None,
) -> WeeklyRadar:
    current_preview = period.start <= utc(now) < period.end
    is_preview = is_preview or current_preview
    canonical: dict[str, PaperRecord] = {}
    for record in records:
        if not period.start <= record.paper.published_at < period.end:
            continue
        previous = canonical.get(record.paper.id)
        if previous is not None and previous != record:
            raise ValueError("Conflicting canonical records for the same paper ID")
        canonical[record.paper.id] = record
    scoped: dict[str, PeriodPaperScore] = {}
    for score in scores:
        if score.mode != mode or score.paper_id not in canonical:
            continue
        if score.data_cutoff != utc(data_cutoff):
            raise ValueError("Paper score cutoff disagrees with radar cutoff")
        previous_score = scoped.get(score.paper_id)
        if previous_score is not None and previous_score != score:
            raise ValueError("Conflicting period scores for the same paper ID")
        scoped[score.paper_id] = score
    known = [score for score in scoped.values() if score.ranking_score is not None]
    qualified = sorted(
        (
            score
            for score in known
            if score.ranking_score is not None and score.ranking_score >= settings.weekly_min_score
        ),
        key=lambda score: (-(score.ranking_score or 0.0), score.paper_id),
    )[: settings.weekly_max_papers]
    statuses = analysis_statuses or {}
    papers = [
        WeeklyPaperSelection(
            paper_id=score.paper_id,
            rank=rank,
            title=canonical[score.paper_id].paper.title,
            paper_url=canonical[score.paper_id].paper.abs_url
            or canonical[score.paper_id].paper.pdf_url
            or (
                f"https://doi.org/{canonical[score.paper_id].paper.doi}"
                if canonical[score.paper_id].paper.doi
                else None
            ),
            importance_score=score.scores.importance,
            trending_score=score.scores.trending,
            overall_score=score.scores.overall,
            publication_score=score.publication_score,
            retrospective_score=score.retrospective_score,
            selection_reason=_selection_reason(score),
            quality_basis=score.quality_basis,
            coverage=score.coverage,
            analysis_status=statuses.get(score.paper_id),
            paper_version=(
                score.source_paper.arxiv_id
                if score.source_paper is not None
                else canonical[score.paper_id].paper.arxiv_id
            ),
        )
        for rank, score in enumerate(qualified, 1)
    ]
    start = coverage_start or period.start_date
    end = coverage_end or (utc(now).date() if current_preview else period.end_date)
    complete = (
        is_complete
        and not is_preview
        and start == period.start_date
        and end == period.end_date
        and utc(now) >= period.end
        and utc(data_cutoff) >= period.end
    )
    selected_scores = [score.ranking_score for score in qualified]
    aggregate: float | None
    if selected_scores:
        aggregate = radar_score(
            [value for value in selected_scores if value is not None],
            settings.radar_scoring.weekly_coefficients,
            settings.radar_scoring,
        )
    elif complete and len(known) == len(canonical):
        aggregate = 0.0
    else:
        aggregate = None
    status: RadarStatus = "draft"
    if not is_preview:
        if any(paper.analysis_status == "needs_review" for paper in papers):
            status = "needs_review"
        elif (
            complete
            and aggregate is not None
            and all(paper.analysis_status in _REVIEWED for paper in papers)
        ):
            status = "review"
    explanation = (
        f"{len(papers)} selected from {len(canonical)} publication-week candidates; "
        f"{len(known)} have a known {mode} score. "
        f"Selection requires a score >= {settings.weekly_min_score:g}/100, "
        f"with a ceiling of {settings.weekly_max_papers}; the soft minimum never pads the list. "
        "The aggregate uses a quality-transformed, fixed-denominator peak and pooled tail."
    )
    if aggregate is None:
        explanation += " Unknown evidence or incomplete discovery is not a measured weak week."
    return WeeklyRadar(
        **_provenance(settings, now, data_cutoff, mode, status, complete, explanation).model_dump(),
        id=period.id,
        year=period.year,
        week=period.week,
        start_date=period.start_date,
        end_date=period.end_date,
        papers=papers,
        radar_score=aggregate if mode == "publication" else None,
        retrospective_score=aggregate if mode == "retrospective" else None,
        paper_count=len(papers),
        candidate_count=len(canonical),
        scored_candidate_count=len(known),
        coverage_start=start,
        coverage_end=end,
        is_preview=is_preview,
        quality_threshold=settings.weekly_min_score,
        below_minimum=len(papers) < settings.weekly_min_papers,
        discovery_complete=complete,
    )


def _owned_weeks(
    weeks: list[WeeklyRadar], mode: RankingMode, year: int, month: int | None = None
) -> list[WeeklyRadar]:
    unique: dict[str, WeeklyRadar] = {}
    for week in weeks:
        owner = week_for(week.start_date).month_key
        if week.is_preview or week.mode != mode or owner[0] != year:
            continue
        if month is not None and owner[1] != month:
            continue
        if week.id in unique and unique[week.id] != week:
            raise ValueError("Conflicting weekly radars for the same period")
        unique[week.id] = week
    return [unique[key] for key in sorted(unique)]


def _references(children: list[WeeklyRadar] | list[MonthlyRadar]) -> list[PeriodReference]:
    refs = [
        PeriodReference(id=child.id, score=child.ranking_score, paper_count=child.paper_count)
        for child in children
    ]
    return sorted(refs, key=lambda ref: (ref.score is None, -(ref.score or 0.0), ref.id))


def _aggregate_status(
    children: list[WeeklyRadar] | list[MonthlyRadar], complete: bool
) -> RadarStatus:
    if any(child.status == "needs_review" for child in children):
        return "needs_review"
    if (
        complete
        and children
        and all(child.status in _REVIEWED and child.ranking_score is not None for child in children)
    ):
        return "review"
    return "draft"


def _aggregate(
    refs: list[PeriodReference], settings: Settings, level: Literal["monthly", "yearly"]
) -> float | None:
    known = [ref.score for ref in refs if ref.score is not None]
    coefficients = (
        settings.radar_scoring.monthly_coefficients
        if level == "monthly"
        else settings.radar_scoring.yearly_coefficients
    )
    return radar_score(known, coefficients, settings.radar_scoring) if known else None


def build_monthly(
    year: int,
    month: int,
    weeks: list[WeeklyRadar],
    settings: Settings,
    *,
    now: datetime,
    mode: RankingMode = "publication",
) -> MonthlyRadar:
    first = date(year, month, 1)
    last = date(year, month, monthrange(year, month)[1])
    required = [period for period in iter_weeks(first, last) if period.month_key == (year, month)]
    children = _owned_weeks(weeks, mode, year, month)
    refs = _references(children)
    required_ids = {period.id for period in required}
    complete = (
        {child.id for child in children} == required_ids
        and all(child.is_complete for child in children)
        and utc(now).date() > last
        and utc(now) >= max(period.end for period in required)
    )
    aggregate = _aggregate(refs, settings, "monthly")
    known_count = sum(ref.score is not None for ref in refs)
    explanation = (
        "ISO weeks belong to the month of their Thursday (the majority of their days). "
        f"{len(children)}/{len(required)} required weeks supplied; "
        f"{known_count} known and {len(refs) - known_count} unknown weekly scores. "
        "Missing or unknown weeks are not substituted with zero; "
        "the monthly score uses known weekly scores and monthly quality coefficients."
    )
    return MonthlyRadar(
        **_provenance(
            settings,
            now,
            max((child.data_cutoff for child in children), default=utc(now)),
            mode,
            _aggregate_status(children, complete),
            complete,
            explanation,
        ).model_dump(),
        id=f"{year:04}-{month:02}",
        year=year,
        month=month,
        score=aggregate if mode == "publication" else None,
        retrospective_score=aggregate if mode == "retrospective" else None,
        best_week=next((ref.id for ref in refs if ref.score is not None), None),
        top_weeks=refs,
        paper_count=sum(child.paper_count for child in children),
        known_week_count=known_count,
        unknown_week_count=len(refs) - known_count,
    )


def build_yearly(
    year: int,
    months: list[MonthlyRadar],
    weeks: list[WeeklyRadar],
    settings: Settings,
    *,
    now: datetime,
    mode: RankingMode = "publication",
) -> YearlyRadar:
    unique: dict[str, MonthlyRadar] = {}
    for month in months:
        if month.year != year or month.mode != mode:
            continue
        if month.id in unique and unique[month.id] != month:
            raise ValueError("Conflicting monthly radars for the same period")
        unique[month.id] = month
    children = [unique[key] for key in sorted(unique)]
    refs = _references(children)
    owned_weeks = _owned_weeks(weeks, mode, year)
    week_refs = _references(owned_weeks)
    required_weeks = [
        period
        for period in iter_weeks(date(year, 1, 1), date(year, 12, 31))
        if period.month_key[0] == year
    ]
    complete = (
        {child.month for child in children} == set(range(1, 13))
        and all(child.is_complete for child in children)
        and {week.id for week in owned_weeks} == {period.id for period in required_weeks}
        and all(week.is_complete for week in owned_weeks)
        and utc(now).date() > date(year, 12, 31)
        and utc(now) >= max(period.end for period in required_weeks)
    )
    selected: dict[str, float] = {}
    for week in owned_weeks:
        for paper in week.papers:
            value = paper.publication_score if mode == "publication" else paper.retrospective_score
            if value is not None:
                selected[paper.paper_id] = max(value, selected.get(paper.paper_id, value))
    top_papers = sorted(selected, key=lambda paper_id: (-selected[paper_id], paper_id))
    aggregate = _aggregate(refs, settings, "yearly")
    known_count = sum(ref.score is not None for ref in refs)
    explanation = (
        "The yearly score uses known monthly scores with yearly quality coefficients. "
        f"{len(children)}/12 months supplied; {known_count} known and "
        f"{len(refs) - known_count} unknown monthly scores. "
        "Weekly highlights use Thursday year ownership; paper IDs are distinct canonical "
        "references ordered by their mode-specific paper scores. Unknown is not zero."
    )
    return YearlyRadar(
        **_provenance(
            settings,
            now,
            max(
                [child.data_cutoff for child in children]
                + [week.data_cutoff for week in owned_weeks],
                default=utc(now),
            ),
            mode,
            _aggregate_status(children, complete),
            complete,
            explanation,
        ).model_dump(),
        id=f"{year:04}",
        year=year,
        score=aggregate if mode == "publication" else None,
        retrospective_score=aggregate if mode == "retrospective" else None,
        best_month=next((ref.id for ref in refs if ref.score is not None), None),
        top_months=refs,
        best_weeks=week_refs,
        top_papers=top_papers,
        paper_count=sum(child.paper_count for child in children),
        known_month_count=known_count,
        unknown_month_count=len(refs) - known_count,
    )
