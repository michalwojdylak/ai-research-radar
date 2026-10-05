import json
import math
from datetime import UTC, date, datetime, timedelta
from itertools import permutations
from pathlib import Path

import pytest
from pydantic import ValidationError

from research_radar.config import RadarScoring, Settings
from research_radar.models.paper import Paper
from research_radar.models.radars import (
    MonthlyRadar,
    PeriodPaperScore,
    RadarStatus,
    RankingMode,
    WeeklyRadar,
    YearlyRadar,
)
from research_radar.models.ranking import (
    PaperRecord,
    PaperScores,
    QualityDimensions,
    ScoreBreakdown,
    ScoreComponent,
    ScoreDimension,
)
from research_radar.periods import (
    iter_weeks,
    parse_week,
    previous_week,
    utc,
    week_for,
)
from research_radar.pipeline.generate import HUMAN_END, HUMAN_START, extract_human_block
from research_radar.radars.hierarchy import build_monthly, build_weekly, build_yearly
from research_radar.radars.output import render_markdown, validate_radar_files, write_radar
from research_radar.radars.scoring import radar_score

NOW = datetime(2025, 1, 12, 12, tzinfo=UTC)
COEFFICIENTS = [1.0, 0.7, 0.5, 0.35, 0.2]


def settings(**values: object) -> Settings:
    return Settings(_env_file=None, **values)


def record(identifier: str, published_at: datetime) -> PaperRecord:
    return PaperRecord(
        paper=Paper(
            id=identifier,
            arxiv_id=identifier + "v1",
            title=f"Primary paper {identifier}",
            abstract="Primary abstract",
            authors=["Researcher"],
            categories=["cs.AI"],
            published_at=published_at,
            abs_url=f"https://arxiv.org/abs/{identifier}",
        ),
        scores=PaperScores(overall=1),
    )


def period_score(
    identifier: str,
    value: float | None,
    cutoff: datetime,
    mode: RankingMode = "publication",
) -> PeriodPaperScore:
    return PeriodPaperScore(
        paper_id=identifier,
        mode=mode,
        data_cutoff=cutoff,
        scores=PaperScores(
            importance=value,
            trending=0 if value is not None else None,
            overall=value,
            quality=QualityDimensions(
                novelty=ScoreDimension(
                    score=value,
                    reason="The primary abstract describes mechanism A.",
                    confidence=0.8,
                )
                if value is not None
                else None,
            ),
            overall_breakdown=ScoreBreakdown(
                score=value,
                coverage=0.75,
                components=[
                    ScoreComponent(name="novelty", value=value, weight=0.2, contribution=19)
                ],
            ),
        ),
        publication_score=value if mode == "publication" else 70,
        retrospective_score=value if mode == "retrospective" else None,
        coverage=0.75,
        quality_basis="publication_document_proxy",
        quality_observed_at=NOW - timedelta(days=1),
    )


def week(
    values: list[float | None],
    identifier: str = "2024-W01",
    *,
    config: Settings | None = None,
    mode: RankingMode = "publication",
    complete: bool = True,
    preview: bool = False,
) -> WeeklyRadar:
    period = parse_week(identifier)
    inputs = [
        record(f"{identifier}-{index:02}", period.start + timedelta(days=1))
        for index, _ in enumerate(values)
    ]
    cutoff = period.end if mode == "publication" else NOW
    scores = [
        period_score(paper.paper.id, value, cutoff, mode)
        for paper, value in zip(inputs, values, strict=True)
    ]
    return build_weekly(
        period,
        inputs,
        scores,
        config or settings(),
        now=NOW,
        data_cutoff=cutoff,
        mode=mode,
        is_complete=complete,
        is_preview=preview,
        analysis_statuses={paper.paper.id: "review" for paper in inputs},
    )


def january() -> list[WeeklyRadar]:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "january_radars.json").read_text(encoding="utf-8")
    )
    return [week(item["scores"], item["id"]) for item in fixture["weeks"]]


def all_year(values: list[float]) -> list[WeeklyRadar]:
    return [
        week(values, period.id)
        for period in iter_weeks(date(2024, 1, 1), date(2024, 12, 31))
        if period.month_key[0] == 2024
    ]


def test_score_matches_exact_peak_and_bounded_mean_tail_formula() -> None:
    config = RadarScoring()
    values = [97, 96, 95, 94, 93, 92, 91]
    q = [((value - 70) / 30) ** 2 for value in values]
    strength = sum(c * value for c, value in zip(COEFFICIENTS[:4], q[:4], strict=True))
    strength += 0.2 * sum(q[4:]) / 3
    expected = 100 * (0.8 * 0.97 + 0.2 * strength / sum(COEFFICIENTS))
    assert radar_score(values, COEFFICIENTS, config) == pytest.approx(expected)
    assert radar_score([95], COEFFICIENTS, config) == pytest.approx(
        100 * (0.8 * 0.95 + 0.2 * ((25 / 30) ** 2) / sum(COEFFICIENTS))
    )
    assert radar_score([], COEFFICIENTS, config) == 0


def test_quality_beats_quantity_and_repeated_tail_is_bounded() -> None:
    config = RadarScoring()

    def score(values: list[float]) -> float:
        return radar_score(values, COEFFICIENTS, config)

    assert score([95, 94, 93]) > score([95] + [60] * 100)
    assert score([95, 95]) > score([70] * 10)
    assert score([95] * 5) == score([95] * 500)
    assert score([100] * 5) == pytest.approx(100)
    assert score([0] * 100) == 0
    assert score([70] * 100) == pytest.approx(56)
    previous = 0.0
    for count in range(1, 12):
        current = score([95] * count)
        assert previous <= current <= 100
        previous = current
    assert score([95, 90, 80, 70, 70]) == score([95, 90, 80, 70] + [70] * 100)


def test_scoring_is_stable_and_uses_supplied_configuration() -> None:
    config = RadarScoring(peak_share=0.3, quality_floor=50, quality_power=3)
    values = [75, 95, 80]
    expected = radar_score(values, [1, 0.4], config)
    assert expected != radar_score(values, COEFFICIENTS, RadarScoring())
    for order in permutations(values):
        assert radar_score(list(order), [1, 0.4], config) == expected


@pytest.mark.parametrize("invalid", [-1, 101, math.inf, -math.inf, math.nan])
def test_invalid_scores_are_rejected_not_clipped(invalid: float) -> None:
    with pytest.raises(ValueError, match="Scores"):
        radar_score([95, invalid], COEFFICIENTS, RadarScoring())


@pytest.mark.parametrize(
    "invalid",
    [[], [1], [0, 0], [1, -0.1], [0.5, 1], [1, math.inf], [1, math.nan], [1e308, 1e308]],
)
def test_invalid_coefficients_are_rejected_even_for_empty_scores(invalid: list[float]) -> None:
    with pytest.raises(ValueError, match="Coefficients"):
        radar_score([], invalid, RadarScoring())


def test_unvalidated_nonfinite_configuration_is_rejected() -> None:
    config = RadarScoring.model_construct(quality_power=math.inf)
    with pytest.raises(ValueError, match="configuration"):
        radar_score([95], COEFFICIENTS, config)


@pytest.mark.parametrize(
    ("day", "identifier", "owner"),
    [
        (date(2024, 1, 1), "2024-W01", (2024, 1)),
        (date(2024, 1, 31), "2024-W05", (2024, 2)),
        (date(2024, 2, 29), "2024-W09", (2024, 2)),
        (date(2021, 1, 1), "2020-W53", (2020, 12)),
        (date(2024, 12, 30), "2025-W01", (2025, 1)),
    ],
)
def test_iso_periods_and_thursday_ownership(
    day: date, identifier: str, owner: tuple[int, int]
) -> None:
    period = week_for(day)
    assert period.id == identifier
    assert parse_week(identifier) == period
    assert period.month_key == owner
    assert period.start_date.weekday() == 0
    assert period.end_date.weekday() == 6
    assert period.end - period.start == timedelta(days=7)
    assert list(iter_weeks(day, day)) == [period]


def test_previous_week_and_utc_normalization() -> None:
    assert previous_week(datetime(2024, 1, 8, 15, tzinfo=UTC)).id == "2024-W01"
    assert previous_week(datetime(2021, 1, 4, tzinfo=UTC)).id == "2020-W53"
    assert utc(datetime(2024, 1, 1)) == datetime(2024, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError):
        parse_week("2024-W54")
    with pytest.raises(ValueError):
        parse_week("January")
    with pytest.raises(ValueError):
        list(iter_weeks(date(2024, 2, 1), date(2024, 1, 1)))


@pytest.mark.parametrize(("count", "limit", "expected"), [(0, 14, 0), (2, 14, 2), (17, 14, 14)])
def test_variable_selection_counts_and_hard_ceiling(count: int, limit: int, expected: int) -> None:
    radar = week([95] * count, config=settings(weekly_max_papers=limit))
    assert radar.paper_count == len(radar.papers) == expected
    assert radar.candidate_count == radar.scored_candidate_count == count
    assert [paper.rank for paper in radar.papers] == list(range(1, expected + 1))
    assert radar.status == "review"
    assert radar.is_complete


def test_threshold_and_soft_minimum_never_pad_a_weak_week() -> None:
    config = settings(weekly_min_score=80, weekly_min_papers=5)
    radar = week([95, 80, 79, 40, None], config=config)
    assert radar.paper_count == 2
    assert radar.below_minimum
    assert radar.quality_threshold == 80
    weak = week([79, 60, 20], config=config)
    assert weak.paper_count == 0
    assert weak.radar_score == 0
    assert weak.below_minimum
    assert settings().weekly_min_papers == 0
    assert week([0], config=settings(weekly_min_score=0)).paper_count == 1


def test_input_order_and_duplicates_do_not_change_selection() -> None:
    period = parse_week("2024-W01")
    records = [record(identifier, period.start) for identifier in ["c", "b", "a"]]
    scores = [period_score(item.paper.id, 95, period.end) for item in records]
    config = settings()
    expected = build_weekly(period, records, scores, config, now=NOW, data_cutoff=period.end)
    assert [paper.paper_id for paper in expected.papers] == ["a", "b", "c"]
    for inputs in permutations(records):
        for scoped in permutations(scores):
            assert (
                build_weekly(
                    period,
                    list(inputs) + [inputs[0]],
                    list(scoped) + [scoped[0]],
                    config,
                    now=NOW,
                    data_cutoff=period.end,
                )
                == expected
            )
    conflicting = scores[0].model_copy(update={"publication_score": 90})
    with pytest.raises(ValueError, match="Conflicting period scores"):
        build_weekly(
            period, records, scores + [conflicting], config, now=NOW, data_cutoff=period.end
        )


def test_membership_uses_original_publication_not_update_or_score_date() -> None:
    period = parse_week("2024-W01")
    older = record("older", period.start - timedelta(seconds=1))
    older.paper.updated_at = period.start + timedelta(days=2)
    next_week = record("next", period.end)
    first = record("first", period.start)
    last = record("last", period.end - timedelta(seconds=1))
    records = [older, next_week, first, last]
    scores = [period_score(item.paper.id, 95, period.end) for item in records]
    radar = build_weekly(period, records, scores, settings(), now=NOW, data_cutoff=period.end)
    assert radar.candidate_count == 2
    assert [paper.paper_id for paper in radar.papers] == ["first", "last"]


def test_source_fields_reasons_and_scores_come_from_the_supplied_inputs() -> None:
    period = parse_week("2024-W01")
    canonical = record("source", period.start)
    score = period_score("source", 95, period.end)
    score.source_paper = canonical.paper.model_copy(update={"arxiv_id": "sourcev2"})
    radar = build_weekly(period, [canonical], [score], settings(), now=NOW, data_cutoff=period.end)
    paper = radar.papers[0]
    assert paper.title == canonical.paper.title
    assert paper.paper_url == canonical.paper.abs_url
    assert paper.overall_score == 95  # The current record's overall=1 is not consulted.
    assert paper.trending_score == 0
    assert paper.paper_version == "sourcev2"
    assert "primary abstract describes mechanism A" in paper.selection_reason
    assert "confidence 80%" in paper.selection_reason
    assert "contribution 19.00" in paper.selection_reason
    assert "quality basis: publication_document_proxy" in paper.selection_reason
    assert "verified" not in paper.selection_reason.casefold()
    assert "beats" not in paper.selection_reason.casefold()
    assert radar.prompt_versions.analysis == settings().analysis_prompt_version
    assert radar.scoring_configuration == settings().radar_scoring
    assert radar.ranking_version == settings().ranking_version
    assert radar.data_cutoff == period.end
    assert radar.generated_at == NOW


def test_unknown_empty_weak_and_partial_discovery_are_distinct() -> None:
    unknown = week([None, None])
    assert unknown.radar_score is None
    assert unknown.status == "draft"
    assert unknown.candidate_count == 2
    assert unknown.scored_candidate_count == 0
    assert week([]).radar_score == 0
    assert week([60]).radar_score == 0
    assert week([60, None]).radar_score is None
    assert week([60, None]).status == "draft"
    assert week([], complete=False).radar_score is None
    assert week([60], complete=False).radar_score is None
    provisional = week([95], complete=False)
    assert provisional.radar_score is not None
    assert not provisional.is_complete
    assert provisional.status == "draft"
    period = parse_week("2024-W01")
    missing_score = build_weekly(
        period,
        [record("no-score", period.start)],
        [],
        settings(),
        now=NOW,
        data_cutoff=period.end,
    )
    assert missing_score.radar_score is None
    assert missing_score.candidate_count == 1
    partial_coverage = build_weekly(
        period,
        [],
        [],
        settings(),
        now=NOW,
        data_cutoff=period.end,
        coverage_end=period.end_date - timedelta(days=1),
    )
    assert not partial_coverage.is_complete
    assert partial_coverage.radar_score is None


@pytest.mark.parametrize(
    ("analysis_status", "expected"),
    [
        ("review", "review"),
        ("approved", "review"),
        ("published", "review"),
        ("needs_review", "needs_review"),
        ("draft", "draft"),
        (None, "draft"),
    ],
)
def test_selected_analysis_review_gate(
    analysis_status: RadarStatus | None, expected: RadarStatus
) -> None:
    period = parse_week("2024-W01")
    statuses = {"a": analysis_status} if analysis_status is not None else {}
    radar = build_weekly(
        period,
        [record("a", period.start)],
        [period_score("a", 95, period.end)],
        settings(),
        now=NOW,
        data_cutoff=period.end,
        analysis_statuses=statuses,
    )
    assert radar.status == expected


def test_preview_always_draft_and_scores_are_mode_isolated() -> None:
    assert week([95], preview=True).status == "draft"
    assert not week([95], preview=True).is_complete
    period = parse_week("2024-W01")
    inputs = [record("a", period.start)]
    retro = period_score("a", 96, NOW, "retrospective")
    publication = period_score("a", 75, period.end)
    radar = build_weekly(
        period, inputs, [retro, publication], settings(), now=NOW, data_cutoff=period.end
    )
    assert radar.papers[0].publication_score == 75
    assert radar.retrospective_score is None
    retrospective = build_weekly(
        period,
        inputs,
        [publication, retro],
        settings(),
        now=NOW,
        data_cutoff=NOW,
        mode="retrospective",
    )
    assert retrospective.radar_score is None
    assert retrospective.retrospective_score is not None
    assert retrospective.papers[0].retrospective_score == 96
    with pytest.raises(ValueError, match="cutoff"):
        build_weekly(
            period,
            inputs,
            [publication.model_copy(update={"data_cutoff": NOW})],
            settings(),
            now=NOW,
            data_cutoff=period.end,
        )


@pytest.mark.parametrize("mode", ["publication", "retrospective"])
def test_current_incomplete_week_is_automatically_preview_and_mode_isolated(
    tmp_path: Path, mode: RankingMode
) -> None:
    now = datetime(2024, 1, 3, 12, tzinfo=UTC)
    period = parse_week("2024-W01")
    source = record("current", period.start)
    config = settings(data_dir=tmp_path)
    radar = build_weekly(
        period,
        [source],
        [period_score(source.paper.id, 95, now, mode)],
        config,
        now=now,
        data_cutoff=now,
        mode=mode,
        is_preview=False,
        is_complete=True,
        analysis_statuses={source.paper.id: "needs_review"},
    )
    assert radar.is_preview
    assert not radar.is_complete
    assert radar.status == "draft"
    assert radar.coverage_end == now.date()
    _, paths = write_radar(radar, config)
    prefix = "radars/retrospective" if mode == "retrospective" else "radars"
    assert paths[0] == tmp_path / prefix / "previews/weeks/2024/2024-W01.json"
    assert validate_radar_files(tmp_path) == 1
    assert build_monthly(2024, 1, [radar], config, now=NOW, mode=mode).top_weeks == []
    assert build_yearly(2024, [], [radar], config, now=NOW, mode=mode).best_weeks == []


def test_publication_and_retrospective_previews_cannot_overwrite_each_other(tmp_path: Path) -> None:
    config = settings(data_dir=tmp_path)
    _, publication_paths = write_radar(week([95], preview=True), config)
    before = [path.read_bytes() for path in publication_paths]
    _, retrospective_paths = write_radar(week([99], preview=True, mode="retrospective"), config)
    assert [path.read_bytes() for path in publication_paths] == before
    assert set(publication_paths).isdisjoint(retrospective_paths)
    assert validate_radar_files(tmp_path) == 2


def test_january_fixture_highlights_exceptional_weeks_not_quantity() -> None:
    weeks = january()
    assert [radar.paper_count for radar in weeks] == [2, 0, 1, 3]
    radar = build_monthly(2024, 1, weeks, settings(), now=NOW)
    assert radar.is_complete
    assert radar.status == "review"
    assert radar.best_week == "2024-W01"
    assert [ref.id for ref in radar.top_weeks[:2]] == ["2024-W01", "2024-W03"]
    assert radar.paper_count == 6
    assert radar.known_week_count == 4
    assert radar.unknown_week_count == 0
    assert radar.score == pytest.approx(
        radar_score(
            [week.radar_score for week in weeks],
            settings().radar_scoring.monthly_coefficients,
            settings().radar_scoring,
        )
    )


def test_monthly_thursday_boundary_deduplication_and_mode_isolation() -> None:
    january_weeks = january()
    boundary = week([95], "2024-W05")
    retro = week([100], mode="retrospective")
    preview = week([100], "2024-W02", preview=True)
    inputs = [*january_weeks, boundary, january_weeks[0], retro, preview]
    result = build_monthly(2024, 1, inputs, settings(), now=NOW)
    assert result == build_monthly(2024, 1, january_weeks, settings(), now=NOW)
    february = build_monthly(2024, 2, inputs, settings(), now=NOW)
    assert [ref.id for ref in february.top_weeks] == ["2024-W05"]
    assert not february.is_complete
    retrospective = build_monthly(2024, 1, inputs, settings(), now=NOW, mode="retrospective")
    assert retrospective.score is None
    assert retrospective.retrospective_score is not None
    assert retrospective.known_week_count == 1
    assert retrospective.top_weeks[0].score == retro.retrospective_score


def test_unknown_monthly_children_do_not_turn_into_zero() -> None:
    all_unknown = build_monthly(
        2024, 1, [week([None]), week([None], "2024-W02")], settings(), now=NOW
    )
    assert all_unknown.score is None
    assert all_unknown.best_week is None
    assert all_unknown.known_week_count == 0
    assert all_unknown.unknown_week_count == 2
    assert not all_unknown.is_complete
    mixed = build_monthly(2024, 1, [week([None]), week([50], "2024-W02")], settings(), now=NOW)
    assert mixed.score == 0
    assert mixed.best_week == "2024-W02"
    assert mixed.top_weeks[0].score == 0
    assert mixed.top_weeks[1].score is None
    assert (mixed.known_week_count, mixed.unknown_week_count) == (1, 1)
    assert build_monthly(2024, 1, [], settings(), now=NOW).score is None


def test_month_is_not_complete_when_required_week_missing_or_unfinished() -> None:
    weeks = january()
    assert not build_monthly(2024, 1, weeks[:3], settings(), now=NOW).is_complete
    unfinished = week([95], "2024-W04", complete=False)
    assert not build_monthly(2024, 1, weeks[:3] + [unfinished], settings(), now=NOW).is_complete
    assert not build_monthly(
        2024, 1, weeks, settings(), now=datetime(2024, 1, 20, tzinfo=UTC)
    ).is_complete
    assert not build_monthly(
        2024, 1, weeks, settings(), now=datetime(2024, 1, 29, tzinfo=UTC)
    ).is_complete
    custom = settings(
        radar_scoring=RadarScoring(monthly_coefficients=[1, 0.1], yearly_coefficients=[1, 0.05])
    )
    assert (
        build_monthly(2024, 1, weeks, custom, now=NOW).score
        != build_monthly(2024, 1, weeks, settings(), now=NOW).score
    )


def test_cross_year_week_belongs_to_december_by_thursday_not_publication_day() -> None:
    weeks = [
        week([95], period.id)
        for period in iter_weeks(date(2020, 12, 1), date(2020, 12, 31))
        if period.month_key == (2020, 12)
    ]
    assert weeks[-1].id == "2020-W53"
    december = build_monthly(2020, 12, weeks, settings(), now=NOW)
    assert december.is_complete
    assert len(december.top_weeks) == 5
    assert not build_monthly(
        2020, 12, weeks, settings(), now=datetime(2021, 1, 1, tzinfo=UTC)
    ).is_complete
    january = build_monthly(2021, 1, weeks, settings(), now=NOW)
    assert january.top_weeks == []
    year = build_yearly(2020, [december], weeks, settings(), now=NOW)
    assert "2020-W53" in {ref.id for ref in year.best_weeks}
    next_year = build_yearly(2021, [december], weeks, settings(), now=NOW)
    assert next_year.top_papers == []
    assert next_year.best_weeks == []


def test_monthly_and_yearly_entire_output_is_independent_of_input_order() -> None:
    weeks = january()
    config = settings()
    month = build_monthly(2024, 1, weeks, config, now=NOW)
    for order in permutations(weeks):
        assert build_monthly(2024, 1, list(order), config, now=NOW) == month
    months = [
        month,
        build_monthly(2024, 2, [week([80], "2024-W06")], config, now=NOW),
        build_monthly(2024, 3, [], config, now=NOW),
    ]
    expected = build_yearly(2024, months, weeks, config, now=NOW)
    for order in permutations(months):
        assert build_yearly(2024, list(order), list(reversed(weeks)), config, now=NOW) == expected


def test_exceptional_period_year_beats_constant_mediocre() -> None:
    mediocre = all_year([75] * 10)
    exceptional = all_year([50] * 10)
    exceptional[0] = week([99, 98], "2024-W01")
    config = settings()
    mediocre_months = [
        build_monthly(2024, month, mediocre, config, now=NOW) for month in range(1, 13)
    ]
    exceptional_months = [
        build_monthly(2024, month, exceptional, config, now=NOW) for month in range(1, 13)
    ]
    steady = build_yearly(2024, mediocre_months, mediocre, config, now=NOW)
    peak = build_yearly(2024, exceptional_months, exceptional, config, now=NOW)
    assert peak.score > steady.score
    assert peak.is_complete and steady.is_complete
    assert peak.best_month == "2024-01"
    assert peak.best_weeks[0].id == "2024-W01"
    assert peak.top_papers == ["2024-W01-00", "2024-W01-01"]
    assert peak.paper_count == 2
    assert peak.known_month_count == 12
    assert peak.unknown_month_count == 0
    assert peak.status == "review"


def test_yearly_distinct_references_modes_unknowns_and_year_ownership() -> None:
    weeks = january()
    month = build_monthly(2024, 1, weeks, settings(), now=NOW)
    unknown = build_monthly(2024, 2, [week([None], "2024-W06")], settings(), now=NOW)
    retro_week = week([100], mode="retrospective")
    retro_month = build_monthly(2024, 1, [retro_week], settings(), now=NOW, mode="retrospective")
    next_year = week([100], "2025-W01")
    year = build_yearly(
        2024,
        [month, unknown, month, retro_month],
        [*weeks, weeks[0], retro_week, next_year, week([100], preview=True)],
        settings(),
        now=NOW,
    )
    assert year.best_month == "2024-01"
    assert year.known_month_count == year.unknown_month_count == 1
    assert not year.is_complete
    assert len(year.best_weeks) == 4
    assert len(year.top_papers) == len(set(year.top_papers)) == 6
    assert year.top_papers[0] == "2024-W03-00"
    assert year.retrospective_score is None
    retro = build_yearly(
        2024, [month, retro_month], [retro_week], settings(), now=NOW, mode="retrospective"
    )
    assert retro.score is None
    assert retro.retrospective_score is not None
    assert retro.top_papers == [retro_week.papers[0].paper_id]
    all_unknown = build_yearly(2024, [unknown], [], settings(), now=NOW)
    assert all_unknown.score is None
    assert all_unknown.best_month is None
    assert all_unknown.unknown_month_count == 1
    assert build_yearly(2024, [], [], settings(), now=NOW).score is None


def test_whole_schema_round_trips_and_rejects_incoherent_ids_or_modes() -> None:
    weekly = week([95])
    monthly = build_monthly(2024, 1, january(), settings(), now=NOW)
    yearly = build_yearly(2024, [monthly], january(), settings(), now=NOW)
    for radar in [weekly, monthly, yearly]:
        assert type(radar).model_validate_json(radar.model_dump_json()) == radar
        payload = radar.model_dump()
        payload["invented_evidence"] = "not allowed"
        with pytest.raises(ValidationError):
            type(radar).model_validate(payload)
    payload = weekly.model_dump()
    payload["week"] = 2
    with pytest.raises(ValidationError):
        WeeklyRadar.model_validate(payload)
    payload = monthly.model_dump()
    payload["retrospective_score"] = 80
    with pytest.raises(ValidationError):
        MonthlyRadar.model_validate(payload)
    payload = yearly.model_dump()
    payload["year"] = 2023
    with pytest.raises(ValidationError):
        YearlyRadar.model_validate(payload)


def test_weekly_markdown_sections_metadata_and_empty_human_block() -> None:
    radar = week([95, 94])
    markdown = render_markdown(radar)
    for heading in [
        "TL;DR",
        "Why should I care?",
        "Papers worth reading",
        "Why these papers?",
        "How strong was this week?",
        "Limitations",
        "Michał's take",
        "Resources",
        "Radar metadata",
    ]:
        assert f"## {heading}" in markdown
    for paper in radar.papers:
        assert paper.paper_url in markdown
        assert paper.selection_reason in markdown
    for value in [radar.scoring_version, radar.ranking_version, radar.data_cutoff.isoformat()]:
        assert value in markdown
    assert "- Mode: `publication`" in markdown
    assert "- Preview: false" in markdown
    assert "- Complete: true" in markdown
    assert extract_human_block(markdown) == ""
    assert "No qualifying papers selected." in render_markdown(week([60]))
    assert "missing evidence" in render_markdown(week([None]))


def test_output_paths_validation_and_preview_retrospective_separation(tmp_path: Path) -> None:
    config = settings(data_dir=tmp_path / "data")
    primary = week([95])
    monthly = build_monthly(2024, 1, january(), config, now=NOW)
    yearly = build_yearly(2024, [monthly], january(), config, now=NOW)
    expected = [
        (primary, "radars/weeks/2024/2024-W01.json"),
        (monthly, "radars/months/2024/2024-01.json"),
        (yearly, "radars/years/2024.json"),
        (week([96], mode="retrospective"), "radars/retrospective/weeks/2024/2024-W01.json"),
        (week([97], preview=True), "radars/previews/weeks/2024/2024-W01.json"),
    ]
    for radar, relative in expected:
        result, paths = write_radar(radar, config)
        assert result == radar
        assert paths == [
            config.data_dir / relative,
            (config.data_dir / relative).with_suffix(".md"),
        ]
        assert type(radar).model_validate_json(paths[0].read_text(encoding="utf-8")) == result
    assert validate_radar_files(config.data_dir) == 5
    month_md = (config.data_dir / "radars/months/2024/2024-01.md").read_text(encoding="utf-8")
    year_md = (config.data_dir / "radars/years/2024.md").read_text(encoding="utf-8")
    assert "../../weeks/2024/2024-W01.md" in month_md
    assert "../months/2024/2024-01.md" in year_md
    assert "../weeks/2024/2024-W01.md" in year_md
    assert primary.papers[0].paper_id in year_md


def test_regeneration_preserves_markdown_and_json_human_take(tmp_path: Path) -> None:
    config = settings(data_dir=tmp_path)
    radar = week([95])
    _, paths = write_radar(radar, config)
    json_path, md_path = paths
    markdown = md_path.read_text(encoding="utf-8")
    md_path.write_text(
        markdown.replace(HUMAN_START, HUMAN_START + "\nMy **own** editorial perspective."),
        encoding="utf-8",
    )
    result, again = write_radar(week([94]), config)
    assert again == paths
    assert result.michaels_take == "My **own** editorial perspective."
    assert extract_human_block(md_path.read_text(encoding="utf-8")) == result.michaels_take
    assert (
        WeeklyRadar.model_validate_json(json_path.read_text(encoding="utf-8")).michaels_take
        == result.michaels_take
    )
    assert result.radar_score != radar.radar_score
    assert validate_radar_files(tmp_path) == 1
    md_path.unlink()
    preserved, _ = write_radar(radar, config)
    assert preserved.michaels_take == result.michaels_take
    assert extract_human_block(md_path.read_text(encoding="utf-8")) == result.michaels_take


def test_incoming_machine_opinion_is_not_written(tmp_path: Path) -> None:
    config = settings(data_dir=tmp_path)
    radar = week([95])
    radar.michaels_take = "An autogenerated personal opinion must not become editorial content."
    result, paths = write_radar(radar, config)
    assert result.michaels_take is None
    assert extract_human_block(paths[1].read_text(encoding="utf-8")) == ""


@pytest.mark.parametrize("status", ["approved", "published"])
def test_approved_content_is_frozen_and_prior_model_returned(
    tmp_path: Path, status: RadarStatus
) -> None:
    config = settings(data_dir=tmp_path)
    original = week([95])
    _, paths = write_radar(original, config)
    prior = WeeklyRadar.model_validate_json(paths[0].read_text(encoding="utf-8"))
    prior.status = status
    paths[0].write_text(prior.model_dump_json(indent=2), encoding="utf-8")
    paths[1].write_text(
        paths[1]
        .read_text(encoding="utf-8")
        .replace("## TL;DR", "## TL;DR\n\nHuman approved text."),
        encoding="utf-8",
    )
    before = [path.read_bytes() for path in paths]
    recalculated = week([99])
    recalculated.scoring_configuration = RadarScoring(peak_share=0.1)
    result, written = write_radar(recalculated, config)
    assert result == prior
    assert written == []
    assert [path.read_bytes() for path in paths] == before


def test_conflicting_takes_fail_before_either_file_changes(tmp_path: Path) -> None:
    config = settings(data_dir=tmp_path)
    radar = week([95])
    _, paths = write_radar(radar, config)
    prior = WeeklyRadar.model_validate_json(paths[0].read_text(encoding="utf-8"))
    prior.michaels_take = "JSON human take"
    paths[0].write_text(prior.model_dump_json(), encoding="utf-8")
    paths[1].write_text(
        paths[1]
        .read_text(encoding="utf-8")
        .replace(HUMAN_START, HUMAN_START + "\nDifferent Markdown human take"),
        encoding="utf-8",
    )
    before = [path.read_bytes() for path in paths]
    with pytest.raises(ValueError, match="Conflicting human"):
        write_radar(week([96]), config)
    assert [path.read_bytes() for path in paths] == before
    with pytest.raises(ValueError, match="Conflicting human"):
        validate_radar_files(tmp_path)


@pytest.mark.parametrize("damage", ["missing_end", "duplicate_start", "reversed", "missing_both"])
def test_bad_human_boundaries_are_never_overwritten(tmp_path: Path, damage: str) -> None:
    config = settings(data_dir=tmp_path)
    radar = week([95])
    _, paths = write_radar(radar, config)
    content = paths[1].read_text(encoding="utf-8")
    if damage == "missing_end":
        content = content.replace(HUMAN_END, "")
    elif damage == "duplicate_start":
        content += HUMAN_START
    elif damage == "reversed":
        content = HUMAN_END + content.replace(HUMAN_END, "")
    else:
        content = content.replace(HUMAN_START, "").replace(HUMAN_END, "")
    paths[1].write_text(content, encoding="utf-8")
    before = [path.read_bytes() for path in paths]
    with pytest.raises(ValueError, match="boundaries"):
        write_radar(week([96]), config)
    assert [path.read_bytes() for path in paths] == before


def test_output_updates_only_exact_period_paths(tmp_path: Path) -> None:
    config = settings(data_dir=tmp_path)
    _, january_paths = write_radar(week([95]), config)
    _, february_paths = write_radar(week([95], "2024-W06"), config)
    unrelated = [path.read_bytes() for path in february_paths]
    write_radar(week([96]), config)
    assert [path.read_bytes() for path in february_paths] == unrelated
    assert january_paths[0].exists()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("paper_count", 9),
        ("scored_candidate_count", 0),
        ("candidate_count", 0),
        ("radar_score", 90),
        ("quality_threshold", 99),
        ("type", "monthly_radar"),
        ("id", "2024-W02"),
        ("mode", "retrospective"),
    ],
)
def test_validator_rejects_schema_and_semantic_corruption(
    tmp_path: Path, field: str, bad_value: object
) -> None:
    config = settings(data_dir=tmp_path)
    _, paths = write_radar(week([95]), config)
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload[field] = bad_value
    paths[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_radar_files(tmp_path)


def test_validator_rejects_wrong_paths_missing_pairs_and_false_completeness(tmp_path: Path) -> None:
    config = settings(data_dir=tmp_path)
    monthly = build_monthly(2024, 1, january()[:2], config, now=NOW)
    _, paths = write_radar(monthly, config)
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["is_complete"] = True
    paths[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Complete months"):
        validate_radar_files(tmp_path)
    payload["is_complete"] = False
    paths[0].write_text(json.dumps(payload), encoding="utf-8")
    misplaced = paths[0].with_name("2024-02.json")
    paths[0].rename(misplaced)
    with pytest.raises(ValueError, match="path"):
        validate_radar_files(tmp_path)
    misplaced.rename(paths[0])
    paths[1].unlink()
    with pytest.raises(ValueError, match="Markdown is missing"):
        validate_radar_files(tmp_path)


@pytest.mark.parametrize("discriminator", [None, "complete", [], {}, "monthly_radar"])
def test_output_refuses_missing_or_ambiguous_model_discriminator(
    tmp_path: Path, discriminator: object
) -> None:
    config = settings(data_dir=tmp_path)
    radar = week([95])
    _, paths = write_radar(radar, config)
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    if discriminator is None:
        payload.pop("type")
    else:
        payload["type"] = discriminator
    paths[0].write_text(json.dumps(payload), encoding="utf-8")
    before = [path.read_bytes() for path in paths]
    with pytest.raises(ValueError):
        write_radar(radar, config)
    with pytest.raises(ValueError):
        validate_radar_files(tmp_path)
    assert [path.read_bytes() for path in paths] == before


def test_full_year_and_retrospective_aggregates_validate(tmp_path: Path) -> None:
    config = settings(data_dir=tmp_path)
    weeks = all_year([95])
    months = [build_monthly(2024, month, weeks, config, now=NOW) for month in range(1, 13)]
    year = build_yearly(2024, months, weeks, config, now=NOW)
    write_radar(year, config)
    retro_week = week([95], mode="retrospective")
    retro_month = build_monthly(2024, 1, [retro_week], config, now=NOW, mode="retrospective")
    retro_year = build_yearly(
        2024, [retro_month], [retro_week], config, now=NOW, mode="retrospective"
    )
    for radar in [retro_month, retro_year]:
        write_radar(radar, config)
        assert radar.mode == "retrospective"
        assert radar.score is None
        assert radar.retrospective_score is not None
    assert validate_radar_files(tmp_path) == 3
