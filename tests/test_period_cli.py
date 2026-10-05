import io
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError
from rich.console import Console

from research_radar import cli
from research_radar.config import RadarScoring, Settings
from research_radar.models.analysis import PromptVersions
from research_radar.models.history import BackfillReport
from research_radar.models.radars import (
    MonthlyRadar,
    PeriodReference,
    RankingMode,
    WeeklyPaperSelection,
    WeeklyRadar,
    YearlyRadar,
)
from research_radar.models.run import RunReport
from research_radar.periods import parse_week, previous_week
from research_radar.pipeline import backfill
from research_radar.storage.database import Database
from research_radar.storage.repository import Repository

NOW = datetime(2025, 1, 6, 12, tzinfo=UTC)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        database_path=tmp_path / "radar.sqlite3",
        cache_dir=tmp_path / "cache",
        openai_api_key=None,
        github_token=None,
        semantic_scholar_api_key=None,
        huggingface_token=None,
        openalex_email=None,
        dry_run=True,
    )


def _week(
    identifier: str,
    score: float | None = 80,
    *,
    count: int = 1,
    mode: RankingMode = "publication",
    preview: bool = False,
    complete: bool = True,
) -> WeeklyRadar:
    period = parse_week(identifier)
    return WeeklyRadar(
        id=period.id,
        year=period.year,
        week=period.week,
        start_date=period.start_date,
        end_date=period.end_date,
        coverage_start=period.start_date,
        coverage_end=period.end_date,
        radar_score=score if mode == "publication" else None,
        retrospective_score=score if mode == "retrospective" else None,
        paper_count=count,
        papers=[
            WeeklyPaperSelection(
                paper_id=f"paper-{identifier}-{rank}",
                title=f"Title {identifier} paper {rank}",
                rank=rank,
                publication_score=91 if mode == "publication" else None,
                retrospective_score=97 if mode == "retrospective" else None,
                selection_reason="Supported quality exceeds threshold.",
            )
            for rank in range(1, count + 1)
        ],
        quality_threshold=70,
        generated_at=NOW,
        data_cutoff=min(period.end, NOW) if mode == "publication" else NOW,
        ranking_version="weekly-test",
        scoring_version="paper-test",
        scoring_configuration=RadarScoring(),
        prompt_versions=PromptVersions(relevance="v1", analysis="v1", critic="v1"),
        is_complete=complete and not preview,
        is_preview=preview,
        discovery_complete=complete,
        mode=mode,
        explanation="Primary-document evidence; unknown metrics excluded.",
    )


def _capture() -> tuple[Console, io.StringIO]:
    output = io.StringIO()
    return Console(file=output, width=240, force_terminal=False), output


@pytest.mark.parametrize(
    "command",
    ["best-week", "best-month", "best-year", "preview-week", "retrospective", "calibration"],
)
def test_period_commands_and_defaults(command: str) -> None:
    options = cli.parser().parse_args([command])
    assert options.command == command
    assert options.limit == 20
    assert options.mode == "publication"


def test_backfill_defaults_are_strict_resumable_and_no_llm() -> None:
    options = cli.parser().parse_args(["backfill", "--from", "2023-01-01"])
    assert options.start_date == date(2023, 1, 1)
    assert options.end_date == datetime.now(UTC).date()
    assert options.resume
    assert options.skip_analysis
    assert not options.analyze
    assert not options.classify
    assert not options.publication_proxy
    assert options.workers == 4


def test_backfill_explicit_options_and_analysis_alias() -> None:
    options = cli.parser().parse_args(
        [
            "backfill",
            "--year",
            "2024",
            "--workers",
            "16",
            "--skip-enrichment",
            "--analyze-selected",
            "--classify",
            "--confirm-expensive",
            "--no-resume",
            "--publication-proxy",
        ]
    )
    assert (options.start_date, options.end_date) == (date(2024, 1, 1), date(2024, 12, 31))
    assert options.analyze and not options.skip_analysis
    assert options.classify and options.confirm_expensive
    assert options.publication_proxy and options.skip_enrichment
    assert options.workers == 16 and not options.resume


@pytest.mark.parametrize(
    "arguments",
    [
        ["backfill"],
        ["backfill", "--year", "2024", "--from", "2024-01-01"],
        ["backfill", "--year", "2024", "--to", "2024-12-31"],
        ["retrospective", "--year", "2024", "--from", "2024-01-01"],
        ["backfill", "--from", "2024-02-30"],
        ["backfill", "--from", "2024-03-01", "--to", "2024-02-29"],
        ["backfill", "--from", "2023-01-01", "--to", "9998-12-31"],
        ["backfill", "--year", "9998"],
        ["backfill", "--year", "2024", "--workers", "0"],
        ["backfill", "--year", "2024", "--workers", "17"],
        ["backfill", "--year", "2024", "--analyze"],
        ["backfill", "--year", "2024", "--classify"],
        ["backfill", "--year", "2024", "--skip-analysis", "--analyze", "--confirm-expensive"],
        ["analyze-selected"],
        ["analyze-selected", "--period", "2024-W08", "--best-weeks", "20"],
        ["analyze-selected", "--best-weeks", "0"],
        ["analyze-selected", "--period", "2024-W08", "--top", "11"],
        ["run", "--days", "7", "--period", "2024-W08"],
        ["preview-week", "--days", "7"],
        ["best-week", "--workers", "2"],
        ["run", "--analyze", "--confirm-expensive"],
        ["run", "--resume"],
        ["best-week", "--period", "2024-W08"],
        ["best-month", "--from", "2024-04-01", "--to", "2024-03-31"],
    ],
)
def test_invalid_options_are_rejected_before_processing(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        cli.parser().parse_args(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize("identifier", ["2024W08", "2024-W08"])
def test_week_alias_is_canonicalized(identifier: str) -> None:
    options = cli.parser().parse_args(["analyze-selected", "--period", identifier])
    assert options.period == "2024-W08"


@pytest.mark.parametrize("identifier", ["2024-W00", "2024-W53", "2024-W54", "not-a-week"])
def test_invalid_iso_weeks_are_rejected(identifier: str) -> None:
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["run", "--period", identifier])


def test_year_and_leap_day_range_boundaries() -> None:
    options = cli.parser().parse_args(["backfill", "--from", "2020-12-31", "--to", "2021-01-04"])
    assert options.start_date == date(2020, 12, 31)
    assert options.end_date == date(2021, 1, 4)
    assert cli.parser().parse_args(["run", "--period", "2020-W53"]).period == "2020-W53"
    assert cli.parser().parse_args(
        ["retrospective", "--from", "2024-02-29", "--to", "2024-03-01"]
    ).start_date == date(2024, 2, 29)
    assert previous_week(datetime(2021, 1, 4, 7, tzinfo=UTC)).id == "2020-W53"


def test_date_error_does_not_echo_secret_like_input(capsys: pytest.CaptureFixture[str]) -> None:
    secret = "not-a-date-secret-token"
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["backfill", "--from", secret])
    assert secret not in capsys.readouterr().err


def _setup_main(monkeypatch: pytest.MonkeyPatch, settings: Settings, arguments: list[str]) -> Mock:
    monkeypatch.setattr(sys, "argv", ["research-radar", *arguments])
    monkeypatch.setattr(cli, "Settings", lambda **overrides: settings.model_copy(update=overrides))
    printer = Mock()
    monkeypatch.setattr(cli, "print_report", printer)
    return printer


def test_default_run_dispatches_weekly_not_legacy(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    printer = _setup_main(monkeypatch, settings, [])
    report = RunReport(id="test", started_at=NOW, stage="weekly", period_id="2025-W01")
    weekly = AsyncMock(return_value=report)
    legacy = AsyncMock()
    monkeypatch.setattr(cli, "run_weekly", weekly)
    monkeypatch.setattr(cli, "run_pipeline", legacy)
    cli.main()
    weekly.assert_awaited_once_with(settings, period_id=None, preview=False)
    legacy.assert_not_called()
    assert printer.call_args.args[0].period_id == previous_week(NOW).id


def test_explicit_days_remains_legacy(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    _setup_main(monkeypatch, settings, ["run", "--days", "30", "--top", "3"])
    legacy = AsyncMock(return_value=RunReport(id="test", started_at=NOW, stage="run"))
    weekly = AsyncMock()
    monkeypatch.setattr(cli, "run_pipeline", legacy)
    monkeypatch.setattr(cli, "run_weekly", weekly)
    cli.main()
    assert legacy.await_args.args[0].research_days == 30
    assert legacy.await_args.args[0].top_papers == 3
    assert legacy.await_args.args[1] == "run"
    weekly.assert_not_called()
    assert "Deprecated rolling run" in capsys.readouterr().out


def test_targeted_run_and_preview_dispatch(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    _setup_main(monkeypatch, settings, ["run", "--period", "2024W08"])
    weekly = AsyncMock(return_value=RunReport(id="test", started_at=NOW, stage="weekly"))
    monkeypatch.setattr(cli, "run_weekly", weekly)
    cli.main()
    weekly.assert_awaited_once_with(settings, period_id="2024-W08", preview=False)
    _setup_main(monkeypatch, settings, ["preview-week", "--top", "3"])
    weekly.reset_mock()
    cli.main()
    assert weekly.await_args.kwargs == {"period_id": None, "preview": True}
    assert weekly.await_args.args[0].weekly_max_papers == 3


@pytest.mark.parametrize(
    "arguments",
    [
        ["backfill", "--year", "2024", "--analyze"],
        ["backfill", "--year", "2024", "--classify"],
    ],
)
def test_cost_guard_precedes_settings_and_providers(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["research-radar", *arguments])
    constructor = Mock(side_effect=AssertionError("must not initialize settings"))
    provider = Mock(side_effect=AssertionError("must not initialize a provider"))
    monkeypatch.setattr(cli, "Settings", constructor)
    monkeypatch.setattr(backfill, "OpenAIProvider", provider)
    with pytest.raises(SystemExit):
        cli.main()
    constructor.assert_not_called()
    provider.assert_not_called()


def test_backfill_dispatch_keeps_strict_defaults_even_with_settings_proxy(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    settings.publication_proxy = True
    _setup_main(monkeypatch, settings, ["backfill", "--year", "2024"])
    runner = AsyncMock(
        return_value=BackfillReport(
            job_id="fixture",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            total_weeks=53,
            completed_weeks=53,
        )
    )
    monkeypatch.setattr(backfill, "run_backfill", runner)
    cli.main()
    options = runner.await_args.args[1]
    assert options.start_date == date(2024, 1, 1)
    assert options.end_date == date(2024, 12, 31)
    assert not options.analyze and not options.classify
    assert options.resume and not options.publication_proxy


def test_analyze_selected_dispatch_has_one_global_budget(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    _setup_main(
        monkeypatch,
        settings,
        [
            "analyze-selected",
            "--best-weeks",
            "20",
            "--top",
            "10",
            "--mode",
            "retrospective",
        ],
    )
    runner = AsyncMock(
        return_value=RunReport(
            id="test",
            started_at=NOW,
            stage="analyze-selected",
        )
    )
    monkeypatch.setattr(cli, "analyze_selected", runner)
    cli.main()
    runner.assert_awaited_once()
    assert runner.await_args.args[0].top_papers == 10
    assert runner.await_args.kwargs == {
        "period_id": None,
        "best_weeks": 20,
        "mode": "retrospective",
        "confirm_expensive": False,
    }


def test_environment_budget_cost_guard_before_selected_analysis(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    settings.top_papers = 11
    _setup_main(monkeypatch, settings, ["analyze-selected", "--period", "2024-W08"])
    runner = AsyncMock()
    monkeypatch.setattr(cli, "analyze_selected", runner)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    runner.assert_not_called()


def test_retrospective_dispatch_only_reuses_local_state(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    _setup_main(monkeypatch, settings, ["retrospective", "--year", "2024"])
    runner = AsyncMock(return_value=[])
    weekly = AsyncMock(side_effect=AssertionError("no network orchestration"))
    legacy = AsyncMock(side_effect=AssertionError("no legacy enrichment"))
    monkeypatch.setattr(cli, "run_retrospective", runner)
    monkeypatch.setattr(cli, "run_weekly", weekly)
    monkeypatch.setattr(cli, "run_pipeline", legacy)
    cli.main()
    runner.assert_awaited_once_with(
        settings, start_date=date(2024, 1, 1), end_date=date(2024, 12, 31)
    )
    weekly.assert_not_called()
    legacy.assert_not_called()


def test_best_week_mode_identity_stable_ties_unknown_and_preview(settings: Settings) -> None:
    with Database(settings.database_path) as database:
        repository = Repository(database)
        for week in [
            _week("2024-W02", 90),
            _week("2024-W01", 90),
            _week("2024-W03", None, count=0),
            _week("2024-W01", 81, mode="retrospective"),
            _week("2025-W02", 99, preview=True),
        ]:
            repository.save_weekly(week)
        arguments = cli.parser().parse_args(["best-week", "--year", "2024"])
        radars = cli._best(repository, arguments)
        assert {radar.id for radar in radars} == {"2024-W01", "2024-W02", "2024-W03"}
        console, output = _capture()
        cli.print_radars(radars, repository, console)
        text = output.getvalue()
        assert text.index("2024-W01") < text.index("2024-W02")
        assert "Unknown scores: 1" in text
        assert "publication" in text and "91.00" in text
        assert "2025-W02" not in text
        arguments = cli.parser().parse_args(["best-week", "--mode", "retrospective"])
        console, output = _capture()
        cli.print_radars(cli._best(repository, arguments), repository, console)
        assert "81.00" in output.getvalue() and "97.00" in output.getvalue()
        assert "retrospective" in output.getvalue()
        assert "2024-W02" not in output.getvalue()


def test_best_week_iso_year_not_calendar_year(settings: Settings) -> None:
    with Database(settings.database_path) as database:
        repository = Repository(database)
        repository.save_weekly(_week("2020-W53"))
        repository.save_weekly(_week("2021-W01"))
        radars = cli._best(
            repository,
            cli.parser().parse_args(
                [
                    "best-week",
                    "--year",
                    "2020",
                    "--from",
                    "2021-01-01",
                    "--to",
                    "2021-01-03",
                ]
            ),
        )
        assert [radar.id for radar in radars] == ["2020-W53"]


def test_month_and_year_tables_expose_drilldowns_and_period_filters(settings: Settings) -> None:
    week = _week("2024-W01")
    provenance = {
        name: getattr(week, name)
        for name in (
            "generated_at",
            "data_cutoff",
            "ranking_version",
            "scoring_version",
            "scoring_configuration",
            "prompt_versions",
        )
    }
    ref = PeriodReference(id=week.id, score=80, paper_count=1)
    month = MonthlyRadar(
        id="2024-01",
        year=2024,
        month=1,
        score=75,
        best_week=week.id,
        top_weeks=[ref],
        known_week_count=1,
        unknown_week_count=3,
        paper_count=1,
        explanation="One known week.",
        **provenance,
    )
    year = YearlyRadar(
        id="2024",
        year=2024,
        score=72,
        best_month=month.id,
        top_months=[PeriodReference(id=month.id, score=75, paper_count=1)],
        best_weeks=[ref],
        known_month_count=1,
        unknown_month_count=11,
        paper_count=1,
        explanation="One known month.",
        **provenance,
    )
    with Database(settings.database_path) as database:
        repository = Repository(database)
        repository.save_weekly(week)
        repository.save_monthly(month)
        repository.save_yearly(year)
        console, output = _capture()
        cli.print_radars([month, year], repository, console)
        text = output.getvalue()
        assert "best week: 2024-W01" in text
        assert "best month: 2024-01" in text
        assert "best weeks: 2024-W01" in text
        assert "INCOMPLETE" in text
        assert "Known/unknown weeks: 1/3" in text
        assert "Known/unknown months: 1/11" in text
        assert "91.00" in text
        assert (
            cli._best(
                repository,
                cli.parser().parse_args(
                    [
                        "best-month",
                        "--from",
                        "2024-02-01",
                    ]
                ),
            )
            == []
        )
        assert [
            radar.id
            for radar in cli._best(
                repository,
                cli.parser().parse_args(
                    [
                        "best-year",
                        "--year",
                        "2024",
                    ]
                ),
            )
        ] == ["2024"]


def test_calibration_unknown_is_not_a_weak_zero_paper_week() -> None:
    weeks = [
        _week("2024-W01", 0, count=0),
        _week("2024-W02", 80, count=1),
        _week("2024-W03", 80, count=3),
        _week("2024-W04", 80, count=9),
        _week("2024-W05", 80, count=10),
        _week("2024-W06", None, count=0),
    ]
    console, output = _capture()
    cli.print_calibration(weeks, console)
    text = output.getvalue()
    for label in ("0", "1", "2–3", "4–9", ">=10"):
        assert label in text
    assert "Unknown-score weeks (excluded from bins): 1" in text
    assert "quota failure" in text
    # Every known-score bin has exactly one week; unknown is not another zero.
    rows = [line for line in text.splitlines() if "│" in line]
    assert len(rows) == 5
    assert all(row.split("│")[2].strip() == "1" for row in rows)


def test_preview_report_is_incomplete_and_never_global_ranking(settings: Settings) -> None:
    preview = _week("2025-W02", preview=True)
    with Database(settings.database_path) as database:
        Repository(database).save_weekly(preview)
    report = RunReport(
        id="test",
        started_at=NOW,
        stage="preview-week",
        period_id=preview.id,
        is_preview=True,
    )
    console, output = _capture()
    cli.print_report(report, settings, console)
    text = output.getvalue()
    assert "INCOMPLETE, not canonical/approved" in text
    assert "incomplete preview" in text
    assert "Top Trending" not in text
    with pytest.raises(ValidationError):
        WeeklyRadar.model_validate({**preview.model_dump(), "status": "approved"})


def test_best_and_calibration_commands_never_load_full_paper_corpus(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setattr(
        Repository, "list_records", Mock(side_effect=AssertionError("full corpus not needed"))
    )
    for arguments in (["best-week"], ["best-month"], ["best-year"], ["calibration"]):
        _setup_main(monkeypatch, settings, arguments)
        cli.main()
