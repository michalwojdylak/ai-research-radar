import argparse
import asyncio
import calendar
import logging
import re
from collections.abc import Iterable
from datetime import UTC, date, datetime
from typing import TypedDict, cast, overload

from pydantic import ValidationError
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.text import Text

from research_radar.config import Settings
from research_radar.models.radars import MonthlyRadar, RankingMode, WeeklyRadar, YearlyRadar
from research_radar.models.run import RunReport
from research_radar.periods import parse_week
from research_radar.pipeline.generate import validate_generated
from research_radar.pipeline.runner import STAGES, run_pipeline
from research_radar.pipeline.weekly import analyze_selected, run_retrospective, run_weekly
from research_radar.storage.database import Database
from research_radar.storage.repository import Repository


class CLIOverrides(TypedDict, total=False):
    research_days: int
    top_papers: int
    dry_run: bool
    weekly_max_papers: int


type Radar = WeeklyRadar | MonthlyRadar | YearlyRadar
PERIOD_COMMANDS = (
    "backfill",
    "analyze-selected",
    "best-week",
    "best-month",
    "best-year",
    "retrospective",
    "preview-week",
    "calibration",
)


def _date(value: str) -> date:
    try:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError
        return date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Use a valid date in YYYY-MM-DD format.") from None


def _period(value: str) -> str:
    if re.fullmatch(r"\d{4}W\d{2}", value):
        value = value[:4] + "-" + value[4:]
    try:
        return parse_week(value).id
    except ValueError:
        raise argparse.ArgumentTypeError("Use a valid ISO week, for example 2024-W08.") from None


def _positive(value: str) -> int:
    try:
        result = int(value)
        if result < 1:
            raise ValueError
        return result
    except ValueError:
        raise argparse.ArgumentTypeError("Use a positive integer.") from None


def _workers(value: str) -> int:
    result = _positive(value)
    if result > 16:
        raise argparse.ArgumentTypeError("--workers must be between 1 and 16.")
    return result


def _year(value: str) -> int:
    result = _positive(value)
    if result > 9998:
        raise argparse.ArgumentTypeError("Use a year between 1 and 9998.")
    return result


class RadarArgumentParser(argparse.ArgumentParser):
    @overload
    def parse_args(
        self, args: Iterable[str] | None = None, namespace: None = None
    ) -> argparse.Namespace: ...

    @overload
    def parse_args[T](self, args: Iterable[str] | None, namespace: T) -> T: ...

    @overload
    def parse_args[T](self, *, namespace: T) -> T: ...

    def parse_args(
        self, args: Iterable[str] | None = None, namespace: object | None = None
    ) -> object:
        options = cast(argparse.Namespace, super().parse_args(args, namespace))
        command = options.command
        allowed: dict[str, set[str]] = {
            "days": {"run", "discover"},
            "period": {"run", "analyze-selected"},
            "best_weeks": {"analyze-selected"},
            "workers": {"backfill"},
            "skip_enrichment": {"backfill"},
            "analyze": {"backfill"},
            "classify": {"backfill"},
            "publication_proxy": {"backfill"},
            "no_resume": {"backfill"},
            "resume": {"backfill"},
            "year": {
                "backfill",
                "retrospective",
                "calibration",
                "best-week",
                "best-month",
                "best-year",
            },
            "from_date": {"backfill", "retrospective", "best-week", "best-month", "best-year"},
            "to_date": {"backfill", "retrospective", "best-week", "best-month", "best-year"},
            "mode": {"analyze-selected", "calibration", "best-week", "best-month", "best-year"},
            "limit": {"best-week", "best-month", "best-year", "retrospective"},
            "confirm_expensive": {"backfill", "analyze-selected"},
            "skip_analysis": {"backfill"},
        }
        for name, commands in allowed.items():
            if getattr(options, name) not in (None, False) and command not in commands:
                self.error(
                    f"--{name.replace('_date', '').replace('_', '-')} is not valid for {command}."
                )
        if command in {"backfill", "retrospective"}:
            if options.year is not None and (
                options.from_date is not None or options.to_date is not None
            ):
                self.error("--year cannot be combined with --from or --to.")
            if command == "backfill" and options.year is None and options.from_date is None:
                self.error("backfill requires --from or --year.")
            today = datetime.now(UTC).date()
            if options.year is not None:
                options.start_date = date(options.year, 1, 1)
                options.end_date = min(date(options.year, 12, 31), today)
            else:
                options.start_date = options.from_date or date(2023, 1, 1)
                options.end_date = options.to_date or today
            if options.end_date > today:
                self.error("--to cannot be in the future.")
            if options.start_date > options.end_date:
                self.error("The start date must not follow the end date.")
        elif (
            options.from_date is not None
            and options.to_date is not None
            and options.from_date > options.to_date
        ):
            self.error("--from must not follow --to.")
        if options.days is not None and options.period is not None:
            self.error("--days is a legacy rolling window and cannot be combined with --period.")
        if command == "backfill" and (options.analyze or options.classify):
            if not options.confirm_expensive:
                self.error(
                    "--analyze and --classify require --confirm-expensive before processing."
                )
        if command == "analyze-selected":
            if (options.period is None) == (options.best_weeks is None):
                self.error("analyze-selected requires exactly one of --period or --best-weeks.")
            if options.top is not None and options.top > 10 and not options.confirm_expensive:
                self.error("A budget above ten requires --confirm-expensive.")
        options.mode = options.mode or "publication"
        options.limit = options.limit if options.limit is not None else 20
        options.workers = options.workers if options.workers is not None else 4
        options.resume = not options.no_resume
        options.skip_analysis = not options.analyze
        return options


def parser() -> argparse.ArgumentParser:
    root = RadarArgumentParser(
        prog="research-radar",
        description="Weekly AI research Radars: best week → month → year, for human review.",
    )
    root.add_argument(
        "command", nargs="?", default="run", choices=["run", *STAGES, "validate", *PERIOD_COMMANDS]
    )
    root.add_argument("--days", type=_positive, help="Opt-in legacy rolling discovery window")
    root.add_argument("--top", type=_positive, help="Global deep-analysis budget (default 10)")
    root.add_argument("--period", type=_period, help="Stored/publication ISO week, e.g. 2024-W08")
    root.add_argument(
        "--best-weeks", type=_positive, help="Select from this many best stored weeks"
    )
    root.add_argument("--from", dest="from_date", type=_date, help="Inclusive start date")
    root.add_argument("--to", dest="to_date", type=_date, help="Inclusive end date (default today)")
    root.add_argument("--year", type=_year, help="Calendar range, or ISO year for best-week")
    root.add_argument("--mode", choices=["publication", "retrospective"], default=None)
    root.add_argument("--limit", type=_positive, help="Maximum ranked periods shown (default 20)")
    root.add_argument("--workers", type=_workers, help="Backfill workers, 1–16 (default 4)")
    root.add_argument(
        "--skip-enrichment", action="store_true", help="Avoid metadata enrichment calls"
    )
    analysis = root.add_mutually_exclusive_group()
    analysis.add_argument(
        "--skip-analysis", action="store_true", help="Skip deep analysis (default)"
    )
    analysis.add_argument(
        "--analyze",
        "--analyze-selected",
        dest="analyze",
        action="store_true",
        help="Enable selected backfill deep analysis; requires confirmation",
    )
    root.add_argument(
        "--confirm-expensive", action="store_true", help="Confirm explicit LLM spending"
    )
    root.add_argument(
        "--classify",
        action="store_true",
        help="Enable backfill LLM classification; requires confirmation",
    )
    root.add_argument(
        "--publication-proxy",
        action="store_true",
        help="Allow labelled approximate historical primary-document quality estimates",
    )
    resume = root.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume", action="store_true", help="Resume backfill checkpoints (default)"
    )
    resume.add_argument("--no-resume", action="store_true", help="Reprocess the backfill range")
    dry = root.add_mutually_exclusive_group()
    dry.add_argument(
        "--dry-run", action="store_true", help="Write local output without GitHub writes"
    )
    dry.add_argument(
        "--no-dry-run", action="store_true", help="Explicitly enable draft PR submission"
    )
    return root


def _range(radar: Radar) -> tuple[date, date]:
    if isinstance(radar, WeeklyRadar):
        return radar.start_date, radar.end_date
    if isinstance(radar, MonthlyRadar):
        return date(radar.year, radar.month, 1), date(
            radar.year, radar.month, calendar.monthrange(radar.year, radar.month)[1]
        )
    return date(radar.year, 1, 1), date(radar.year, 12, 31)


def _drilldown(radar: Radar) -> str:
    if isinstance(radar, MonthlyRadar):
        return f"best week: {radar.best_week or 'unknown'}; weeks: " + ", ".join(
            child.id for child in radar.top_weeks
        )
    if isinstance(radar, YearlyRadar):
        return f"best month: {radar.best_month or 'unknown'}; best weeks: " + ", ".join(
            child.id for child in radar.best_weeks
        )
    return ", ".join(paper.paper_id for paper in radar.papers)


def _top_paper(radar: Radar, repository: Repository) -> str:
    week: WeeklyRadar | None = None
    if isinstance(radar, WeeklyRadar):
        week = radar
    elif isinstance(radar, MonthlyRadar) and radar.best_week:
        week = repository.get_weekly(radar.best_week, mode=radar.mode)
    elif isinstance(radar, YearlyRadar) and radar.best_weeks:
        week = repository.get_weekly(radar.best_weeks[0].id, mode=radar.mode)
    if week is None or not week.papers:
        return "Unknown / no selection"
    paper = week.papers[0]
    score = paper.publication_score if radar.mode == "publication" else paper.retrospective_score
    value = "unknown" if score is None else f"{score:.2f}"
    return f"{paper.title} [{paper.paper_id}] — {value}"


def print_radars(
    radars: list[Radar], repository: Repository, console: Console, *, limit: int = 20
) -> None:
    known = sorted(
        (radar for radar in radars if radar.ranking_score is not None),
        key=lambda radar: (-float(radar.ranking_score or 0), radar.id),
    )
    unknown = sorted(
        (radar for radar in radars if radar.ranking_score is None), key=lambda radar: radar.id
    )
    modes = ", ".join(sorted({radar.mode for radar in radars}))
    table = Table(
        title=f"Best periods — {modes or 'no stored periods'} effective score "
        "(unknown periods are not ranked)"
    )
    for label in (
        "Rank",
        "Period",
        "Date range",
        "Score",
        "Selected",
        "Top paper / score",
        "Mode / scope",
        "Explanation / drilldown",
    ):
        table.add_column(label)
    for rank, radar in enumerate(known[:limit], 1):
        start, end = _range(radar)
        scope = (
            "incomplete preview"
            if isinstance(radar, WeeklyRadar) and radar.is_preview
            else ("complete" if radar.is_complete else "INCOMPLETE")
        )
        explanation = radar.explanation
        if isinstance(radar, MonthlyRadar):
            explanation += (
                f" Known/unknown weeks: {radar.known_week_count}/{radar.unknown_week_count}."
            )
        elif isinstance(radar, YearlyRadar):
            explanation += (
                f" Known/unknown months: {radar.known_month_count}/{radar.unknown_month_count}."
            )
        table.add_row(
            *(
                Text(value)
                for value in (
                    str(rank),
                    radar.id,
                    f"{start} – {end}",
                    f"{radar.ranking_score:.2f}",
                    str(radar.paper_count),
                    _top_paper(radar, repository),
                    f"{radar.mode}\n{scope}",
                    f"{explanation}\n{_drilldown(radar)}",
                )
            )
        )
    console.print(table)
    if unknown:
        console.print(
            f"Unknown scores: {len(unknown)} period(s), not weak/zero and not ranked: "
            + ", ".join(
                f"{radar.id} ({radar.mode}, "
                f"{'complete' if radar.is_complete else 'INCOMPLETE'}, "
                f"selected={radar.paper_count})"
                for radar in unknown[:limit]
            ),
            style="yellow",
            markup=False,
        )
    if not radars:
        console.print("No stored Radars match this scope.", markup=False)
    if any(not radar.is_complete for radar in radars):
        console.print(
            "Warning: incomplete coverage is not a complete historical result.", style="yellow"
        )


def _best(repository: Repository, arguments: argparse.Namespace) -> list[Radar]:
    mode: RankingMode = arguments.mode
    if arguments.command == "best-week":
        return list(
            repository.list_weekly(
                mode=mode,
                year=arguments.year,
                start_date=arguments.from_date,
                end_date=arguments.to_date,
            )
        )
    radars: list[Radar]
    if arguments.command == "best-month":
        radars = list(repository.list_monthly(mode=mode, year=arguments.year))
    else:
        radars = [
            radar
            for radar in repository.list_yearly(mode=mode)
            if arguments.year is None or radar.year == arguments.year
        ]
    return [
        radar
        for radar in radars
        if (arguments.from_date is None or _range(radar)[1] >= arguments.from_date)
        and (arguments.to_date is None or _range(radar)[0] <= arguments.to_date)
    ]


def print_calibration(
    weeks: list[WeeklyRadar], console: Console, *, mode: RankingMode = "publication"
) -> None:
    labels = ("0", "1", "2–3", "4–9", ">=10")
    counts = dict.fromkeys(labels, 0)
    unknown = 0
    for week in weeks:
        if week.ranking_score is None:
            unknown += 1
            continue
        size = week.paper_count
        label = (
            "0"
            if size == 0
            else "1"
            if size == 1
            else "2–3"
            if size < 4
            else ("4–9" if size < 10 else ">=10")
        )
        counts[label] += 1
    table = Table(title=f"Weekly selection calibration — {mode} (stored, non-preview weeks)")
    table.add_column("Selected papers")
    table.add_column("Weeks with known scores")
    for label in labels:
        table.add_row(label, str(counts[label]))
    console.print(table)
    console.print(f"Unknown-score weeks (excluded from bins): {unknown}", markup=False)
    console.print(
        f"Incomplete weeks: {sum(not week.is_complete for week in weeks)}; "
        "zero selected papers is valid, not a quota failure.",
        markup=False,
    )


def print_report(
    report: RunReport, settings: Settings, console: Console, *, mode: RankingMode = "publication"
) -> None:
    console.rule("AI Research Radar")
    table = Table(show_header=False, box=None)
    for label, value in (
        ("Discovered", report.discovered),
        ("Relevant", report.relevant),
        ("Enriched", report.enriched),
        ("Ranked", report.ranked),
        ("Deep analyzed", report.deep_analyzed),
        ("Critic passed", report.critic_passed),
        ("Needs review", report.needs_review),
        ("Generated", report.generated),
        ("Skipped analysis", report.skipped_analysis),
        ("Failures", len(report.failures)),
    ):
        table.add_row(f"{label}:", str(value))
    console.print(table)
    console.print(f"Run status: {report.status}")
    with Database(settings.database_path) as database:
        repository = Repository(database)
        if report.period_id:
            radar = repository.get_weekly(report.period_id, mode=mode, preview=report.is_preview)
            if radar:
                console.print(
                    "Current-week preview — INCOMPLETE, not canonical/approved."
                    if report.is_preview
                    else f"Publication week: {report.period_id}",
                    markup=False,
                )
                hierarchy: list[Radar] = [radar]
                if not report.is_preview:
                    year, month = parse_week(radar.id).month_key
                    monthly = repository.get_monthly(f"{year:04}-{month:02}", mode=radar.mode)
                    yearly = repository.get_yearly(f"{year:04}", mode=radar.mode)
                    hierarchy.extend(item for item in (monthly, yearly) if item is not None)
                for item in hierarchy:
                    print_radars([item], repository, console)
            rankings = None
        elif report.stage in {"weekly", "preview-week", "analyze-selected", "backfill"}:
            rankings = None
        else:
            rankings = repository.latest_rankings()
    if rankings is not None:
        for title, entries in (
            ("Top Trending", rankings.trending),
            ("Top Important", rankings.most_important),
        ):
            console.print(f"\n[bold]{title}[/bold]")
            for entry in entries[: settings.top_papers]:
                console.print(
                    f"{entry.rank}. {entry.title} — {entry.score:.2f} "
                    f"(coverage {entry.coverage:.0%})",
                    markup=False,
                )
    if report.failures:
        console.print("\n[bold yellow]Failures are recorded in SQLite.[/bold yellow]")
        for failure in report.failures:
            console.print(
                f"{failure.stage}: {failure.paper_id or 'pipeline'} — {failure.error_type}",
                markup=False,
            )


def main() -> None:
    arguments = parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=False, markup=False)],
    )
    # Third-party DEBUG/INFO HTTP logs can disclose query parameters; keep them quiet.
    for name in ("httpx", "httpcore", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)
    console = Console()
    try:
        overrides: CLIOverrides = {}
        if arguments.days is not None:
            overrides["research_days"] = arguments.days
        if arguments.top is not None:
            overrides["top_papers"] = arguments.top
        if arguments.dry_run:
            overrides["dry_run"] = True
        elif arguments.no_dry_run:
            overrides["dry_run"] = False
        if arguments.command == "preview-week" and arguments.top is not None:
            overrides["weekly_max_papers"] = arguments.top
        settings = Settings(**overrides)
        if (
            arguments.command == "analyze-selected"
            and settings.top_papers > 10
            and not arguments.confirm_expensive
        ):
            console.print("A global budget above ten requires --confirm-expensive.", style="red")
            raise SystemExit(2)
        if arguments.command == "validate":
            count = validate_generated(settings.data_dir)
            console.print(f"Validated {count} generated JSON documents.")
            return
        if arguments.command.startswith("best-") or arguments.command == "calibration":
            with Database(settings.database_path) as database:
                repository = Repository(database)
                if arguments.command == "calibration":
                    print_calibration(
                        repository.list_weekly(mode=arguments.mode, year=arguments.year),
                        console,
                        mode=arguments.mode,
                    )
                else:
                    console.rule(
                        {
                            "best-week": "BEST WEEKS",
                            "best-month": "BEST MONTHS",
                            "best-year": "BEST YEARS",
                        }[arguments.command]
                    )
                    print_radars(
                        _best(repository, arguments), repository, console, limit=arguments.limit
                    )
            return
        if arguments.command == "backfill":
            from research_radar.pipeline.backfill import BackfillOptions, run_backfill

            options = BackfillOptions(
                start_date=arguments.start_date,
                end_date=arguments.end_date,
                workers=arguments.workers,
                skip_enrichment=arguments.skip_enrichment,
                analyze=arguments.analyze,
                confirm_expensive=arguments.confirm_expensive,
                resume=arguments.resume,
                classify=arguments.classify,
                publication_proxy=arguments.publication_proxy,
            )
            result = asyncio.run(run_backfill(settings, options))
            console.print(result.model_dump_json(indent=2), markup=False)
            return
        if arguments.command == "retrospective":
            radars = asyncio.run(
                run_retrospective(
                    settings,
                    start_date=arguments.start_date,
                    end_date=arguments.end_date,
                )
            )
            with Database(settings.database_path) as database:
                print_radars(list(radars), Repository(database), console, limit=arguments.limit)
            return
        if arguments.command == "analyze-selected":
            report = asyncio.run(
                analyze_selected(
                    settings,
                    period_id=arguments.period,
                    best_weeks=arguments.best_weeks,
                    mode=arguments.mode,
                    confirm_expensive=arguments.confirm_expensive,
                )
            )
            if report.period_id is None:
                console.print(
                    f"Selected across {arguments.best_weeks} best weeks; "
                    f"global deep-analysis budget: {settings.top_papers}.",
                    markup=False,
                )
        elif arguments.command == "preview-week" or (
            arguments.command == "run" and arguments.days is None
        ):
            report = asyncio.run(
                run_weekly(
                    settings,
                    period_id=arguments.period,
                    preview=arguments.command == "preview-week",
                )
            )
        else:
            if arguments.command == "run" and arguments.days is not None:
                console.print(
                    "Deprecated rolling run (--days): using the legacy per-paper pipeline. "
                    "Omit --days for the previous complete ISO-week Radar.",
                    style="yellow",
                )
            report = asyncio.run(run_pipeline(settings, arguments.command))
        if arguments.mode == "retrospective":
            print_report(report, settings, console, mode="retrospective")
        else:
            print_report(report, settings, console)
    except ValidationError as error:
        details = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in error.errors(include_input=False, include_url=False)
        )
        console.print(f"Validation failed: {details}", style="red", markup=False)
        raise SystemExit(1) from None
    except (ValueError, OSError) as error:
        console.print(f"[red]Unable to complete: {type(error).__name__}.[/red]")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
