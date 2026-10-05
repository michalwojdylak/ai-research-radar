import argparse
import asyncio
import logging
from typing import TypedDict

from pydantic import ValidationError
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from research_radar.config import Settings
from research_radar.models.run import RunReport
from research_radar.pipeline.generate import validate_generated
from research_radar.pipeline.runner import STAGES, run_pipeline
from research_radar.storage.database import Database
from research_radar.storage.repository import Repository


class CLIOverrides(TypedDict, total=False):
    research_days: int
    top_papers: int
    dry_run: bool


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="research-radar",
        description="AI research discovery, ranking, and human-review drafts.",
    )
    root.add_argument("command", nargs="?", default="run", choices=["run", *STAGES, "validate"])
    root.add_argument("--days", type=int, help="Discovery window in days")
    root.add_argument("--top", type=int, help="Total number of papers selected for deep analysis")
    dry = root.add_mutually_exclusive_group()
    dry.add_argument(
        "--dry-run", action="store_true", help="Write local output without GitHub writes"
    )
    dry.add_argument(
        "--no-dry-run", action="store_true", help="Explicitly enable draft PR submission"
    )
    return root


def print_report(report: RunReport, settings: Settings, console: Console) -> None:
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
        rankings = Repository(database).latest_rankings()
    if rankings:
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
        settings = Settings(**overrides)
        if arguments.command == "validate":
            count = validate_generated(settings.data_dir)
            console.print(f"Validated {count} generated JSON documents.")
            return
        report = asyncio.run(run_pipeline(settings, arguments.command))
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
