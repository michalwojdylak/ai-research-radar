import json
import math
import posixpath
import re
from calendar import monthrange
from datetime import date
from pathlib import Path, PurePosixPath
from typing import cast
from urllib.parse import urlsplit

from research_radar.config import Settings
from research_radar.models.radars import (
    MonthlyRadar,
    PeriodReference,
    WeeklyRadar,
    YearlyRadar,
)
from research_radar.periods import iter_weeks, parse_week
from research_radar.pipeline.generate import HUMAN_END, HUMAN_START, extract_human_block
from research_radar.pipeline.rank import atomic_text
from research_radar.radars.scoring import radar_score

type Radar = WeeklyRadar | MonthlyRadar | YearlyRadar


def _relative_path(radar: Radar) -> Path:
    root = Path("radars")
    if radar.mode == "retrospective":
        root /= "retrospective"
    if isinstance(radar, WeeklyRadar):
        if radar.is_preview:
            root /= "previews"
        return root / "weeks" / f"{radar.year:04}" / f"{radar.id}.json"
    if isinstance(radar, MonthlyRadar):
        return root / "months" / f"{radar.year:04}" / f"{radar.id}.json"
    return root / "years" / f"{radar.id}.json"


def _human(markdown: str) -> str:
    if (
        markdown.count(HUMAN_START) != 1
        or markdown.count(HUMAN_END) != 1
        or markdown.index(HUMAN_START) >= markdown.index(HUMAN_END)
    ):
        raise ValueError("Ambiguous human editorial boundaries; refusing to overwrite")
    return extract_human_block(markdown)


def _number(value: float | None) -> str:
    return f"{value:.2f}/100" if value is not None else "Unknown"


def _child_link(radar: Radar, identifier: str, kind: str) -> str:
    parent = PurePosixPath(_relative_path(radar).as_posix()).parent
    root = PurePosixPath("radars")
    if radar.mode == "retrospective":
        root /= "retrospective"
    child = root / kind / identifier[:4] / f"{identifier}.md"
    return posixpath.relpath(str(child), str(parent))


def _reference_lines(radar: Radar, refs: list[PeriodReference], kind: str) -> list[str]:
    return [
        f"- [{ref.id}]({_child_link(radar, ref.id, kind)}) — {_number(ref.score)}, "
        f"{ref.paper_count} selected papers"
        for ref in refs
    ] or ["No child radars supplied; scores remain unknown."]


def render_markdown(radar: Radar) -> str:
    title = (
        "Weekly AI Research Radar"
        if isinstance(radar, WeeklyRadar)
        else "Monthly AI Research Radar"
        if isinstance(radar, MonthlyRadar)
        else "Yearly AI Research Radar"
    )
    heading = (
        f"AI Research Radar — Week {radar.week:02}, {radar.year}"
        if isinstance(radar, WeeklyRadar)
        else f"{title} — {radar.id}"
    )
    lines = [
        f"# {heading}",
        "",
        f"> Deterministic research selection · Status: **{radar.status}** · Mode: **{radar.mode}**",
        "> This draft requires human review. It is not Michał's personal opinion.",
        "",
        "## TL;DR",
        "",
        f"{radar.paper_count} selected papers. "
        f"{'Publication radar' if radar.mode == 'publication' else 'Retrospective'} "
        f"score: **{_number(radar.ranking_score)}**. "
        f"Period coverage: **{'complete' if radar.is_complete else 'incomplete'}**.",
        "",
        "## Why should I care?",
        "",
        radar.explanation,
        "",
    ]
    if isinstance(radar, WeeklyRadar):
        lines.extend(
            [
                "## Radar score",
                "",
                _number(radar.ranking_score),
                "",
                "## Why was this week important?",
                "",
                radar.explanation,
                "",
                "## Papers worth reading",
                "",
                f"Original publication week: {radar.start_date}–{radar.end_date}.",
                "",
            ]
        )
        for paper in radar.papers:
            title = f"[{paper.title}]({paper.paper_url})" if paper.paper_url else paper.title
            value = (
                paper.publication_score
                if radar.mode == "publication"
                else paper.retrospective_score
            )
            lines.extend(
                [
                    f"### {paper.rank}. {title}",
                    "",
                    f"{radar.mode.title()} score: {_number(value)}. "
                    f"Importance: {_number(paper.importance_score)}; "
                    f"trending: {_number(paper.trending_score)}; "
                    f"overall: {_number(paper.overall_score)}.",
                    "",
                    paper.selection_reason,
                    "",
                ]
            )
        if not radar.papers:
            lines.extend(
                [
                    "No qualifying papers selected."
                    if radar.ranking_score is not None
                    else "No papers selected; missing evidence or discovery coverage "
                    "remains unknown.",
                    "",
                ]
            )
        lines.extend(
            [
                "## Why these papers?",
                "",
                f"Only known {radar.mode} scores >= {radar.quality_threshold:g}/100 qualify. "
                "Scores are sorted descending, with stable paper IDs breaking ties. "
                "The soft minimum does not add weak or unscored candidates.",
                "",
                "## How strong was this week?",
                "",
                f"Radar score: {_number(radar.ranking_score)}. "
                "The peak and transformed quality slots reward exceptional work; "
                "the bounded mean tail does not reward paper count.",
                "",
                "## Week in context",
                "",
                f"This is the original publication period {radar.id} "
                f"({radar.start_date}–{radar.end_date}), not the date of later attention. "
                f"Ranking uses the {radar.mode} view with cutoff "
                f"{radar.data_cutoff.isoformat()}. "
                + (
                    "This is an incomplete preview, not a finalized Radar."
                    if radar.is_preview
                    else "Follow its Thursday-owned month and year for the wider hierarchy."
                ),
                "",
            ]
        )
    elif isinstance(radar, MonthlyRadar):
        lines.extend(
            [
                "## Best weeks",
                "",
                *_reference_lines(radar, radar.top_weeks, "weeks"),
                "",
                f"Best known week: {radar.best_week or 'Unknown'}. "
                f"Known weeks: {radar.known_week_count}; "
                f"unknown weeks: {radar.unknown_week_count}.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Best months",
                "",
                *_reference_lines(radar, radar.top_months, "months"),
                "",
                "## Best weeks",
                "",
                *_reference_lines(radar, radar.best_weeks, "weeks"),
                "",
                "## Papers worth reading",
                "",
                *[f"- Paper ID: `{paper_id}`" for paper_id in radar.top_papers],
                "",
                f"Best known month: {radar.best_month or 'Unknown'}. "
                f"Known months: {radar.known_month_count}; unknown months: "
                f"{radar.unknown_month_count}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Limitations",
            "",
            "Unknown observations remain unknown, not zero. Publication-document proxies "
            "are approximations, not contemporaneous observations. Scores and component "
            "reasons are selection signals, not independently verified claims or proof "
            "of superiority. Incomplete periods are provisional.",
            "",
            "## Michał's take",
            "",
            HUMAN_START,
            "",
            radar.michaels_take or "",
            "",
            HUMAN_END,
            "",
            "## Resources",
            "",
        ]
    )
    if isinstance(radar, WeeklyRadar):
        lines.extend(
            f"- [{paper.title}]({paper.paper_url})" for paper in radar.papers if paper.paper_url
        )
    else:
        refs = radar.top_weeks if isinstance(radar, MonthlyRadar) else radar.top_months
        lines.extend(
            _reference_lines(radar, refs, "weeks" if isinstance(radar, MonthlyRadar) else "months")
        )
    prompts = radar.prompt_versions
    lines.extend(
        [
            "",
            "## Radar metadata",
            "",
            f"- Radar ID: `{radar.id}`",
            f"- Type: `{radar.type}`",
            f"- Mode: `{radar.mode}`",
            f"- Generated at: {radar.generated_at.isoformat()}",
            f"- Data cutoff: {radar.data_cutoff.isoformat()}",
            f"- Ranking version: `{radar.ranking_version}`",
            f"- Scoring version: `{radar.scoring_version}`",
            f"- Prompt versions: relevance={prompts.relevance}, analysis={prompts.analysis}, "
            f"critic={prompts.critic}",
            f"- Complete: {str(radar.is_complete).lower()}",
            f"- Status: `{radar.status}`",
            f"- Selected paper count: {radar.paper_count}",
            f"- Radar scoring configuration: `{radar.scoring_configuration.model_dump_json()}`",
            f"- Paper scoring weights: `{radar.paper_scoring_weights.model_dump_json()}`",
            f"- Retrospective weights: `{radar.retrospective_weights.model_dump_json()}`",
        ]
    )
    if isinstance(radar, WeeklyRadar):
        lines.extend(
            [
                f"- ISO week: {radar.year}-W{radar.week:02}",
                f"- Publication dates: {radar.start_date}–{radar.end_date}",
                f"- Discovery coverage: {radar.coverage_start}–{radar.coverage_end}",
                f"- Discovery complete: {str(radar.discovery_complete).lower()}",
                f"- Preview: {str(radar.is_preview).lower()}",
                f"- Candidates: {radar.candidate_count}",
                f"- Scored candidates: {radar.scored_candidate_count}",
                f"- Quality threshold: {radar.quality_threshold:g}/100",
                f"- Below soft minimum: {str(radar.below_minimum).lower()}",
            ]
        )
    markdown = "\n".join(lines).rstrip() + "\n"
    _human(markdown)
    return markdown


def write_radar[T: Radar](radar: T, settings: Settings) -> tuple[T, list[Path]]:
    """Update exactly one period, preserving independently edited human content."""
    radar = cast(T, type(radar).model_validate(radar.model_dump()))
    json_path = settings.data_dir / _relative_path(radar)
    md_path = json_path.with_suffix(".md")
    previous: T | None = None
    if json_path.exists():
        previous_payload = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(previous_payload, dict) or previous_payload.get("type") != radar.type:
            raise ValueError("Existing radar type discriminator is missing or inconsistent")
        previous = cast(T, type(radar).model_validate(previous_payload))
        if (previous.id, previous.mode) != (radar.id, radar.mode):
            raise ValueError("Radar identity collision; refusing to overwrite")
        _validate_semantics(previous)
        if isinstance(previous, WeeklyRadar) and isinstance(radar, WeeklyRadar):
            if previous.is_preview != radar.is_preview:
                raise ValueError("Radar preview identity collision; refusing to overwrite")
    human = _human(md_path.read_text(encoding="utf-8")) if md_path.exists() else None
    take = previous.michaels_take if previous is not None else None
    if take and human and take.strip() != human.strip():
        raise ValueError("Conflicting human editorial content")
    if previous is not None and previous.status in {"approved", "published"}:
        return previous, []
    _validate_semantics(radar)
    result = radar.model_copy(deep=True)
    result.michaels_take = human or take or None
    markdown = render_markdown(result)
    atomic_text(json_path, result.model_dump_json(indent=2) + "\n")
    atomic_text(md_path, markdown)
    return cast(T, result), [json_path, md_path]


def _ordered_refs(refs: list[PeriodReference]) -> None:
    if len({ref.id for ref in refs}) != len(refs):
        raise ValueError("Duplicate child radar IDs")
    if refs != sorted(refs, key=lambda ref: (ref.score is None, -(ref.score or 0.0), ref.id)):
        raise ValueError("Child radar references must be deterministically score-ordered")


def _check_aggregate(
    radar: MonthlyRadar | YearlyRadar,
    refs: list[PeriodReference],
    known_count: int,
    unknown_count: int,
) -> None:
    _ordered_refs(refs)
    values = [ref.score for ref in refs if ref.score is not None]
    if (known_count, unknown_count) != (len(values), len(refs) - len(values)):
        raise ValueError("Known and unknown child counts must agree with references")
    configuration = radar.scoring_configuration
    coefficients = (
        configuration.monthly_coefficients
        if isinstance(radar, MonthlyRadar)
        else configuration.yearly_coefficients
    )
    expected = radar_score(values, coefficients, configuration) if values else None
    if expected is None:
        if radar.ranking_score is not None:
            raise ValueError("Unknown child scores must not invent an aggregate")
    elif radar.ranking_score is None or not math.isclose(radar.ranking_score, expected):
        raise ValueError("Aggregate score disagrees with scoring configuration and children")


def _validate_semantics(radar: Radar) -> None:
    configuration = radar.scoring_configuration
    for coefficients in (
        configuration.weekly_coefficients,
        configuration.monthly_coefficients,
        configuration.yearly_coefficients,
    ):
        radar_score([], coefficients, configuration)
    if radar.data_cutoff > radar.generated_at:
        raise ValueError("Data cutoff cannot follow generation time")
    if isinstance(radar, WeeklyRadar):
        if not radar.paper_count <= radar.scored_candidate_count <= radar.candidate_count:
            raise ValueError("Weekly candidate counts are inconsistent")
        if radar.is_preview and radar.status != "draft":
            raise ValueError("Previews must be draft and incomplete")
        period = parse_week(radar.id)
        if radar.is_complete and (
            not radar.discovery_complete
            or (radar.coverage_start, radar.coverage_end) != (radar.start_date, radar.end_date)
            or radar.generated_at < period.end
            or radar.data_cutoff < period.end
        ):
            raise ValueError("Complete weeks require full, finished discovery coverage")
        selected: list[tuple[float, str]] = []
        for paper in radar.papers:
            value = (
                paper.publication_score
                if radar.mode == "publication"
                else paper.retrospective_score
            )
            if value is None or value < radar.quality_threshold:
                raise ValueError("Selected papers require a known qualifying mode-specific score")
            selected.append((value, paper.paper_id))
            if paper.paper_url is not None:
                parsed = urlsplit(paper.paper_url)
                if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
                    raise ValueError("Paper URLs must be primary HTTP(S) source links")
        if selected != sorted(selected, key=lambda item: (-item[0], item[1])):
            raise ValueError("Paper selections must be deterministically score-ordered")
        if radar.status in {"review", "approved", "published"} and any(
            paper.analysis_status not in {"review", "approved", "published"}
            for paper in radar.papers
        ):
            raise ValueError("Reviewed radars require reviewed selected analyses")
        if radar.status in {"review", "approved", "published"} and (
            not radar.is_complete or radar.ranking_score is None
        ):
            raise ValueError("Reviewed radars require completed known period coverage")
        if radar.ranking_score is not None:
            if not selected and (
                not radar.is_complete or radar.scored_candidate_count < radar.candidate_count
            ):
                raise ValueError("Unknown or incomplete discovery must not certify an empty week")
            expected = radar_score(
                [value for value, _ in selected],
                radar.scoring_configuration.weekly_coefficients,
                radar.scoring_configuration,
            )
            if not math.isclose(radar.ranking_score, expected, abs_tol=1e-10):
                raise ValueError("Weekly score disagrees with selected scores and configuration")
        elif selected:
            raise ValueError("Known selected papers require a mode-specific radar score")
        elif radar.is_complete and radar.scored_candidate_count == radar.candidate_count:
            raise ValueError("A completed, fully scored empty selection requires a zero score")
    elif isinstance(radar, MonthlyRadar):
        _check_aggregate(radar, radar.top_weeks, radar.known_week_count, radar.unknown_week_count)
        for ref in radar.top_weeks:
            if parse_week(ref.id).month_key != (radar.year, radar.month):
                raise ValueError("Monthly child weeks require Thursday month ownership")
        best = next((ref.id for ref in radar.top_weeks if ref.score is not None), None)
        if radar.best_week != best or radar.paper_count != sum(
            ref.paper_count for ref in radar.top_weeks
        ):
            raise ValueError("Monthly highlights or paper count disagree with child references")
        if radar.is_complete:
            first = date(radar.year, radar.month, 1)
            last = date(radar.year, radar.month, monthrange(radar.year, radar.month)[1])
            required = [
                period
                for period in iter_weeks(first, last)
                if period.month_key == (radar.year, radar.month)
            ]
            if {ref.id for ref in radar.top_weeks} != {period.id for period in required} or (
                radar.generated_at < max(period.end for period in required)
                or radar.generated_at.date() <= last
            ):
                raise ValueError("Complete months require every finished Thursday-owned week")
    else:
        _check_aggregate(
            radar, radar.top_months, radar.known_month_count, radar.unknown_month_count
        )
        _ordered_refs(radar.best_weeks)
        for ref in radar.top_months:
            if not re.fullmatch(rf"{radar.year:04}-(0[1-9]|1[0-2])", ref.id):
                raise ValueError("Yearly child months must belong to the requested year")
        for ref in radar.best_weeks:
            if parse_week(ref.id).month_key[0] != radar.year:
                raise ValueError("Yearly child weeks require Thursday year ownership")
        best = next((ref.id for ref in radar.top_months if ref.score is not None), None)
        if radar.best_month != best or len(set(radar.top_papers)) != len(radar.top_papers):
            raise ValueError("Yearly highlights require consistent months and distinct paper IDs")
        if radar.paper_count != sum(ref.paper_count for ref in radar.top_months):
            raise ValueError("Yearly paper count must agree with child month references")
        if radar.is_complete:
            required_weeks = [
                period
                for period in iter_weeks(date(radar.year, 1, 1), date(radar.year, 12, 31))
                if period.month_key[0] == radar.year
            ]
            if (
                {ref.id for ref in radar.top_months}
                != {f"{radar.year:04}-{month:02}" for month in range(1, 13)}
                or {ref.id for ref in radar.best_weeks} != {period.id for period in required_weeks}
                or (
                    radar.generated_at < max(period.end for period in required_weeks)
                    or radar.generated_at.date() <= date(radar.year, 12, 31)
                )
            ):
                raise ValueError("Complete years require all finished child periods")


def validate_radar_files(data_dir: Path) -> int:
    """Validate all canonical, retrospective and preview radar JSON/Markdown pairs."""
    count = 0
    root = data_dir / "radars"
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        models: dict[str, type[WeeklyRadar] | type[MonthlyRadar] | type[YearlyRadar]] = {
            "weekly_radar": WeeklyRadar,
            "monthly_radar": MonthlyRadar,
            "yearly_radar": YearlyRadar,
        }
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("type"), str)
            or payload["type"] not in models
        ):
            raise ValueError(f"Unknown radar document type: {path}")
        radar = models[payload["type"]].model_validate(payload)
        _validate_semantics(radar)
        if path != data_dir / _relative_path(radar):
            raise ValueError(f"Radar path disagrees with ID, type, mode or preview: {path}")
        markdown_path = path.with_suffix(".md")
        if not markdown_path.exists():
            raise ValueError(f"Radar Markdown is missing: {markdown_path}")
        markdown = markdown_path.read_text(encoding="utf-8")
        human = _human(markdown)
        if human and radar.michaels_take and human != radar.michaels_take.strip():
            raise ValueError(f"Conflicting human editorial content: {path}")
        if f"- Radar ID: `{radar.id}`" not in markdown or f"- Mode: `{radar.mode}`" not in markdown:
            raise ValueError(f"Radar Markdown identity or mode disagrees with JSON: {path}")
        count += 1
    for markdown_path in root.rglob("*.md"):
        if not markdown_path.with_suffix(".json").exists():
            raise ValueError(f"Radar JSON is missing: {markdown_path}")
    return count
