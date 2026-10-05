import hashlib
import logging
import re
import unicodedata
from pathlib import Path

from research_radar.analysis.analyzer import analysis_input_fingerprint
from research_radar.config import Settings
from research_radar.models.analysis import AnalysisResult, ResearchPaper
from research_radar.models.paper import PaperLinks, SourceReference
from research_radar.models.ranking import PaperRecord, Rankings
from research_radar.models.run import RunReport
from research_radar.pipeline.common import PAPER_ERRORS, report_failure
from research_radar.pipeline.rank import atomic_text
from research_radar.storage.repository import Repository

logger = logging.getLogger(__name__)
HUMAN_START = "<!-- HUMAN EDITORIAL SECTION -->"
HUMAN_END = "<!-- END HUMAN EDITORIAL SECTION -->"
SECTIONS = [
    ("TL;DR", "tldr"),
    ("Why should I care?", "why_should_i_care"),
    ("What problem does it solve?", "problem"),
    ("What is the core idea?", "core_idea"),
    ("What is actually new?", "whats_new"),
    ("How does it work?", "how_it_works"),
    ("What did they compare against?", "comparisons"),
    ("Key results", "key_results"),
    ("Does it actually beat previous work?", "beats_previous_work"),
    ("Why does it matter?", "why_it_matters"),
    ("What could happen next?", "what_next"),
]


def slug_for(record: PaperRecord) -> str:
    ascii_title = (
        unicodedata.normalize("NFKD", record.paper.title).encode("ascii", "ignore").decode()
    )
    title = re.sub(r"[^a-z0-9]+", "-", ascii_title.casefold()).strip("-")[:100].rstrip("-")
    suffix = hashlib.sha256(record.paper.id.encode()).hexdigest()[:8]
    return f"{title or 'paper'}-{suffix}"


def to_research_paper(record: PaperRecord, analysis: AnalysisResult, slug: str) -> ResearchPaper:
    paper = record.paper
    enrichment = record.enrichment
    return ResearchPaper(
        **analysis.content.model_dump(),
        id=paper.id,
        slug=slug,
        title=paper.title,
        authors=paper.authors,
        published_at=paper.published_at,
        venue=enrichment.venue,
        categories=paper.categories,
        abstract=paper.abstract,
        links=PaperLinks(
            arxiv=paper.abs_url,
            pdf=paper.pdf_url,
            doi=f"https://doi.org/{paper.doi}" if paper.doi else None,
            github=enrichment.github_url,
            huggingface=enrichment.hf_paper_url,
        ),
        metrics=record.metrics,
        scores=record.scores,
        michaels_take=None,
        status=analysis.status,
        full_text_available=analysis.full_text_available,
        input_truncated=analysis.input_truncated,
        analysis_confidence=analysis.confidence,
        critic=analysis.critic,
        generated_at=analysis.generated_at,
        analysis_model=analysis.model,
        critic_model=analysis.critic_model,
        prompt_versions=analysis.prompt_versions,
    )


def render_markdown(paper: ResearchPaper, *, human_block: str | None = None) -> str:
    lines = [
        f"# {paper.title}",
        "",
        f"{', '.join(paper.authors)} · {paper.published_at:%Y-%m-%d} · {paper.venue or 'Unknown'}",
        "",
        f"> AI-assisted research analysis · Status: **{paper.status}** · "
        f"Evidence confidence: **{paper.analysis_confidence}**",
        "> This draft requires human review. It is not Michał's personal opinion.",
        "",
    ]
    evidence: dict[str, list[SourceReference]] = {
        item.section: item.sources for item in paper.section_sources
    }

    def section(heading: str, key: str, content: str) -> None:
        lines.extend([f"## {heading}", "", content, ""])
        if sources := evidence.get(key):
            lines.extend(
                [
                    "Sources: "
                    + " · ".join(f"[{source.title}]({source.url})" for source in sources),
                    "",
                ]
            )

    for heading, key in SECTIONS[:2]:
        section(heading, key, str(getattr(paper, key)))
    timeline_sources = paper.sources + [
        source for section in paper.section_sources for source in section.sources
    ]
    timeline_urls = {source.title: source.url for source in timeline_sources}
    if paper.links.arxiv:
        timeline_urls[paper.title] = paper.links.arxiv
    known_urls = set(timeline_urls.values())
    timeline_lines: list[str] = []
    for event in paper.timeline:
        reference = event.related_paper
        url = reference if reference in known_urls else timeline_urls.get(reference or "")
        timeline_lines.append(
            f"- **{event.year} — {event.title}:** {event.description}"
            + (f" ([source]({url}))" if url else "")
        )
    timeline = "\n".join(timeline_lines)
    section("Research timeline", "timeline", timeline or "Unknown")
    for heading, key in SECTIONS[2:9]:
        section(heading, key, str(getattr(paper, key)))
    section(
        "Limitations",
        "limitations",
        f"{paper.limitations}\n\n### Authors' limitations\n\n{paper.authors_limitations}"
        f"\n\n### Analysis-derived limitations\n\n{paper.analysis_limitations}",
    )
    section(
        "Can I reproduce it?",
        "reproducibility",
        f"**{paper.reproducibility}**\n\n{paper.reproducibility_reason}",
    )
    for heading, key in SECTIONS[9:]:
        section(heading, key, str(getattr(paper, key)))
    lines.extend(
        [
            "## Michał's take",
            "",
            HUMAN_START,
            "",
            human_block if human_block is not None else paper.michaels_take or "",
            "",
            HUMAN_END,
            "",
        ]
    )
    if paper.questions_for_michael:
        lines.extend(["### Questions for Michał", ""])
        lines.extend(f"- {question}" for question in paper.questions_for_michael)
        lines.append("")
    lines.extend(["## Resources", ""])
    resources = list(paper.sources)
    lines.extend(f"- [{source.title}]({source.url}) ({source.source_type})" for source in resources)
    for name, url in paper.links.model_dump().items():
        if url and all(source.url != url for source in resources):
            lines.append(f"- [{name}]({url})")
    lines.extend(
        [
            "",
            "## Research metadata",
            "",
            f"- Paper ID: `{paper.id}`",
            f"- Categories: {', '.join(paper.categories)}",
            f"- Full text available: {str(paper.full_text_available).lower()}",
            f"- Input truncated: {str(paper.input_truncated).lower()}",
            f"- Analysis model: `{paper.analysis_model}`",
            f"- Critic model: `{paper.critic_model}`",
            f"- Prompt versions: relevance={paper.prompt_versions.relevance}, "
            f"analysis={paper.prompt_versions.analysis}, critic={paper.prompt_versions.critic}",
        ]
    )
    for score_name in ("importance", "trending", "overall"):
        value = getattr(paper.scores, score_name)
        breakdown = getattr(paper.scores, f"{score_name}_breakdown")
        lines.append(
            f"- {score_name.title()}: {f'{value:.2f}' if value is not None else 'Unknown'} "
            f"(evidence coverage: {breakdown.coverage:.0%})"
        )
    lines.extend(["", "### Critic review", "", f"Passed: {paper.critic.passed}", ""])
    lines.extend(
        f"- **{issue.severity}:** {issue.issue} Correction: {issue.correction}"
        for issue in paper.critic.issues
    )
    markdown = "\n".join(lines).rstrip() + "\n"
    if markdown.count(HUMAN_START) != 1 or markdown.count(HUMAN_END) != 1:
        raise ValueError("Ambiguous human editorial boundaries; refusing to generate")
    return markdown


def _human_block(path: Path) -> str | None:
    if not path.exists():
        return None
    return extract_human_block(path.read_text(encoding="utf-8"))


def extract_human_block(markdown: str) -> str:
    if markdown.count(HUMAN_START) == 1 and markdown.count(HUMAN_END) == 1:
        return markdown.split(HUMAN_START, 1)[1].split(HUMAN_END, 1)[0].strip()
    if HUMAN_END in markdown or markdown.count(HUMAN_START) > 1:
        raise ValueError("Ambiguous human editorial boundaries; refusing to overwrite")
    # Preserve editorial text in older drafts that predate the explicit end marker.
    if "## Michał's take" in markdown:
        return (
            markdown.split("## Michał's take", 1)[1]
            .split("\n## ", 1)[0]
            .replace(HUMAN_START, "")
            .strip()
        )
    raise ValueError("Existing Markdown has no editorial boundary; refusing to overwrite")


def generate(repository: Repository, settings: Settings, report: RunReport) -> list[Path]:
    rankings = repository.latest_rankings()
    if rankings is None:
        logger.warning("No rankings available; run rank before generate.")
        return []
    existing_by_id: dict[str, tuple[Path, ResearchPaper]] = {}
    for existing_path in (settings.data_dir / "papers").rglob("*.json"):
        previous = ResearchPaper.model_validate_json(existing_path.read_text(encoding="utf-8"))
        if previous.id in existing_by_id:
            raise ValueError("Duplicate canonical paper IDs in generated JSON")
        existing_by_id[previous.id] = (existing_path, previous)
    paths: list[Path] = []
    for paper_id in rankings.selected_ids[: settings.top_papers]:
        record = repository.get_record(paper_id)
        analysis = repository.latest_analysis(paper_id)
        if record is None or analysis is None:
            logger.warning("No analysis for selected paper %s; no prose generated.", paper_id)
            continue
        try:
            existing = existing_by_id.get(paper_id)
            if existing and existing[1].status in {"approved", "published"}:
                logger.info("Preserving human-approved content: %s", paper_id)
                continue
            if analysis.input_fingerprint != analysis_input_fingerprint(record):
                logger.warning(
                    "Analysis evidence changed for %s; run analyze before generate.", paper_id
                )
                raise ValueError("Analysis evidence has changed; run analyze before generate")
            slug = existing[1].slug if existing else slug_for(record)
            paper = to_research_paper(record, analysis, slug)
            json_path = (
                existing[0]
                if existing
                else (settings.data_dir / "papers" / f"{paper.published_at:%Y/%m}" / f"{slug}.json")
            )
            editorial_date = existing[1].published_at if existing else paper.published_at
            md_path = (
                settings.data_dir / "articles" / "drafts" / (f"{editorial_date:%Y-%m-%d}-{slug}.md")
            )
            if existing:
                previous = existing[1]
                if previous.id != paper.id:
                    raise ValueError("Content identity collision; refusing to overwrite")
                paper.michaels_take = previous.michaels_take
            human = _human_block(md_path)
            if human:
                if paper.michaels_take and paper.michaels_take.strip() != human:
                    logger.warning(
                        "Editorial JSON and Markdown disagree for %s; preserving both files.",
                        paper.id,
                    )
                    raise ValueError("Conflicting human editorial content")
                paper.michaels_take = human
            markdown = render_markdown(paper, human_block=human or None)
            atomic_text(json_path, paper.model_dump_json(indent=2) + "\n")
            atomic_text(md_path, markdown)
            paths.extend([json_path, md_path])
            report.generated += 1
        except PAPER_ERRORS as error:
            report_failure(report, "generate", error, paper_id)
    return paths


def validate_generated(data_dir: Path) -> int:
    count = 0
    for path in sorted((data_dir / "papers").rglob("*.json")):
        ResearchPaper.model_validate_json(path.read_text(encoding="utf-8"))
        count += 1
    for path in sorted((data_dir / "rankings").glob("*.json")):
        Rankings.model_validate_json(path.read_text(encoding="utf-8"))
        count += 1
    return count
