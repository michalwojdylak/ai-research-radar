from datetime import UTC, datetime
from pathlib import Path

import pytest

from research_radar.analysis.analyzer import analysis_input_fingerprint
from research_radar.config import Settings
from research_radar.models.analysis import (
    AnalysisContent,
    AnalysisResult,
    CriticResult,
    PromptVersions,
    ResearchPaper,
    TimelineEvent,
)
from research_radar.models.paper import Paper
from research_radar.models.ranking import PaperRecord, Rankings
from research_radar.models.run import RunReport
from research_radar.pipeline.generate import (
    HUMAN_END,
    HUMAN_START,
    generate,
    render_markdown,
    to_research_paper,
    validate_generated,
)
from research_radar.storage.database import Database
from research_radar.storage.repository import Repository


@pytest.fixture
def output_record() -> PaperRecord:
    return PaperRecord(
        paper=Paper(
            id="2610.00001",
            arxiv_id="2610.00001",
            title="Evaluation for AI",
            abstract="The abstract describes evaluation.",
            authors=["Researcher"],
            published_at=datetime(2026, 10, 1, tzinfo=UTC),
            categories=["cs.AI"],
            abs_url="https://arxiv.org/abs/2610.00001",
        )
    )


def analysis_for(record: PaperRecord) -> AnalysisResult:
    fixture = Path(__file__).parent / "fixtures" / "analysis.json"
    return AnalysisResult(
        paper_id=record.paper.id,
        content=AnalysisContent.model_validate_json(fixture.read_text(encoding="utf-8")),
        critic=CriticResult(passed=True, issues=[]),
        status="review",
        full_text_available=False,
        confidence="reduced",
        generated_at=datetime(2026, 10, 5, tzinfo=UTC),
        model="test",
        critic_model="test",
        prompt_versions=PromptVersions(relevance="v1", analysis="v1", critic="v1"),
        cache_key="test",
        input_fingerprint=analysis_input_fingerprint(record),
    )


def test_markdown_contains_all_sections_and_no_take(output_record: PaperRecord) -> None:
    paper = to_research_paper(output_record, analysis_for(output_record), "evaluation-for-ai")
    markdown = render_markdown(paper)
    for heading in (
        "TL;DR",
        "Why should I care?",
        "Research timeline",
        "What problem does it solve?",
        "What is the core idea?",
        "What is actually new?",
        "How does it work?",
        "What did they compare against?",
        "Key results",
        "Does it actually beat previous work?",
        "Limitations",
        "Can I reproduce it?",
        "Why does it matter?",
        "What could happen next?",
        "Michał's take",
        "Resources",
        "Research metadata",
    ):
        assert f"## {heading}" in markdown
    assert markdown.split(HUMAN_START)[1].split(HUMAN_END)[0].strip() == ""
    assert paper.michaels_take is None
    assert "Unknown" in markdown


def test_timeline_title_is_resolved_to_supplied_url(output_record: PaperRecord) -> None:
    paper = to_research_paper(output_record, analysis_for(output_record), "evaluation-for-ai")
    paper.timeline = [
        TimelineEvent(
            year="2026",
            title=paper.title,
            description="The current paper.",
            related_paper=paper.title,
        )
    ]
    markdown = render_markdown(paper)
    assert "([source](https://arxiv.org/abs/2610.00001))" in markdown
    assert "([source](Evaluation for AI))" not in markdown


def test_generation_preserves_human_edit_and_stable_slug(
    tmp_path: Path, output_record: PaperRecord
) -> None:
    settings = Settings(data_dir=tmp_path / "data", _env_file=None)
    with Database(tmp_path / "state.sqlite3") as database:
        repository = Repository(database)
        repository.save_record(output_record)
        repository.save_analysis(analysis_for(output_record))
        repository.save_rankings(
            Rankings(timestamp=datetime.now(UTC), selected_ids=[output_record.paper.id])
        )
        report = RunReport(id="test", started_at=datetime.now(UTC), stage="generate")
        paths = generate(repository, settings, report)
        json_path, md_path = paths
        original = md_path.read_text(encoding="utf-8")
        md_path.write_text(
            original.replace(HUMAN_START, HUMAN_START + "\nMy own take."), encoding="utf-8"
        )
        output_record.paper.title = "Evaluation for AI: revised title"
        repository.save_record(output_record)
        repository.save_analysis(analysis_for(output_record))
        paths_again = generate(repository, settings, report)
        assert paths_again == paths
        assert "My own take." in md_path.read_text(encoding="utf-8")
        generated = ResearchPaper.model_validate_json(json_path.read_text(encoding="utf-8"))
        assert generated.michaels_take == "My own take."
        assert generated.title.endswith("revised title")
        assert validate_generated(settings.data_dir) == 1


def test_stale_analysis_is_not_exported(tmp_path: Path, output_record: PaperRecord) -> None:
    with Database(tmp_path / "state.sqlite3") as database:
        repository = Repository(database)
        repository.save_record(output_record)
        repository.save_analysis(analysis_for(output_record))
        repository.save_rankings(
            Rankings(timestamp=datetime.now(UTC), selected_ids=[output_record.paper.id])
        )
        output_record.paper.abstract = "Changed evidence"
        repository.save_record(output_record)
        report = RunReport(id="test", started_at=datetime.now(UTC), stage="generate")
        settings = Settings(data_dir=tmp_path / "data", _env_file=None)
        assert generate(repository, settings, report) == []
        assert len(report.failures) == 1


def test_approved_content_is_never_overwritten(tmp_path: Path, output_record: PaperRecord) -> None:
    settings = Settings(data_dir=tmp_path / "data", _env_file=None)
    with Database(tmp_path / "state.sqlite3") as database:
        repository = Repository(database)
        repository.save_record(output_record)
        repository.save_analysis(analysis_for(output_record))
        repository.save_rankings(
            Rankings(timestamp=datetime.now(UTC), selected_ids=[output_record.paper.id])
        )
        report = RunReport(id="test", started_at=datetime.now(UTC), stage="generate")
        json_path, md_path = generate(repository, settings, report)
        approved = ResearchPaper.model_validate_json(json_path.read_text(encoding="utf-8"))
        approved.status = "approved"
        json_path.write_text(approved.model_dump_json(indent=2), encoding="utf-8")
        json_before = json_path.read_bytes()
        md_before = md_path.read_bytes()
        output_record.paper.abstract = "Changed after human approval"
        repository.save_record(output_record)
        assert generate(repository, settings, report) == []
        assert json_path.read_bytes() == json_before
        assert md_path.read_bytes() == md_before


def test_editorial_conflict_preserves_both_files(
    tmp_path: Path, output_record: PaperRecord
) -> None:
    settings = Settings(data_dir=tmp_path / "data", _env_file=None)
    with Database(tmp_path / "state.sqlite3") as database:
        repository = Repository(database)
        repository.save_record(output_record)
        repository.save_analysis(analysis_for(output_record))
        repository.save_rankings(
            Rankings(timestamp=datetime.now(UTC), selected_ids=[output_record.paper.id])
        )
        report = RunReport(id="test", started_at=datetime.now(UTC), stage="generate")
        json_path, md_path = generate(repository, settings, report)
        paper = ResearchPaper.model_validate_json(json_path.read_text(encoding="utf-8"))
        paper.michaels_take = "JSON editorial perspective"
        json_path.write_text(paper.model_dump_json(), encoding="utf-8")
        md_path.write_text(
            md_path.read_text(encoding="utf-8").replace(
                HUMAN_START, HUMAN_START + "\nMarkdown editorial perspective"
            ),
            encoding="utf-8",
        )
        before = (json_path.read_bytes(), md_path.read_bytes())
        assert generate(repository, settings, report) == []
        assert (json_path.read_bytes(), md_path.read_bytes()) == before
        assert report.failures[-1].stage == "generate"
