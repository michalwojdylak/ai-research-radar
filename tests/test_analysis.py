import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx
import pymupdf
import pytest
from pydantic import BaseModel, SecretStr

from research_radar.analysis import AnalysisService, LLMResponseError, OpenAIProvider
from research_radar.analysis.analyzer import analysis_input_fingerprint
from research_radar.analysis.critic import source_issues
from research_radar.analysis.pdf import FullTextLoader, PaperText, trusted_pdf_url
from research_radar.config import Settings
from research_radar.models.analysis import AnalysisContent, CriticIssue, CriticResult
from research_radar.models.paper import Paper, SourceReference
from research_radar.models.ranking import PaperRecord
from research_radar.prompts import load_prompt


class QueueProvider:
    def __init__(self, *responses: object) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    async def generate_structured[T: BaseModel](
        self, *, system_prompt: str, user_prompt: str, schema: type[T]
    ) -> T:
        index = len(self.calls)
        self.calls.append((system_prompt, user_prompt))
        value = self.responses[min(index, len(self.responses) - 1)]
        if isinstance(value, Exception):
            raise value
        return cast(T, value)


def record() -> PaperRecord:
    return PaperRecord(
        paper=Paper(
            id="2601.00001",
            title="Neural representations",
            abstract="We introduce a representation learning method.",
            authors=["Ada Researcher"],
            published_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 2, tzinfo=UTC),
            categories=["cs.LG"],
            abs_url="https://arxiv.org/abs/2601.00001",
        )
    )


def configuration(path: Path) -> Settings:
    return Settings(cache_dir=path / "cache", _env_file=None)


def content(tldr: str = "Fact (paper reports): a representation method.") -> AnalysisContent:
    source = SourceReference(
        title="Neural representations",
        url="https://arxiv.org/abs/2601.00001",
        source_type="paper",
    )
    return AnalysisContent(
        tldr=tldr,
        why_should_i_care="Interpretation: potentially useful.",
        timeline=[],
        problem="Fact (paper reports): representation learning.",
        core_idea="Fact (paper reports): a new method.",
        whats_new="Unknown beyond the abstract.",
        how_it_works="Unknown without full text.",
        comparisons="Unknown.",
        key_results="Unknown numerical results.",
        beats_previous_work="Unknown.",
        limitations="Unknown: full text unavailable.",
        authors_limitations="Unknown.",
        analysis_limitations="Only the abstract is available.",
        reproducibility="Unknown",
        reproducibility_reason="No implementation evidence.",
        why_it_matters="Interpretation: potentially useful representations.",
        what_next="Speculation: evaluate reproducibility.",
        sources=[source],
        section_sources=[{"section": "tldr", "sources": [source]}],
    )


def failing_critic() -> CriticResult:
    return CriticResult(
        passed=False,
        issues=[
            CriticIssue(
                severity="high", issue="Unsupported claim", correction="Replace claim with Unknown."
            )
        ],
    )


def test_inline_links_cannot_bypass_source_validation() -> None:
    draft = content()
    draft.key_results = "Claim [source](https://example.com/invented)."
    issues = source_issues(draft, record(), {"https://arxiv.org/abs/2601.00001"})
    assert any(issue.severity == "high" and "example.com" in issue.issue for issue in issues)


def test_inline_primary_link_with_markdown_punctuation_is_supported() -> None:
    draft = content()
    draft.key_results = "Unknown; see [paper](https://arxiv.org/abs/2601.00001)."
    assert source_issues(draft, record(), {"https://arxiv.org/abs/2601.00001"}) == []


async def test_abstract_fallback_review_and_no_human_opinion(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    analyst = QueueProvider(content())
    critic = QueueProvider(CriticResult(passed=True, issues=[]))
    service = AnalysisService(configuration(tmp_path), analyst, critic)
    try:
        result = await service.analyze(record())
    finally:
        await service.aclose()
    assert result.status == "review"
    assert result.confidence == "reduced"
    assert not result.full_text_available
    assert "michaels_take" not in result.content.model_dump()
    assert result.input_fingerprint == analysis_input_fingerprint(record())
    assert "using abstract" in caplog.text
    evidence = json.loads(analyst.calls[0][1])["evidence"]
    assert json.loads(critic.calls[0][1])["evidence"] == evidence
    assert evidence["primary_paper"]["authors"] == ["Ada Researcher"]


async def test_critic_high_retry_then_pass(tmp_path: Path) -> None:
    analyst = QueueProvider(content(), content("Unknown until further evidence."))
    critic = QueueProvider(failing_critic(), CriticResult(passed=True, issues=[]))
    service = AnalysisService(configuration(tmp_path), analyst, critic)
    try:
        result = await service.analyze(record())
    finally:
        await service.aclose()
    assert result.status == "review"
    assert len(analyst.calls) == len(critic.calls) == 2
    correction = json.loads(analyst.calls[1][1])["corrections"][0]
    assert correction["severity"] == "high"
    assert correction["correction"] == "Replace claim with Unknown."


async def test_critic_maximum_three_attempts_and_cache(tmp_path: Path) -> None:
    analyst = QueueProvider(content())
    critic = QueueProvider(failing_critic())
    service = AnalysisService(configuration(tmp_path), analyst, critic)
    try:
        result = await service.analyze(record())
        again = await service.analyze(record())
    finally:
        await service.aclose()
    assert result.status == "needs_review"
    assert len(analyst.calls) == len(critic.calls) == 3
    assert again == result


@pytest.mark.parametrize("location", ["sources", "section_sources", "timeline"])
async def test_unknown_sources_override_critic_pass(tmp_path: Path, location: str) -> None:
    draft = content().model_dump()
    foreign = {"title": "Invented", "url": "https://invented.example/paper", "source_type": "paper"}
    if location == "sources":
        draft["sources"] = [foreign]
    elif location == "section_sources":
        draft["section_sources"] = [{"section": "key_results", "sources": [foreign]}]
    else:
        draft["timeline"] = [
            {
                "year": "2020",
                "title": "Invented",
                "description": "Unknown",
                "related_paper": foreign["url"],
            }
        ]
    service = AnalysisService(
        configuration(tmp_path),
        QueueProvider(AnalysisContent.model_validate(draft)),
        QueueProvider(CriticResult(passed=True, issues=[])),
    )
    try:
        result = await service.analyze(record())
    finally:
        await service.aclose()
    assert result.status == "needs_review"
    assert not result.critic.passed
    assert any(issue.severity == "high" for issue in result.critic.issues)


@pytest.mark.parametrize(
    "bad",
    [
        {
            "passed": True,
            "issues": [{"severity": "high", "issue": "Contradiction", "correction": "Fix"}],
        },
        {"passed": True},
        RuntimeError("Critic unavailable"),
    ],
)
async def test_critic_malformed_and_errors_never_pass(tmp_path: Path, bad: object) -> None:
    service = AnalysisService(configuration(tmp_path), QueueProvider(content()), QueueProvider(bad))
    try:
        with pytest.raises((LLMResponseError, RuntimeError)):
            await service.analyze(record())
    finally:
        await service.aclose()


@pytest.mark.parametrize(
    "bad", [{"tldr": "Incomplete"}, {**content().model_dump(), "michaels_take": "I endorse this"}]
)
async def test_invalid_analysis_is_not_prose_parsed(tmp_path: Path, bad: object) -> None:
    service = AnalysisService(
        configuration(tmp_path),
        QueueProvider(bad),
        QueueProvider(CriticResult(passed=True, issues=[])),
    )
    try:
        with pytest.raises(LLMResponseError):
            await service.analyze(record())
    finally:
        await service.aclose()


async def test_analysis_cache_prompt_model_content_invalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = configuration(tmp_path)
    analyst = QueueProvider(content())
    critic = QueueProvider(CriticResult(passed=True, issues=[]))
    service = AnalysisService(settings, analyst, critic)
    try:
        first = await service.analyze(record())
        assert await service.analyze(record()) == first
        assert len(analyst.calls) == len(critic.calls) == 1
        changed = record()
        changed.paper.abstract = "Changed primary abstract."
        assert (await service.analyze(changed)).cache_key != first.cache_key
        settings.analysis_prompt_version = "v2"
        await service.analyze(record())
        settings.analysis_model = "new-analysis-model"
        await service.analyze(record())
        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "analysis.md").write_text("Changed analysis prompt.", encoding="utf-8")
        await service.analyze(record())
        assert len(analyst.calls) == 5
        settings.critic_prompt_version = "v2"
        await service.analyze(record())
        assert len(critic.calls) >= 3
    finally:
        await service.aclose()


async def test_observation_timestamp_does_not_regenerate_analysis(tmp_path: Path) -> None:
    analyst = QueueProvider(content())
    critic = QueueProvider(CriticResult(passed=True, issues=[]))
    service = AnalysisService(configuration(tmp_path), analyst, critic)
    item = record()
    item.metrics.observed_at = datetime(2026, 1, 3, tzinfo=UTC)
    try:
        first = await service.analyze(item)
        item.metrics.observed_at = datetime(2026, 1, 4, tzinfo=UTC)
        second = await service.analyze(item)
    finally:
        await service.aclose()
    assert first == second
    assert len(analyst.calls) == len(critic.calls) == 1
    assert first.input_fingerprint == analysis_input_fingerprint(item)


def test_fingerprint_tracks_evidence_and_paper_revision_not_ranking() -> None:
    item = record()
    original = analysis_input_fingerprint(item)
    item.scores.overall = 95
    item.metrics.observed_at = datetime(2026, 1, 3, tzinfo=UTC)
    assert analysis_input_fingerprint(item) == original
    item.paper.updated_at = datetime(2026, 1, 4, tzinfo=UTC)
    assert analysis_input_fingerprint(item) != original
    item.paper.updated_at = record().paper.updated_at
    item.metrics.citations = 10
    assert analysis_input_fingerprint(item) != original
    item.metrics.citations = None
    item.paper.abstract = "Revised primary abstract."
    assert analysis_input_fingerprint(item) != original


def pdf_bytes() -> bytes:
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Primary evidence: neural representations are evaluated.")
        return document.tobytes()


async def test_pdf_download_text_cache_and_updated_identity(tmp_path: Path) -> None:
    calls: list[str] = []
    data = pdf_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=data, headers={"content-type": "application/pdf"})

    paper = record().paper
    paper.pdf_url = "https://arxiv.org/pdf/2601.00001"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        loader = FullTextLoader(configuration(tmp_path), client)
        first = await loader.load(paper)
        assert first.full_text_available and "Primary evidence" in first.text
        assert await loader.load(paper) == first
        assert len(calls) == 1
        paper.updated_at = datetime(2026, 1, 3, tzinfo=UTC)
        await loader.load(paper)
        assert len(calls) == 2
        paper.pdf_url += "v2"
        await loader.load(paper)
        assert len(calls) == 3


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"not a pdf"),
        httpx.Response(404),
        httpx.Response(200, content=b"%PDF-" + b"x" * 2000),
        httpx.Response(302, headers={"location": "http://127.0.0.1/private.pdf"}),
        httpx.Response(200, headers={"content-length": "2000000"}, content=b"%PDF-"),
    ],
)
async def test_pdf_expected_failure_reduced_confidence(
    tmp_path: Path, response: httpx.Response, caplog: pytest.LogCaptureFixture
) -> None:
    settings = configuration(tmp_path)
    settings.max_pdf_bytes = 1000
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return response

    item = record()
    item.paper.pdf_url = "https://arxiv.org/pdf/2601.00001"
    service = AnalysisService(
        settings, QueueProvider(content()), QueueProvider(CriticResult(passed=True, issues=[]))
    )
    await service.full_text.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service.full_text = FullTextLoader(settings, client)
        result = await service.analyze(item)
    await service.aclose()
    assert not result.full_text_available
    assert result.confidence == "reduced"
    assert "reduced confidence" in caplog.text
    assert len(calls) == 1


async def test_pdf_safe_redirect_and_fulltext_standard(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.host == "export.arxiv.org":
            return httpx.Response(302, headers={"location": "https://arxiv.org/pdf/2601.00001"})
        return httpx.Response(200, content=pdf_bytes())

    item = record()
    item.paper.pdf_url = "https://export.arxiv.org/pdf/2601.00001"
    analyst = QueueProvider(content())
    service = AnalysisService(
        configuration(tmp_path), analyst, QueueProvider(CriticResult(passed=True, issues=[]))
    )
    await service.full_text.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service.full_text = FullTextLoader(configuration(tmp_path), client)
        result = await service.analyze(item)
    await service.aclose()
    assert result.full_text_available and result.confidence == "standard"
    assert len(calls) == 2
    assert "Primary evidence" in analyst.calls[0][1]


@pytest.mark.parametrize(
    "url",
    [
        "https://arxiv.org.evil.example/pdf/1",
        "https://127.0.0.1/pdf/1",
        "https://user@arxiv.org/pdf/1",
        "https://arxiv.org:8000/pdf/1",
        "https://arxiv.org/abs/1",
        "https://arxiv.org/pdf/%2e%2e/private",
    ],
)
def test_untrusted_pdf_urls(url: str) -> None:
    with pytest.raises(ValueError):
        trusted_pdf_url(url)


async def test_truncated_text_reduces_confidence(tmp_path: Path) -> None:
    settings = configuration(tmp_path)
    settings.max_analysis_characters = 1500
    service = AnalysisService(
        settings, QueueProvider(content()), QueueProvider(CriticResult(passed=True, issues=[]))
    )
    item = record()
    evidence, available, truncated = service._evidence(
        item, PaperText(text="a" * 10000, full_text_available=True), []
    )
    await service.aclose()
    assert len(evidence) <= settings.max_analysis_characters
    assert available and truncated


def test_packaged_and_editable_prompts_stay_equal() -> None:
    package = Path(__file__).parents[1] / "src" / "research_radar" / "prompts"
    root = Path(__file__).parents[1] / "prompts"
    for name in ("relevance", "analysis", "critic"):
        assert (package / f"{name}.md").read_text() == (root / f"{name}.md").read_text()
    assert "version: custom-version" in load_prompt("analysis", "custom-version")


@pytest.mark.parametrize("kind", ["refusal", "unparsed", "incomplete", "empty", "malformed"])
async def test_openai_validates_structured_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    message = SimpleNamespace(
        refusal="No" if kind == "refusal" else None,
        parsed=None if kind == "unparsed" else content(),
    )
    if kind == "malformed":
        message.parsed = AnalysisContent.model_construct(tldr="Incomplete")
    choice = SimpleNamespace(
        message=message, finish_reason="length" if kind == "incomplete" else "stop"
    )

    async def parse(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(choices=[] if kind == "empty" else [choice])

    async def close() -> None:
        return None

    fake_client = SimpleNamespace(
        beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))),
        close=close,
    )
    monkeypatch.setattr("research_radar.analysis.llm.AsyncOpenAI", lambda **_: fake_client)
    settings = configuration(tmp_path)
    settings.openai_api_key = SecretStr("test")
    provider = OpenAIProvider(settings, "mock")
    with pytest.raises(LLMResponseError):
        await provider.generate_structured(
            system_prompt="System v1", user_prompt="Paper", schema=AnalysisContent
        )
    await provider.aclose()


async def test_openai_provider_direct_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []

    async def parse(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(refusal=None, parsed=content()), finish_reason="stop"
                )
            ]
        )

    async def close() -> None:
        return None

    fake_client = SimpleNamespace(
        beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))),
        close=close,
    )
    monkeypatch.setattr("research_radar.analysis.llm.AsyncOpenAI", lambda **_: fake_client)
    settings = configuration(tmp_path)
    settings.openai_api_key = SecretStr("test")
    provider = OpenAIProvider(settings, "mock")
    for _ in range(2):
        await provider.generate_structured(
            system_prompt="System v1", user_prompt="Paper 2601.00001", schema=AnalysisContent
        )
    assert len(calls) == 1
    await provider.generate_structured(
        system_prompt="System v2", user_prompt="Paper 2601.00001", schema=AnalysisContent
    )
    assert len(calls) == 2
    await provider.aclose()
