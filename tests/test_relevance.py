import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel

from research_radar.analysis.llm import LLMResponseError
from research_radar.config import Settings
from research_radar.models.paper import Paper
from research_radar.models.ranking import QualityDimensions, RelevanceResult, ScoreDimension
from research_radar.ranking.relevance import RelevanceFilter


class RelevanceProvider:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[str, str]] = []

    async def generate_structured[T: BaseModel](
        self, *, system_prompt: str, user_prompt: str, schema: type[T]
    ) -> T:
        self.calls.append((system_prompt, user_prompt))
        if isinstance(self.result, Exception):
            raise self.result
        return cast(T, self.result)


def paper(**changes: object) -> Paper:
    return Paper.model_validate(
        {
            "id": "2601.00001",
            "title": "Learning robust neural representations",
            "abstract": "We evaluate a neural network for classification.",
            "authors": ["Ada Researcher"],
            "published_at": datetime(2026, 1, 1, tzinfo=UTC),
            "categories": ["cs.LG"],
            **changes,
        }
    )


def settings(path: Path, **changes: object) -> Settings:
    return Settings.model_validate({"cache_dir": path / "cache", **changes})


async def test_deterministic_is_not_ai_confirmed(tmp_path: Path) -> None:
    result = await RelevanceFilter(settings(tmp_path)).classify(paper())
    assert result is not None
    assert result.classifier == "deterministic"
    assert result.confidence < 0.5
    assert "not AI-confirmed" in result.reason
    assert all(value is None for value in result.quality.model_dump().values())


async def test_prefilter_rejects_unrelated_without_llm(tmp_path: Path) -> None:
    provider = RelevanceProvider(RuntimeError("Must not call"))
    result = await RelevanceFilter(settings(tmp_path), provider).classify(
        paper(
            title="Ocean tides", abstract="Measurements of sea level.", categories=["physics.ao-ph"]
        )
    )
    assert result is None
    assert not provider.calls


async def test_keyword_accepts_cross_category_and_threshold_rejects(tmp_path: Path) -> None:
    provider = RelevanceProvider(
        RelevanceResult(relevance_score=20, reason="Not relevant", confidence=1)
    )
    result = await RelevanceFilter(settings(tmp_path), provider).classify(
        paper(categories=["physics.ao-ph"], abstract="Machine learning for weather.")
    )
    assert result is None
    assert len(provider.calls) == 1


async def test_only_title_abstract_and_no_prestige(tmp_path: Path) -> None:
    estimate = ScoreDimension(score=99, reason="Unsupported prestige", confidence=1)
    provider = RelevanceProvider(
        RelevanceResult(
            relevance_score=90,
            reason="AI method",
            confidence=0.8,
            quality=QualityDimensions(venue=estimate, author_lab=estimate),
        )
    )
    result = await RelevanceFilter(settings(tmp_path), provider).classify(paper())
    assert result is not None and result.classifier == "llm"
    assert result.quality.venue is None and result.quality.author_lab is None
    assert set(json.loads(provider.calls[0][1])) == {"paper_id", "title", "abstract"}
    assert "UNTRUSTED" in provider.calls[0][0]


@pytest.mark.parametrize(
    "response", [{"relevance_score": 200}, RuntimeError("Provider unavailable")]
)
async def test_malformed_and_provider_errors_propagate(tmp_path: Path, response: object) -> None:
    provider = RelevanceProvider(response)
    expected = LLMResponseError if isinstance(response, dict) else RuntimeError
    with pytest.raises(expected):
        await RelevanceFilter(settings(tmp_path), provider).classify(paper())


async def test_relevance_cache_and_invalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    provider = RelevanceProvider(
        RelevanceResult(relevance_score=90, reason="Method", confidence=0.8)
    )
    configuration = settings(tmp_path)
    classifier = RelevanceFilter(configuration, provider)
    await classifier.classify(paper())
    await classifier.classify(paper())
    assert len(provider.calls) == 1
    await classifier.classify(paper(abstract="Neural network with new abstract."))
    configuration.relevance_model = "different-model"
    await classifier.classify(paper())
    configuration.relevance_prompt_version = "v2"
    await classifier.classify(paper())
    override = tmp_path / "prompts"
    override.mkdir()
    (override / "relevance.md").write_text("Edited relevance instructions", encoding="utf-8")
    await classifier.classify(paper())
    assert len(provider.calls) == 5
