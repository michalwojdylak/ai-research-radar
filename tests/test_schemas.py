from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from research_radar.config import Settings
from research_radar.models.analysis import CriticIssue, CriticResult
from research_radar.models.metrics import MetricsSnapshot
from research_radar.models.paper import Paper, SourceReference
from research_radar.models.ranking import ScoreDimension


def test_absent_metrics_are_not_zero() -> None:
    metrics = MetricsSnapshot(timestamp=datetime.now(UTC))
    assert metrics.citations is None
    assert metrics.github_stars is None
    assert metrics.hf_upvotes is None


def test_invalid_scores_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ScoreDimension(score=101, reason="test", confidence=1)
    with pytest.raises(ValidationError):
        ScoreDimension.model_validate({"score": 50, "reason": "test", "confidence": 1, "fake": 1})


def test_sources_require_http_urls() -> None:
    with pytest.raises(ValidationError):
        SourceReference(title="test", url="javascript:alert(1)", source_type="paper")


def test_critic_cannot_pass_high_issue() -> None:
    with pytest.raises(ValidationError):
        CriticResult(
            passed=True,
            issues=[CriticIssue(severity="high", issue="Invented benchmark", correction="Remove")],
        )


def test_env_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARXIV_CATEGORIES", "cs.AI, cs.LG")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("WEIGHTS__TRENDING__GITHUB_VELOCITY", "0.30")
    settings = Settings(_env_file=None)
    assert settings.arxiv_categories == ["cs.AI", "cs.LG"]
    assert settings.openai_api_key is None
    assert settings.dry_run


def test_dates_normalized_to_utc() -> None:
    paper = Paper(
        id="x",
        title="AI paper",
        abstract="AI",
        authors=[],
        categories=["cs.AI"],
        published_at=datetime(2026, 10, 1),
    )
    assert paper.published_at.tzinfo is UTC


def test_weight_validation() -> None:
    with pytest.raises(ValidationError):
        Settings(weights={"importance": {"novelty": 1.0}}, _env_file=None)
