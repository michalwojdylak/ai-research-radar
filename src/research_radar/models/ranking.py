from datetime import datetime
from typing import Literal

from pydantic import Field

from research_radar.models.metrics import PaperMetrics
from research_radar.models.paper import Enrichment, Paper, RadarModel


class ScoreDimension(RadarModel):
    score: float = Field(ge=0, le=100)
    reason: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class QualityDimensions(RadarModel):
    scientific_impact: ScoreDimension | None = None
    novelty: ScoreDimension | None = None
    empirical_quality: ScoreDimension | None = None
    practical_relevance: ScoreDimension | None = None
    venue: ScoreDimension | None = None
    author_lab: ScoreDimension | None = None


class RelevanceResult(RadarModel):
    relevance_score: int = Field(ge=0, le=100)
    reason: str
    confidence: float = Field(ge=0, le=1)
    classifier: Literal["llm", "deterministic"] = "llm"
    quality: QualityDimensions = Field(default_factory=QualityDimensions)


class ScoreComponent(RadarModel):
    name: str
    value: float | None = Field(default=None, ge=0, le=100)
    weight: float = Field(ge=0)
    contribution: float | None = None


class ScoreBreakdown(RadarModel):
    score: float | None = Field(default=None, ge=0, le=100)
    coverage: float = Field(default=0, ge=0, le=1)
    components: list[ScoreComponent] = Field(default_factory=list)


class PaperScores(RadarModel):
    relevance_score: int | None = Field(default=None, ge=0, le=100)
    importance: float | None = Field(default=None, ge=0, le=100)
    trending: float | None = Field(default=None, ge=0, le=100)
    overall: float | None = Field(default=None, ge=0, le=100)
    freshness_multiplier: float = 1.0
    quality: QualityDimensions = Field(default_factory=QualityDimensions)
    importance_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    trending_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    overall_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)


class PaperRecord(RadarModel):
    paper: Paper
    relevance: RelevanceResult | None = None
    enrichment: Enrichment = Field(default_factory=Enrichment)
    metrics: PaperMetrics = Field(default_factory=PaperMetrics)
    scores: PaperScores = Field(default_factory=PaperScores)


class RankingEntry(RadarModel):
    paper_id: str
    title: str
    rank: int = Field(ge=1)
    score: float
    coverage: float = Field(ge=0, le=1)


class Rankings(RadarModel):
    timestamp: datetime
    trending: list[RankingEntry] = Field(default_factory=list)
    most_important: list[RankingEntry] = Field(default_factory=list)
    selected_ids: list[str] = Field(default_factory=list)
