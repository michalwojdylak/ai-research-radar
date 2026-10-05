from datetime import date, datetime
from typing import Literal

from pydantic import Field

from research_radar.models.paper import Paper, RadarModel
from research_radar.models.ranking import RelevanceResult


class RelevanceObservation(RadarModel):
    observed_at: datetime
    paper: Paper
    relevance: RelevanceResult | None
    primary_only: bool = True
    model: str | None = None
    prompt_version: str | None = None


BackfillPhase = Literal["discovery", "relevance", "enrichment", "ranking", "analysis", "complete"]


class BackfillCheckpoint(RadarModel):
    job_id: str
    period_id: str
    start_date: date
    end_date: date
    phase: BackfillPhase = "discovery"
    status: Literal["pending", "running", "failed", "complete"] = "pending"
    processed_ids: list[str] = Field(default_factory=list)
    analyses_done: list[str] = Field(default_factory=list)
    updated_at: datetime
    error_type: str | None = None
    discovery_cutoff: datetime | None = None
    ranking_fingerprint: str | None = None


class BackfillReport(RadarModel):
    job_id: str
    start_date: date
    end_date: date
    total_weeks: int
    completed_weeks: int = 0
    processed_weeks: int = 0
    failed_weeks: list[str] = Field(default_factory=list)
    deep_analyzed: int = 0
    llm_relevance_enabled: bool = False
    status: Literal["success", "partial", "failed"] = "success"
