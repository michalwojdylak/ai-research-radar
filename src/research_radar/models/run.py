from datetime import datetime
from typing import Literal

from pydantic import Field

from research_radar.models.paper import RadarModel


class PipelineFailure(RadarModel):
    stage: str
    paper_id: str | None = None
    error_type: str
    message: str


class RunReport(RadarModel):
    id: str
    started_at: datetime
    finished_at: datetime | None = None
    stage: str
    status: Literal["running", "success", "partial", "failed"] = "running"
    discovered: int = 0
    relevant: int = 0
    enriched: int = 0
    ranked: int = 0
    deep_analyzed: int = 0
    critic_passed: int = 0
    needs_review: int = 0
    generated: int = 0
    skipped_analysis: int = 0
    failures: list[PipelineFailure] = Field(default_factory=list)
    period_id: str | None = None
    is_preview: bool = False
