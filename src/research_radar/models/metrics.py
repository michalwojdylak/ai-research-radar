from datetime import datetime

from pydantic import Field

from research_radar.models.paper import RadarModel


class MetricsSnapshot(RadarModel):
    timestamp: datetime
    citations: int | None = Field(default=None, ge=0)
    github_stars: int | None = Field(default=None, ge=0)
    hf_upvotes: int | None = Field(default=None, ge=0)
    github_url: str | None = None
    citation_source: str | None = None
    hf_paper_url: str | None = None


class PaperMetrics(RadarModel):
    citations: int | None = Field(default=None, ge=0)
    influential_citations: int | None = Field(default=None, ge=0)
    github_stars: int | None = Field(default=None, ge=0)
    github_forks: int | None = Field(default=None, ge=0)
    github_contributors: int | None = Field(default=None, ge=0)
    github_releases: int | None = Field(default=None, ge=0)
    hf_upvotes: int | None = Field(default=None, ge=0)
    github_star_velocity_24h: float | None = None
    github_star_velocity_7d: float | None = None
    citation_velocity: float | None = None
    hf_velocity: float | None = None
    observed_at: datetime | None = None
