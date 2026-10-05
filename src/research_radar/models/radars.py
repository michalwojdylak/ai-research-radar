from datetime import UTC, date, datetime, time, timedelta
from typing import Literal

from pydantic import Field, model_validator

from research_radar.config import RadarScoring, RetrospectiveWeights, ScoringWeights
from research_radar.models.analysis import PromptVersions
from research_radar.models.paper import Paper, RadarModel
from research_radar.models.ranking import PaperScores

RankingMode = Literal["publication", "retrospective"]
RadarStatus = Literal["draft", "review", "approved", "published", "needs_review"]
QualityBasis = Literal["historical_observation", "publication_document_proxy", "unavailable"]


class PeriodPaperScore(RadarModel):
    paper_id: str
    mode: RankingMode
    data_cutoff: datetime
    scores: PaperScores
    publication_score: float | None = Field(default=None, ge=0, le=100)
    retrospective_score: float | None = Field(default=None, ge=0, le=100)
    quality_basis: QualityBasis = "unavailable"
    quality_observed_at: datetime | None = None
    coverage: float = Field(default=0, ge=0, le=1)
    source_paper: Paper | None = None

    @property
    def ranking_score(self) -> float | None:
        return self.publication_score if self.mode == "publication" else self.retrospective_score


class WeeklyPaperSelection(RadarModel):
    paper_id: str
    rank: int = Field(ge=1)
    title: str = "Unknown"
    paper_url: str | None = None
    importance_score: float | None = Field(default=None, ge=0, le=100)
    trending_score: float | None = Field(default=None, ge=0, le=100)
    overall_score: float | None = Field(default=None, ge=0, le=100)
    publication_score: float | None = Field(default=None, ge=0, le=100)
    retrospective_score: float | None = Field(default=None, ge=0, le=100)
    selection_reason: str
    quality_basis: QualityBasis = "unavailable"
    coverage: float = Field(default=0, ge=0, le=1)
    analysis_status: RadarStatus | None = None
    paper_version: str | None = None


class RadarProvenance(RadarModel):
    mode: RankingMode = "publication"
    ranking_version: str
    scoring_version: str
    prompt_versions: PromptVersions
    generated_at: datetime
    data_cutoff: datetime
    scoring_configuration: RadarScoring
    status: RadarStatus = "draft"
    michaels_take: str | None = None
    is_complete: bool = False
    explanation: str = ""
    paper_scoring_weights: ScoringWeights = Field(default_factory=ScoringWeights)
    retrospective_weights: RetrospectiveWeights = Field(default_factory=RetrospectiveWeights)


class WeeklyRadar(RadarProvenance):
    type: Literal["weekly_radar"] = "weekly_radar"
    id: str = Field(pattern=r"^\d{4}-W\d{2}$")
    year: int
    week: int
    start_date: date
    end_date: date
    papers: list[WeeklyPaperSelection] = Field(default_factory=list)
    radar_score: float | None = Field(default=None, ge=0, le=100)
    retrospective_score: float | None = Field(default=None, ge=0, le=100)
    paper_count: int = Field(ge=0)
    candidate_count: int = Field(default=0, ge=0)
    scored_candidate_count: int = Field(default=0, ge=0)
    coverage_start: date
    coverage_end: date
    is_preview: bool = False
    quality_threshold: float = Field(ge=0, le=100)
    below_minimum: bool = False
    discovery_complete: bool = False

    @property
    def ranking_score(self) -> float | None:
        return self.radar_score if self.mode == "publication" else self.retrospective_score

    @model_validator(mode="after")
    def coherent_week(self) -> "WeeklyRadar":
        start = date.fromisocalendar(self.year, self.week, 1)
        end = date.fromisocalendar(self.year, self.week, 7)
        if self.id != f"{self.year:04}-W{self.week:02}" or (self.start_date, self.end_date) != (
            start,
            end,
        ):
            raise ValueError("Radar IDs and date ranges must agree with their ISO week")
        if not start <= self.coverage_start <= self.coverage_end <= end:
            raise ValueError("Discovery coverage must lie within the ISO week")
        if self.paper_count != len(self.papers) or len({p.paper_id for p in self.papers}) != len(
            self.papers
        ):
            raise ValueError("paper_count must match distinct paper references")
        if [p.rank for p in self.papers] != list(range(1, self.paper_count + 1)):
            raise ValueError("Selections require contiguous ranks")
        if self.is_preview and (self.is_complete or self.status in {"approved", "published"}):
            raise ValueError("Previews cannot be complete, approved, or published")
        cutoff_end = datetime.combine(end + timedelta(days=1), time.min, UTC)
        if self.mode == "publication" and not (
            datetime.combine(start, time.min, UTC) <= self.data_cutoff <= cutoff_end
        ):
            raise ValueError(
                "Publication-mode cutoff must lie within the original publication week"
            )
        if self.status in {"review", "approved", "published"} and any(
            paper.analysis_status == "needs_review" for paper in self.papers
        ):
            raise ValueError("Radars with unresolved paper reviews cannot pass the review gate")
        if self.mode == "publication" and self.retrospective_score is not None:
            raise ValueError("Publication documents cannot carry a retrospective aggregate")
        if self.mode == "retrospective" and self.radar_score is not None:
            raise ValueError("Retrospective scores must not occupy radar_score")
        return self


class PeriodReference(RadarModel):
    id: str
    score: float | None = Field(default=None, ge=0, le=100)
    paper_count: int = Field(ge=0)


class MonthlyRadar(RadarProvenance):
    type: Literal["monthly_radar"] = "monthly_radar"
    id: str = Field(pattern=r"^\d{4}-\d{2}$")
    year: int
    month: int = Field(ge=1, le=12)
    score: float | None = Field(default=None, ge=0, le=100)
    retrospective_score: float | None = Field(default=None, ge=0, le=100)
    best_week: str | None = None
    top_weeks: list[PeriodReference] = Field(default_factory=list)
    paper_count: int = Field(default=0, ge=0)
    known_week_count: int = Field(default=0, ge=0)
    unknown_week_count: int = Field(default=0, ge=0)

    @property
    def ranking_score(self) -> float | None:
        return self.score if self.mode == "publication" else self.retrospective_score

    @model_validator(mode="after")
    def coherent_month(self) -> "MonthlyRadar":
        if self.id != f"{self.year:04}-{self.month:02}":
            raise ValueError("Monthly ID must agree with year and month")
        if self.mode == "publication" and self.retrospective_score is not None:
            raise ValueError("Publication documents cannot carry a retrospective aggregate")
        if self.mode == "retrospective" and self.score is not None:
            raise ValueError("Retrospective scores must not occupy score")
        return self


class YearlyRadar(RadarProvenance):
    type: Literal["yearly_radar"] = "yearly_radar"
    id: str = Field(pattern=r"^\d{4}$")
    year: int
    score: float | None = Field(default=None, ge=0, le=100)
    retrospective_score: float | None = Field(default=None, ge=0, le=100)
    best_month: str | None = None
    top_months: list[PeriodReference] = Field(default_factory=list)
    best_weeks: list[PeriodReference] = Field(default_factory=list)
    top_papers: list[str] = Field(default_factory=list)
    paper_count: int = Field(default=0, ge=0)
    known_month_count: int = Field(default=0, ge=0)
    unknown_month_count: int = Field(default=0, ge=0)

    @property
    def ranking_score(self) -> float | None:
        return self.score if self.mode == "publication" else self.retrospective_score

    @model_validator(mode="after")
    def coherent_year(self) -> "YearlyRadar":
        if self.id != f"{self.year:04}":
            raise ValueError("Yearly ID must agree with year")
        if self.mode == "publication" and self.retrospective_score is not None:
            raise ValueError("Publication documents cannot carry a retrospective aggregate")
        if self.mode == "retrospective" and self.score is not None:
            raise ValueError("Retrospective scores must not occupy score")
        return self
