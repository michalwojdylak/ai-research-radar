from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from research_radar.models.metrics import PaperMetrics
from research_radar.models.paper import PaperLinks, RadarModel, SourceReference
from research_radar.models.ranking import PaperScores

SectionName = Literal[
    "tldr",
    "why_should_i_care",
    "timeline",
    "problem",
    "core_idea",
    "whats_new",
    "how_it_works",
    "comparisons",
    "key_results",
    "beats_previous_work",
    "limitations",
    "reproducibility",
    "why_it_matters",
    "what_next",
]


class TimelineEvent(RadarModel):
    year: str
    title: str
    description: str
    related_paper: str | None = None


class SectionEvidence(RadarModel):
    section: SectionName
    sources: list[SourceReference]


class AnalysisContent(RadarModel):
    tldr: str = Field(min_length=1)
    why_should_i_care: str = Field(min_length=1)
    timeline: list[TimelineEvent]
    problem: str = Field(min_length=1)
    core_idea: str = Field(min_length=1)
    whats_new: str = Field(min_length=1)
    how_it_works: str = Field(min_length=1)
    comparisons: str = Field(min_length=1)
    key_results: str = Field(min_length=1)
    beats_previous_work: str = Field(min_length=1)
    limitations: str = Field(min_length=1)
    authors_limitations: str = Field(min_length=1)
    analysis_limitations: str = Field(min_length=1)
    reproducibility: Literal["Easy", "Moderate", "Hard", "Very Hard", "Unknown"]
    reproducibility_reason: str = Field(min_length=1)
    why_it_matters: str = Field(min_length=1)
    what_next: str = Field(min_length=1)
    sources: list[SourceReference]
    section_sources: list[SectionEvidence]
    questions_for_michael: list[str] = Field(default_factory=list)


class CriticIssue(RadarModel):
    severity: Literal["low", "medium", "high"]
    issue: str
    correction: str


class CriticResult(RadarModel):
    passed: bool
    issues: list[CriticIssue]

    @model_validator(mode="after")
    def consistent_result(self) -> "CriticResult":
        if self.passed and any(issue.severity == "high" for issue in self.issues):
            raise ValueError("A critic cannot pass unresolved high-severity issues")
        return self


class PromptVersions(RadarModel):
    relevance: str
    analysis: str
    critic: str


class AnalysisResult(RadarModel):
    paper_id: str
    content: AnalysisContent
    critic: CriticResult
    status: Literal["review", "needs_review"]
    full_text_available: bool
    input_truncated: bool = False
    confidence: Literal["reduced", "standard"]
    generated_at: datetime
    model: str
    critic_model: str
    prompt_versions: PromptVersions
    cache_key: str
    input_fingerprint: str = ""

    @model_validator(mode="after")
    def review_gate(self) -> "AnalysisResult":
        if self.status == "review" and not self.critic.passed:
            raise ValueError("Only critic-passed analyses can enter review")
        return self


class ResearchPaper(AnalysisContent):
    id: str
    slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    title: str
    authors: list[str]
    published_at: datetime
    venue: str | None
    categories: list[str]
    abstract: str
    links: PaperLinks
    metrics: PaperMetrics
    scores: PaperScores
    michaels_take: str | None = None
    status: Literal["draft", "review", "approved", "published", "needs_review"]
    full_text_available: bool
    input_truncated: bool = False
    analysis_confidence: Literal["reduced", "standard"]
    critic: CriticResult
    generated_at: datetime
    analysis_model: str
    critic_model: str
    prompt_versions: PromptVersions
    arxiv_id: str | None = None
    updated_at: datetime | None = None

    @model_validator(mode="after")
    def publication_gate(self) -> "ResearchPaper":
        if self.status in {"review", "approved", "published"} and not self.critic.passed:
            raise ValueError("Unresolved critic issues require needs_review or draft status")
        return self
