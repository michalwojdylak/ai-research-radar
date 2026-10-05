from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RadarModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_validator("*", mode="after")
    @classmethod
    def normalize_dates(cls, value: object) -> object:
        if isinstance(value, datetime):
            return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return value


class SourceReference(RadarModel):
    title: str
    url: str
    source_type: Literal[
        "paper", "github", "project", "openalex", "semantic_scholar", "huggingface"
    ]

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        from urllib.parse import urlsplit

        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
        ):
            raise ValueError("Sources must have an absolute HTTP(S) URL without credentials")
        return value


class Paper(RadarModel):
    id: str = Field(min_length=1)
    arxiv_id: str | None = None
    title: str = Field(min_length=1)
    abstract: str
    authors: list[str]
    published_at: datetime
    updated_at: datetime | None = None
    categories: list[str]
    pdf_url: str | None = None
    abs_url: str | None = None
    doi: str | None = None


class PaperLinks(RadarModel):
    arxiv: str | None = None
    pdf: str | None = None
    doi: str | None = None
    github: str | None = None
    project: str | None = None
    huggingface: str | None = None


class RelatedWork(RadarModel):
    title: str
    url: str
    year: int | None = None


class RepositoryEvidence(RadarModel):
    repository_url: str
    evidence_url: str
    reason: str


class Enrichment(RadarModel):
    openalex_id: str | None = None
    semantic_scholar_id: str | None = None
    openalex_citations: int | None = Field(default=None, ge=0)
    semantic_scholar_citations: int | None = Field(default=None, ge=0)
    influential_citations: int | None = Field(default=None, ge=0)
    topics: list[str] = Field(default_factory=list)
    related_works: list[RelatedWork] = Field(default_factory=list)
    venue: str | None = None
    publication_year: int | None = None
    github_url: str | None = None
    github_stars: int | None = Field(default=None, ge=0)
    github_forks: int | None = Field(default=None, ge=0)
    github_contributors: int | None = Field(default=None, ge=0)
    github_updated_at: datetime | None = None
    github_releases: int | None = Field(default=None, ge=0)
    github_readme: str | None = None
    repository_evidence: RepositoryEvidence | None = None
    hf_paper_url: str | None = None
    hf_upvotes: int | None = Field(default=None, ge=0)
    hf_trending: bool | None = None
    hf_models: list[str] | None = None
    hf_datasets: list[str] | None = None
    sources: list[SourceReference] = Field(default_factory=list)

    @property
    def citations(self) -> int | None:
        # Counts from different indexes are not additive.
        counts = [
            x for x in (self.openalex_citations, self.semantic_scholar_citations) if x is not None
        ]
        return max(counts) if counts else None
