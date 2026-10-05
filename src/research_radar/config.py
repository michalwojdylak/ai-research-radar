from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from research_radar.models.paper import RadarModel


class ImportanceWeights(RadarModel):
    scientific_impact: float = 0.30
    novelty: float = 0.25
    empirical_quality: float = 0.20
    practical_relevance: float = 0.10
    citation_signal: float = 0.05
    venue: float = 0.05
    author_lab: float = 0.05


class TrendingWeights(RadarModel):
    github_velocity: float = 0.30
    community_attention: float = 0.20
    hf_velocity: float = 0.15
    citation_velocity: float = 0.15
    freshness: float = 0.10
    absolute_attention: float = 0.10


class OverallWeights(RadarModel):
    scientific_impact: float = 0.25
    novelty: float = 0.20
    empirical_quality: float = 0.15
    practical_relevance: float = 0.10
    github_velocity: float = 0.10
    community_interest: float = 0.10
    citation_velocity: float = 0.05
    venue_author: float = 0.05


class ScoringWeights(RadarModel):
    importance: ImportanceWeights = Field(default_factory=ImportanceWeights)
    trending: TrendingWeights = Field(default_factory=TrendingWeights)
    overall: OverallWeights = Field(default_factory=OverallWeights)

    @model_validator(mode="after")
    def validate_weights(self) -> "ScoringWeights":
        for group in (self.importance, self.trending, self.overall):
            values = list(group.model_dump().values())
            if any(value < 0 for value in values) or abs(sum(values) - 1) > 1e-6:
                raise ValueError("Each weight group must contain nonnegative weights summing to 1")
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_nested_delimiter="__"
    )

    openai_api_key: SecretStr | None = None
    github_token: SecretStr | None = None
    semantic_scholar_api_key: SecretStr | None = None
    huggingface_token: SecretStr | None = None
    openalex_email: str | None = None
    research_days: int = Field(default=7, ge=1, le=3650)
    top_papers: int = Field(default=10, ge=1, le=100)
    arxiv_categories: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "stat.ML"]
    )
    dry_run: bool = True
    data_dir: Path = Path("data")
    cache_dir: Path = Path(".cache/research-radar")
    database_path: Path = Path("data/radar.sqlite3")
    concurrency: int = Field(default=4, ge=1, le=16)
    http_timeout: float = Field(default=45, gt=0)
    http_cache_ttl: int = Field(default=21600, ge=0)
    arxiv_page_size: int = Field(default=100, ge=1, le=2000)
    arxiv_request_interval: float = Field(default=3.1, ge=3)
    relevance_threshold: int = Field(default=60, ge=0, le=100)
    relevance_model: str = "gpt-4.1-mini"
    analysis_model: str = "gpt-4.1"
    critic_model: str = "gpt-4.1"
    relevance_prompt_version: str = "v1"
    analysis_prompt_version: str = "v1"
    critic_prompt_version: str = "v1"
    max_analysis_characters: int = Field(default=100000, ge=1000)
    max_pdf_bytes: int = Field(default=30000000, ge=1000)
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    github_repository: str | None = None
    github_base_branch: str = "master"

    @field_validator(
        "openai_api_key",
        "github_token",
        "semantic_scholar_api_key",
        "huggingface_token",
        "openalex_email",
        "github_repository",
        mode="before",
    )
    @classmethod
    def empty_is_unavailable(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("arxiv_categories", mode="before")
    @classmethod
    def parse_categories(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("arxiv_categories")
    @classmethod
    def valid_categories(cls, value: list[str]) -> list[str]:
        import re

        if not value or any(not re.fullmatch(r"[a-zA-Z-]+\.[A-Za-z]+", x) for x in value):
            raise ValueError("Provide comma-separated arXiv subject categories")
        return value
