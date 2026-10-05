"""Strict input contracts shared by parsers and the public HTTP cache."""

import base64
import re
from datetime import datetime
from typing import Annotated, Literal

import httpx
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    TypeAdapter,
    model_validator,
)

from research_radar.ranking.deduplication import canonical_arxiv_id, normalize_doi

Count = Annotated[StrictInt, Field(ge=0)]
Year = Annotated[StrictInt, Field(ge=1, le=9999)]


class ProviderModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)


class OpenAlexSource(ProviderModel):
    display_name: StrictStr | None = None


class OpenAlexLocation(ProviderModel):
    landing_page_url: StrictStr | None = None
    pdf_url: StrictStr | None = None
    source: OpenAlexSource | None = None


class OpenAlexTopic(ProviderModel):
    display_name: StrictStr | None = None


class OpenAlexIds(ProviderModel):
    doi: StrictStr | None = None


class OpenAlexWork(ProviderModel):
    id: StrictStr = Field(pattern=r"^https://openalex\.org/W\d+$")
    display_name: StrictStr | None = None
    doi: StrictStr | None = None
    ids: OpenAlexIds | None = None
    cited_by_count: Count | None = None
    publication_year: Year | None = None
    topics: list[OpenAlexTopic] | None = None
    primary_location: OpenAlexLocation | None = None
    locations: list[OpenAlexLocation] | None = None
    related_works: list[StrictStr] | None = None


class SemanticScholarIds(ProviderModel):
    ArXiv: StrictStr | None = None
    DOI: StrictStr | None = None


class SemanticScholarPDF(ProviderModel):
    url: StrictStr | None = None


class SemanticScholarPaper(ProviderModel):
    paperId: StrictStr = Field(pattern=r"^[A-Za-z0-9_-]+$")
    externalIds: SemanticScholarIds | None = None
    title: StrictStr | None = None
    citationCount: Count | None = None
    influentialCitationCount: Count | None = None
    venue: StrictStr | None = None
    year: Year | None = None
    openAccessPdf: SemanticScholarPDF | None = None


class HuggingFaceAsset(ProviderModel):
    id: StrictStr


class HuggingFacePaper(ProviderModel):
    id: StrictStr
    upvotes: Count | None = None
    githubRepo: StrictStr | None = None
    models: list[StrictStr | HuggingFaceAsset] | None = None
    datasets: list[StrictStr | HuggingFaceAsset] | None = None
    trending: StrictBool | None = Field(
        default=None, validation_alias=AliasChoices("isTrending", "trending")
    )


class HuggingFaceRecord(ProviderModel):
    paper: HuggingFacePaper
    upvotes: Count | None = None
    githubRepo: StrictStr | None = None
    models: list[StrictStr | HuggingFaceAsset] | None = None
    datasets: list[StrictStr | HuggingFaceAsset] | None = None
    trending: StrictBool | None = Field(
        default=None, validation_alias=AliasChoices("isTrending", "trending")
    )


class GitHubRepository(ProviderModel):
    full_name: StrictStr = Field(pattern=r"^[A-Za-z0-9-]+/[A-Za-z0-9_.-]+$")
    private: StrictBool | None = None
    stargazers_count: Count | None = None
    forks_count: Count | None = None
    updated_at: datetime | None = None
    pushed_at: datetime | None = None


class GitHubReadme(ProviderModel):
    content: StrictStr
    encoding: Literal["base64"]

    @model_validator(mode="after")
    def validate_content(self) -> "GitHubReadme":
        decoded = base64.b64decode("".join(self.content.split()), validate=True)
        if len(decoded) > 1_000_000:
            raise ValueError("GitHub README exceeds size limit")
        decoded.decode("utf-8")
        return self


class GitHubSearchItem(ProviderModel):
    full_name: StrictStr
    private: StrictBool | None = None


class GitHubSearchResults(ProviderModel):
    items: list[GitHubSearchItem]
    total_count: Count | None = None
    incomplete_results: StrictBool | None = None


class GitHubContributor(ProviderModel):
    id: Annotated[StrictInt, Field(ge=1)] | None = None
    login: StrictStr | None = None
    email: StrictStr | None = None

    @model_validator(mode="after")
    def require_identity(self) -> "GitHubContributor":
        if self.id is None and not self.login and not self.email:
            raise ValueError("Contributor lacks an identity")
        return self


class GitHubRelease(ProviderModel):
    id: Annotated[StrictInt, Field(ge=1)]


CONTRIBUTORS = TypeAdapter(list[GitHubContributor])
RELEASES = TypeAdapter(list[GitHubRelease])


def validate_provider_response(response: httpx.Response) -> None:
    """Reject malformed API-success bodies before they reach the disk cache."""
    host, path = response.request.url.host, response.request.url.path
    if host == "api.openalex.org" and path.startswith("/works/"):
        work = OpenAlexWork.model_validate_json(response.content)
        identifier = path.removeprefix("/works/")
        if re.fullmatch(r"W\d+", identifier):
            if work.id != f"https://openalex.org/{identifier}":
                raise ValueError("Mismatched OpenAlex work identifier")
        elif doi := normalize_doi(identifier):
            returned_dois = {normalize_doi(work.doi)}
            if work.ids:
                returned_dois.add(normalize_doi(work.ids.doi))
            arxiv = (
                canonical_arxiv_id(doi.removeprefix("10.48550/arxiv."))
                if doi.startswith("10.48550/arxiv.")
                else None
            )
            locations = list(work.locations or [])
            if work.primary_location:
                locations.append(work.primary_location)
            exact_arxiv = arxiv is not None and any(
                canonical_arxiv_id(url) == arxiv
                for location in locations
                for url in (location.landing_page_url, location.pdf_url)
            )
            if doi not in returned_dois and not exact_arxiv:
                raise ValueError("OpenAlex response lacks matching DOI or arXiv identifier")
    elif host == "api.semanticscholar.org" and path.startswith("/graph/v1/paper/"):
        ss_work = SemanticScholarPaper.model_validate_json(response.content)
        identifier = path.removeprefix("/graph/v1/paper/")
        if ss_work.externalIds:
            if identifier.startswith("ARXIV:") and ss_work.externalIds.ArXiv:
                if canonical_arxiv_id(ss_work.externalIds.ArXiv) != canonical_arxiv_id(
                    identifier.removeprefix("ARXIV:")
                ):
                    raise ValueError("Mismatched Semantic Scholar arXiv identifier")
            elif identifier.startswith("DOI:") and ss_work.externalIds.DOI:
                if normalize_doi(ss_work.externalIds.DOI) != normalize_doi(
                    identifier.removeprefix("DOI:")
                ):
                    raise ValueError("Mismatched Semantic Scholar DOI")
    elif host == "huggingface.co" and path.startswith("/api/papers/"):
        record = HuggingFaceRecord.model_validate_json(response.content)
        expected = canonical_arxiv_id(path.removeprefix("/api/papers/"))
        if expected is None or canonical_arxiv_id(record.paper.id) != expected:
            raise ValueError("Mismatched Hugging Face paper identifier")
    elif host == "api.github.com":
        if path == "/search/repositories":
            GitHubSearchResults.model_validate_json(response.content)
        elif path.startswith("/repos/"):
            if path.endswith("/readme"):
                GitHubReadme.model_validate_json(response.content)
            elif path.endswith("/contributors"):
                CONTRIBUTORS.validate_json(response.content)
            elif path.endswith("/releases"):
                RELEASES.validate_json(response.content)
            elif len(path.strip("/").split("/")) == 3:
                repository = GitHubRepository.model_validate_json(response.content)
                if repository.full_name.lower() != path.removeprefix("/repos/").lower():
                    raise ValueError("Mismatched GitHub repository")
