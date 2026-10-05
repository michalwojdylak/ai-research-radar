"""Public repository enrichment requiring direct links or exact README proof."""

import base64
import re
from typing import Literal
from urllib.parse import quote, unquote

from research_radar.collectors.common import (
    EXPECTED_ERRORS,
    Collector,
    github_repository_url,
    log_failure,
    merge_enrichment,
    paper_arxiv_id,
    paper_doi,
)
from research_radar.collectors.schemas import (
    CONTRIBUTORS,
    RELEASES,
    GitHubReadme,
    GitHubRepository,
    GitHubSearchResults,
)
from research_radar.models.paper import (
    Enrichment,
    Paper,
    RepositoryEvidence,
    SourceReference,
)
from research_radar.ranking.deduplication import canonical_arxiv_id

MAX_SEARCH_PAGES = 3
MAX_CANDIDATES = 20
MAX_COUNT_PAGES = 10
_ARXIV_IN_TEXT = re.compile(
    r"(?<![A-Za-z0-9_])(\d{4}\.\d{4,5}|[a-z][a-z0-9.-]*/\d{7})"
    r"(?:v\d+)?(?![A-Za-z0-9_-]|\.\d)",
    re.IGNORECASE,
)


def _readme_matches(text: str, paper: Paper) -> bool:
    text = unquote(text)
    arxiv = paper_arxiv_id(paper)
    if arxiv and any(
        canonical_arxiv_id(match.group()) == arxiv for match in _ARXIV_IN_TEXT.finditer(text)
    ):
        return True
    doi = paper_doi(paper)
    return bool(
        doi
        and re.search(
            rf"(?<![A-Za-z0-9._-]){re.escape(doi)}(?![A-Za-z0-9/_-]|\.[A-Za-z0-9])",
            text,
            re.IGNORECASE,
        )
    )


class GitHubCollector(Collector):
    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.settings.github_token:
            headers["Authorization"] = f"Bearer {self.settings.github_token.get_secret_value()}"
        return headers

    @staticmethod
    def _api_url(repository_url: str) -> str:
        normalized = github_repository_url(repository_url)
        if normalized is None:
            raise ValueError("Invalid public GitHub repository URL")
        owner, repo = normalized.removeprefix("https://github.com/").split("/")
        return f"https://api.github.com/repos/{quote(owner, safe='')}/{quote(repo, safe='')}"

    async def _readme(self, repository_url: str) -> str:
        response = await self.http.get(
            f"{self._api_url(repository_url)}/readme", headers=self._headers()
        )
        readme = GitHubReadme.model_validate_json(response.content)
        decoded = base64.b64decode("".join(readme.content.split()), validate=True)
        if len(decoded) > 1_000_000:
            raise ValueError("GitHub README exceeds size limit")
        return decoded.decode("utf-8")

    async def _count(
        self, repository_url: str, endpoint: Literal["contributors", "releases"]
    ) -> int:
        observed: set[str | int] = set()
        for page in range(1, MAX_COUNT_PAGES + 1):
            params: dict[str, str | int] = {"per_page": 100, "page": page}
            if endpoint == "contributors":
                params["anon"] = "true"
            response = await self.http.get(
                f"{self._api_url(repository_url)}/{endpoint}",
                headers=self._headers(),
                params=params,
            )
            if endpoint == "contributors":
                contributors = CONTRIBUTORS.validate_json(response.content)
                identifiers: list[str | int] = []
                for contributor in contributors:
                    identifier = contributor.id or contributor.login or contributor.email
                    if identifier is None:
                        raise ValueError("Contributor lacks an identity")
                    identifiers.append(identifier)
            else:
                identifiers = [release.id for release in RELEASES.validate_json(response.content)]
            before = len(observed)
            observed.update(identifiers)
            has_next = any(
                'rel="next"' in link or "rel=next" in link
                for link in response.headers.get("link", "").split(",")
            )
            if not has_next and (len(identifiers) < 100 or "link" in response.headers):
                return len(observed)
            if identifiers and len(observed) == before:
                raise ValueError("GitHub pagination repeated results")
        raise ValueError("GitHub count pagination limit reached; exact count unavailable")

    async def _repository(self, evidence: RepositoryEvidence, readme: str | None) -> Enrichment:
        normalized = github_repository_url(evidence.repository_url)
        if normalized is None:
            raise ValueError("Invalid repository evidence")
        result = Enrichment(
            github_url=normalized,
            github_readme=readme,
            repository_evidence=evidence.model_copy(update={"repository_url": normalized}),
            sources=[
                SourceReference(
                    title="Associated code repository", url=normalized, source_type="github"
                )
            ],
        )
        try:
            response = await self.http.get(self._api_url(normalized), headers=self._headers())
            repository = GitHubRepository.model_validate_json(response.content)
            if (
                repository.full_name.lower()
                != normalized.removeprefix("https://github.com/").lower()
                or repository.private is True
            ):
                raise ValueError("Mismatched or private GitHub repository")
            result.github_stars = repository.stargazers_count
            result.github_forks = repository.forks_count
            result.github_updated_at = repository.pushed_at or repository.updated_at
        except EXPECTED_ERRORS as error:
            log_failure("github.repository", error)
            return result
        if readme is None:
            try:
                result.github_readme = await self._readme(normalized)
            except EXPECTED_ERRORS as error:
                log_failure("github.readme", error)
        for endpoint in ("contributors", "releases"):
            try:
                count = await self._count(normalized, endpoint)
                if endpoint == "contributors":
                    result.github_contributors = count
                else:
                    result.github_releases = count
            except EXPECTED_ERRORS as error:
                log_failure(f"github.{endpoint}", error)
        return result

    async def _search(self, paper: Paper) -> tuple[RepositoryEvidence, str] | None:
        identifier = paper_arxiv_id(paper) or paper_doi(paper)
        if not identifier:
            return None
        inspected: set[str] = set()
        for page in range(1, MAX_SEARCH_PAGES + 1):
            response = await self.http.get(
                "https://api.github.com/search/repositories",
                params={"q": f'"{identifier}" in:readme', "per_page": 10, "page": page},
                headers=self._headers(),
            )
            results = GitHubSearchResults.model_validate_json(response.content)
            for item in results.items:
                repository_url = github_repository_url(f"https://github.com/{item.full_name}")
                if item.private is True or repository_url is None or repository_url in inspected:
                    continue
                inspected.add(repository_url)
                try:
                    readme = await self._readme(repository_url)
                    if _readme_matches(readme, paper):
                        return (
                            RepositoryEvidence(
                                repository_url=repository_url,
                                evidence_url=f"{repository_url}#readme",
                                reason="Repository README contains the exact paper arXiv ID or DOI",
                            ),
                            readme,
                        )
                except EXPECTED_ERRORS as error:
                    log_failure("github.search_readme", error)
                if len(inspected) >= MAX_CANDIDATES:
                    return None
            if len(results.items) < 10 or (
                results.total_count is not None and page * 10 >= results.total_count
            ):
                break
        return None

    async def enrich(
        self, paper: Paper, *, metadata_evidence: list[RepositoryEvidence] | None = None
    ) -> Enrichment:
        evidence_list = list(metadata_evidence or [])
        arxiv, doi = paper_arxiv_id(paper), paper_doi(paper)
        paper_url = (
            f"https://arxiv.org/abs/{arxiv}" if arxiv else f"https://doi.org/{doi}" if doi else None
        )
        for match in re.finditer(r"https?://[^\s<>\"'`]+", paper.abstract):
            if repository_url := github_repository_url(match.group().rstrip(".,;:!?)]}")):
                evidence_list.append(
                    RepositoryEvidence(
                        repository_url=repository_url,
                        evidence_url=paper_url or repository_url,
                        reason="Explicit repository link in the paper abstract",
                    )
                )
        seen: set[str] = set()
        fallback = Enrichment()
        for evidence in evidence_list[:MAX_CANDIDATES]:
            normalized = github_repository_url(evidence.repository_url)
            if normalized is None or normalized in seen:
                continue
            seen.add(normalized)
            result = await self._repository(evidence, None)
            if fallback.github_url is None:
                fallback = merge_enrichment(fallback, result)
            if result.github_stars is not None or result.github_forks is not None:
                return result
        if seen:
            return fallback
        found_repository = await self._search(paper)
        if found_repository:
            evidence, readme = found_repository
            return await self._repository(evidence, readme)
        return fallback
