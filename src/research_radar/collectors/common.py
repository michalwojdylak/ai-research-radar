"""Small shared contracts for independent enrichment providers."""

import logging
import re
from urllib.parse import urlsplit
from xml.etree import ElementTree

import httpx

from research_radar.collectors.http import NETWORK_ERRORS, CachedHTTP
from research_radar.config import Settings
from research_radar.models.paper import Enrichment, Paper, RepositoryEvidence
from research_radar.ranking.deduplication import canonical_arxiv_id, normalize_doi

EXPECTED_ERRORS = (
    *NETWORK_ERRORS,
    httpx.HTTPStatusError,
    httpx.DecodingError,
    ValueError,
    ElementTree.ParseError,
)
logger = logging.getLogger(__name__)
_OWNER = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})")
_REPO = re.compile(r"[A-Za-z0-9_.-]{1,100}")
_RESERVED_OWNERS = {
    "about",
    "collections",
    "enterprise",
    "explore",
    "features",
    "login",
    "marketplace",
    "orgs",
    "search",
    "security",
    "settings",
    "sponsors",
    "topics",
    "users",
}


class Collector:
    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient,
        *,
        http: CachedHTTP | None = None,
    ) -> None:
        self.settings = settings
        self.http = http or CachedHTTP(settings, client)


def paper_arxiv_id(paper: Paper) -> str | None:
    return next(
        (
            identifier
            for value in (paper.arxiv_id, paper.id, paper.abs_url, paper.pdf_url)
            if (identifier := canonical_arxiv_id(value)) is not None
        ),
        None,
    )


def github_repository_url(value: str) -> str | None:
    """Normalize a GitHub repo URL, never an API URL or a different host."""
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 80, 443}
    ):
        return None
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2:
        return None
    owner, repo = parts[:2]
    if repo.endswith(".git"):
        repo = repo[:-4]
    if (
        not _OWNER.fullmatch(owner)
        or not _REPO.fullmatch(repo)
        or repo in {".", ".."}
        or owner.lower() in _RESERVED_OWNERS
        or (len(parts) > 2 and parts[2] not in {"tree", "blob", "releases", "issues"})
    ):
        return None
    return f"https://github.com/{owner}/{repo}"


def metadata_repository(value: str | None, evidence_url: str) -> RepositoryEvidence | None:
    if value and (repository := github_repository_url(value)):
        return RepositoryEvidence(
            repository_url=repository,
            evidence_url=evidence_url,
            reason="Explicit GitHub link in identifier-resolved paper API metadata",
        )
    return None


def merge_enrichment(current: Enrichment, incoming: Enrichment) -> Enrichment:
    """Missing values never erase measurements; lists merge deterministically."""
    values = current.model_dump()
    for field, value in incoming.model_dump().items():
        if value is None:
            continue
        if isinstance(value, list):
            existing = values[field]
            if existing is None:
                values[field] = list(value)
                continue
            for item in value:
                if item not in existing:
                    existing.append(item)
        else:
            values[field] = value
    return Enrichment.model_validate(values)


def log_failure(source: str, error: BaseException) -> None:
    status = (
        error.response.status_code if isinstance(error, httpx.HTTPStatusError) else "unavailable"
    )
    logger.warning("enrichment source=%s type=%s status=%s", source, type(error).__name__, status)


def paper_doi(paper: Paper) -> str | None:
    return normalize_doi(paper.doi)
