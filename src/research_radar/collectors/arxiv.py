"""Discovery through the official, paginated arXiv Atom API."""

import logging
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlsplit

import feedparser
import httpx
from pydantic import ValidationError

from research_radar.collectors.http import CachedHTTP
from research_radar.config import Settings
from research_radar.models.paper import Paper
from research_radar.ranking.deduplication import canonical_arxiv_id, normalize_doi

logger = logging.getLogger(__name__)
ARXIV_API = "https://export.arxiv.org/api/query"
MAX_DISCOVERY_PAGES = 100
MAX_DISCOVERY_PAPERS = 10_000


def _date(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Missing arXiv timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing arXiv text field")
    return " ".join(value.split())


def _version(value: str | None) -> int:
    match = re.search(r"v(\d+)$", urlsplit(value or "").path)
    return int(match.group(1)) if match else 0


def _entry(entry: Mapping[str, Any]) -> Paper:
    identifier = canonical_arxiv_id(entry.get("id"))
    if identifier is None:
        raise ValueError("Invalid arXiv identifier")
    version = _version(str(entry["id"]))
    arxiv_id = f"{identifier}v{version}" if version else identifier
    authors = [_text(author.get("name")) for author in entry.get("authors", [])]
    categories = [_text(tag.get("term")) for tag in entry.get("tags", [])]
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    for link in entry.get("links", []):
        href = link.get("href")
        if (
            link.get("type") == "application/pdf"
            and isinstance(href, str)
            and urlsplit(href).hostname in {"arxiv.org", "export.arxiv.org"}
            and canonical_arxiv_id(href) == identifier
        ):
            parsed = urlsplit(href)
            pdf_url = f"https://arxiv.org{parsed.path}"
            break
    return Paper(
        id=identifier,
        arxiv_id=arxiv_id,
        title=_text(entry.get("title")),
        abstract=_text(entry.get("summary")),
        authors=authors,
        published_at=_date(entry.get("published")),
        updated_at=_date(entry["updated"]) if entry.get("updated") else None,
        categories=categories,
        pdf_url=pdf_url,
        abs_url=f"https://arxiv.org/abs/{identifier}",
        doi=normalize_doi(entry.get("arxiv_doi")),
    )


class ArxivCollector:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.http = CachedHTTP(settings, client)

    async def discover(
        self, *, days: int | None = None, now: datetime | None = None
    ) -> list[Paper]:
        window = self.settings.research_days if days is None else days
        if not 1 <= window <= 3650:
            raise ValueError("Discovery days must be between 1 and 3650")
        end = now or datetime.now(UTC)
        end = end.replace(tzinfo=UTC) if end.tzinfo is None else end.astimezone(UTC)
        start_date = end - timedelta(days=window)
        categories = " OR ".join(f"cat:{category}" for category in self.settings.arxiv_categories)
        query = f"({categories}) AND submittedDate:[{start_date:%Y%m%d%H%M} TO {end:%Y%m%d%H%M}]"
        papers: dict[str, Paper] = {}
        seen_entries: set[str] = set()
        offset = 0
        for _ in range(MAX_DISCOVERY_PAGES):
            response = await self.http.get(
                ARXIV_API,
                params={
                    "search_query": query,
                    "start": offset,
                    "max_results": self.settings.arxiv_page_size,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                },
            )
            feed = feedparser.parse(response.content)
            if feed.bozo:
                raise ValueError("Malformed arXiv Atom response")
            entries = feed.entries
            if not entries:
                break
            identifiers = {str(entry.get("id", "")) for entry in entries}
            if identifiers <= seen_entries:
                logger.warning("arxiv pagination repeated page")
                break
            seen_entries.update(identifiers)
            for entry in entries:
                try:
                    paper = _entry(entry)
                except (ValidationError, ValueError, KeyError, TypeError, AttributeError) as error:
                    logger.warning("arxiv entry invalid type=%s", type(error).__name__)
                    continue
                if start_date <= paper.published_at <= end and set(paper.categories).intersection(
                    self.settings.arxiv_categories
                ):
                    old = papers.get(paper.id)
                    if old is None or (
                        paper.updated_at or paper.published_at,
                        _version(paper.arxiv_id),
                    ) > (
                        old.updated_at or old.published_at,
                        _version(old.arxiv_id),
                    ):
                        papers[paper.id] = paper
            offset += len(entries)
            if len(papers) >= MAX_DISCOVERY_PAPERS:
                logger.warning("arxiv discovery paper limit reached")
                break
            total_value = feed.feed.get("opensearch_totalresults")
            total = int(total_value) if total_value is not None else None
            if total is not None and offset >= total:
                break
            if total is None and len(entries) < self.settings.arxiv_page_size:
                break
        else:
            logger.warning("arxiv discovery page limit reached")
        return sorted(
            papers.values(), key=lambda paper: (paper.published_at, paper.id), reverse=True
        )[:MAX_DISCOVERY_PAPERS]
