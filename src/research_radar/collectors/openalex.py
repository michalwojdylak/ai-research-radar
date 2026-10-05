"""Exact-identifier OpenAlex lookup; related works are topical, not ancestry."""

import re
from urllib.parse import quote

from research_radar.collectors.common import (
    EXPECTED_ERRORS,
    Collector,
    log_failure,
    metadata_repository,
    paper_arxiv_id,
    paper_doi,
)
from research_radar.collectors.schemas import OpenAlexWork
from research_radar.models.paper import Enrichment, Paper, RelatedWork, SourceReference
from research_radar.ranking.deduplication import canonical_arxiv_id, normalize_doi

_WORK_ID = re.compile(r"https://openalex.org/(W\d+)")


class OpenAlexCollector(Collector):
    async def enrich(self, paper: Paper) -> Enrichment:
        arxiv = paper_arxiv_id(paper)
        doi = paper_doi(paper) or (f"10.48550/arxiv.{arxiv}" if arxiv else None)
        if doi is None:
            return Enrichment()
        headers = {"Accept": "application/json", "User-Agent": "AI-Research-Radar/0.1"}
        if self.settings.openalex_email:
            headers["User-Agent"] += f" (mailto:{self.settings.openalex_email})"
        response = await self.http.get(
            f"https://api.openalex.org/works/{quote(f'https://doi.org/{doi}', safe='')}",
            headers=headers,
        )
        work = OpenAlexWork.model_validate_json(response.content)
        if _WORK_ID.fullmatch(work.id) is None:
            raise ValueError("Invalid OpenAlex work identifier")
        locations = list(work.locations or [])
        if work.primary_location:
            locations.insert(0, work.primary_location)
        returned_dois = {normalize_doi(work.doi)}
        if work.ids:
            returned_dois.add(normalize_doi(work.ids.doi))
        exact_arxiv = arxiv is not None and any(
            canonical_arxiv_id(url) == arxiv
            for location in locations
            for url in (location.landing_page_url, location.pdf_url)
        )
        if doi not in returned_dois and not exact_arxiv:
            raise ValueError("OpenAlex response lacks matching DOI or arXiv identifier")
        evidence = next(
            (
                evidence
                for location in locations
                for url in (location.landing_page_url, location.pdf_url)
                if (evidence := metadata_repository(url, work.id)) is not None
            ),
            None,
        )
        result = Enrichment(
            openalex_id=work.id,
            openalex_citations=work.cited_by_count,
            publication_year=work.publication_year,
            topics=[topic.display_name for topic in work.topics or [] if topic.display_name],
            venue=(
                work.primary_location.source.display_name or None
                if work.primary_location and work.primary_location.source
                else None
            ),
            github_url=evidence.repository_url if evidence else None,
            repository_evidence=evidence,
            sources=[
                SourceReference(
                    title=work.display_name or paper.title, url=work.id, source_type="openalex"
                )
            ],
        )
        for related_id in (work.related_works or [])[:5]:
            match = _WORK_ID.fullmatch(related_id)
            if match is None:
                continue
            try:
                related_response = await self.http.get(
                    f"https://api.openalex.org/works/{match.group(1)}", headers=headers
                )
                related = OpenAlexWork.model_validate_json(related_response.content)
                if related.id != related_id or not related.display_name:
                    raise ValueError("Invalid related work")
                result.related_works.append(
                    RelatedWork(
                        title=related.display_name,
                        url=related.id,
                        year=related.publication_year,
                    )
                )
            except EXPECTED_ERRORS as error:
                log_failure("openalex.related", error)
        return result
