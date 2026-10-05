"""Semantic Scholar Graph lookups through exact arXiv/DOI identifiers."""

import re
from urllib.parse import quote

from research_radar.collectors.common import (
    Collector,
    metadata_repository,
    paper_arxiv_id,
    paper_doi,
)
from research_radar.collectors.schemas import SemanticScholarPaper
from research_radar.models.paper import Enrichment, Paper, SourceReference
from research_radar.ranking.deduplication import canonical_arxiv_id, normalize_doi


class SemanticScholarCollector(Collector):
    async def enrich(self, paper: Paper) -> Enrichment:
        arxiv, doi = paper_arxiv_id(paper), paper_doi(paper)
        identifier = f"ARXIV:{arxiv}" if arxiv else f"DOI:{doi}" if doi else None
        if identifier is None:
            return Enrichment()
        headers = {"Accept": "application/json"}
        if self.settings.semantic_scholar_api_key:
            headers["x-api-key"] = self.settings.semantic_scholar_api_key.get_secret_value()
        response = await self.http.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{quote(identifier, safe=':')}",
            params={
                "fields": (
                    "paperId,externalIds,title,citationCount,influentialCitationCount,"
                    "venue,year,openAccessPdf"
                )
            },
            headers=headers,
        )
        work = SemanticScholarPaper.model_validate_json(response.content)
        if re.fullmatch(r"[A-Za-z0-9_-]+", work.paperId) is None:
            raise ValueError("Invalid Semantic Scholar paper identifier")
        if work.externalIds:
            if arxiv and work.externalIds.ArXiv:
                if canonical_arxiv_id(work.externalIds.ArXiv) != arxiv:
                    raise ValueError("Mismatched Semantic Scholar arXiv identifier")
            elif not arxiv and doi and work.externalIds.DOI:
                if normalize_doi(work.externalIds.DOI) != doi:
                    raise ValueError("Mismatched Semantic Scholar DOI")
        source_url = f"https://www.semanticscholar.org/paper/{work.paperId}"
        evidence = metadata_repository(
            work.openAccessPdf.url if work.openAccessPdf else None, source_url
        )
        return Enrichment(
            semantic_scholar_id=work.paperId,
            semantic_scholar_citations=work.citationCount,
            influential_citations=work.influentialCitationCount,
            venue=work.venue or None,
            publication_year=work.year,
            github_url=evidence.repository_url if evidence else None,
            repository_evidence=evidence,
            sources=[
                SourceReference(
                    title=work.title or paper.title,
                    url=source_url,
                    source_type="semantic_scholar",
                )
            ],
        )
