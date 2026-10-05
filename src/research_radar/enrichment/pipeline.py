"""Fault-isolated metadata enrichment, with bounded sequential source requests."""

from research_radar.collectors.common import (
    EXPECTED_ERRORS,
    Collector,
    log_failure,
    merge_enrichment,
)
from research_radar.collectors.github import GitHubCollector
from research_radar.collectors.huggingface import HuggingFaceCollector
from research_radar.collectors.openalex import OpenAlexCollector
from research_radar.collectors.semantic_scholar import SemanticScholarCollector
from research_radar.models.paper import Enrichment, Paper, RepositoryEvidence


class EnrichmentPipeline(Collector):
    async def enrich(self, paper: Paper) -> Enrichment:
        result = Enrichment()
        evidence: list[RepositoryEvidence] = []
        for source, collector in (
            ("openalex", OpenAlexCollector),
            ("semantic_scholar", SemanticScholarCollector),
            ("huggingface", HuggingFaceCollector),
        ):
            try:
                incoming = await collector(self.settings, self.http.client, http=self.http).enrich(
                    paper
                )
                result = merge_enrichment(result, incoming)
                if incoming.repository_evidence:
                    evidence.append(incoming.repository_evidence)
            except EXPECTED_ERRORS as error:
                log_failure(source, error)
        try:
            github = await GitHubCollector(self.settings, self.http.client, http=self.http).enrich(
                paper, metadata_evidence=evidence
            )
            result = merge_enrichment(result, github)
        except EXPECTED_ERRORS as error:
            log_failure("github", error)
        return result
