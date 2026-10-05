"""Optional Hugging Face paper attention, without inferred missing counters."""

from research_radar.collectors.common import (
    Collector,
    metadata_repository,
    paper_arxiv_id,
)
from research_radar.collectors.schemas import HuggingFaceAsset, HuggingFaceRecord
from research_radar.models.paper import Enrichment, Paper, SourceReference
from research_radar.ranking.deduplication import canonical_arxiv_id


def _asset_ids(assets: list[str | HuggingFaceAsset] | None) -> list[str] | None:
    if assets is None:
        return None
    return list(dict.fromkeys(asset if isinstance(asset, str) else asset.id for asset in assets))


class HuggingFaceCollector(Collector):
    async def enrich(self, paper: Paper) -> Enrichment:
        arxiv = paper_arxiv_id(paper)
        if arxiv is None:
            return Enrichment()
        headers = {"Accept": "application/json"}
        if self.settings.huggingface_token:
            headers["Authorization"] = (
                f"Bearer {self.settings.huggingface_token.get_secret_value()}"
            )
        response = await self.http.get(
            f"https://huggingface.co/api/papers/{arxiv}", headers=headers
        )
        record = HuggingFaceRecord.model_validate_json(response.content)
        if canonical_arxiv_id(record.paper.id) != arxiv:
            raise ValueError("Mismatched Hugging Face paper identifier")
        source_url = f"https://huggingface.co/papers/{arxiv}"
        evidence = metadata_repository(record.paper.githubRepo or record.githubRepo, source_url)
        models = record.models if record.models is not None else record.paper.models
        datasets = record.datasets if record.datasets is not None else record.paper.datasets
        return Enrichment(
            hf_paper_url=source_url,
            hf_upvotes=record.paper.upvotes if record.paper.upvotes is not None else record.upvotes,
            hf_trending=record.trending if record.trending is not None else record.paper.trending,
            hf_models=_asset_ids(models),
            hf_datasets=_asset_ids(datasets),
            github_url=evidence.repository_url if evidence else None,
            repository_evidence=evidence,
            sources=[SourceReference(title=paper.title, url=source_url, source_type="huggingface")],
        )
