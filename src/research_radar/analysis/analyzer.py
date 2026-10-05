import json
from datetime import UTC, datetime
from typing import Literal

from research_radar.analysis.critic import CriticService
from research_radar.analysis.llm import (
    LLMProvider,
    cached_generate,
    fingerprint,
    read_cached,
    write_cached,
)
from research_radar.analysis.pdf import FullTextLoader, PaperText
from research_radar.config import Settings
from research_radar.models.analysis import (
    AnalysisContent,
    AnalysisResult,
    CriticResult,
    PromptVersions,
)
from research_radar.models.paper import SourceReference
from research_radar.models.ranking import PaperRecord
from research_radar.prompts import load_prompt


def analysis_input_fingerprint(record: PaperRecord) -> str:
    """Stable evidence identity, excluding ranking and metric observation timestamps."""
    return fingerprint(
        {
            "paper": record.paper.model_dump(mode="json"),
            "enrichment": record.enrichment.model_dump(mode="json"),
            "metrics": record.metrics.model_dump(mode="json", exclude={"observed_at"}),
        }
    )


def supplied_sources(record: PaperRecord) -> list[SourceReference]:
    sources = list(record.enrichment.sources)
    paper = record.paper
    entries: tuple[tuple[str, str | None, Literal["paper", "github", "huggingface"]], ...] = (
        (paper.title, paper.abs_url, "paper"),
        (f"{paper.title} (PDF)", paper.pdf_url, "paper"),
        ("Repository", record.enrichment.github_url, "github"),
        ("Hugging Face paper", record.enrichment.hf_paper_url, "huggingface"),
    )
    for title, url, kind in entries:
        if url:
            sources.append(SourceReference(title=title, url=url, source_type=kind))
    for work in record.enrichment.related_works:
        sources.append(SourceReference(title=work.title, url=work.url, source_type="paper"))
    evidence = record.enrichment.repository_evidence
    if evidence:
        sources.append(
            SourceReference(
                title="Repository evidence", url=evidence.evidence_url, source_type="paper"
            )
        )
    return list({source.url: source for source in sources}.values())


class AnalysisService:
    def __init__(
        self, settings: Settings, provider: LLMProvider, critic_provider: LLMProvider
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.critic = CriticService(settings, critic_provider)
        self.full_text = FullTextLoader(settings)

    def _evidence(
        self, record: PaperRecord, extracted: PaperText, sources: list[SourceReference]
    ) -> tuple[str, bool, bool]:
        paper = record.paper
        document: dict[str, object] = {
            "paper_id": paper.id,
            "primary_paper": {
                "title": paper.title,
                "abstract": paper.abstract,
                "authors": paper.authors,
                "published_at": paper.published_at.isoformat(),
                "updated_at": paper.updated_at.isoformat() if paper.updated_at else None,
            },
            "full_text": extracted.text,
            "full_text_available": extracted.full_text_available,
            "input_truncated": extracted.truncated,
            "github_readme": record.enrichment.github_readme,
            "enrichment": record.enrichment.model_dump(
                mode="json", exclude={"github_readme", "sources"}, exclude_none=True
            ),
            "metrics": record.metrics.model_dump(
                mode="json", exclude={"observed_at"}, exclude_none=True
            ),
            "supplied_sources": [source.model_dump() for source in sources],
        }
        # Bound the supplied document, retaining primary abstract before secondary README/text.
        truncated = extracted.truncated
        serialized = json.dumps(document, ensure_ascii=False)
        for key in ("github_readme", "full_text"):
            value = document[key]
            if isinstance(value, str) and len(serialized) > self.settings.max_analysis_characters:
                excess = len(serialized) - self.settings.max_analysis_characters + 32
                document[key] = value[: max(0, len(value) - excess)]
                truncated = True
                serialized = json.dumps(document, ensure_ascii=False)
        available = extracted.full_text_available and bool(document["full_text"])
        document["full_text_available"] = available
        document["input_truncated"] = truncated
        serialized = json.dumps(document, ensure_ascii=False)
        if len(serialized) > self.settings.max_analysis_characters:
            raise ValueError(
                "Paper metadata and abstract exceed the configured analysis input limit"
            )
        return serialized, available, truncated

    async def analyze(self, record: PaperRecord) -> AnalysisResult:
        sources = supplied_sources(record)
        extracted = await self.full_text.load(record.paper)
        evidence, available, truncated = self._evidence(record, extracted, sources)
        analysis_prompt = load_prompt("analysis", self.settings.analysis_prompt_version)
        critic_prompt = load_prompt("critic", self.settings.critic_prompt_version)
        versions = PromptVersions(
            relevance=self.settings.relevance_prompt_version,
            analysis=self.settings.analysis_prompt_version,
            critic=self.settings.critic_prompt_version,
        )
        key = fingerprint(
            {
                "paper_id": record.paper.id,
                "record_fingerprint": analysis_input_fingerprint(record),
                "evidence": evidence,
                "analysis_prompt": analysis_prompt,
                "critic_prompt": critic_prompt,
                "versions": versions.model_dump(),
                "model": self.settings.analysis_model,
                "critic_model": self.settings.critic_model,
                "schema": AnalysisContent.model_json_schema(),
                "critic_schema": CriticResult.model_json_schema(),
                "result_schema": AnalysisResult.model_json_schema(),
                "policy": "analysis-three-attempts-inline-sources-v2",
            }
        )
        path = self.settings.cache_dir / "analyses" / f"{key}.json"
        cached = await read_cached(path, AnalysisResult)
        if cached is not None:
            return cached
        corrections: list[dict[str, str]] = []
        allowed_urls = {source.url for source in sources}
        for attempt in range(3):
            content = await cached_generate(
                self.settings,
                self.provider,
                system_prompt=analysis_prompt,
                user_prompt=json.dumps(
                    {
                        "evidence": json.loads(evidence),
                        "corrections": corrections,
                        "attempt": attempt + 1,
                    },
                    ensure_ascii=False,
                ),
                schema=AnalysisContent,
                model=self.settings.analysis_model,
                paper_id=record.paper.id,
                stage="analysis",
                version=self.settings.analysis_prompt_version,
            )
            critic = await self.critic.review(
                record=record,
                evidence=evidence,
                content=content,
                allowed_urls=allowed_urls,
                attempt=attempt + 1,
            )
            if critic.passed:
                break
            corrections = [issue.model_dump() for issue in critic.issues]
        result = AnalysisResult(
            paper_id=record.paper.id,
            content=content,
            critic=critic,
            status="review" if critic.passed else "needs_review",
            full_text_available=available,
            input_truncated=truncated,
            confidence="standard" if available and not truncated else "reduced",
            generated_at=datetime.now(UTC),
            model=self.settings.analysis_model,
            critic_model=self.settings.critic_model,
            prompt_versions=versions,
            cache_key=key,
            input_fingerprint=analysis_input_fingerprint(record),
        )
        await write_cached(path, result)
        return result

    async def aclose(self) -> None:
        await self.full_text.aclose()
