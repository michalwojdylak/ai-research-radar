import json
import re

from research_radar.analysis.llm import LLMProvider, cached_generate
from research_radar.config import Settings
from research_radar.models.analysis import AnalysisContent, CriticIssue, CriticResult
from research_radar.models.ranking import PaperRecord
from research_radar.prompts import load_prompt


def source_issues(
    content: AnalysisContent, record: PaperRecord, allowed_urls: set[str]
) -> list[CriticIssue]:
    issues: list[CriticIssue] = []
    references = content.sources + [
        source for section in content.section_sources for source in section.sources
    ]
    prose = [
        value
        for name in AnalysisContent.model_fields
        if isinstance(value := getattr(content, name), str)
    ]
    prose.extend(event.description for event in content.timeline)
    prose.extend(content.questions_for_michael)
    inline_urls = {
        match
        for text in prose
        for match in re.findall(r"https?://[^\s<>\"']+", text)
        if not any(
            match.startswith(known) and not match[len(known) :].strip(".,;:!?)]}`*")
            for known in allowed_urls
        )
    }
    unsupported_urls = ({source.url for source in references} - allowed_urls) | inline_urls
    for url in sorted(unsupported_urls):
        issues.append(
            CriticIssue(
                severity="high",
                issue=f"Analysis cites an unsupplied source URL: {url}",
                correction="Remove this reference and unsupported claims; use supplied URLs only.",
            )
        )
    allowed_references = (
        allowed_urls
        | {record.paper.title}
        | {related.title for related in record.enrichment.related_works}
    )
    for event in content.timeline:
        if event.related_paper is not None and event.related_paper not in allowed_references:
            issues.append(
                CriticIssue(
                    severity="high",
                    issue=f"Timeline references unsupplied work: {event.related_paper}",
                    correction="Use an exact supplied title/URL, or null if the work is unknown.",
                )
            )
    return issues


class CriticService:
    def __init__(self, settings: Settings, provider: LLMProvider) -> None:
        self.settings = settings
        self.provider = provider

    async def review(
        self,
        *,
        record: PaperRecord,
        evidence: str,
        content: AnalysisContent,
        allowed_urls: set[str],
        attempt: int = 1,
    ) -> CriticResult:
        result = await cached_generate(
            self.settings,
            self.provider,
            system_prompt=load_prompt("critic", self.settings.critic_prompt_version),
            user_prompt=json.dumps(
                {
                    "evidence": json.loads(evidence),
                    "draft": content.model_dump(),
                    "attempt": attempt,
                },
                ensure_ascii=False,
            ),
            schema=CriticResult,
            model=self.settings.critic_model,
            paper_id=record.paper.id,
            stage="critic",
            version=self.settings.critic_prompt_version,
        )
        deterministic = source_issues(content, record, allowed_urls)
        return CriticResult(
            passed=result.passed and not deterministic,
            issues=result.issues + deterministic,
        )
